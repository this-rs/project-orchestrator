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
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::config::ChatConfig;
use super::post_tool_hook;
use super::skill_hook;
use super::types::{
    classify_api_error, is_out_of_credit_message, truncate_snippet, BackgroundTaskInfo,
    BackgroundTaskKind, ChatEvent, ChatEventPage, ChatRequest, CreateSessionResponse,
    MessageSearchHit, MessageSearchResult, PendingMessage, SessionActivity, SessionWorkLog,
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
    memory::{
        ContextInjector, ConversationMemoryManager, GetMessagesOptions, MemoryConfig,
        MessageSearchOptions, MessageSort,
    },
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    StreamDelta, StreamEventData,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::expand_tilde;
mod recovery;
pub use recovery::{RecoveryReport, ScanReport};

/// Broadcast channel buffer size for WebSocket subscribers
const BROADCAST_BUFFER: usize = 256;

/// How long a resumed CLI must stay up after its handshake before the resume is
/// trusted (an unknown `--resume` target makes it exit within a second).
const RESUME_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);
/// How long a provider switch waits for a streaming turn to stop before it reads what the turn
/// had produced (`settle_cut_turn`). A cut turn ends in milliseconds; this only bounds a stuck one.
const CUT_TURN_SETTLE: std::time::Duration = std::time::Duration::from_secs(3);

/// How long an interrupted turn of the Claude CLI waits for the CLI's own `result` of
/// that turn before it ends (see `stream_response`): a CLI answers an interrupt at
/// once, a dead one does not hold the session longer than this.
const INTERRUPTED_TURN_RESULT_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Longest wait for one instance's health while building the Auto pool. A
/// health check can start a process (ACP) or wait on a slow endpoint (a local
/// server loading a model); past this the instance counts as unavailable for
/// this decision instead of holding the opening of the conversation.
pub(crate) const ROUTING_HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
/// Longest wait for one instance's model catalog while building the Auto pool.
/// The catalogs are fetched all at once, so this bounds the whole catalog phase.
pub(crate) const ROUTING_CATALOG_TIMEOUT: Duration = Duration::from_secs(2);

/// How often the watchdog looks for a legacy session whose CLI process is gone (ZOMBIE-CLI).
const DEAD_CLI_WATCH_INTERVAL: Duration = Duration::from_secs(5);

/// How often the watchdog looks for a turn waiting on a tool result that will never come
/// (LOST-TOOL-RESULT, issue #1056). A turn's clock is minutes, not seconds.
const LOST_TOOL_RESULT_WATCH_INTERVAL: Duration = Duration::from_secs(60);

/// How often the release sweep looks for sessions whose segment is `Done`
/// (`release_done_sessions`): the settle window of `segment_state`, so a finished
/// mission leaves the live set within about a minute.
const DONE_RELEASE_INTERVAL: Duration = Duration::from_millis(super::segment_state::SETTLE_MS);

/// A turn whose last persisted event is a `tool_use` of one of PO's OWN MCP tools, older
/// than this, is dead: those hops are bounded (120 s for the REST proxy in `mcp::http_client`,
/// 300 s for the native MCP client), so a result eight times past the larger bound is not
/// late — it is lost. Nothing kills a turn faster than a lost result can hurt it: the turn
/// is abandoned, its queued messages wait for the next one.
const LOST_TOOL_RESULT_GRACE_MCP: Duration = Duration::from_secs(600);

/// Same, for a tool that runs something (`Bash`, a script, a build): those have no bound of
/// their own, so they are given a much longer silence before the turn is called dead. A
/// command still running after an hour has outlived every budget this harness gives a turn.
const LOST_TOOL_RESULT_GRACE_COMMAND: Duration = Duration::from_secs(3600);

/// Whether a tool name belongs to one of PO's own MCP servers (`mcp__<server>__<tool>`) —
/// i.e. hops this server serves, and whose budgets it therefore knows. The native tools
/// (`Bash`, `Read`, ...) do not travel through it and keep the command grace.
fn is_own_mcp_tool(tool: &str) -> bool {
    tool.starts_with("mcp__")
}

/// How long the detached write of the episode a closed session leaves may take
/// (`record_close_episode`: a few graph round trips). Past it, it is given up and logged.
const CLOSE_EPISODE_BUDGET: Duration = Duration::from_secs(10);

/// Upper bound of building the Auto pool, whatever the number of instances: the
/// health checks run at once (`ROUTING_HEALTH_TIMEOUT`), then the catalogs at once
/// (`ROUTING_CATALOG_TIMEOUT`), then the capability probes are waited for together
/// (the standard probe policy's `wait`); one second of margin for the rest, which
/// is local work.
pub(crate) const ROUTING_POOL_BOUND: Duration = ROUTING_HEALTH_TIMEOUT
    .saturating_add(ROUTING_CATALOG_TIMEOUT)
    .saturating_add(super::provider::cognitive::decider::ProbePolicy::standard().wait)
    .saturating_add(Duration::from_secs(1));
/// Longest wait for the health check that precedes opening a non-Claude-Code
/// session (A30).
pub(crate) const OPEN_PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest wait for `provider.open`. Below the 60 s HTTP request timeout so the
/// person reads why the conversation did not open, not a bare HTTP 408.
pub(crate) const PROVIDER_OPEN_TIMEOUT: Duration = Duration::from_secs(30);
// An Auto opening (pool, preflight, open) ends before the 60 s HTTP request timeout.
const _: () = assert!(
    ROUTING_POOL_BOUND
        .saturating_add(OPEN_PREFLIGHT_TIMEOUT)
        .saturating_add(PROVIDER_OPEN_TIMEOUT)
        .as_secs()
        < 60
);

/// `provider.health()` bounded by `limit`; a timeout reads as unavailable,
/// with a typed `Timeout` error.
pub(crate) async fn bounded_health(
    provider: &dyn nexus_claude::agent::AgentProvider,
    limit: Duration,
) -> nexus_claude::agent::ProviderHealth {
    match tokio::time::timeout(limit, provider.health()).await {
        Ok(health) => health,
        Err(_) => nexus_claude::agent::ProviderHealth {
            status: nexus_claude::agent::HealthStatus::Unavailable,
            version: None,
            detail: Some(format!("no health answer within {} s", limit.as_secs())),
            error: Some(nexus_claude::agent::ProviderError::Timeout {
                after_ms: limit.as_millis() as u64,
            }),
            login_hint: None,
            checked_at_ms: chrono::Utc::now().timestamp_millis().max(0) as u64,
        },
    }
}

/// The instance's key is read from the vault. `vault:` is the only indirection
/// to the vault an instance can store (`parse_credential_ref` accepts `none`,
/// `vault:` and `env:`).
pub(crate) fn keyed_in_vault(instance: &super::provider::settings::InstanceRecord) -> bool {
    instance.credential_ref.trim_start().starts_with("vault:")
}

/// The candidates Auto may route to that depend on the vault: the local
/// instances keyed in it, and Claude Code when the Anthropic catalog is
/// offline because of the locked vault (`claude_code_waits_on_vault`: its
/// models are then rejected as "catalog offline"). Only those the router could
/// take: allowed for the project (`allowed`, the consent the pool applies), and
/// among the ticked models when the menu restricts the pool. Remote instances
/// never enter the Auto pool.
pub(crate) fn vault_backed_candidates(
    instances: &[super::provider::settings::InstanceRecord],
    ticked: Option<&[super::types::RoutingPoolEntry]>,
    allowed: impl Fn(&str) -> bool,
    claude_code_waits_on_vault: bool,
) -> Vec<String> {
    let claude_code =
        claude_code_waits_on_vault.then(|| super::provider::resolver::CLAUDE_CODE.to_string());
    claude_code
        .into_iter()
        .chain(
            instances
                .iter()
                .filter(|i| !super::provider::resolver::is_remote_instance(&i.id))
                .filter(|i| keyed_in_vault(i))
                .map(|i| i.id.clone()),
        )
        .filter(|id| allowed(id))
        .filter(|id| ticked.is_none_or(|pool| pool.iter().any(|e| e.provider == *id)))
        .collect()
}

/// Whether opening in Auto must be refused because the vault is locked: the
/// router's choice applies (`full`, a stage that applies it, nothing named) and
/// some candidate depends on the locked vault. Routing around it would be a
/// silent fallback (A30). When the choice is not applied (stage `shadow`, a
/// pilot at `advisory`), the declared rules open the conversation and nothing
/// is routed around: no refusal.
pub(crate) fn auto_refused_by_locked_vault(
    mode: super::provider::cognitive::ProviderRoutingMode,
    applied: bool,
    named: bool,
    vault_locked: bool,
    vault_backed: &[String],
) -> bool {
    mode == super::provider::cognitive::ProviderRoutingMode::Full
        && applied
        && !named
        && vault_locked
        && !vault_backed.is_empty()
}

/// Most models offered instead of unlocking the vault.
pub(crate) const MAX_LOCKED_VAULT_FALLBACKS: usize = 3;

/// How many pool members the `cognitive decision` debug line names: the pool is the whole
/// catalog, and an unbounded list in a log line is noise (review nit).
const MAX_LOGGED_POOL: usize = 8;

/// The models Auto could still open on without the vault, best first: its own
/// pick, then the eligible alternatives by score. None of them depends on the
/// vault (those came back unhealthy or offline, but the filter says it
/// outright). Empty when every usable model needs the vault: unlocking is then
/// the only way.
pub(crate) fn locked_vault_fallbacks(
    decision: &super::provider::cognitive::decision::CognitiveDecision,
    vault_backed: &[String],
) -> Vec<super::provider::errors::FallbackModel> {
    let mut eligible: Vec<_> = decision
        .alternatives
        .iter()
        .filter(|a| a.rejected.is_none())
        .collect();
    eligible.sort_by(|a, b| {
        b.score
            .unwrap_or(f64::MIN)
            .total_cmp(&a.score.unwrap_or(f64::MIN))
    });
    let mut out: Vec<super::provider::errors::FallbackModel> = Vec::new();
    for pick in decision
        .chosen
        .iter()
        .chain(eligible.into_iter().map(|a| &a.pick))
    {
        if vault_backed.contains(&pick.provider_id)
            || out
                .iter()
                .any(|f| f.provider_id == pick.provider_id && f.model == pick.model)
        {
            continue;
        }
        out.push(super::provider::errors::FallbackModel {
            provider_id: pick.provider_id.clone(),
            model: pick.model.clone(),
        });
        if out.len() == MAX_LOCKED_VAULT_FALLBACKS {
            break;
        }
    }
    out
}

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

/// How long the background-task PID claim (`async_pid_claim`) keeps re-reading
/// the CLI's descendants before giving up. The process it looks for is forked
/// by the tool the CLI is about to run, and its birth is not synchronised with
/// the `ToolUse` we announce: a single reading misses it as soon as the fork
/// lands after that reading (measured under llvm-cov on task `FLAKY-PARITE`,
/// where `cancel_task` then had no PID to signal). Bounded on purpose: the claim
/// is a detached task per background task, and an unbounded poller would spin
/// for the whole session after any process that dies before it is observed.
pub(crate) const PID_CLAIM_WINDOW: Duration = Duration::from_secs(10);

/// Interval between two readings inside [`PID_CLAIM_WINDOW`]: five looks per
/// second, cheap (one `pgrep -P`) and fine-grained enough to catch a fork
/// anywhere in a multi-second window.
pub(crate) const PID_CLAIM_POLL: Duration = Duration::from_millis(500);

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
    /// Per live session, the skills and personas injected into it (see `activation_registry`).
    pub activations:
        std::collections::HashMap<Uuid, crate::chat::activation_registry::SessionActivations>,
    /// Per live, tapped session, what it is doing in its current segment
    /// (`chat::segment_state`), classified when the snapshot was taken.
    pub states: std::collections::HashMap<Uuid, crate::chat::segment_state::Verdict>,
}

/// A test seam of the release loop (see `ChatManager::release_hook`).
#[cfg(test)]
pub(crate) type ReleaseHook = Arc<dyn Fn(&str) + Send + Sync>;

/// A peer's message that resumes a released session (`inject_team_message`).
pub(crate) struct TeamInbound {
    pub from_session_id: Uuid,
    pub from_name: String,
    pub text: String,
}

/// The event a resume stores and broadcasts for its inbound message, with its stored type:
/// a person's `user_message`, or a peer's typed `team_message` that keeps its sender.
pub(crate) fn inbound_event(
    message: &str,
    team: Option<&TeamInbound>,
) -> (ChatEvent, &'static str) {
    match team {
        Some(t) => (
            ChatEvent::TeamMessage {
                from_session_id: t.from_session_id,
                from_name: t.from_name.clone(),
                content: t.text.clone(),
            },
            "team_message",
        ),
        None => (
            ChatEvent::UserMessage {
                content: message.to_string(),
            },
            "user_message",
        ),
    }
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
        let verdict = self.states.get(&id).cloned();
        SessionActivity {
            live: self.live.contains(&id),
            streaming: self.streaming.contains(&id),
            pending_permissions: self.pending_permissions.get(&id).map_or(0, |p| p.len()),
            monitors,
            bash_tasks,
            state: verdict.as_ref().map(|v| v.state),
            state_reason: verdict.map(|v| v.reason),
        }
    }
}

/// CACHE-POOL: whether a newly-opened session should track the project's pending
/// objectives for itself (`ActiveSession::objective_tracking`). An ordinary conversation
/// does; a delegation's child — conversation sub-agent, MCP sub-agent, runner, gate,
/// trigger, any session with a `spawned_by` — does not. The objectives belong to
/// whoever opened it, not to it: measured 17/10/2026, a child that inherited the
/// parent's reminder answered it with an extra, unrequested turn (~9.4k cache tokens).
/// One function for both construction sites (fresh open and restart recovery) so the
/// rule cannot drift between them.
pub(crate) fn tracks_own_objectives(has_spawner: bool) -> bool {
    !has_spawner
}

/// An active chat session with a live Claude CLI subprocess
pub struct ActiveSession {
    /// The model's context window (tokens) the session was opened with, `None` when
    /// unknown: sizes the global cap of each turn's enrichment
    /// (`composer::turn_enrichment_cap`).
    pub context_window: Option<u64>,
    /// The anchor mode of the session, fixed when it was opened or resumed, and the
    /// cache its turns share: nothing re-reads the environment per turn.
    pub anchor: super::anchor_resolver::AnchorSession,
    /// Mirrors the sub-agents of this Claude Code session into read-only child
    /// conversations (`chat::mirror`, P18c): fed by `stream_response`, passive. `None` in tests.
    pub mirror_tap: Option<Arc<super::mirror::MirrorTap>>,
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
    /// Generation of the turn: bumped each time `stream_response` installs a new
    /// `interrupt_token`. The interrupt watchdog compares it to know whether the
    /// turn it cancelled is still the one running (`abandon_turn_if_stuck`).
    pub turn_generation: Arc<AtomicU64>,
    /// Where the turns and the OOB listener are in the CLI's output: the listener
    /// never takes a finished turn's message for background output
    /// (`oob_listener::ResultCursor`).
    pub result_cursor: Arc<super::oob_listener::ResultCursor>,
    /// The task running the current turn (`stream_response`, then whatever it
    /// drains next). Registered by `track_stream_task` at every spawn; aborted by
    /// the interrupt watchdog when the turn ignores its cancellation.
    pub stream_task: Arc<std::sync::Mutex<Option<tokio::task::AbortHandle>>>,
    /// How long one post-stream step may hold a turn of this session
    /// (`post_stream::StepBudget`). `POST_STREAM_STEP_BUDGET` everywhere but in
    /// tests, which shorten it to watch a step overrun.
    pub post_stream_budget: Duration,
    /// The permission requests of the CLI still waiting for an answer.
    /// Key: request_id, Value: the tool input, and the tool.
    /// When the user responds Allow, we include this input in `updatedInput`
    /// so the CLI doesn't lose the original command/parameters.
    pub pending_permission_inputs:
        Arc<tokio::sync::Mutex<std::collections::HashMap<String, PendingPermission>>>,
    /// The outage state of the provider instance this session runs on
    /// (`chat::provider::outage`): the legacy (Claude Code) loop's half of
    /// hold/resume. One [`super::provider::outage::TurnHold`] per session, beside
    /// the retry configuration the whole chat shares — fed from the loop's own text
    /// taxonomy (`outage::outcome_of_legacy_failure`), it lets a turn whose retry
    /// budget is spent be **held** while the instance is down instead of being
    /// closed on an error the user would read as theirs.
    pub provider_outage: Arc<std::sync::Mutex<super::provider::outage::TurnHold>>,
    /// What the user granted "for the session" (`chat::session_grants`): the backend
    /// answers itself the later requests of THIS session they cover; never handed to
    /// the CLI as a rule.
    pub session_grants: Arc<std::sync::Mutex<super::session_grants::SessionGrants>>,
    /// Whether auto-continue is on for this session: the backend sends "Continue" after
    /// a turn stopped on its turn limit (error_max_turns). Always on (decision 7b95342a),
    /// unless the operator's `CHAT_AUTO_CONTINUE=false` turned interactive sessions off;
    /// the WebSocket `set_auto_continue` message no longer changes it.
    pub auto_continue: Arc<AtomicBool>,
    /// The current chain of auto-continues (continuations in a row). Past
    /// `max_auto_continues` the turn stays stopped and the chain starts again from zero
    /// (`post_stream::auto_continue_allowed`).
    pub auto_continue_count: Arc<AtomicU32>,
    /// Maximum number of auto-continues in a row (0 = unlimited):
    /// `post_stream::MAX_AUTO_CONTINUES` for every session.
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
    /// The CLI's descendants read when a `Bash` call STARTS (its
    /// `ContentBlockStart`, input still empty), keyed by `tool_use_id`, taken
    /// back by the second pass of `track_background_task_start`. The CLI cannot
    /// fork the tool before its input is complete, so this is the reference a
    /// background task's PID claim compares against: read at the full
    /// `ToolUse` instead, it already held a process the CLI forked right after
    /// printing the call, which was then never claimed (task 51f87943).
    pub tool_start_descendants: Arc<Mutex<HashMap<String, Vec<u32>>>>,
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

/// What the legacy (Claude Code) loop needs to persist and broadcast an outage
/// notice (`chat::provider::outage`), gathered once per `stream_response` call.
///
/// The notice travels on the **same** channel as every other session fact — the
/// persisted event stream (`ChatEvent::ProviderOutage`) and the live broadcast —
/// exactly as the agent engine emits it (`AgentSessionHandle::emit_outage`), so a
/// client that does not know the variant ignores it and no second channel exists.
struct OutageSink<'a> {
    graph: &'a Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
    next_seq: &'a Arc<AtomicI64>,
    events_tx: &'a broadcast::Sender<ChatEvent>,
    nats: &'a Option<Arc<crate::events::NatsEmitter>>,
    session_id: &'a str,
}

impl OutageSink<'_> {
    /// Persists then broadcasts one transition of the instance's state. Persisted
    /// directly (not through the per-attempt `events_to_persist` buffer, which a
    /// retry clears): a `down` that opens a hold must not be dropped by the very
    /// retry it announces.
    async fn emit(&self, notice: super::provider::outage::OutageNotice) {
        let event = notice.to_chat_event(chrono::Utc::now());
        let _ = self.events_tx.send(event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(self.session_id, event.clone());
        }
        if let Some(uuid) = self.session_uuid {
            let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
            let record = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: uuid,
                seq,
                event_type: event.event_type().to_string(),
                data: serde_json::to_string(&event).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            if let Err(e) = self.graph.store_chat_events(uuid, vec![record]).await {
                warn!(error = %e, "Failed to persist a provider outage notice (non-fatal)");
            }
        }
    }
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

/// The approval scopes the Claude Code engine (legacy path) offers: `once`, and `session`
/// kept by the backend (`chat::session_grants`). `always` (P11b) is added per session, by a
/// session that keeps lasting rules (`offer_lasting_scope`).
/// Declared in its `system_init.capabilities` and checked by
/// [`ChatManager::deliver_permission_answer`]: what is accepted is what is declared.
pub const LEGACY_PERMISSION_SCOPES: &[super::types::PermissionAnswerScope] = &[
    super::types::PermissionAnswerScope::Once,
    super::types::PermissionAnswerScope::Session,
];

/// `system_init.capabilities` of the Claude Code engine: the WHOLE capabilities object, as
/// the agent engine stamps it on a Claude Code session (`AgentSessionHandle::emit_one`:
/// the provider's capabilities without `always`), its `permission_scopes` being exactly
/// [`LEGACY_PERMISSION_SCOPES`] (what [`ChatManager::deliver_permission_answer`] accepts).
pub fn legacy_capabilities() -> serde_json::Value {
    static CAPABILITIES: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    CAPABILITIES
        .get_or_init(|| {
            let mut caps = serde_json::to_value(
                nexus_claude::providers::claude_code::ClaudeCodeConfig::default()
                    .capabilities(None),
            )
            .unwrap_or_else(|_| serde_json::json!({}));
            if let Some(fields) = caps.as_object_mut() {
                fields.insert(
                    "permission_scopes".to_string(),
                    serde_json::json!(LEGACY_PERMISSION_SCOPES),
                );
            }
            caps
        })
        .clone()
}

/// Why a permission answer was not delivered.
#[derive(Debug)]
pub enum PermissionDeliveryError {
    /// No instance holds the session: the CLI that asked is gone.
    SessionDead(String),
    /// The CLI is alive but the request is no longer waiting (already
    /// answered, or never asked).
    NotPending,
    /// The approval asks for a scope (`session`, `always`) this session does not
    /// declare, or cannot express for this request: REFUSED, never answered as a
    /// narrower scope. The request is still waiting.
    ScopeUnsupported(super::types::PermissionAnswerScope),
    Failed(anyhow::Error),
}

impl std::fmt::Display for PermissionDeliveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionDead(id) => write!(f, "Session {id} not found or inactive"),
            Self::NotPending => write!(f, "Permission request is no longer pending"),
            Self::ScopeUnsupported(scope) => write!(
                f,
                "permission scope {} is not supported by this session",
                scope.as_str()
            ),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PermissionDeliveryError {}

/// The `error` of a NATS `control_response` RPC whose request is no longer waiting
/// on the owning instance: the asking instance turns it back into
/// [`PermissionDeliveryError::NotPending`].
pub(crate) const PERMISSION_NOT_PENDING_RPC: &str = "permission_not_pending";

/// The `error` of a NATS `control_response` RPC whose scope the owning instance
/// refused: the asking instance turns it back into
/// [`PermissionDeliveryError::ScopeUnsupported`]. Also the `code` of the WS error.
pub(crate) const PERMISSION_SCOPE_UNSUPPORTED_RPC: &str = "permission_scope_unsupported";

/// A permission request of the CLI waiting for the user's answer
/// (`ActiveSession::pending_permission_inputs`).
#[derive(Debug, Clone, Default)]
pub struct PendingPermission {
    /// The original tool input, replayed as `updatedInput` (an empty `{}` would erase it).
    pub input: serde_json::Value,
    /// The tool asked for.
    pub tool: String,
}

/// What delivering a permission answer to a legacy session reads, borrowed from the
/// manager or from the NATS RPC listener of the session.
struct PermissionAnswer<'a> {
    active_sessions: &'a RwLock<HashMap<String, ActiveSession>>,
    graph: &'a Arc<dyn GraphStore>,
    nats: Option<&'a Arc<crate::events::NatsEmitter>>,
    event_emitter: &'a Option<Arc<dyn crate::events::EventEmitter>>,
}

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
    /// The session is a read-only mirror of a Claude Code sub-agent (`chat::mirror`): nothing
    /// was sent (409 `mirror_read_only`).
    MirrorReadOnly,
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

/// How long a stopped turn may take to end before the interrupt watchdog
/// ends it (`ChatManager::abandon_turn_if_stuck`). A turn that honours its
/// cancellation leaves its stream loop within milliseconds; its post-stream
/// steps are each bounded by `post_stream::POST_STREAM_STEP_BUDGET` and run
/// after the Stop is acknowledged. The acknowledgement itself includes the wait
/// for the CLI's own `result` of the stopped turn, inside the loop and bounded by
/// [`INTERRUPTED_TURN_RESULT_GRACE`]: the grace covers that wait with a margin.
///
/// They used to be equal (5 s each): a CLI that never answered its interrupt
/// held the drain for the whole 5 s, and the watchdog, armed a moment earlier,
/// fired at the same time. When it won (a loaded machine: the Code Coverage
/// job), it aborted a turn that was ending on its own, inside its loop, and the
/// turn's events, collected for one write at the end of the loop, were lost:
/// the stopped turn had neither its text nor its end on the thread, and looked
/// cut at the next boot (task 39427f40).
pub(crate) const INTERRUPT_GRACE: Duration =
    Duration::from_secs(INTERRUPTED_TURN_RESULT_GRACE.as_secs() + 5);
const _: () = assert!(
    INTERRUPT_GRACE.as_secs() >= INTERRUPTED_TURN_RESULT_GRACE.as_secs() + 5,
    "the interrupt watchdog must outlast the wait for the interrupted turn's result"
);

/// Outcome of [`ChatManager::interrupt_scoped`]. Surfaced to REST callers
/// so a client can tell a real interrupt from a silent no-op: before this,
/// `interrupt()` returned `Ok(())` whether it had stopped a live turn or
/// found nothing at all, and the UI had no way to tell the two apart —
/// it just span on "Stopping…" forever.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterruptOutcome {
    /// `true` when a live turn was actually interrupted locally. On the agent
    /// engine this is the provider's own `turn_interrupted` verdict; on the
    /// legacy engine it means the session was in `active_sessions` and the
    /// flag + token + control_request were applied. `false` means nothing
    /// local was stopped — the session may live on another instance (see
    /// `routed`), or not be running at all.
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
// (record, the deferred tools the provider was built with — CTX2, a changed list
// rebuilds it —, the provider)
pub(crate) type NativeCacheEntry = (
    super::provider::settings::InstanceRecord,
    Arc<Vec<String>>,
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
    /// Spend cap of the session (`SessionLimits::max_cost_usd`); `None` = none (PYSR-3).
    pub max_cost_usd: Option<f64>,
    pub kind: nexus_claude::agent::ProviderKind,
    /// Working directory ON THE REMOTE machine, for a `claude_code_remote`
    /// instance. `Some` also means: no project-orchestrator MCP server, no extra
    /// directories (the provider refuses both, and the local paths mean nothing
    /// there).
    pub remote_cwd: Option<&'a str>,
    /// The provider can be given MCP servers for this session
    /// ([`ChatManager::carries_per_session_mcp`]). `false` (OpenClaw's ACP bridge,
    /// whatever its capabilities say): no project-orchestrator MCP server, and the
    /// session reports `project_orchestrator_tools` among what it does not do.
    pub per_session_mcp: bool,
    /// A person lifted the session's file scope ([`ChatSessionNode::file_scope_lifted`]). It
    /// takes effect only for a native session in trust with normal access ([`nt::attach`]).
    ///
    /// [`nt::attach`]: super::provider::nexus_tools::attach
    pub file_scope_lifted: bool,
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
    /// The person the session belongs to (`ChatSessionNode::owner`): with the project,
    /// whose lasting approvals it keeps (`chat::lasting_rules`).
    owner: Option<&'a str>,
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
    pub routing_mode: Option<super::provider::cognitive::ConversationRoutingMode>,
    /// The pairs the request restricted routing to (`mixed`).
    pub routing_pool: Option<Vec<super::types::RoutingPoolEntry>>,
    pub permission_mode: Option<&'a str>,
    /// The message of the turn about to start.
    pub message: &'a str,
    /// Index of the first turn the router counts itself (legacy engine).
    pub next_turn: u32,
    /// The conversation's segmentation (segments + boundaries) carried across a provider
    /// move: the new router resumes it instead of a fresh (or stale store) segmentation.
    /// `None` = nothing to resume.
    pub segments: Option<super::provider::cognitive::segment::SegmentsSnapshot>,
    /// The person the conversation belongs to (`ChatSessionNode::owner`): their "models Auto
    /// may use" (AUTO-L) bound an Auto conversation without a pool of its own.
    pub owner: Option<&'a str>,
}

/// Manages chat sessions and their lifecycle
pub struct ChatManager {
    pub(crate) graph: Arc<dyn GraphStore>,
    #[allow(dead_code)]
    pub(crate) search: Arc<dyn SearchStore>,
    /// The server's embedding provider, when one is configured. Handed to
    /// `KnowledgeInjectionStage` so that knowledge retrieval can add a dense
    /// (cosine) channel to BM25. `None` is a supported state, not a failure:
    /// retrieval then stays lexical-only, exactly as it was before.
    pub(crate) embeddings: Option<Arc<dyn crate::embeddings::EmbeddingProvider>>,
    pub(crate) config: ChatConfig,
    pub(crate) active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    /// Sessions of the provider-neutral path (`CHAT_PROVIDER_PATH=agent`).
    pub(crate) agent_runtime: Arc<super::agent_runtime::AgentRuntime>,
    /// Where the agent path finds a provider instance.
    pub(crate) provider_source: Arc<dyn super::agent_runtime::ProviderSource>,
    /// The CLI path the built-in Claude Code of the agent engine opens with: the one
    /// `update_claude_cli_path` sets, as the historical engine reads it (M2).
    builtin_cli_path: super::agent_runtime::ClaudeCliPath,
    /// Whether the restart recovery resumes a cut Claude Code turn of the agent engine
    /// whatever its permission mode and stored conversation. `false` on every server: there a turn is resumed
    /// through `--resume` only when its CLI asks before each tool and named its
    /// conversation (`recovery::claude_code_left_alone`; review of #726, point 3). Only the
    /// recovery tests set it, their fake being `claude-code` but no CLI.
    recovery_resumes_claude_code: bool,
    /// The cognitive router (R2), when wired. `None` keeps the declarative
    /// resolution exactly as it was.
    pub(crate) cognitive_routing: Option<super::provider::cognitive::decider::CognitiveRouting>,
    /// Open cognitive decisions of live chat sessions (session id -> decision id),
    /// closed with what happened when the session is closed.
    pub(crate) open_decisions:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Uuid>>>,
    /// The objective ledger of each live session (N1): written once per turn by
    /// `after_turn`, read at close. Off unless `CONDUCTION` is set.
    pub(crate) objective_ledgers: std::sync::Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, crate::neurons::conduction::ObjectiveLedger>,
        >,
    >,
    /// Whether the Claude Code CLI is on this server (B2): what the resolver knows of the
    /// built-in instance's health. See [`super::cli_version::ClaudeCliProbe`].
    claude_cli: super::cli_version::ClaudeCliProbe,
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
    /// The host's home, whose git/gh credentials the native shell mirrors (decision 2238b1e4).
    /// `None` in tests that must not depend on the machine's own files.
    pub(crate) host_home: Option<std::path::PathBuf>,
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
    /// One resume at a time per session (review of #684, finding 2): two messages to a
    /// session that is not live would each resume it, and the one that fails would undo
    /// what the other set up.
    pub(crate) resume_locks: Arc<SessionLocks>,
    /// The `refs_v1` switch (on unless `REFS_V1=0`): whether the API layer folds
    /// `refs` into messages and the server announces the capability.
    pub(crate) refs_v1: bool,
    /// Where the documents attached to a message live: the agent engine reads the
    /// attached images from it to send them inline (`documents.storage_dir`).
    pub(crate) document_store: crate::documents::store::DocumentStore,
    /// Root of the native sessions' transcripts (`<root>/<instance id>/<id>.json`, P14);
    /// `None`: kept in memory, lost with the process (nexus' default, the tests' too).
    pub(crate) native_transcripts: Option<std::path::PathBuf>,
    /// The declared ACP agents and the root of their `HOME`s (`with_acp_homes`,
    /// `with_acp_commands`); the default reads `CHAT_PROVIDER_ACP_COMMANDS` and keeps
    /// nexus' default `HOME` under the user's data directory.
    pub(crate) acp: super::provider::native_factory::AcpSetup,
    /// Codex program and home override for tests (`with_codex_program`, `with_codex_home`);
    /// the default uses the `codex` binary on `PATH` and nexus' default `CODEX_HOME`.
    pub(crate) codex: super::provider::native_factory::CodexSetup,
    /// How far the anchor resolver drives the context (`PO_ANCHOR_CONTEXT`, default `shadow`).
    pub(crate) anchor_mode: super::anchor_resolver::AnchorContextMode,
    /// What the turns of every session share in anchor mode (resolutions, shadow runs).
    pub(crate) anchor_cache: Arc<super::anchor_resolver::AnchorCache>,
    /// The held context of each agent session this process opened ([`HeldSlot`]):
    /// one per session id, shared by every state built for it, so a reopen orders
    /// against the writes still running and keeps memory when the store is behind.
    /// One small entry per session opened since the start (its held context is
    /// empty once delivered).
    pub(crate) held_slots: std::sync::Mutex<HashMap<String, Arc<HeldSlot>>>,
    /// The manager itself, once in its `Arc` ([`Self::enable_subagents`]): what the sub-agents
    /// of native sessions open their child sessions through (P18, `subagents.rs`).
    pub(crate) self_ref: std::sync::OnceLock<std::sync::Weak<ChatManager>>,
    /// The permission mode of each live session of the agent engine, as last set
    /// (`set_session_permission_mode`): what a child's ceiling reads (A17), the agent engine
    /// keeping its mode in the harness where `live_session_permission_mode` cannot see it.
    pub(crate) agent_modes: std::sync::Mutex<HashMap<String, String>>,
    /// The sub-agents running for each parent session (P18): a slot is taken before the
    /// envelope counts the live children and held until the sub-agent ends, so parallel `Agent`
    /// calls cannot all count the same children and pass the cap together.
    pub(crate) subagent_slots: std::sync::Mutex<HashMap<String, usize>>,
    /// The openings of children in flight, per parent, from EVERY path (P18-suite, point 6):
    /// counted by the envelope with the live children (`envelope::ChildOpenings`).
    pub(crate) child_openings: super::envelope::ChildOpenings,
    /// The cap on a parent's live children in force now (PLAFOND-RT): read at every delegation,
    /// replaced at runtime by `set_max_live_children`, seeded from the stored setting at startup.
    pub(crate) live_children_cap: std::sync::atomic::AtomicUsize,
    /// The tools the native harness defers (CTX2, `chat::deferred_tools`): fixed into a
    /// native provider when it is built, so a change applies to the sessions opened after it.
    pub(crate) deferred_tools: std::sync::RwLock<Arc<Vec<String>>>,
    /// The read-only mirrors of Claude Code sub-agents this process wrote (`chat::mirror`,
    /// P18c): idempotency across the taps of a session, and the live view of the open ones.
    pub(crate) mirrors: Arc<super::mirror::MirrorRegistry>,
    /// Whether the Claude Code sessions opened here get a mirror tap (always, but in tests).
    pub(crate) subagent_mirrors: bool,
    /// The taps that record each live session's event tail, so the snapshot can say what
    /// it is doing (`chat::segment_tap`, `chat::segment_state`).
    pub(crate) segment_taps: Arc<super::segment_tap::SegmentTaps>,
    /// The warm delegation sessions this server may resume instead of opening a new child
    /// (CACHE-POOL, `chat::session_pool`): a closed sub-agent session whose prefix is still
    /// in the provider's cache, reused by the next delegation of the SAME conversation, on
    /// the SAME engine and model, with the SAME envelope, consent and tool policy. Empty
    /// when the pool finds nothing, which only means a fresh session, as before.
    pub(crate) session_pool: Arc<super::session_pool::SessionPool>,
    /// The sessions a cascade closed in the last minutes (B36-RACE,
    /// `chat::cancelled_parents`): a delegated child whose parent is here is refused at its
    /// activation, inside the critical section that would make it live.
    pub(crate) cancelled_parents: Arc<super::cancelled_parents::CancelledParents>,
    /// Rate-limiting state for `inject_team_message`: sliding window per
    /// (from_session_id, to_session_id) pair. Cleared lazily (no background task).
    pub(crate) team_msg_rate: tokio::sync::Mutex<HashMap<(Uuid, String), Vec<std::time::Instant>>>,
    /// The Claude Code sessions `release_done_sessions` closed because their segment was
    /// Done. Only those are resumed by a team message (`inject_team_message`); any other
    /// close (a person's, the idle cleanup's) clears the mark.
    pub(crate) released_done: std::sync::Mutex<std::collections::HashSet<String>>,
    /// Test seam: runs in the release loop after the second verdict read, before the
    /// recheck, so a test can make a session busy at exactly that point.
    #[cfg(test)]
    pub(crate) release_hook: std::sync::Mutex<Option<ReleaseHook>>,
    /// RL5: the ONE door a provider call goes through — RL2's counters and RL4's
    /// breakers behind a single gate. Shared by every session: the boot injects the
    /// process-wide instance with [`ChatManager::with_provider_limits`]. The default
    /// here is an in-memory, unlimited one, so a session a test builds keeps the path
    /// it had — and no test shares another one's counters.
    pub(crate) provider_limits: Arc<super::provider::limits::Gate>,
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
    /// The per-turn routers: the session's records the compaction decision (shadow).
    routing: Option<Arc<super::agent_hooks::TurnRouting>>,
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
            routing: None,
        }
    }

    /// Attach the per-turn routers, so a compaction records its routing decision.
    pub(crate) fn with_routing(mut self, routing: Arc<super::agent_hooks::TurnRouting>) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Records, without holding the compaction back, the cognitive decision about the
    /// summary, when it is not applied (`agent_hooks::record_compaction_decision`: an applied
    /// one is taken by the harness's `compaction_directive`, N-C1).
    fn record_routing_decision(&self) {
        let Some(router) = self
            .routing
            .as_ref()
            .and_then(|routing| routing.get(&self.session_id))
        else {
            return;
        };
        tokio::spawn(async move {
            super::agent_hooks::record_compaction_decision(&router).await;
        });
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
        self.record_routing_decision();

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
        // Always allow compaction to proceed. The instructions go out as
        // `systemMessage`: the CLI appends a PreCompact callback's systemMessage to
        // the summary's custom instructions, ignores `reason`, and rejects a
        // `hookSpecificOutput` (no PreCompact variant). Measured on Claude Code
        // 2.1.295 (task M2c); the nexus Claude Code facade answers the same way.
        // The CLI ALSO shows this text to the user, in its "Compacted…" line: it is
        // not a hidden channel. Re-check on each CLI bump (docs/parity/engines.md).
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                system_message: custom_instructions,
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

/// The payload of the CRUD `created` event of a chat session (decision ed628dad), cut to
/// what the event's readers need (review nexus-41 of #727, finding 1): the event reaches
/// every authenticated `/ws/events` subscriber, with no owner nor project filter, so it
/// carries no `cwd` (an absolute path that names the OS user), no `model`, no
/// `execution_place`, no `switched_from` and none of the run/plan/task ids of an origin.
/// What stays: `project_slug`, and `spawned_by` reduced to `{type, parent_session_id}` —
/// enough for a client to tell "a child of mine" and refetch its filtered `/children`,
/// which carries the rest (the `tool_use_id` join included). `spawned_by` is absent for a
/// session nothing spawned, `parent_session_id` for an origin that names none (`trigger`).
fn session_created_payload(request: &ChatRequest, project_slug: Option<&str>) -> serde_json::Value {
    session_created_payload_of(request.spawned_by.as_deref(), project_slug)
}

/// [`session_created_payload`] from the session's stored `spawned_by` (a mirror of a Claude
/// Code sub-agent, `chat::mirror`, has no request): the same trimmed payload.
pub(crate) fn session_created_payload_of(
    spawned_by: Option<&str>,
    project_slug: Option<&str>,
) -> serde_json::Value {
    let mut payload = serde_json::json!({ "project_slug": project_slug });
    if let Some(spawned_by) =
        spawned_by.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
    {
        let mut trimmed = serde_json::Map::new();
        if let Some(kind) = spawned_by.get("type") {
            trimmed.insert("type".into(), kind.clone());
        }
        if let Some(parent) = spawned_by.get("parent_session_id").filter(|v| !v.is_null()) {
            trimmed.insert("parent_session_id".into(), parent.clone());
        }
        payload["spawned_by"] = serde_json::Value::Object(trimmed);
    }
    payload
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

/// Whether the request opens a child of a conversation — a native parent's `Agent`
/// call (its `tool_use_id` is set) or a sub-conversation it spawned. Such a child is a
/// sub-task, so its routing signature is built from the sub-task (description, prompt,
/// `subagent_type`), never from a task node or a stated class alone.
fn is_subagent_spawn(request: &ChatRequest) -> bool {
    request
        .spawned_by
        .as_deref()
        .and_then(parse_spawned_by)
        .is_some_and(|ctx| ctx.spawn_type == "conversation")
}

/// Whether the opening has a stated fallback in the session that sent it, so Auto on a
/// locked vault is not refused but opens on that session's pair (`parent_pair_of`):
/// an `Agent` call PO routes (`route_auto`, AUTO-SUB), or a delegation (`delegate_task`)
/// sent by an open conversation (DELEG-500 — the vault re-locks at every restart, and
/// each such delegation answered `credentials_locked`, a bare 500 until #825). A
/// delegation nobody sent (a person, from the plan page) has no parent: refused as
/// any Auto opening is.
fn falls_back_on_parent(request: &ChatRequest) -> bool {
    request.route_auto
        || request
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)
            .is_some_and(|ctx| ctx.spawn_type == "delegation" && ctx.parent_session_id.is_some())
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
/// ENV-WS: whether the project a new session ends with was DEDUCED (the request named
/// none, the server found one from the cwd) rather than chosen. The only input from the
/// request is the ABSENCE of a project: nothing a caller writes can mark a project as
/// deduced, so no caller can widen its children by claiming it.
pub(crate) fn project_was_inferred(requested: Option<&str>, resolved: Option<&str>) -> bool {
    requested.is_none() && resolved.is_some()
}

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

/// One shadow turn of the posture of a session (T7, `chat::posture`): derived from its
/// focus anchors, fitted to the capability snapshot of its opening, tracked with
/// hysteresis and journaled (target `posture_shadow`). It changes nothing the turn does:
/// no prompt, tool or enrichment reads it.
pub(crate) fn observe_posture(
    session_id: Uuid,
    node: &crate::neo4j::models::ChatSessionNode,
    anchors: &[super::anchor::Anchor],
    admitted: Option<&super::anchor_resolver::ResolvedScope>,
) -> super::posture::PostureObservation {
    let caps: Option<nexus_claude::agent::Capabilities> = node
        .capabilities
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok());
    let invariants = super::posture::Invariants {
        access: node.access,
        permission_mode: node.permission_mode.clone(),
    };
    super::posture::book().observe_turn(session_id, anchors, admitted, &invariants, caps.as_ref())
}

/// Mode `shadow`: the resolver does not run for the turn, so the posture is observed in
/// a detached task on the session's anchors (one read), never delaying the turn. The
/// resolver's admission is not applied there; nothing of it reaches the model.
fn spawn_posture_shadow(
    graph: Arc<dyn GraphStore>,
    session_id: Uuid,
    node: crate::neo4j::models::ChatSessionNode,
) {
    tokio::spawn(async move {
        match graph.list_session_anchors(session_id).await {
            Ok(anchors) => {
                observe_posture(session_id, &node, &anchors, None);
            }
            Err(e) => {
                warn!(target: "posture_shadow", session_id = %session_id, error = %e, "posture shadow: anchors unreadable (ignored)")
            }
        }
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
///
/// `window_tokens`: the session's context window (`None`: unknown). The pipeline's
/// block is held to `composer::turn_enrichment_cap` of it, whole sections only
/// ([`super::enrichment::EnrichmentContext::to_system_prompt_markdown_capped`]); the
/// anchor live block, a few lines, is not counted.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn enrichment_for_turn(
    graph: &Arc<dyn GraphStore>,
    pipeline: &super::enrichment::EnrichmentPipeline,
    session_id: &str,
    message: &str,
    protocol: TurnProtocol,
    excluded_note_ids: std::collections::HashSet<String>,
    anchor: &super::anchor_resolver::AnchorSession,
    window_tokens: Option<u64>,
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
                // Posture, shadow (T7): from the resolver's admitted anchors, no I/O.
                observe_posture(uuid, &node, &r.anchors, Some(&r.scope));
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
    // Bind the project the turn actually runs on, once, on the session node (P18).
    // A session opened without a project resolved one from its working directory on
    // every turn, but the node kept `project_slug = null`: the enrichment knew the
    // project while `/api/agents/live` reported an agent "without a project", and a
    // dashboard could not place it in its project. The write refuses a session that
    // already has a slug, so an explicit project is never overwritten.
    if node
        .project_slug
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .is_empty()
    {
        if let Some(slug) = project_slug.as_deref().filter(|s| !s.trim().is_empty()) {
            if let Err(e) = graph.bind_chat_session_project(uuid, slug).await {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "could not bind the resolved project on the session"
                );
            }
        }
    }
    if mode == AnchorContextMode::Shadow {
        spawn_posture_shadow(graph.clone(), uuid, node.clone());
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
    // The synaptic layer (T8) observes the turn when it is in shadow for the
    // project: a detached task, never awaited, whose result is journaled and
    // never reaches this prompt.
    if let Some(pid) = project_id {
        crate::neurons::synaptic_shadow::spawn_turn(graph.clone(), uuid, pid, node.created_at);
    }
    // A restricted provider (Claude Code) is not sent the project's provider-sensitive
    // notes by the per-turn injection either: kept out of the system prompt alone, they
    // would come back in the message of the next turn.
    let restricted_content =
        super::content_policy::provider_content_policy(node.provider_id.as_deref().unwrap_or(""))
            .is_restricted();
    let input = super::enrichment::EnrichmentInput {
        message: message.to_string(),
        session_id: uuid,
        project_slug,
        project_id,
        cwd: Some(node.cwd),
        protocol_run_id: protocol.run_id,
        protocol_state: protocol.state,
        excluded_note_ids,
        restricted_content,
        reasoning_path_tracker: protocol.reasoning_path_tracker,
    };
    let ctx = pipeline.execute(&input).await;
    let mut parts: Vec<String> = Vec::new();
    if !live_block.is_empty() {
        // The live block opens the enrichment, like the other stages: never the prefix.
        parts.push(live_block);
    }
    if ctx.has_content() {
        debug!(
            "[enrichment] Prompt enriched: {} sections, {}ms (hints: {:?})",
            ctx.sections.len(),
            ctx.total_time_ms,
            ctx.hints.keys().collect::<Vec<_>>()
        );
        // Clean markdown prepended to the user message (replaces the old XML-wrapped
        // <enrichment_context> format).
        let md = ctx
            .to_system_prompt_markdown_capped(super::composer::turn_enrichment_cap(window_tokens));
        if !md.is_empty() {
            parts.push(md);
        }
    }
    // The retrieval trigger (PR3 of plan 42d49a31): an INVITATION to consult the graph,
    // appended to the user turn — never to the system prompt, whose prefix must stay
    // byte-identical for the cache. Off by default (`RETRIEVAL_TRIGGER`): off, `parts`
    // holds exactly what the three branches above held before.
    if let Some(invite) = super::retrieval_trigger::for_turn(super::retrieval_trigger::Facts {
        message,
        first_turn: node.message_count <= 1,
    }) {
        parts.push(invite);
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
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
    /// The session's state for the end of its turns, as `ActiveSession` holds it on
    /// the Claude Code engine.
    session: AgentTurnState,
    search: Arc<dyn SearchStore>,
    event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    context_injector: Option<Arc<ContextInjector>>,
    /// The budget of each end-of-turn step (`post_stream::POST_STREAM_STEP_BUDGET`).
    step_budget: Duration,
    /// A step made to never answer (tests only, `None` in production).
    stall_step: Option<&'static str>,
    /// Where routing decisions are stored (`CognitiveRouting::store`): a change of model a
    /// turn could not make is written back on its decision. `None`: nowhere.
    routing_store: Option<Arc<dyn super::provider::cognitive::store::RoutingArmStore>>,
    /// The decisions still open, keyed by session id (shared with [`ChatManager`]),
    /// so a turn's ending can feed its reward to the arm it routed.
    open_decisions: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Uuid>>>,
    objective_ledgers: std::sync::Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, crate::neurons::conduction::ObjectiveLedger>,
        >,
    >,
    /// Mirrors the sub-agents of a Claude Code session into read-only child conversations
    /// (`chat::mirror`, P18c), from [`Self::observe`]. `None` for every other provider.
    mirror: Option<Arc<super::mirror::MirrorTap>>,
    /// The host's door to the graph hooks when the session's provider runs none
    /// (`provider::hooks`, B14). `None`: the provider runs them, or a runner session.
    hook_fallback: Option<Arc<super::provider::hooks::HookFallback>>,
}

/// What a session of the agent engine keeps for the end of its turns: the fields of
/// `ActiveSession` the Claude Code engine's post-stream reads.
#[derive(Clone, Default)]
pub(crate) struct AgentTurnState {
    /// Records the conversation in memory (`None`: no memory configured).
    memory: Option<Arc<Mutex<ConversationMemoryManager>>>,
    rfc_accumulator: Arc<Mutex<super::observation_detector::RfcAccumulator>>,
    /// The protocol run the session was spawned in (`spawned_by`), for the enrichment.
    protocol_run_id: Option<Uuid>,
    protocol_state: Option<String>,
    reasoning_path_tracker: super::feedback::ReasoningPathTracker,
    objectives: super::post_stream::ObjectiveCounters,
    /// The tools the session used (files, steps), live from its events.
    work_log: Arc<std::sync::Mutex<SessionWorkLog>>,
    /// What waits for the session's next turn after a compaction: the re-injected
    /// context and the objective reminder of that turn ([`HeldContext`]). Never a
    /// turn of its own (P8c): a turn started only to carry it was measured against
    /// the window right after the compaction and compacted again (measured in
    /// `routing_modes_e2e_tests` on integ/p8). Persisted with the session (it
    /// survives an idle close, a restart, a resume) and cleared only once a turn
    /// carrying it was answered ([`super::agent_runtime::TurnOutcome::answered`], in `after_turn`).
    held: Arc<HeldSlot>,
    /// Tool calls announced and not yet answered: id → (tool, input so far).
    open_tools: Arc<std::sync::Mutex<HashMap<String, (String, serde_json::Value)>>>,
    /// The model's context window (tokens) the session was opened with, `None` when
    /// unknown: sizes the global cap of each turn's enrichment
    /// (`composer::turn_enrichment_cap`).
    context_window: Option<u64>,
    /// The user's own text (`shown`) of the last turn `prepare` saw, so a context
    /// refusal ([`ProviderError::ContextTooSmall`]) can be replayed without the
    /// injected context. `None`: no turn prepared yet. Shared across clones of the
    /// state (the same pattern as [`Self::held`]).
    last_user_message: Arc<std::sync::Mutex<Option<String>>>,
}

/// A session's [`HeldContext`] in this process: what waits in memory (the truth
/// while the process lives) and the one ordered way it reaches the store.
#[derive(Default)]
pub(crate) struct HeldSlot {
    memory: std::sync::Mutex<HeldMemory>,
    /// Orders the store writes of the session: each write takes it INSIDE its task,
    /// then stores what memory holds at that moment, so the last write to land is
    /// the latest state (a clear can never land after a newer store, among the
    /// writes that land within their bound — see [`Self::write_once`] for one that
    /// was dropped). A reopen ([`Self::reopen`]) takes it too. Every holder bounds
    /// its store call by [`HELD_WRITE_BUDGET_FACTOR`] × the step budget (a write and
    /// the reopen's read alike: one that never answers is dropped, the slot kept
    /// as it stands), so the lock is never held past that, and waiting for it is
    /// bounded by the writes queued before.
    write: tokio::sync::Mutex<()>,
    /// A background retry of a failed write runs ([`Self::retry_in_background`]):
    /// at most one per session.
    retrying: std::sync::atomic::AtomicBool,
}

/// What memory holds for a session, and how it stands against the store.
#[derive(Debug, Clone)]
struct HeldMemory {
    held: HeldContext,
    /// Moves at each change of `held` (`HeldSlot::edit`): a write or a reload knows
    /// whether memory changed since it read it.
    generation: u64,
    /// The generation the store is known to hold; `None`: unknown (a write failed
    /// or ran out of time). Memory and store diverge — the slot is *unsynced* —
    /// whenever this is not the current generation: from the first change of a
    /// turn (`after_turn` empties memory long before its write) until a write of
    /// that state landed.
    stored: Option<u64>,
}

impl Default for HeldMemory {
    /// A slot nobody wrote yet: in step with the store, which the first open of the
    /// session in this process reads.
    fn default() -> Self {
        Self {
            held: HeldContext::default(),
            generation: 0,
            stored: Some(0),
        }
    }
}

/// A held-context write gets this many step budgets before its store call is
/// dropped (the lock released, the slot left unsynced): a store that never answers
/// does not make every later turn of the session wait for the lock.
pub(crate) const HELD_WRITE_BUDGET_FACTOR: u32 = 2;

/// The background retries of a failed held-context write: the first after
/// [`HELD_RETRY_FIRST_DELAY`], each next one twice later, at most
/// [`HELD_RETRY_ATTEMPTS`]; past them the next turn writes again.
pub(crate) const HELD_RETRY_FIRST_DELAY: Duration = Duration::from_secs(1);
pub(crate) const HELD_RETRY_ATTEMPTS: u32 = 6;

impl HeldSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, HeldMemory> {
        self.memory.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What goes in front of the next turn ([`HeldContext::prefix`]).
    fn prefix(&self) -> Option<String> {
        self.lock().held.prefix()
    }

    /// What memory holds.
    #[cfg(test)]
    pub(crate) fn held(&self) -> HeldContext {
        self.lock().held.clone()
    }

    /// Changes what memory holds; a change moves the generation (the store no
    /// longer holds it, [`Self::unsynced`]).
    fn edit<R>(&self, f: impl FnOnce(&mut HeldContext) -> R) -> R {
        let mut m = self.lock();
        let before = m.held.clone();
        let out = f(&mut m.held);
        if m.held != before {
            m.generation += 1;
        }
        out
    }

    /// The store may not hold what memory holds: memory changed since the last
    /// write that landed (a turn under way, a write still running), or a write
    /// failed. The next turn writes again, and a reopen in this process keeps
    /// memory instead of reloading the store.
    fn unsynced(&self) -> bool {
        let m = self.lock();
        m.stored != Some(m.generation)
    }

    /// Replaces memory with what the store holds, read while memory was at
    /// generation `seen` and in step with the store. Refused (memory kept) when
    /// memory moved meanwhile: the store read is older than memory.
    fn reload(&self, stored: HeldContext, seen: u64) -> bool {
        let mut m = self.lock();
        if m.generation != seen || m.stored != Some(seen) {
            return false;
        }
        if m.held != stored {
            m.held = stored;
            m.generation += 1;
            m.stored = Some(m.generation);
        }
        true
    }

    /// Replaces memory with what the store holds, as a new state of the session
    /// starts from it: only a slot in step with the store reloads it, and only if
    /// memory did not move during the read ([`Self::reload`]). Ordered after the
    /// writes already started (their lock, waited for within `budget`; past it the
    /// read goes on, the generation check alone keeping it safe). The read is
    /// bounded like a write ([`HELD_WRITE_BUDGET_FACTOR`] × `budget`): a store that
    /// does not answer, or cannot be read, keeps memory and releases the lock.
    async fn reopen(&self, graph: &Arc<dyn GraphStore>, session_id: &str, budget: Duration) {
        let _order = tokio::time::timeout(budget, self.write.lock()).await.ok();
        let seen = {
            let m = self.lock();
            (m.stored == Some(m.generation)).then_some(m.generation)
        };
        let Some(seen) = seen else {
            return;
        };
        let bound = budget * HELD_WRITE_BUDGET_FACTOR;
        let read = match tokio::time::timeout(bound, HeldContext::load(graph, session_id)).await {
            Ok(read) => read,
            Err(_) => Err(anyhow!(
                "the store did not answer within {} ms",
                bound.as_millis()
            )),
        };
        match read {
            Ok(stored) => {
                self.reload(stored, seen);
            }
            Err(e) => warn!(
                session_id,
                "held context: the store could not be read, memory kept: {e:#}"
            ),
        }
    }

    /// Writes what memory holds to the store, ordered after the writes already
    /// started, the store call bounded ([`HELD_WRITE_BUDGET_FACTOR`] × `budget`).
    /// The slot is in step only if memory did not move since the write read it.
    ///
    /// A store call dropped by the bound may still land on the server, later. If a
    /// newer write lands first, the dropped one overwrites it there while the slot
    /// believes itself in step: no retry, and the newer state is lost at a restart
    /// (memory stays right while the process lives). It takes a store slower than
    /// the bound (60 s in production). A write conditional on a version would close
    /// it, but the generation lives in this process only (it restarts at 0, and
    /// another instance has its own): it would need a durable version and a delete
    /// turned into a versioned tombstone in the `GraphStore` setting API.
    async fn write_once(
        &self,
        graph: &Arc<dyn GraphStore>,
        session_id: &str,
        budget: Duration,
    ) -> Result<()> {
        let _order = self.write.lock().await;
        let (held, generation) = {
            let m = self.lock();
            (m.held.clone(), m.generation)
        };
        let bound = budget * HELD_WRITE_BUDGET_FACTOR;
        let out = match tokio::time::timeout(bound, held.store(graph, session_id)).await {
            Ok(out) => out,
            Err(_) => Err(anyhow!(
                "the store did not answer within {} ms: dropped, retried later",
                bound.as_millis()
            )),
        };
        self.lock().stored = match out {
            Ok(()) => Some(generation),
            Err(_) => None,
        };
        out
    }

    /// Writes what memory holds to the store, in a task of its own
    /// ([`super::post_stream::spawn_write`]: never dropped, a failure logged),
    /// ordered after the writes already started. A failed write leaves the slot
    /// [`Self::unsynced`] and is retried in the background
    /// ([`Self::retry_in_background`]), then by the next turn. Returns the step
    /// (for the report) and the write to wait for.
    fn spawn_write(
        self: &Arc<Self>,
        graph: &Arc<dyn GraphStore>,
        session_id: &str,
        budget: Duration,
    ) -> (&'static str, tokio::task::JoinHandle<()>) {
        let step = if self.lock().held.is_empty() {
            "clear_held_context"
        } else {
            "hold_context"
        };
        let slot = Arc::clone(self);
        let graph = graph.clone();
        let sid = session_id.to_string();
        let write = super::post_stream::spawn_write(session_id.to_string(), step, async move {
            let out = slot.write_once(&graph, &sid, budget).await;
            if out.is_err() {
                slot.retry_in_background(graph, sid, budget);
            }
            out
        });
        (step, write)
    }

    /// Retries a failed write without waiting for a turn (a session closed, idle
    /// closed or simply left keeps no stale context in the store for a restart to
    /// reload): after [`HELD_RETRY_FIRST_DELAY`], then twice later each time, at
    /// most [`HELD_RETRY_ATTEMPTS`] times, until the slot is in step. One retry
    /// loop per session at a time.
    fn retry_in_background(
        self: &Arc<Self>,
        graph: Arc<dyn GraphStore>,
        sid: String,
        budget: Duration,
    ) {
        use std::sync::atomic::Ordering::SeqCst;
        if self.retrying.swap(true, SeqCst) {
            return;
        }
        let slot = Arc::clone(self);
        tokio::spawn(async move {
            let mut delay = HELD_RETRY_FIRST_DELAY;
            let mut in_step = false;
            for attempt in 1..=HELD_RETRY_ATTEMPTS {
                tokio::time::sleep(delay).await;
                if !slot.unsynced() {
                    in_step = true;
                    break;
                }
                match slot.write_once(&graph, &sid, budget).await {
                    Ok(()) => {
                        info!(session_id = %sid, attempt, "held context write retried: landed")
                    }
                    Err(e) => {
                        warn!(session_id = %sid, attempt, "held context write retry failed: {e:#}")
                    }
                }
                if !slot.unsynced() {
                    in_step = true;
                    break;
                }
                delay *= 2;
            }
            slot.retrying.store(false, SeqCst);
            // A write that failed after the last check found the slot in step saw
            // this loop still running and left the retry to it: retry it now. Not
            // after the attempts ran out (the next turn writes again).
            if in_step && slot.lock().stored.is_none() {
                slot.retry_in_background(graph, sid, budget);
            }
        });
    }
}

/// What waits for a session's next turn after a compaction (`AgentTurnState::held`).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct HeldContext {
    /// The project context re-injected after the compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// The objective reminder of the turn that compacted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reminder: Option<String>,
}

impl HeldContext {
    /// Where it is stored: one setting document per session.
    const KEY: &'static str = "held_context";

    fn scope(session_id: &str) -> String {
        format!("session:{session_id}")
    }

    pub fn is_empty(&self) -> bool {
        self.context.is_none() && self.reminder.is_none()
    }

    /// What goes in front of the next turn: the context, then the reminder.
    pub fn prefix(&self) -> Option<String> {
        match (&self.context, &self.reminder) {
            (None, None) => None,
            (Some(c), None) => Some(c.clone()),
            (None, Some(r)) => Some(r.clone()),
            (Some(c), Some(r)) => Some(format!(
                "{c}

{r}"
            )),
        }
    }

    /// The re-injected context cut to fit the model's window: at most
    /// [`HELD_CONTEXT_WINDOW_SHARE`] of a known window (four characters a token),
    /// never more than the builder's own cap (`MAX_MARKDOWN_CHARS`, 6000). It cannot
    /// by itself push a turn over nexus' compaction threshold (80 % of the window)
    /// unless the history alone is already past 75 %.
    pub fn cap(context: &str, window_tokens: Option<u64>) -> String {
        const BUILDER_CAP: usize = 6000;
        let limit = window_tokens
            .map(|w| ((w as f64) * HELD_CONTEXT_WINDOW_SHARE * 4.0) as usize)
            .unwrap_or(BUILDER_CAP)
            .min(BUILDER_CAP);
        if context.chars().count() <= limit {
            return context.to_string();
        }
        const TAIL: &str = "
[… cut to fit the context window]
</system-reminder>";
        let keep = limit.saturating_sub(TAIL.chars().count());
        let mut out: String = context.chars().take(keep).collect();
        out.push_str(TAIL);
        out
    }

    /// The held context of `session_id`, as stored (none: empty; an unreadable
    /// document: empty, said). A store that cannot be read is an error, not an
    /// empty context: the caller keeps what it has.
    pub async fn load(graph: &Arc<dyn GraphStore>, session_id: &str) -> Result<Self> {
        let raw = graph
            .get_llm_setting(&Self::scope(session_id), Self::KEY)
            .await?;
        Ok(match raw {
            Some(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
                warn!(session_id, "held context unreadable, dropped: {e}");
                Self::default()
            }),
            None => Self::default(),
        })
    }

    /// Stores it (empty: clears it).
    pub async fn store(&self, graph: &Arc<dyn GraphStore>, session_id: &str) -> Result<()> {
        if self.is_empty() {
            return Self::clear(graph, session_id).await;
        }
        graph
            .put_llm_setting(
                &Self::scope(session_id),
                Self::KEY,
                &serde_json::to_string(self)?,
            )
            .await
    }

    pub async fn clear(graph: &Arc<dyn GraphStore>, session_id: &str) -> Result<()> {
        graph
            .delete_llm_setting(&Self::scope(session_id), Self::KEY)
            .await
            .map(|_| ())
    }
}

/// Largest share of the model's window the held context may take.
pub(crate) const HELD_CONTEXT_WINDOW_SHARE: f64 = 0.05;

/// The decision of a model change that could not be made says so: not applied, and why
/// (`not_changed: <code>`). Nothing to write without a store.
async fn record_not_changed(
    store: Option<&Arc<dyn super::provider::cognitive::store::RoutingArmStore>>,
    mut decision: super::provider::cognitive::decision::CognitiveDecision,
    code: &str,
) {
    let Some(store) = store else {
        return;
    };
    decision.applied = false;
    decision.reason = format!("{}; not_changed: {code}", decision.reason);
    if let Err(error) = store.put_decision(&decision).await {
        warn!(decision_id = %decision.id, %error, "the refused change was not recorded on its decision");
    }
}

impl ManagerTurnServices {
    /// One end-of-turn step under its own budget ([`run_step`]); `stall_step` makes
    /// the named one never answer (tests only: proves the budgets are per step).
    async fn step<F: std::future::Future>(
        &self,
        after: &mut super::agent_runtime::AfterTurn,
        session_id: &str,
        name: &'static str,
        fut: F,
    ) -> Option<F::Output> {
        let stalled = self.stall_step == Some(name);
        run_step(after, session_id, name, self.step_budget, async move {
            if stalled {
                std::future::pending::<()>().await;
            }
            fut.await
        })
        .await
    }

    fn flush_open_tools(&self) {
        let open: Vec<(String, serde_json::Value)> = self
            .session
            .open_tools
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain()
            .map(|(_, call)| call)
            .collect();
        if open.is_empty() {
            return;
        }
        let mut log = self
            .session
            .work_log
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (tool, input) in open {
            log.record_tool_use(&tool, &input);
        }
    }

    fn work_summary(&self) -> String {
        self.session
            .work_log
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .to_summary_markdown()
    }
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
        // The user's own text, kept for a context refusal's replay (`adapt_context`):
        // what `prepare` was shown, before any injection. Set before the `/compact`
        // early return too, so the last prepared turn is always the last user message.
        *self
            .session
            .last_user_message
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(shown.to_string());
        // `/compact` reaching a provider here is one whose CLI reads the command itself
        // (Claude Code; the others are compacted by the runtime, ISO3): it must head the text
        // the CLI reads, or the CLI hands it to the model as a sentence. Nothing in front.
        if sent == shown && super::manual_compaction::compact_command(shown).is_some() {
            if let Some(router) = self.turn_routing.get(session_id) {
                router.set_turn_input(shown);
            }
            return shown.to_string();
        }
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
        // What the user typed goes to memory, as `stream_response` records it.
        if let Some(mm) = &self.session.memory {
            mm.lock().await.record_user_message(&turn.memory_text);
        }
        let prepared = match enrichment_for_turn(
            &self.graph,
            &self.enrichment_pipeline,
            session_id,
            &turn.enrichment_text,
            TurnProtocol {
                run_id: self.session.protocol_run_id,
                state: self.session.protocol_state.clone(),
                reasoning_path_tracker: Some(self.session.reasoning_path_tracker.clone()),
            },
            turn.excluded_note_ids.clone(),
            &self.anchor,
            self.session.context_window,
        )
        .await
        {
            Some(md) => prepend_enrichment(&md, &body),
            None => body,
        };
        // The advice the post-tool hook gave after the previous turns' tools, when the
        // provider runs no hook (B14): the host's door, in front of the next turn.
        // Only read here, as the held context below: `after_turn` drops it once a
        // turn carrying it was answered (`HookFallback::settle`).
        let prepared = match self.hook_fallback.as_ref().and_then(|f| f.carried()) {
            Some(advice) => prepend_enrichment(&advice, &prepared),
            None => prepared,
        };
        // What waits since the last compaction goes in front of this turn, whatever
        // started it. Only read here: `after_turn` clears it once a turn carrying it
        // was answered (`TurnOutcome::answered`), so a turn refused before it was
        // sent, failed, or stopped before any attempt was answered keeps it.
        let held = self.session.held.prefix();
        let prepared = match held {
            Some(context) => prepend_enrichment(&context, &prepared),
            None => prepared,
        };
        // The hook of the turn only sees the length of the text: hand it the text of
        // THIS turn, whatever started it (a message, the queue, a hint, another
        // instance), right before it is sent — what the user typed, not the
        // `<po-refs>`/`<po-attachments>` blocks around it.
        if let Some(router) = self.turn_routing.get(session_id) {
            router.set_turn_input(shown);
        }
        prepared
    }

    async fn model_for_images(&self, session_id: &str) -> Option<String> {
        let router = self.turn_routing.get(session_id)?;
        super::agent_hooks::model_for_images(&router).await
    }

    async fn image_model_not_applied(&self, session_id: &str, _model: &str) {
        let Some(router) = self.turn_routing.get(session_id) else {
            return;
        };
        // The decision said the image model was applied; it was not (as the legacy
        // `set_model` that fails, `apply_turn_directive`).
        if let Some(decision) = router.forget_image_change() {
            record_not_changed(self.routing_store.as_ref(), decision, "set_model_failed").await;
        }
    }

    async fn model_after_limit(
        &self,
        session_id: &str,
        limit: &nexus_claude::agent::ProviderError,
    ) -> Option<String> {
        let router = self.turn_routing.get(session_id)?;
        super::agent_hooks::model_after_limit(&router, limit).await
    }

    async fn adapt_context(
        &self,
        _session_id: &str,
        input: &nexus_claude::agent::TurnInput,
    ) -> Option<nexus_claude::agent::TurnInput> {
        // The user's own text, without the injected knowledge blocks, the held prefix
        // and the hook advice that `prepare` put in front of it, and without the
        // machine blocks a stored message carries (`<po-refs>`, `<po-attachments>`).
        let shown = self
            .session
            .last_user_message
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        let mut reduced =
            nexus_claude::agent::TurnInput::text(crate::refs::turn::visible_text(&shown));
        // The images stay: an image is not context bloat, and dropping it would change
        // what the person asked. They are already the `InputBlock::Image` blocks of the
        // refused input, so keep those as they are.
        reduced
            .blocks
            .extend(input.blocks.iter().filter_map(|block| match block {
                nexus_claude::agent::InputBlock::Image {
                    media_type,
                    data_base64,
                } => Some(nexus_claude::agent::InputBlock::Image {
                    media_type: media_type.clone(),
                    data_base64: data_base64.clone(),
                }),
                _ => None,
            }));
        Some(reduced)
    }

    async fn continuation(&self, session_id: &str) -> String {
        let ctx = super::post_stream::PostStreamContext::build(
            &self.graph,
            Uuid::parse_str(session_id).ok(),
        )
        .await;
        let work_summary = self.work_summary();
        super::post_stream::continuation_message(
            &self.graph,
            ctx.project_slug.as_deref(),
            &work_summary,
        )
        .await
    }

    async fn images(
        &self,
        shown: &str,
    ) -> std::result::Result<Vec<super::message_attachments::AttachedImage>, String> {
        super::message_attachments::load_images(&self.graph, &self.documents, shown).await
    }

    /// Also tells the session lists (the CRUD bus) when the session starts and stops
    /// streaming, as the Claude Code engine does at the start of a turn and in
    /// `finalize_streaming_status` (P18b round 3, fix F): the runtime emits
    /// `streaming_status` exactly there — `false` only once the queue is empty
    /// (`AgentSessionHandle::next_queued`) — so the pending-queue semantics are the
    /// runtime's own. Without it, a list (a parent's children) never learned that a
    /// native session had finished.
    fn publish(&self, session_id: &str, event: &ChatEvent) {
        if let (ChatEvent::StreamingStatus { is_streaming }, Some(emitter)) =
            (event, &self.event_emitter)
        {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                session_id,
                serde_json::json!({ "is_streaming": is_streaming }),
                None,
            );
        }
        if let Some(nats) = &self.nats {
            nats.publish_chat_event(session_id, event.clone());
        }
    }

    /// The work log records a tool once its input is final: a call announced before
    /// its input was complete (Claude Code, ACP: `tool_use` with `{}`, then
    /// `tool_use_input_resolved`) is logged when its result (or its cancel) arrives,
    /// with the resolved input.
    fn observe(&self, _session_id: &str, event: &ChatEvent) {
        // A Claude Code sub-agent's activity is mirrored, passively (P18c).
        if let Some(tap) = &self.mirror {
            tap.observe(event);
        }
        // A provider that runs no hook: the host sees the tools and the compaction
        // signal in its place (B14).
        if let Some(fallback) = &self.hook_fallback {
            fallback.observe(event);
        }
        let mut open = self
            .session
            .open_tools
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match event {
            ChatEvent::ToolUse {
                id, tool, input, ..
            } => {
                open.insert(id.clone(), (tool.clone(), input.clone()));
            }
            ChatEvent::ToolUseInputResolved { id, input, .. } => {
                if let Some(call) = open.get_mut(id) {
                    call.1 = input.clone();
                }
            }
            ChatEvent::ToolResult { id, .. } | ChatEvent::ToolCancelled { id, .. } => {
                if let Some((tool, input)) = open.remove(id) {
                    drop(open);
                    self.session
                        .work_log
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .record_tool_use(&tool, &input);
                }
            }
            _ => {}
        }
    }

    /// What a Claude Code sub-agent still running in the background prints between the
    /// parent's turns reaches its mirror in the form a turn shows it (P18c); the parent
    /// itself shows it flattened, as before, and its work log is not touched.
    fn observe_out_of_turn(&self, _session_id: &str, events: &[nexus_claude::agent::AgentEvent]) {
        let Some(tap) = &self.mirror else { return };
        for event in events {
            for structured in super::provider::event_map::structured_out_of_turn(event) {
                tap.observe(&structured);
            }
        }
    }

    /// The post-stream steps of the Claude Code engine, with the same functions:
    /// post-compaction re-injection, objective tracking, memory / feedback /
    /// observations. Interrupt cleanup and auto-continue are the runtime's.
    async fn after_turn(
        &self,
        session_id: &str,
        outcome: &super::agent_runtime::TurnOutcome,
    ) -> super::agent_runtime::AfterTurn {
        let mut after = super::agent_runtime::AfterTurn::default();
        let uuid = Uuid::parse_str(session_id).ok();
        // Gap 2: feed the bandit the outcome of the turn it routed. Only for the
        // OPEN decision of a turn that reached a terminal `done`; an overridden
        // decision is skipped — `close_routing_decision` already feeds it.
        if let (Some(store), Some(stop_reason)) =
            (self.routing_store.clone(), outcome.stop_reason.as_deref())
        {
            let decision_id = self
                .open_decisions
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(session_id)
                .copied();
            if let Some(decision_id) = decision_id {
                if let Ok(Some(decision)) = store.decision(decision_id).await {
                    if let Some((arm, observation)) =
                        super::provider::cognitive::decision::turn_settlement(
                            &decision,
                            Some(stop_reason),
                            &outcome.subtype,
                            outcome.is_error,
                            // A refusal of the account is not the model failing (QUOTA).
                            outcome.error_code.as_deref(),
                            outcome.quota_refusal,
                        )
                    {
                        if let Err(error) = store.observe(&arm, &observation).await {
                            warn!(
                                %session_id, %decision_id, %error,
                                "feeding the turn reward failed"
                            );
                        }
                    }
                }
            }
        }
        {
            let state = crate::neurons::conduction::state_from_stop(
                outcome.stop_reason.as_deref(),
                outcome.is_error,
                &outcome.subtype,
            );
            let mut ledgers = self
                .objective_ledgers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let ledger = ledgers.entry(session_id.to_string()).or_insert_with(|| {
                crate::neurons::conduction::ObjectiveLedger::new(
                    crate::neurons::conduction::Objective::None,
                )
            });
            let _ = ledger.settle(
                state,
                chrono::Utc::now(),
                outcome.stop_reason.clone().unwrap_or_default(),
            );
        }
        // The turn carried what waited since the last compaction and was answered:
        // dropped in memory before anything else (before the first await, so the
        // backstop cannot skip it), a compaction of this same turn holding its own
        // context again below. The store follows in ONE ordered write at the end.
        let previous = if outcome.answered {
            self.session.held.edit(std::mem::take)
        } else {
            HeldContext::default()
        };
        let dropped = !previous.is_empty();
        // Each step under its own budget, as `PostStreamHandler`'s (`StepBudget`): a
        // step that never answers is dropped and said, the next ones still run.
        let ctx = self
            .step(
                &mut after,
                session_id,
                "context",
                super::post_stream::PostStreamContext::build(&self.graph, uuid),
            )
            .await
            .unwrap_or_default();
        // 1. Post-compaction context re-injection, a Stop included: unlike the Claude
        // Code engine, nothing is queued as a turn (a Stop would drop it), the context
        // waits in front of the next turn. Skipping it on a Stop lost the context the
        // turn delivered AND the one of its own compaction.
        if outcome.compacted {
            let snapshot = self
                .session
                .work_log
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .snapshot();
            if let Some((hint, event)) = self
                .step(
                    &mut after,
                    session_id,
                    "post_compaction",
                    super::post_stream::post_compaction_recovery(
                        &self.graph,
                        session_id,
                        ctx.project_slug.as_deref(),
                        Some(snapshot),
                        outcome.context_window,
                    ),
                )
                .await
            {
                // Kept for the next turn, never queued as a turn of its own, and
                // no bigger than the window allows (`HeldContext::cap`).
                if let Some(hint) = hint {
                    let hint = HeldContext::cap(&hint, outcome.context_window);
                    self.session.held.edit(|h| h.context = Some(hint));
                }
                after.events.push(event);
            }
            // The compaction summarized what this turn carried: when no fresh context
            // came (the step ran out of time), what it carried waits again — the
            // context and its reminder (a newer reminder of this turn replaces it
            // below) — rather than nothing.
            self.session.held.edit(|held| {
                if held.context.is_none() {
                    held.context = previous.context.clone();
                    if held.reminder.is_none() {
                        held.reminder = previous.reminder.clone();
                    }
                }
            });
        }
        // 2. Objective tracking, as on the Claude Code engine. After a turn that
        // compacted, its reminder waits with the re-injected context, in front of
        // the next turn, instead of starting a turn on a history just compacted.
        {
            let had_conclusive = outcome
                .tools
                .iter()
                .any(|(tool, input)| is_conclusive_tool(tool, input));
            let had_productive = outcome
                .tools
                .iter()
                .any(|(tool, input)| !is_conclusive_tool(tool, input));
            let turn = super::post_stream::ObjectiveTurn {
                had_productive_tool_use: had_productive,
                had_conclusive_tool_use: had_conclusive,
                auto_continue_allowed: outcome.auto_continue_allowed,
                hit_error_max_turns: outcome.hit_turn_limit,
                interrupted: outcome.interrupted,
            };
            let reminder = self
                .step(
                    &mut after,
                    session_id,
                    "objective_tracking",
                    super::post_stream::objective_reminder(
                        &self.graph,
                        ctx.project_slug.as_deref(),
                        &self.session.objectives,
                        turn,
                        self.work_summary(),
                    ),
                )
                .await
                .flatten();
            if let Some(reminder) = reminder {
                info!(session_id, "Objective tracker: injecting reminder");
                if outcome.compacted {
                    self.session.held.edit(|h| h.reminder = Some(reminder));
                } else {
                    after.hints.push(reminder);
                }
            }
        }
        // What waits for the next turn is stored with the session: ONE write of what
        // memory holds, ordered after the session's earlier writes (`HeldSlot`), so a
        // clear can never land after the store of this turn's compaction. Also when
        // an earlier write failed (`unsynced`): retried until the store follows, so a
        // restart, an idle close or a relay never reloads a context already delivered.
        // Waited for within the step budget; past it the clients learn it is late
        // (`persistence_delayed`) and the write goes on.
        if outcome.compacted || dropped || self.session.held.unsynced() {
            let (step, write) =
                self.session
                    .held
                    .spawn_write(&self.graph, session_id, self.step_budget);
            if let Some(late) =
                super::post_stream::await_write(session_id, step, self.step_budget, write).await
            {
                after.events.push(late);
            }
        }
        // Tools still open when the turn ended (an error, the turn limit) are
        // logged as they stand: the work log misses none.
        self.flush_open_tools();
        // 3. Memory / feedback / observations (the store write runs on its own).
        self.step(
            &mut after,
            session_id,
            "feedback",
            super::post_stream::record_turn_feedback(
                super::post_stream::TurnFeedback {
                    graph: &self.graph,
                    search: &self.search,
                    event_emitter: &self.event_emitter,
                    session_id,
                    session_uuid: uuid,
                    project_id: ctx.project_id,
                    rfc_accumulator: Some(self.session.rfc_accumulator.clone()),
                },
                &outcome.assistant_text,
                &self.session.memory,
                &self.context_injector,
            ),
        )
        .await;
        // The citations of the answer, validated and announced on the thread,
        // as the Claude Code engine's `handle_feedback` does: the runtime emits
        // what comes back, and persists it (its own sequence), so a replay has
        // it. Readable citations draw a chip, the others stay plain text.
        if let Some(event) = self
            .step(
                &mut after,
                session_id,
                "cites",
                super::feedback::cited_refs_event(&self.graph, session_id, &outcome.assistant_text),
            )
            .await
            .flatten()
        {
            after.events.push(event);
        }
        // A provider that runs no hook: the post-tool hook is played over the turn's
        // tool results, its advice said as a system hint and carried in front of the
        // next turn (`HookFallback::carried`), B14.
        if let Some(fallback) = self.hook_fallback.clone() {
            // What this turn carried is dropped first (answered) or kept, before the
            // advice of its own tools is added.
            fallback.settle(outcome.answered);
            if let Some(advice) = self
                .step(&mut after, session_id, "post_tool_hooks", async move {
                    fallback.after_turn().await
                })
                .await
            {
                after.events.extend(
                    advice
                        .into_iter()
                        .map(|content| ChatEvent::SystemHint { content }),
                );
            }
        }
        after
    }
}

/// One end-of-turn step of the agent engine under its own `budget`: past it the step
/// is dropped (`post_stream::bounded`) and an `error` event says which one
/// (`post_stream_step_abandoned`, `reason` = the step), as the Claude Code engine's
/// `StepBudget::run` reports it.
pub(crate) async fn run_step<F: std::future::Future>(
    after: &mut super::agent_runtime::AfterTurn,
    session_id: &str,
    step: &'static str,
    budget: Duration,
    fut: F,
) -> Option<F::Output> {
    let out = super::post_stream::bounded(session_id, step, budget, fut).await;
    if out.is_none() {
        after.events.push(ChatEvent::Error {
            message: format!(
                "Error: the end-of-turn step '{step}' took longer than {} s: skipped",
                budget.as_secs()
            ),
            parent_tool_use_id: None,
            code: Some(super::post_stream::STEP_ABANDONED_CODE.to_string()),
            reason: Some(step.to_string()),
            index: None,
            request_id: None,
        });
    }
    out
}

/// What another instance asks of a session of the agent engine this instance
/// runs (NATS RPC `rpc.chat.{id}.send`): the message types the Claude Code
/// listener (`spawn_nats_rpc_listener`) answers, served by the session handle.
pub(crate) async fn agent_rpc(
    handle: &Arc<super::agent_runtime::AgentSessionHandle>,
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
            // A scope this instance does not know is refused, never read as once.
            let scope = match field("scope")
                .map(serde_json::from_value::<super::types::PermissionAnswerScope>)
            {
                None => Ok(super::types::PermissionAnswerScope::Once),
                Some(Ok(scope)) => Ok(scope),
                Some(Err(_)) => Err(()),
            };
            let answered = match (
                field("request_id").and_then(|v| v.as_str().map(str::to_string)),
                scope,
            ) {
                (None, _) => Err(PermissionDeliveryError::Failed(anyhow!(
                    "a permission answer needs its request_id"
                ))),
                (Some(_), Err(())) => Err(PermissionDeliveryError::ScopeUnsupported(
                    super::types::PermissionAnswerScope::Once,
                )),
                (Some(request_id), Ok(scope)) => {
                    handle
                        .answer_permission_scoped(&request_id, allow, scope)
                        .await
                }
            };
            return match answered {
                Ok(()) => crate::events::ChatRpcResponse {
                    success: true,
                    error: None,
                },
                Err(PermissionDeliveryError::ScopeUnsupported(_)) => {
                    crate::events::ChatRpcResponse {
                        success: false,
                        error: Some(PERMISSION_SCOPE_UNSUPPORTED_RPC.to_string()),
                    }
                }
                Err(e) => crate::events::ChatRpcResponse {
                    success: false,
                    error: Some(e.to_string()),
                },
            };
        }
        "set_auto_continue" => {
            // Accepted and ignored: auto-continue is always on (decision 7b95342a).
            debug!(session_id = %handle.session_id, "NATS RPC: set_auto_continue ignored (always on)");
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

/// The person behind these claims, as the owner of what they open (P11b): the `sub` of a
/// person's session JWT, of their MCP token, or of an agent session token minted for them.
/// `None` for the server's own service account, the anonymous user (no-auth mode, the
/// standalone MCP server) and any other kind of token (a vault token opens nothing).
pub(crate) fn person_behind(claims: &crate::auth::jwt::Claims) -> Option<&str> {
    use crate::auth::jwt::{
        ANONYMOUS_USER_ID, SERVICE_ACCOUNT_EMAIL, TOKEN_TYPE_AGENT_SESSION, TOKEN_TYPE_MCP,
    };
    let kind_of_person = match claims.token_type.as_deref() {
        None => true,
        Some(kind) => kind == TOKEN_TYPE_MCP || kind == TOKEN_TYPE_AGENT_SESSION,
    };
    (kind_of_person
        && claims.email != SERVICE_ACCOUNT_EMAIL
        && claims.sub != ANONYMOUS_USER_ID.to_string()
        && !claims.sub.is_empty())
    .then_some(claims.sub.as_str())
}

/// The MCP server names the native engine keeps for the servers the backend attaches
/// (nexus-tools, the browser). A server of the session under one of them would REPLACE
/// the backend's (nexus only attaches its defaults when the name is free), and its tools
/// would be taken for the built-in ones: nexus gives every tool of a `nexus` server its
/// canonical name, so a `mcp__nexus__Read` that writes or calls the network would get the
/// session grant of the read-only built-in (`session_grants`), and a `browser_*` tool
/// would be classified by name.
pub(crate) const RESERVED_NATIVE_SERVERS: [&str; 2] = [
    nexus_claude::providers::native::NEXUS_TOOLS_SERVER,
    nexus_claude::providers::native::BROWSER_SERVER,
];

/// Refuses (typed, `invalid`) a native session whose spec already has a server under a
/// reserved name ([`RESERVED_NATIVE_SERVERS`]), before the backend attaches its own.
pub(crate) fn refuse_reserved_native_servers(
    servers: &std::collections::BTreeMap<String, nexus_claude::agent::McpServerSpec>,
) -> Result<()> {
    match RESERVED_NATIVE_SERVERS
        .iter()
        .find(|name| servers.contains_key(**name))
    {
        Some(name) => Err(anyhow::Error::new(
            nexus_claude::agent::ProviderError::invalid(format!(
                "an MCP server of the session is named `{name}`: that name is reserved for the \
                 native engine's own tools; rename the server"
            )),
        )),
        None => Ok(()),
    }
}

/// Adds the servers the backend attaches to a native session (`nexus`, `browser`). A name
/// already taken is refused, never kept ([`RESERVED_NATIVE_SERVERS`]).
pub(crate) fn attach_native_servers(
    servers: &mut std::collections::BTreeMap<String, nexus_claude::agent::McpServerSpec>,
    attached: std::collections::BTreeMap<String, nexus_claude::agent::McpServerSpec>,
) -> Result<()> {
    for (name, server) in attached {
        if servers.contains_key(&name) {
            return Err(anyhow::Error::new(
                nexus_claude::agent::ProviderError::invalid(format!(
                    "an MCP server of the session is named `{name}`: that name is reserved for \
                     the native engine's own tools; rename the server"
                )),
            ));
        }
        servers.insert(name, server);
    }
    Ok(())
}

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

/// Name of the project-orchestrator MCP server in a session's spec.
pub(crate) const PO_MCP_SERVER: &str = "project-orchestrator";

/// Whether a native session keeps its conversation in MEMORY only: the transcript root it
/// would write to lies inside a git work tree, which the factory refuses
/// (`native_factory::build_native_for_sessions`, the SAME predicate). The session works and
/// says it will not resume after a restart (`resume_after_restart`, CTX6 follow-up).
pub(crate) fn transcripts_kept_in_memory(
    kind: nexus_claude::agent::ProviderKind,
    root: Option<&std::path::Path>,
) -> bool {
    matches!(kind, nexus_claude::agent::ProviderKind::Native)
        && root.is_some_and(super::provider::transcripts::inside_git_worktree)
}

/// What a session about to open lacks that the HOST knows and its provider's
/// capabilities may not say, as `degraded_features` entries: `nexus_tools`
/// ([`lacks_nexus_tools`]) and `project_orchestrator_tools` when the spec carries no
/// project-orchestrator MCP server (a remote Claude Code, OpenClaw's ACP bridge).
pub(crate) fn host_degraded(
    kind: nexus_claude::agent::ProviderKind,
    spec: &nexus_claude::agent::SessionSpec,
    configured: Option<&std::path::Path>,
) -> Vec<String> {
    let mut missing = Vec::new();
    if lacks_nexus_tools(kind, spec, configured) {
        missing.push(super::agent_runtime::NEXUS_TOOLS_FEATURE.to_string());
    }
    if !spec.mcp_servers.contains_key(PO_MCP_SERVER) {
        missing.push(super::agent_runtime::PO_TOOLS_FEATURE.to_string());
    }
    missing
}

/// The warning the server logs when a Claude Code session runs on the agent engine and
/// lacks something (`degraded`, what its `system_init` announces): `None` for another
/// provider, and `None` when nothing is missing — the old warning listed nothing and
/// still said "these features are NOT available" (M2, gap D1 of M0).
pub(crate) fn forced_engine_warning(provider_id: &str, degraded: &[String]) -> Option<String> {
    (provider_id == super::provider::resolver::CLAUDE_CODE && !degraded.is_empty()).then(|| {
        format!(
            "Claude Code is running on the agent engine without: {}",
            degraded.join(", ")
        )
    })
}

/// Whether `spec` carries the project-orchestrator server WITHOUT a session token while
/// the server signs its tokens (`signing`): that server shows its tools and authorises
/// none ([`spec_tool_profile`]). A session opened without its caller's identity (the
/// restart recovery) is in that case.
pub(crate) fn po_server_unauthorised(
    spec: &nexus_claude::agent::SessionSpec,
    signing: bool,
) -> bool {
    signing
        && matches!(
            spec.mcp_servers.get(PO_MCP_SERVER),
            Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) if !env.contains_key("PO_AUTH_TOKEN")
        )
}

/// The `result` that ends a turn of the Claude CLI stopped by a person: the CLI's own
/// `result` of that turn when it was read in time, else one made here; either way with
/// `stop_reason: interrupted`, the neutral reason the agent engine gives, and in the
/// stored form of a stopped turn ([`stored_interrupted_result`]): a Stop is not a failure,
/// so the thread reloaded shows no error block under the turn (follow-up review of #711,
/// finding 2). `elapsed`, the time the turn ran, is its duration when the CLI gave none
/// (a `result` made here, or a CLI's `0`): never a false "0 ms".
pub(crate) fn interrupted_turn_end(
    read: Option<Message>,
    elapsed: std::time::Duration,
) -> ChatEvent {
    let from_cli = read
        .as_ref()
        .map(ChatManager::message_to_events)
        .and_then(|events| {
            events
                .into_iter()
                .find(|e| matches!(e, ChatEvent::Result { .. }))
        });
    let elapsed_ms = u64::try_from(elapsed.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let ended = match from_cli {
        Some(ChatEvent::Result {
            session_id,
            duration_ms,
            cost_usd,
            subtype,
            is_error,
            num_turns,
            result_text,
            cost,
            usage,
            model,
            ..
        }) => ChatEvent::Result {
            session_id,
            duration_ms: if duration_ms == 0 {
                elapsed_ms
            } else {
                duration_ms
            },
            cost_usd,
            subtype,
            is_error,
            num_turns,
            result_text,
            cost,
            usage,
            model,
            stop_reason: Some("interrupted".to_string()),
            error_code: None,
            error_message: None,
        },
        _ => ChatEvent::Result {
            session_id: String::new(),
            duration_ms: elapsed_ms,
            cost_usd: None,
            subtype: INTERRUPTED_SUBTYPE.to_string(),
            is_error: false,
            num_turns: None,
            result_text: None,
            cost: None,
            usage: None,
            model: None,
            stop_reason: Some("interrupted".to_string()),
            error_code: None,
            error_message: None,
        },
    };
    stored_interrupted_result(&ended).unwrap_or(ended)
}

/// `subtype` of the `result` of a turn stopped by a person, as stored: the one the native
/// provider gives (`done(Interrupted, "interrupted", false)`), neither a success nor an error.
pub(crate) const INTERRUPTED_SUBTYPE: &str = "interrupted";

/// The stored form of a `result` whose `stop_reason` is `interrupted`, `None` for any other
/// event. The Claude CLI answers an interrupt with `error_during_execution` and
/// `is_error: true`, often with an empty `result`: stored as is, the reloaded thread shows an
/// empty error block under every stopped turn while the live client saw none (follow-up
/// review of #711, finding 2). Stored instead with the native provider's neutral form —
/// `subtype: interrupted`, `is_error: false` — by both engines (the legacy loop through
/// [`interrupted_turn_end`], the agent runtime when it persists a `result`), so a thread
/// reloads the same whichever engine ran it. Only the STORED copy: what is broadcast live
/// (the runner and the sub-agents read `is_error` there) keeps the provider's form.
pub(crate) fn stored_interrupted_result(event: &ChatEvent) -> Option<ChatEvent> {
    match event {
        ChatEvent::Result {
            session_id,
            duration_ms,
            cost_usd,
            num_turns,
            result_text,
            cost,
            usage,
            model,
            stop_reason,
            ..
        } if stop_reason.as_deref() == Some("interrupted") => Some(ChatEvent::Result {
            session_id: session_id.clone(),
            duration_ms: *duration_ms,
            cost_usd: *cost_usd,
            subtype: INTERRUPTED_SUBTYPE.to_string(),
            is_error: false,
            num_turns: *num_turns,
            result_text: result_text.clone().filter(|t| !t.trim().is_empty()),
            cost: cost.clone(),
            usage: usage.clone(),
            model: model.clone(),
            stop_reason: stop_reason.clone(),
            error_code: None,
            error_message: None,
        }),
        _ => None,
    }
}

/// The sentinel a config/default uses to mean "let the harness choose the model", NOT a
/// model name. `chat.default_model: auto` reaches a session as the model string `"auto"`;
/// passed verbatim it becomes `--model auto`, which the Claude Code CLI rejects
/// (`[claude-code:unrecognized_model]`) — every turn then fails with "There's an issue with
/// the selected model (auto)". Treated as the sentinel, the model is left unset: the CLI
/// (and nexus) use the account's own default, and PO's cognitive router still overrides it
/// when a pool/mode is configured. Case-insensitive, trimmed.
pub(crate) fn is_auto_model(model: &str) -> bool {
    model.trim().eq_ignore_ascii_case("auto")
}

/// Whether an opening is decided by PO because the SERVER default is `auto`.
///
/// `chat.default_model: auto` is the operator's standing choice of Auto: a pilot
/// conversation that names nothing (no provider, no model, no pool, no mode, no alias, no
/// run) must then be routed by PO like one the user put in Auto from the menu — NOT left to
/// the declared rules, which (with no primary configured) open Claude Code with the `auto`
/// sentinel, i.e. the CLI's own account default, outside the person's Auto list and never
/// re-decided (measured 25/10/2026: settings `executors_only` + stage `auto` + default
/// `auto` → opening decision `applied: false`, session model `auto`, CLI on Opus 4.6).
///
/// A model named `auto` counts as nothing named: it is the sentinel, never a model. An
/// executor (spawned, delegated, runner) or an opening PO already routes (`route_auto`) is
/// not concerned: its routing is decided by its parent or by the settings' executor mode.
pub(crate) fn opens_on_server_default_auto(default_model: &str, request: &ChatRequest) -> bool {
    is_auto_model(default_model)
        && request.routing_mode.is_none()
        && request.routing_pool.is_none()
        && request.provider.is_none()
        && request
            .model
            .as_deref()
            .is_none_or(|m| m.trim().is_empty() || is_auto_model(m))
        && request.task_alias.is_none()
        && request.persona_alias.is_none()
        && request.run_provider.is_none()
        && request.spawned_by.is_none()
        && request.runner_context.is_none()
        && !request.route_auto
}

/// The opening request as the router and the session record read it when
/// [`opens_on_server_default_auto`]: the conversation is in Auto (`routing_mode: full`, so the
/// opening decision is taken at the `auto` stage in `full`, the person's Auto list applies,
/// the record says `full` and the per-turn router starts in Auto) and the `auto` sentinel is
/// not a named model. `None` when the request already says what it wants.
pub(crate) fn routed_by_server_default(
    default_model: &str,
    request: &ChatRequest,
) -> Option<ChatRequest> {
    opens_on_server_default_auto(default_model, request).then(|| {
        let mut routed = request.clone();
        routed.routing_mode = Some(super::provider::cognitive::ConversationRoutingMode::Full);
        routed.model = None;
        routed
    })
}

/// Opens (or resumes) a session with `spec` through `open`. A provider that learned,
/// between the reading of its capabilities and this opening, that it takes no MCP
/// server per session (two first sessions opened at once on an ACP agent that refuses
/// them: the other one taught it) answers `Unsupported { per_session_mcp }`: the
/// session is asked ONCE more without its MCP servers. `true` beside the session: it
/// was opened without them (it lacks the PO tools, [`PO_TOOLS_FEATURE`]).
///
/// Only an ACP agent (`kind`) is asked again: its spec holds no server but the PO one
/// (nothing else is dropped without being said). Any other kind's refusal is returned
/// as it is.
///
/// [`PO_TOOLS_FEATURE`]: super::agent_runtime::PO_TOOLS_FEATURE
pub(crate) async fn open_dropping_refused_mcp<F, Fut>(
    kind: nexus_claude::agent::ProviderKind,
    spec: nexus_claude::agent::SessionSpec,
    open: F,
) -> std::result::Result<
    (Arc<dyn nexus_claude::agent::AgentSession>, bool),
    nexus_claude::agent::ProviderError,
>
where
    F: Fn(nexus_claude::agent::SessionSpec) -> Fut,
    Fut: std::future::Future<
        Output = std::result::Result<
            Arc<dyn nexus_claude::agent::AgentSession>,
            nexus_claude::agent::ProviderError,
        >,
    >,
{
    let acp = kind == nexus_claude::agent::ProviderKind::Acp;
    let without = (acp && !spec.mcp_servers.is_empty()).then(|| {
        let mut without = spec.clone();
        without.mcp_servers.clear();
        without
    });
    match (open(spec).await, without) {
        (Err(nexus_claude::agent::ProviderError::Unsupported { capability }), Some(without))
            if capability == "per_session_mcp" =>
        {
            open(without).await.map(|session| (session, true))
        }
        (outcome, _) => outcome.map(|session| (session, false)),
    }
}

/// Whether a session whose spec carried MCP servers (`requested`: the PO server,
/// with the token minted for it) came out WITHOUT them: the host opened it again
/// without them (`dropped`, [`open_dropping_refused_mcp`]), or the provider did on its
/// own — an ACP agent that refuses `mcpServers` at `session/new` is asked again
/// without them by nexus, which only says so in the session's capabilities
/// (`per_session_mcp: false`). Those are the SESSION's (nexus #87): a concurrent opening
/// of the same instance that was refused does not turn a session that kept its servers to
/// `false`, so its token is not revoked. Either way the token went to the agent in the refused
/// request and nothing will ever present it legitimately: it is revoked.
pub(crate) fn opened_without_mcp(
    requested: bool,
    dropped: bool,
    session: &nexus_claude::agent::Capabilities,
) -> bool {
    requested && (dropped || !session.per_session_mcp)
}

/// What an opening (or a resume) of the agent engine has set up before its session
/// is live — the PO token minted for its spec, its per-turn router — undone when the
/// opening stops short of [`ChatManager::finish_agent_open`]: an authorization
/// refused, a spec that cannot be built, a provider unavailable or that refuses the
/// session, a window too small, or the opening future dropped. [`Self::disarm`]
/// once the session is handed over.
///
/// Undoes only what THIS opening set up (review of #684, finding 2): the token whose
/// `jti` it minted ([`Self::minted`]), the router it registered when that router is
/// still the session's ([`Self::registered`], `Arc::ptr_eq`). A concurrent resume of
/// the same session that succeeded keeps its own. Resumes of one session are also
/// serialised ([`SessionLocks`]): the session's `nexus-tools` token is kept per
/// session, not per minting, and a new PO token supersedes the session's others.
struct OpeningCleanup {
    session_id: String,
    turn_routing: Arc<super::agent_hooks::TurnRouting>,
    /// The router this opening registered.
    router: Option<Arc<super::agent_hooks::TurnRouter>>,
    /// The `jti` of the PO token minted for this opening's spec: `None` until the spec
    /// is built (a token minted by a build that failed is then the session's only one).
    jti: Option<String>,
    armed: bool,
}

impl OpeningCleanup {
    fn new(session_id: &str, turn_routing: &Arc<super::agent_hooks::TurnRouting>) -> Self {
        Self {
            session_id: session_id.to_string(),
            turn_routing: Arc::clone(turn_routing),
            router: None,
            jti: None,
            armed: true,
        }
    }

    /// The router this opening registered (`None`: routing is not configured).
    fn registered(&mut self, router: Option<Arc<super::agent_hooks::TurnRouter>>) {
        self.router = router;
    }

    /// The spec this opening built: the `jti` of the PO token it carries.
    fn minted(&mut self, spec: &nexus_claude::agent::SessionSpec, secret: Option<&str>) {
        self.jti = minted_jti(spec, secret);
    }

    /// Revokes the PO token of this opening (the session's `nexus-tools` token with
    /// it); every token of the session when the opening never knew its own.
    fn revoke_token(&self) {
        match self.jti.as_deref() {
            Some(jti) => {
                crate::auth::agent_tokens::revoke_minted(&self.session_id, jti);
            }
            None => {
                crate::auth::agent_tokens::revoke_session(&self.session_id);
            }
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for OpeningCleanup {
    fn drop(&mut self) {
        if self.armed {
            // Nothing will ever use this opening's token.
            self.revoke_token();
            if let Some(router) = &self.router {
                self.turn_routing.remove_if(&self.session_id, router);
            }
        }
    }
}

/// The `jti` of the PO token `spec` carries (its project-orchestrator server), read
/// with the server's signing key. `None` without a key, a server or a token.
pub(crate) fn minted_jti(
    spec: &nexus_claude::agent::SessionSpec,
    secret: Option<&str>,
) -> Option<String> {
    let secret = secret?;
    match spec.mcp_servers.get(PO_MCP_SERVER) {
        Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => env
            .get("PO_AUTH_TOKEN")
            .and_then(|token| crate::auth::jwt::decode_jwt(token, secret).ok())
            .and_then(|claims| claims.jti),
        _ => None,
    }
}

/// One lock per session id, held for as long as someone holds its guard: resumes of
/// one session run one after the other. An entry whose lock nobody holds is dropped.
#[derive(Default)]
pub(crate) struct SessionLocks(
    std::sync::Mutex<HashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>>,
);

impl SessionLocks {
    /// Waits for the session's lock.
    pub(crate) async fn lock(&self, session_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.0.lock().unwrap_or_else(|e| e.into_inner());
            locks.retain(|_, lock| lock.strong_count() > 0);
            match locks.get(session_id).and_then(std::sync::Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(tokio::sync::Mutex::new(()));
                    locks.insert(session_id.to_string(), Arc::downgrade(&lock));
                    lock
                }
            }
        };
        lock.lock_owned().await
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        let mut locks = self.0.lock().unwrap_or_else(|e| e.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        locks.len()
    }
}

/// Where a resume starts from: the live handle of the session (or none) as the caller saw
/// it before it awaited anything ([`ChatManager::resume_point`]). Opaque: only the manager
/// reads it, under the session's resume lock.
pub struct ResumeFrom(Option<Arc<super::agent_runtime::AgentSessionHandle>>);

/// Test-only pause at the start of `try_remote_send`, keyed by session id: a test makes a
/// message wait there (where a real deployment waits for the NATS RPC timeout) while
/// another resume of the same session completes.
#[cfg(test)]
pub(crate) mod remote_send_hook {
    use std::collections::HashMap;
    use std::sync::{Arc, LazyLock, Mutex};
    use tokio::sync::{oneshot, Notify};

    struct Hook {
        entered: Option<oneshot::Sender<()>>,
        release: Arc<Notify>,
    }

    static HOOKS: LazyLock<Mutex<HashMap<String, Hook>>> = LazyLock::new(Default::default);

    /// Arms a pause for `session_id`: the receiver fires when a `try_remote_send` reaches
    /// it; that call then waits until the returned `Notify` is notified. One use.
    pub(crate) fn arm(session_id: &str) -> (oneshot::Receiver<()>, Arc<Notify>) {
        let (tx, rx) = oneshot::channel();
        let release = Arc::new(Notify::new());
        HOOKS.lock().unwrap().insert(
            session_id.to_string(),
            Hook {
                entered: Some(tx),
                release: Arc::clone(&release),
            },
        );
        (rx, release)
    }

    pub(super) async fn wait(session_id: &str) {
        let hook = HOOKS.lock().unwrap().remove(session_id);
        if let Some(mut hook) = hook {
            if let Some(entered) = hook.entered.take() {
                let _ = entered.send(());
            }
            hook.release.notified().await;
        }
    }
}

/// The window (tokens) a Claude Code session is budgeted for when nobody knows its
/// model's window (CTX4, `ChatManager::session_context_window`): 200K, the smallest
/// window of the Claude models the CLI serves (some have 1M). It gives the PO prompt's
/// dynamic context and the per-turn enrichment the 40 000-character budget Claude
/// Code sessions had before CTX4 (`composer::compute_dynamic_budget` clamps any
/// window from ~67K tokens up to 40 000), so an unreachable catalog (OAuth login, no
/// API key) does not cut the main path to the unknown-window 10 000.
pub(crate) const CLAUDE_CODE_FLOOR_WINDOW: u64 = 200_000;

/// The context window (tokens) frozen in a session's capability snapshot (the
/// `capabilities` JSON stored at its opening: `{"context_window": {"value": N, ...}}`),
/// `None` when there is no snapshot, it is unreadable, or it states no window.
pub(crate) fn snapshot_context_window(snapshot: Option<&str>) -> Option<u64> {
    let caps: serde_json::Value = serde_json::from_str(snapshot?).ok()?;
    caps.get("context_window")?
        .get("value")?
        .as_u64()
        .filter(|w| *w > 0)
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

/// How long a session must have done nothing before a full table may close it for a new one.
const RECLAIM_IDLE_GRACE: Duration = Duration::from_secs(120);

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
        embeddings: Option<&Arc<dyn crate::embeddings::EmbeddingProvider>>,
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
        // Hybrid retrieval (BM25 ∪ dense cosine). With no provider the stage keeps
        // its lexical-only behaviour, unchanged.
        let ki_stage = if let Some(provider) = embeddings {
            ki_stage.with_embeddings(provider.clone())
        } else {
            ki_stage
        };
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
        // Controlled by ENRICHMENT_REFLEX env var (default: true). With an embedding
        // provider it also recalls the closed sessions whose first message is close to
        // this one (the session episodes); without one, scars and co-changes only.
        let reflex_stage = crate::reflex::stage::ReflexStage::new(graph.clone());
        let reflex_stage = if let Some(provider) = embeddings {
            reflex_stage.with_embeddings(provider.clone())
        } else {
            reflex_stage
        };
        pipeline.add_parallel_stage(Box::new(reflex_stage));
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
            Self::build_enrichment_pipeline(&graph, &search, None, None, None, None);
        let agent_runtime = Arc::new(
            super::agent_runtime::AgentRuntime::new(graph.clone())
                .declaring_read_only_mcp_tools(config.read_only_mcp_tools.clone()),
        );
        let builtin_cli_path: super::agent_runtime::ClaudeCliPath =
            Arc::new(std::sync::RwLock::new(config.claude_cli_path.clone()));
        let provider_source: Arc<dyn super::agent_runtime::ProviderSource> = Arc::new(
            super::agent_runtime::BuiltinProviders::following(Arc::clone(&builtin_cli_path)),
        );
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            embeddings: None,
            agent_runtime,
            provider_source,
            builtin_cli_path,
            recovery_resumes_claude_code: false,
            cognitive_routing: None,
            open_decisions: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            objective_ledgers: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            claude_cli: super::cli_version::ClaudeCliProbe::default(),
            native_cache: Arc::new(RwLock::new(HashMap::new())),
            native_probers: Arc::new(RwLock::new(HashMap::new())),
            context_injector: None,
            memory_config: None,
            event_emitter: None,
            vault: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
            host_home: dirs::home_dir(),
            env_config,
            enrichment_pipeline,
            trajectory_collector: Arc::new(std::sync::RwLock::new(None)),
            reasoning_engine: None,
            neural_routing_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dual_track_router: Arc::new(std::sync::RwLock::new(None)),
            nn_router: None,
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            turn_routing: Arc::new(super::agent_hooks::TurnRouting::default()),
            resume_locks: Arc::new(SessionLocks::default()),
            refs_v1: crate::refs::flag::from_env(),
            document_store: crate::documents::store::DocumentStore::new(
                crate::documents::store::default_storage_dir(),
            ),
            native_transcripts: None,
            acp: Default::default(),
            anchor_mode: super::anchor_resolver::AnchorContextMode::from_env(),
            anchor_cache: Arc::default(),
            held_slots: Default::default(),
            self_ref: Default::default(),
            agent_modes: Default::default(),
            subagent_slots: Default::default(),
            child_openings: Default::default(),
            live_children_cap: std::sync::atomic::AtomicUsize::new(
                super::envelope::MAX_LIVE_CHILDREN,
            ),
            deferred_tools: std::sync::RwLock::new(Arc::new(
                super::deferred_tools::default_deferred_tools(),
            )),
            mirrors: Default::default(),
            subagent_mirrors: true,
            segment_taps: Default::default(),
            session_pool: Arc::new(super::session_pool::SessionPool::from_env()),
            cancelled_parents: Default::default(),
            team_msg_rate: Default::default(),
            released_done: Default::default(),
            #[cfg(test)]
            release_hook: Default::default(),
            codex: Default::default(),
            provider_limits: Arc::new(super::provider::limits::Gate::unlimited()),
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
            Self::build_enrichment_pipeline(&graph, &search, None, None, None, None);
        let agent_runtime = Arc::new(
            super::agent_runtime::AgentRuntime::new(graph.clone())
                .declaring_read_only_mcp_tools(config.read_only_mcp_tools.clone()),
        );
        let builtin_cli_path: super::agent_runtime::ClaudeCliPath =
            Arc::new(std::sync::RwLock::new(config.claude_cli_path.clone()));
        let provider_source: Arc<dyn super::agent_runtime::ProviderSource> = Arc::new(
            super::agent_runtime::BuiltinProviders::following(Arc::clone(&builtin_cli_path)),
        );
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            embeddings: None,
            agent_runtime,
            provider_source,
            builtin_cli_path,
            recovery_resumes_claude_code: false,
            cognitive_routing: None,
            open_decisions: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            objective_ledgers: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            claude_cli: super::cli_version::ClaudeCliProbe::default(),
            native_cache: Arc::new(RwLock::new(HashMap::new())),
            native_probers: Arc::new(RwLock::new(HashMap::new())),
            context_injector,
            memory_config: Some(memory_config),
            event_emitter: None,
            vault: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
            host_home: dirs::home_dir(),
            env_config,
            enrichment_pipeline,
            trajectory_collector: Arc::new(std::sync::RwLock::new(None)),
            reasoning_engine: None,
            neural_routing_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dual_track_router: Arc::new(std::sync::RwLock::new(None)),
            nn_router: None,
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            turn_routing: Arc::new(super::agent_hooks::TurnRouting::default()),
            resume_locks: Arc::new(SessionLocks::default()),
            refs_v1: crate::refs::flag::from_env(),
            document_store: crate::documents::store::DocumentStore::new(
                crate::documents::store::default_storage_dir(),
            ),
            native_transcripts: None,
            acp: Default::default(),
            anchor_mode: super::anchor_resolver::AnchorContextMode::from_env(),
            anchor_cache: Arc::default(),
            held_slots: Default::default(),
            self_ref: Default::default(),
            agent_modes: Default::default(),
            subagent_slots: Default::default(),
            child_openings: Default::default(),
            live_children_cap: std::sync::atomic::AtomicUsize::new(
                super::envelope::MAX_LIVE_CHILDREN,
            ),
            deferred_tools: std::sync::RwLock::new(Arc::new(
                super::deferred_tools::default_deferred_tools(),
            )),
            mirrors: Default::default(),
            subagent_mirrors: true,
            segment_taps: Default::default(),
            session_pool: Arc::new(super::session_pool::SessionPool::from_env()),
            cancelled_parents: Default::default(),
            team_msg_rate: Default::default(),
            released_done: Default::default(),
            #[cfg(test)]
            release_hook: Default::default(),
            codex: Default::default(),
            provider_limits: Arc::new(super::provider::limits::Gate::unlimited()),
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

    /// Put the dedicated `HOME` of each ACP instance under `root` (`<root>/<instance>/home`)
    /// instead of nexus' default in the user's data directory. Applies to the providers
    /// built afterwards; the tests use it so that nothing is written in the real one.
    pub fn with_acp_homes(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.acp.homes = Some(root.into());
        self
    }

    /// Declare the ACP agents here instead of `CHAT_PROVIDER_ACP_COMMANDS` (the tests:
    /// the process environment is shared by every test running at the same time).
    pub fn with_acp_commands(
        mut self,
        commands: std::collections::BTreeMap<String, super::provider::settings::AcpCommand>,
    ) -> Self {
        self.acp.commands = Some(commands);
        self
    }

    /// Override the `codex` binary path for tests (`fake_codex`), so that tests neither
    /// need `codex` on `PATH` nor depend on a real Codex login.
    pub fn with_codex_program(mut self, program: impl Into<std::path::PathBuf>) -> Self {
        self.codex.program = Some(program.into());
        self
    }

    /// Override the `CODEX_HOME` root for tests: each instance gets
    /// `<root>/<instance-id>`, so no two instances collide and nothing is written in
    /// the user's real Codex data directory.
    pub fn with_codex_home(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.codex.home = Some(root.into());
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
    /// Read by the sub-agent opening too (AUTO-SUB, review point 4).
    pub(crate) async fn model_is_pinned(&self, session_id: &str) -> bool {
        matches!(
            self.graph
                .get_llm_setting(&Self::pin_scope(session_id), Self::MODEL_PIN_KEY)
                .await,
            Ok(Some(value)) if value == "true"
        )
    }

    /// The models of `provider_id` a pool lets the per-turn decision choose among: `None`
    /// without a pool (or with one model only: strict, a pin), `Some(empty)` when no ticked
    /// model is on this provider (the conversation then stays on its model here; a pool
    /// ticked across providers may still move it to another provider before a turn,
    /// decision R-S2, `agent_hooks::plan_provider_move`).
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
        // The conversation's segments and boundaries are persisted on a confirmed boundary,
        // when the process wired a graph-backed segment store (`CognitiveRouting.segment_store`).
        // `None` keeps them in the session's segmenter only.
        let segment_store = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.segment_store.clone());
        // S4: the conversation's segmentation, read once here. A conversation reopened after
        // a restart continues where it stopped; a fresh one has nothing to resume. Bounded,
        // and never fatal: `read_segments_snapshot` starts a fresh segmentation rather than
        // delaying or failing the turn.
        //
        // A provider move carries the live segmentation in the opening (fresher than the
        // store, which is written on a confirmed boundary only): prefer it, so a move never
        // resumes from a stale store entry.
        let segments = match turn.segments.clone() {
            Some(snapshot) => Some(snapshot),
            None => match segment_store.as_ref() {
                Some(store) => Self::read_segments_snapshot(store, session_id).await,
                None => None,
            },
        };
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
                    settings.mode = mode.pilot_mode();
                }
                settings
            }
            Err(error) => {
                warn!(session_id, error = %error, "routing settings unreadable: no per-turn routing");
                return None;
            }
        };
        // AUTO-L: the person's list, read once here (a change applies from the next opening or
        // resume); the router applies it only while the conversation is Auto without a pool of
        // its own. Unreadable = every model, never a refused turn.
        let auto_models = match turn.owner {
            Some(person) => {
                super::provider::cognitive::auto_models::load(self.graph.as_ref(), person)
                    .await
                    .unwrap_or_else(|error| {
                        warn!(session_id, %error, "auto_models unreadable: every model applies");
                        None
                    })
            }
            _ => None,
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
                // R-S2: a pool ticked across two providers or more (the menu's `mixed`)
                // makes the provider it names its PILOT, not an imposed provider: PO may
                // move the conversation to the other provider, like `full` restricted to
                // the pool. A `full` conversation keeps its named provider explicit
                // (decision R-S4, `routes_among_pool`).
                provider_imposed: turn.provider_imposed
                    && !turn.moved_in
                    && !(turn.routing_mode
                        != Some(super::provider::cognitive::ConversationRoutingMode::Full)
                        && super::types::pool_spans_providers(turn.routing_pool.as_deref())),
                allowed_models,
                routing_pool: turn.routing_pool.clone(),
                auto_models,
                // Auto, or two models ticked or more: the user's choice for THIS
                // conversation, decided at the `auto` stage (decision R-S1).
                conversation_routes: super::provider::cognitive::conversation_routes(
                    turn.routing_mode,
                    turn.routing_pool.as_ref().map_or(0, Vec::len),
                ),
                current_model: model.to_owned(),
                next_turn: turn.next_turn,
                moved_in: turn.moved_in,
                segment_store,
                segments,
            },
        ));
        router.set_last_message(turn.message);
        // R-S3: a move is a change of model; its relayed turn guards the next one of its class.
        router.seed_relayed_turn();
        self.turn_routing.insert(session_id, Arc::clone(&router));
        Some(router)
    }

    /// Legacy engine: asks the same per-turn decision as the agent engine and, when it
    /// names another model, sends the `set_model` control frame before the message.
    pub(crate) async fn apply_turn_directive(&self, session_id: &str, message: &str) {
        let Some(router) = self.turn_routing.get(session_id) else {
            return;
        };
        // The routing reads what the user typed, not the blocks around it, and whether an
        // image goes with it.
        router.set_turn_input(message);
        let typed = crate::refs::turn::visible_text(message);
        let message = typed.as_str();
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
                // The stored decision says the change did not go out (as a refused move).
                if let Some(decision) = router.decision_of_turn(ctx.turn_index) {
                    self.record_refused_change(decision, "set_model_failed")
                        .await;
                }
            }
        }
    }

    /// What the conversation can really do for its next turn, given its routing (F-R4,
    /// decision 11cefdb2): `snapshot_images` is what the capabilities snapshot of its
    /// opening model says (`None`: no snapshot stored). The live router answers
    /// (`agent_hooks::effective_images`); without one (the session is not open here, or no
    /// cognitive router is configured) the snapshot stands, said so (`no_router`). `None`:
    /// nothing to say (no snapshot, no router).
    pub(crate) async fn effective_capabilities(
        &self,
        session_id: &str,
        snapshot_images: Option<bool>,
    ) -> Option<super::types::EffectiveCapabilities> {
        use super::types::{EffectiveCapabilities, EffectiveCapability, EffectiveCause};
        let images = match self.turn_routing.get(session_id) {
            Some(router) => super::agent_hooks::effective_images(&router, snapshot_images).await?,
            None => EffectiveCapability::snapshot(snapshot_images?, EffectiveCause::NoRouter),
        };
        Some(EffectiveCapabilities { images })
    }

    /// Mode `full` (or a pool ticked across providers, R-S2), before a turn starts, on both
    /// engines: when the router's decision names
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
        // The routing reads what the user typed, not the blocks around it, and whether an
        // image goes with it.
        router.set_turn_input(message);
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
                // A move is a completed transition, not a per-turn decision left open: close
                // it so the store's open decisions are only the turns' own (an in-place switch
                // keeps the move's decision on the SAME thread, which otherwise reads as the
                // next turn's).
                if let Some(store) = self
                    .cognitive_routing
                    .as_ref()
                    .and_then(|r| r.store.clone())
                {
                    if let Err(error) = store
                        .set_outcome(decision.id, decision.outcome.unwrap_or_default())
                        .await
                    {
                        warn!(%decision.id, %error, "closing the move's routing decision failed");
                    }
                }
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

    /// A terminal provider failure — the chosen provider's retries exhausted, an
    /// out-of-credit refusal (402), or a non-retryable send error — must not end the
    /// chat. The conversation falls back to the NEXT eligible provider of its routing
    /// pool and resumes there, exactly as an explicit provider move
    /// ([`Self::move_provider_before_turn`]) would, through the same relay.
    ///
    /// The failed instance is first marked out of credit, so the next decision (and
    /// any later fallback) does not pick it again. Returns `true` when the conversation
    /// moved; `false` keeps the current error-ending behaviour (no eligible provider,
    /// or a relay the target refused). A fallback is attempted at most once per turn by
    /// the caller: it is a single relay, and a session already moved this turn is no
    /// longer live (the relay closed it), so it cannot be moved twice.
    pub(crate) async fn fallback_provider_after_failure(
        &self,
        session_id: &str,
        failed_provider: &str,
        message: &str,
    ) -> bool {
        if let Some(routing) = self.cognitive_routing.as_ref() {
            routing.mark_out_of_credit(failed_provider);
        }
        let Some(pick) = self.next_fallback_pick(session_id, failed_provider).await else {
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
            Ok(_) => {
                info!(
                    from_session = session_id,
                    to_provider = %pick.provider_id,
                    model = %pick.model,
                    failed = %failed_provider,
                    "the failed provider was abandoned: the conversation fell back to another provider"
                );
                true
            }
            Err(error) => {
                warn!(
                    session_id,
                    failed = %failed_provider,
                    to_provider = %pick.provider_id,
                    error = %error,
                    "the fallback provider move was refused: the conversation keeps its error"
                );
                false
            }
        }
    }

    /// The next provider the conversation may fall back to after `failed_provider`
    /// failed terminally. An explicit routing pool wins and is the ONLY place it may
    /// fall back to (never the project pool); a `primary`/pinned conversation is never
    /// moved at all. Only a conversation with no explicit pin consults the project
    /// pool (the consented models of every instance). Out-of-credit instances are
    /// excluded wherever the pick is made.
    async fn next_fallback_pick(
        &self,
        session_id: &str,
        failed_provider: &str,
    ) -> Option<super::provider::cognitive::decision::Pick> {
        use super::provider::cognitive::decision::Pick;
        let uuid = Uuid::parse_str(session_id).ok()?;
        let node = self.graph.get_chat_session(uuid).await.ok().flatten()?;
        let out_of_credit = |provider: &str| {
            self.cognitive_routing
                .as_ref()
                .is_some_and(|routing| routing.out_of_credit(provider))
        };
        // An explicit routing pool: fall back within it only, in the user's order. A
        // pool of one (a pin) offers nothing, and it never falls through to the
        // project pool.
        if let Some(pool) = node
            .routing_pool
            .as_deref()
            .and_then(|json| serde_json::from_str::<Vec<super::types::RoutingPoolEntry>>(json).ok())
        {
            return pool
                .into_iter()
                .find(|entry| entry.provider != failed_provider && !out_of_credit(&entry.provider))
                .map(|entry| Pick::new(entry.provider, entry.model));
        }
        // A `primary` conversation ("never substituted"), a provider the request
        // imposed (`routed_by: request`), or a model pinned by hand: never moved.
        if node.routing_mode.as_deref() == Some("primary")
            || node.routed_by.as_deref() == Some("request")
            || self.model_is_pinned(session_id).await
        {
            return None;
        }
        let pool = self.routing_pool_known(node.project_slug.as_deref()).await;
        let next = pool.into_iter().find(|facts| {
            facts.provider_id != failed_provider && !out_of_credit(&facts.provider_id)
        })?;
        Some(Pick::new(next.provider_id, next.model))
    }

    /// The decision of a model change that could not be sent says so: not applied, and why.
    async fn record_refused_change(
        &self,
        decision: super::provider::cognitive::decision::CognitiveDecision,
        code: &str,
    ) {
        let store = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.store.clone());
        record_not_changed(store.as_ref(), decision, code).await;
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
        // The sessions this manager opens report the limits their provider states to the same
        // registry the router reads (QUOTA).
        self.agent_runtime.set_quota(Arc::clone(&routing.quota));
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

    /// Hand the ADMISSION GATE (RL5) the process-wide instance built at boot.
    ///
    /// Every session counts against the SAME counters and breakers: a per-session
    /// gate would hand out the configured allowance once per session, which is
    /// exactly the waste RL5 exists to prevent.
    pub fn with_provider_limits(mut self, gate: Arc<super::provider::limits::Gate>) -> Self {
        self.agent_runtime.set_gate(Arc::clone(&gate));
        self.provider_limits = gate;
        self
    }

    /// The admission gate (RL5): what a caller consults BEFORE a provider call, and
    /// what `GET /api/chat/providers/{id}/limits` reads its measured numbers from.
    pub fn provider_limits(&self) -> &Arc<super::provider::limits::Gate> {
        &self.provider_limits
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
            self.embeddings.as_ref(),
        );
        drop(tc_guard);
        self
    }

    /// Attach the server's embedding provider to the knowledge injection stage.
    ///
    /// Rebuilds the enrichment pipeline so `KnowledgeInjectionStage` retrieves
    /// notes and decisions through **both** channels: BM25 (exact symbols, slugs,
    /// error codes) and dense cosine over the precomputed note embeddings
    /// (paraphrase, a question sharing no keyword with the note), fused by
    /// Reciprocal Rank Fusion. Only the query is embedded — one short string per
    /// turn — and inside the stage's existing per-query timeout.
    ///
    /// Omitting this call costs recall, not correctness: retrieval then stays
    /// BM25-only, exactly as it was before the dense channel existed.
    pub fn with_embeddings(
        mut self,
        provider: Arc<dyn crate::embeddings::EmbeddingProvider>,
    ) -> Self {
        self.embeddings = Some(provider.clone());
        let tc_guard = self.trajectory_collector.read().unwrap();
        self.enrichment_pipeline = Self::build_enrichment_pipeline(
            &self.graph,
            &self.search,
            self.reasoning_engine.as_ref(),
            tc_guard.as_ref(),
            Some(&self.mcp_registry),
            Some(&provider),
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
    /// Reads the existing file as a `crate::yaml::Value` tree, updates only the
    /// `chat.permissions` subtree, and writes back atomically (tmp + rename).
    /// This preserves all other config sections (auth, server, neo4j, etc.)
    /// without needing `Serialize` on those structs.
    fn persist_permission_to_yaml(
        yaml_path: &std::path::Path,
        permission: &super::config::PermissionConfig,
    ) -> Result<()> {
        use std::io::Write;

        // 1. Read existing YAML as a Value tree (or start from empty mapping)
        let mut doc: crate::yaml::Value = if yaml_path.exists() {
            let contents = std::fs::read_to_string(yaml_path)
                .with_context(|| format!("Reading {}", yaml_path.display()))?;
            crate::yaml::value_from_str(&contents)
                .with_context(|| format!("Parsing {}", yaml_path.display()))?
        } else {
            crate::yaml::Value::Mapping(crate::yaml::Mapping::new())
        };

        // 2. Ensure doc is a mapping
        let root = doc
            .as_mapping_mut()
            .ok_or_else(|| anyhow!("config.yaml root is not a YAML mapping"))?;

        // 3. Ensure chat section exists as a mapping
        let chat_key = "chat".to_string();
        if !root.contains_key(&chat_key) {
            root.insert(
                chat_key.clone(),
                crate::yaml::Value::Mapping(crate::yaml::Mapping::new()),
            );
        }
        let chat_section = root
            .get_mut(&chat_key)
            .and_then(|v| v.as_mapping_mut())
            .ok_or_else(|| anyhow!("chat section is not a YAML mapping"))?;

        // 4. Serialize PermissionConfig to a YAML Value and insert
        let perm_value = crate::yaml::to_value(permission)
            .context("Serializing PermissionConfig to YAML value")?;
        chat_section.insert("permissions".to_string(), perm_value);

        // 5. Serialize the full document back to YAML string
        let yaml_str =
            crate::yaml::to_string(&doc).context("Serializing config document to YAML")?;

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
        super::cli_version::set_configured_cli_path(path.clone());
        *self
            .builtin_cli_path
            .write()
            .unwrap_or_else(|e| e.into_inner()) = path.clone();
        self.env_config.write().await.claude_cli_path = path;
        self.claude_cli.forget();
    }

    /// Looks for the Claude Code CLI on the machine before a default chat is routed to it
    /// (B2): without it, the built-in instance is unhealthy for the resolver. What the
    /// server runs with; a manager built without it assumes the CLI is there.
    pub fn detecting_claude_cli(mut self) -> Self {
        self.claude_cli = super::cli_version::ClaudeCliProbe::detecting();
        self
    }

    /// A fixed answer to "is the Claude Code CLI on this server?": tests that must not
    /// depend on the machine they run on (CI has no `claude`).
    #[cfg(test)]
    pub(crate) fn with_claude_cli_present(mut self, present: bool) -> Self {
        self.claude_cli = super::cli_version::ClaudeCliProbe::Fixed(present);
        self
    }

    /// The recovery resumes cut Claude Code turns of the agent engine like any other,
    /// whatever their permission mode and stored conversation: for the recovery tests, whose fake provider answers as
    /// `claude-code` but is no CLI (the field `recovery_resumes_claude_code`).
    #[cfg(test)]
    pub(crate) fn resuming_claude_code_at_boot(mut self) -> Self {
        self.recovery_resumes_claude_code = true;
        self
    }

    /// Whether the built-in Claude Code instance can open a session here: its CLI is on
    /// the server (the configured path, else the usual places). Cached for a minute.
    pub async fn claude_code_usable(&self) -> bool {
        let configured = self.env_config.read().await.claude_cli_path.clone();
        self.claude_cli.present(configured.as_deref())
    }

    /// The CLI was just installed or upgraded: look for it again at the next opening.
    pub fn forget_claude_cli_detection(&self) {
        self.claude_cli.forget();
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
        let mut doc: crate::yaml::Value = if yaml_path.exists() {
            let contents = std::fs::read_to_string(yaml_path)
                .with_context(|| format!("Reading {}", yaml_path.display()))?;
            crate::yaml::value_from_str(&contents)
                .with_context(|| format!("Parsing {}", yaml_path.display()))?
        } else {
            crate::yaml::Value::Mapping(crate::yaml::Mapping::new())
        };

        // 2. Ensure doc is a mapping
        let root = doc
            .as_mapping_mut()
            .ok_or_else(|| anyhow!("config.yaml root is not a YAML mapping"))?;

        // 3. Ensure chat section exists as a mapping
        let chat_key = "chat".to_string();
        if !root.contains_key(&chat_key) {
            root.insert(
                chat_key.clone(),
                crate::yaml::Value::Mapping(crate::yaml::Mapping::new()),
            );
        }
        let chat_section = root
            .get_mut(&chat_key)
            .and_then(|v| v.as_mapping_mut())
            .ok_or_else(|| anyhow!("chat section is not a YAML mapping"))?;

        // 4. Set/remove individual fields surgically
        let path_key = "process_path".to_string();
        let cli_key = "claude_cli_path".to_string();
        let auto_key = "auto_update_cli".to_string();

        match &env.process_path {
            Some(val) => {
                chat_section.insert(path_key, crate::yaml::Value::String(val.clone()));
            }
            None => {
                chat_section.shift_remove(&path_key);
            }
        }
        match &env.claude_cli_path {
            Some(val) => {
                chat_section.insert(cli_key, crate::yaml::Value::String(val.clone()));
            }
            None => {
                chat_section.shift_remove(&cli_key);
            }
        }
        // Only write auto_update_cli when true (omit when false to keep YAML clean)
        if env.auto_update_cli {
            chat_section.insert(auto_key, crate::yaml::Value::Bool(true));
        } else {
            chat_section.shift_remove(&auto_key);
        }
        // Only write auto_update_app when false (default is true, so omit when true to keep YAML clean)
        let auto_app_key = "auto_update_app".to_string();
        if !env.auto_update_app {
            chat_section.insert(auto_app_key, crate::yaml::Value::Bool(false));
        } else {
            chat_section.shift_remove(&auto_app_key);
        }

        // 5. Serialize the full document back to YAML string
        let yaml_str =
            crate::yaml::to_string(&doc).context("Serializing config document to YAML")?;

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
                        // and tell the CLI at once, as every local Stop does: the turn waits
                        // for the CLI's result of the interrupted turn before it ends.
                        {
                            let sessions = active_sessions.read().await;
                            if let Some(session) = sessions.get(&session_id) {
                                session.interrupt_token.cancel();
                                if let Some(ref tx) = session.stdin_tx {
                                    let _ = tx.try_send(InteractiveClient::build_interrupt_json());
                                }
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
                                            nats.publish(&reply_to, payload).await
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
        let cognitive_routing = self.cognitive_routing.clone();

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
                                    let _ = nats.publish(&reply_to, payload).await;
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
                                let _ = nats.publish(&reply_to, payload).await;
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
                            // A permission granted on another instance: the same path as
                            // a local answer (deliver_permission_answer) — the control_response
                            // carries subtype, request_id and behavior, goes over stdin_tx
                            // (never the client lock, held by the turn that waits for it),
                            // and the request must still be waiting: one RPC, one write.
                            let answer = serde_json::from_str::<serde_json::Value>(message).ok();
                            let allow = answer
                                .as_ref()
                                .and_then(|v| v.get("allow").and_then(|a| a.as_bool()))
                                .unwrap_or(false);
                            let request_id = answer.as_ref().and_then(|v| {
                                v.get("request_id")
                                    .and_then(|r| r.as_str())
                                    .map(str::to_string)
                            });
                            let scope = answer
                                .as_ref()
                                .and_then(|v| v.get("scope").cloned())
                                .map(serde_json::from_value::<super::types::PermissionAnswerScope>);

                            info!(
                                session_id = %session_id,
                                request_id = ?request_id,
                                allow,
                                "NATS RPC: answering a permission request of the CLI"
                            );

                            match (request_id, scope) {
                                (None, _) => crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(
                                        "a permission answer needs its request_id".to_string(),
                                    ),
                                },
                                // A scope this instance does not know is refused, never read as once.
                                (Some(_), Some(Err(_))) => crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(PERMISSION_SCOPE_UNSUPPORTED_RPC.to_string()),
                                },
                                (Some(request_id), scope) => {
                                    let scope = scope.and_then(Result::ok).unwrap_or_default();
                                    let ctx = PermissionAnswer {
                                        active_sessions: &active_sessions,
                                        graph: &graph,
                                        nats: Some(&nats),
                                        event_emitter: &event_emitter,
                                    };
                                    match Self::deliver_permission_answer(
                                        ctx,
                                        &session_id,
                                        &request_id,
                                        allow,
                                        scope,
                                        true,
                                    )
                                    .await
                                    {
                                        Ok(()) => crate::events::ChatRpcResponse {
                                            success: true,
                                            error: None,
                                        },
                                        Err(PermissionDeliveryError::NotPending) => {
                                            crate::events::ChatRpcResponse {
                                                success: false,
                                                error: Some(PERMISSION_NOT_PENDING_RPC.to_string()),
                                            }
                                        }
                                        Err(PermissionDeliveryError::ScopeUnsupported(_)) => {
                                            crate::events::ChatRpcResponse {
                                                success: false,
                                                error: Some(
                                                    PERMISSION_SCOPE_UNSUPPORTED_RPC.to_string(),
                                                ),
                                            }
                                        }
                                        Err(e) => crate::events::ChatRpcResponse {
                                            success: false,
                                            error: Some(e.to_string()),
                                        },
                                    }
                                }
                            }
                        } else if request.message_type == "set_auto_continue" {
                            // Accepted and ignored, for older frontends: auto-continue is
                            // always on (decision 7b95342a).
                            debug!(
                                session_id = %session_id,
                                "NATS RPC: set_auto_continue ignored (always on)"
                            );
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
                            let cognitive_routing_clone = cognitive_routing.clone();

                            let active_sessions_for_turn = active_sessions.clone();
                            let session_id_for_turn = session_id.clone();
                            let turn_task = tokio::spawn(async move {
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
                                    cognitive_routing_clone,
                                )
                                .await;
                            });
                            Self::track_stream_task(
                                &active_sessions_for_turn,
                                &session_id_for_turn,
                                turn_task.abort_handle(),
                            )
                            .await;

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
                            if let Err(e) = nats.publish(&reply_to, payload).await {
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

    /// The model a moved conversation opens on: the explicit one when the user named it,
    /// otherwise the TARGET provider's own default (never the OLD provider's model name, which
    /// a provider like ACP would refuse). Mirrors [`Self::create_session_resolved`]'s model
    /// resolution for a non-Claude target.
    async fn resolve_target_model(&self, provider: &str, model: Option<&str>) -> Result<String> {
        if let Some(model) = model {
            return Ok(model.to_string());
        }
        if provider == super::provider::resolver::CLAUDE_CODE {
            return Ok(self.resolve_model(None));
        }
        super::provider::store::instance(self.graph.as_ref(), provider)
            .await?
            .and_then(|i| i.default_model)
            .ok_or_else(|| {
                anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                    "this provider instance has no default model: name one",
                ))
            })
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
    #[cfg(test)]
    pub(crate) async fn build_system_prompt_anchored(
        &self,
        place: super::neutral_place::ExecutionPlace,
        cwd: &str,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        context_window: Option<u64>,
        session_id: &str,
        scaffolding_override: Option<u8>,
    ) -> (String, std::collections::HashSet<String>) {
        self.build_system_prompt_anchored_for(
            &super::composer::PromptAudience::default(),
            place,
            cwd,
            project_slug,
            user_message,
            model,
            context_window,
            session_id,
            scaffolding_override,
        )
        .await
    }

    /// Build the anchored system prompt for a given audience
    /// (`PromptAudience::of` the session's `spawned_by` and provider): the child of a
    /// conversation gets its frame, an engine other than Claude Code its own sub-agent
    /// words. The production openings (open, resume, reopen) pass it; the default is
    /// a root session on Claude Code.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn build_system_prompt_anchored_for(
        &self,
        audience: &super::composer::PromptAudience,
        place: super::neutral_place::ExecutionPlace,
        cwd: &str,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        context_window: Option<u64>,
        session_id: &str,
        scaffolding_override: Option<u8>,
    ) -> (String, std::collections::HashSet<String>) {
        use super::anchor_resolver::AnchorContextMode;
        let (prompt, ids) = self
            .build_system_prompt_for(
                audience,
                project_slug,
                user_message,
                model,
                context_window,
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
    ///
    /// `context_window`: the model's window in tokens as the catalog knows it
    /// ([`Self::session_context_window`], `None` = unknown), resolved once per opening:
    /// it sizes the dynamic context (`composer::compute_dynamic_budget`).
    pub async fn build_system_prompt(
        &self,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        context_window: Option<u64>,
        session_id: Option<&str>,
        scaffolding_override: Option<u8>,
    ) -> (String, std::collections::HashSet<String>) {
        self.build_system_prompt_for(
            &super::composer::PromptAudience::default(),
            project_slug,
            user_message,
            model,
            context_window,
            session_id,
            scaffolding_override,
        )
        .await
    }

    /// [`Self::build_system_prompt`] for a given audience (a child of a conversation,
    /// an engine with its own sub-agent words); the default audience is a root session
    /// on Claude Code and leaves the prompt as it always was.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn build_system_prompt_for(
        &self,
        audience: &super::composer::PromptAudience,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        context_window: Option<u64>,
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
                iso_scene: audience.displays_scenes(),
                context_window,
                audience: audience.clone(),
                ..Default::default()
            };
            return (FsmPromptComposer::compose(&input), empty_ids);
        };

        // Fetch raw context from Neo4j
        let mut ctx = match fetch_project_context(&self.graph, slug).await {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!(
                    "Failed to fetch project context for '{}': {} — using base prompt only",
                    slug, e
                );
                let input = ComposerInput {
                    cite_refs: self.refs_v1,
                    iso_scene: audience.displays_scenes(),
                    context_window,
                    audience: audience.clone(),
                    ..Default::default()
                };
                return (FsmPromptComposer::compose(&input), empty_ids);
            }
        };

        // A restricted provider (Claude Code: Anthropic's safeguards) does not get the
        // project's provider-sensitive notes in its prompt. Injected, they make the
        // endpoint's safety classifier refuse EVERY turn of the session — even a bare
        // "salut" — because it reads the system prompt. The content is legitimate; it is
        // kept for a permissive provider, and left out here so a normal conversation is not
        // refused for context it never needed (decision 3fccf7c5; the auto-route of a
        // conversation that does need it is the planned follow-up).
        if audience.content_policy.is_restricted() {
            use super::content_policy::tags_are_sensitive;
            let before = ctx.guidelines.len()
                + ctx.gotchas.len()
                + ctx.global_guidelines.len()
                + ctx.global_gotchas.len();
            ctx.guidelines.retain(|n| !tags_are_sensitive(&n.tags));
            ctx.gotchas.retain(|n| !tags_are_sensitive(&n.tags));
            ctx.global_guidelines
                .retain(|n| !tags_are_sensitive(&n.tags));
            ctx.global_gotchas.retain(|n| !tags_are_sensitive(&n.tags));
            let omitted = before
                - (ctx.guidelines.len()
                    + ctx.gotchas.len()
                    + ctx.global_guidelines.len()
                    + ctx.global_gotchas.len());
            if omitted > 0 {
                info!(
                    session_id = session_id.unwrap_or(""),
                    slug,
                    omitted,
                    "left provider-sensitive notes out of a restricted provider's prompt"
                );
            }
        }

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
            context_window,
            message_embedding: message_embedding.as_ref(),
            external_tools_available,
            cite_refs: self.refs_v1,
            iso_scene: audience.displays_scenes(),
            audience: audience.clone(),
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
        self.live_session_snapshot_at(super::segment_tap::now_ms())
            .await
    }

    /// [`Self::live_session_snapshot`], the segment states classified at `now_ms` (the clock
    /// the model is given: `chat::segment_state` never reads one).
    pub(crate) async fn live_session_snapshot_at(&self, now_ms: u64) -> LiveSessionSnapshot {
        let mut snap = LiveSessionSnapshot::default();
        let mut inputs = Vec::new();
        let mut tasks = Vec::new();
        // Per session that gets a segment state: its broadcast (tapped) and its queue length.
        let mut channels: Vec<(Uuid, broadcast::Sender<ChatEvent>)> = Vec::new();
        let mut queues = Vec::new();
        let mut queued: HashMap<Uuid, usize> = HashMap::new();
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
                    channels.push((id, s.events_tx.clone()));
                    queues.push((id, s.pending_messages.clone()));
                }
            }
        }
        // The agent engine's sessions are live too, with the background tasks their
        // provider reports running (P4); their permissions are not tracked here (P11).
        for handle in self.agent_runtime.handles().await {
            if let Ok(id) = handle.session_id.parse::<Uuid>() {
                snap.live.insert(id);
                if handle.is_streaming.load(Ordering::SeqCst) {
                    snap.streaming.insert(id);
                }
                let counts = count_background_tasks(&handle.active_background_tasks().await);
                if counts != (0, 0) {
                    snap.background_tasks.insert(id, counts);
                }
                channels.push((id, handle.events_tx.clone()));
                queued.insert(id, handle.queued_len().await);
            }
        }
        for (id, queue) in queues {
            queued.insert(id, queue.lock().await.len());
        }
        // An open mirror of a Claude Code sub-agent works as long as its sub-agent (P18c).
        for id in self.mirrors.open_ids() {
            snap.live.insert(id);
            snap.streaming.insert(id);
        }
        // What each live session was given, and forget the sessions that ended.
        crate::chat::activation_registry::retain_live(&snap.live);
        for id in &snap.live {
            let got = crate::chat::activation_registry::get(*id);
            if !got.skills.is_empty() || !got.personas.is_empty() {
                snap.activations.insert(*id, got);
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
        self.classify_segments(&mut snap, channels, &queued, now_ms);
        snap
    }

    /// Fills `snap.states` (`chat::segment_state`): taps each session on its broadcast
    /// the first time it is seen (and asks its record, once, who it answers to), forgets
    /// the taps of sessions that left, then classifies each tapped session at one clock
    /// value. A mirror has no channel of its own here: it gets no state.
    fn classify_segments(
        &self,
        snap: &mut LiveSessionSnapshot,
        channels: Vec<(Uuid, broadcast::Sender<ChatEvent>)>,
        queued: &HashMap<Uuid, usize>,
        now: u64,
    ) {
        use super::segment_state::Activity;
        let taps = &self.segment_taps;
        taps.retain(&channels.iter().map(|(id, _)| *id).collect());
        for (id, tx) in &channels {
            if let Some(tap) = taps.ensure(*id, tx) {
                let graph = self.graph.clone();
                let id = *id;
                tokio::spawn(async move {
                    if let Ok(Some(node)) = graph.get_chat_session(id).await {
                        tap.set_addressee(super::segment_tap::addressee_of(
                            node.spawned_by.as_deref(),
                        ));
                    }
                    // What the session said before its tap existed (its opening turn, a
                    // thread resumed after a restart): the stored tail, with its times. A
                    // store that cannot answer leaves the tail as it is — no evidence.
                    let cap = super::segment_state::TAIL_CAP as i64;
                    let Ok(count) = graph.count_chat_events(id).await else {
                        return;
                    };
                    let Ok(records) = graph
                        .get_chat_events_paginated(id, (count - cap).max(0), cap)
                        .await
                    else {
                        return;
                    };
                    let older: Vec<(ChatEvent, u64)> = records
                        .iter()
                        .filter_map(|r| {
                            let event = serde_json::from_str::<ChatEvent>(&r.data).ok()?;
                            Some((event, r.created_at.timestamp_millis().max(0) as u64))
                        })
                        .collect();
                    tap.seed(&older);
                });
            }
        }
        for (id, _) in channels {
            let a = snap.activity_for(id);
            let activity = Activity {
                streaming: a.streaming,
                pending_permissions: a.pending_permissions,
                monitors: a.monitors,
                bash_tasks: a.bash_tasks,
                queued_messages: queued.get(&id).copied().unwrap_or(0),
                ..Default::default()
            };
            if let Some(verdict) = taps.classify(id, activity, now) {
                snap.states.insert(id, verdict);
            }
        }
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
        if let Some(mode) = self
            .active_sessions
            .read()
            .await
            .get(session_id)
            .and_then(|s| s.permission_mode.clone())
        {
            return Some(mode);
        }
        // A session of the agent engine (native, Codex, ACP): the mode last set on it.
        if !self.agent_runtime.owns(session_id).await {
            return None;
        }
        self.agent_modes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(session_id)
            .cloned()
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

    /// `dirs` clamped to what `parent_session_id` could already see when it was
    /// itself opened — its `cwd` and its `add_dirs` as STORED (frozen at its own
    /// creation, never re-resolved) (P18b, fix 1). Called only for a request whose
    /// `spawned_by` names a parent: a workspace added a project after the parent
    /// opened must not widen what a later child of that parent is handed, even
    /// though the workspace itself is validly inherited. A parent that cannot be
    /// read grants nothing: refusing a directory is recoverable, leaking one is not.
    async fn clamp_spawned_add_dirs(
        &self,
        dirs: Vec<String>,
        parent_session_id: &str,
    ) -> Vec<String> {
        let parent = match Uuid::parse_str(parent_session_id) {
            Ok(parent_uuid) => self
                .graph
                .get_chat_session(parent_uuid)
                .await
                .ok()
                .flatten(),
            Err(_) => None,
        };
        let Some(parent) = parent else {
            // Fails closed, and says so: the warn below would otherwise blame a widening
            // when the truth is that the parent could not be read at all.
            if !dirs.is_empty() {
                tracing::warn!(
                    parent = %parent_session_id,
                    dropped = ?dirs,
                    "a spawned session's parent is unreadable: none of its resolved add_dirs is granted"
                );
            }
            return Vec::new();
        };
        let mut roots = vec![parent.cwd];
        roots.extend(parent.add_dirs.unwrap_or_default());
        let dropped: Vec<&String> = dirs
            .iter()
            .filter(|d| !roots.iter().any(|r| super::envelope::is_within(r, d)))
            .collect();
        if !dropped.is_empty() {
            tracing::warn!(
                parent = %parent_session_id,
                dropped = ?dropped,
                "a spawned session's resolved add_dirs widened past its parent's roots; dropped"
            );
        }
        dirs.into_iter()
            .filter(|d| roots.iter().any(|r| super::envelope::is_within(r, d)))
            .collect()
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

    /// The owner of the session a relayed conversation leaves (`relay.from_session_id`), read
    /// from its node. `None` without a relay, or when it cannot be read.
    async fn relayed_owner(&self, relay: Option<&super::relay::RelayedFrom>) -> Option<String> {
        let from = Uuid::parse_str(&relay?.from_session_id).ok()?;
        self.graph
            .get_chat_session(from)
            .await
            .ok()
            .flatten()?
            .owner
    }

    /// The owner of the session `request` names as its parent (`spawned_by.parent_session_id`),
    /// read from the parent's node. `None` without a parent, or when it cannot be read: the
    /// child is then left as it always was without claims (P18b round 3, fix A).
    async fn spawning_parent_owner(&self, request: &ChatRequest) -> Option<String> {
        let parent = request
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)?
            .parent_session_id?;
        let parent = Uuid::parse_str(&parent).ok()?;
        self.graph
            .get_chat_session(parent)
            .await
            .ok()
            .flatten()?
            .owner
    }

    /// Whether the session `session_id` is recorded and was spawned by nothing (no
    /// `spawned_by`): a conversation a person opened. Only such a session may run sub-agents
    /// (P18, depth 1). An unknown or unreadable record answers `false`.
    pub(crate) async fn spawned_by_nothing(&self, session_id: &str) -> bool {
        let Ok(sid) = Uuid::parse_str(session_id) else {
            return false;
        };
        matches!(
            self.graph.get_chat_session(sid).await,
            Ok(Some(node)) if node.spawned_by.as_deref().is_none_or(|s| s.trim().is_empty())
        )
    }

    /// The lineage of a session something spawned (its record has a `spawned_by`): depth 1
    /// under its parent session, or under whatever spawned it when it names none. `None` for a
    /// session a person opened, or an unknown record.
    pub(crate) async fn spawned_lineage(
        &self,
        session_id: &str,
    ) -> Option<nexus_claude::agent::Lineage> {
        let sid = Uuid::parse_str(session_id).ok()?;
        let node = self.graph.get_chat_session(sid).await.ok()??;
        let spawned_by = node.spawned_by.filter(|s| !s.trim().is_empty())?;
        let ctx = parse_spawned_by(&spawned_by);
        let parent = ctx
            .as_ref()
            .and_then(|c| c.parent_session_id.clone())
            .unwrap_or_default();
        Some(nexus_claude::agent::Lineage {
            root_session_id: parent.clone(),
            parent_session_id: parent,
            depth: 1,
            spawned_by: ctx.map_or_else(|| "unknown".to_string(), |c| c.spawn_type),
        })
    }

    /// Whether a given session is currently streaming.  Used by `team::TeamHooks`.
    pub(crate) async fn session_is_streaming(&self, session_id: &str) -> bool {
        self.active_sessions
            .read()
            .await
            .get(session_id)
            .is_some_and(|s| s.is_streaming.load(Ordering::SeqCst))
    }

    /// All active sessions that are direct children of `parent_session_id`.
    ///
    /// "Direct child" = a session whose `spawned_by` field names `parent_session_id`.
    /// Returns only sessions that are currently in `active_sessions` (alive).
    pub(crate) async fn direct_children(&self, parent_session_id: &str) -> Vec<String> {
        let active: Vec<String> = self.active_sessions.read().await.keys().cloned().collect();

        let mut children = Vec::new();
        for id in active {
            if let Ok(uuid) = Uuid::parse_str(&id) {
                if let Ok(Some(node)) = self.graph.get_chat_session(uuid).await {
                    let is_child = node
                        .spawned_by
                        .as_deref()
                        .is_some_and(|s| s.contains(parent_session_id));
                    if is_child {
                        children.push(id);
                    }
                }
            }
        }
        children
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

    /// The person a session opened by these claims belongs to (`ChatSessionNode::owner`,
    /// P11b):
    /// - a signed-in person, with their session JWT or their MCP token, is the owner;
    /// - an agent session's token hands down the owner of the session it was minted for (a
    ///   delegated session belongs to the person behind its parent). When that parent
    ///   cannot be read, the token's own `sub` (the person it was minted for, copied by
    ///   `generate_session_token`) is the owner; a token minted for the server itself is
    ///   then REFUSED: a session is never left without an owner by a read that failed
    ///   (`owner` is written once, at open);
    /// - the server itself (a plan run, a protocol), an anonymous caller or no claims at
    ///   all: nobody.
    pub(crate) async fn session_owner(
        &self,
        claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<Option<String>> {
        let Some(claims) = claims else {
            return Ok(None);
        };
        if claims.is_agent_session() {
            let parent = crate::auth::jwt::agent_session_binding(claims)
                .and_then(|binding| binding.session_id.parse::<Uuid>().ok());
            let read = match parent {
                Some(parent) => self.graph.get_chat_session(parent).await.ok().flatten(),
                None => None,
            };
            if let Some(parent) = read {
                return Ok(parent.owner);
            }
            return match person_behind(claims) {
                Some(sub) => Ok(Some(sub.to_string())),
                None => Err(anyhow!(
                    "the session that opens this one cannot be read, and its token names no \
                     person: the new session would have no owner"
                )),
            };
        }
        Ok(person_behind(claims).map(str::to_string))
    }

    /// Where the lasting approvals of every person are kept (`chat::lasting_rules`),
    /// `None` when this server keeps none (`ChatConfig::lasting_rules_dir`).
    pub fn lasting_rule_store(&self) -> Option<super::lasting_rules::LastingRuleStore> {
        self.config
            .lasting_rules_dir
            .clone()
            .map(super::lasting_rules::LastingRuleStore::new)
    }

    /// The lasting rules a session of `owner` on `project`, running in `cwd`, keeps and
    /// obeys; `None` (no `always` offered) without a store, an owner, a project whose
    /// record can be read, or when `cwd` is not the project's root (symbolic links
    /// resolved): a rule is bound to that root (`LastingScope::of_session`).
    pub(crate) async fn lasting_scope(
        &self,
        owner: Option<&str>,
        project: Option<&str>,
        cwd: &str,
    ) -> Option<super::lasting_rules::LastingScope> {
        let store = self.lasting_rule_store()?;
        let (owner, slug) = (owner?.to_string(), project?.to_string());
        let root = self
            .graph
            .get_project_by_slug(&slug)
            .await
            .ok()
            .flatten()
            .map(|p| std::path::PathBuf::from(crate::expand_tilde(&p.root_path)))?;
        let cwd = std::path::PathBuf::from(crate::expand_tilde(cwd));
        tokio::task::spawn_blocking(move || {
            super::lasting_rules::LastingScope::of_session(
                Some(&store),
                Some(owner.as_str()),
                Some(slug.as_str()),
                Some(&root),
                &cwd,
            )
        })
        .await
        .ok()
        .flatten()
    }

    /// The grants of a session opened (`resumed: false`) or resumed here, in `project_dir`
    /// (its `.mcp.json` decides the read-only MCP declarations), keeping the lasting rules
    /// of `lasting`; `asker` is who asks its permission requests. Whether `claude --resume`
    /// restores the Bash tool's last directory is not proven, so a resumed session starts
    /// moved (no lasting rule bound to the project's root covers a command there; step
    /// 55f02275, P11b follow-up) unless it is known its shell never left the root
    /// ([`Self::resumed_shell_moved`]). The first move is recorded with the session.
    pub(crate) async fn session_grants_for(
        &self,
        session_id: &str,
        project_dir: Option<&std::path::Path>,
        lasting: Option<super::lasting_rules::LastingScope>,
        resumed: bool,
        asker: super::session_grants::Asker,
    ) -> super::session_grants::SessionGrants {
        let id = Uuid::parse_str(session_id).ok();
        let moved = resumed && self.resumed_shell_moved(session_id, id, asker).await;
        let recorder = id.map(|id| {
            let graph = self.graph.clone();
            super::session_grants::MoveRecorder::new(move || {
                let graph = graph.clone();
                let Ok(runtime) = tokio::runtime::Handle::try_current() else {
                    // A resume still finds the move in the session's events.
                    warn!(session_id = %id, "No runtime to record that the session's shell may have moved");
                    return;
                };
                runtime.spawn(async move {
                    if let Err(e) = graph.set_chat_session_shell_moved(id).await {
                        // A resume still finds the move in the session's events.
                        warn!(session_id = %id, error = %e, "Failed to record that the session's shell may have moved");
                    }
                });
            })
        });
        super::session_grants::SessionGrants::declaring(
            super::session_grants::declarations_for_project(
                &self.config.read_only_mcp_tools,
                project_dir,
            ),
        )
        .with_lasting(lasting)
        .with_shell_moved(moved)
        .recording_moves(recorder)
    }

    /// Whether a session being resumed may have its shell outside its project root. Moved
    /// unless every check says it never left:
    /// - a live session of this id here (a resume replacing it) whose shell moved: moved;
    /// - the stored flag (`shell_moved`): `true`, absent (a session recorded before the flag
    ///   existed: its lasting command rules are asked again once) or unreadable: moved;
    /// - `false` is only an optimisation that can lag (its write is in the background):
    ///   the session's stored events are replayed through the same predicate
    ///   ([`super::session_grants::history_moved_the_shell`]); a move there, an event that
    ///   cannot be read back, or a failed read: moved (review of #716, S1).
    async fn resumed_shell_moved(
        &self,
        session_id: &str,
        id: Option<Uuid>,
        asker: super::session_grants::Asker,
    ) -> bool {
        let Some(id) = id else {
            return true;
        };
        let live_moved = self.active_sessions.read().await.get(session_id).map(|s| {
            s.session_grants
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .shell_moved()
        });
        if live_moved == Some(true) {
            return true;
        }
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            if handle.shell_moved() {
                return true;
            }
        }
        match self.graph.get_chat_session_shell_moved(id).await {
            Ok(Some(false)) => {}
            _ => return true,
        }
        const PAGE: i64 = 500;
        let mut after = 0;
        loop {
            let page = match self.graph.get_chat_events(id, after, PAGE).await {
                Ok(page) => page,
                Err(e) => {
                    warn!(session_id, error = %e, "Cannot read the session's events back: its shell is taken as moved");
                    return true;
                }
            };
            let mut events = Vec::new();
            for record in &page {
                after = after.max(record.seq);
                if !matches!(
                    record.event_type.as_str(),
                    "tool_use" | "tool_use_input_resolved"
                ) {
                    continue;
                }
                match serde_json::from_str::<ChatEvent>(&record.data) {
                    Ok(event) => events.push(event),
                    Err(e) => {
                        warn!(session_id, seq = record.seq, error = %e, "A stored tool call cannot be read back: the session's shell is taken as moved");
                        return true;
                    }
                }
            }
            if super::session_grants::history_moved_the_shell(asker, &events) {
                return true;
            }
            if (page.len() as i64) < PAGE {
                return false;
            }
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

        // `auto` is a sentinel, not a model name (see [`is_auto_model`]): left unset the CLI
        // uses the account's default instead of a `--model auto` it would reject. A named
        // model is passed through.
        if !is_auto_model(model) {
            builder = builder.model(model);
        }

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
                            let mut result = match &t.content {
                                Some(ContentValue::Text(s)) => serde_json::Value::String(s.clone()),
                                Some(ContentValue::Structured(v)) => {
                                    serde_json::Value::Array(v.clone())
                                }
                                None => serde_json::Value::Null,
                            };
                            // A tool result is not the agent's words: a `#kind:uuid`
                            // inside it is data, and is never a citation of the agent.
                            crate::refs::cite::neutralize_value(&mut result);
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
                    error_code: None,
                    error_message: None,
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
                        request_id: None,
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
                            capabilities: Some(legacy_capabilities()),
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

    /// The ONE CRUD `created` event of a session this server just opened, whatever the
    /// engine (legacy CLI or agent runtime) and whoever asked (HTTP, MCP, an `Agent` call,
    /// a provider switch, the runner): the callers emit none of their own (lot review of
    /// #721, finding 1).
    fn emit_session_created(
        &self,
        session_id: Uuid,
        request: &ChatRequest,
        project_slug: Option<&str>,
    ) {
        if let Some(ref emitter) = self.event_emitter {
            emitter.emit_created(
                crate::events::EntityType::ChatSession,
                &session_id.to_string(),
                session_created_payload(request, project_slug),
                None,
            );
        }
    }

    /// The passive observer that mirrors the sub-agents of the Claude Code session
    /// `session_id` into read-only child conversations (`chat::mirror`, P18c), on the
    /// historical engine: what the CLI prints out of turn goes through `oob_listener`, which
    /// the tap does not see.
    pub(crate) fn mirror_tap(&self, session_id: &str) -> Option<Arc<super::mirror::MirrorTap>> {
        self.spawn_mirror_tap(session_id, false)
    }

    /// The same on the agent engine, whose out-of-turn pump shows the tap what the CLI
    /// prints between turns (`TurnServices::observe_out_of_turn`): a background
    /// sub-agent's mirror follows it until the CLI notifies its end.
    pub(crate) fn agent_mirror_tap(
        &self,
        session_id: &str,
    ) -> Option<Arc<super::mirror::MirrorTap>> {
        self.spawn_mirror_tap(session_id, true)
    }

    fn spawn_mirror_tap(
        &self,
        session_id: &str,
        follows_out_of_turn: bool,
    ) -> Option<Arc<super::mirror::MirrorTap>> {
        if !self.subagent_mirrors {
            return None;
        }
        Some(super::mirror::MirrorTap::spawn(
            session_id,
            super::mirror::MirrorDeps {
                graph: self.graph.clone(),
                event_emitter: self.event_emitter.clone(),
                nats: self.nats.clone(),
                registry: Arc::clone(&self.mirrors),
            },
            follows_out_of_turn,
        ))
    }

    /// No sub-agent mirror for the sessions this manager opens (tests: the parent's stream
    /// compared with and without the tap).
    #[cfg(test)]
    pub(crate) fn without_subagent_mirrors(mut self) -> Self {
        self.subagent_mirrors = false;
        self
    }

    /// Refuses a message to a read-only mirror of a Claude Code sub-agent
    /// ([`super::mirror::MirrorReadOnly`], 409 `mirror_read_only`) before anything sends it,
    /// resumes or proxies it. A session that cannot be read is not refused here (what
    /// follows reports it).
    pub async fn refuse_if_mirror(&self, session_id: &str) -> Result<()> {
        if self.mirrors.is_known_mirror(session_id) || self.mirrors.is_open(session_id) {
            return Err(super::mirror::MirrorReadOnly.into());
        }
        // A session live here is a normal one (a mirror has no process): no graph read on
        // the path of every write of a live session (a parent answering its sub-agent's
        // permission never waits on the store).
        if self.is_session_active(session_id).await {
            return Ok(());
        }
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            return Ok(());
        };
        match self.graph.get_chat_session(uuid).await {
            Ok(Some(node)) if node.mirror_of.is_some() => Err(super::mirror::MirrorReadOnly.into()),
            _ => Ok(()),
        }
    }

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
        self.create_session_relayed_at(request, relay, Uuid::new_v4())
            .await
    }

    /// [`Self::create_session_relayed`] under an id the caller has ALREADY answered with
    /// ([`Self::create_session_detached`]): everything the opening persists has to carry
    /// that id, since the caller's client is already streaming `/ws/chat/<id>`.
    async fn create_session_relayed_at(
        &self,
        request: &ChatRequest,
        relay: Option<&super::relay::RelayedFrom>,
        session_id: Uuid,
    ) -> Result<CreateSessionResponse> {
        // The menu's ticks, read once: one model ticked is strict, an empty pool is none.
        let settled = request.settled_routing();
        let request = settled.as_ref().unwrap_or(request);
        // A delegation (an agent caller, a spawned session) follows the settings' stage: it
        // cannot hand its own routing to PO (decision R-S1). Its Auto or pool is dropped, not
        // refused: the delegation still opens, routed as a session that asked nothing.
        let delegated = (request.spawned_by.is_some()
            && (request.routing_mode.is_some() || request.routing_pool.is_some()))
        .then(|| {
            warn!("a delegated session cannot choose its routing: routing_mode and routing_pool dropped");
            ChatRequest {
                routing_mode: None,
                routing_pool: None,
                ..request.clone()
            }
        });
        let request = delegated.as_ref().unwrap_or(request);
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

    /// Opens a session WITHOUT waiting for it (issue #1056): the caller gets the id now,
    /// and the opening — graph reads (owner, project, provider), CLI spawn, first message —
    /// goes on in a task of its own.
    ///
    /// WHY. `POST /api/chat/sessions` used to answer only once the WHOLE opening was over.
    /// Measured on 2026-11-01 (issue #1056): ~107 s went by before the session was even
    /// persisted — on a loaded machine, in the graph reads that precede it — while the MCP
    /// caller's HTTP budget is 120 s (`mcp::http_client`). A parent asking for a child
    /// through `chat(action: "send_message")` could then stay `streaming` for tens of
    /// minutes with neither a result nor an error.
    ///
    /// WHAT THE CALLER KEEPS. The id (a session's stream is `/ws/chat/<id>`, and it is
    /// known before anything is resolved), the place and the access, which the request
    /// alone decides. What it no longer gets is the REFUSAL as an HTTP error: a refused
    /// opening is reported as an `error` event in `notice_to` (the conversation that asked),
    /// since no response is left to carry it. That is the trade issue #1056 asks for — a
    /// bounded call, and the failure said where the conversation will read it.
    ///
    /// `opening` is the caller's in-flight ticket (P18-suite, point 6): it moves INTO the
    /// task, so the live-children cap counts this opening until it is over. Without it, a
    /// parent firing several detached openings would walk past its own cap.
    pub fn create_session_detached(
        self: &Arc<Self>,
        request: ChatRequest,
        notice_to: Option<String>,
        opening: Option<super::envelope::OpeningTicket>,
    ) -> CreateSessionResponse {
        let session_id = Uuid::new_v4();
        // What the request alone decides, answered now. The opening may narrow the access
        // further; it can never widen what is said here, and a session says its `access`
        // again on every read once it exists.
        let execution_place = if request.cwd.trim().is_empty()
            || super::neutral_place::is_neutral_path(&request.cwd)
        {
            super::neutral_place::ExecutionPlace::Neutral
        } else {
            super::neutral_place::ExecutionPlace::Project
        };
        // Belonging, not place: a session that names no project and no workspace is
        // read-only by default, whatever directory it runs in (`SessionAccess::for_open`).
        let unanchored = matches!(
            execution_place,
            super::neutral_place::ExecutionPlace::Neutral
        ) && request.project_slug.is_none()
            && request.workspace_slug.is_none();
        let access = super::provider::policy::SessionAccess::for_open(request.access, unanchored);
        let manager = Arc::clone(self);
        let spawned = request;
        let id = session_id.to_string();
        let id_in_task = id.clone();
        tokio::spawn(async move {
            // Held for the whole opening, dropped with this task.
            let _opening = opening;
            if let Err(e) = manager
                .create_session_relayed_at(&spawned, None, session_id)
                .await
            {
                warn!(
                    session_id = %id_in_task,
                    error = %e,
                    "a detached opening failed: the caller was already told the id"
                );
                manager
                    .emit_session_event(
                        notice_to.as_deref(),
                        ChatEvent::Error {
                            message: format!(
                                "Opening session {id_in_task} failed: {e}. Nothing is running \
                                 under that id; the conversation is free."
                            ),
                            parent_tool_use_id: None,
                            code: Some("session_open_failed".to_string()),
                            reason: None,
                            index: None,
                            request_id: None,
                        },
                    )
                    .await;
            }
        });
        CreateSessionResponse {
            session_id: id,
            stream_url: format!("/ws/chat/{session_id}"),
            execution_place,
            access,
            notices: vec![
                "this session is opening in the background: the id answers now and the \
                 conversation fills as the provider opens. A refused opening is reported as an \
                 error in the conversation that asked for it."
                    .to_string(),
            ],
        }
    }

    /// Persists AND broadcasts one event for a session of EITHER engine: the agent engine
    /// through its handle (which owns the sequence and the broadcast), the Claude Code
    /// engine through its live `ActiveSession`, and — for a session nobody holds any more —
    /// into the graph alone.
    ///
    /// Best effort by construction: a notice that fails to persist is logged and never fails
    /// the caller. `None` (no conversation to tell) is a no-op, not a mistake.
    pub(crate) async fn emit_session_event(&self, session_id: Option<&str>, event: ChatEvent) {
        let Some(session_id) = session_id.filter(|s| !s.trim().is_empty()) else {
            return;
        };
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            handle.emit(event).await;
            return;
        }
        let (tx, seq) = {
            let sessions = self.active_sessions.read().await;
            match sessions.get(session_id) {
                Some(s) => (
                    Some(s.events_tx.clone()),
                    Some(s.next_seq.fetch_add(1, Ordering::SeqCst)),
                ),
                None => (None, None),
            }
        };
        if let Some(tx) = tx {
            let _ = tx.send(event.clone());
        }
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            warn!(
                session_id,
                "a session notice names no session id: not persisted"
            );
            return;
        };
        // A session nobody holds: its stored sequence is the only authority there is.
        let seq = match seq {
            Some(seq) => seq,
            None => match self.graph.get_latest_chat_event_seq(uuid).await {
                Ok(seq) => seq + 1,
                Err(e) => {
                    warn!(session_id, error = %e, "cannot read the last sequence: notice not persisted");
                    return;
                }
            },
        };
        let record = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq,
            event_type: event.event_type().to_string(),
            data: serde_json::to_string(&event).unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        if let Err(e) = self.graph.store_chat_events(uuid, vec![record]).await {
            warn!(session_id, error = %e, "failed to persist a session notice (non-fatal)");
        }
    }

    /// [`Self::create_session_relayed`] once `request.cwd` is settled (a project's, or
    /// the neutral directory of `session_id`).
    async fn create_session_in_place(
        &self,
        request: &ChatRequest,
        relay: Option<&super::relay::RelayedFrom>,
        session_id: Uuid,
    ) -> Result<CreateSessionResponse> {
        let mut opened_on: Option<String> = None;
        let result = self
            .create_session_resolved(request, relay, session_id, &mut opened_on)
            .await;
        // A failure once the provider is resolved names THAT provider (B6), not the one the
        // HTTP layer would assume.
        result.map_err(|e| match opened_on {
            Some(provider_id) => super::provider::errors::on_provider(&provider_id, e),
            None => e,
        })
    }

    /// [`Self::create_session_in_place`]; `opened_on` receives the provider as soon as it
    /// is resolved.
    async fn create_session_resolved(
        &self,
        request: &ChatRequest,
        relay: Option<&super::relay::RelayedFrom>,
        session_id: Uuid,
        opened_on: &mut Option<String>,
    ) -> Result<CreateSessionResponse> {
        // `chat.default_model: auto` and nothing named: the conversation is PO's to route,
        // as if the user had put it in Auto (see `routed_by_server_default`). Read once here,
        // so the decision, the record (`routing_mode`) and the per-turn router agree.
        let routed = routed_by_server_default(&self.config.default_model, request);
        let request = routed.as_ref().unwrap_or(request);
        // Who the session belongs to, settled before anything is opened: a refusal (no owner
        // can be named for a delegated session) leaves nothing behind.
        let owner = match self.session_owner(request.user_claims.as_ref()).await? {
            // No claims at all, and a parent named: the child of a parent that was itself
            // reopened without its owner's claims (the restart recovery keeps none) belongs
            // to the parent's person, not to nobody (P18b round 3, fix A). Only the owner
            // is inherited: no claims means no token is minted, so the child's MCP access
            // stays exactly as restricted as before (H6).
            // A conversation the router moved (no claims: nobody asked) stays its person's: the
            // new session belongs to the owner of the one it continues, so their "models Auto
            // may use" (AUTO-L) and their permission answers follow the conversation.
            None if request.user_claims.is_none() => match self.relayed_owner(relay).await {
                Some(owner) => Some(owner),
                None => self.spawning_parent_owner(request).await,
            },
            owner => owner,
        };
        // Check max sessions: both engines' live sessions count.
        {
            if self.active_session_count().await >= self.config.max_sessions {
                // Full: an idle session (finished sub-agent, abandoned chat) must not hold a
                // slot for the half hour the cleanup task waits (issue #876). Free the most
                // idle one, then count again; refuse, typed (429), only if none can go.
                self.reclaim_idle_slot(RECLAIM_IDLE_GRACE).await;
                let active = self.active_session_count().await;
                if active >= self.config.max_sessions {
                    return Err(anyhow::Error::new(
                        super::provider::errors::SessionLimitReached {
                            active,
                            limit: self.config.max_sessions,
                        },
                    ));
                }
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
        *opened_on = Some(provider_choice.provider_id.clone());
        // A conversation routed among its pool opens on the model PO chose among it; the
        // pilot the menu names is only the fallback when no decision applied (R-S4).
        let decided_among_pool = request.routes_among_pool()
            && provider_choice.routed_by == super::provider::resolver::RoutedBy::Auto;
        let model = match request
            .model
            .as_deref()
            .filter(|m| !m.is_empty() && !decided_among_pool)
        {
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

        // The window the prompt and the per-turn enrichment are budgeted for (CTX4),
        // resolved once for this opening.
        let session_window = self
            .session_context_window(&provider_choice.provider_id, &model, None)
            .await;

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
                // A child of a conversation is told so, and an engine other than Claude
                // Code gets its own sub-agent words (P18-suite, point 7).
                let audience = super::composer::PromptAudience::of(
                    request.spawned_by.as_deref(),
                    &provider_choice.provider_id,
                );
                self.build_system_prompt_anchored_for(
                    &audience,
                    if super::neutral_place::is_neutral_path(&request.cwd) {
                        super::neutral_place::ExecutionPlace::Neutral
                    } else {
                        super::neutral_place::ExecutionPlace::Project
                    },
                    &request.cwd,
                    project_slug.as_deref(),
                    routing_message,
                    Some(&model),
                    session_window,
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
        // A spawned session's workspace is inherited (`ParentEnvelope::apply`), but
        // `resolve_add_dirs` above re-reads the workspace's CURRENT membership: if a
        // project was added to it after the parent was itself opened, the resolution
        // just above hands a later-spawned child a directory the parent's own envelope
        // never granted it (P18b, fix 1). Clamp back to what the parent could already
        // see, frozen on the parent's own node at ITS creation.
        let resolved_add_dirs = match request
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)
            .and_then(|ctx| ctx.parent_session_id)
        {
            Some(parent_id) => {
                self.clamp_spawned_add_dirs(resolved_add_dirs, &parent_id)
                    .await
            }
            None => resolved_add_dirs,
        };

        // What the session may do: fixed here, persisted on the node, kept by every resume.
        // Read-only by default only for a session that belongs to NOTHING: no project and no
        // workspace. Where it runs does not decide it (`neutral_place`: "no tool permission
        // is derived from it"), so a `work` project — no source tree, so a neutral cwd — and
        // a workspace that holds no project yet are ordinary sessions and can write.
        let unanchored = super::neutral_place::is_neutral_path(&request.cwd)
            && project_slug.is_none()
            && request.workspace_slug.is_none();
        let access = super::provider::policy::SessionAccess::for_open(request.access, unanchored);
        // The record both engines keep (`session_record`): the title and preview come
        // from the opening message as typed, and the memory conversation is named
        // here (a relayed session keeps the one it continues).
        let title_preview = super::session_record::title_and_preview(&request.message);
        let conversation_id = relay.and_then(|r| r.conversation_id.clone()).or_else(|| {
            self.memory_config.as_ref().map(|cfg| {
                ConversationMemoryManager::new(cfg.clone())
                    .conversation_id()
                    .to_string()
            })
        });
        let session_node = ChatSessionNode {
            file_scope_lifted: request.file_scope_lifted,
            mirror_of: None,
            id: session_id,
            cli_session_id: None,
            project_slug: project_slug.clone(),
            workspace_slug: request.workspace_slug.clone(),
            cwd: request.cwd.clone(),
            title: title_preview.as_ref().map(|(t, _)| t.clone()),
            model: model.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            // The opening message is the first user message (persisted below as
            // `user_message`); every later one bumps the count.
            message_count: 1,
            total_cost_usd: None,
            conversation_id: conversation_id.clone(),
            preview: title_preview.map(|(_, p)| p),
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
            // Written BEFORE the provider opens (M2): a resume routes on it, whether or not
            // the capability snapshot (written after the open, non-fatal) reaches the graph.
            engine: Some(
                if self.engine_is_agent(&provider_choice.provider_id) {
                    ChatSessionNode::ENGINE_AGENT
                } else {
                    ChatSessionNode::ENGINE_LEGACY
                }
                .to_string(),
            ),
            resume_token: None,
            execution_place: if super::neutral_place::is_neutral_path(&request.cwd) {
                super::neutral_place::ExecutionPlace::Neutral
            } else {
                super::neutral_place::ExecutionPlace::Project
            },
            // ENV-WS: decided HERE, by the server, from the request having no project.
            project_inferred: project_was_inferred(
                request.project_slug.as_deref(),
                project_slug.as_deref(),
            ),
            access,
            owner,
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
        // Hybrid routing: every provider but Claude Code is served by the agent engine;
        // Claude Code stays on the historical engine unless `CHAT_PROVIDER_PATH=agent`
        // (`engine_is_agent`). The agent engine ports hooks, queue, auto-continue,
        // retry, compaction and NATS: what a session lacks is in its `system_init`.
        if self.engine_is_agent(&provider_choice.provider_id) {
            let opened = self
                .open_agent_session(AgentOpen {
                    request,
                    session_id,
                    provider_id: &provider_choice.provider_id,
                    model: &model,
                    system_prompt: &system_prompt,
                    add_dirs: &resolved_add_dirs,
                    project_slug: project_slug.as_deref(),
                    owner: session_node.owner.as_deref(),
                    relay,
                    access,
                })
                .await?;
            // Its `created` went out from `open_agent_session`, before its first turn.
            return Ok(opened);
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
                    segments: None,
                    owner: session_node.owner.as_deref(),
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
        let (session_hooks, lifecycle_hooks) = {
            let context_source = match &spawned_ctx {
                Some(ctx) if ctx.task_id.is_some() => {
                    CompactionContextSource::Task(ctx.task_id.unwrap())
                }
                _ => match project_slug.as_deref() {
                    Some(slug) => CompactionContextSource::Session(slug.to_string()),
                    None => CompactionContextSource::None,
                },
            };
            let mut table = self.graph_hook_table(GraphHookInput {
                session_id: session_id.to_string(),
                context_source,
                work_log: work_log.clone(),
                // PreToolUse/PostToolUse are skipped for runner sessions — they inject
                // ~1000 chars of context per tool call (persona, notes, redirect suggestions),
                // which accelerates compaction and wastes tokens. Runner agents already have
                // full task context via the prompt.
                tool_knowledge: request.runner_context.is_none(),
                announce: Some(events_tx.clone()),
            });
            // ISO1: Claude Code's own events (UserPromptSubmit, Stop, SubagentStop,
            // Notification) consult the same hooks as on the agent engine and the native harness.
            let lifecycle = self.serve_lifecycle_hooks(
                &mut table,
                &session_id.to_string(),
                &request.cwd,
                request.permission_mode.clone(),
            );
            (table, lifecycle)
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
        // ISO1: the CLI sends no SessionStart to SDK callbacks: the host asks it itself.
        super::agent_hooks::ask_session_start(&lifecycle_hooks, "startup", &session_id.to_string())
            .await;

        // Create ConversationMemoryManager for message recording
        // It records into the conversation named on the node at creation.
        let memory_manager = match (&self.memory_config, &conversation_id) {
            (Some(mem_config), Some(conversation_id)) => {
                let mm = ConversationMemoryManager::new(mem_config.clone())
                    .with_conversation_id(conversation_id.clone());
                debug!(
                    "Created ConversationMemoryManager for session {} with conversation_id {}",
                    session_id, conversation_id
                );
                Some(Arc::new(Mutex::new(mm)))
            }
            _ => None,
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
                Some(Arc::clone(&lifecycle_hooks)),
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
        // Auto-continue is always on, capped (decision 7b95342a): a runner session always
        // continues; an interactive one unless the operator's kill switch is off.
        let is_runner_session = request.runner_context.is_some();
        let (auto_continue_on, max_auto_continues) =
            super::post_stream::session_auto_continue(is_runner_session, self.config.auto_continue);
        let auto_continue = Arc::new(AtomicBool::new(auto_continue_on));
        let auto_continue_count = Arc::new(AtomicU32::new(0));
        // OOB-trigger cap (T7 of plan 9a1684b2): conservative for runner
        // sessions, generous for interactive (user can always interrupt).
        let oob_trigger_cap: u32 = if is_runner_session {
            OOB_TRIGGER_CAP_RUNNER
        } else {
            OOB_TRIGGER_CAP_INTERACTIVE
        };
        // Resolved before the sessions are locked (it reads the project's record).
        // Boxed: the opening's future is already large.
        let lasting = Box::pin(self.lasting_scope(
            session_node.owner.as_deref(),
            session_node.project_slug.as_deref(),
            &request.cwd,
        ))
        .await;
        let session_grants = self
            .session_grants_for(
                &session_id.to_string(),
                Some(std::path::Path::new(&crate::expand_tilde(&request.cwd))),
                lasting,
                false,
                super::session_grants::Asker::ClaudeCode,
            )
            .await;
        // Its Claude Code sub-agents are mirrored (P18c, `chat::mirror`).
        let mirror_tap = self.mirror_tap(&session_id.to_string());
        let parent = super::cancelled_parents::parent_of(request.spawned_by.as_deref());
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // B36-RACE: a child whose parent a cascade closed while it was opening is
            // refused HERE, under the lock that would make it live (`chat::cancelled_parents`).
            let admitted = self
                .cancelled_parents
                .admit(&session_id.to_string(), parent.as_deref());
            if let Err(refused) = admitted {
                drop(sessions);
                self.refuse_legacy_activation(&session_id.to_string(), client, &refused)
                    .await;
                return Err(anyhow::Error::new(refused));
            }
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
                    context_window: session_window,
                    anchor: self.anchor_session(),
                    mirror_tap: mirror_tap.clone(),
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
                    turn_generation: Arc::new(AtomicU64::new(0)),
                    result_cursor: Default::default(),
                    stream_task: Arc::new(std::sync::Mutex::new(None)),
                    post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
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
                    // CACHE-POOL: a delegation's child (conversation sub-agent, MCP sub-agent,
                    // runner, gate, trigger — any spawned_by) is not the conversation the
                    // project's pending tasks belong to. Reminding it of the PARENT's objectives
                    // is noise it never asked for (measured 17/10/2026: an extra unrequested
                    // turn, ~9.4k cache tokens, to answer a reminder meant for someone else).
                    objective_tracking: tracks_own_objectives(request.spawned_by.is_some()),
                    objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
                    objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
                    work_log: work_log.clone(),
                    oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
                    provider_outage: Arc::new(std::sync::Mutex::new(
                        super::provider::outage::TurnHold::with_policies(
                            super::provider::resolver::CLAUDE_CODE,
                            super::provider::outage::OutagePolicy::from_env(),
                            super::provider::outage::HoldPolicy::from_env(),
                        ),
                    )),
                    session_grants: Arc::new(std::sync::Mutex::new(session_grants)),
                    oob_trigger_cap,
                    oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
                    oob_capped_warned: Arc::new(AtomicBool::new(false)),
                    cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_tools_cap: CANCEL_TOOLS_CAP,
                    cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
                    active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
                    tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
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
        // cancel_task asked on another instance (request/reply).
        self.spawn_nats_cancel_task_listener(
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

        // The lists (and other instances, via NATS) learn the session now: BEFORE
        // anything can start a turn of it — the out-of-band listener below may, and the
        // first turn is spawned further down — so `created` precedes its first CRUD
        // `updated {is_streaming}` (review of frontend #364).
        self.emit_session_created(session_id, request, project_slug.as_deref());

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
            let relayed = relay.event(
                &session_id.to_string(),
                &provider_choice.provider_id,
                &model,
            );
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

        // The title and preview were set on the node at creation (`session_record`).

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
        let cognitive_routing = self.cognitive_routing.clone();

        let session_id_for_turn = session_id_str.clone();
        let turn_task = tokio::spawn(async move {
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
                cognitive_routing,
            )
            .await;
        });
        Self::track_stream_task(
            &self.active_sessions,
            &session_id_for_turn,
            turn_task.abort_handle(),
        )
        .await;

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
    /// returns `false` and the second pass performs the actual insert. The
    /// first pass of a Bash call still reads the CLI's descendants
    /// (`ActiveSession::tool_start_descendants`): that reading, taken before
    /// the CLI can fork the tool, is the reference of the PID claim.
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
        // Task 51f87943: the reference of a Bash background task's PID claim is
        // read when the call STARTS (first pass, `ContentBlockStart`, empty
        // input): the CLI cannot fork the tool before the input is complete. Read
        // on the second pass instead, it already held a process the CLI forked
        // right after printing the call — never a newcomer, never claimed, and
        // `cancel_task` then signalled nothing. Every Bash call is read (its
        // `run_in_background` flag is not known yet) and the reading is taken
        // back on the second pass, whatever the flag says.
        let early_before: Option<Vec<u32>> = if tool_name == "Bash" {
            let (early, cli_pid) = {
                let sessions = active_sessions.read().await;
                let Some(active) = sessions.get(session_id) else {
                    return false;
                };
                (active.tool_start_descendants.clone(), active.child_pid)
            };
            if input.as_object().is_none_or(serde_json::Map::is_empty) {
                if let Some(cli_pid) = cli_pid {
                    let pids = Self::get_descendant_pids(cli_pid);
                    early
                        .lock()
                        .await
                        .entry(tool_use_id.to_string())
                        .or_insert(pids);
                }
                return false;
            }
            let taken = early.lock().await.remove(tool_use_id);
            taken
        } else {
            None
        };

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

        // Capture the descendant snapshot BEFORE the subprocess forks: the one
        // read when the Bash call started when there is one (task 51f87943),
        // else now. This must run lock-free because `pgrep -P` may take
        // 10-50 ms. Plan fc35b25e (T2): the async claim spawned below compares
        // this snapshot to readings taken every PID_CLAIM_POLL (up to
        // PID_CLAIM_WINDOW) to discover the new PID.
        let before_pids: Vec<u32> = match early_before {
            Some(pids) => pids,
            None => child_pid_for_before
                .map(Self::get_descendant_pids)
                .unwrap_or_default(),
        };

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
                    signalled: false,
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
            entry.signalled = false;

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
    /// PID at every reading so a `resume_session` that races with the
    /// claim window does not silently associate the new tool with an
    /// obsolete CLI.
    async fn lookup_child_pid(
        session_id: &str,
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
    ) -> Option<u32> {
        let sessions = active_sessions.read().await;
        sessions.get(session_id).and_then(|s| s.child_pid)
    }

    /// Async PID claim for a freshly-tracked background task. Returns as soon as
    /// the subprocess it looks for is visible, and gives up after
    /// [`PID_CLAIM_WINDOW`]. On success it assigns the newest new PID to the
    /// task entry and broadcasts a fresh `ActiveTasksUpdate` so the frontend can
    /// surface the PID (useful for debugging).
    ///
    /// The process is forked by the tool the CLI is about to run, and its birth
    /// is NOT synchronised with the `ToolUse` we announce: how long the fork
    /// takes depends on the load of the machine. A single reading taken 1 s
    /// later therefore missed it whenever the fork landed after that reading,
    /// and `cancel_task` then had no PID to signal at all (task
    /// `FLAKY-PARITE`: `cancel_task` red once, green on a rerun of the SAME
    /// head, under llvm-cov). So the claim re-reads the CLI's descendants every
    /// [`PID_CLAIM_POLL`] and stops at the first reading whose diff against
    /// `before_pids` is not empty — the first new PID found is the claim.
    ///
    /// Race-safety:
    /// - No locks are held during the sleeps.
    /// - If the entry was removed before the claim lands (rapid cancel) the
    ///   claim no-ops silently.
    /// - If the diff stays empty until the deadline (the subprocess crashed
    ///   before the first reading, or two ToolUses raced and the first claim won
    ///   both candidates), we log a warning and leave `pid: None`. The
    ///   `cancel_task` fallback (T4) handles this case.
    ///
    /// The window is BOUNDED, deliberately: this is a detached task per
    /// background task, so an unbounded poller would spin for the whole life of
    /// the session each time a background process dies before it is ever
    /// observed. Ten seconds is far past every measured fork latency and stays
    /// negligible next to `BACKGROUND_TASK_IDLE_DEATH_SECS`. Widening the single
    /// sleep instead would delay every claim without making it deterministic.
    ///
    /// Plan fc35b25e (T2). Tested in `tests::test_pid_claim_*` and in
    /// `tests::a_pid_born_late_in_the_claim_window_is_still_claimed`.
    async fn async_pid_claim(
        session_id: String,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: broadcast::Sender<ChatEvent>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        tool_use_id: String,
        before_pids: Vec<u32>,
    ) {
        let deadline = tokio::time::Instant::now() + PID_CLAIM_WINDOW;
        let mut reads: u32 = 0;
        let candidates = loop {
            // Let the subprocess fork; the loop comes back until it is visible
            // (or the window closes).
            tokio::time::sleep(PID_CLAIM_POLL).await;
            reads += 1;

            // Re-resolve child_pid on every reading (a `resume_session` between
            // the callsite and now would have replaced the CLI).
            let Some(child_pid) = Self::lookup_child_pid(&session_id, &active_sessions).await
            else {
                debug!(
                    session_id = %session_id,
                    tool_use_id = %tool_use_id,
                    "async_pid_claim: session gone, skipping"
                );
                return;
            };

            let after_pids = Self::get_descendant_pids(child_pid);
            let candidates = Self::pid_discovery_diff(&before_pids, &after_pids);
            if !candidates.is_empty() {
                break candidates;
            }
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    session_id = %session_id,
                    tool_use_id = %tool_use_id,
                    before_count = before_pids.len(),
                    after_count = after_pids.len(),
                    reads = reads,
                    window_ms = PID_CLAIM_WINDOW.as_millis() as u64,
                    "async_pid_claim: no new PID in the whole window (subprocess crashed or claim raced)"
                );
                return;
            }
        };

        // The loop only breaks on a non-empty diff.
        let claimed = candidates[0];

        // Update the task's pid and capture a fresh snapshot. A task stopped
        // before its pid was known (`cancel_task` within the claim's first
        // second: marked `pending_removal_at`, nothing signalled) is signalled
        // here, once: nothing else would, and a `cancel_task` asked again finds
        // it signalled (review of #691, finding 1).
        //
        // Only when the claim is unambiguous (review of #694, point 2): several new
        // processes in the same second (two tools started together) means the
        // first candidate may be the neighbour's. The pid is still recorded, as
        // before, but nothing is sent automatically: the task stays marked
        // stopping and NOT signalled, so an explicit Stop asked again signals it
        // (it is not the no-op), as before #691.
        let unambiguous = candidates.len() == 1;
        if !unambiguous {
            debug!(
                session_id = %session_id,
                tool_use_id = %tool_use_id,
                candidates = candidates.len(),
                "async_pid_claim: several new processes; a stop before the claim is left to an explicit Stop"
            );
        }
        let mut stop_now = false;
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
            if unambiguous && task.pending_removal_at.is_some() && !task.signalled {
                task.signalled = true;
                stop_now = true;
            }
            tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>()
        };

        if stop_now {
            let killed = Self::kill_subtree(claimed);
            info!(
                session_id = %session_id,
                tool_use_id = %tool_use_id,
                root_pid = claimed,
                killed_count = killed.len(),
                "async_pid_claim: the task was stopped before its pid was known: SIGINT'd subtree"
            );
        }

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
                    signalled: false,
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
        let (tasks_arc, cli_reports_live_tasks, cli_background_tasks_arc, child_pid) = {
            let sessions = active_sessions.read().await;
            match sessions.get(session_id) {
                Some(active) => (
                    active.active_background_tasks.clone(),
                    active.cli_background_tasks.load(Ordering::Relaxed) > 0,
                    active.cli_background_tasks.clone(),
                    active.child_pid,
                ),
                None => return,
            }
        };
        // A dead CLI (zombie or reaped) cannot have live tasks; its last count is stale.
        // Clear it here so the idle-death logic below can do its job without being
        // falsely kept alive by a count that will never reach 0 (ZOMBIE-CLI).
        let cli_reports_live_tasks =
            if cli_reports_live_tasks && process_alive(child_pid) == Some(false) {
                cli_background_tasks_arc.store(0, Ordering::Relaxed);
                false
            } else {
                cli_reports_live_tasks
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
                    // Found dead: no signal may go to its pid any more (it may
                    // name another process by now). A `cancel_task` before the
                    // purge is the no-op (review of #694, point 1).
                    info.signalled = true;
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

    /// Marks the session's provider instance out of credit in the cognitive
    /// router when a legacy-engine turn ended on an out-of-credit refusal
    /// reported as TEXT (HTTP 402). The legacy CLI reports a 402 as raw text,
    /// so the router's typed `ProviderError` check never sees it; the manager
    /// calls this on the error text instead. The provider id is read from the
    /// session node (frozen at open; `claude-code` when the node predates the
    /// field), so nothing is plumbed through the turn.
    async fn mark_out_of_credit_if_needed(
        routing: Option<&super::provider::cognitive::decider::CognitiveRouting>,
        graph: &Arc<dyn GraphStore>,
        session_uuid: Option<Uuid>,
        error_text: &str,
    ) {
        let Some(routing) = routing else {
            return;
        };
        if !is_out_of_credit_message(error_text) {
            return;
        }
        let provider_id = match session_uuid {
            Some(uuid) => match graph.get_chat_session(uuid).await {
                Ok(Some(node)) => node
                    .provider_id
                    .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string()),
                _ => super::provider::resolver::CLAUDE_CODE.to_string(),
            },
            None => super::provider::resolver::CLAUDE_CODE.to_string(),
        };
        warn!(
            provider = %provider_id,
            "legacy turn ended out of credit; excluding the provider instance for a cooldown"
        );
        routing.mark_out_of_credit(&provider_id);
    }

    /// The conversation's persisted segmentation, or `None` when there is nothing to
    /// resume. Never a failure, and never a delay beyond `READ_TIMEOUT`: an absent, empty,
    /// slow or unreadable store starts a fresh segmentation — a missing episodic memory
    /// costs a boundary, never a turn.
    async fn read_segments_snapshot(
        store: &Arc<dyn super::provider::cognitive::segment_store::SegmentStore>,
        conversation: &str,
    ) -> Option<super::provider::cognitive::segment::SegmentsSnapshot> {
        /// How long the stored segmentation is waited for when a router is built.
        const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1_000);
        let read = async {
            let segments = store.segments(conversation).await.ok()?;
            let boundaries = store.boundaries(conversation).await.ok()?;
            Some(super::provider::cognitive::segment::SegmentsSnapshot {
                segments,
                boundaries,
            })
        };
        match tokio::time::timeout(READ_TIMEOUT, read).await {
            Ok(Some(snapshot)) if !snapshot.is_empty() => Some(snapshot),
            Ok(_) => None,
            Err(_) => {
                warn!(
                    conversation,
                    "conversation segments unreadable in time: fresh segmentation"
                );
                None
            }
        }
    }

    /// Feeds one **final** failed call of the legacy (Claude Code) loop to its
    /// session's `TurnHold`, in the loop's own text taxonomy
    /// (`outage::outcome_of_legacy_failure`) — the same detector the agent engine
    /// feeds from a typed `ProviderError`. Emits the notice when this outcome
    /// flipped the instance (a fresh outage announces `down`).
    ///
    /// Returns `true` when the text was a *health* failure — the outcome that
    /// counts against the instance — which is what tells the caller a spent retry
    /// budget may be **held** for a down instance rather than closing the turn.
    async fn record_legacy_outage(
        outage: &Arc<std::sync::Mutex<super::provider::outage::TurnHold>>,
        text: &str,
        sink: &OutageSink<'_>,
    ) -> bool {
        let Some((outcome, cause)) = super::provider::outage::outcome_of_legacy_failure(text)
        else {
            return false;
        };
        let notice = outage.lock().unwrap_or_else(|e| e.into_inner()).record(
            outcome,
            Instant::now(),
            Some(cause),
        );
        if let Some(notice) = notice {
            sink.emit(notice).await;
        }
        true
    }

    /// The attempt that answered: the first success after an outage ends the hold
    /// (`TurnHold::record` clears the held state) and announces `up`.
    async fn record_legacy_outage_up(
        outage: &Arc<std::sync::Mutex<super::provider::outage::TurnHold>>,
        sink: &OutageSink<'_>,
    ) {
        let notice = outage.lock().unwrap_or_else(|e| e.into_inner()).record(
            super::provider::circuit_breaker::Outcome::Success,
            Instant::now(),
            None,
        );
        if let Some(notice) = notice {
            sink.emit(notice).await;
        }
    }

    /// Whether a turn whose ordinary retry budget is spent may go on
    /// (`TurnHold::hold_on`): `Some(wait)` while the instance is down and the hold
    /// budget still allows — wait `wait`, then probe the instance again. `None`
    /// when the failure must stand (the instance is not down, or the hold budget
    /// is spent). No second state machine: this is the agent engine's own hold.
    fn hold_legacy_turn(
        outage: &Arc<std::sync::Mutex<super::provider::outage::TurnHold>>,
    ) -> Option<Duration> {
        outage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hold_on(Instant::now())
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
        cognitive_routing: Option<super::provider::cognitive::decider::CognitiveRouting>,
    ) {
        // The tap that mirrors this session's Claude Code sub-agents (P18c, `chat::mirror`):
        // it sees each event AFTER it was broadcast, and never holds the turn.
        let mirror_tap = active_sessions
            .read()
            .await
            .get(&session_id)
            .and_then(|s| s.mirror_tap.clone());
        // Helper closure: emit a ChatEvent to local broadcast + NATS (if configured)
        let emit_chat = |event: ChatEvent,
                         tx: &broadcast::Sender<ChatEvent>,
                         nats: &Option<Arc<crate::events::NatsEmitter>>,
                         sid: &str| {
            let _ = tx.send(event.clone());
            if let Some(tap) = &mirror_tap {
                tap.observe(&event);
            }
            if let Some(ref nats) = nats {
                nats.publish_chat_event(sid, event);
            }
        };

        // Create a NEW CancellationToken and store it in ActiveSession BEFORE
        // setting is_streaming=true. This ensures the token always matches
        // what interrupt() will cancel (fixes Gaps 1, 2, 4, 7, 10).
        let (interrupt_token, result_cursor) = {
            let mut sessions = active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                let token = CancellationToken::new();
                session.interrupt_token = token.clone();
                session.interrupt_flag.store(false, Ordering::SeqCst);
                // A new turn: a watchdog armed for the previous one stands down.
                session.turn_generation.fetch_add(1, Ordering::SeqCst);
                (token, session.result_cursor.clone())
            } else {
                warn!(
                    "Session {} no longer in active_sessions at stream start",
                    session_id
                );
                release_unstarted_turn(&is_streaming, &events_tx, &nats, &session_id);
                return;
            }
        };

        // When the turn started: the duration of a stopped turn the CLI gave none
        // (`interrupted_turn_end`).
        let turn_started = std::time::Instant::now();
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
            let (protocol, anchor, window) = {
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
                let window = sessions.get(&session_id).and_then(|s| s.context_window);
                (protocol, anchor, window)
            };
            // `/compact`: the CLI reads the command only at the head of the text (ISO3).
            if super::manual_compaction::compact_command(&prompt).is_some() {
                prompt
            } else {
                match enrichment_for_turn(
                    &graph,
                    &enrichment_pipeline,
                    &session_id,
                    &prompt,
                    protocol,
                    turn.excluded_note_ids.clone(),
                    &anchor,
                    window,
                )
                .await
                {
                    Some(enrichment_md) => prepend_enrichment(&enrichment_md, &prompt),
                    None => prompt,
                }
            }
        };

        // After the (enriched) visible text: the pointers, then the documents.
        let prompt = format!("{prompt}{}", turn.model_tail);

        // Events are persisted in Neo4j — the WebSocket replay handles late-joining clients.

        // Variables that persist across retry iterations (needed after the retry loop)
        let session_uuid = Uuid::parse_str(&session_id).ok();
        let mut assistant_text_parts: Vec<String> = Vec::new();
        let mut events_to_persist: Vec<ChatEventRecord> = Vec::new();
        // When each tool call of the turn really ran (`tool_clock`): a `tool_timing`
        // is stored and sent right after the result or the cancellation it times.
        let tool_clock = super::tool_clock::ToolClock::for_session(&session_id);

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
        let (
            pending_perm_inputs,
            stdin_tx_for_auto_allow,
            work_log,
            session_grants,
            provider_outage,
        ) = {
            let guard = active_sessions.read().await;
            match guard.get(&session_id) {
                Some(s) => (
                    s.pending_permission_inputs.clone(),
                    s.stdin_tx.clone(),
                    s.work_log.clone(),
                    s.session_grants.clone(),
                    s.provider_outage.clone(),
                ),
                None => (
                    Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
                    None,
                    Arc::new(Mutex::new(SessionWorkLog::default())),
                    Arc::default(),
                    Arc::new(std::sync::Mutex::new(
                        super::provider::outage::TurnHold::new(
                            super::provider::resolver::CLAUDE_CODE,
                        ),
                    )),
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

        // What a transition of the instance's state needs to be persisted and
        // broadcast: built once, used by every failure site below.
        let outage_sink = OutageSink {
            graph: &graph,
            session_uuid,
            next_seq: &next_seq,
            events_tx: &events_tx,
            nats: &nats,
            session_id: &session_id,
        };

        'retry_loop: loop {
            // A refused turn is never sent (and so never retried).
            let Some(input) = cli_input.clone() else {
                break 'retry_loop;
            };
            // Per-attempt state — reset on each retry iteration
            let mut should_retry = false;
            let mut last_retry_error = String::new();
            // Outage state of this attempt: did it feed the detector a *health*
            // failure, and did it get a stream at all (`OutageSink`/`TurnHold`)?
            let mut outage_failed = false;
            let mut attempt_answered = false;
            // Set when the retry budget is spent and the instance is down: how long
            // to wait before probing it again (`TurnHold::hold_on`).
            let mut held_for: Option<Duration> = None;

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
                        // The outage detector sees this failed call, in the loop's
                        // own text taxonomy: a 5xx, an overload, a rate limit, a
                        // timeout or an unreachable endpoint counts against the
                        // instance (`OutageSink`).
                        outage_failed =
                            Self::record_legacy_outage(&provider_outage, &err_str, &outage_sink)
                                .await;
                        // A health failure whose retry budget is spent is **held**
                        // while the instance is down (`TurnHold::hold_on`), probed at
                        // the hold's cadence: a turn that never started showed
                        // nothing, so nothing is replayed.
                        if outage_failed
                            && !interrupt_flag.load(Ordering::SeqCst)
                            && retry_attempt >= retry_config.max_attempts
                        {
                            held_for = Self::hold_legacy_turn(&provider_outage);
                        }

                        if kind.is_retryable()
                            && retry_attempt < retry_config.max_attempts
                            && !interrupt_flag.load(Ordering::SeqCst)
                        {
                            // Retryable — set flag, will handle after client lock release
                            should_retry = true;
                            last_retry_error = err_str;
                        } else if let Some(probe) = held_for {
                            // The instance is down and the hold still allows: the
                            // turn goes on instead of being closed on an error the
                            // user would read as theirs.
                            warn!(
                                session_id = %session_id,
                                probe_ms = probe.as_millis() as u64,
                                "Provider instance is down: holding the turn instead of closing it"
                            );
                            should_retry = true;
                            last_retry_error = err_str;
                        } else {
                            // Not retryable or max retries exceeded — propagate error
                            error!("Error starting stream for session {}: {}", session_id, e);

                            // A turn that never started leaves the session half-dead:
                            // the CLI could not accept the write, whether the error
                            // names the channel or not. Forget it so the next user
                            // message routes to resume_session (a fresh CLI) instead
                            // of re-hitting the same broken stdin. Only the entry of
                            // THIS client: a resume that replaced it meanwhile keeps
                            // its session and its NATS listeners (review of #685,
                            // finding 2).
                            warn!(
                                "CLI could not start the stream for session {} ({}), removing from \
                                 active_sessions so next message triggers resume_session",
                                session_id, err_str
                            );
                            if Self::forget_if_client(&active_sessions, &session_id, &client).await
                            {
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
                                    request_id: None,
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
                    attempt_answered = true;
                    let mut stream = std::pin::pin!(s);

                    // Track the current parent_tool_use_id from stream events.
                    // When a sub-agent is active, its stream Messages carry parent_tool_use_id.
                    // We capture this so permission requests (which arrive via a separate control
                    // channel without parent info) can be attributed to the correct agent.
                    let mut current_parent_tool_use_id: Option<String> = None;

                    // Helper closure: process an SDK control message (permission or AskUserQuestion).
                    // Returns Some(ChatEvent) if a can_use_tool was parsed, None otherwise.
                    // Persisted here, NOT broadcast: the caller stores a permission request's
                    // input first, then announces it (`announce_control_event`).
                    // The caller is responsible for adding the event to streaming_events (async)
                    // and for auto-allowing AskUserQuestion events via stdin_tx.
                    let handle_control_msg = |control_msg: serde_json::Value,
                                              events_to_persist: &mut Vec<ChatEventRecord>,
                                              next_seq: &std::sync::atomic::AtomicI64,
                                              current_parent: Option<String>|
                     -> Option<ChatEvent> {
                        let mut event = parse_permission_control_msg(&control_msg, current_parent)?;
                        mark_always_eligibility(&mut event, &session_grants);
                        tool_clock.observe(&event, chrono::Utc::now());

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

                        Some(event)
                    };

                    // The request leaves the backend: broadcast to the WebSocket clients and
                    // a light signal on /ws/events (ids only, never the command/question).
                    // Called AFTER a permission request is in `pending_perm_inputs`: a strict
                    // answer (`route_permission_response` with `require_pending`) sent the
                    // moment a client sees the request finds it pending, never `NotPending`
                    // (review of #712).
                    let announce_control_event = |event: &ChatEvent| {
                        emit_chat(event.clone(), &events_tx, &nats, &session_id);
                        notify_attention_for_chat_event(&event_emitter, &session_id, event);
                    };

                    // A decision the backend made itself (a request a session grant covers): persisted
                    // and broadcast like the request it answers, and the attention it raised cleared
                    // (`attention_changed`, as for a decision of the user).
                    let record_decision =
                        |event: ChatEvent,
                         events_to_persist: &mut Vec<ChatEventRecord>,
                         next_seq: &std::sync::atomic::AtomicI64| {
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
                            emit_chat(event, &events_tx, &nats, &session_id);
                            notify_attention(
                                &event_emitter,
                                AttentionSubject::Session(session_id.to_string()),
                                AttentionReason::PermissionDecision,
                            );
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
                    let mut interrupted_in_stream = false;
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
                                    interrupted_in_stream = true;
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
                                                let mut granted_decision: Option<ChatEvent> = None;
                                                // A regular permission is pending BEFORE anyone sees it.
                                                if let ChatEvent::PermissionRequest { ref id, ref input, ref tool, .. } = evt {
                                                    if !id.is_empty() {
                                                        store_pending_perm_input(&pending_perm_inputs, id, input, tool).await;
                                                    }
                                                }
                                                announce_control_event(&evt);
                                                // AskUserQuestion: close the control request with a denial that ends
                                                // the turn; the user's answer comes as the next turn (§15.6). An
                                                // `allow` without answers would make the CLI report "the user did not
                                                // answer" and the model would go on without it.
                                                // Do NOT store in pending_perm_inputs (it's not a permission).
                                                if let ChatEvent::AskUserQuestion { ref id, .. } = evt {
                                                    if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                        let control_response = question_closing_response(id);
                                                        if let Ok(json) = serde_json::to_string(&control_response) {
                                                            let _ = tx.send(json).await;
                                                            info!(request_id = %id, "Closed AskUserQuestion control request (answer comes as the next turn)");
                                                        }
                                                    }
                                                } else if let ChatEvent::PermissionRequest { ref id, ref input, ref tool, .. } = evt {
                                                    // Regular permission (its input stored above): a session grant may answer it
                                                    if !id.is_empty() {
                                                        if let Some(decision) = answer_granted_request(&session_grants, &pending_perm_inputs, &tool_clock, stdin_tx_for_auto_allow.as_ref(), id, tool, input).await {
                                                            record_decision(decision.clone(), &mut events_to_persist, &next_seq);
                                                            granted_decision = Some(decision);
                                                        }
                                                    }
                                                }
                                                streaming_events.lock().await.push(evt);
                                                if let Some(decision) = granted_decision {
                                                    streaming_events.lock().await.push(decision);
                                                }
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
                                                let mut granted_decision: Option<ChatEvent> = None;
                                                    // Pending before anyone sees it (same as above).
                                                    if let ChatEvent::PermissionRequest { ref id, ref input, ref tool, .. } = evt {
                                                        if !id.is_empty() {
                                                            store_pending_perm_input(&pending_perm_inputs, id, input, tool).await;
                                                        }
                                                    }
                                                    announce_control_event(&evt);
                                                    // AskUserQuestion: close it, as above (§15.6)
                                                    if let ChatEvent::AskUserQuestion { ref id, .. } = evt {
                                                        if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                            let control_response = question_closing_response(id);
                                                            if let Ok(json) = serde_json::to_string(&control_response) {
                                                                // try_send here because we're in a sync-ish drain loop
                                                                let _ = tx.try_send(json);
                                                                info!(request_id = %id, "Closed AskUserQuestion control request (drain)");
                                                            }
                                                        }
                                                    } else if let ChatEvent::PermissionRequest { ref id, ref input, ref tool, .. } = evt {
                                                        if !id.is_empty() {
                                                        if let Some(decision) = answer_granted_request(&session_grants, &pending_perm_inputs, &tool_clock, stdin_tx_for_auto_allow.as_ref(), id, tool, input).await {
                                                            record_decision(decision.clone(), &mut events_to_persist, &next_seq);
                                                            granted_decision = Some(decision);
                                                        }
                                                        }
                                                    }
                                                    streaming_events.lock().await.push(evt);
                                                    if let Some(decision) = granted_decision {
                                                        streaming_events.lock().await.push(decision);
                                                    }
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
                                    interrupted_in_stream = true;
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
                                    result_cursor.turn_read_result();
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
                                    // One final call of the legacy loop fed to the
                                    // outage detector, in its own text taxonomy.
                                    outage_failed = Self::record_legacy_outage(
                                        &provider_outage,
                                        text,
                                        &outage_sink,
                                    )
                                    .await;
                                    if kind.is_retryable()
                                        && text_empty
                                        && retry_attempt < retry_config.max_attempts
                                        && !interrupt_flag.load(Ordering::SeqCst)
                                    {
                                        should_retry = true;
                                        last_retry_error = text.clone();
                                        break; // break inner stream loop
                                    }
                                    // A turn that showed nothing but its error is held
                                    // for a down instance rather than closed: `text_empty`
                                    // is the very guard the ordinary retry uses, so an
                                    // executed tool is never replayed by the hold either.
                                    if outage_failed
                                        && text_empty
                                        && retry_attempt >= retry_config.max_attempts
                                        && !interrupt_flag.load(Ordering::SeqCst)
                                    {
                                        held_for = Self::hold_legacy_turn(&provider_outage);
                                        if let Some(probe) = held_for {
                                            warn!(
                                                session_id = %session_id,
                                                probe_ms = probe.as_millis() as u64,
                                                "Provider instance is down: holding the turn instead of closing it"
                                            );
                                            should_retry = true;
                                            last_retry_error = text.clone();
                                            break; // break inner stream loop
                                        }
                                    }
                                    // A non-retryable refusal: a 402 (out of credit)
                                    // must cool the instance down via the router, not
                                    // repeat it on the next message.
                                    Self::mark_out_of_credit_if_needed(
                                        cognitive_routing.as_ref(),
                                        &graph,
                                        session_uuid,
                                        text.as_str(),
                                    )
                                    .await;
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
                                    let event = offer_lasting_scope(event, &session_grants);
                                    if let Some(call) = super::session_grants::observed_call(
                                        super::session_grants::Asker::ClaudeCode,
                                        &event,
                                    ) {
                                        session_grants
                                            .lock()
                                            .unwrap_or_else(|p| p.into_inner())
                                            .observe_tool_use(&call);
                                    }
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

                                    let timing = tool_clock.observe(&event, chrono::Utc::now());

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
                                    // The timing of the call this event ended: stored in the
                                    // turn's batch right after it, sent right after it.
                                    if let Some(timing) = timing {
                                        if let Some(uuid) = session_uuid {
                                            events_to_persist.push(ChatEventRecord {
                                                id: Uuid::new_v4(),
                                                session_id: uuid,
                                                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                                                event_type: timing.event_type().to_string(),
                                                data: serde_json::to_string(&timing)
                                                    .unwrap_or_default(),
                                                created_at: chrono::Utc::now(),
                                            });
                                        }
                                        streaming_events.lock().await.push(timing.clone());
                                        emit_chat(timing, &events_tx, &nats, &session_id);
                                    }
                                }
                            }
                            Err(e) => {
                                let err_str = format!("{}", e);
                                let kind = classify_api_error(&err_str);
                                let text_empty = streaming_text.lock().await.is_empty();
                                // The outage detector sees this failed call, in the
                                // loop's own text taxonomy (`OutageSink`).
                                outage_failed = Self::record_legacy_outage(
                                    &provider_outage,
                                    &err_str,
                                    &outage_sink,
                                )
                                .await;
                                // A health failure whose retry budget is spent is held
                                // while the instance is down: `text_empty` is the same
                                // guard the ordinary retry uses — nothing is replayed.
                                if outage_failed
                                    && text_empty
                                    && retry_attempt >= retry_config.max_attempts
                                    && !interrupt_flag.load(Ordering::SeqCst)
                                {
                                    held_for = Self::hold_legacy_turn(&provider_outage);
                                }

                                if kind.is_retryable()
                                    && text_empty
                                    && retry_attempt < retry_config.max_attempts
                                    && !interrupt_flag.load(Ordering::SeqCst)
                                {
                                    // Retryable stream error with 0 tokens — will retry
                                    should_retry = true;
                                    last_retry_error = err_str;
                                } else if let Some(probe) = held_for {
                                    warn!(
                                        session_id = %session_id,
                                        probe_ms = probe.as_millis() as u64,
                                        "Provider instance is down: holding the turn instead of closing it"
                                    );
                                    should_retry = true;
                                    last_retry_error = err_str;
                                } else {
                                    // Not retryable — propagate error
                                    error!("Stream error for session {}: {}", session_id, e);
                                    // A 402 (out of credit) reported as text must cool
                                    // the instance down via the router.
                                    Self::mark_out_of_credit_if_needed(
                                        cognitive_routing.as_ref(),
                                        &graph,
                                        session_uuid,
                                        &err_str,
                                    )
                                    .await;
                                    let cli_gone = err_str.contains("ended before a Result");
                                    if cli_gone {
                                        warn!(
                                            "CLI left before answering for session {}, removing from \
                                             active_sessions so the next message resumes it",
                                            session_id
                                        );
                                        // Only the entry of THIS client (review of
                                        // #685, finding 2).
                                        if Self::forget_if_client(
                                            &active_sessions,
                                            &session_id,
                                            &client,
                                        )
                                        .await
                                        {
                                            notify_attention(
                                                &event_emitter,
                                                AttentionSubject::Session(session_id.to_string()),
                                                AttentionReason::SessionInactive,
                                            );
                                        }
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
                                            request_id: None,
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

                    // An interrupted turn still gets its `result` from the CLI, AFTER the
                    // interrupt. Read it here, before the turn ends: left unread, it reaches
                    // the NEXT turn (a fresh subscription to the CLI's output, started by
                    // the drain of the queued message), which ends on it at once — its real
                    // answer then has no turn to read it and every later turn is one
                    // `result` behind. Bounded: a CLI that never answers does not hold the
                    // session.
                    if interrupted_in_stream {
                        let drained = tokio::time::timeout(INTERRUPTED_TURN_RESULT_GRACE, async {
                            while let Some(item) = stream.next().await {
                                match item {
                                    Ok(msg @ Message::Result { .. }) => {
                                        result_cursor.turn_read_result();
                                        return Some(msg);
                                    }
                                    Ok(_) => continue,
                                    Err(_) => return None,
                                }
                            }
                            None
                        })
                        .await
                        .ok()
                        .flatten();
                        debug!(
                            session_id = %session_id,
                            result_read = drained.is_some(),
                            "interrupted turn: waited for the CLI's result"
                        );
                        // The stopped turn ends on the thread: stored without an end, it
                        // would look cut by a restart at the next boot, and the restart
                        // recovery would close it with a false "the server restarted"
                        // (review of #711, finding 4). Stored only (the agent engine ends an
                        // interrupted turn with such a `result` too); the live clients
                        // already see the turn stop (`streaming_status`).
                        if let Some(uuid) = session_uuid {
                            let ended = interrupted_turn_end(drained, turn_started.elapsed());
                            events_to_persist.push(ChatEventRecord {
                                id: Uuid::new_v4(),
                                session_id: uuid,
                                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                                event_type: ended.event_type().to_string(),
                                data: serde_json::to_string(&ended).unwrap_or_default(),
                                created_at: chrono::Utc::now(),
                            });
                        }
                    }
                } // end if let Some(s) = stream_ok
            } // client lock released here

            // The attempt answered, with no health failure of its own: the instance
            // is up as far as this session knows (a refusal it answered is not a
            // health failure — `outage::outcome_of_legacy_failure`). The first
            // success after an outage ends the hold and announces `up`.
            if attempt_answered && !outage_failed {
                Self::record_legacy_outage_up(&provider_outage, &outage_sink).await;
            }

            // ===== RETRY DECISION =====
            // If should_retry is set (from stream item error or Result is_error),
            // emit Retrying event, sleep with backoff, and continue the retry loop.
            if should_retry {
                retry_attempt += 1;
                // A held turn probes the instance at the hold's cadence, not on the
                // retry curve the ordinary budget follows.
                let delay = match held_for {
                    Some(probe) => probe.as_millis() as u64,
                    None => retry_config.delay_for_attempt(retry_attempt),
                };
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
                    // The stopped turn ends on the thread, as a Stop during the stream
                    // (`interrupted_turn_end`): ended by `retrying` alone, it would look cut
                    // at the next boot (follow-up review of #711, finding 3).
                    if let Some(uuid) = session_uuid {
                        let ended = interrupted_turn_end(None, turn_started.elapsed());
                        events_to_persist.push(ChatEventRecord {
                            id: Uuid::new_v4(),
                            session_id: uuid,
                            seq: next_seq.fetch_add(1, Ordering::SeqCst),
                            event_type: ended.event_type().to_string(),
                            data: serde_json::to_string(&ended).unwrap_or_default(),
                            created_at: chrono::Utc::now(),
                        });
                    }
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
        // All post-stream logic delegated to PostStreamHandler (see post_stream.rs).
        //
        // Nothing below may hold the turn for good. Before this, one store
        // call that never answered (Neo4j, Meilisearch: no client timeout)
        // kept `is_streaming` true past the reach of any Stop — the loop
        // above, the only reader of `interrupt_token`, was over — and the
        // queued messages were never sent (note 1a31aadb, 2026-10-05; session
        // b4c28b9f, 2026-10-09). So:
        // - the turn's events are written in a task of their own, started
        //   first: no step before them, and no abandoned turn, can lose them;
        // - every other step runs under the session's step budget: past it
        //   the step is dropped, logged and reported (`StepBudget`);
        // - the interrupt watchdog (`abandon_turn_if_stuck`) ends a stopped
        //   turn that still does not get there.
        let persist_write = super::post_stream::spawn_persist_events(
            &graph,
            session_uuid,
            &session_id,
            events_to_persist,
        );
        let steps = super::post_stream::StepBudget {
            session_id: session_id.clone(),
            budget: super::post_stream::step_budget_of(&active_sessions, &session_id).await,
            events_tx: events_tx.clone(),
            nats: nats.clone(),
        };
        let post_ctx = steps
            .run(
                "context",
                super::post_stream::PostStreamContext::build(&graph, session_uuid),
            )
            .await
            .unwrap_or_default();
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
        steps
            .run(
                "post_compaction",
                post_handler.handle_post_compaction(needs_post_compaction_injection),
            )
            .await;

        // 2. Lock-free interrupt + ToolCancelled
        steps
            .run(
                "interrupt_cleanup",
                post_handler
                    .handle_interrupt_cleanup(&stdin_tx_for_auto_allow, &pending_tool_calls),
            )
            .await;

        // 3. Auto-continue (its own pause comes on top of the budget)
        let auto_continue_allowed = steps
            .run_for(
                "auto_continue",
                steps.budget
                    + std::time::Duration::from_millis(super::post_stream::AUTO_CONTINUE_DELAY_MS),
                post_handler.handle_auto_continue(hit_error_max_turns),
            )
            .await
            .unwrap_or(false);

        // 4. Objective tracking — uses had_productive_tool_use (not had_tool_use)
        // so that conclusive-only turns (git commit/push) still trigger reminders.
        // Also passes had_conclusive_tool_use so that mixed turns (Edit + git commit)
        // still check for pending objectives.
        // A turn refused before it was sent (its images) did not stop the agent:
        // no objective reminder starts a turn without the user's picture.
        if cli_input.is_some() {
            steps
                .run(
                    "objective_tracking",
                    post_handler.handle_objective_tracking(
                        had_productive_tool_use,
                        had_conclusive_tool_use,
                        auto_continue_allowed,
                        hit_error_max_turns,
                    ),
                )
                .await;
        }

        // The turn is over: a sub-agent mirror still open is closed (P18c).
        if let Some(tap) = &mirror_tap {
            tap.turn_ended();
        }
        // 5. Streaming status update (a lock, nothing remote: never bounded)
        let has_pending = post_handler.finalize_streaming_status().await;

        // 6. The turn's events, written since the loop ended: wait for them up
        // to the budget (the history stays in order when the store is healthy);
        // past it the write goes on alone and the clients are told.
        if let Some(write) = persist_write {
            steps.wait_for_write("persist_events", write).await;
        }

        // 7. Memory / feedback / RFC (its store write runs in its own task)
        steps
            .run(
                "feedback",
                post_handler.handle_feedback(
                    &assistant_text_parts,
                    &memory_manager,
                    &context_injector,
                ),
            )
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
            cognitive_routing,
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
        #[cfg(test)]
        remote_send_hook::wait(session_id).await;
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
            let provider_id = handle.provider_id.clone();
            match handle.send_message(message).await {
                Ok(()) => return Ok(()),
                // A terminal provider failure (out of credit, a non-retryable send
                // error) must not end the chat: fall back to the next eligible provider
                // and resume there. Without one, the error is surfaced unchanged (ERR-1).
                Err(error)
                    if error
                        .chain()
                        .find_map(|cause| {
                            cause.downcast_ref::<nexus_claude::agent::ProviderError>()
                        })
                        .is_some_and(|err| {
                            super::provider::cognitive::decider::is_out_of_credit(err)
                                || !err.retryable()
                        }) =>
                {
                    if self
                        .fallback_provider_after_failure(session_id, &provider_id, message)
                        .await
                    {
                        return Ok(());
                    }
                    return Err(error);
                }
                Err(error) => return Err(error),
            }
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
        let cognitive_routing = self.cognitive_routing.clone();

        let session_id_for_turn = session_id_str.clone();
        let turn_task = tokio::spawn(async move {
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
                cognitive_routing,
            )
            .await;
        });
        Self::track_stream_task(
            &self.active_sessions,
            &session_id_for_turn,
            turn_task.abort_handle(),
        )
        .await;

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

    /// Wakes a session for news from ELSEWHERE that it has to answer — the end of a
    /// delegation it launched — instead of leaving it idle until a human types again.
    ///
    /// On the agent engine the news starts (or waits behind) a turn of the
    /// `BackgroundOutput` kind: the transcript says why the agent went on by itself, and
    /// it is not counted as a user message. It is the same path the out-of-turn pump
    /// takes for what the provider reports (`chat::oob_listener::wake_for_background`).
    ///
    /// On the historical engine the news follows [`Self::inject_hint`]: queued behind a
    /// running turn, a turn of its own when the session is idle.
    pub async fn wake_with_notification(&self, session_id: &str, content: &str) -> Result<()> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            handle.wake_with_background(content).await?;
            return Ok(());
        }
        self.inject_hint(session_id, content).await
    }

    /// Relay a team message from `from_session_id` into `to_session_id`.
    ///
    /// # Behaviour
    /// - **If the target is streaming**: the message is queued (priority 0, same
    ///   as user messages); drain delivers it as its own turn after the current
    ///   one ends. No interrupt.
    /// - **If the target is idle**: the team message is persisted + broadcast
    ///   immediately, and a new `stream_response` turn is spawned with the
    ///   `[team {from_name}] {text}` prefix so the model can distinguish it from
    ///   human input.
    ///
    /// The `from_session_id` and `from_name` are asserted by the backend — they
    /// cannot be forged by model text. `text` is the raw message from the peer;
    /// it is stored as-is in `ChatEvent::TeamMessage`; only the prompt passed to
    /// the model receives the `[team …]` prefix.
    ///
    /// Rate limiting: at most `TEAM_MSG_RATE_MAX` messages per minute from any
    /// single `(from_session_id, to_session_id)` pair.
    pub async fn inject_team_message(
        &self,
        from_session_id: Uuid,
        from_name: &str,
        to_session_id: &str,
        text: &str,
    ) -> Result<()> {
        // Rate limiting — guard against runaway loops between two agents.
        const TEAM_MSG_RATE_MAX: usize = 10; // messages per minute
        const TEAM_MSG_RATE_WINDOW: std::time::Duration = std::time::Duration::from_secs(60);
        {
            let mut rate = self.team_msg_rate.lock().await;
            let key = (from_session_id, to_session_id.to_string());
            let now = std::time::Instant::now();
            let window = rate.entry(key).or_default();
            window.retain(|t| now.duration_since(*t) < TEAM_MSG_RATE_WINDOW);
            if window.len() >= TEAM_MSG_RATE_MAX {
                warn!(
                    from_session_id = %from_session_id,
                    to_session_id = %to_session_id,
                    "inject_team_message: rate limit exceeded ({}/min), dropping",
                    TEAM_MSG_RATE_MAX
                );
                return Ok(());
            }
            window.push(now);
        }

        let session_state = {
            let sessions = self.active_sessions.read().await;
            sessions.get(to_session_id).map(|s| {
                (
                    s.client.clone(),
                    s.events_tx.clone(),
                    s.interrupt_flag.clone(),
                    s.memory_manager.clone(),
                    s.next_seq.clone(),
                    s.pending_messages.clone(),
                    s.is_streaming.clone(),
                    s.streaming_text.clone(),
                    s.streaming_events.clone(),
                    s.sdk_control_rx.clone(),
                    s.auto_continue.clone(),
                )
            })
        };

        let Some((
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
        )) = session_state
        else {
            // Not live on the Claude Code engine. A session released because its segment
            // was Done is resumed by the message, as a person's message would resume it;
            // anything else (closed by a person, idle cleanup, never existed, held by the
            // agent engine) stays "not found".
            let released = self
                .released_done
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(to_session_id);
            if !released || self.agent_runtime.owns(to_session_id).await {
                return Err(anyhow!(
                    "inject_team_message: target session {} not found",
                    to_session_id
                ));
            }
            let team = TeamInbound {
                from_session_id,
                from_name: from_name.to_string(),
                text: text.to_string(),
            };
            let from = self.resume_point(to_session_id).await;
            let mut opened_on = None;
            return self
                .resume_session_on(
                    to_session_id,
                    &format!("[team {}] {}", from_name, text),
                    None,
                    from,
                    &mut opened_on,
                    Some(&team),
                )
                .await
                .map_err(|e| {
                    anyhow!(
                        "inject_team_message: target session {} not found, and resuming it failed: {}",
                        to_session_id,
                        e
                    )
                });
        };

        let session_uuid = Uuid::parse_str(to_session_id).ok();
        let prompt_content = format!("[team {}] {}", from_name, text);

        // Try to atomically claim the streaming slot (same approach as OOB listener).
        let won_race = is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();

        if !won_race {
            // A stream is already running — queue without interrupt; drain will
            // pick it up with priority 0 after the current turn ends.
            let mut queue = pending_messages.lock().await;
            queue.push_back(super::types::PendingMessage::team_message(
                text.to_string(),
                from_session_id,
                from_name.to_string(),
            ));
            debug!(
                from_session_id = %from_session_id,
                to_session_id = %to_session_id,
                "inject_team_message: session streaming — queued for drain"
            );
            return Ok(());
        }

        // Won the race — session is now "streaming" (flag = true). Persist and
        // broadcast the team_message event, then spawn stream_response directly.
        if let Some(uuid) = session_uuid {
            let event_record = crate::neo4j::models::ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: uuid,
                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                event_type: "team_message".to_string(),
                data: serde_json::to_string(&ChatEvent::TeamMessage {
                    from_session_id,
                    from_name: from_name.to_string(),
                    content: text.to_string(),
                })
                .unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            let graph = self.graph.clone();
            let to_id = to_session_id.to_string();
            tokio::spawn(async move {
                if let Err(e) = graph.store_chat_events(uuid, vec![event_record]).await {
                    warn!(
                        to_session_id = %to_id,
                        "inject_team_message: failed to persist team_message: {}", e
                    );
                }
            });
        }
        let team_event = ChatEvent::TeamMessage {
            from_session_id,
            from_name: from_name.to_string(),
            content: text.to_string(),
        };
        let _ = events_tx.send(team_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(to_session_id, team_event);
        }

        // Spawn stream_response: the team message IS the turn prompt.
        let session_id = to_session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let context_injector = self.context_injector.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();
        let cognitive_routing = self.cognitive_routing.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        tokio::spawn(async move {
            ChatManager::stream_response(
                client,
                events_tx,
                prompt_content,
                session_id,
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
                sdk_control_rx,
                auto_continue,
                retry_config,
                enrichment_pipeline,
                search,
                documents,
                cognitive_routing,
            )
            .await;
        });

        Ok(())
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
                let streaming = session.is_streaming.load(Ordering::SeqCst);
                info!(
                    session_id = %session_id,
                    op = op.name(),
                    id = ?op.id(),
                    found = outcome.found,
                    interrupt = outcome.interrupt,
                    streaming,
                    held_left = messages.len(),
                    "queue op applied (legacy engine)"
                );
                self.publish_pending_queue(session_id, &session.events_tx, messages);
                if outcome.interrupt && streaming {
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
        if self.refuse_if_mirror(session_id).await.is_err() {
            return Err(MessageDeliveryError::MirrorReadOnly);
        }
        // Read before anything awaits: see `resume_point`.
        let from = self.resume_point(session_id).await;
        if self.is_session_active(session_id).await {
            match self.queue_user_message(session_id, content).await {
                Ok(_) => return Ok(DeliveryRoute::Local),
                Err(send_err) => {
                    warn!(
                        session_id = %session_id,
                        error = %send_err,
                        "queue_user_message failed, attempting resume_session as fallback"
                    );
                    return match self
                        .resume_session_from(session_id, content, claims, from)
                        .await
                    {
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
        self.resume_session_from(session_id, content, claims, from)
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
        self.send_permission_response_inner(
            session_id,
            request_id,
            allow,
            super::types::PermissionAnswerScope::Once,
            false,
        )
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
        scope: super::types::PermissionAnswerScope,
        require_pending: bool,
    ) -> std::result::Result<(), PermissionDeliveryError> {
        let answer = PermissionAnswer {
            active_sessions: &self.active_sessions,
            graph: &self.graph,
            nats: self.nats.as_ref(),
            event_emitter: &self.event_emitter,
        };
        Self::deliver_permission_answer(
            answer,
            session_id,
            request_id,
            allow,
            scope,
            require_pending,
        )
        .await
    }

    /// The body of [`Self::send_permission_response_inner`], without `self`: the
    /// NATS RPC listener of a legacy session answers through it too, so a permission
    /// granted on another instance is the same `control_response` (subtype, request_id,
    /// behavior) written on `stdin_tx`, never under the client lock that a turn holds.
    ///
    ///
    /// `session`: the CLI gets a plain allow (never `updatedPermissions`: no rule is handed
    /// to the CLI); the grant (`session_grants::grant_for`) is kept in the session and the
    /// backend answers the later requests it covers. `always` (P11b): a plain allow too, none
    /// of the CLI's `permission_suggestions` applied; the lasting grant
    /// (`session_grants::lasting_grant_for`) is written for the session's person and project
    /// once the CLI has the answer. Either grant covers the identical call from BEFORE the
    /// CLI can read the answer (the CLI asks it again at once). An `always` or a `session`
    /// the call cannot be granted for
    /// (or a session that keeps no lasting rule): refused
    /// ([`PermissionDeliveryError::ScopeUnsupported`]), the request stays waiting.
    async fn deliver_permission_answer(
        ctx: PermissionAnswer<'_>,
        session_id: &str,
        request_id: &str,
        allow: bool,
        scope: super::types::PermissionAnswerScope,
        require_pending: bool,
    ) -> std::result::Result<(), PermissionDeliveryError> {
        // What is accepted is what the engine declares (`system_init.capabilities`), as on
        // the agent engine: a scope it does not declare is refused, never narrowed.
        // `always` is offered by a session that keeps lasting rules (`legacy_offered_scopes`).
        if allow
            && !LEGACY_PERMISSION_SCOPES.contains(&scope)
            && scope != super::types::PermissionAnswerScope::Always
        {
            return Err(PermissionDeliveryError::ScopeUnsupported(scope));
        }
        let (stdin_tx, pending_perm_inputs, events_tx, session_uuid, next_seq, session_grants) = {
            let mut sessions = ctx.active_sessions.write().await;
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
                session.session_grants.clone(),
            )
        };

        // Retrieve the original tool input stored when the permission_request arrived.
        // This is needed because the CLI's Zod schema for permission responses requires:
        //   Allow: { behavior: "allow", updatedInput: <record> }
        //   Deny:  { behavior: "deny",  message: <string> }
        // The `updatedInput` field REPLACES the original tool input in the CLI, so we
        // MUST pass back the original input — an empty {} would erase command/file_path/etc.
        // The answers to one request are handed over one at a time, from the claim to the
        // end of the write (`ToolClock::answering`): the CLI gets them in the order the
        // tool clock kept them.
        let tool_clock = super::tool_clock::ToolClock::for_session(session_id);
        let answering = tool_clock.answering(request_id).await;
        let claimed = pending_perm_inputs.lock().await.remove(request_id);
        if require_pending && claimed.is_none() {
            return Err(PermissionDeliveryError::NotPending);
        }
        // The entry is only really consumed once the decision reached the CLI:
        // on a failed send it is put back so a retry is still `pending`.
        let restore_claim = |entry: Option<PendingPermission>| {
            let pending = pending_perm_inputs.clone();
            let request_id = request_id.to_string();
            async move {
                if let Some(entry) = entry {
                    pending.lock().await.insert(request_id, entry);
                }
            }
        };
        // `always` kept as a lasting rule (`false`: as a session grant, the shell having
        // possibly left the project's root; review of #714, A).
        let mut for_good = false;
        // How long the approval lasts is the backend's business: `session` is a grant kept
        // here, `always` a lasting rule of the backend (P11b). Never a narrower or wider answer.
        let grant = match (allow, scope) {
            (false, _) | (true, super::types::PermissionAnswerScope::Once) => None,
            (true, super::types::PermissionAnswerScope::Session) => {
                // With this session's declared read-only MCP tools.
                let grant = claimed.as_ref().and_then(|c| {
                    session_grants
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .grant_for(&legacy_asked_call(c))
                });
                match grant {
                    Some(grant) => Some(grant),
                    None => {
                        restore_claim(claimed).await;
                        return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                    }
                }
            }
            // The lasting grant of the call (stricter than `session`), when this session
            // keeps lasting rules; written once the CLI has the answer.
            (true, super::types::PermissionAnswerScope::Always) => {
                let grant = claimed.as_ref().and_then(|c| {
                    session_grants
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .always_grant(&legacy_asked_call(c))
                });
                match grant {
                    Some((grant, lasting)) => {
                        for_good = lasting;
                        Some(grant)
                    }
                    None => {
                        restore_claim(claimed).await;
                        return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                    }
                }
            }
        };
        let original_input = claimed
            .as_ref()
            .map(|c| c.input.clone())
            .unwrap_or_else(|| serde_json::json!({}));

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
                restore_claim(claimed).await;
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
            scope = scope.as_str(),
            "Sending permission control response to CLI (via stdin_tx, lock-free)"
        );

        // The grant covers, from now on, the requests that arrive while the CLI takes the
        // answer and the rule is written: the CLI asks the identical call again as soon as
        // it has the answer (about 1 ms after the write, measured on the parity scenario,
        // while the next statements here ran 50 µs after it; a thread preempted in between
        // left the second call to the user — FLAKY-PARITE-ALWAYS). Begun BEFORE the write,
        // as on the agent engine (`agent_runtime::PendingGrant`); taken back if it fails.
        let lasting =
            for_good && claimed.is_some() && scope == super::types::PermissionAnswerScope::Always;
        if let Some(grant) = &grant {
            let mut grants = session_grants.lock().unwrap_or_else(|p| p.into_inner());
            if lasting {
                grants.begin_lasting(grant.clone());
            } else {
                grants.begin_pending(grant.clone());
            }
        }
        // On the tool clock BEFORE the CLI has it: a fast tool's result cannot overtake it.
        let mark = tool_clock.decided(request_id, allow, chrono::Utc::now());
        if let Err(e) = stdin_tx.send(json).await {
            tool_clock.undecided(request_id, mark);
            if let Some(grant) = &grant {
                let mut grants = session_grants.lock().unwrap_or_else(|p| p.into_inner());
                if lasting {
                    grants.abort_lasting(grant);
                } else {
                    grants.end_pending(grant, false);
                }
            }
            restore_claim(claimed).await;
            return Err(PermissionDeliveryError::Failed(anyhow!(
                "Failed to send permission control response: {}",
                e
            )));
        }
        #[cfg(test)]
        answer_probe::cli_has_answer(session_id).await;
        drop(answering);
        // Kept for good only now that the CLI has the answer. A lasting rule that cannot be
        // written leaves the call granted for this session, and the decision says `session`.
        let mut kept = scope;
        match (&grant, &claimed, scope) {
            (Some(grant), Some(c), super::types::PermissionAnswerScope::Always) if lasting => {
                let place = {
                    let grants = session_grants.lock().unwrap_or_else(|p| p.into_inner());
                    // The shell may have moved since: then nothing is written.
                    (!grants.cwd_bound(&legacy_asked_call(c)))
                        .then(|| grants.lasting_scope())
                        .flatten()
                };
                let written =
                    super::lasting_rules::keep(place, &legacy_asked_call(c), grant, session_id)
                        .await;
                session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .end_lasting(grant, written);
                if !written {
                    kept = super::types::PermissionAnswerScope::Session;
                }
            }
            (Some(grant), _, _) => {
                if scope == super::types::PermissionAnswerScope::Always {
                    kept = super::types::PermissionAnswerScope::Session;
                }
                // Pending since before the write: kept for the rest of the session.
                session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .end_pending(grant, true)
            }
            (None, _, _) => {}
        }

        // Persist and broadcast the permission decision so it survives session reload.
        let decision_event = ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
            scope: kept.lasting(allow),
            rule: grant
                .as_ref()
                .map(super::session_grants::SessionGrant::describe),
        };

        // Broadcast to all connected WebSocket clients
        let _ = events_tx.send(decision_event.clone());
        if let Some(nats) = ctx.nats {
            nats.publish_chat_event(session_id, decision_event.clone());
        }
        notify_attention(
            ctx.event_emitter,
            AttentionSubject::Session(session_id.to_string()),
            AttentionReason::PermissionDecision,
        );

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
            if let Err(e) = ctx.graph.store_chat_events(uuid, vec![record]).await {
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
    ///
    /// `scope` says how long an approval lasts; a scope the session does not declare
    /// is `ScopeUnsupported` wherever the session lives (never a narrower answer).
    pub async fn route_permission_response(
        &self,
        session_id: &str,
        request_id: &str,
        allow: bool,
        scope: super::types::PermissionAnswerScope,
        require_pending: bool,
    ) -> std::result::Result<DeliveryRoute, PermissionDeliveryError> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle
                .answer_permission_scoped(request_id, allow, scope)
                .await
                .map(|()| DeliveryRoute::Local);
        }
        if self.is_session_active(session_id).await {
            return self
                .send_permission_response_inner(
                    session_id,
                    request_id,
                    allow,
                    scope,
                    require_pending,
                )
                .await
                .map(|()| DeliveryRoute::Local);
        }
        // The message_type "control_response" tells the receiving instance to
        // answer the permission, not to send a message. Both engines answer with
        // `request_id`: the CLI matches the control_response to its can_use_tool by it.
        // The owner answers strictly (a request no longer waiting is refused), so the
        // refusal comes back typed. Asked ONCE: a resend would write the answer twice.
        let payload = serde_json::json!({
            "allow": allow,
            "request_id": request_id,
            "scope": scope,
        })
        .to_string();
        let Some(ref nats) = self.nats else {
            return Err(PermissionDeliveryError::SessionDead(session_id.to_string()));
        };
        match nats
            .request_send_message(session_id, &payload, "control_response")
            .await
        {
            Some(answer) if answer.success => Ok(DeliveryRoute::Remote),
            Some(answer) if answer.error.as_deref() == Some(PERMISSION_NOT_PENDING_RPC) => {
                Err(PermissionDeliveryError::NotPending)
            }
            Some(answer) if answer.error.as_deref() == Some(PERMISSION_SCOPE_UNSUPPORTED_RPC) => {
                Err(PermissionDeliveryError::ScopeUnsupported(scope))
            }
            Some(answer) => {
                debug!(
                    session_id = %session_id,
                    error = ?answer.error,
                    "the remote instance did not deliver the permission answer"
                );
                Err(PermissionDeliveryError::SessionDead(session_id.to_string()))
            }
            None => Err(PermissionDeliveryError::SessionDead(session_id.to_string())),
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
    ///
    /// Only the entry whose CLI was found dead is removed: if a resume replaced it
    /// meanwhile (between the read and the write lock), the new session and its
    /// NATS listeners are left alone (review of #673, finding 4).
    pub(crate) async fn evict_if_cli_dead(&self, session_id: &str) -> bool {
        let client = match self.active_sessions.read().await.get(session_id) {
            Some(s) => s.client.clone(),
            None => return false,
        };
        let dead = match client.try_lock() {
            Ok(c) => !c.is_alive().await,
            Err(_) => false,
        };
        if !dead || !Self::forget_if_client(&self.active_sessions, session_id, &client).await {
            return false;
        }
        warn!(
            session_id = %session_id,
            "CLI process is gone, dropped the session from active_sessions so the message resumes it"
        );
        true
    }

    /// Removes a legacy session from `active_sessions` AND ends its per-session
    /// NATS listeners (`nats_cancel`), only while the entry is still the one of
    /// `client` (`Arc::ptr_eq`, under the write lock): a resume that replaced it
    /// keeps its session and its listeners (review of #673, finding 4; of #685,
    /// finding 2). Whether it was removed.
    ///
    /// A removal that leaves the listeners running keeps an instance answering for
    /// a session it no longer holds: after a resume on another instance, its `gone`
    /// could reach a canceller before the real owner's answer (review N1 of #663).
    /// Every removal of a dead CLI goes through here; the others cancel
    /// `nats_cancel` themselves (`close_session`, a replaced session).
    async fn forget_if_client(
        active_sessions: &RwLock<HashMap<String, ActiveSession>>,
        session_id: &str,
        client: &Arc<Mutex<InteractiveClient>>,
    ) -> bool {
        let removed = {
            let mut sessions = active_sessions.write().await;
            match sessions.get(session_id) {
                Some(s) if Arc::ptr_eq(&s.client, client) => sessions.remove(session_id),
                _ => None,
            }
        };
        match removed {
            Some(session) => {
                session.nats_cancel.cancel();
                true
            }
            None => false,
        }
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
        // A read-only mirror of a Claude Code sub-agent takes no message (P18c).
        if self.refuse_if_mirror(session_id).await.is_err() {
            return Err(MessageDeliveryError::MirrorReadOnly);
        }
        // A registered session whose CLI has left would swallow the message (the
        // write goes to a closed stdin and the turn ends empty): drop it so the
        // message takes the resume path below.
        self.evict_if_cli_dead(session_id).await;
        // Read before anything awaits: see `resume_point`.
        let from = self.resume_point(session_id).await;
        if self.is_session_active(session_id).await {
            match self.send_message(session_id, content).await {
                Ok(()) => return Ok(DeliveryRoute::Local),
                Err(send_err) => {
                    warn!(
                        session_id = %session_id,
                        error = %send_err,
                        "send_message failed, attempting resume_session as fallback"
                    );
                    return match self
                        .resume_session_from(session_id, content, claims, from)
                        .await
                    {
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
        self.resume_session_from(session_id, content, claims, from)
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
    /// Records whether a person lifted the session's file scope (decision fce98b8a): persisted
    /// on the node (a resume launches `nexus-tools` from it) and live (a process launched
    /// `--unscoped` is confined by its hooks once the lift ends). Persistence failures are
    /// logged: the live state still confines.
    pub async fn set_file_scope_lifted(&self, session_id: &str, lifted: bool) {
        crate::auth::agent_tokens::set_scope_lift_live(session_id, lifted);
        // The same two places set and clear a person's choice of trust, which Rock'n roll's
        // network gate reads (`trust_opens_network`).
        crate::auth::agent_tokens::set_person_trust(session_id, lifted);
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self
                .graph
                .update_chat_session_file_scope_lifted(uuid, lifted)
                .await
            {
                warn!(session_id = %session_id, error = %e, "Failed to persist the file-scope lift of a session (non-fatal)");
            }
        }
        info!(session_id = %session_id, file_scope_lifted = lifted, "file scope lift of the session changed");
    }

    /// A PERSON switched the session to trust (the WebSocket handler checked owner and person):
    /// the file scope is lifted. It takes effect on the live process if it was launched
    /// `--unscoped`, otherwise at the next launch (resume).
    pub async fn lift_file_scope(&self, session_id: &str) {
        self.set_file_scope_lifted(session_id, true).await;
    }

    pub async fn set_session_permission_mode(&self, session_id: &str, mode: &str) -> Result<()> {
        self.refuse_if_mirror(session_id).await?;
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
            // Leaving trust ends a person's lift of the file scope (decision fce98b8a): only a
            // person sets it again ([`Self::lift_file_scope`]).
            if pair.neutral != nexus_claude::agent::PolicyMode::Trust {
                self.set_file_scope_lifted(session_id, false).await;
            }
            // What a child's ceiling reads (A17, P18): kept live, and persisted so a resume and
            // the envelope after a restart see the mode the session is in, not its opening one.
            self.agent_modes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(session_id.to_string(), pair.legacy.to_string());
            if let Ok(uuid) = Uuid::parse_str(session_id) {
                if let Err(e) = self
                    .graph
                    .update_chat_session_permission_mode(uuid, pair.legacy)
                    .await
                {
                    warn!(session_id = %session_id, error = %e, "Failed to persist the permission mode of an agent session (non-fatal)");
                }
            }
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
        if super::envelope::mode_rank(mode) < super::envelope::TRUST_RANK {
            self.set_file_scope_lifted(session_id, false).await;
        }

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

        // A read-only mirror never moves to another provider (P18c): the move would open a
        // writable session holding the same `tool_use_id`, and write into the mirror's thread.
        self.refuse_if_mirror(session_id).await?;
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

        // The model the moved conversation opens on, resolved BEFORE anything is torn down:
        // the explicit one, else the target's own default (never the old provider's model name).
        let target_model = self.resolve_target_model(provider, model).await?;

        // Pre-flight: the consent/gate/journal check runs BEFORE anything is stopped or closed,
        // so a refused move never cuts a running turn nor tears the old process down (the old
        // session stays exactly as it was: process live, resume token intact).
        self.authorize_provider_switch(&node, provider, &target_model, claims.as_ref())
            .await?;

        // A turn still streaming is cut by the move: keep the answer it had produced before the
        // conversation so far is read from what was stored.
        self.settle_cut_turn(session_id).await;

        // The conversation so far, from what was stored.
        let records = self.graph.get_chat_events(previous, 0, 5_000).await?;
        let events: Vec<ChatEvent> = records
            .iter()
            .filter_map(|record| serde_json::from_str(&record.data).ok())
            .collect();
        // A relay may use 40 % of the target's window when it is known.
        let window = match self.provider_for(provider).await {
            Ok(target) => target
                .capabilities(Some(&target_model))
                .context_window
                .map(|w| w.value),
            Err(_) => None,
        };
        let mut relayed = relay::RelayedFrom {
            relay: relay::render_relay(
                &events,
                &current,
                provider,
                relay::budget_for_window(window),
            ),
            from_session_id: session_id.to_string(),
            from_provider: current.clone(),
            moved_by: moved_by.to_string(),
            // The memory conversation goes on in the same session.
            conversation_id: node.conversation_id.clone(),
            segments: None,
            next_turn: 0,
        };
        // Carry the live router's segmenter state (segments + boundaries) and its segmenter
        // turn counter, BEFORE teardown destroys the live router: the reopened router resumes
        // from them instead of resetting the conversation's segmentation.
        if let Some(router) = self.turn_routing.get(session_id) {
            relayed.segments = Some(router.snapshot_segments());
            relayed.next_turn = router.next_turn();
        }
        let rendered = &relayed.relay;

        let by_router = moved_by == relay::MOVED_BY_AUTO;
        let routed_by = if by_router {
            relay::MOVED_BY_AUTO
        } else {
            "request"
        };
        // The routing the target keeps, as `create_session_resolved` persisted on a NEW
        // session before the in-place flow: a router move keeps the conversation's mode and
        // pool (a pool without a mode reads `full`, or the reopened target would decide its
        // model again among the pool — review of #713, C1-bis); a user move imposes the
        // target and keeps none.
        let carried_mode = if by_router {
            node.routing_mode
                .as_deref()
                .and_then(|m| serde_json::from_value(serde_json::Value::String(m.to_owned())).ok())
                .or_else(|| {
                    node.routing_pool
                        .is_some()
                        .then_some(super::provider::cognitive::ConversationRoutingMode::Full)
                })
        } else {
            None
        };
        let carried_pool = if by_router {
            node.routing_pool.clone()
        } else {
            None
        };

        // Re-target the SAME id in place: the node is updated, the old provider process is
        // closed quietly (`close_session_for_switch`, which emits no `session_closed` and,
        // unlike `close_session`, does not remove a neutral directory), then the same id is
        // reopened on the target with its stored history relayed in front of the first message.
        // The engine written is the one the target runs on, so a later resume of the SAME
        // session routes on the branch the switch reopened with (agent vs legacy).
        let engine = if self.engine_is_agent(provider) {
            ChatSessionNode::ENGINE_AGENT
        } else {
            ChatSessionNode::ENGINE_LEGACY
        };
        self.graph
            .update_chat_session_provider(previous, provider, &target_model, routed_by, engine)
            .await?;
        // The channel a client is LISTENING on, taken while it still exists. An in-place
        // switch is meant to replace it: `close_session_for_switch` drops the session (and,
        // with it, its sender) and the reopen makes a NEW channel. A socket that subscribed
        // to the old one cannot hear the switch on the new one — it is not attached to it —
        // and, measured on the agent engine (2026-11-04, `agent_e2e_tests`, in-place move in
        // Auto), it does not hear `RecvError::Closed` either: something around the reopened
        // session keeps the old sender alive, so that channel stays open and silent forever.
        // The tab then sat on its typing indicator until the page was reloaded. The switch is
        // therefore ANNOUNCED on the channel it replaces, below.
        let replaced_channel = self.get_events_tx(session_id).await.ok();
        if let Err(error) = self.close_session_for_switch(session_id).await {
            debug!(%session_id, %error, "closing the provider process before an in-place switch");
        }
        if let Err(error) = self
            .reopen_session_on_provider(session_id, message, claims.as_ref(), Some(&relayed))
            .await
        {
            // The move failed after the node was retargeted and the old process closed:
            // restore EVERY cleared field (provider, model, routed_by, resume_token,
            // capabilities, cli_session_id, engine), so a later resume finds the old session.
            self.graph
                .restore_chat_session_provider(&node, &current)
                .await?;
            return Err(error);
        }
        // The target keeps the routing a router move carried (or none, for a user move),
        // persisted after the reopen so a later resume routes on it.
        self.graph
            .update_chat_session_routing(
                previous,
                carried_mode.as_ref().map(|m| m.as_str()),
                carried_pool.as_deref(),
                Some(routed_by),
            )
            .await?;

        // The SAME thread states the switch — with the relay counts, so a bounded relay
        // is never a silent truncation; there is no new session to follow.
        let switched = relayed.event(session_id, provider, &target_model);
        self.emit_on_session(session_id, switched.clone()).await;
        // …and the client attached to the channel this switch just replaced hears the same
        // statement on it, then nothing more. That frame is the ONLY way it can learn the
        // session went on without it: `provider_switched` is what makes a socket re-open
        // itself on the session (the tab followed the switch already, from the response of
        // the route it called — a router move has no such response). Sent on the channel
        // directly, not through `emit_on_session`: the event is persisted and published
        // once, above, and a second pass would give every other client a duplicate.
        if let Some(replaced) = replaced_channel {
            let _ = replaced.send(switched);
        }

        let note = serde_json::json!({
            "from_session": session_id,
            "from_provider": current,
            "relayed_entries": rendered.included,
            "omitted_entries": rendered.omitted,
        });
        if let Err(error) = self
            .graph
            .put_llm_setting(&format!("handoff:{session_id}"), "note", &note.to_string())
            .await
        {
            warn!(session_id = %session_id, %error, "recording the provider handoff failed (non-fatal)");
        }
        Ok(super::types::SwitchProviderResponse {
            session_id: session_id.to_string(),
            stream_url: format!("/ws/chat/{session_id}"),
            previous_session_id: session_id.to_string(),
            relayed_entries: rendered.included,
            omitted_entries: rendered.omitted,
            conversation_id: relayed.conversation_id.clone(),
        })
    }

    /// Stops the turn a provider switch is about to cut, and keeps the answer it had produced.
    ///
    /// On the agent engine a turn stores its answer only at its end, while the text streamed so
    /// far lives in the session's buffer, which is cleared when the turn ends. Without this, a
    /// conversation moved while the answer was streaming lost that answer: the target read the
    /// question with no reply, and the thread showed nothing for it. The longest text the buffer
    /// held is stored as the assistant's answer, before the relay is read. A session that is not
    /// streaming is left as it is. Bounded: the wait never holds the switch longer than
    /// `CUT_TURN_SETTLE`.
    async fn settle_cut_turn(&self, session_id: &str) {
        let Some(handle) = self.agent_runtime.get(session_id).await else {
            return;
        };
        if !handle.is_streaming.load(Ordering::SeqCst) {
            return;
        }
        let mut partial = handle.streaming_text.lock().await.clone();
        let _ = self.interrupt(session_id).await;
        let deadline = tokio::time::Instant::now() + CUT_TURN_SETTLE;
        while handle.is_streaming.load(Ordering::SeqCst) && tokio::time::Instant::now() < deadline {
            let text = handle.streaming_text.lock().await.clone();
            if text.len() > partial.len() {
                partial = text;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let text = handle.streaming_text.lock().await.clone();
        if text.len() > partial.len() {
            partial = text;
        }
        if partial.trim().is_empty() {
            return;
        }
        handle
            .emit(ChatEvent::AssistantText {
                content: partial,
                parent_tool_use_id: None,
            })
            .await;
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
        self.refuse_if_mirror(session_id).await?;
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
    ///   session's provider (one session = one provider); when the ticks span two providers
    ///   or more, the router may also move the conversation to the other provider before a
    ///   turn, like `full` restricted to the pool (decision R-S2).
    pub async fn set_session_routing(
        &self,
        session_id: &str,
        body: &super::types::SessionRoutingRequest,
    ) -> Result<ChatSessionNode> {
        use super::provider::cognitive::ConversationRoutingMode as Mode;
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
                router.reroute(Mode::Full, None, None, false);
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
                        router.reroute(Mode::Primary, None, None, true);
                    }
                    (Mode::Primary, None, "request")
                }
                _ => {
                    let allowed = Self::allowed_models_of(Some(&pool), &provider)
                        .filter(|models| !models.is_empty())
                        .ok_or_else(|| anyhow::Error::new(Refused::OtherProvider(provider)))?;
                    self.unpin_session_model(session_id).await;
                    if let Some(router) = &router {
                        // The whole pool, every provider: a pool ticked across providers
                        // may move the conversation from the next turn on (R-S2).
                        router.reroute(Mode::Mixed, Some(allowed), Some(pool.clone()), false);
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

    /// PYSR-3b: a live session that holds a spend cap (a delegation, a sub-agent of one:
    /// [`Self::session_limits_of`]) is not moved to a model its engine cannot hold that cap on
    /// (a native model without a price): the engine checks the cap against the price at the
    /// opening only, so the session would go on spending with nothing counted. Refused as the
    /// engine refuses such an opening, `unsupported("cost")` (a 422 over HTTP); the model in
    /// force stays. A dormant session is not refused here: its resume leaves the cap out and
    /// says so (`degraded_features: cost_cap`).
    async fn refuse_a_model_that_drops_the_cost_cap(
        &self,
        session_id: &str,
        model: &str,
    ) -> Result<()> {
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            return Ok(());
        };
        let Some(node) = self.graph.get_chat_session(uuid).await.ok().flatten() else {
            return Ok(());
        };
        if node.model == model {
            return Ok(());
        }
        let capped = self
            .session_limits_of(&node)
            .await
            .is_some_and(|limits| limits.request_caps().1.is_some());
        if !capped {
            return Ok(());
        }
        let provider_id = node
            .provider_id
            .clone()
            .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string());
        let provider = self.provider_for(&provider_id).await?;
        if super::provider::policy::model_change_drops_the_cost_cap(
            provider.kind(),
            provider.capabilities(Some(&node.model)).cost,
            provider.capabilities(Some(model)).cost,
        ) {
            warn!(
                session_id,
                from = %node.model,
                to = model,
                "model change refused: the session's spend cap cannot be held on that model"
            );
            return Err(anyhow::Error::new(
                nexus_claude::agent::ProviderError::unsupported("cost"),
            ));
        }
        Ok(())
    }

    /// `manual`: the user asked for it, which ends the automatic model routing of the
    /// session. The router's own changes pass `false`.
    async fn set_session_model_inner(
        &self,
        session_id: &str,
        model: &str,
        manual: bool,
    ) -> Result<bool> {
        let live_agent = self.agent_runtime.get(session_id).await;
        // Before anything is pinned or flagged: a refused change changes nothing.
        if live_agent.is_some() {
            self.refuse_a_model_that_drops_the_cost_cap(session_id, model)
                .await?;
        }
        if manual {
            if let Some(router) = self.turn_routing.get(session_id) {
                router.mark_manual();
            }
            self.pin_session_model(session_id).await;
            self.flag_routing_override(session_id).await;
        }
        if let Some(handle) = live_agent {
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
                    reason: None,
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
                    reason: None,
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

    /// The auto-continue state of a session, as `auto_continue_state_changed` reports it.
    ///
    /// Auto-continue is always on (decision 7b95342a): there is no per-session toggle any
    /// more, and the WebSocket / NATS `set_auto_continue` message is accepted and ignored.
    /// A live session answers its own flag (a runner always continues); any other session
    /// answers the operator's `CHAT_AUTO_CONTINUE` (on unless turned off). The flag an
    /// older version persisted on the session node is not read.
    pub async fn get_auto_continue_state(&self, session_id: &str) -> Result<bool> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return Ok(handle.auto_continue.load(Ordering::Relaxed));
        }
        let sessions = self.active_sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            return Ok(session.auto_continue.load(Ordering::Relaxed));
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
            .with_work_log(work_log)
            .with_routing(Arc::clone(&self.turn_routing));
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
        // PreToolUse → the session's tool clock notes when the engine takes a call up
        // (every session, runner or not: it adds nothing to the context).
        super::tool_clock::clock_the_table(
            &mut hooks,
            super::tool_clock::ToolClock::for_session(&session_id),
        );
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

    /// ISO1: the hooks the HISTORICAL engine answers Claude Code's own events with. Registers
    /// their callbacks in the session's table (`agent_hooks::serve_lifecycle`).
    fn serve_lifecycle_hooks(
        &self,
        table: &mut super::agent_hooks::HookTable,
        session_id: &str,
        cwd: &str,
        permission_mode: Option<String>,
    ) -> Arc<dyn nexus_claude::agent::SessionHooks> {
        super::agent_hooks::serve_lifecycle(table, session_id, cwd, permission_mode, |hooks| {
            self.provider_source.wrap_lifecycle(hooks)
        })
    }

    /// The hooks a resumed Claude CLI session is given: the SAME as at its creation
    /// (`create_session` → [`Self::graph_hook_table`]), B14.
    fn resume_hook_table(
        &self,
        session_node: &ChatSessionNode,
        events_tx: &broadcast::Sender<ChatEvent>,
        work_log: &Arc<Mutex<SessionWorkLog>>,
    ) -> super::agent_hooks::HookTable {
        // The compaction is told about the task a runner works on, else the project,
        // as at creation.
        let context_source = match session_node
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)
            .and_then(|ctx| ctx.task_id)
        {
            Some(task_id) => CompactionContextSource::Task(task_id),
            None => match session_node.project_slug.as_deref() {
                Some(slug) => CompactionContextSource::Session(slug.to_string()),
                None => CompactionContextSource::None,
            },
        };
        self.graph_hook_table(GraphHookInput {
            session_id: session_node.id.to_string(),
            context_source,
            work_log: Arc::clone(work_log),
            // A runner has its task context in the prompt: no per-tool hook, neither
            // the skill activation nor the redirect advice (the resume used to
            // register the latter unconditionally).
            tool_knowledge: !is_runner_spawned(session_node.spawned_by.as_deref()),
            announce: Some(events_tx.clone()),
        })
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

    /// The live handle (if any) of `session_id`, read by a caller BEFORE it checks whether
    /// the session is live and before it asks another instance (`try_remote_send`, up to
    /// the NATS RPC timeout): handed to [`Self::resume_session_from`], a resume that finds
    /// ANOTHER live handle once it holds the session's lock knows a resume succeeded while
    /// the caller was looking, and delivers to it (follow-up of the review of #696).
    pub async fn resume_point(&self, session_id: &str) -> ResumeFrom {
        ResumeFrom(self.agent_runtime.get(session_id).await)
    }

    /// [`Self::resume_session_from`], from the live handle of now: for a caller that did
    /// not look at the session before (a caller that checked `is_session_active` or tried
    /// another instance first reads [`Self::resume_point`] before, and passes it).
    pub async fn resume_session(
        &self,
        session_id: &str,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<()> {
        let from = self.resume_point(session_id).await;
        self.resume_session_from(session_id, message, user_claims, from)
            .await
    }

    /// Resumes `session_id` and delivers `message`, starting from `from`: the live handle
    /// the caller saw before it awaited anything (its liveness check, a failed send, the
    /// NATS RPC). A resume that finds ANOTHER live handle once it holds the session's lock
    /// was overtaken by a resume that succeeded meanwhile, and delivers to it (review of
    /// #696, finding 1, and its follow-up: the caller's window).
    pub async fn resume_session_from(
        &self,
        session_id: &str,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        from: ResumeFrom,
    ) -> Result<()> {
        let mut opened_on: Option<String> = None;
        let result = self
            .resume_session_on(session_id, message, user_claims, from, &mut opened_on, None)
            .await;
        // A failure once the session is read names ITS provider (B6).
        result.map_err(|e| match opened_on {
            Some(provider_id) => super::provider::errors::on_provider(&provider_id, e),
            None => e,
        })
    }

    /// [`Self::resume_session_from`]; `opened_on` receives the session's provider as soon
    /// as the session is read.
    async fn resume_session_on(
        &self,
        session_id: &str,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        from: ResumeFrom,
        opened_on: &mut Option<String>,
        team: Option<&TeamInbound>,
    ) -> Result<()> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;
        let seen = from.0;

        // Load session from Neo4j
        let mut session_node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found in database", session_id))?;
        // A read-only mirror of a Claude Code sub-agent has no process to resume (P18c).
        if session_node.mirror_of.is_some() {
            return Err(super::mirror::MirrorReadOnly.into());
        }
        *opened_on = Some(
            session_node
                .provider_id
                .clone()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string()),
        );
        self.ready_for_resume(&mut session_node).await?;

        // A resume that carries no identity (the restart recovery, or a resume that
        // arrives without claims) re-derives it from the owner stored at open — without
        // it, `po_mcp_env` mints no `agent_session` token and the PO tools lose their auth
        // (P20-suite-2). A session with no owner stays without one, rather than forging it.
        let rederived_claims;
        let user_claims: Option<&crate::auth::jwt::Claims> = match user_claims {
            Some(claims) => Some(claims),
            None => {
                rederived_claims = self.resume_identity(&session_node).await;
                rederived_claims.as_ref()
            }
        };

        // The engine is frozen at opening (M2): a session written as opened on the agent
        // engine resumes there, whether or not its capability snapshot was stored.
        if session_node.opened_on_agent() {
            return self
                .resume_agent_session(&session_node, message, user_claims, seen)
                .await;
        }

        self.reopen_legacy_session_for(&session_node, message, user_claims, None, true, team)
            .await
    }

    /// Reopens a session on the legacy Claude Code CLI (the historical engine): spawns the
    /// CLI and delivers `message`. It is the shared legacy branch of [`Self::resume_session_on`];
    /// `relay` names an in-place provider switch, whose first message goes out relay-prefixed
    /// and whose router is re-seeded from the carried segmenter state.
    async fn reopen_legacy_session(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        relay: Option<&super::relay::RelayedFrom>,
    ) -> Result<()> {
        self.reopen_legacy_session_with(node, message, user_claims, relay, true)
            .await
    }

    /// [`Self::reopen_legacy_session`]; `fresh_if_refused == false` (the restart recovery)
    /// refuses to fall back to a fresh CLI when `--resume` is refused: a cut turn handed to
    /// a CLI that lost the conversation would continue blind, so the caller closes it.
    pub(super) async fn reopen_legacy_session_with(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        relay: Option<&super::relay::RelayedFrom>,
        fresh_if_refused: bool,
    ) -> Result<()> {
        self.reopen_legacy_session_for(node, message, user_claims, relay, fresh_if_refused, None)
            .await
    }

    /// [`Self::reopen_legacy_session_with`]; `team` makes the stored and broadcast inbound
    /// event the peer's typed `team_message` instead of a user message.
    async fn reopen_legacy_session_for(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        relay: Option<&super::relay::RelayedFrom>,
        fresh_if_refused: bool,
        team: Option<&TeamInbound>,
    ) -> Result<()> {
        let sid = node.id.to_string();
        let session_id: &str = sid.as_str();
        let uuid = node.id;
        let session_node = node;
        // (P20-suite-2) As on a resume: re-derive the person from the stored owner when the
        // reopen has no caller claims, so the session token can be minted again.
        let rederived_claims;
        let user_claims: Option<&crate::auth::jwt::Claims> = match user_claims {
            Some(claims) => Some(claims),
            None => {
                rederived_claims = self.resume_identity(node).await;
                rederived_claims.as_ref()
            }
        };
        let cli_session_id = session_node.cli_session_id.as_deref();

        if let Some(cli_id) = cli_session_id {
            info!("Resuming session {} with CLI ID {}", session_id, cli_id);
        } else {
            info!(
                "Starting fresh CLI for session {} (no previous cli_session_id)",
                session_id
            );
        }

        // The window the prompt and the per-turn enrichment are budgeted for (CTX4).
        let session_window = self
            .session_context_window(
                session_node
                    .provider_id
                    .as_deref()
                    .filter(|p| !p.is_empty())
                    .unwrap_or(super::provider::resolver::CLAUDE_CODE),
                &session_node.model,
                session_node.capabilities.as_deref(),
            )
            .await;
        // Build options - with resume flag only if we have a cli_session_id
        let (system_prompt, _included_note_ids) = self
            .build_system_prompt_anchored_for(
                &super::composer::PromptAudience::of(
                    session_node.spawned_by.as_deref(),
                    session_node.provider_id.as_deref().unwrap_or(""),
                ),
                session_node.execution_place,
                &session_node.cwd,
                session_node.project_slug.as_deref(),
                message,
                Some(&session_node.model),
                session_window,
                session_id,
                None,
            )
            .await;

        // Create broadcast channel early so CompactionNotifier can use the sender
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);

        // Create work_log early so CompactionNotifier can reference it
        let work_log = Arc::new(Mutex::new(SessionWorkLog::default()));

        // Build session hooks: the same table as at creation (B14).
        let mut session_hooks = self.resume_hook_table(session_node, &events_tx, &work_log);
        // ISO1: Claude Code's own events consult the session's hooks (see create_session).
        let lifecycle_hooks = self.serve_lifecycle_hooks(
            &mut session_hooks,
            session_id,
            &session_node.cwd,
            session_node.permission_mode.clone(),
        );

        // A resume keeps the access the session was opened with and can only narrow it.
        let access = super::provider::policy::SessionAccess::for_resume(session_node.access, None);
        let resume_add_dirs = session_node.add_dirs.clone().unwrap_or_default();

        // A move re-seeds the router (the quiet close of the switch removed it): a router
        // move (`auto`) keeps the conversation's routing (mode, pool) and lets it choose
        // again; a user move imposes the target. The legacy engine counts its first routed
        // turn after the relayed one, hence `next_turn: 1`.
        if let Some(relay) = relay {
            let moved_in = relay.moved_by == super::relay::MOVED_BY_AUTO;
            let routing_mode = if moved_in {
                session_node
                    .routing_mode
                    .as_deref()
                    .and_then(|m| {
                        serde_json::from_value(serde_json::Value::String(m.to_owned())).ok()
                    })
                    .or_else(|| {
                        (session_node.routing_pool.is_some())
                            .then_some(super::provider::cognitive::ConversationRoutingMode::Full)
                    })
            } else {
                None
            };
            let routing_pool = if moved_in {
                session_node
                    .routing_pool
                    .as_deref()
                    .and_then(|json| serde_json::from_str(json).ok())
            } else {
                None
            };
            self.register_turn_router(
                session_id,
                super::provider::resolver::CLAUDE_CODE,
                &session_node.model,
                session_node.project_slug.as_deref(),
                OpeningTurn {
                    routing_mode,
                    routing_pool,
                    explicit_model: false,
                    permission_mode: session_node.permission_mode.as_deref(),
                    provider_imposed: !moved_in && session_node.routing_pool.is_none(),
                    moved_in,
                    message,
                    next_turn: 1,
                    segments: relay.segments.clone(),
                    owner: session_node.owner.as_deref(),
                },
            )
            .await;
        }

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
                if resume_with.is_some() && fresh_if_refused {
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
                if !fresh_if_refused {
                    let _ = candidate.disconnect().await;
                    return Err(anyhow!(
                        "the Claude CLI refused to resume its conversation (--resume)"
                    ));
                }
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
        // ISO1: the CLI sends no SessionStart to SDK callbacks: the host asks it itself.
        let source = if resume_with.is_some() {
            "resume"
        } else {
            "startup"
        };
        super::agent_hooks::ask_session_start(&lifecycle_hooks, source, session_id).await;

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
                Some(Arc::clone(&lifecycle_hooks)),
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
        // Auto-continue is always on (decision 7b95342a): the flag an older version
        // persisted on the session node is no longer read; only the kill switch counts.
        let (auto_continue_on, max_auto_continues) =
            super::post_stream::session_auto_continue(false, self.config.auto_continue);
        let auto_continue = Arc::new(AtomicBool::new(auto_continue_on));
        // Resolved before the sessions are locked (it reads the project's record).
        // Boxed: the opening's future is already large.
        let lasting = Box::pin(self.lasting_scope(
            session_node.owner.as_deref(),
            session_node.project_slug.as_deref(),
            &session_node.cwd,
        ))
        .await;
        // A resumed session starts moved unless it is known its shell never left the root.
        let session_grants = self
            .session_grants_for(
                session_id,
                Some(std::path::Path::new(&crate::expand_tilde(
                    &session_node.cwd,
                ))),
                lasting,
                true,
                super::session_grants::Asker::ClaudeCode,
            )
            .await;
        // Its Claude Code sub-agents are mirrored (P18c): a new tap per resume, the registry
        // keeps one mirror per call.
        let mirror_tap = self.mirror_tap(session_id);
        let parent =
            super::cancelled_parents::parent_checked_on_resume(session_node.spawned_by.as_deref());
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // B36-RACE: a child the warm pool resumes for a new delegation, whose parent a
            // cascade closed, is refused HERE, under the lock that would make it live. A person
            // resuming it passes (B36-RACE-RESUME, `parent_checked_on_resume`).
            if let Err(refused) = self.cancelled_parents.admit(session_id, parent.as_deref()) {
                drop(sessions);
                self.refuse_legacy_activation(session_id, client, &refused)
                    .await;
                return Err(anyhow::Error::new(refused));
            }
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
                    context_window: session_window,
                    anchor: self.anchor_session(),
                    mirror_tap: mirror_tap.clone(),
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
                    turn_generation: Arc::new(AtomicU64::new(0)),
                    result_cursor: Default::default(),
                    stream_task: Arc::new(std::sync::Mutex::new(None)),
                    post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
                    pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                        std::collections::HashMap::new(),
                    )),
                    auto_continue: auto_continue.clone(),
                    auto_continue_count: Arc::new(AtomicU32::new(0)),
                    max_auto_continues,
                    rfc_accumulator: Arc::new(Mutex::new(
                        super::observation_detector::RfcAccumulator::new(),
                    )),
                    protocol_run_id: None,
                    protocol_state: None,
                    reasoning_path_tracker: super::feedback::ReasoningPathTracker::new(),
                    // CACHE-POOL: a session reopened on restart that was itself a delegation's
                    // child stays one — it must not start receiving the parent's objective
                    // reminders just because it crossed a restart.
                    objective_tracking: tracks_own_objectives(session_node.spawned_by.is_some()),
                    objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
                    objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
                    work_log: work_log.clone(),
                    // Resumed sessions = interactive: use the generous cap (50/5min).
                    // T7 of plan 9a1684b2.
                    oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
                    provider_outage: Arc::new(std::sync::Mutex::new(
                        super::provider::outage::TurnHold::with_policies(
                            super::provider::resolver::CLAUDE_CODE,
                            super::provider::outage::OutagePolicy::from_env(),
                            super::provider::outage::HoldPolicy::from_env(),
                        ),
                    )),
                    session_grants: Arc::new(std::sync::Mutex::new(session_grants)),
                    oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
                    oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
                    oob_capped_warned: Arc::new(AtomicBool::new(false)),
                    cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_tools_cap: CANCEL_TOOLS_CAP,
                    cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
                    active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
                    tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
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
        self.spawn_nats_cancel_task_listener(
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

        // The inbound event: a person's message, or a peer's (typed, with its sender).
        let (inbound, inbound_type) = inbound_event(message, team);
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: inbound_type.to_string(),
            data: serde_json::to_string(&inbound).unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let _ = self.graph.store_chat_events(uuid, vec![user_event]).await;
        if team.is_none() {
            // The message that resumes the session is a user message: it counts
            // (`session_record`), as on the agent engine.
            if let Err(e) = super::session_record::count_user_message(&self.graph, uuid).await {
                warn!(session_id, error = %e, "Failed to count the resuming message (non-fatal)");
            }
        }

        // Emit the inbound event on local broadcast + NATS
        let _ = events_tx.send(inbound.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, inbound);
        }
        if team.is_none() {
            self.notify_attention(session_id, AttentionReason::UserMessage);
        }

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = super::relay::prefixed(relay.and_then(|r| r.text()), message);
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();
        let cognitive_routing = self.cognitive_routing.clone();

        let session_id_for_turn = session_id_str.clone();
        let turn_task = tokio::spawn(async move {
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
                cognitive_routing,
            )
            .await;
        });
        Self::track_stream_task(
            &self.active_sessions,
            &session_id_for_turn,
            turn_task.abort_handle(),
        )
        .await;

        Ok(())
    }

    /// What every resume checks before it reopens a session, a person's
    /// ([`Self::resume_session_from`]) as the restart recovery's (`recovery.rs`, which
    /// used to skip it: review of #697, finding 9): a session stored without a slug
    /// (all-projects mode) gets the project inferred from its cwd, as `create_session`
    /// does; its provider stays the one frozen at open (A16) — the legacy path only drives
    /// Claude Code, so a session bound to another provider cannot be resumed there — and its
    /// ENGINE is frozen too (M2): a session opened on the agent engine resumes there and a
    /// legacy one on the legacy engine, whatever `CHAT_PROVIDER_PATH` says now. Switching
    /// the default to the agent engine breaks no conversation, and neither does switching
    /// back to `legacy`.
    pub(crate) async fn ready_for_resume(&self, node: &mut ChatSessionNode) -> Result<()> {
        if node.project_slug.is_none() {
            node.project_slug = infer_session_project(
                self.graph.as_ref(),
                self.anchor_mode,
                Some(node.id),
                node.execution_place,
                &node.cwd,
            )
            .await;
            // ENV-WS: a project found here was deduced, not chosen.
            node.project_inferred = node.project_slug.is_some();
        }
        let frozen = super::provider::resolver::resolve_for_open(
            node.provider_id.as_deref(),
            true,
            None,
            &super::provider::resolver::BuiltinCatalog,
        )
        .map_err(anyhow::Error::new)?;
        if !node.opened_on_agent() && frozen.provider_id != super::provider::resolver::CLAUDE_CODE {
            return Err(anyhow::Error::new(
                super::provider::resolver::ResolveError::Unavailable {
                    provider_id: frozen.provider_id,
                    role: super::provider::resolver::Role::Pilot,
                },
            ));
        }
        // A session of the agent engine (`ChatSessionNode::opened_on_agent`) resumes there whatever
        // `CHAT_PROVIDER_PATH` says now: the engine is frozen at opening, like the provider.
        Ok(())
    }

    /// Retrieve full event history for a session (including tool_use, tool_result, etc.).
    ///
    /// Reads structured `ChatEventRecord` from Neo4j (the same data the WebSocket
    /// replay uses), so every event type is preserved — not just text.
    ///
    /// Falls back to the message memory for sessions that have no ChatEvent
    /// nodes yet (text only, no tool_use).
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

        // Fallback: sessions without persisted chat events — read from the message memory (text only)
        let session_node = self.graph.get_chat_session(uuid).await?.unwrap();
        if let Some(conversation_id) = session_node.conversation_id {
            let injector = match self.context_injector.as_ref() {
                Some(injector) => injector,
                None => {
                    return Ok(ChatEventPage {
                        events: vec![],
                        total_count: 0,
                        has_more: false,
                        offset: offset_val as usize,
                        limit: limit_val as usize,
                    });
                }
            };

            let options = GetMessagesOptions::new()
                .limit(limit_val as usize)
                .offset(offset_val as usize)
                .sort_by(MessageSort::CreatedAt)
                .newest_first(false);

            let paginated = injector
                .provider()
                .get_conversation_messages(&conversation_id, Some(options))
                .await
                .map_err(|e| anyhow!("Meilisearch query failed: {}", e))?;

            let meili_total = paginated.total_count;

            // Convert MessageDocument → ChatEventRecord (text-only approximation)
            let events: Vec<ChatEventRecord> = paginated
                .messages
                .into_iter()
                .enumerate()
                .map(|(i, msg)| {
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

            // The title rule of every session (`session_record`).
            if let Some((title, preview)) = first_user_msg
                .as_deref()
                .and_then(super::session_record::title_and_preview)
            {
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

    /// Brings the record of the sessions the agent engine served before it kept one
    /// (P8) to what it would hold today (`session_record`), from their persisted
    /// events: `message_count` = their `user_message` events (raised, never lowered),
    /// `total_cost_usd` from their `result` events when the record has none, and the
    /// title and preview of their first user message when they have none.
    /// Idempotent: a second run changes nothing. A session of the legacy engine (no
    /// provider snapshot) is left alone, that engine kept its record; a live one too.
    /// Returns the number of sessions updated.
    ///
    /// Run it AT REST, or on the instance that holds the live sessions: it skips the
    /// sessions live on THIS instance only. A session live on another instance
    /// (NATS) can still be counted twice until the cluster-wide follow-up (plan
    /// 5ad54c48-6317-4dc0-ad99-096242cc038d) lands.
    pub async fn backfill_agent_session_records(&self) -> Result<usize> {
        use super::session_record::{next_total_cost, title_and_preview, CostFigure};
        const PAGE: usize = 200;
        // Listed in full first: the list is ordered by `updated_at`, which each
        // write below changes (paging while writing would skip sessions).
        let mut sessions = Vec::new();
        let mut offset = 0;
        loop {
            let (page, _) = self
                .graph
                .list_chat_sessions(None, None, PAGE, offset, true)
                .await
                .context("Failed to list sessions")?;
            let fetched = page.len();
            offset += fetched;
            // The sessions the agent engine served (`opened_on_agent`: the engine written at
            // creation, else the snapshot), and any provider other than Claude Code (only the
            // agent engine serves one), in case neither was written.
            sessions.extend(
                page.into_iter()
                    .filter(|s| s.opened_on_agent() || !s.served_by_claude_code()),
            );
            if fetched < PAGE {
                break;
            }
        }
        let mut updated = 0;
        for session in sessions {
            // A live session keeps its own record: its handle may be between the
            // event and the write (`user_message` stored, count not yet bumped; a
            // `result` stored, cost not yet added), the backfill would count it twice.
            if self.agent_runtime.owns(&session.id.to_string()).await {
                continue;
            }
            let provider_id = session.provider_id.as_deref().unwrap_or_default();
            let events = self.graph.get_chat_events(session.id, -1, 100_000).await?;
            let parsed: Vec<ChatEvent> = events
                .iter()
                .filter(|e| e.event_type == "user_message" || e.event_type == "result")
                .filter_map(|e| serde_json::from_str(&e.data).ok())
                .collect();
            let users: Vec<&str> = parsed
                .iter()
                .filter_map(|e| match e {
                    ChatEvent::UserMessage { content } => Some(content.as_str()),
                    _ => None,
                })
                .collect();
            let message_count = i64::try_from(users.len())
                .ok()
                .filter(|n| *n > session.message_count);
            let total_cost_usd = if session.total_cost_usd.is_none() {
                // A Claude Code on another machine reports the session's total.
                let figure = match super::provider::store::instance(
                    self.graph.as_ref(),
                    provider_id,
                )
                .await
                {
                    Ok(Some(record))
                        if record.kind == super::provider::settings::KIND_CLAUDE_CODE_REMOTE =>
                    {
                        CostFigure::SessionTotal
                    }
                    // Claude Code forced onto the agent engine: the CLI's total too.
                    _ if provider_id == super::provider::resolver::CLAUDE_CODE => {
                        CostFigure::SessionTotal
                    }
                    _ => CostFigure::Turn,
                };
                parsed
                    .iter()
                    .filter_map(|e| match e {
                        ChatEvent::Result { cost_usd, .. } => Some(*cost_usd),
                        _ => None,
                    })
                    .fold(None, |total, usd| {
                        next_total_cost(total, usd, figure).or(total)
                    })
            } else {
                None
            };
            let (title, preview) = match (&session.title, users.first()) {
                (None, Some(first)) => match title_and_preview(first) {
                    Some((t, p)) => (Some(t), Some(p)),
                    None => (None, None),
                },
                _ => (None, None),
            };
            if message_count.is_none() && total_cost_usd.is_none() && title.is_none() {
                continue;
            }
            self.graph
                .update_chat_session(
                    session.id,
                    None,
                    title,
                    message_count,
                    total_cost_usd,
                    None,
                    preview,
                )
                .await?;
            updated += 1;
        }
        Ok(updated)
    }

    /// Search messages across all sessions via full-text search.
    ///
    /// Uses the message memory raw search (bypassing the SDK's `retrieve_context`
    /// which applies token_budget / max_context_items limits designed for LLM
    /// context injection, not UI search).
    pub async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<MessageSearchResult>> {
        use std::collections::HashMap as StdHashMap;

        let injector = match self.context_injector.as_ref() {
            Some(injector) => injector,
            None => return Ok(vec![]),
        };

        // Query the memory provider directly — no token budget, no max_context_items
        let options = MessageSearchOptions::new().limit(limit * 5); // Fetch extra to allow grouping by session
        let hits: Vec<nexus_claude::memory::MessageSearchHit> = injector
            .provider()
            .search_messages(query, &options)
            .await
            .map_err(|e| anyhow!("Meilisearch search failed: {}", e))?;

        if hits.is_empty() {
            return Ok(vec![]);
        }

        // Group results by conversation_id
        let mut by_conversation: StdHashMap<String, Vec<MessageSearchHit>> = StdHashMap::new();
        for hit in &hits {
            let doc = &hit.document;
            let score = hit.ranking_score;
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
        // An open mirror of a Claude Code sub-agent is followed live (P18c).
        if let Some(rx) = self.mirrors.subscribe(session_id) {
            return Ok(rx);
        }
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

    /// Whether the session is working: a turn running (or a mirrored turn) or a background
    /// task its provider still reports. An active session that does neither has answered
    /// and only waits for its next message or for the idle cleanup.
    pub async fn is_session_working(&self, session_id: &str) -> bool {
        if self.is_session_streaming(session_id).await {
            return true;
        }
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return !handle.active_background_tasks().await.is_empty();
        }
        let tasks = {
            let sessions = self.active_sessions.read().await;
            sessions.get(session_id).map(|s| {
                (
                    s.active_background_tasks.clone(),
                    s.cli_background_tasks.clone(),
                    s.child_pid,
                )
            })
        };
        match tasks {
            Some((background, cli_background, child_pid)) => {
                let cli_count = cli_background.load(Ordering::Relaxed);
                // A dead CLI (zombie or reaped) cannot have live background tasks; its last
                // reported count is stale. Clear it so the session frees its child slot
                // instead of holding it until the idle-cleanup timeout (ZOMBIE-CLI).
                if cli_count > 0 && process_alive(child_pid) == Some(false) {
                    cli_background.store(0, Ordering::Relaxed);
                    return !background.lock().await.is_empty();
                }
                !background.lock().await.is_empty() || cli_count > 0
            }
            None => false,
        }
    }

    /// Check if a session is currently streaming
    pub async fn is_session_streaming(&self, session_id: &str) -> bool {
        if self.mirrors.is_open(session_id) {
            return true;
        }
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

    /// Remember the task that runs the turn of `session_id`, so the interrupt
    /// watchdog can abort it. Called right after every `tokio::spawn` of
    /// `stream_response`; the drain's recursive call runs in the same task.
    pub(crate) async fn track_stream_task(
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
        session_id: &str,
        handle: tokio::task::AbortHandle,
    ) {
        let sessions = active_sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            *session
                .stream_task
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(handle);
        }
    }

    /// Arm [`Self::abandon_turn_if_stuck`] for `session_id`, in the background.
    pub(crate) fn spawn_interrupt_watchdog(&self, session_id: &str, grace: Duration) {
        let active_sessions = self.active_sessions.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        let session_id = session_id.to_string();
        tokio::spawn(async move {
            Self::abandon_turn_if_stuck(active_sessions, event_emitter, nats, session_id, grace)
                .await;
        });
    }

    /// The turn of `session_id` was told to stop (flag, token, control request).
    /// Wait up to `grace` for it to end on its own; when it does not, end it from
    /// here: the task is aborted, the status goes back to idle, the clients learn
    /// why. Returns whether the turn had to be abandoned.
    ///
    /// Why a turn can ignore its stop: the cancellation token is read by the
    /// stream loop's `select!` only. Once the loop is over (post-stream steps:
    /// persistence, memory) or while the loop body sits on an await of its own
    /// (a store call), nothing reads it. Observed on 2026-10-09 (session
    /// b4c28b9f): the CLI had answered, `result` was published, and the turn
    /// stayed "streaming" for two hours; two Stops were delivered to nothing.
    ///
    /// A turn that ends, or a new turn that starts (`turn_generation` moved),
    /// before the grace is over stands the watchdog down.
    ///
    /// Aborting the task drops what it held. Past the loop that is nothing
    /// the next turn needs. Inside the loop, the SDK control receiver taken for
    /// the turn goes with it: permission requests of that CLI are no longer
    /// answered until the session is resumed. A session stuck for good was the
    /// alternative.
    ///
    /// What the turn already wrote is safe: past the loop its events are written
    /// in a task of their own (`post_stream::spawn_persist_events`), which the
    /// abort does not reach. Inside the loop they are not written yet (the loop
    /// collects them for one batch at its end): an abort there loses them, and
    /// the `turn_abandoned` message says so.
    ///
    /// Messages held during the turn stay queued: the end of the next turn
    /// drains them, as it always does.
    pub(crate) async fn abandon_turn_if_stuck(
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        session_id: String,
        grace: Duration,
    ) -> bool {
        let (is_streaming, generation, turn) = {
            let sessions = active_sessions.read().await;
            let Some(session) = sessions.get(&session_id) else {
                return false;
            };
            (
                session.is_streaming.clone(),
                session.turn_generation.clone(),
                session.turn_generation.load(Ordering::SeqCst),
            )
        };
        if !is_streaming.load(Ordering::SeqCst) {
            return false;
        }
        // The runtime clock (not `std`): paused in tests, it moves with `sleep`.
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                break;
            }
            tokio::time::sleep(left.min(Duration::from_millis(100))).await;
            if !is_streaming.load(Ordering::SeqCst) || generation.load(Ordering::SeqCst) != turn {
                return false;
            }
        }

        let (task, events_tx, streaming_text, streaming_events) = {
            let sessions = active_sessions.read().await;
            let Some(session) = sessions.get(&session_id) else {
                return false;
            };
            if !session.is_streaming.load(Ordering::SeqCst)
                || session.turn_generation.load(Ordering::SeqCst) != turn
            {
                return false;
            }
            let task = session
                .stream_task
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            (
                task,
                session.events_tx.clone(),
                session.streaming_text.clone(),
                session.streaming_events.clone(),
            )
        };
        let task_aborted = task.map(|handle| handle.abort()).is_some();
        warn!(
            session_id = %session_id,
            grace_ms = grace.as_millis() as u64,
            task_aborted,
            "Stop ignored by the turn: abandoned by the watchdog, the session is idle again"
        );

        is_streaming.store(false, Ordering::SeqCst);
        streaming_text.lock().await.clear();
        streaming_events.lock().await.clear();
        for event in [
            ChatEvent::Error {
                message: format!(
                    "The turn was stopped, but it did not end within {}s and was abandoned. \
                     The conversation is free again; messages held during the turn are still queued \
                     and go out after the next one. If the turn was stopped before its answer was \
                     complete, part of it may be missing after a reload.",
                    grace.as_secs()
                ),
                parent_tool_use_id: None,
                code: Some("turn_abandoned".to_string()),
                reason: None,
                index: None,
                request_id: None,
            },
            ChatEvent::StreamingStatus {
                is_streaming: false,
            },
        ] {
            let _ = events_tx.send(event.clone());
            if let Some(ref nats) = nats {
                nats.publish_chat_event(&session_id, event);
            }
        }
        if let Some(ref emitter) = event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &session_id,
                serde_json::json!({ "is_streaming": false }),
                None,
            );
        }
        true
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
            // The native engine's Stop is a flag its event loop alone reads: a turn past
            // that loop (its end-of-turn work) or wedged on an await inside it never sees
            // it, and `is_streaming` would stay true for good — measured on 03/11/2026
            // (session 16c189ca): two hours of a composer that could not send, every Stop
            // delivered to nothing. The Claude Code engine arms `abandon_turn_if_stuck`
            // below for exactly that reason, with the same grace; the native one arms the
            // same verdict on its handle.
            handle.arm_abandon_watchdog(
                handle.turn_generation(),
                super::agent_runtime::INTERRUPT_GRACE_DEFAULT,
            );
            let diagnostic = outcome.diagnostic;
            return Ok(InterruptOutcome {
                delivered: outcome.turn_interrupted,
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

            // The token only reaches the stream loop. A turn past its loop —
            // in its post-stream steps, or wedged on an await inside the loop
            // body — ignores it, and `is_streaming` would stay true for good.
            // The watchdog gives it `INTERRUPT_GRACE` to end, then ends it.
            self.spawn_interrupt_watchdog(session_id, INTERRUPT_GRACE);
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
    /// If the session is not local, the request goes over NATS
    /// (`events.chat.{session_id}.cancel_tools`, request/reply) to the owning
    /// instance, which applies ITS cap, stops the tools and answers what it did
    /// or its typed refusal; no answer is a typed error (`cancel_relay`:
    /// `owner_unreachable`, `owner_timeout`, `session_gone`, `owner_protocol`),
    /// never a success. A session no instance holds is `owner_unreachable`, with or without NATS (one
    /// rule for every topology, review N2).
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
                // Session not local: the instance that holds it answers over NATS
                // (request/reply, its cap applies) with what it did, its refusal, or
                // the typed reason there is no answer (`cancel_relay`) — never a
                // success it did not report.
                if let Some(ref nats) = self.nats {
                    return super::cancel_relay::relay(
                        nats,
                        nats.cancel_tools_subject(session_id),
                        serde_json::json!({}),
                        super::cancel_relay::CancelKind::Tools,
                    )
                    .await;
                }
                // Single instance: no instance holds it live — the same typed answer
                // as with NATS and nobody answering (`owner_unreachable`).
                debug!(session_id = %session_id, "cancel_running_tools: no such live session");
                return Err(anyhow::Error::new(
                    super::cancel_relay::CancelRelayError::OwnerUnreachable,
                ));
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

        // Cross-instance fan-out of the ChatEvent (so remote clients of this
        // session see the cancel in their feed). NOT the cancel signal: this
        // instance holds the session and just did it; publishing it reached this
        // instance's own listener, which signalled the tools a second time and
        // counted the cap twice (review N8).
        if let Some(ref nats) = self.nats {
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
    ///    marks it only. `killed_pids` is empty in this edge case; the
    ///    claim, when it finds the pid of a task marked stopping, sends the
    ///    SIGINT itself (review of #691, finding 1).
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
    /// Cross-instance: if no session of this instance holds it, the request
    /// goes over NATS (`events.chat.{id}.cancel_task`, request/reply) to the
    /// instance that does — Claude Code (`spawn_nats_cancel_task_listener`) or
    /// agent engine (`spawn_agent_nats_listeners`) — and its answer is returned:
    /// what it stopped, `capped`, or its refusal (the `ProviderError` back in the
    /// `Err`). No answer at all is a typed error (`cancel_relay`: `owner_unreachable`
    /// when nobody subscribes, `owner_timeout`, `session_gone`, `owner_protocol`),
    /// never a success. A session no instance holds is `owner_unreachable`, with or without NATS.
    ///
    /// ## Agent engine
    ///
    /// `AgentSessionHandle::cancel_task`: the provider stops the task
    /// (`cancel_tools(task { id })`), a provider that cannot is a typed refusal
    /// (`Err` holding `ProviderError::Unsupported`, `error { code: cancel_refused }`
    /// on the wire), never a success.
    ///
    /// ## Returns
    ///
    /// - `CancelTaskResult { task_id, killed_pids, capped: false }` on
    ///   success — `killed_pids` is the list of PIDs (root + descendants)
    ///   that received SIGINT in V2. Empty in the claim-race fallback
    ///   path described above.
    /// - `capped: true` if the rate cap was hit (no broadcast, no map
    ///   mutation, no kill).
    /// - Returns `Ok(...)` when the session is held but the task_id is
    ///   not (idempotent — clicking Stop twice on the same task is fine);
    ///   a session no instance holds is `Err(owner_unreachable)`.
    pub async fn cancel_task(&self, session_id: &str, task_id: &str) -> Result<CancelTaskResult> {
        // The agent engine: the session handle stops the task through its provider
        // (`AgentSession::cancel_tools(task { id })`), under the same per-session cap,
        // and refuses, typed, what the provider cannot do.
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.cancel_task(task_id).await;
        }
        if let Some(result) =
            Self::cancel_legacy_task(&self.active_sessions, &self.nats, session_id, task_id).await
        {
            return Ok(result);
        }
        // Not held here: the instance that holds the session answers over NATS
        // (`spawn_nats_cancel_task_listener`, `spawn_agent_nats_listeners`) with what
        // it did, or why it could not.
        // No answer at all is a typed error (`cancel_relay`), never a success.
        if let Some(ref nats) = self.nats {
            return super::cancel_relay::relay(
                nats,
                nats.cancel_task_subject(session_id),
                serde_json::json!({ "task_id": task_id }),
                super::cancel_relay::CancelKind::Task,
            )
            .await;
        }
        // Single instance: no instance holds it live — the same typed answer as
        // with NATS and nobody answering (`owner_unreachable`).
        debug!(
            session_id = %session_id,
            task_id = %task_id,
            "cancel_task: no such live session"
        );
        Err(anyhow::Error::new(
            super::cancel_relay::CancelRelayError::OwnerUnreachable,
        ))
    }

    /// [`Self::cancel_task`] on a Claude Code session of THIS instance; `None` when
    /// it holds no such session. Callable without the manager (the NATS listener).
    async fn cancel_legacy_task(
        active_sessions: &RwLock<HashMap<String, ActiveSession>>,
        nats: &Option<Arc<crate::events::NatsEmitter>>,
        session_id: &str,
        task_id: &str,
    ) -> Option<CancelTaskResult> {
        let session_state = {
            let sessions = active_sessions.read().await;
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

        let (history, cap, window, events_tx, tasks_arc) = session_state?;

        let no_op = |capped: bool| CancelTaskResult {
            task_id: task_id.to_string(),
            killed_pids: Vec::new(),
            capped,
        };

        // One critical section on the task map for the whole decision: whether
        // the task was already signalled, the rate cap, the mark and the pid.
        //
        // - A task already SIGNALLED (by a stop that knew its pid, or by the PID
        //   claim for a stop that came first) is a no-op, not counted in the cap:
        //   a retry after a `session_gone` / `owner_timeout` from a real owner
        //   that did stop it, a second click, or a concurrent cancel waiting on
        //   this lock, never SIGINTs the same pid twice (reviews of #685,
        //   finding 3, and of #691, findings 1 and 2).
        // - A task marked stopping but NOT signalled (stopped before its pid was
        //   known, or marked by the death poller) is not a no-op: it is signalled
        //   now if its pid is known; otherwise the PID claim will do it.
        // - Marking and reading the pid together means a concurrent claim cannot
        //   populate the pid AFTER we read it (plan fc35b25e, T4).
        // Lock order: tasks, then the cap history (taken nowhere else first).
        let (snapshot, task_pid) = {
            let mut tasks = tasks_arc.lock().await;
            if tasks.get(task_id).is_some_and(|t| t.signalled) {
                debug!(
                    session_id = %session_id,
                    task_id = %task_id,
                    "cancel_task: already signalled; idempotent no-op"
                );
                return Some(no_op(false));
            }
            if !Self::check_and_record_cancel_cap(&history, cap, window).await {
                warn!(
                    session_id = %session_id,
                    task_id = %task_id,
                    cap = cap,
                    window_secs = window.as_secs(),
                    "cancel_task: rate cap hit, refusing"
                );
                return Some(no_op(true));
            }
            let pid = if let Some(entry) = tasks.get_mut(task_id) {
                entry
                    .pending_removal_at
                    .get_or_insert_with(std::time::Instant::now);
                let captured = entry.pid;
                entry.signalled = captured.is_some();
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
        // otherwise the claim signals it when it finds the pid (the task is
        // marked stopping), and this answer has no pid.
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
                    "cancel_task: no PID stored yet (claim pending: it signals the task when it finds the pid; or the subprocess crashed before discovery)"
                );
                Vec::new()
            }
        };

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(nats) = nats {
            nats.publish_chat_event(session_id, event);
        }

        Some(CancelTaskResult {
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
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.active_background_tasks().await;
        }
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
                // `biased`: a listener whose session was removed (its token
                // cancelled) stops before reading a request it would answer
                // `gone` (review of #673, finding 1).
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS cancel_tools listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(msg) = msg else { break; };
                        // A request (`cancel_running_tools` on another instance)
                        // gets an answer; a plain publish (the local path's
                        // fan-out) does not.
                        let reply_to = msg.reply;
                        let asker_v2 = super::cancel_relay::asker_v2(&msg.payload);

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
                            // The asker learns it at once, typed (`session_gone`).
                            if let Some(reply_to) = reply_to {
                                super::cancel_relay::publish(
                                    &nats,
                                    reply_to,
                                    asker_v2,
                                    &super::cancel_relay::CancelReply::<CancelToolsResult>::Gone,
                                )
                                .await;
                            }
                            break;
                        };

                        // Apply the cap on the SIGINT side too — protects
                        // against a remote instance flooding the NATS
                        // subject. Mirrors the cap applied by the
                        // local-path of `cancel_running_tools`.
                        let capped = !Self::check_and_record_cancel_cap(&history, cap, window).await;
                        let killed = if capped {
                            warn!(
                                session_id = %session_id,
                                cap = cap,
                                window_secs = window.as_secs(),
                                "NATS cancel_tools listener: rate cap hit, dropping signal"
                            );
                            Vec::new()
                        } else {
                            let killed = Self::kill_descendants(cli_pid);
                            info!(
                                session_id = %session_id,
                                cli_pid = ?cli_pid,
                                descendant_count = killed.len(),
                                "NATS cancel_tools received, SIGINT sent to descendants"
                            );
                            killed
                        };
                        if let Some(reply_to) = reply_to {
                            let result = CancelToolsResult {
                                cli_pid,
                                killed_pids: killed,
                                capped,
                            };
                            super::cancel_relay::publish(
                                &nats,
                                reply_to,
                                asker_v2,
                                &super::cancel_relay::CancelReply::Result(result),
                            )
                            .await;
                        }
                    }
                }
            }
        });
    }

    /// Spawn the per-session NATS responder for `cancel_task` asked on another
    /// instance (P12): request/reply on `events.chat.{id}.cancel_task`, answered
    /// with what [`Self::cancel_legacy_task`] did here (`cancel_relay::CancelReply`) —
    /// the cap applies on this side, as for a local call. One task per request,
    /// bounded by `OWNER_CANCEL_BOUND`. When the session has left: `gone`, then the
    /// listener ends.
    fn spawn_nats_cancel_task_listener(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        cancel: CancellationToken,
    ) {
        let Some(nats) = self.nats.clone() else {
            return;
        };
        let session_id = session_id.to_string();
        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_cancel_task(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(session_id = %session_id, "Failed to subscribe to NATS cancel_task: {}", e);
                    return;
                }
            };
            let nats_opt = Some(Arc::clone(&nats));
            loop {
                // `biased`: see `spawn_nats_cancel_tools_listener`.
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    msg = subscriber.next() => {
                        let Some(msg) = msg else { break };
                        let Some(reply_to) = msg.reply else { continue };
                        let task_id = super::cancel_relay::task_id_of(&msg.payload);
                        let asker_v2 = super::cancel_relay::asker_v2(&msg.payload);
                        // The session left this instance (a removal path that did not
                        // end this listener): `gone`, which the asker only keeps if no
                        // instance holding the session answers (`GONE_GRACE`).
                        if !active_sessions.read().await.contains_key(&session_id) {
                            super::cancel_relay::publish(
                                &nats,
                                reply_to,
                                asker_v2,
                                &super::cancel_relay::CancelReply::<CancelTaskResult>::Gone,
                            )
                            .await;
                            break;
                        }
                        // One task per request, bounded: a slow cancel never holds
                        // the next one, and the asker always gets an answer.
                        let (nats, nats_opt, sessions, sid) = (
                            Arc::clone(&nats),
                            nats_opt.clone(),
                            Arc::clone(&active_sessions),
                            session_id.clone(),
                        );
                        tokio::spawn(async move {
                            super::cancel_relay::answer(
                                &nats,
                                reply_to,
                                asker_v2,
                                super::cancel_relay::CancelKind::Task,
                                async {
                                    Self::cancel_legacy_task(&sessions, &nats_opt, &sid, &task_id)
                                        .await
                                        .ok_or_else(|| {
                                            anyhow::Error::new(
                                                super::cancel_relay::CancelRelayError::SessionGone {
                                                    kind: super::cancel_relay::CancelKind::Task,
                                                },
                                            )
                                        })
                                },
                            )
                            .await;
                        });
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
            // kill(0, _) signals the caller's OWN process group: a root pid of 0 (never a
            // real CLI) would SIGINT the whole test run (ZOMBIE-CLI, Unit Tests on #1002).
            if root_pid == 0 {
                return Vec::new();
            }
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

    /// The server's warning when a Claude Code session on the agent engine lacks
    /// something (`degraded`: the full list its `system_init` announces). Silent when it
    /// lacks nothing: the engine ports hooks, queue, auto-continue, retry, compaction and
    /// NATS, so a local Claude Code loses none of them ([`forced_engine_warning`]).
    fn warn_if_forced(&self, provider_id: &str, degraded: &[String]) {
        if let Some(message) = forced_engine_warning(provider_id, degraded) {
            warn!(features = ?degraded, "{message}");
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

    /// A vault is wired and locked now.
    fn vault_locked(&self) -> bool {
        self.vault
            .as_ref()
            .is_some_and(|v| !v.is_unlocked(chrono::Utc::now()))
    }

    /// The vault was just unlocked (vault page, chat card, secret request card). What
    /// Auto measured while it was locked no longer holds: the health of the instances
    /// is checked again at the next opening, and an Anthropic catalog the locked vault
    /// left offline refreshes now (bounded), so the message sent again right after the
    /// unlock is routed with the vault, not around it (review of #689, finding 1).
    pub async fn vault_unlocked(&self) {
        let Some(routing) = self.cognitive_routing.as_ref() else {
            return;
        };
        routing.health.clear();
        if let Some(catalog) = routing.claude_code_catalog.as_ref() {
            catalog.vault_unlocked().await;
        }
    }

    /// The models ticked for the conversation being opened, `None` when nothing restricts it.
    fn routing_pool_of(request: &ChatRequest) -> Option<&[super::types::RoutingPoolEntry]> {
        request.routing_pool.as_deref().filter(|p| !p.is_empty())
    }

    /// The pair (provider, model) of the session that spawned `request`: the explicit
    /// fallback of a routed sub-agent (AUTO-SUB, review point 1). A routed child carries
    /// neither a pool nor a pair of its own — naming one would make its request "named" and
    /// the router would never run — so when the router applies nothing, this is what opens
    /// it instead of the declared rules (project rule, global rule, configured default). A
    /// DeepSeek Auto parent would otherwise see its `Agent` child open on `claude-code`
    /// with no visible reason. The parent's id rides the `spawned_by` the child already
    /// carries; an unknown parent (a pruned node, a test) is no fallback.
    ///
    /// The pair is NOT put back into the request's `provider`/`model`: it is used at the
    /// `request` level of the resolution, only once the router had its say.
    async fn parent_pair_of(
        &self,
        request: &ChatRequest,
    ) -> Option<super::provider::resolver::Candidate> {
        use super::provider::resolver;
        let parent_id = request
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)?
            .parent_session_id?;
        let parent = self
            .graph
            .get_chat_session(Uuid::parse_str(&parent_id).ok()?)
            .await
            .ok()??;
        // A node written before the harness names no provider: it is Claude Code (M0/A4).
        let provider = parent
            .provider_id
            .clone()
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| resolver::CLAUDE_CODE.to_owned());
        let model = (!parent.model.is_empty()).then(|| parent.model.clone());
        Some(resolver::Candidate::new(provider, model))
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
        // The project's consents, or outside any project those given for the sessions
        // without one (B3).
        let consents = store::consents_for(self.graph.as_ref(), project_slug).await?;
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
        // Without its CLI on the server, the built-in instance is not a default (B2): a chat
        // that names nothing goes to the only other usable instance, never to a 424. One
        // that names claude-code still gets the typed 424 `cli_not_found` (install action).
        let store_catalog = catalog::StoreCatalog::new(instances, &consents)
            .with_claude_cli(self.claude_code_usable().await);
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
        let routing_store = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.store.clone());
        // Read ONCE, before the pool is built: the refusal below and the health the pool
        // caches follow the same state of the vault (review of #689, finding 1).
        let vault_locked = self.vault_locked();
        let mut cognitive = match (&self.cognitive_routing, routing_instances) {
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
                        vault_locked,
                    )
                    .await
                {
                    Ok(decision) => decision,
                    // Auto refused on a locked vault: the opening fails, it does not fall
                    // back to the declared rules.
                    Err(error)
                        if error
                            .downcast_ref::<super::provider::errors::AutoVaultLocked>()
                            .is_some() =>
                    {
                        return Err(error);
                    }
                    Err(error) => {
                        warn!(%error, "cognitive routing skipped: the declared rules apply");
                        None
                    }
                }
            }
            _ => None,
        };
        let mut auto_pick = cognitive
            .as_ref()
            .filter(|d| d.applied)
            .and_then(|d| d.chosen.clone());
        // A pool of two models or more (R-S4): what opens the conversation when the decision
        // among the pool does not — the menu's pilot (its provider AND its model), else the
        // first ticked model, never a default the user did not tick.
        let pool_fallback = request.routes_among_pool().then(|| {
            match (
                input.request.as_ref().map(|c| c.provider_id.clone()),
                request.model.as_deref().filter(|m| !m.is_empty()),
            ) {
                (Some(provider), Some(model)) => {
                    Some(resolver::Candidate::new(provider, Some(model.to_owned())))
                }
                (Some(_), None) => input.request.clone(),
                (None, _) => Self::routing_pool_of(request)
                    .and_then(<[_]>::first)
                    .map(|first| {
                        resolver::Candidate::new(first.provider.clone(), Some(first.model.clone()))
                    }),
            }
        });
        // AUTO-SUB (review point 1): the pair of the session that spawned this child. Read
        // once, applied below ONLY when the router applies nothing — the declared rules
        // must never substitute a default for a child whose parent named its engine.
        // Never over an engine the request itself names (DELEG-500): a delegation that
        // names one keeps it, and its refusal stays the request's own. That includes the
        // run level, where `inherit_parent_engine` already puts the parent's pair (and the
        // task and persona levels): the fallback is for an opening that names NOTHING.
        let names_an_engine = input.request.is_some()
            || input.task.is_some()
            || input.persona.is_some()
            || input.run.is_some();
        let parent_fallback = if falls_back_on_parent(request) && !names_an_engine {
            self.parent_pair_of(request).await
        } else {
            None
        };
        let declared_project_rule = input.project_rule.clone();
        if let Some(pick) = &auto_pick {
            input.project_rule = Some(resolver::Candidate::new(
                pick.provider_id.clone(),
                Some(pick.model.clone()),
            ));
            // One applied among the pool: the decision opens the conversation, the pilot
            // only stands behind it (`pool_fallback`).
            if pool_fallback.is_some() {
                input.request = None;
            }
        } else if let Some(fallback) = &pool_fallback {
            input.request = fallback.clone();
        } else if let Some(fallback) = &parent_fallback {
            // AUTO-SUB: nothing applied and no pool stands behind the opening — the engine
            // of the session this child came from opens it, at the explicit `request`
            // level, so no declared rule can substitute a configured default.
            input.request = Some(fallback.clone());
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
        let mut choice = match resolver::resolve(&input, &store_catalog) {
            Ok(choice) => choice,
            // The pick among the pool was applied but the resolver refuses it (its view of
            // the instance — health, consent — differs from the decider's): a pilot never
            // falls back inside the resolver, so the pool's own fallback opens the
            // conversation instead of the opening failing (review of #713, C2). The decision
            // is stored back as not applied, with why.
            Err(refused)
                if auto_pick.is_some()
                    && (pool_fallback.is_some() || parent_fallback.is_some()) =>
            {
                // Rewritten BEFORE the fallback is resolved: the pick was not applied, whether
                // the fallback then opens the conversation or is refused too (nothing opens).
                if let Some(decision) = cognitive.as_mut() {
                    decision.applied = false;
                    decision.reason =
                        format!("{}; not_opened: {}", decision.reason, refused.code());
                    if let Some(store) = routing_store.as_ref() {
                        if let Err(error) = store.put_decision(decision).await {
                            warn!(decision_id = %decision.id, %error, "the refused opening was not recorded on its decision");
                        }
                    }
                }
                input.project_rule = declared_project_rule;
                // What stands behind the refused pick opens the conversation: the pool's
                // pilot, else the parent's pair (AUTO-SUB).
                input.request = pool_fallback.clone().flatten().or(parent_fallback.clone());
                auto_pick = None;
                resolver::resolve(&input, &store_catalog)
                    .map_err(|e| anyhow::Error::new(e.in_scope(project_slug)))?
            }
            Err(refused) => return Err(anyhow::Error::new(refused.in_scope(project_slug))),
        };
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
    /// stored. Empty when the cognitive router is not wired. The native models whose
    /// tools are unknown are probed (bounded) unless the settings say `primary` +
    /// `shadow`, where routing spends no request.
    pub(crate) async fn routing_pool_for(
        &self,
        project_slug: Option<&str>,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::cognitive::decider::ProbeScope;
        if self.cognitive_routing.is_none() {
            return Vec::new();
        }
        let scope =
            match super::provider::cognitive::load_routing(self.graph.as_ref(), project_slug).await
            {
                Ok((settings, _)) => ProbeScope::for_settings(settings.mode, settings.stage),
                Err(_) => ProbeScope::None,
            };
        self.routing_pool_scoped(project_slug, &scope).await
    }

    /// [`Self::routing_pool_for`] without any probe: what is already known (a report
    /// must not spend requests).
    pub(crate) async fn routing_pool_known(
        &self,
        project_slug: Option<&str>,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::cognitive::decider::ProbeScope;
        self.routing_pool_scoped(project_slug, &ProbeScope::None)
            .await
    }

    async fn routing_pool_scoped(
        &self,
        project_slug: Option<&str>,
        scope: &super::provider::cognitive::decider::ProbeScope,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::{catalog, store};
        let Some(routing) = self.cognitive_routing.as_ref() else {
            return Vec::new();
        };
        let Ok(instances) = store::instances(self.graph.as_ref()).await else {
            return Vec::new();
        };
        let consents = store::consents_for(self.graph.as_ref(), project_slug)
            .await
            .unwrap_or_default();
        let store_catalog = catalog::StoreCatalog::new(instances.clone(), &consents);
        self.routing_pool(routing, &instances, &store_catalog, scope)
            .await
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

    /// Starts, each in its own task, the probes of `reserved` on `native`. Each task
    /// records its own outcome: a caller that stops waiting (a turn's timeout, the pool's
    /// short wait) loses neither a request already sent nor the model's mark.
    fn spawn_probes(
        routing: &super::provider::cognitive::decider::CognitiveRouting,
        native: &Arc<nexus_claude::providers::native::NativeProvider>,
        instance: &str,
        reserved: Vec<super::provider::cognitive::decider::Reservation>,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        use super::provider::cognitive::decider::{record_probe, ProbeOutcome};
        let policy = routing.probe_policy;
        reserved
            .into_iter()
            .map(|reservation| {
                let (native, log, instance) = (
                    Arc::clone(native),
                    Arc::clone(&routing.probed),
                    instance.to_owned(),
                );
                tokio::spawn(async move {
                    let model = reservation.model;
                    let outcome = match tokio::time::timeout(
                        policy.timeout,
                        native.refresh_capabilities(&model),
                    )
                    .await
                    {
                        Ok(Ok(_)) => ProbeOutcome::Concluded,
                        Ok(Err(error)) => ProbeOutcome::of_error(&error),
                        Err(_) => ProbeOutcome::Failed,
                    };
                    if outcome != ProbeOutcome::Concluded {
                        debug!(%instance, %model, ?outcome, "routing capability probe did not conclude");
                    }
                    record_probe(
                        &mut log.lock().unwrap_or_else(|e| e.into_inner()),
                        &instance,
                        &model,
                        reservation.ticket,
                        outcome,
                        std::time::Instant::now(),
                    );
                })
            })
            .collect()
    }

    /// Probes in the background, without anyone waiting, every model of `instance` in
    /// `models` whose tools are unknown, within the instance's budget: as seats free up,
    /// the next models are reserved, until none is left to probe now. Used when a
    /// conversation that routes (Auto, ticked models) opens: its per-turn pools read the
    /// stored settings, which may forbid probing (review of #709, C4a).
    async fn probe_rest_in_background(&self, instance: &str, models: Vec<String>) {
        use super::provider::cognitive::decider::{probing, reserve_probes};
        let Some(routing) = self.cognitive_routing.clone() else {
            return;
        };
        let Some(native) = self.native_probers.read().await.get(instance).cloned() else {
            return;
        };
        let instance = instance.to_owned();
        tokio::spawn(async move {
            use nexus_claude::agent::AgentProvider;
            // Bounded: each round either starts a probe or waits for one to end.
            for _ in 0..(models.len() * 4 + 8) {
                let now = std::time::Instant::now();
                let wanted: Vec<String> = models
                    .iter()
                    .filter(|m| !native.capabilities(Some(m)).tools)
                    .cloned()
                    .collect();
                let reserved = reserve_probes(
                    &mut routing.probed.lock().unwrap_or_else(|e| e.into_inner()),
                    &instance,
                    wanted,
                    now,
                    &routing.probe_policy,
                );
                if reserved.is_empty() {
                    let busy = probing(
                        &routing.probed.lock().unwrap_or_else(|e| e.into_inner()),
                        &instance,
                        now,
                        &routing.probe_policy,
                    );
                    if !busy {
                        return;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    continue;
                }
                futures::future::join_all(Self::spawn_probes(
                    &routing, &native, &instance, reserved,
                ))
                .await;
            }
        });
    }

    /// The pool the cognitive router chooses from: every model of every
    /// reachable instance, with the capabilities and the price nexus reports,
    /// the health (one probe per instance per window) and the project's consent.
    /// The native models `scope` covers whose tools are not known yet are probed,
    /// within `routing.probe_policy`: every instance's probes start before any is
    /// waited for, and the build waits `wait` at most for all of them together; what
    /// ends later serves the next builds.
    async fn routing_pool(
        &self,
        routing: &super::provider::cognitive::decider::CognitiveRouting,
        instances: &[super::provider::settings::InstanceRecord],
        store_catalog: &super::provider::catalog::StoreCatalog,
        scope: &super::provider::cognitive::decider::ProbeScope,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::cognitive::candidates::ModelFacts;
        use super::provider::cognitive::decider::reserve_probes;
        use super::provider::resolver::{self, InstanceCatalog};
        use std::time::Instant;

        struct Gathered {
            id: String,
            provider: Arc<dyn nexus_claude::agent::AgentProvider>,
            healthy: bool,
            allowed: bool,
            entries: Vec<(String, Option<nexus_claude::agent::ModelPrice>)>,
        }

        let mut ids = vec![resolver::CLAUDE_CODE.to_string()];
        ids.extend(
            instances
                .iter()
                .filter(|i| !resolver::is_remote_instance(&i.id))
                .map(|i| i.id.clone()),
        );
        let mut providers = Vec::new();
        for id in ids {
            // The legacy engine can only drive Claude Code.
            if !self.engine_is_agent(&id) && id != resolver::CLAUDE_CODE {
                continue;
            }
            let Ok(provider) = self.provider_for(&id).await else {
                continue;
            };
            providers.push((id, provider));
        }
        // Health of every instance the cache does not know, all at once and each
        // bounded: one slow endpoint or agent process must not hold the opening
        // of the conversation (an unbounded, sequential check here returned HTTP
        // 408 on an Auto opening). Stored with the time it ENDED, so a slow
        // answer is not already stale when cached. Never stored for an instance
        // keyed in the vault while the vault is locked: that "unavailable" ends
        // with the unlock, and a cached one would route around the vault for a
        // minute after it (review of #689, finding 1).
        let vault_locked = self.vault_locked();
        let vault_keyed: std::collections::HashSet<&str> = instances
            .iter()
            .filter(|i| keyed_in_vault(i))
            .map(|i| i.id.as_str())
            .collect();
        let unknown: Vec<_> = providers
            .iter()
            .filter(|(id, _)| routing.health.get(id, Instant::now()).is_none())
            .map(|(id, provider)| {
                let cache = !(vault_locked && vault_keyed.contains(id.as_str()));
                async move {
                    let ok = bounded_health(provider.as_ref(), ROUTING_HEALTH_TIMEOUT)
                        .await
                        .status
                        != nexus_claude::agent::HealthStatus::Unavailable;
                    if cache {
                        routing.health.put(id, ok, Instant::now());
                    }
                    (id.clone(), ok)
                }
            })
            .collect();
        let measured: HashMap<String, bool> = futures::future::join_all(unknown)
            .await
            .into_iter()
            .collect();
        let healthy_of = |id: &str| match measured.get(id) {
            Some(ok) => *ok,
            None => routing.health.get(id, Instant::now()).unwrap_or(false),
        };
        // The catalogs of the healthy instances, all at once and each bounded: a
        // sequential wait here added `ROUTING_CATALOG_TIMEOUT` per instance (review
        // of #689, finding 4).
        let mut catalogs: HashMap<String, Vec<nexus_claude::agent::ModelInfo>> =
            futures::future::join_all(providers.iter().filter(|(id, _)| healthy_of(id)).map(
                |(id, provider)| async move {
                    let models = tokio::time::timeout(ROUTING_CATALOG_TIMEOUT, provider.catalog())
                        .await
                        .ok()
                        .and_then(Result::ok)
                        .unwrap_or_default();
                    (id.clone(), models)
                },
            ))
            .await
            .into_iter()
            .collect();
        let mut gathered = Vec::new();
        let mut probes = Vec::new();
        for (id, provider) in providers {
            let now = Instant::now();
            let healthy = healthy_of(&id);
            let default_model = instances
                .iter()
                .find(|i| i.id == id)
                .and_then(|i| i.default_model.clone());
            let models = catalogs.remove(&id).unwrap_or_default();
            let allowed = store_catalog.is_allowed_for_project(&id);
            // A model the catalogue itself lists without tools is not worth a probe.
            let no_tools_listed: std::collections::HashSet<String> = models
                .iter()
                .filter(|m| m.supports_tools == Some(false))
                .map(|m| m.id.clone())
                .collect();
            // Answered by the catalogue: no path (this pool, a routed conversation's
            // background probes) may reserve them (review of #709, N2).
            if id != resolver::CLAUDE_CODE && !no_tools_listed.is_empty() {
                super::provider::cognitive::decider::mark_listed_without_tools(
                    &mut routing.probed.lock().unwrap_or_else(|e| e.into_inner()),
                    &id,
                    &no_tools_listed,
                );
            }
            let mut entries: Vec<(String, Option<nexus_claude::agent::ModelPrice>)> =
                models.into_iter().map(|m| (m.id, m.pricing)).collect();
            if let Some(model) = &default_model {
                if !entries.iter().any(|(m, _)| m == model) {
                    entries.push((model.clone(), None));
                }
            }
            // A native model nobody probed reports no tools and no window, so the hard
            // constraints would drop it for good: probe the instance's models the scope
            // covers, the default first, then the catalogue's order. Bounded: a model
            // the catalogue says has no tools is never probed, an answer (tools,
            // `ModelNoTools`, a refusal of the model) is never sent again, a failure
            // backs off, and an instance never has more than `per_instance` running.
            if id != resolver::CLAUDE_CODE && healthy {
                let wanted: Vec<String> = default_model
                    .iter()
                    .chain(entries.iter().map(|(m, _)| m))
                    .filter(|model| scope.covers(&id, model))
                    .filter(|model| !no_tools_listed.contains(*model))
                    .filter(|model| !provider.capabilities(Some(model)).tools)
                    .cloned()
                    .collect();
                let concrete = self.native_probers.read().await.get(&id).cloned();
                if let (false, Some(native)) = (wanted.is_empty(), concrete) {
                    let reserved = reserve_probes(
                        &mut routing.probed.lock().unwrap_or_else(|e| e.into_inner()),
                        &id,
                        wanted,
                        now,
                        &routing.probe_policy,
                    );
                    probes.extend(Self::spawn_probes(routing, &native, &id, reserved));
                }
            }
            gathered.push(Gathered {
                id,
                provider,
                healthy,
                allowed,
                entries,
            });
        }
        // Every instance's probes are running: wait a little for all of them together,
        // never for the slowest instance after another (review of #709, C1).
        if !probes.is_empty() {
            let _ =
                tokio::time::timeout(routing.probe_policy.wait, futures::future::join_all(probes))
                    .await;
        }
        let mut pool = Vec::new();
        for Gathered {
            id,
            provider,
            healthy,
            allowed,
            entries,
        } in gathered
        {
            let mut seen = std::collections::HashSet::new();
            // Claude Code declares no catalog and no window of its own: its candidates
            // are the live Anthropic catalog's models, each with the window the Models
            // API reports. Offline, they stay in the pool with the reason they cannot
            // be chosen ("window unknown: catalog offline"), never a guessed window.
            if id == resolver::CLAUDE_CODE {
                if let Some(catalog) = routing.claude_code_catalog.as_ref() {
                    let snapshot = catalog.snapshot().await;
                    let prices: std::collections::HashMap<_, _> = entries.iter().cloned().collect();
                    for facts in super::provider::cognitive::candidates::claude_code_facts(
                        &id,
                        &snapshot,
                        |model| provider.capabilities(Some(model)),
                        |model| prices.get(model).copied().flatten(),
                        Some(healthy),
                        allowed,
                    ) {
                        seen.insert(facts.model.clone());
                        pool.push(facts);
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
        // QUOTA: an account that refuses until a known instant is not unhealthy, it is exhausted, and
        // the filter says so with the instant it is back (a reject reason of its own).
        super::provider::cognitive::candidates::apply_quota(&mut pool, &routing.quota);
        pool
    }

    /// Asks the cognitive router about the session being opened. The decision is
    /// stored by the decider whether it is applied or not.
    /// Auto on a locked vault (`vault_locked`, read by the caller before anything is
    /// measured) is refused with [`super::provider::errors::AutoVaultLocked`] instead.
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
        vault_locked: bool,
    ) -> Result<Option<super::provider::cognitive::decision::CognitiveDecision>> {
        use super::provider::cognitive::{
            candidates::Slot,
            decider::ProbeScope,
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
            super::provider::resolver::Role::Executor => {
                if is_subagent_spawn(request) {
                    TaskSignature::from_subagent(
                        request.subagent_type.as_deref(),
                        &crate::refs::turn::visible_text(&request.message),
                        project_slug,
                        ContextHints::default(),
                    )
                } else {
                    TaskSignature::from_delegation(
                        request.task_class.as_deref(),
                        None,
                        1,
                        project_slug,
                        ContextHints::default(),
                    )
                }
            }
        };
        routing.set_hints(PriorHints::from_aliases(aliases));
        // A provider, model, alias or run the caller named is never substituted. The
        // provider and model the menu names WITH a pool of two models or more are its pilot,
        // not a choice: the opening is decided among the pool (decision R-S4), the pilot is
        // only the fallback when that decision does not apply.
        let pilot_only = request.routes_among_pool();
        let pilot_named =
            request.provider.is_some() || request.model.as_deref().is_some_and(|m| !m.is_empty());
        let named = (!pilot_only && pilot_named)
            || request.task_alias.is_some()
            || request.persona_alias.is_some()
            || request.run_provider.is_some();
        // The conversation's own mode replaces the settings' (the chat menu: Auto = full).
        let mut settings = settings;
        // Auto, or two models ticked or more: the user's choice for THIS conversation is
        // decided at the `auto` stage, whatever the settings' stage (decision R-S1).
        settings.stage = super::provider::cognitive::conversation_stage(
            settings.stage,
            super::provider::cognitive::conversation_routes(
                request.routing_mode,
                Self::routing_pool_of(request).map_or(0, <[_]>::len),
            ),
        );
        if let Some(mode) = request.routing_mode {
            settings.mode = mode.pilot_mode();
        }
        let ticked = Self::routing_pool_of(request);
        // The models ticked in the menu are routed like `full` inside them, whatever the
        // settings' mode.
        if ticked.is_some() {
            settings.mode = super::provider::cognitive::ProviderRoutingMode::Full;
        }
        // AUTO-SUB: a sub-agent PO routes (its parent was in Auto) is decided over the full
        // pool at the `auto` stage, whatever the settings — a spawned session normally
        // cannot hand its routing over, but this is PO's own choice, not the child's.
        if request.route_auto {
            settings.mode = super::provider::cognitive::ProviderRoutingMode::Full;
            settings.stage = super::provider::cognitive::LearningStage::Auto;
        }
        debug!(
            route_auto = request.route_auto,
            subagent = is_subagent_spawn(request),
            named = named,
            mode = ?settings.mode,
            stage = ?settings.stage,
            "cognitive decision settings"
        );
        // What the probes may cost: nothing for a named choice (the session opening
        // probes its own model) or in `primary` + `shadow`; only the ticked models when
        // there are some; every model otherwise.
        let scope = match ticked {
            _ if named => ProbeScope::None,
            Some(ticked) => ProbeScope::Only(
                ticked
                    .iter()
                    .map(|e| (e.provider.clone(), e.model.clone()))
                    .collect(),
            ),
            None => ProbeScope::for_settings(settings.mode, settings.stage),
        };
        let mut pool = self
            .routing_pool(routing, instances, store_catalog, &scope)
            .await;
        // The models ticked in the menu are the only candidates.
        if let Some(ticked) = ticked {
            pool.retain(|facts| {
                ticked
                    .iter()
                    .any(|e| e.provider == facts.provider_id && e.model == facts.model)
            });
        }
        // AUTO-L: an Auto pilot with no models ticked chooses among its person's list (with
        // the fallback to every model, stated in the reason, when nothing listed can serve).
        let person_code = if !named
            && ticked.is_none()
            && (matches!(role, super::provider::resolver::Role::Pilot)
                || is_subagent_spawn(request))
            && settings.mode == super::provider::cognitive::ProviderRoutingMode::Full
        {
            match self
                .session_owner(request.user_claims.as_ref())
                .await
                .ok()
                .flatten()
            {
                Some(person) => {
                    match super::provider::cognitive::auto_models::load(
                        self.graph.as_ref(),
                        &person,
                    )
                    .await
                    {
                        Ok(Some(list)) => Some(super::provider::cognitive::auto_models::restrict(
                            &mut pool, &list, &signature,
                        )),
                        Ok(None) => None,
                        Err(error) => {
                            warn!(%error, "auto_models unreadable: every model applies");
                            None
                        }
                    }
                }
                None => None,
            }
        } else {
            None
        };
        let mut decide = DecideRequest::new(signature, settings, pool);
        decide.narrowed.extend(person_code);
        decide.slot = if named {
            Slot::Explicit
        } else {
            Slot::Automatic
        };
        decide.trust = request.permission_mode.as_deref() == Some("bypassPermissions");
        decide.session_id = session_id;
        let routes = !named
            && super::provider::cognitive::conversation_routes(
                request.routing_mode,
                ticked.map_or(0, <[_]>::len),
            );
        let unknown: Vec<(String, String)> = decide
            .pool
            .iter()
            .filter(|facts| !facts.supports_tools)
            .map(|facts| (facts.provider_id.clone(), facts.model.clone()))
            .collect();
        // Auto with the vault locked: the candidates that depend on it came back
        // unavailable (or "catalog offline"), so the router would pick among what is
        // left — a silent fallback, and in practice a slow local agent whose opening
        // ran past the HTTP timeout (408). Refuse as for a named instance, BEFORE the
        // decision is stored: nothing is persisted that no session will ever close
        // (review of #689, nit 6). The offer is what the router would have chosen
        // without the vault, previewed, not decided.
        //
        // AUTO-SUB (review point 2): a request `route_auto` is not a user turning Auto on
        // for a conversation — it is PO's own choice for an `Agent` call of a parent that
        // is ALREADY open and holds a working pair. The fallback here is not silent, it is
        // that pair (`parent_pair_of`), so the refusal buys nothing: it would only make the
        // tool call fail on a vault that re-locked while the parent idled.
        // DELEG-500: a delegation sent by a conversation that is already open has the same
        // stated fallback (`falls_back_on_parent`): it is not refused either.
        if vault_locked && !named && !falls_back_on_parent(request) {
            let waits = match routing.claude_code_catalog.as_ref() {
                Some(catalog) => catalog.waits_on_vault().await,
                None => false,
            };
            let vault_backed = vault_backed_candidates(
                instances,
                ticked,
                |id| {
                    super::provider::resolver::InstanceCatalog::is_allowed_for_project(
                        store_catalog,
                        id,
                    )
                },
                waits,
            );
            if !vault_backed.is_empty() {
                let preview = routing.decider.preview(&decide).await?;
                if auto_refused_by_locked_vault(
                    preview.mode,
                    preview.applied,
                    named,
                    vault_locked,
                    &vault_backed,
                ) {
                    info!(
                        candidates = ?vault_backed,
                        "Auto routing refused: the vault is locked and candidates depend on it"
                    );
                    return Err(anyhow::Error::new(
                        super::provider::errors::AutoVaultLocked {
                            fallbacks: locked_vault_fallbacks(&preview, &vault_backed),
                        },
                    ));
                }
            }
        }
        let decision = routing.decider.decide(&decide).await?;
        debug!(
            chosen = ?decision.chosen,
            applied = decision.applied,
            role = ?decide.signature.role,
            pool_len = decide.pool.len(),
            pool = ?decide
                .pool
                .iter()
                .take(MAX_LOGGED_POOL)
                .map(|f| f.provider_id.as_str())
                .collect::<Vec<_>>(),
            "cognitive decision"
        );
        // A conversation that routes (Auto, models ticked) is decided turn by turn from
        // pools that follow the stored settings, which may forbid probing: the candidates
        // of its provider not known yet are probed now, in the background, within the
        // instance's budget (review of #709, C4a).
        if routes {
            if let Some(chosen) = decision.chosen.as_ref() {
                let rest: Vec<String> = unknown
                    .into_iter()
                    .filter(|(provider, _)| *provider == chosen.provider_id)
                    .map(|(_, model)| model)
                    .collect();
                if !rest.is_empty() {
                    self.probe_rest_in_background(&chosen.provider_id, rest)
                        .await;
                }
            }
        }
        Ok(Some(decision))
    }

    /// The context window (tokens) of `model` on `provider_id` as the model catalog
    /// knows it, from the same sources as routing and compaction
    /// (`routing_pool_scoped`, A1: nexus is the source of truth): for Claude Code, which
    /// declares no window of its own, the live Anthropic catalog's `max_input_tokens`
    /// first; then what nexus reports for the model (`capabilities(model)`: configured,
    /// probed, or from its catalog). `None` when nobody knows it — never guessed from
    /// the model's name.
    pub(crate) async fn known_context_window(&self, provider_id: &str, model: &str) -> Option<u64> {
        if provider_id == super::provider::resolver::CLAUDE_CODE {
            if let Some(catalog) = self
                .cognitive_routing
                .as_ref()
                .and_then(|routing| routing.claude_code_catalog.as_ref())
            {
                let from_catalog = catalog
                    .snapshot()
                    .await
                    .models
                    .iter()
                    .find(|m| m.id == model)
                    .and_then(|m| m.max_input_tokens);
                if from_catalog.is_some() {
                    return from_catalog;
                }
            }
        }
        let provider = self.provider_for(provider_id).await.ok()?;
        provider
            .capabilities(Some(model))
            .context_window
            .map(|w| w.value)
    }

    /// The window a session's system prompt is budgeted for (CTX4), resolved ONCE per
    /// opening: the window frozen in the session's capability snapshot (`snapshot`,
    /// what its provider reported when it was opened) when it holds one, else
    /// [`Self::known_context_window`], else — for the Claude Code ENGINE only —
    /// [`CLAUDE_CODE_FLOOR_WINDOW`]. The prompt is built once per opening and the
    /// composer is pure in this window, so the budget — hence the cached prefix —
    /// does not move between the turns of a session.
    ///
    /// Why an engine floor for Claude Code: its sessions often run on the CLI's OAuth
    /// login with no API key, so the live Anthropic catalog can be unreachable (static
    /// fallback, no `max_input_tokens`) and nexus declares no window for the CLI. The
    /// CLI states no window at `system_init`; it reports one only in its `result`
    /// (`modelUsage.*.contextWindow`), i.e. after the prompt is built, and taking it
    /// then would move the budget mid-session. So an unknown window on that engine
    /// keeps the budget it always had (40 000 characters, the 200K-token window it
    /// used to assume from the model's name), decided by the engine, never the name.
    /// A native provider with an unknown window keeps the 10 000 fallback.
    pub(crate) async fn session_context_window(
        &self,
        provider_id: &str,
        model: &str,
        snapshot: Option<&str>,
    ) -> Option<u64> {
        if let Some(window) = snapshot_context_window(snapshot) {
            return Some(window);
        }
        if let Some(window) = self.known_context_window(provider_id, model).await {
            return Some(window);
        }
        (provider_id == super::provider::resolver::CLAUDE_CODE).then_some(CLAUDE_CODE_FLOOR_WINDOW)
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
        // The deferred tools are part of what the provider is built with (CTX2): a
        // changed list rebuilds it for the sessions opened from now on; a running
        // session keeps the provider, hence the list, it was opened with.
        let deferred = self.deferred_tools();
        {
            let cache = self.native_cache.read().await;
            if let Some((cached, cached_deferred, provider)) = cache.get(provider_id) {
                if *cached == record && **cached_deferred == *deferred {
                    return Ok(Arc::clone(provider));
                }
            }
        }
        let (provider, concrete) = native_factory::build_provider_with_handle(
            &record,
            self.vault.clone(),
            self.native_transcripts.as_deref(),
            &self.acp,
            &self.codex,
            &deferred,
        )
        .map_err(anyhow::Error::new)?;
        {
            let mut probers = self.native_probers.write().await;
            match concrete {
                Some(native) => probers.insert(provider_id.to_string(), native),
                None => probers.remove(provider_id),
            };
        }
        // A rebuilt provider knows no probe: neither does the router.
        if let Some(routing) = &self.cognitive_routing {
            super::provider::cognitive::decider::forget_instance(
                &mut routing.probed.lock().unwrap_or_else(|e| e.into_inner()),
                provider_id,
            );
        }
        self.native_cache.write().await.insert(
            provider_id.to_string(),
            (record, deferred, Arc::clone(&provider)),
        );
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

    /// Whether a session can be given MCP servers (the project-orchestrator server
    /// among them): what its provider says NOW (`capabilities(model).per_session_mcp`:
    /// declared for an ACP agent that takes none — `build_acp` —, or learned by nexus
    /// at a refusal). A resumed session whose frozen snapshot says it carried none
    /// (`snapshot`) is not given one either: a refusal learned before a restart is not
    /// asked again. The snapshot also wins over a declaration changed since to
    /// `per_session_mcp: true`: a declaration applies to the sessions opened after it
    /// (docs/guides/providers.md, OpenClaw).
    pub(crate) fn carries_per_session_mcp(
        provider: &dyn nexus_claude::agent::AgentProvider,
        model: &str,
        snapshot: Option<&str>,
    ) -> bool {
        let refused_before = snapshot
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .and_then(|caps| {
                caps.get("per_session_mcp")
                    .and_then(serde_json::Value::as_bool)
            })
            == Some(false);
        provider.capabilities(Some(model)).per_session_mcp && !refused_before
    }

    /// The consent/gate/journal check of an in-place provider switch, run BEFORE the old
    /// provider process is closed: a refused move (`NotAllowed`, `EndpointRefused`, an
    /// un-gated trust, an unjournalable send) returns here while the old session is still
    /// exactly as it was (process live, resume token intact). It is the same
    /// [`Self::authorize_provider_use`] the opening runs, keyed on the policy mode the target
    /// would open with.
    async fn authorize_provider_switch(
        &self,
        node: &ChatSessionNode,
        provider: &str,
        model: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<()> {
        // As on a resume: re-derive the person from the stored owner when the move has no
        // caller claims (a router move), so the journal names who the content belongs to.
        let rederived_claims;
        let user_claims = match user_claims {
            Some(claims) => Some(claims),
            None => {
                rederived_claims = self.resume_identity(node).await;
                rederived_claims.as_ref()
            }
        };
        let mode = self.policy_mode_for(node).await.ok_or_else(|| {
            anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                "unknown permission mode or malformed tool pattern",
            ))
        })?;
        self.authorize_provider_use(ProviderUse {
            provider_id: provider,
            model,
            mode,
            project_slug: node.project_slug.as_deref(),
            claims: user_claims,
            session_id: &node.id.to_string(),
        })
        .await
    }

    /// The `PolicyMode` a session of `node`'s permission mode and access would open with
    /// (the same derivation [`Self::build_agent_spec_with_access`] uses for `spec.policy.mode`),
    /// `None` when the mode cannot be parsed.
    async fn policy_mode_for(
        &self,
        node: &ChatSessionNode,
    ) -> Option<nexus_claude::agent::PolicyMode> {
        let perm = self.permission_config.read().await;
        let mode = node
            .permission_mode
            .clone()
            .unwrap_or_else(|| perm.mode.clone());
        let access = super::provider::policy::SessionAccess::for_resume(node.access, None);
        super::provider::policy::tool_policy_with_access(
            &mode,
            &perm.allowed_tools,
            &perm.disallowed_tools,
            access,
        )
        .map(|policy| policy.mode)
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
        // The project's consent, or outside any project the one given for the sessions
        // without one (B3) — never the other scope's.
        let consented = store::consents_for(self.graph.as_ref(), project_slug)
            .await?
            .iter()
            .any(|c| super::provider::settings::consent_holds(c, &record));
        if !consented {
            return Err(refuse(
                resolver::ResolveError::NotAllowed(provider_id.into()).in_scope(project_slug),
            ));
        }
        let slug = project_slug;
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
            return Err(refuse(resolver::ResolveError::EndpointRefused(
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

    /// Whether this server's `nexus-tools` (the one native sessions get) embeds Obscura: then the
    /// built-in `obscura` search engine exists ([`super::provider::nexus_tools::BUILTIN_OBSCURA_ID`]).
    pub async fn nexus_tools_embeds_obscura(&self) -> bool {
        match self
            .config
            .nexus_tools_path
            .as_deref()
            .and_then(super::provider::native_factory::runnable_nexus_tools)
        {
            Some(program) => super::provider::nexus_tools::embeds_obscura(&program).await,
            None => false,
        }
    }

    /// The browser a native session of THIS installation is attached to, when its project allows
    /// it: the separate executable the operator configured (`OBSCURA_BROWSER` /
    /// `chat.obscura_browser`), else the copy embedded in `nexus-tools` (built with the `obscura`
    /// feature). Found is still not authorised: [`super::provider::nexus_tools::attach`] gives it
    /// only to a project that allowed it.
    ///
    /// This is the ONE resolution: the opening of a session calls it, and
    /// [`Self::provides_browser`] — what the Network tools panel reports (N23c) — calls it too.
    /// There is no second reading of the configuration.
    pub(crate) fn browser_of_installation(
        &self,
        program: &std::path::Path,
        embedded_obscura: bool,
    ) -> Option<nexus_claude::providers::native::BrowserTools> {
        use super::provider::native_factory::runnable_nexus_tools;
        self.config
            .nexus_browser_path
            .as_deref()
            .and_then(runnable_nexus_tools)
            .map(nexus_claude::providers::native::BrowserTools::new)
            .or_else(|| {
                embedded_obscura.then(|| {
                    nexus_claude::providers::native::BrowserTools::embedded_in_nexus_tools(
                        program.to_path_buf(),
                    )
                })
            })
    }

    /// Whether THIS installation PROVIDES the browser at all (N23c): its `nexus-tools` runs and
    /// [`Self::browser_of_installation`] — the resolution of the attachment itself — finds a
    /// browser. A project may authorise the browser either way (the two are different floors: the
    /// operator's configuration and the project's consent); when the installation provides none,
    /// the authorisation takes effect only once it does. The real case is a build without the
    /// `obscura` feature (the macOS Intel legacy archive, 10.15–13.3).
    pub async fn provides_browser(&self) -> bool {
        match self
            .config
            .nexus_tools_path
            .as_deref()
            .and_then(super::provider::native_factory::runnable_nexus_tools)
        {
            Some(program) => {
                let embedded_obscura = super::provider::nexus_tools::embeds_obscura(&program).await;
                self.browser_of_installation(&program, embedded_obscura)
                    .is_some()
            }
            None => false,
        }
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
            max_cost_usd,
            kind,
            remote_cwd,
            per_session_mcp,
            hooks,
            file_scope_lifted,
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
        // A session (re)built in a mode other than trust is no longer lifted (decision fce98b8a):
        // a REST resume in a lower mode clears the stored flag, so a later switch back to trust
        // by anyone but a person keeps it scoped.
        let file_scope_lifted =
            if file_scope_lifted && policy.mode != nexus_claude::agent::PolicyMode::Trust {
                self.set_file_scope_lifted(session_id, false).await;
                false
            } else {
                file_scope_lifted
            };
        // The project is read before the hooks scope is consumed: it is what the consent of
        // the network tools is tied to (A28).
        let project_slug = hooks.as_ref().and_then(|scope| scope.project_slug.clone());
        // The PO server's environment holds a token bound to the session: minted only
        // for a session that is given the server.
        let po_env = if remote_cwd.is_none() && per_session_mcp {
            Some(
                self.po_mcp_env(
                    Some(&mode),
                    user_claims,
                    Some(session_id),
                    access
                        .tool_profile()
                        .or_else(|| third_party_tool_profile(third_party, policy.mode)),
                    third_party,
                )
                .await,
            )
        } else {
            None
        };
        // A remote session runs where `remote_cwd` says; the local path of the
        // project (and a local `~`) means nothing on that machine.
        let mut spec = SessionSpec::new(match remote_cwd {
            Some(remote) => remote.to_string(),
            None => expand_tilde(cwd),
        });
        // `auto` is a sentinel, not a model name: left unset, Claude Code uses the account's
        // own default instead of being handed `--model auto` (which it rejects). Only Claude
        // Code sees this sentinel — a native/ACP instance resolves its own default model, never
        // `auto` — so the special case is scoped to that kind.
        spec.model = (!(kind == nexus_claude::agent::ProviderKind::ClaudeCode
            && is_auto_model(model)))
        .then(|| model.to_string());
        // What a provider kind refuses, it is not given (a refusal here would be a
        // typed `unsupported` at open): ACP has no extra dirs; Codex and ACP have no turn
        // limits. The system prompt IS given to ACP (ISO4): it has no system role, and nexus
        // delivers it as the leading text block of the first `session/prompt`.
        let is_acp = kind == nexus_claude::agent::ProviderKind::Acp;
        let has_turn_limits = matches!(
            kind,
            nexus_claude::agent::ProviderKind::ClaudeCode
                | nexus_claude::agent::ProviderKind::Native
        );
        let text = if remote_cwd.is_some() {
            format!("{system_prompt}\n\n{REMOTE_SESSION_NOTICE}")
        } else {
            system_prompt.to_string()
        };
        spec.system_prompt = Some(SystemPromptSpec {
            text,
            mode: SystemPromptMode::Replace,
        });
        spec.policy = policy;
        // A remote Claude Code cannot carry an MCP server (its configuration holds
        // the session token, which must not reach another machine's command line),
        // nor can an agent that refuses per-session MCP servers (OpenClaw's ACP
        // bridge, `per_session_mcp`): the PO tools are NOT given to it, and
        // `system_init.degraded_features` says so (`project_orchestrator_tools`,
        // [`host_degraded`]).
        // The vault token minted for the session's PO server, for a local Claude Code's own
        // shell too (below), as the historical engine gives it.
        let vault_token = po_env
            .as_ref()
            .and_then(|env| env.get("PO_VAULT_TOKEN").cloned());
        if let Some(env) = po_env {
            spec.mcp_servers.insert(
                PO_MCP_SERVER.to_string(),
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
        // The temporary directory of the session's `nexus-tools` (its command outputs): one of
        // the directories its hooks hold it to once it leaves trust.
        let mut gate_temp_dir: Option<std::path::PathBuf> = None;
        // The names of the native engine's own servers are reserved (P11b): a server of
        // the session named `nexus` would replace nexus-tools, and its tools would pass for
        // the built-in ones (`mcp__nexus__Read` granted whole for the session).
        if kind == nexus_claude::agent::ProviderKind::Native {
            refuse_reserved_native_servers(&spec.mcp_servers)?;
        }
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
                    // Built with the `obscura` feature, nexus-tools carries the keyless search
                    // engine and the browser (decision 1e9d7501): asked once per executable.
                    let embedded_obscura = nt::embeds_obscura(&program).await;
                    // The browser is held to the same rule: an absolute, executable path. Without
                    // one, the copy embedded in nexus-tools (`nexus-tools obscura mcp`). Found is
                    // still not authorised: `attach` gives it only to a project that allowed it.
                    // The SAME call answers the Network tools panel (`provides_browser`, N23c).
                    let browser = self.browser_of_installation(&program, embedded_obscura);
                    let attachment = nt::attach(
                        &nt::NexusToolsConfig {
                            program,
                            browser,
                            // The same rule as the others: an absolute, executable path.
                            // Unconfigured, the engine embedded in nexus-tools is the default
                            // and `dsh-obscura-nav` on the PATH only the fallback (OBS-R).
                            obscura: super::config::obscura_engine_program(
                                self.config.obscura_path.as_deref(),
                                embedded_obscura,
                                super::config::ChatConfig::obscura_on_path,
                                runnable_nexus_tools,
                            ),
                            embedded_obscura,
                            deny_paths: nt::host_deny_paths(self.config_yaml_path.as_deref()),
                            // The native shell's git/gh mirror the host's credentials (decision 2238b1e4).
                            host_home: self.host_home.clone(),
                        },
                        &nt::GraphConsents(self.graph.clone()),
                        self.vault.as_deref(),
                        nt::AttachInput {
                            session_id,
                            cwd: &spec.cwd,
                            extra_dirs: &spec.extra_dirs,
                            // The file scope is lifted inside `attach`, from the policy, the
                            // access and the person's choice (decision fce98b8a). A lift chosen
                            // mid-conversation takes effect at the next launch (resume); a
                            // session that leaves trust is confined by `ToolAccessHooks`.
                            access,
                            lifted_by_person: file_scope_lifted,
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
                    attach_native_servers(&mut spec.mcp_servers, attachment.servers)?;
                    gate_temp_dir = attachment.temp_dir.clone();
                    // The list the gate checks a SEARCH against, kept where the runtime can read
                    // it: a search names no destination of its own, so a question about it is
                    // about these engines (N23b, CONSENT-WIRE).
                    super::provider::nexus_tools::remember_search_origins(
                        session_id,
                        &attachment.search_origins,
                    );
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
        // The native harness writes this id in its per-request log line (`native model
        // request`, `host_session_id`), so a model request is tied to this conversation
        // (CTX0). Other providers ignore an extension that is not theirs.
        if kind == nexus_claude::agent::ProviderKind::Native {
            spec.extensions.insert(
                "native".to_string(),
                serde_json::json!({ "host_session_id": session_id }),
            );
        }
        if has_turn_limits {
            spec.max_turns = u32::try_from(self.config.max_turns).ok();
        }
        spec.limits.max_tokens = max_tokens;
        spec.limits.max_cost_usd = max_cost_usd;
        // A native session has no wall-clock cap on a turn and no token budget unless the
        // operator sets one (ISO5, Claude Code has neither): `chat.native_turn_timeout_secs`,
        // `chat.native_session_max_tokens`. A budget the request names wins.
        if kind == nexus_claude::agent::ProviderKind::Native {
            spec.limits.turn_timeout_ms = self
                .config
                .native_turn_timeout_secs
                .map(|secs| secs.saturating_mul(1000));
            spec.limits.max_tokens = max_tokens.or(self.config.native_session_max_tokens);
        }
        spec.env = EnvSpec {
            inherit: Self::child_env_inherit_names(),
            set: Default::default(),
        };
        // What the historical engine sets in a local Claude CLI's environment, so its shell
        // behaves the same on the agent engine (M2): the operator's PATH (`process_path`),
        // and what `orchestrator secret get` needs to read a granted secret through a pipe.
        // The operator's PATH reaches a local NATIVE session too: its shell is nexus-tools,
        // which inherits PATH from the backend process, the bare launchd one
        // (`/usr/bin:/bin:/usr/sbin:/sbin`) when the server runs as a service: `cargo`, `npm`...
        // were not found (ISO5, measured on session 36d623c8). The secrets below stay
        // Claude Code's.
        if kind == nexus_claude::agent::ProviderKind::Native && remote_cwd.is_none() {
            if let Some(path) = self.env_config.read().await.process_path.clone() {
                spec.env.set.insert("PATH".to_string(), path);
            }
        }
        if kind == nexus_claude::agent::ProviderKind::ClaudeCode && remote_cwd.is_none() {
            if let Some(path) = self.env_config.read().await.process_path.clone() {
                spec.env.set.insert("PATH".to_string(), path);
            }
            if let Some(token) = vault_token {
                spec.env.set.insert("PO_VAULT_TOKEN".to_string(), token);
                spec.env.set.insert(
                    "PO_SERVER_URL".to_string(),
                    format!("http://127.0.0.1:{}", self.config.server_port),
                );
            }
        }
        // A runner session (`plan run`, a protocol executor) is read here, before `hooks` moves:
        // nobody watches it, so it is not a person a question can be put to (N23b).
        let runner = hooks.as_ref().is_some_and(|scope| scope.runner);
        // The knowledge graph's hooks, for the providers that run hooks in their loop. A
        // remote session has no local project to resolve a file against.
        // A provider that runs none gets the host's fallback (`provider::hooks`, B14).
        let runs_hooks = super::provider::hooks::provider_takes_hooks(kind, remote_cwd.is_none());
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
        // Whether a PERSON is in front of this session (N23b): the only case in which a
        // destination the project has not consented to becomes a QUESTION instead of a refusal.
        // The same predicate the answer paths use for a durable decision
        // (`Claims::may_keep_lasting_approvals`: never an agent token, an MCP token, a vault token
        // or the server's account), and never a session somebody else runs — a runner has nobody
        // watching it, and a spawned session's user is its parent session (`delegate_task`, the
        // `Agent` tool), not a person in front of a dialog. Read from the session's own shape,
        // never from anything the model says.
        let spawned_lineage = self.spawned_lineage(session_id).await;
        let person_in_front = user_claims.is_some_and(|c| c.may_keep_lasting_approvals())
            && !runner
            && spawned_lineage.is_none();
        // The `Agent` tool of the native harness (P18): its calls run as child sessions of this
        // one (A18). Only for a local session nothing spawned (depth 1, A17), once the manager
        // can hand itself over (`enable_subagents`).
        if kind == nexus_claude::agent::ProviderKind::Native && remote_cwd.is_none() {
            // A spawned native session: the harness's own depth guard applies (`lineage`), and
            // it asks no question nobody would see (its user is a parent session, not a person).
            if let Some(lineage) = spawned_lineage {
                spec.lineage = Some(lineage);
                spec.policy
                    .deny
                    .push(nexus_claude::agent::ToolPattern::tool(
                        nexus_claude::providers::native::builtin::ASK_USER_QUESTION_TOOL,
                    ));
            }
            if let Some(manager) = self.weak_self() {
                if self.spawned_by_nothing(session_id).await {
                    spec.hooks = Some(Arc::new(super::subagents::SubagentHooks::new(
                        spec.hooks.take(),
                        manager,
                        session_id,
                        user_claims.cloned(),
                    )));
                }
            }
            // TEAM-1: TeamHooks wraps whatever hooks are active so far (SubagentHooks or
            // GraphSessionHooks) and intercepts `Agent(subagent_type: "team_message" /
            // "team_status")` before delegating to the real sub-agent runner.  Installed for
            // ALL native sessions (root and child) so that children receive queued
            // TeamMessages through the same drain path.
            if let Some(manager) = self.weak_self() {
                spec.hooks = Some(Arc::new(super::team::TeamHooks::new(
                    spec.hooks.take(),
                    manager,
                    session_id,
                )));
            }
        }
        // The consent of the project and the revocation of the token are checked before ANY
        // other hook, and whether or not the session has the knowledge-graph hooks.
        if let Some(search_origins) = gate {
            let mut gate_hooks = super::provider::nexus_tools::ToolAccessHooks::new(
                spec.hooks.take(),
                Arc::new(super::provider::nexus_tools::GraphConsents(
                    self.graph.clone(),
                )),
                session_id,
                project_slug,
                search_origins,
            )
            .with_scope_dirs(
                spec.cwd.clone(),
                spec.extra_dirs
                    .iter()
                    .cloned()
                    .chain(gate_temp_dir)
                    .collect(),
            )
            // N23b: a destination the project has not consented to is ASKED about in the chat,
            // when a person is in front of this session; refused as before otherwise.
            .with_person_in_front(person_in_front);
            // Rock'n roll is autonomy: a public origin lacking the project's consent is opened
            // and the person is told. It opens the network of a session only when a PERSON put
            // it in trust, with a normal access (the rule of the file-scope lift, decision
            // fce98b8a): never the runner, a protocol executor, a delegated task or a session
            // nobody owns, which run in `bypassPermissions` with nobody in front of them.
            // `set_file_scope_lifted` keeps the flag live from there on.
            crate::auth::agent_tokens::set_person_trust(
                session_id,
                super::provider::nexus_tools::lifts_file_scope(
                    &spec.policy,
                    access,
                    file_scope_lifted,
                ),
            );
            if let Some(manager) = self.weak_self() {
                let probe_session = session_id.to_owned();
                let in_trust: super::provider::nexus_tools::TrustProbe = Arc::new(move || {
                    super::provider::nexus_tools::trust_opens_network(
                        access,
                        crate::auth::agent_tokens::person_trust(&probe_session)
                            && !crate::auth::agent_tokens::is_out_of_trust(&probe_session),
                    )
                });
                let notice_session = session_id.to_owned();
                let notice: super::provider::nexus_tools::TrustNotice =
                    Arc::new(move |content: String| {
                        let (manager, session_id) = (manager.clone(), notice_session.clone());
                        Box::pin(async move {
                            // Bounded: the call waits for the notice, never for ever.
                            let delivered = async {
                                let manager = manager.upgrade()?;
                                let handle = manager.agent_runtime.get(&session_id).await?;
                                handle.emit(ChatEvent::SystemHint { content }).await;
                                Some(())
                            };
                            matches!(
                                tokio::time::timeout(std::time::Duration::from_secs(2), delivered)
                                    .await,
                                Ok(Some(()))
                            )
                        })
                    });
                gate_hooks = gate_hooks.with_trust(in_trust, notice);
            }
            spec.hooks = Some(Arc::new(gate_hooks));
        }
        Ok(spec)
    }

    /// Opens a session on the provider-neutral engine and sends the first
    /// message. The `ChatSession` node is already persisted by `create_session`.
    /// Whether the person named the provider of this request (a pin, or a strict conversation),
    /// as opposed to the router choosing it (Auto, mixed pools) or a delegated child.
    fn provider_is_explicit(request: &ChatRequest) -> bool {
        use super::provider::cognitive::ConversationRoutingMode as Mode;
        if request.spawned_by.is_some() {
            return false;
        }
        match request.routing_mode {
            Some(Mode::Primary) => true,
            Some(Mode::Mixed | Mode::Full) => false,
            None => request.provider.is_some(),
        }
    }

    /// Refuses an opening on an explicitly chosen provider whose quota is used up.
    fn refuse_explicit_provider_if_exhausted(
        &self,
        provider_id: &str,
        explicit: bool,
    ) -> Result<()> {
        if !explicit {
            return Ok(());
        }
        let Some(held) = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.quota.exhausted(provider_id))
        else {
            return Ok(());
        };
        Err(super::provider::errors::on_provider(
            provider_id,
            anyhow::Error::new(nexus_claude::agent::ProviderError::RateLimited {
                retry_after_ms: None,
                resets_at_ms: Some(held.resets_at_ms),
            }),
        ))
    }

    async fn open_agent_session(&self, o: AgentOpen<'_>) -> Result<CreateSessionResponse> {
        let AgentOpen {
            request,
            session_id,
            provider_id,
            model,
            system_prompt,
            add_dirs,
            project_slug,
            owner,
            relay,
            access,
        } = o;
        // QUOTA: a provider the person chose is never swapped for another one behind their back.
        // When its quota is used up the opening says so, with the instant it serves again, and
        // lets them decide (wait, pick another provider, resume at the reset).
        self.refuse_explicit_provider_if_exhausted(
            provider_id,
            Self::provider_is_explicit(request),
        )?;
        let provider = self.provider_for(provider_id).await?;
        let remote_cwd = self.remote_cwd_of(provider_id).await?;
        let per_session_mcp = Self::carries_per_session_mcp(provider.as_ref(), model, None);
        let sid = session_id.to_string();
        // A cap the engine cannot enforce would make it refuse to open (a delegation routed to
        // DeepSeek without a price, to Codex or to ACP): it is left out, and said.
        let (max_tokens, max_cost_usd, dropped) = super::provider::policy::enforceable_limits(
            provider.kind(),
            provider.capabilities(Some(model)).cost,
            request.max_tokens,
            request.max_cost_usd,
        );
        if !dropped.is_empty() {
            warn!(
                session_id = %sid,
                provider_id,
                model,
                ?dropped,
                "this engine cannot enforce these caps: the session opens without them"
            );
        }
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
        // From here, a failure undoes the router and the token (finding 5 of #682).
        let mut cleanup = OpeningCleanup::new(&sid, &self.turn_routing);
        let router = self
            .register_turn_router(
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
                    segments: None,
                    owner,
                },
            )
            .await;
        cleanup.registered(router);
        let mut spec = self
            .build_agent_spec_with_access(
                AgentSpecInput {
                    file_scope_lifted: request.file_scope_lifted,
                    cwd: &request.cwd,
                    model,
                    system_prompt,
                    permission_mode,
                    add_dirs,
                    user_claims: request.user_claims.as_ref(),
                    session_id: &sid,
                    third_party: provider_id != super::provider::resolver::CLAUDE_CODE,
                    max_tokens,
                    max_cost_usd,
                    kind: provider.kind(),
                    remote_cwd: remote_cwd.as_deref(),
                    per_session_mcp,
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
        // ISO2: sub-agent type tool restrictions — the child cannot use tools
        // the type definition denies.
        for tool_name in &request.denied_tools {
            spec.policy
                .deny
                .push(nexus_claude::agent::ToolPattern::tool(tool_name));
        }
        cleanup.minted(&spec, self.config.jwt_secret.as_deref());
        let tool_policy = super::provider::policy::wire_tool_policy(&spec.policy);
        self.authorize_provider_use(ProviderUse {
            provider_id,
            model: spec.model.as_deref().unwrap_or(model),
            mode: spec.policy.mode,
            project_slug,
            claims: request.user_claims.as_ref(),
            session_id: &sid,
        })
        .await?;
        // Preflight at EVERY opening (A30), not only when the instance was saved:
        // is the provider reachable and logged in now?
        if provider_id != super::provider::resolver::CLAUDE_CODE {
            let health = bounded_health(provider.as_ref(), OPEN_PREFLIGHT_TIMEOUT).await;
            if health.status == nexus_claude::agent::HealthStatus::Unavailable {
                return Err(anyhow::Error::new(health.error.unwrap_or(
                    nexus_claude::agent::ProviderError::EndpointUnreachable {
                        detail: "the provider reports itself unavailable".to_string(),
                    },
                )));
            }
        }
        let tool_profile = spec_tool_profile(&spec);
        let mut host_missing = host_degraded(
            provider.kind(),
            &spec,
            self.config.nexus_tools_path.as_deref(),
        );
        if transcripts_kept_in_memory(provider.kind(), self.native_transcripts.as_deref()) {
            host_missing.push(super::agent_runtime::RESUME_AFTER_RESTART_FEATURE.to_string());
        }
        // A cap left out is not only a log line: the session's `system_init` says so in its
        // `degraded_features` (PYSR-3b), so whoever watches the child — the session that
        // delegated, its person — reads that its budget is not held.
        host_missing.extend(super::agent_runtime::dropped_cap_features(&dropped));
        // The project's files are on this machine unless the instance is remote.
        let project_dir =
            (!super::provider::resolver::is_remote_instance(provider_id)).then(|| spec.cwd.clone());
        let mcp_names: Vec<String> = spec.mcp_servers.keys().cloned().collect();
        // Bounded below the HTTP request timeout: a provider that does not open in
        // time is a typed `timeout` the person can read, not a bare 408.
        let (session, dropped) = match tokio::time::timeout(
            PROVIDER_OPEN_TIMEOUT,
            open_dropping_refused_mcp(provider.kind(), spec, |spec| provider.open(spec)),
        )
        .await
        {
            Ok(result) => match result {
                Ok(ok) => ok,
                Err(error) => {
                    if super::provider::cognitive::decider::is_out_of_credit(&error) {
                        if let Some(routing) = self.cognitive_routing.as_ref() {
                            routing.mark_out_of_credit(provider_id);
                        }
                    }
                    return Err(anyhow::Error::new(error));
                }
            },
            Err(_) => {
                return Err(anyhow::Error::new(
                    nexus_claude::agent::ProviderError::Timeout {
                        after_ms: PROVIDER_OPEN_TIMEOUT.as_millis() as u64,
                    },
                ));
            }
        };
        let without_mcp = opened_without_mcp(per_session_mcp, dropped, session.capabilities());
        if without_mcp {
            // The token minted for the refused spec's PO server: nothing holds it.
            cleanup.revoke_token();
            host_missing.push(super::agent_runtime::PO_TOOLS_FEATURE.to_string());
        }
        // ... and does the model's window hold the tool schemas the session was
        // given, with room left to work? (known only once the provider has probed)
        if provider_id != super::provider::resolver::CLAUDE_CODE {
            if let Err(e) = window_holds_the_tools(session.capabilities(), tool_profile) {
                let _ = session.close().await;
                return Err(anyhow::Error::new(e));
            }
        }
        // Lasting approvals only for a session whose files are on this machine, running in
        // its project's root.
        let lasting = match &project_dir {
            Some(dir) => {
                Box::pin(self.lasting_scope(owner, project_slug, &dir.display().to_string())).await
            }
            None => None,
        };
        self.finish_agent_open(
            &sid,
            provider_id,
            provider.kind(),
            session,
            1,
            tool_policy,
            host_missing,
            !per_session_mcp || without_mcp,
            project_dir,
            lasting,
            false,
            mcp_names,
            super::cancelled_parents::parent_of(request.spawned_by.as_deref()),
        )
        .await?;
        // Live: the token and the router are the session's now (a refusal above left
        // `cleanup` armed, which undoes them).
        cleanup.disarm();
        // The session is open: the lists learn it now — a sub-agent opened by an `Agent`
        // call included (it never goes through an HTTP handler) — and BEFORE its first
        // turn starts, so `created` precedes that session's first CRUD `updated` (a turn
        // that fails at once ends `is_streaming: false` right away; review of frontend
        // #364: a client that saw the end before the start counted the child running).
        self.emit_session_created(session_id, request, project_slug);
        if let Some(handle) = self.agent_runtime.get(&sid).await {
            // As on Claude Code: always on, capped (decision 7b95342a); a runner always
            // continues, an interactive session unless the operator's kill switch is off.
            let (on, max) = super::post_stream::session_auto_continue(
                request.runner_context.is_some(),
                self.config.auto_continue,
            );
            handle.configure_auto_continue(on, max);
        }
        if let Some(handle) = self.agent_runtime.get(&sid).await {
            // A relayed conversation says so first, on the thread.
            if let Some(relay) = relay {
                handle.emit(relay.event(&sid, provider_id, model)).await;
            }
            if !request.message.is_empty() {
                // Counted on the node at creation (`message_count: 1`).
                handle.opening_message_counted();
                let sent = handle
                    .send_message_relayed(
                        &request.message,
                        &super::relay::prefixed(relay.and_then(|r| r.text()), &request.message),
                    )
                    .await;
                // A send that failed before its turn took the flag: the next message
                // the user sends is counted.
                if sent.is_err() {
                    handle.forget_opening_count();
                }
                sent?;
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
    /// manager as it is configured NOW (the pipeline is replaced after construction)
    /// and the session's own end-of-turn state ([`Self::agent_turn_state`]).
    fn turn_services_with(
        &self,
        session: AgentTurnState,
        mirror: Option<Arc<super::mirror::MirrorTap>>,
        hook_fallback: Option<Arc<super::provider::hooks::HookFallback>>,
    ) -> Arc<dyn super::agent_runtime::TurnServices> {
        Arc::new(ManagerTurnServices {
            graph: self.graph.clone(),
            enrichment_pipeline: self.enrichment_pipeline.clone(),
            turn_routing: Arc::clone(&self.turn_routing),
            documents: self.document_store.clone(),
            nats: self.nats.clone(),
            anchor: self.anchor_session(),
            session,
            search: self.search.clone(),
            event_emitter: self.event_emitter.clone(),
            context_injector: self.context_injector.clone(),
            step_budget: super::post_stream::POST_STREAM_STEP_BUDGET,
            stall_step: None,
            routing_store: self
                .cognitive_routing
                .as_ref()
                .and_then(|routing| routing.store.clone()),
            open_decisions: std::sync::Arc::clone(&self.open_decisions),
            objective_ledgers: std::sync::Arc::clone(&self.objective_ledgers),
            mirror,
            hook_fallback,
        })
    }

    /// The host's door to the graph hooks of `session_id` when its provider runs none
    /// ([`super::provider::hooks::HookPlan`]); `None` when the provider runs them, or
    /// for a runner session (no per-tool hook, at creation as at resume).
    async fn hook_fallback_for(
        &self,
        session_id: &str,
        provider_id: &str,
        kind: nexus_claude::agent::ProviderKind,
        caps: &nexus_claude::agent::Capabilities,
    ) -> Option<Arc<super::provider::hooks::HookFallback>> {
        let local = self
            .remote_cwd_of(provider_id)
            .await
            .ok()
            .flatten()
            .is_none();
        if !super::provider::hooks::HookPlan::for_session(kind, caps, local).needs_fallback() {
            return None;
        }
        let node = self
            .graph
            .get_chat_session(Uuid::parse_str(session_id).ok()?)
            .await
            .ok()
            .flatten()?;
        if is_runner_spawned(node.spawned_by.as_deref()) {
            return None;
        }
        // One ledger for the session: the fallback resets it on the compaction signal,
        // the redirect hook gives an advice once per ledger window (B14).
        let ledger = Arc::new(super::hook_ledger::HookLedger::new());
        let redirect =
            Self::redirect_hook(self.graph.clone(), self.hook_session_project(session_id))
                .with_ledger(Arc::clone(&ledger));
        Some(Arc::new(super::provider::hooks::HookFallback::new(
            Arc::new(redirect),
            ledger,
            session_id,
            node.cwd,
        )))
    }

    /// The held context of `session_id` ([`HeldSlot`]), as a new state of the session
    /// starts from it: the store's (a first open in this process, a restart, another
    /// instance that played turns since), once the writes of an earlier state of the
    /// session in this process landed (waited for within the step budget). When
    /// memory and store may diverge (a write failed or still runs, a turn of an
    /// earlier handle changed memory and has not written it yet) or the store
    /// cannot be read in time ([`HeldSlot::reopen`]), memory.
    async fn held_slot(&self, session_id: &str) -> Arc<HeldSlot> {
        let slot = {
            let mut slots = self.held_slots.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(slots.entry(session_id.to_string()).or_default())
        };
        // Only a slot in step with the store reloads it, and only if memory did not
        // move during the read (an earlier handle's `after_turn` still runs: `close`
        // does not stop it, `adopt` replaces a live handle).
        slot.reopen(
            &self.graph,
            session_id,
            super::post_stream::POST_STREAM_STEP_BUDGET,
        )
        .await;
        slot
    }

    /// The end-of-turn state of the agent session `session_id`, from its node: its
    /// memory conversation, the protocol run it was spawned in. Objective tracking is
    /// on, as for every Claude Code session, EXCEPT for a child of a conversation: a
    /// sub-agent's objective is the prompt it was given, never the project's backlog.
    async fn agent_turn_state(&self, session_id: &str) -> AgentTurnState {
        let node = match Uuid::parse_str(session_id) {
            Ok(uuid) => self.graph.get_chat_session(uuid).await.ok().flatten(),
            Err(_) => None,
        };
        let memory = match (
            &self.memory_config,
            node.as_ref().and_then(|n| n.conversation_id.clone()),
        ) {
            (Some(config), Some(conversation_id)) => Some(Arc::new(Mutex::new(
                ConversationMemoryManager::new(config.clone())
                    .with_conversation_id(conversation_id),
            ))),
            _ => None,
        };
        let spawned = node
            .as_ref()
            .and_then(|n| n.spawned_by.as_deref())
            .and_then(parse_spawned_by);
        // A child of a conversation is a sub-agent: its objective is the prompt it was
        // given. The objective reminder reads the PROJECT's pending tasks
        // (`post_stream::objective_reminder`), so on a child it nags about work the child
        // never took — and the child pays a whole extra billed turn to answer it.
        // Measured 17/10/2026 on a throwaway child: 9 413 cache-creation tokens, 600
        // output tokens, $0.0094, for a task list that was never its own.
        let subagent = spawned
            .as_ref()
            .is_some_and(|ctx| ctx.spawn_type == "conversation");
        // What waited for this session's next turn before it closed or the server
        // restarted.
        let held = self.held_slot(session_id).await;
        // The window the per-turn enrichment is capped for (CTX4): the one frozen at the
        // opening (registered after the snapshot is stored), else the catalog's.
        let context_window = match node.as_ref() {
            Some(n) => {
                self.session_context_window(
                    n.provider_id
                        .as_deref()
                        .filter(|p| !p.is_empty())
                        .unwrap_or(super::provider::resolver::CLAUDE_CODE),
                    &n.model,
                    n.capabilities.as_deref(),
                )
                .await
            }
            None => None,
        };
        AgentTurnState {
            memory,
            context_window,
            protocol_run_id: spawned.as_ref().and_then(|s| s.protocol_run_id),
            held,
            protocol_state: spawned.and_then(|s| s.protocol_state),
            objectives: super::post_stream::ObjectiveCounters {
                enabled: !subagent,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Records what the provider reported (frozen capabilities, resume token)
    /// and registers the live session. `host_missing`: what the host knows the session
    /// lacks ([`host_degraded`]: its `nexus-tools` server, the project-orchestrator
    /// server), which it reports as features it does not have rather than leaving only
    /// a line in the server's log.
    /// `project_dir`: the session's project directory on THIS machine (`None` for a remote
    /// session), whose `.mcp.json` decides which read-only MCP declarations hold for it
    /// (`session_grants::declarations_for_project`). `lasting`: the person and project whose
    /// lasting approvals the session keeps and obeys (`None`: `always` is not offered).
    /// `resumed`: a resumed session starts with its shell taken as moved unless it is known
    /// it never moved ([`Self::session_grants_for`]).
    #[allow(clippy::too_many_arguments)]
    async fn finish_agent_open(
        &self,
        session_id: &str,
        provider_id: &str,
        kind: nexus_claude::agent::ProviderKind,
        session: Arc<dyn nexus_claude::agent::AgentSession>,
        first_seq: i64,
        tool_policy: serde_json::Value,
        mut host_missing: Vec<String>,
        without_mcp: bool,
        project_dir: Option<std::path::PathBuf>,
        lasting: Option<super::lasting_rules::LastingScope>,
        resumed: bool,
        mcp_names: Vec<String>,
        parent: Option<String>,
    ) -> std::result::Result<(), super::cancelled_parents::ParentCancelled> {
        // What the provider's kind tells beyond its capabilities (an unknown window on
        // the native harness, no sub-agents and no fallback).
        for feature in super::agent_runtime::kind_degraded(kind, session.capabilities()) {
            if !host_missing.contains(&feature) {
                host_missing.push(feature);
            }
        }
        let mut announced = super::agent_runtime::degraded_features(session.capabilities());
        for feature in &host_missing {
            if !announced.contains(feature) {
                announced.push(feature.clone());
            }
        }
        self.warn_if_forced(provider_id, &announced);
        if let Some(router) = self.turn_routing.get(session_id) {
            router.set_model_live(session.capabilities().set_model_live);
            router.set_compaction_model(session.capabilities().compaction_model);
        }
        // The frozen snapshot keeps what the opening FOUND: a session opened without
        // its MCP servers (`without_mcp`: its provider takes none, it refused them, or
        // its snapshot already said so) says `per_session_mcp: false`, whatever a
        // provider that has learned nothing yet (a fresh one after a restart) answers.
        // Otherwise the next resume after another restart would send them again, to be
        // refused again ([`Self::carries_per_session_mcp`]).
        let mut snapshot = session.capabilities().clone();
        if without_mcp {
            snapshot.per_session_mcp = false;
        }
        let capabilities = serde_json::to_string(&snapshot).unwrap_or_default();
        let token = session.resume_token().map(|t| t.to_wire());
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self
                .graph
                .update_chat_session_harness(uuid, Some(&capabilities), token.as_deref())
                .await
            {
                warn!(session_id, error = %e, "Failed to persist the provider snapshot (non-fatal)");
            }
            // PYSR-2: what a later resume must not lose. Rewritten at every opening and
            // every in-place switch, so it describes the servers the session LAST ran with.
            // A session opened WITHOUT its servers (the provider takes none or refused
            // them: `without_mcp`) ran with none, and its resume rebuilds the spec without
            // them: recording the ones asked for would refuse every one of its resumes.
            let attached: &[String] = if without_mcp { &[] } else { &mcp_names };
            if let Err(e) = self
                .graph
                .set_chat_session_fingerprint(
                    uuid,
                    &super::resume_guard::fingerprint_json(attached),
                )
                .await
            {
                warn!(session_id, error = %e, "Failed to persist the resume fingerprint (non-fatal)");
            }
        }
        let kind_name = provider_kind_name(&kind);
        // A provider that runs no hook gets the host's door to them (B14).
        let hook_fallback = self
            .hook_fallback_for(session_id, provider_id, kind, session.capabilities())
            .await;
        let handle = self
            .agent_runtime
            .adopt_admitting(
                session_id,
                provider_id,
                session,
                first_seq,
                &kind_name,
                tool_policy,
                Some(
                    self.turn_services_with(
                        self.agent_turn_state(session_id).await,
                        // Claude Code runs its sub-agents itself: they get read-only mirrors
                        // (P18c). A native session's sub-agents are real children already.
                        matches!(kind, nexus_claude::agent::ProviderKind::ClaudeCode)
                            .then(|| self.agent_mirror_tap(session_id))
                            .flatten(),
                        hook_fallback,
                    ),
                ),
                host_missing,
                self.session_grants_for(
                    session_id,
                    project_dir.as_deref(),
                    lasting,
                    resumed,
                    super::session_grants::Asker::from_provider_kind(&kind_name),
                )
                .await,
                // B36-RACE: refused under the runtime's write lock when a cascade closed
                // the parent while this child was opening (`chat::cancelled_parents`).
                || self.cancelled_parents.admit(session_id, parent.as_deref()),
            )
            .await
            .inspect_err(|refused| {
                info!(session_id, parent = %refused.parent_session_id, "delegated session refused: its parent was cancelled");
            })?;
        // A failed turn is retried as the chat's configuration says, on both engines.
        handle.configure_retry(self.config.retry.clone());
        // A provider instance the detector declares DOWN holds the turn instead of
        // closing it, as the operator configured (`CHAT_PROVIDER_OUTAGE_*`,
        // `CHAT_PROVIDER_HOLD_*`): the agent engine's control half of outage
        // detection (`chat::provider::outage`).
        handle.configure_outage(
            super::provider::outage::OutagePolicy::from_env(),
            super::provider::outage::HoldPolicy::from_env(),
        );
        self.spawn_agent_nats_listeners(handle);
        Ok(())
    }

    /// The loop of an agent-engine cancel listener: hands each request to `on_request`
    /// until the session closes. `biased`, the close first: once `closed` is
    /// cancelled no request is read any more, even one already waiting, so a session
    /// being closed (or adopted again) does not ask its provider session to cancel
    /// (review of #685, finding 1; the legacy listeners do the same since #673).
    async fn serve_agent_cancels<M>(
        mut requests: impl futures::Stream<Item = M> + Unpin,
        closed: CancellationToken,
        mut on_request: impl FnMut(M),
    ) {
        loop {
            tokio::select! {
                biased;
                _ = closed.cancelled() => break,
                msg = requests.next() => {
                    let Some(msg) = msg else { break };
                    on_request(msg);
                }
            }
        }
    }

    /// `work` (a cancel asked of an agent-engine session from another instance),
    /// unless the session closes before it answers: then `session_gone`, which a
    /// v2 asker reads as provisional (the real owner, if any, still answers),
    /// instead of what a provider session being closed would say (`failed`,
    /// `refused`: final answers that would beat the real owner's). A request read
    /// just before the close lands here (review of #685, finding 1).
    ///
    /// The close is watched for the whole cancel, not only at its start (review
    /// of #691, finding 3): a close during `work` drops it and answers `gone`, and
    /// an outcome that comes back once the session is closed is replaced by
    /// `gone` too. The cancel may then have reached the provider already (the
    /// signal may have left): `gone` says nothing about that, which is what the
    /// 410 tells the user ("may or may not have been stopped"; cancel-task is
    /// retryable, a task signalled twice is still one stopped task).
    async fn unless_agent_closed<T>(
        closed: &CancellationToken,
        kind: super::cancel_relay::CancelKind,
        work: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        let gone =
            || anyhow::Error::new(super::cancel_relay::CancelRelayError::SessionGone { kind });
        if closed.is_cancelled() {
            return Err(gone());
        }
        let outcome = tokio::select! {
            biased;
            _ = closed.cancelled() => return Err(gone()),
            outcome = work => outcome,
        };
        if closed.is_cancelled() {
            return Err(gone());
        }
        outcome
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
            // there finds no local session and asks over NATS): the handle applies
            // the cap and stops the tools, as `spawn_nats_cancel_tools_listener`
            // does for a Claude Code session, and the asker gets what it did or
            // the typed refusal. One task per request, bounded
            // (`cancel_relay::OWNER_CANCEL_BOUND`).
            let (nats, handle, sid) = (nats.clone(), Arc::clone(&handle), sid.clone());
            tokio::spawn(async move {
                let Ok(sub) = nats.subscribe_cancel_tools(&sid).await else {
                    warn!(session_id = %sid, "agent engine: NATS cancel_tools subscription failed");
                    return;
                };
                let closed = handle.closed.clone();
                Self::serve_agent_cancels(sub, closed, move |msg: crate::events::transport::Message| {
                    let (nats, handle) = (nats.clone(), Arc::clone(&handle));
                    tokio::spawn(async move {
                        let kind = super::cancel_relay::CancelKind::Tools;
                        let work =
                            Self::unless_agent_closed(&handle.closed, kind, handle.cancel_tools());
                        let asker_v2 = super::cancel_relay::asker_v2(&msg.payload);
                        match msg.reply {
                            Some(reply_to) => {
                                super::cancel_relay::answer(&nats, reply_to, asker_v2, kind, work)
                                    .await
                            }
                            None => {
                                if let Err(e) = work.await {
                                    warn!(session_id = %handle.session_id, error = %e, "agent engine: NATS cancel_tools failed");
                                }
                            }
                        }
                    });
                })
                .await;
            });
        }
        {
            // `cancel_task` asked on another instance: the handle stops the task
            // (cap included) and the asker gets what it did, or the typed refusal.
            // One task per request, bounded.
            let (nats, handle, sid) = (nats.clone(), Arc::clone(&handle), sid.clone());
            tokio::spawn(async move {
                let Ok(sub) = nats.subscribe_cancel_task(&sid).await else {
                    warn!(session_id = %sid, "agent engine: NATS cancel_task subscription failed");
                    return;
                };
                let closed = handle.closed.clone();
                Self::serve_agent_cancels(
                    sub,
                    closed,
                    move |msg: crate::events::transport::Message| {
                        let Some(reply_to) = msg.reply else { return };
                        let task_id = super::cancel_relay::task_id_of(&msg.payload);
                        let (nats, handle) = (nats.clone(), Arc::clone(&handle));
                        tokio::spawn(async move {
                            let kind = super::cancel_relay::CancelKind::Task;
                            super::cancel_relay::answer(
                                &nats,
                                reply_to,
                                super::cancel_relay::asker_v2(&msg.payload),
                                kind,
                                Self::unless_agent_closed(
                                    &handle.closed,
                                    kind,
                                    handle.cancel_task(&task_id),
                                ),
                            )
                            .await;
                        });
                    },
                )
                .await;
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
                                let _ = nats.publish(&reply_to, payload).await;
                            }
                        }
                    }
                }
            });
        }
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
                            Ok(request) => agent_rpc(&handle, &request).await,
                            Err(e) => crate::events::ChatRpcResponse {
                                success: false,
                                error: Some(format!("Invalid request: {e}")),
                            },
                        };
                        if let (Some(reply_to), Ok(payload)) = (msg.reply, serde_json::to_vec(&response)) {
                            let _ = nats.publish(&reply_to, payload).await;
                        }
                    }
                }
            }
        });
    }

    /// Reopens a session of the agent engine that is no longer live, from its
    /// persisted resume token (a new session when it never got one), then
    /// delivers `message`.
    ///
    /// `seen`: the live handle the caller started from, read before it awaited anything
    /// (`None`: the session was not live; a dead handle: a resume after a failed send).
    /// The identity a resumed session reopens under when its caller brought none (the
    /// restart recovery, or a resume that arrives without claims): the person who opened
    /// it, read from `ChatSessionNode.owner` (their `sub`) and resolved through the user
    /// record for the `email`/`name` the session token copies. `None` when the session has
    /// no owner, or the user is gone — the session then stays without an identity (its PO
    /// tools run unauthorised) rather than forge one.
    async fn resume_identity(&self, node: &ChatSessionNode) -> Option<crate::auth::jwt::Claims> {
        let owner = node.owner.as_deref()?;
        let id = Uuid::parse_str(owner).ok()?;
        let user = self.graph.get_user_by_id(id).await.ok().flatten()?;
        Some(crate::auth::jwt::Claims::person(
            owner,
            &user.email,
            &user.name,
        ))
    }

    async fn resume_agent_session(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        seen: Option<Arc<super::agent_runtime::AgentSessionHandle>>,
    ) -> Result<()> {
        let sid = node.id.to_string();
        // One resume of a session at a time (reviews of #684, finding 2, and #696, finding
        // 1): two messages to a session that is not live — or two failed sends on one dead
        // handle — would each resume it, each minting a token that supersedes the other's,
        // the second ending the first's handle, and the one that fails would undo what the
        // other set up. Once it holds the lock, a resume that finds a live handle OTHER than
        // the one it started from was overtaken by a resume that succeeded: its message goes
        // to that session, through `send_message` (routing applies in `full` mode). The same
        // handle (dead, a resume after a failed send) or none: it resumes. When the send to
        // the handle that overtook it fails (its process died while this message waited, the
        // NATS RPC included: a handle leaves the runtime only by `close`), it resumes over
        // that handle, as after any failed send (review of #706, finding 1). Still under the
        // lock: no new race.
        let _resuming = self.resume_locks.lock(&sid).await;
        if let Some(now) = self.agent_runtime.get(&sid).await {
            if seen.as_ref().is_none_or(|seen| !Arc::ptr_eq(seen, &now)) {
                match self.send_message(&sid, message).await {
                    Ok(()) => return Ok(()),
                    Err(error) => warn!(
                        session_id = %sid,
                        %error,
                        "the session that overtook this resume refused the message: resuming over it"
                    ),
                }
            }
        }
        self.reopen_agent_session(node, message, user_claims).await
    }

    /// Reopens a session IN PLACE on another provider (B-SW phase 1): the SAME `session_id`
    /// continues, on the target provider/model it was just re-targeted to, replaying its
    /// stored history through `relay` in front of the first message. It IS the shared resume
    /// path of [`Self::resume_session_on`]: the branch on the engine the node was re-targeted
    /// to (the agent runtime, or the legacy Claude Code CLI), with the router re-seeded from
    /// the carried segmenter state. Unlike a resume, the consent/gate/journal check already
    /// ran in `authorize_provider_switch` (before the old process closed), so the leaf
    /// reopens skip it.
    async fn reopen_session_on_provider(
        &self,
        session_id: &str,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        relay: Option<&super::relay::RelayedFrom>,
    ) -> Result<()> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;
        let node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {session_id} not found in database"))?;
        // The engine written by `update_chat_session_provider` names the branch: a session
        // re-targeted to a native provider reopens on the agent runtime; to Claude Code on
        // the legacy CLI — the SAME branch a later resume takes (`resume_session_on`).
        if node.opened_on_agent() {
            self.reopen_agent_session_with(&node, message, user_claims, relay)
                .await
        } else {
            self.reopen_legacy_session(&node, message, user_claims, relay)
                .await
        }
    }

    /// Reopens a session of the agent engine that is no longer live (the caller holds the
    /// session's resume lock): from its persisted resume token, or a fresh open, then
    /// delivers `message`.
    pub(super) async fn reopen_agent_session(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<()> {
        self.reopen_agent_session_with(node, message, user_claims, None)
            .await
    }

    /// The hooks a reopened session of the agent engine gets: the same as at its
    /// opening (`open_agent_session`), B14.
    fn resume_agent_hook_scope(node: &ChatSessionNode) -> AgentHookScope {
        AgentHookScope {
            project_slug: node.project_slug.clone(),
            task_id: node
                .spawned_by
                .as_deref()
                .and_then(parse_spawned_by)
                .and_then(|ctx| ctx.task_id),
            // A runner session gets no per-tool hook, as at its opening (the reopen
            // used to give it every one).
            runner: is_runner_spawned(node.spawned_by.as_deref()),
        }
    }

    /// [`Self::reopen_agent_session`] with `relay`: an in-place provider switch reopens on
    /// the node it was just re-targeted to (its provider/model are the target's), the first
    /// message goes out relay-prefixed, and the router is re-seeded from the carried
    /// segmenter state. `relay == None` is the plain resume.
    async fn reopen_agent_session_with(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
        relay: Option<&super::relay::RelayedFrom>,
    ) -> Result<()> {
        let sid = node.id.to_string();
        // (P20-suite-2) The restart recovery reopens the session with no caller: re-derive
        // the person from the owner stored at open, so the session token can be minted
        // again and the PO tools keep their auth. No owner, or a gone user: stays without
        // one (the tools run unauthorised rather than forging an identity).
        let rederived_claims;
        let user_claims: Option<&crate::auth::jwt::Claims> = match user_claims {
            Some(claims) => Some(claims),
            None => {
                rederived_claims = self.resume_identity(node).await;
                rederived_claims.as_ref()
            }
        };
        let provider_id = node
            .provider_id
            .clone()
            .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string());
        let provider = self.provider_for(&provider_id).await?;
        let remote_cwd = self.remote_cwd_of(&provider_id).await?;
        let per_session_mcp = Self::carries_per_session_mcp(
            provider.as_ref(),
            &node.model,
            node.capabilities.as_deref(),
        );
        let (system_prompt, _) = self
            .build_system_prompt_anchored_for(
                &super::composer::PromptAudience::of(node.spawned_by.as_deref(), &provider_id),
                node.execution_place,
                &node.cwd,
                node.project_slug.as_deref(),
                message,
                Some(&node.model),
                self.session_context_window(
                    &provider_id,
                    &node.model,
                    node.capabilities.as_deref(),
                )
                .await,
                &sid,
                None,
            )
            .await;
        // From here, a failure undoes the router and the token (finding 5 of #682).
        let mut cleanup = OpeningCleanup::new(&sid, &self.turn_routing);
        // A move by the router (`auto`) keeps the conversation's routing (mode, pool) and
        // lets it choose again; a move by the user imposes the target (`routed_by: request`).
        // A plain resume keeps the mode the conversation was opened with, and starts counting
        // its turns again (the harness is the authority on the index).
        let moved_in = relay.is_some_and(|r| r.moved_by == super::relay::MOVED_BY_AUTO);
        let routing_mode = if relay.is_some() && !moved_in {
            None
        } else {
            node.routing_mode
                .as_deref()
                .and_then(|m| serde_json::from_value(serde_json::Value::String(m.to_owned())).ok())
                // The router moves only a conversation `full` in effect; a pool kept without
                // a mode says so (review of #713, C1-bis).
                .or_else(|| {
                    (moved_in && node.routing_pool.is_some())
                        .then_some(super::provider::cognitive::ConversationRoutingMode::Full)
                })
        };
        let routing_pool = if relay.is_some() && !moved_in {
            None
        } else {
            node.routing_pool
                .as_deref()
                .and_then(|json| serde_json::from_str(json).ok())
        };
        let router = self
            .register_turn_router(
                &sid,
                &provider_id,
                &node.model,
                node.project_slug.as_deref(),
                OpeningTurn {
                    routing_mode,
                    routing_pool,
                    explicit_model: false,
                    permission_mode: node.permission_mode.as_deref(),
                    // A provider the request named stays imposed across a resume, and a user
                    // move imposes its target (a pool, carried only by a router move, keeps
                    // the provider free); a router move never imposes it.
                    provider_imposed: node.routed_by.as_deref() == Some("request")
                        && node.routing_pool.is_none(),
                    moved_in,
                    message,
                    next_turn: 0,
                    segments: relay.and_then(|r| r.segments.clone()),
                    owner: node.owner.as_deref(),
                },
            )
            .await;
        cleanup.registered(router);
        // A resume keeps the access the session was opened with and can only narrow it.
        let access = super::provider::policy::SessionAccess::for_resume(node.access, None);
        // PYSR-3b: a resumed delegation (or a sub-agent of one) keeps its delay and budget —
        // re-derived from the same constants, less what it already spent (the engine process
        // a resume opens counts from zero). A cap this engine cannot hold is left out and
        // said, as at the opening.
        let (max_tokens, max_cost_usd, dropped_caps) = match self.session_limits_of(node).await {
            Some(limits) => {
                let (tokens, usd) = limits.request_caps();
                super::provider::policy::enforceable_limits(
                    provider.kind(),
                    provider.capabilities(Some(&node.model)).cost,
                    tokens,
                    usd,
                )
            }
            None => (None, None, Vec::new()),
        };
        if !dropped_caps.is_empty() {
            warn!(
                session_id = %sid,
                provider_id,
                model = %node.model,
                dropped = ?dropped_caps,
                "this engine cannot enforce these caps: the session resumes without them"
            );
        }
        let spec = self
            .build_agent_spec_with_access(
                AgentSpecInput {
                    file_scope_lifted: node.file_scope_lifted,
                    cwd: &node.cwd,
                    model: &node.model,
                    system_prompt: &system_prompt,
                    permission_mode: node.permission_mode.as_deref(),
                    add_dirs: node.add_dirs.as_deref().unwrap_or(&[]),
                    user_claims,
                    session_id: &sid,
                    third_party: provider_id != super::provider::resolver::CLAUDE_CODE,
                    max_tokens,
                    max_cost_usd,
                    kind: provider.kind(),
                    remote_cwd: remote_cwd.as_deref(),
                    per_session_mcp,
                    hooks: Some(Self::resume_agent_hook_scope(node)),
                },
                access,
            )
            .await?;
        cleanup.minted(&spec, self.config.jwt_secret.as_deref());
        let tool_policy = super::provider::policy::wire_tool_policy(&spec.policy);
        if relay.is_none() {
            // A resume is a new sending of the project's content: consent, guard and
            // journal apply again (a consent may have been revoked meanwhile). A switch
            // already passed `authorize_provider_switch` before the old process closed.
            self.authorize_provider_use(ProviderUse {
                provider_id: &provider_id,
                model: &node.model,
                mode: spec.policy.mode,
                project_slug: node.project_slug.as_deref(),
                claims: user_claims,
                session_id: &sid,
            })
            .await?;
            // PYSR-2: a plain resume may not lose an MCP server the conversation was
            // opened with. Nothing is written yet: the router and the token this reopen
            // minted are undone by `cleanup` on the way out.
            let now: Vec<String> = spec.mcp_servers.keys().cloned().collect();
            let stored = self
                .graph
                .get_chat_session_fingerprint(node.id)
                .await
                .unwrap_or(None);
            if let Err(incompatible) =
                super::resume_guard::check_servers_kept(stored.as_deref(), &now)
            {
                return Err(anyhow::Error::new(incompatible));
            }
        }
        // A native resume token names a transcript FILE on disk, and that file is written
        // only when a turn COMMITS (`loop.rs`). A session cut before its first committed
        // turn — a fresh open, notably the first turn right after a provider switch —
        // therefore carries a token that names NOTHING, and every resume then failed with
        // "unknown transcript: nothing to resume", bricking the session for good even
        // though the graph holds the whole conversation (a switch cut mid-first-turn).
        // When the transcript the token names is absent, do NOT resume it: reopen FRESH
        // with the stored history relayed in front of the message (the same rendering a
        // provider switch uses), so a conversation is never lost.
        let lost_native = relay.is_none()
            && node.resume_token.as_deref().is_some_and(|wire| {
                super::provider::transcripts::named_transcript_missing(
                    self.native_transcripts.as_deref(),
                    &provider_id,
                    wire,
                ) == Some(true)
            });
        let fallback_relay = if lost_native {
            let records = self
                .graph
                .get_chat_events(node.id, 0, 5_000)
                .await
                .unwrap_or_default();
            let events: Vec<ChatEvent> = records
                .iter()
                .filter_map(|record| serde_json::from_str(&record.data).ok())
                .collect();
            let window = provider
                .capabilities(Some(node.model.as_str()))
                .context_window
                .map(|w| w.value);
            warn!(
                session_id = %sid,
                provider = %provider_id,
                "resume: the native transcript is gone; reopening fresh with the stored history relayed"
            );
            Some(super::relay::RelayedFrom {
                relay: super::relay::render_relay(
                    &events,
                    &provider_id,
                    &provider_id,
                    super::relay::budget_for_window(window),
                ),
                from_session_id: sid.clone(),
                from_provider: provider_id.clone(),
                // Not a router move: a resume that repairs a lost transcript never imposes
                // a target and never counts as an automatic `full`-mode move.
                moved_by: super::relay::MOVED_BY_USER.to_string(),
                conversation_id: node.conversation_id.clone(),
                segments: None,
                next_turn: 0,
            })
        } else {
            None
        };
        // No resume token survives a move across provider families, nor a repair of a lost
        // transcript: both open fresh (the second replays the history as a relay instead).
        let token = if relay.is_some() || fallback_relay.is_some() {
            None
        } else {
            node.resume_token
                .as_deref()
                .and_then(|raw| nexus_claude::agent::ResumeToken::from_wire(raw).ok())
        };
        let mut host_missing = host_degraded(
            provider.kind(),
            &spec,
            self.config.nexus_tools_path.as_deref(),
        );
        if transcripts_kept_in_memory(provider.kind(), self.native_transcripts.as_deref()) {
            host_missing.push(super::agent_runtime::RESUME_AFTER_RESTART_FEATURE.to_string());
        }
        // A cap left out is said to whoever watches the session (PYSR-3b).
        host_missing.extend(super::agent_runtime::dropped_cap_features(&dropped_caps));
        // The project's files are on this machine unless the instance is remote.
        let project_dir = (!super::provider::resolver::is_remote_instance(&provider_id))
            .then(|| spec.cwd.clone());
        // A resume without its owner's identity (the restart recovery: the owner's claims
        // are not stored on the session) carries the PO server but no token for it while
        // the server signs its tokens: every PO tool would be refused. Said, not silent
        // (review of #697, finding 8b).
        if po_server_unauthorised(&spec, self.config.jwt_secret.is_some())
            && !host_missing
                .iter()
                .any(|f| f == super::agent_runtime::PO_TOOLS_FEATURE)
        {
            host_missing.push(super::agent_runtime::PO_TOOLS_FEATURE.to_string());
        }
        let mcp_names: Vec<String> = spec.mcp_servers.keys().cloned().collect();
        let opened = open_dropping_refused_mcp(provider.kind(), spec, |spec| {
            let provider = Arc::clone(&provider);
            let token = token.clone();
            async move {
                match token {
                    Some(token) => provider.resume(spec, token).await,
                    None => provider.open(spec).await,
                }
            }
        })
        .await;
        let (session, dropped) = match opened {
            Ok(ok) => ok,
            Err(error) => {
                if super::provider::cognitive::decider::is_out_of_credit(&error) {
                    if let Some(routing) = self.cognitive_routing.as_ref() {
                        routing.mark_out_of_credit(&provider_id);
                    }
                }
                return Err(anyhow::Error::new(error));
            }
        };
        let without_mcp = opened_without_mcp(per_session_mcp, dropped, session.capabilities());
        if without_mcp {
            // As at an opening: the token went out in the refused request.
            cleanup.revoke_token();
            if !host_missing
                .iter()
                .any(|f| f == super::agent_runtime::PO_TOOLS_FEATURE)
            {
                host_missing.push(super::agent_runtime::PO_TOOLS_FEATURE.to_string());
            }
        }
        let latest = self
            .graph
            .get_latest_chat_event_seq(node.id)
            .await
            .unwrap_or(0);
        let lasting = match &project_dir {
            Some(dir) => {
                Box::pin(self.lasting_scope(
                    node.owner.as_deref(),
                    node.project_slug.as_deref(),
                    &dir.display().to_string(),
                ))
                .await
            }
            None => None,
        };
        self.finish_agent_open(
            &sid,
            &provider_id,
            provider.kind(),
            session,
            latest + 1,
            tool_policy,
            host_missing,
            !per_session_mcp || without_mcp,
            project_dir,
            lasting,
            // A provider switch is NOT a resume: the shell never left the project root, so the
            // session's lasting `always` rules still bind (a resume starts optimistically
            // moved, which would make every `always` command be asked again). Only a plain
            // resume (no relay) may have moved.
            relay.is_none(),
            mcp_names,
            // B36-RACE: a warm child the pool resumes for a delegation is refused like a fresh
            // one; a person resuming it passes (B36-RACE-RESUME).
            super::cancelled_parents::parent_checked_on_resume(node.spawned_by.as_deref()),
        )
        .await?;
        cleanup.disarm();
        let handle = self
            .agent_runtime
            .get(&sid)
            .await
            .ok_or_else(|| anyhow!("Session {sid} vanished while resuming"))?;
        // A resumed session is interactive: on unless the operator's kill switch is off,
        // capped (as on Claude Code; the flag an older version persisted is not read).
        let (on, max) = super::post_stream::session_auto_continue(false, self.config.auto_continue);
        handle.configure_auto_continue(on, max);
        if let Some(relay) = relay {
            // The stored thread keeps showing `message` alone; the model sees the relay in
            // front of it (never stored).
            handle
                .send_message_relayed(message, &super::relay::prefixed(relay.text(), message))
                .await
        } else if let Some(lost) = fallback_relay.as_ref() {
            // The lost-transcript repair: the same as a switch — the thread shows `message`
            // alone, the model reads the rebuilt history in front of it (never stored).
            handle
                .send_message_relayed(message, &super::relay::prefixed(lost.text(), message))
                .await
        } else {
            handle.send_message(message).await
        }
    }

    /// A legacy session refused at its activation (B36-RACE): never inserted, its MCP
    /// token revoked and its CLI disconnected (dropped after 5 s, which kills it).
    async fn refuse_legacy_activation(
        &self,
        session_id: &str,
        client: Arc<Mutex<InteractiveClient>>,
        refused: &super::cancelled_parents::ParentCancelled,
    ) {
        info!(session_id, parent = %refused.parent_session_id, "delegated session refused: its parent was cancelled");
        crate::auth::agent_tokens::revoke_session(session_id);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            client.lock().await.disconnect().await
        })
        .await;
    }

    /// Close an active session: interrupt first, then disconnect and remove.
    ///
    /// T3 fix (Gap 5): call interrupt() BEFORE removing the session from
    /// active_sessions so the stream loop can observe the flag/token. Then
    /// disconnect with a 5s timeout — if the CLI hangs, drop the client to
    /// trigger SIGKILL via the Drop impl.
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        self.close_session_with_cascade(session_id, false).await
    }

    /// Close a session and, when `cascade` is set, every session that descends
    /// from it (its delegated children and their children, through SPAWNED_BY).
    ///
    /// Top-down (B36): a session is closed BEFORE its children are read, so a
    /// session whose children have been listed can no longer delegate. A
    /// delegation already in flight when its parent closed can still land
    /// after that read, so the tree is read again until a pass finds nothing
    /// new. A descendant that is no longer live is not closed again (its own
    /// children are still reached). Each closed session loses its neutral
    /// directory. A descendant that cannot be closed does not stop the rest:
    /// the first error is reported.
    ///
    /// A delegation whose node is written but whose session is not live yet (its
    /// provider still opening) is seen INACTIVE by the read below and skipped: it is
    /// refused at its activation instead (B36-RACE, `chat::cancelled_parents`). Each
    /// session the cascade closes gets a tombstone BEFORE it is closed — so before its
    /// children are read — and every activation of a child checks its parent's tombstone
    /// inside the critical section that inserts it in the live map. A child inserted
    /// BEFORE the tombstone is seen live by the read (which comes after) and closed here;
    /// one inserted AFTER reads the tombstone and is refused. The tombstone's mutex being
    /// taken inside that critical section, no third order exists. Nothing waits: a
    /// cancel with no delegation in flight costs one map insert per closed session.
    pub async fn close_session_with_cascade(&self, session_id: &str, cascade: bool) -> Result<()> {
        if cascade {
            self.cancelled_parents.mark(session_id);
        }
        let mut result = self.close_session_inner(session_id).await;
        // The neutral directory (if the session had one) goes with the session; a resume
        // makes it again.
        super::neutral_place::remove_for(session_id);
        if !cascade {
            return result;
        }
        let mut seen = std::collections::HashSet::from([session_id.to_string()]);
        loop {
            let mut found_new = false;
            let mut frontier = vec![session_id.to_string()];
            // Visited in this pass: a malformed link cycle cannot loop forever.
            let mut visited = std::collections::HashSet::from([session_id.to_string()]);
            while let Some(parent) = frontier.pop() {
                for child in self.session_children(&parent).await {
                    if seen.insert(child.clone()) {
                        found_new = true;
                        // Its own in-flight delegations are refused too — even when it is
                        // closed already: a delegation it started before closing may still
                        // be opening.
                        self.cancelled_parents.mark(&child);
                        if self.is_session_active(&child).await {
                            if let Err(e) = self.close_session_inner(&child).await {
                                tracing::warn!(session_id = %child, error = %e, "cascade: closing a descendant failed");
                                if result.is_ok() {
                                    result = Err(e);
                                }
                            }
                            super::neutral_place::remove_for(&child);
                        }
                    }
                    // Read only once the child is closed (or was already).
                    if visited.insert(child.clone()) {
                        frontier.push(child);
                    }
                }
            }
            if !found_new {
                return result;
            }
        }
    }

    /// The sessions spawned by `session_id` through SPAWNED_BY. A graph read
    /// that fails is "no children": nothing is invented.
    async fn session_children(&self, session_id: &str) -> Vec<String> {
        let Ok(parent_id) = Uuid::parse_str(session_id) else {
            return Vec::new();
        };
        self.graph
            .get_session_children(parent_id)
            .await
            .map(|children| children.into_iter().map(|c| c.id.to_string()).collect())
            .unwrap_or_default()
    }

    /// Delete a session: close it if it is live, remove what it keeps on disk (the
    /// transcript of a native session, P14), then the node. `Ok(false)`: no such
    /// session. A CLOSED session keeps its transcript (it stays resumable): only a
    /// deletion removes it, and before the node, which is what names it. A transcript
    /// that cannot be removed fails the deletion (the conversation would stay on disk
    /// with nothing left to find it); a retry removes both.
    ///
    /// A parent takes its descendants with it (P18-suite, point 9): they are closed first by
    /// the cascade (B36: top-down, a delegation in flight refused at its activation), then
    /// DELETED, deepest first, the parent last. Deleted rather than re-parented: a spawned
    /// session is listed only under its parent (the session lists hide every session that
    /// has a `spawned_by`), so a child that outlived its parent would be reachable from
    /// nowhere while its `spawned_by` still named a parent that no longer exists — and a
    /// child is a part of its parent's conversation (its delegated work), which the person
    /// asked to delete. A descendant that cannot be deleted fails the deletion BEFORE the
    /// parent goes, so a retry still finds the tree and finishes it.
    pub async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        // Before the close, as before: an open mirror's tap must not write it again.
        self.mirrors.deleted(session_id);
        let root = session_id.to_string();
        let _ = self.close_session_with_cascade(&root, true).await;
        // The descendants, read once the whole tree is closed; parents before children.
        let mut tree = Vec::new();
        let mut seen = std::collections::HashSet::from([root.clone()]);
        let mut frontier = vec![root];
        while let Some(parent) = frontier.pop() {
            for child in self.session_children(&parent).await {
                if seen.insert(child.clone()) {
                    tree.push(child.clone());
                    frontier.push(child);
                }
            }
        }
        for child in tree.iter().rev() {
            if let Ok(child) = Uuid::parse_str(child) {
                self.delete_one_session(child).await?;
            }
        }
        self.delete_one_session(session_id).await
    }

    /// [`Self::delete_session`] of ONE session, already closed: its transcript, then its node.
    async fn delete_one_session(&self, session_id: Uuid) -> Result<bool> {
        // An open mirror of a Claude Code sub-agent: its tap drops it and never writes it again.
        self.mirrors.deleted(session_id);
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
        self.close_session_inner_with(
            session_id,
            true,
            crate::episodes::models::close_reason::USER_CLOSED,
        )
        .await
    }

    /// Closes a session and records WHY in its close episode. The reason is the caller's
    /// knowledge — a sub-agent's call returning is not a person closing the window — and
    /// it is the only thing that makes two closed sessions comparable after the fact.
    pub async fn close_session_because(&self, session_id: &str, reason: &str) -> Result<()> {
        self.close_session_inner_with(session_id, true, reason)
            .await
    }

    /// Closes the legacy sessions whose CLI process is gone, with the typed reason
    /// `process_died` on their `session_closed`. Returns the ids closed.
    ///
    /// Why it matters (ZOMBIE-CLI): a killed CLI stays `<defunct>` until something waits
    /// for it, and its session stays "live" with its child slot held. Closing the session
    /// runs `disconnect()`, whose `child.wait()` reaps the zombie, and removes the session
    /// from the live table, which frees the parent's slot and purges its background tasks.
    pub(crate) async fn close_sessions_of_dead_clis(&self) -> Vec<String> {
        let dead: Vec<String> = {
            let sessions = self.active_sessions.read().await;
            sessions
                .iter()
                .filter(|(_, s)| process_alive(s.child_pid) == Some(false))
                .map(|(id, _)| id.clone())
                .collect()
        };
        // The agent engine (the default): the provider said, out of a turn, that its process is
        // gone. A turn in flight would have noticed and detached the handle by itself; an idle
        // session never did, and held its child slot.
        let mut dead = dead;
        for handle in self.agent_runtime.handles().await {
            if handle.provider_is_dead() && !dead.contains(&handle.session_id) {
                dead.push(handle.session_id.clone());
            }
        }
        for id in &dead {
            warn!(session_id = %id, "CLI process is gone: closing its session (process_died)");
            if let Err(e) = self
                .close_session_inner_with(
                    id,
                    true,
                    crate::episodes::models::close_reason::PROCESS_DIED,
                )
                .await
            {
                warn!(session_id = %id, error = %e, "closing a session of a dead CLI failed");
            }
        }
        dead
    }

    /// Runs [`Self::close_sessions_of_dead_clis`] every few seconds, so a dead CLI frees its
    /// slot within seconds rather than after the idle cleanup's half hour.
    pub fn start_dead_cli_watchdog(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(DEAD_CLI_WATCH_INTERVAL);
            loop {
                ticker.tick().await;
                manager.close_sessions_of_dead_clis().await;
            }
        });
    }

    /// Arms the LOST-TOOL-RESULT watchdog (issue #1056).
    ///
    /// What it repairs: a session left `streaming` because the answer to one of its tool
    /// calls never came back. Measured on 2026-11-01 — a parent asked for a child through
    /// `chat(action: "send_message")`, the child was created and finished, and the parent
    /// stayed `streaming` for 26+ minutes, no result and no error, until a human
    /// interrupted it. No existing watchdog covers that case: the interrupt watchdog needs a
    /// stop to have been asked, and the dead-CLI one needs a dead process.
    ///
    /// It only ever FIRES AN INTERRUPT — the ordinary, engine-agnostic way to end a turn —
    /// after saying why in the conversation. It never closes a session: the work is still
    /// there, and the next message resumes it.
    pub fn start_lost_tool_result_watchdog(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(LOST_TOOL_RESULT_WATCH_INTERVAL);
            loop {
                ticker.tick().await;
                manager.abandon_turns_whose_tool_result_never_came().await;
            }
        });
    }

    /// One tick of the LOST-TOOL-RESULT watchdog: ends the turns whose last persisted event
    /// is a tool call that got no result long ago, and returns the sessions it ended.
    ///
    /// The three cheap filters come first, in memory (`session_activity_map`): a turn is a
    /// candidate only if it is streaming, holds no background subprocess (a monitor or a
    /// `run_in_background` shell legitimately says nothing for minutes) and waits on no
    /// permission. Only then is the graph asked what the LAST event was — two indexed reads,
    /// for the few sessions that pass. A store that cannot answer means "not judged", never
    /// "guess": a wrong interrupt costs a turn.
    pub(crate) async fn abandon_turns_whose_tool_result_never_came(&self) -> Vec<String> {
        let now = chrono::Utc::now();
        let mut abandoned = Vec::new();
        for (id, activity) in self.session_activity_map().await {
            if !activity.streaming
                || activity.background_tasks() > 0
                || activity.pending_permissions > 0
            {
                continue;
            }
            let Some((tool, at)) = self.last_event_unanswered_tool_use(id).await else {
                continue;
            };
            let silent_for = now.signed_duration_since(at).to_std().unwrap_or_default();
            let grace = if is_own_mcp_tool(&tool) {
                LOST_TOOL_RESULT_GRACE_MCP
            } else {
                LOST_TOOL_RESULT_GRACE_COMMAND
            };
            if silent_for < grace {
                continue;
            }
            warn!(
                session_id = %id,
                tool = %tool,
                silent_secs = silent_for.as_secs(),
                grace_secs = grace.as_secs(),
                "a tool result is lost: the turn waited past every budget of that call, ending it"
            );
            self.emit_session_event(
                Some(&id.to_string()),
                ChatEvent::Error {
                    message: format!(
                        "The call to `{tool}` never came back: this turn waited {} min for its \
                         result, which is past every budget that call has, so the turn was ended. \
                         The session is free again and your next message continues the \
                         conversation. Only this turn's answer was lost.",
                        silent_for.as_secs() / 60
                    ),
                    parent_tool_use_id: None,
                    code: Some("tool_result_lost".to_string()),
                    reason: None,
                    index: None,
                    request_id: None,
                },
            )
            .await;
            if let Err(e) = self.interrupt(&id.to_string()).await {
                warn!(session_id = %id, error = %e, "ending a turn stuck on a lost tool result failed");
                continue;
            }
            abandoned.push(id.to_string());
        }
        abandoned
    }

    /// The last persisted event of `session_id` when it is a `tool_use`, i.e. when nothing —
    /// no result, no answer — has been written since that call. `None` for a session with no
    /// event yet, and for any other last event: a finished turn, a message, a question.
    async fn last_event_unanswered_tool_use(
        &self,
        session_id: Uuid,
    ) -> Option<(String, chrono::DateTime<chrono::Utc>)> {
        let count = match self.graph.count_chat_events(session_id).await {
            Ok(count) => count,
            Err(e) => {
                debug!(session_id = %session_id, error = %e, "cannot count its events: session not judged");
                return None;
            }
        };
        if count == 0 {
            return None;
        }
        let last = match self
            .graph
            .get_chat_events_paginated(session_id, count - 1, 1)
            .await
        {
            Ok(mut events) => events.pop()?,
            Err(e) => {
                debug!(session_id = %session_id, error = %e, "cannot read its last event: session not judged");
                return None;
            }
        };
        if last.event_type != "tool_use" {
            return None;
        }
        let data: serde_json::Value = serde_json::from_str(&last.data).ok()?;
        let tool = data
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        Some((tool, last.created_at))
    }

    /// Stores the episode a closed session leaves (`episodes::collector::collect_session_episode`).
    ///
    /// Called on an ANNOUNCED close only — never on an in-place switch, whose thread goes on.
    /// Best effort, bounded and detached: a graph that fails or hangs is logged, and the
    /// close neither waits for it nor fails because of it.
    /// Turns what a session really had in context into `SYNAPSE` edges (T5).
    ///
    /// Extracted from the close so it can be tested without a live session: the rule needs a
    /// project (the co-activation never welds two projects' notes together) and it needs at
    /// least two notes. `None` says « nothing to do », which is not a failure — a session that
    /// read one note has no pair to weld.
    pub(crate) async fn reinforce_session_knowledge(
        graph: std::sync::Arc<dyn crate::neo4j::traits::GraphStore>,
        session: Uuid,
        notes: Vec<Uuid>,
    ) -> Option<crate::graph::rules::RuleOutcome> {
        if notes.len() < 2 {
            return None;
        }
        // The session carries a slug, not an id: the co-activation is scoped to the project
        // the notes must belong to.
        let project = match graph.get_chat_session(session).await {
            Ok(Some(s)) => match s.project_slug.as_deref() {
                Some(slug) => match graph.get_project_by_slug(slug).await {
                    Ok(Some(p)) => Some(p.id),
                    _ => None,
                },
                None => None,
            },
            _ => None,
        };
        match crate::graph::rules::run_with(
            graph.as_ref(),
            crate::graph::rules::SESSION_CO_ACTIVATION.name,
            crate::graph::rules::Measurement::SessionNotes {
                session,
                project,
                notes,
            },
        )
        .await
        {
            Ok(outcome) => Some(outcome),
            Err(e) => {
                warn!(session_id = %session, error = %e, "the session's co-activation failed");
                None
            }
        }
    }

    /// Links to their tasks and plans the commits a session made, before its episode is
    /// collected.
    ///
    /// This is the piece that was missing. `TaskEnricher` already turned git into
    /// `Commit` + `TOUCHES` + `RESOLVED_BY`, but was wired only into the plan runner: a task
    /// finished by a CHAT kept `get_task_commits() == []`. The delivery was real; the link
    /// was never made (RM-4, decision `c23f494b`).
    ///
    /// Best effort, and deliberately narrow: a session with no linked task, or no linked
    /// plan, is left alone — a commit is never attributed to a task by guesswork — and only
    /// the COMMIT half of the enrichment runs (`TaskEnricher::link_commits`), so a close
    /// does not litter the graph with an auto-note per session.
    pub(crate) async fn link_session_commits(
        graph: std::sync::Arc<dyn crate::neo4j::traits::GraphStore>,
        session: Uuid,
    ) {
        let Ok(Some(node)) = graph.get_chat_session(session).await else {
            return;
        };
        let Ok(links) = graph.get_session_links(session).await else {
            return;
        };
        if links.linked_tasks.is_empty() || links.linked_plans.is_empty() {
            return;
        }
        // The evidence follows the REPOSITORY (`cwd`), not the session's project slug,
        // which is measured to lie: the session of a backend task carried `nexus-sdk`.
        let project = match node.project_slug.as_deref() {
            Some(slug) => graph
                .get_project_by_slug(slug)
                .await
                .ok()
                .flatten()
                .map(|p| p.id),
            None => None,
        };
        let enricher = crate::runner::enricher::TaskEnricher::new(graph.clone());
        for task in &links.linked_tasks {
            for plan in &links.linked_plans {
                let (linked, _) = enricher
                    .link_commits(task.id, plan.id, project, node.created_at, &node.cwd)
                    .await;
                if linked > 0 {
                    info!(
                        session_id = %session,
                        task_id = %task.id,
                        commits = linked,
                        "the commits of a closed session are linked to its task"
                    );
                }
            }
        }
    }

    fn record_close_episode(&self, session_id: &str, reason: &str) {
        let Ok(id) = Uuid::parse_str(session_id) else {
            return;
        };
        let graph = self.graph.clone();
        let embeddings = self.embeddings.clone();
        let reason = reason.to_string();
        let work = async move {
            // The counter before anything reads it: a user message whose count write is still
            // in flight (the drain's, past its step budget) is counted by the episode as by
            // the session (T0). Half the budget, so the episode is still written when the
            // write hangs.
            if !crate::chat::drain::count_writes_settled(id, CLOSE_EPISODE_BUDGET / 2).await {
                warn!(
                    session_id = %id,
                    "the message counter of a closed session was still being written: its episode may count one message short"
                );
            }
            // The links first: the episode collected below names the commits of the window,
            // and the graph must already know which task they resolve.
            Self::link_session_commits(graph.clone(), id).await;
            let Some(mut episode) =
                crate::episodes::collector::collect_session_episode(graph.as_ref(), id, &reason)
                    .await?
            else {
                return Ok(false);
            };
            // The vector the reflex stage recalls it by (`reflex::episode_recall`): the
            // first message, which is short. No provider, or a provider that fails: the
            // episode is kept, it is only not recallable by similarity.
            if let Some(provider) = embeddings {
                match provider.embed_text(&episode.stimulus.request).await {
                    Ok(vector) if !vector.is_empty() => episode.embedding = Some(vector),
                    Ok(_) => {}
                    Err(e) => {
                        debug!(session_id = %id, error = %e, "close episode left without embedding")
                    }
                }
            }
            graph.save_session_episode(&episode).await
        };
        // What the session really had in context becomes edges, once. The registry that
        // holds it is in memory and this is its last moment: after the close, the notes it
        // welded are all that is left.
        let graph = self.graph.clone();
        let activated = crate::chat::activation_registry::get(id).notes;
        if activated.len() >= 2 {
            tokio::spawn(async move {
                match tokio::time::timeout(
                    CLOSE_EPISODE_BUDGET,
                    Self::reinforce_session_knowledge(graph, id, activated),
                )
                .await
                {
                    Ok(Some(outcome)) => debug!(
                        session_id = %id,
                        written = outcome.written,
                        "{}",
                        outcome.measure
                    ),
                    Ok(None) => debug!(session_id = %id, "nothing to co-activate"),
                    Err(_) => warn!(session_id = %id, "the session's co-activation timed out"),
                }
            });
        }

        // Detached: the close never waits on the graph for its episode (a cascade or a
        // deletion closes many sessions in a row). The write cannot outlive a deletion:
        // it is one statement that MATCHes the session, and a deletion removes the session
        // and its episode in one statement.
        tokio::spawn(async move {
            match tokio::time::timeout(CLOSE_EPISODE_BUDGET, work).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    warn!(session_id = %id, error = %e, "the session's close episode was not stored")
                }
                Err(_) => warn!(session_id = %id, "storing the session's close episode timed out"),
            }
        });
    }

    /// Tears the session down WITHOUT announcing `session_closed`: an in-place provider
    /// switch, whose same thread must see only `provider_switched`, never `session_closed`.
    /// Unlike [`Self::close_session`], it does not remove a neutral directory.
    async fn close_session_for_switch(&self, session_id: &str) -> Result<()> {
        // No episode is written here (announce = false), so the reason is only carried by
        // the event: it names the switch rather than pretending it was a close.
        self.close_session_inner_with(
            session_id,
            false,
            crate::episodes::models::close_reason::SWITCHED,
        )
        .await
    }

    async fn close_session_inner_with(
        &self,
        session_id: &str,
        announce: bool,
        reason: &str,
    ) -> Result<()> {
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            self.anchor_cache.forget(uuid);
        }
        // Any close but the Done release (which marks the session AFTER this) ends the
        // team's right to resume it.
        self.released_done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(session_id);
        self.turn_routing.remove(session_id);
        self.close_routing_decision(session_id).await;
        let objective_state = self
            .objective_ledgers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_id)
            .map(|ledger| ledger.state().clone());
        debug!(%session_id, objective_state = ?objective_state, "objective ledger at close");
        if self.agent_runtime.owns(session_id).await {
            self.agent_modes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(session_id);
            crate::auth::agent_tokens::revoke_session(session_id);
            // Every instance learns the session is gone, as on the legacy path: the
            // `session_closed` the runtime emits is published like every event. A quiet
            // close (an in-place switch) skips that announcement.
            if announce {
                self.agent_runtime
                    .close_with_reason(session_id, reason)
                    .await?;
                self.record_close_episode(session_id, reason);
            } else {
                self.agent_runtime.close_for_switch(session_id).await?;
            }
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
            // Its NATS listeners end with it (review N1).
            session.nats_cancel.cancel();
            (
                session.client,
                session.protocol_run_id,
                session.protocol_state,
                session.events_tx,
            )
        };
        self.notify_attention(session_id, AttentionReason::SessionInactive);

        // A45: tell every client (and every instance) the session is gone. A quiet close
        // (an in-place switch) skips it: the same thread sees only `provider_switched`.
        if announce {
            let closed = ChatEvent::SessionClosed {
                session_id: session_id.to_string(),
                reason: Some(reason.to_string()),
            };
            let _ = events_tx.send(closed.clone());
            if let Some(ref nats) = self.nats {
                nats.publish_chat_event(session_id, closed);
            }
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

        // After the disconnect: the node holds the session as it ended. A quiet close (an
        // in-place switch) is not an end, and leaves no episode.
        if announce {
            self.record_close_episode(session_id, reason);
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
                expired.extend(manager.expired_agent_sessions(timeout).await);

                for id in expired {
                    info!("Cleaning up timed-out session {}", id);
                    if let Err(e) = manager.close_session(&id).await {
                        warn!("Failed to close timed-out session {}: {}", id, e);
                    }
                }
            }
        });
    }

    /// Runs [`Self::release_done_sessions`] every [`DONE_RELEASE_INTERVAL`].
    pub fn start_done_release_task(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(DONE_RELEASE_INTERVAL);
            loop {
                ticker.tick().await;
                manager.release_done_sessions().await;
            }
        });
    }

    /// Releases the sessions whose current segment is `Done` (`chat::segment_state`):
    /// their turn ended, their output went to whoever spawned them (or there was none),
    /// no watch, no queue, nothing planned. Returns the sessions it closed.
    ///
    /// It closes ON THAT STATE ONLY, never on a duration: a session that reports to a
    /// person, watches, or waits on one is never `Done`, however long it idles (that is
    /// still the idle cleanup's business, unchanged). The close is the idle cleanup's own,
    /// [`Self::close_session`]: the thread stays in the graph, readable, and the next
    /// message resumes the session the ordinary way.
    ///
    /// Both engines. A Claude Code session that is `Done` is closed too, which frees its
    /// child slot: a team message to a session that is no longer live resumes it
    /// (`inject_team_message`), so closing early no longer cuts its team off.
    ///
    /// The verdict is read twice — on a first snapshot, then again on a fresh one just
    /// before closing — so a message that arrived in between keeps its session.
    pub(crate) async fn release_done_sessions(&self) -> Vec<String> {
        self.release_done_sessions_at(super::segment_tap::now_ms())
            .await
    }

    /// [`Self::release_done_sessions`] with the clock given (tests move it past the settle
    /// window instead of sleeping through it).
    pub(crate) async fn release_done_sessions_at(&self, now_ms: u64) -> Vec<String> {
        use super::segment_state::SessionState;
        let done = |snap: &LiveSessionSnapshot| -> std::collections::HashSet<Uuid> {
            snap.states
                .iter()
                .filter(|(_, v)| v.state == SessionState::Done)
                .map(|(id, _)| *id)
                .collect()
        };
        let candidates: Vec<Uuid> = done(&self.live_session_snapshot_at(now_ms).await)
            .into_iter()
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let still = self.live_session_snapshot_at(now_ms).await;
        let still_done = done(&still);
        let mut released = Vec::new();
        for id in candidates {
            if !still_done.contains(&id) {
                continue;
            }
            let sid = id.to_string();
            #[cfg(test)]
            {
                let hook = self
                    .release_hook
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone();
                if let Some(hook) = hook {
                    hook(&sid);
                }
            }
            // A message may have landed since the second read: only an idle session with an
            // empty queue is closed, or its new turn would be interrupted.
            if !self.is_idle_for_release(&sid).await {
                continue;
            }
            // A delegated child or an open sub-agent mirror still running: the parent is
            // not finished (the verdict counts monitors and bash tasks only).
            let has_live_child = self
                .session_children(&sid)
                .await
                .iter()
                .filter_map(|c| c.parse::<Uuid>().ok())
                .any(|c| still.live.contains(&c));
            if has_live_child {
                continue;
            }
            let reason = still.states.get(&id).map(|v| v.reason.clone());
            info!(session_id = %sid, reason = ?reason, "Session done: its mission is over, releasing it");
            let legacy = !self.agent_runtime.owns(&sid).await;
            match self.close_session(&sid).await {
                Ok(()) => {
                    if legacy {
                        self.released_done
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .insert(sid.clone());
                    }
                    released.push(sid)
                }
                Err(e) => warn!(session_id = %sid, error = %e, "Failed to release a done session"),
            }
        }
        released
    }

    /// Whether `session_id` is still idle right now: no turn running, nothing queued.
    pub(crate) async fn is_idle_for_release(&self, session_id: &str) -> bool {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return !handle.is_streaming.load(Ordering::SeqCst) && handle.queued_len().await == 0;
        }
        let sessions = self.active_sessions.read().await;
        match sessions.get(session_id) {
            Some(s) => {
                !s.is_streaming.load(Ordering::SeqCst) && s.pending_messages.lock().await.is_empty()
            }
            None => false,
        }
    }

    /// The agent engine's sessions the idle cleanup closes, by the rule of the
    /// Claude Code engine ([`session_is_expired`]): a running turn or a background
    /// task its provider still reports running (P4) keeps an idle session.
    async fn expired_agent_sessions(&self, timeout: Duration) -> Vec<String> {
        let mut expired = Vec::new();
        for handle in self.agent_runtime.handles().await {
            let idle = handle.idle_for();
            let is_streaming = handle.is_streaming.load(Ordering::SeqCst);
            let background_tasks = handle.active_background_tasks().await.len();
            if session_is_expired(idle, timeout, is_streaming, background_tasks) {
                expired.push(handle.session_id.clone());
            } else if idle > timeout {
                debug!(
                    session_id = %handle.session_id,
                    idle_secs = idle.as_secs(),
                    is_streaming,
                    background_tasks,
                    "Idle agent session kept alive: work still in progress"
                );
            }
        }
        expired
    }

    /// Closes the most idle session that is doing nothing (no running turn, no background
    /// task) and has been for `grace`, on either engine, to make room for a
    /// new one. Closing is what the idle cleanup does half an hour later: the session stays
    /// in the graph and resumes on the next message. Returns whether a slot was freed.
    pub(crate) async fn reclaim_idle_slot(&self, grace: Duration) -> bool {
        let mut best: Option<(Duration, String)> = None;
        let mut consider = |idle: Duration, id: &str| {
            if idle >= grace && best.as_ref().is_none_or(|(b, _)| idle > *b) {
                best = Some((idle, id.to_string()));
            }
        };
        let legacy: Vec<_> = {
            let sessions = self.active_sessions.read().await;
            sessions
                .iter()
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
        for (id, idle, streaming, background, cli_background) in legacy {
            let busy = streaming.load(Ordering::Relaxed)
                || background
                    .lock()
                    .await
                    .len()
                    .max(cli_background.load(Ordering::Relaxed))
                    > 0;
            if !busy {
                consider(idle, &id);
            }
        }
        for handle in self.agent_runtime.handles().await {
            let busy = handle.is_streaming.load(Ordering::SeqCst)
                || !handle.active_background_tasks().await.is_empty();
            if !busy {
                consider(handle.idle_for(), &handle.session_id);
            }
        }
        let Some((idle, id)) = best else {
            return false;
        };
        info!(session_id = %id, idle_secs = idle.as_secs(), "Session table full: closing the most idle session");
        match self.close_session(&id).await {
            Ok(()) => true,
            Err(e) => {
                warn!("Failed to reclaim idle session {}: {}", id, e);
                false
            }
        }
    }

    /// Get the number of currently active sessions, on both engines.
    pub async fn active_session_count(&self) -> usize {
        let legacy = self.active_sessions.read().await.len();
        legacy + self.agent_runtime.len().await
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
            // A process that has exited but not been reaped still answers kill(0):
            // it is a zombie (`<defunct>`), not a running process. Counting it alive
            // kept a dead CLI's background task and its child slot forever.
            return Some(!is_zombie(pid));
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

/// Is `pid` a zombie: exited, its status not yet collected by its parent?
#[cfg(unix)]
fn is_zombie(pid: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        // `/proc/<pid>/stat` is `pid (comm) S ...`; the state follows the last `)`.
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| {
                stat.rsplit(')')
                    .next()
                    .map(|rest| rest.trim_start().starts_with('Z'))
            })
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .map(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .trim_start()
                    .starts_with('Z')
            })
            .unwrap_or(false)
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
#[path = "manager/tests/idle_expiry_tests.rs"]
mod idle_expiry_tests;

/// The wire name of a provider kind (`claude_code`, `native`, `codex`...), from which a
/// session's `session_grants::Asker` is derived. Never a guess: a kind that does not
/// serialize to a name is `unknown`, whose requests are never granted for the session
/// (`Asker::Other`, the most restrictive).
fn provider_kind_name<K: serde::Serialize + ?Sized>(kind: &K) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Parse a raw SDK control message into a [`ChatEvent::PermissionRequest`] if it is
/// a `can_use_tool` request.  Returns `None` for any other subtype.
///
/// This is the pure-logic core extracted from the `handle_control_msg` closure inside
/// `stream_response`.  Having it as a standalone function makes it directly testable
/// The call a pending request of the CLI asks for, as the grants compare it.
fn legacy_asked_call(pending: &PendingPermission) -> super::session_grants::AskedCall {
    super::session_grants::AskedCall {
        asker: super::session_grants::Asker::ClaudeCode,
        tool: pending.tool.clone(),
        canonical: None,
        input: pending.input.clone(),
    }
}

/// Marks a permission request of the CLI that may be answered `always` (the session keeps
/// lasting rules and the comparator would approve this very call for good): the client
/// offers `always` only then.
fn mark_always_eligibility(
    event: &mut ChatEvent,
    grants: &std::sync::Mutex<super::session_grants::SessionGrants>,
) {
    if let ChatEvent::PermissionRequest {
        tool,
        input,
        always,
        ..
    } = event
    {
        let call = super::session_grants::AskedCall {
            asker: super::session_grants::Asker::ClaudeCode,
            tool: tool.clone(),
            canonical: None,
            input: input.clone(),
        };
        *always = grants
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .lasting_grant_for(&call)
            .map(|_| true);
    }
}

/// `system_init` of a legacy session, offering `always` when the session keeps lasting
/// rules (`SessionGrants::keeps_lasting`): what [`ChatManager::deliver_permission_answer`]
/// accepts. Any other event is returned as it is.
fn offer_lasting_scope(
    mut event: ChatEvent,
    grants: &std::sync::Mutex<super::session_grants::SessionGrants>,
) -> ChatEvent {
    if let ChatEvent::SystemInit {
        capabilities: Some(caps),
        ..
    } = &mut event
    {
        let keeps = grants
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keeps_lasting(super::session_grants::Asker::ClaudeCode);
        if keeps {
            if let Some(scopes) = caps
                .get_mut("permission_scopes")
                .and_then(|s| s.as_array_mut())
            {
                if !scopes.iter().any(|s| s == "always") {
                    scopes.push(serde_json::json!("always"));
                }
            }
        }
    }
    event
}

/// Store a pending permission input in the shared map.
/// Extracted as a standalone async fn so that `tokio::select!` blocks don't
/// struggle with type inference on `Arc<Mutex<HashMap>>` inline locks.
async fn store_pending_perm_input(
    map: &Arc<tokio::sync::Mutex<std::collections::HashMap<String, PendingPermission>>>,
    id: &str,
    input: &serde_json::Value,
    tool: &str,
) {
    map.lock().await.insert(
        id.to_string(),
        PendingPermission {
            input: input.clone(),
            tool: tool.to_string(),
        },
    );
}

/// A permission request of the CLI the session's grants cover (`chat::session_grants`):
/// answered here, once, with its original input; the request leaves the pending map.
/// Returns the decision to record, `None` when nothing covers it or the write failed
/// (the user is then asked as usual). The answer is on the session's tool clock before
/// the CLI has it, taken back when the write fails, as for an answer of the user.
async fn answer_granted_request(
    grants: &std::sync::Mutex<super::session_grants::SessionGrants>,
    pending: &tokio::sync::Mutex<std::collections::HashMap<String, PendingPermission>>,
    clock: &super::tool_clock::ToolClock,
    stdin_tx: Option<&tokio::sync::mpsc::Sender<String>>,
    id: &str,
    tool: &str,
    input: &serde_json::Value,
) -> Option<ChatEvent> {
    let call = super::session_grants::AskedCall {
        asker: super::session_grants::Asker::ClaudeCode,
        tool: tool.to_string(),
        canonical: None,
        input: input.clone(),
    };
    let cover = Box::pin(super::session_grants::cover(grants, &call)).await?;
    let tx = stdin_tx?;
    let line = serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": id,
            "response": { "behavior": "allow", "updatedInput": input }
        }
    })
    .to_string();
    // Claimed before the write, like an answer of the user: whoever claims first answers,
    // never two answers to one request (a request already claimed is left alone). Given
    // back if the write fails: the request then still waits for the user.
    let answering = clock.answering(id).await;
    let claimed = pending.lock().await.remove(id)?;
    let mark = clock.decided(id, true, chrono::Utc::now());
    if tx.send(line).await.is_err() {
        clock.undecided(id, mark);
        pending.lock().await.insert(id.to_string(), claimed);
        return None;
    }
    drop(answering);
    info!(request_id = %id, rule = %cover.rule, lasting = cover.lasting, "permission covered by a grant: answered by the backend");
    Some(ChatEvent::PermissionDecision {
        id: id.to_string(),
        allow: true,
        scope: Some(if cover.lasting {
            super::types::PermissionAnswerScope::Always
        } else {
            super::types::PermissionAnswerScope::Session
        }),
        rule: Some(cover.rule),
    })
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

/// The control response that closes an `AskUserQuestion` request (§15.6): a denial with
/// the question-posed message, so the turn ends and the user's answer comes as the next
/// turn. Must match `nexus_claude`'s `QUESTION_POSED_MESSAGE` (the SDK's own copy).
fn question_closing_response(request_id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": request_id,
            "response": {
                "behavior": "deny",
                "message": "The question was posed to the user. Their answer will arrive in their next message; do not guess it.",
            }
        }
    })
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
        tool_use_id: request_data
            .get("toolUseId")
            .or_else(|| request_data.get("tool_use_id"))
            .and_then(|v| v.as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        // Set by the session (`mark_always_eligibility`), which knows its lasting rules.
        always: None,
        consent: None,
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
#[path = "manager/tests/runner_spawn_tests.rs"]
mod runner_spawn_tests;

/// Release a streaming claim for a turn that will not run: `stream_response`
/// found the session gone from `active_sessions` before the turn started. The
/// caller (OOB trigger, drain) may already have set `is_streaming`; without
/// this, the session reads "streaming" forever and nothing finalizes it.
pub(crate) fn release_unstarted_turn(
    is_streaming: &AtomicBool,
    events_tx: &broadcast::Sender<ChatEvent>,
    nats: &Option<Arc<crate::events::NatsEmitter>>,
    session_id: &str,
) {
    is_streaming.store(false, Ordering::SeqCst);
    let event = ChatEvent::StreamingStatus {
        is_streaming: false,
    };
    let _ = events_tx.send(event.clone());
    if let Some(nats) = nats {
        nats.publish_chat_event(session_id, event);
    }
}

#[cfg(test)]
#[path = "manager/tests/unstarted_turn_tests.rs"]
mod unstarted_turn_tests;

#[cfg(test)]
#[path = "manager/tests/tests.rs"]
mod tests;

/// Fixtures for handler tests that need a LIVE session (a registered
/// `ActiveSession` whose CLI stdin is a channel the test can read).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn chat_config() -> ChatConfig {
        ChatConfig::default()
    }

    /// Register a live session. Returns the receiver of what would be written
    /// to the CLI's stdin and the queue of messages received while streaming.
    ///
    /// The CLI is `fake_claude`, the scripted stand-in for the Claude CLI that
    /// nexus builds (`NEXUS_FAKES_DIR`, the same binaries as the agent e2e tests),
    /// RUNNING: connected, it waits for its stdin to close (600 s at most, see
    /// [`connected_fake_cli`]), so the session is
    /// live for `evict_if_cli_dead` as it would be with the real CLI. No `claude`
    /// has to be installed, and a missing fake FAILS instead of skipping. What the
    /// session writes to the CLI goes through the `stdin` channel the test reads.
    pub(crate) async fn insert_live_session(
        manager: &ChatManager,
        session_id: &str,
        is_streaming: bool,
        pending_permissions: &[&str],
    ) -> (
        tokio::sync::mpsc::Receiver<String>,
        Arc<Mutex<VecDeque<PendingMessage>>>,
    ) {
        let client = connected_fake_cli().await;
        insert_session_with_client(
            manager,
            session_id,
            client,
            is_streaming,
            pending_permissions,
        )
        .await
        .expect("the session is registered")
    }

    /// A client connected to a running `fake_claude` that stays up until its
    /// stdin closes (the client is dropped, or the test process ends) or for at
    /// most 600 s: the transcript (`tests/fixtures/fake_claude/live_until_eof.jsonl`,
    /// checked in, so nothing is left in the temp dir) waits for EOF with a 600 s
    /// timeout, the same bound as the fake's watchdog set below.
    pub(crate) async fn connected_fake_cli() -> InteractiveClient {
        let transcript = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fake_claude/live_until_eof.jsonl");
        let mut options = nexus_claude::ClaudeCodeOptions {
            model: Some("test".into()),
            cli_path: Some(crate::chat::agent_e2e_tests::fake_bin("fake_claude")),
            ..Default::default()
        };
        options.env.insert(
            "FAKE_CLAUDE_TRANSCRIPT".into(),
            transcript.display().to_string(),
        );
        // The fake's watchdog (30 s by default) must not end a CLI a slow test holds;
        // same bound as the `wait_eof` timeout of the transcript.
        options
            .env
            .insert("FAKE_CLAUDE_MAX_RUNTIME_MS".into(), "600000".into());
        let mut client =
            InteractiveClient::new(options).expect("an explicit CLI path is never searched for");
        client.connect().await.expect("fake_claude starts");
        assert!(client.is_alive().await, "the fake CLI is running");
        client
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
            pending.insert(
                (*id).to_string(),
                super::PendingPermission {
                    input: serde_json::json!({ "command": "ls" }),
                    tool: "Bash".to_string(),
                },
            );
        }
        let session = ActiveSession {
            context_window: None,
            anchor: Default::default(),
            mirror_tap: None,
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
            turn_generation: Arc::new(AtomicU64::new(0)),
            result_cursor: Default::default(),
            stream_task: Arc::new(std::sync::Mutex::new(None)),
            post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(pending)),
            provider_outage: Arc::new(std::sync::Mutex::new(
                crate::chat::provider::outage::TurnHold::new(
                    crate::chat::provider::resolver::CLAUDE_CODE,
                ),
            )),
            session_grants: Arc::default(),
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
            tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
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
#[path = "manager/tests/agent_env_tests.rs"]
mod agent_env_tests;

/// `ManagerTurnServices::prepare` × `refs::turn`: what the agent engine sends for
/// a turn whose message carries `#` references (H3 enrichment + refs_v1).
#[cfg(test)]
#[path = "manager/tests/refs_turn_services_tests.rs"]
mod refs_turn_services_tests;

#[cfg(test)]
#[path = "manager/tests/auto_open_bounds_tests.rs"]
mod auto_open_bounds_tests;

/// The end-of-turn steps of the agent engine are bounded one by one, as the Claude
/// Code engine's (`StepBudget::run`): a step that never answers is dropped and
/// said, the next one still runs (before, one budget held the three together, so a
/// slow re-injection lost the memory and the feedback too).
#[cfg(test)]
#[path = "manager/tests/agent_step_budget_tests.rs"]
mod agent_step_budget_tests;

/// What waits for a session's next turn after a compaction (`HeldContext`): kept
/// across a restart, dropped only once a turn carrying it was sent, cut to the
/// window; and the end-of-turn steps each under their own budget.
#[cfg(test)]
#[path = "manager/tests/held_context_tests.rs"]
mod held_context_tests;

/// F-R4: the host's own order — `prepare` hands the router the turn's input (its images
/// included), so `model_for_images`, asked right after, routes an image turn to a model that
/// reads it. Played with the REAL services and the real decision.
#[cfg(test)]
#[path = "manager/tests/image_turn_services_tests.rs"]
mod image_turn_services_tests;

/// R0 — the per-turn reward, through a REAL native turn: a `FakeProvider` plays the
/// turn, `AgentRuntime::play()` reads its terminal `done`, `ManagerTurnServices::after_turn`
/// feeds the bandit. No `TurnOutcome` is built by hand here: before this module the
/// capture (`play`) and the settlement (`after_turn`) were each tested alone, so a
/// field `play` forgot to fill would have left every test green and the posterior still.
#[cfg(test)]
#[path = "manager/tests/turn_reward_play_tests.rs"]
mod turn_reward_play_tests;
/// What a test runs the moment the Claude Code CLI can read a permission answer of the
/// user ([`ChatManager::deliver_permission_answer`]), before the function goes on: the
/// CLI's next request lands there in the worst case (FLAKY-PARITE-ALWAYS).
#[cfg(test)]
pub(crate) mod answer_probe {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    type Probe =
        Box<dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send>;

    /// One probe per session (a session id is never shared between tests).
    static PROBES: LazyLock<Mutex<HashMap<String, Probe>>> = LazyLock::new(Default::default);

    /// Runs `probe` once, when the CLI of `session_id` has the next answer.
    pub(crate) fn on_cli_has_answer(session_id: &str, probe: Probe) {
        PROBES
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(session_id.to_string(), probe);
    }

    /// Drops the probe of `session_id` if it never ran (the CLI never had an answer).
    pub(crate) fn forget(session_id: &str) {
        PROBES
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(session_id);
    }

    pub(super) async fn cli_has_answer(session_id: &str) {
        let probe = PROBES
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(session_id);
        if let Some(probe) = probe {
            probe().await;
        }
    }
}

/// The CLI asks the identical call again as soon as it reads the answer of the first one:
/// a grant the user gave (`session`, `always`) covers it from that moment, not only once
/// the rest of the answer (the lasting rule's write) is done (FLAKY-PARITE-ALWAYS).
#[cfg(test)]
#[path = "manager/tests/grant_before_the_cli_tests.rs"]
mod grant_before_the_cli_tests;

/// The backend's own answer to a request a session grant covers is on the tool clock
/// like an answer of the user (Claude Code engine).
#[cfg(test)]
#[path = "manager/tests/granted_answer_clock_tests.rs"]
mod granted_answer_clock_tests;
