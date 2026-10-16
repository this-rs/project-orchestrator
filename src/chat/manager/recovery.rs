//! Agent turns cut by a restart of the server.
//!
//! A session of the agent engine lives in the memory of the server: its provider process, its
//! stream, its turn. A restart ends all of it, and nothing used to notice: the session stayed
//! "streaming" in the interface until the user typed again. The events of a session are
//! persisted, so a cut turn is READ from them: the last meaningful event is neither a `result`
//! nor an `error` (nor a deliberate `session_closed`).
//!
//! What it does, in two steps:
//!
//! 1. [`ChatManager::scan_interrupted_turns`] (fast, at boot): for each recent session, judges
//!    its tail ([`judge`]). A turn that cannot be continued by itself (it was waiting for an
//!    unanswered question or approval) is CLOSED with an `error` event, so the interface shows
//!    it and the next boot does not look at it again. So is every cut turn recovery does not
//!    resume: Claude Code CLI sessions, orchestrator children (the plan runner, pipeline,
//!    gate-retry and delegation relaunch the TASK in a new session), turns cut more than
//!    [`RECOVERY_WINDOW`] ago. A turn that can be continued is queued, with a context
//!    reconstructed from the stored events so the model knows what it was doing.
//! 2. [`ChatManager::drive_recovery`] (in the background): reopens each queued session and tells
//!    the model what happened. A session whose provider reads its key from the vault
//!    (`vault:<name>`) waits while the vault is locked; the others resume at once. Each attempt
//!    holds the session's resume lock, shared with a person's resume: a session somebody
//!    brought back first is skipped, never written into. The startup never waits for a person;
//!    what still waits when the wait ends is closed with an `error` event.
//!
//! A session cut again and again does not resume for ever: past [`MAX_AUTO_RESUMES`] it is
//! closed with an `error` event.
//!
//! # User identity
//!
//! `ChatSessionNode` does not persist the owner's JWT claims (a schema limitation). Recovery
//! reopens with `user_claims: None`, which is accepted by `authorize_provider_use`. The PO MCP
//! server then gets no session token: when the server signs its tokens, the resumed session
//! says it lacks the PO tools (`degraded_features`) instead of losing them silently. Replaying
//! the owner's identity requires storing it on the session node (a future schema migration).

use super::*;

/// A turn whose last event is older than this is closed, not resumed: a turn cut an hour ago
/// is not one somebody is still waiting for. Judged on the actual last event timestamp
/// (`updated_at` is not refreshed during a running turn); `updated_at` older than twice the
/// window only shortens the tail read.
pub(crate) const RECOVERY_WINDOW: chrono::Duration = chrono::Duration::hours(1);

/// How many automatic resumes one session gets before it is closed instead.
pub(crate) const MAX_AUTO_RESUMES: usize = 2;

/// How many of the last events are read to judge a session and count prior resumes.
/// Large enough to span several resume cycles (each adds a few events) and to capture
/// the full context of the cut turn.
const TAIL: i64 = 200;

/// Starts every resume message, and is how a later boot counts the resumes of a session.
pub(crate) const RESUME_MARKER: &str = "[PO restart recovery]";

/// What the tail of a session says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// The last turn ended (or the session never ran): nothing to do.
    Settled,
    /// The turn was cut and can be continued. `dangling_tool` is the tool whose call has no
    /// result, when the cut fell there.
    Resume { dangling_tool: Option<String> },
    /// The turn was cut and cannot be continued by itself: close it, with this reason.
    Close(&'static str),
}

/// Events that say nothing about whether a turn ended: timing, hints, bookkeeping, and
/// everything that can be written on an IDLE session after its turn's `result` (review of
/// #711, finding 1). When one of them does start a turn (a background output given to the
/// model), the turn's own events follow it and decide; a cut right after it is judged on
/// what came before, which errs on the safe side (left as it is rather than closed).
fn is_bookkeeping(event_type: &str) -> bool {
    matches!(
        event_type,
        "tool_timing"
            | "system_hint"
            | "tool_use_input_resolved"
            | "permission_decision"
            // Settings changed on an idle session, and the announce of a (re)opened one.
            | "permission_mode_changed"
            | "model_changed"
            | "auto_continue_state_changed"
            | "system_init"
            // Out of turn: the output of a background command or monitor (both engines
            // store it after the turn ended: `oob_listener`, `pump_out_of_band`), the
            // cancel of a background task, the state of the task list and the queue.
            | "background_output"
            | "tool_cancelled"
            | "tools_cancelled"
            | "active_tasks_update"
            | "pending_queue"
            | "secret_request_resolved"
            | "conversation_relayed"
            // Transient or announcing events (no turn content of their own).
            | "stream_delta"
            | "streaming_status"
            | "retrying"
            | "auto_continue"
            | "compaction_started"
            | "compaction_recovery"
            | "compact_boundary"
            | "refs_resolved"
            // The citations of the answer: announced after the turn's own result.
            | "cites_resolved"
            // The agent engine stores a background task's notification out of turn as a
            // `workflow` (`task_notification`, follow-up review of #711).
            | "workflow"
    )
}

/// How many times this session has already been automatically resumed, counting only the
/// resumes that belong to the CURRENT turn (after the last real user message). With TAIL=200
/// earlier turns' resume markers would otherwise accumulate and trigger false closes. A
/// marker followed by a `session_closed` before the next marker is an attempt of the
/// recovery that failed and was tried again (review of #711, finding 2), not a restart.
fn count_resumes(tail: &[ChatEventRecord]) -> usize {
    // Find the start of the current turn: the last user_message that is NOT a resume marker.
    let turn_start = tail
        .iter()
        .rposition(|e| e.event_type == "user_message" && !e.data.contains(RESUME_MARKER))
        .map(|i| i + 1)
        .unwrap_or(0);
    let mut resumes = 0;
    let mut open_marker = false;
    for event in &tail[turn_start..] {
        if event.event_type == "user_message" && event.data.contains(RESUME_MARKER) {
            if open_marker {
                resumes += 1;
            }
            open_marker = true;
        } else if event.event_type == "session_closed" {
            open_marker = false;
        }
    }
    resumes + usize::from(open_marker)
}

/// Whether the `spawned_by` JSON string identifies a session whose work is relaunched by
/// whoever owns it: a plan runner, the pipeline engine, a gate-retry, a delegation — or the
/// `Agent` tool call of a parent chat session. Those sessions are relaunched by their owner
/// at boot; recovery leaving them alone avoids a double-resume.
///
/// A `SpawnedBy::Conversation` child carrying a `tool_use_id` is a sub-agent hosted by the
/// `Agent` tool (P18, `chat::subagents`): its parent's own resume re-issues that call, so
/// resuming the child too would run the same work twice (two agents editing the same files),
/// its answer would reach nobody, and it would hold one of the parent's
/// [`MAX_LIVE_CHILDREN`](crate::chat::envelope::MAX_LIVE_CHILDREN) slots for nothing (P18b,
/// fix 3). A `Conversation` child WITHOUT a `tool_use_id` was opened by
/// `chat(send_message)`: nothing re-issues it, so it is resumed like any other session.
/// `SpawnedBy::Trigger` is not owned either: external triggers do not re-fire after a
/// restart.
fn is_orchestrator_child(spawned_by: &str) -> bool {
    is_agent_tool_child(spawned_by) || is_owned_by_a_run(spawned_by)
}

/// A sub-agent of the `Agent` tool: a `SpawnedBy::Conversation` carrying a `tool_use_id`.
fn is_agent_tool_child(spawned_by: &str) -> bool {
    matches!(
        crate::chat::types::SpawnedBy::from_json_str(spawned_by),
        Some(crate::chat::types::SpawnedBy::Conversation {
            tool_use_id: Some(_),
            ..
        })
    )
}

/// A session a run drives: runner, pipeline, gate retry, delegation.
fn is_owned_by_a_run(spawned_by: &str) -> bool {
    use crate::chat::types::SpawnedBy;
    matches!(
        SpawnedBy::from_json_str(spawned_by),
        Some(
            SpawnedBy::Runner { .. }
                | SpawnedBy::Pipeline { .. }
                | SpawnedBy::Gate { .. }
                | SpawnedBy::Delegation { .. }
        )
    )
}

/// Judges the last events of a session (oldest first).
pub(crate) fn judge(tail: &[ChatEventRecord]) -> Verdict {
    let Some(last) = tail.iter().rev().find(|e| !is_bookkeeping(&e.event_type)) else {
        return Verdict::Settled;
    };
    match last.event_type.as_str() {
        // `session_closed`: the session was closed on purpose (a turn running then was
        // stopped by it, not cut by a restart); `session_error`: it ended in error.
        "result" | "error" | "session_closed" | "session_error" => Verdict::Settled,

        "ask_user_question" => Verdict::Close(
            "The server restarted while this session waited for an answer; ask again.",
        ),

        "permission_request" => {
            // If a permission_decision follows the request (even though it is bookkeeping and
            // thus skipped by the last-event search), the user already answered: the tool was
            // approved and in flight when the server stopped. Resume rather than close.
            //
            // Match by decision id == request id (not just position), and require allow:true so
            // a denial is treated as "not answered" and the session is closed instead.
            let request_id = extract_json_str(&last.data, "id");
            let answered = request_id.as_deref().is_some_and(|rid| {
                tail.iter().any(|e| {
                    e.seq > last.seq
                        && e.event_type == "permission_decision"
                        && extract_json_str(&e.data, "id").as_deref() == Some(rid)
                        && extract_json_bool(&e.data, "allow").unwrap_or(false)
                })
            });
            if answered {
                if count_resumes(tail) >= MAX_AUTO_RESUMES {
                    return Verdict::Close(
                        "The server restarted again while this turn was running; it was not resumed again.",
                    );
                }
                // The approved tool was in flight when the server stopped: tell the model its
                // result is unknown (it may have run fully, partly, or not at all).
                let dangling_tool = extract_json_str(&last.data, "tool");
                Verdict::Resume { dangling_tool }
            } else {
                Verdict::Close(
                    "The server restarted while this session waited for a permission answer; ask again.",
                )
            }
        }

        other => {
            if count_resumes(tail) >= MAX_AUTO_RESUMES {
                return Verdict::Close(
                    "The server restarted again while this turn was running; it was not resumed again.",
                );
            }
            let dangling_tool = (other == "tool_use")
                .then(|| tool_name(&last.data))
                .flatten();
            Verdict::Resume { dangling_tool }
        }
    }
}

fn tool_name(data: &str) -> Option<String> {
    extract_json_str(data, "tool")
}

/// Extract a string field from a JSON blob stored in `ChatEventRecord.data`.
fn extract_json_str(data: &str, field: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(data)
        .ok()?
        .get(field)?
        .as_str()
        .map(str::to_string)
}

/// Extract a bool field from a JSON blob stored in `ChatEventRecord.data`.
fn extract_json_bool(data: &str, field: &str) -> Option<bool> {
    serde_json::from_str::<serde_json::Value>(data)
        .ok()?
        .get(field)?
        .as_bool()
}

/// Rebuilds a readable summary of the cut turn from its stored events, so the model can
/// continue without re-running what was already done.
///
/// The native provider's transcript is saved only at the end of a completed turn; after a
/// restart, its resume token points to the last COMPLETED turn. The model therefore has no
/// in-context memory of the cut turn's work. Injecting this summary into the resume message
/// bridges that gap.
pub(crate) fn reconstruct_cut_turn(tail: &[ChatEventRecord]) -> String {
    // Find the start of the cut turn: the last user_message that is NOT a resume marker.
    let turn_start = tail
        .iter()
        .rposition(|e| e.event_type == "user_message" && !e.data.contains(RESUME_MARKER));
    let Some(start_idx) = turn_start else {
        return String::new();
    };
    // Map tool_use id → index in `lines` so results are matched by id, not position.
    // Parallel tool calls (two tool_use events before either result) would be misattributed
    // with a positional approach; id-based matching is correct for any ordering.
    let mut tool_call_lines: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut lines: Vec<String> = Vec::new();
    for event in &tail[start_idx..] {
        match event.event_type.as_str() {
            "user_message" if !event.data.contains(RESUME_MARKER) => {
                let content = serde_json::from_str::<serde_json::Value>(&event.data)
                    .ok()
                    .and_then(|v| {
                        v.get("content")
                            .and_then(|c| c.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "(message)".to_string());
                lines.push(format!("User: {}", truncate_str(&content, 300)));
            }
            "tool_use" => {
                let v = serde_json::from_str::<serde_json::Value>(&event.data).ok();
                let id = v
                    .as_ref()
                    .and_then(|v| v.get("id"))
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = v
                    .as_ref()
                    .and_then(|v| v.get("tool"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("unknown");
                let input_preview = v
                    .as_ref()
                    .and_then(|v| v.get("input"))
                    .map(|i| truncate_str(&i.to_string(), 60))
                    .unwrap_or_default();
                let idx = lines.len();
                lines.push(format!("Called: {name}({input_preview})"));
                if !id.is_empty() {
                    tool_call_lines.insert(id, idx);
                }
            }
            "tool_result" => {
                // Match the result to its call by id (same id as the originating tool_use).
                let id = extract_json_str(&event.data, "id");
                if let Some(idx) = id.as_deref().and_then(|id| tool_call_lines.get(id)) {
                    lines[*idx].push_str(" → result received");
                }
            }
            _ => {}
        }
    }
    lines.join("\n")
}

fn truncate_str(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// What the resumed model is told.
///
/// `cut_turn_context` is the output of [`reconstruct_cut_turn`]: an empty string when no
/// events from the cut turn are stored (e.g. the turn was cut before the first tool call).
pub(crate) fn resume_message(dangling_tool: Option<&str>, cut_turn_context: &str) -> String {
    let tool = match dangling_tool {
        Some(tool) => format!(
            " Your last tool call ({tool}) has no result: it may have run fully, partly or not \
             at all. Check the state before repeating it."
        ),
        None => String::new(),
    };
    let context = if cut_turn_context.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nEvents saved before the restart (for context; do not re-run them):\n{cut_turn_context}"
        )
    };
    format!(
        "{RESUME_MARKER} The server restarted while you were working, so your turn was cut.\
         {context}\n\
         Continue from where you were; do not start over, and do not repeat what is already done.{tool}"
    )
}

/// A session found cut and queued for resume.
#[derive(Debug, Clone)]
pub(crate) struct PendingResume {
    pub node: ChatSessionNode,
    pub message: String,
    /// The last event seen when it was queued (or after this recovery's own failed
    /// attempt): anything newer means somebody else acted.
    pub seen_seq: i64,
    /// Attempts that failed with a transient provider error so far.
    pub failures: u32,
}

/// How many times a resume that failed with a transient provider error (provider not
/// ready yet at boot, overloaded, unreachable, timed out) is tried again before the turn
/// is closed (review of #697, finding 8c).
pub(crate) const MAX_TRANSIENT_RETRIES: u32 = 3;

/// Why a cut turn of a session recovery does not resume was closed.
const CLOSED_LEFT_TO_OWNER: &str = "The server restarted while this turn was running. This session is driven by its owner (plan run, pipeline, delegation, or the `Agent` call of a parent session), which relaunches the work itself: it is not resumed here.";
const CLOSED_MIRROR: &str = "The server restarted while the Claude Code sub-agent this conversation mirrors was running. A mirror is read-only and is never resumed: its parent conversation shows what happened.";
const CLOSED_LEGACY: &str = "The server restarted while this turn was running, and this session is not resumed automatically: send your message again to continue.";
const CLOSED_CLAUDE_CODE: &str = "The server restarted while this Claude Code turn was running. It is not resumed without you (it could run tools unattended): send your message again to continue.";
const CLOSED_STALE: &str = "The server restarted while this turn was running, too long ago to resume it automatically: send your message again to continue.";
const CLOSED_ABANDONED: &str = "The server restarted while this turn was running, and the vault stayed locked, so it could not be resumed: send your message again.";
const CLOSED_FAILED: &str = "The server restarted while this turn was running, and the session could not be resumed: send your message again.";

/// How many of the last events are read for a session that is closed rather than resumed
/// when its turn was cut (enough to find its last meaningful event).
const SHORT_TAIL: i64 = 20;

/// A session whose last activity is older than this is not touched at all, cut or not: the
/// older a thread, the less its "streaming" look matters and the more a wrong judgement of
/// its tail would cost (review of #711, finding 1: rather left streaming than falsely
/// closed). It also bounds what one boot reads and writes.
pub(crate) const CLOSE_HORIZON: chrono::Duration = chrono::Duration::days(7);

/// What the scan found.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub scanned: usize,
    /// Cut turns closed with an `error` event: waiting for a person, resumed too often, or
    /// in a session recovery does not resume (Claude Code CLI, orchestrator child, out of
    /// the window).
    pub closed: usize,
}

/// How a recovery ended.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    pub resumed: usize,
    /// Left alone because the session moved on by itself (the user wrote, or another
    /// resume brought it back): seen under the session's resume lock, nothing written.
    pub skipped: usize,
    /// Closed with an `error` event because the resume failed for good.
    pub failed: usize,
    /// Attempts that failed with a transient error and were tried again.
    pub retried: usize,
    /// Still waiting for the vault when the wait ended: closed with an `error` event.
    pub abandoned: usize,
}

/// What one attempt at a queued session did.
enum Attempt {
    Resumed,
    Skipped,
    /// The vault is locked and the session needs it: later.
    Wait,
    /// A transient failure: later, counted.
    Retry,
    Failed,
}

impl ChatManager {
    /// Reads the recent sessions and queues the cut turns that can be resumed (step 1).
    ///
    /// Every cut turn recovery does not resume is CLOSED with an `error` event, so it no
    /// longer shows "streaming": the sessions of the Claude Code CLI (nothing reopens them
    /// at boot), the children of an orchestrator (it relaunches the TASK, in a new session),
    /// and turns cut longer than [`RECOVERY_WINDOW`] ago. A session opened after the scan
    /// began, or live on this server (a run relaunched meanwhile), is not looked at.
    pub(crate) async fn scan_interrupted_turns(&self) -> Result<(Vec<PendingResume>, ScanReport)> {
        let (sessions, _) = self
            .graph
            .list_chat_sessions(None, None, 200, 0, true)
            .await?;
        let started = chrono::Utc::now();
        let since = started - RECOVERY_WINDOW;
        let mut pending = Vec::new();
        let mut report = ScanReport::default();
        for node in sessions {
            let sid = node.id.to_string();
            if node.created_at >= started || self.is_session_active(&sid).await {
                continue;
            }
            // `updated_at` is set at least when a turn starts: older than the horizon, so is
            // the turn.
            if node.updated_at < started - CLOSE_HORIZON {
                continue;
            }
            // Not resumed here, closed when cut: a session of the historical engine (the
            // Claude Code CLI); a Claude Code session of the agent engine, unless this server
            // was told otherwise (M2: unattended, its CLI could run tools in `acceptEdits` or
            // `bypassPermissions` with nobody there; kept closed until decided explicitly);
            // an owner-managed one (runner, pipeline, gate-retry, delegation, or a sub-agent
            // of an `Agent` call) is relaunched by its owner. A `chat(send_message)`
            // sub-session and a trigger-spawned session have no owner: resumed.
            let left_alone = if node.mirror_of.is_some() {
                // A read-only mirror of a Claude Code sub-agent (P18c) has no process: never
                // resumed, only closed when cut.
                Some(CLOSED_MIRROR)
            } else if !node.opened_on_agent() {
                Some(CLOSED_LEGACY)
            } else if node.served_by_claude_code() && !self.recovery_resumes_claude_code {
                Some(CLOSED_CLAUDE_CODE)
            } else if node
                .spawned_by
                .as_deref()
                .is_some_and(is_orchestrator_child)
            {
                Some(CLOSED_LEFT_TO_OWNER)
            } else {
                None
            };
            // `updated_at` is not refreshed during a running turn: a session that looks
            // stale by it may still be resumable (fine filter below, on the last event).
            // Only a session that can be resumed needs its whole tail.
            let candidate = left_alone.is_none() && node.updated_at >= since - RECOVERY_WINDOW;
            let depth = if candidate { TAIL } else { SHORT_TAIL };
            report.scanned += 1;
            let latest = match self.graph.get_latest_chat_event_seq(node.id).await {
                Ok(seq) => seq,
                Err(e) => {
                    warn!("recovery: events of {} unreadable: {e}", node.id);
                    continue;
                }
            };
            let tail = match self
                .graph
                .get_chat_events(node.id, (latest - depth).max(0), depth)
                .await
            {
                Ok(tail) => tail,
                Err(e) => {
                    warn!("recovery: events of {} unreadable: {e}", node.id);
                    continue;
                }
            };
            let last_event_at = tail.last().map(|e| e.created_at).unwrap_or(node.updated_at);
            if last_event_at < started - CLOSE_HORIZON {
                continue;
            }
            match judge(&tail) {
                Verdict::Settled => {}
                Verdict::Close(reason) => {
                    self.close_cut_session(&node, reason).await;
                    report.closed += 1;
                }
                Verdict::Resume { dangling_tool } => {
                    let closed_because =
                        left_alone.or((last_event_at < since).then_some(CLOSED_STALE));
                    if let Some(reason) = closed_because {
                        self.close_cut_session(&node, reason).await;
                        report.closed += 1;
                        continue;
                    }
                    let cut_context = reconstruct_cut_turn(&tail);
                    pending.push(PendingResume {
                        message: resume_message(dangling_tool.as_deref(), &cut_context),
                        node,
                        seen_seq: latest,
                        failures: 0,
                    });
                }
            }
        }
        Ok((pending, report))
    }

    /// [`Self::close_cut_turn`] for the session `node`. A sub-agent of an `Agent` call first
    /// refuses the permission requests its thread leaves undecided, through the helper the
    /// live path runs before closing a child (`deny_waiting_permissions`): nobody will ever
    /// answer them (its parent re-issues the call in a new child), and since round 3 the
    /// child has an owner, in whose attention they would stay pending for 30 days (lot
    /// review of #721, finding 4).
    async fn close_cut_session(&self, node: &ChatSessionNode, reason: &str) {
        if node.spawned_by.as_deref().is_some_and(is_agent_tool_child) && node.mirror_of.is_none() {
            self.deny_waiting_permissions(&node.id.to_string()).await;
        }
        self.close_cut_turn(node.id, reason).await;
    }

    /// Ends a cut turn on the thread, so it shows and is not judged again. The event goes
    /// after the LAST stored event, read now (an attempt of this recovery may have written
    /// some since the scan), and is published like the events of a live session: to the
    /// other instances (NATS), and as a session update for the lists (review of #697,
    /// finding 8a). Never called on a session live here.
    async fn close_cut_turn(&self, session_id: Uuid, reason: &str) {
        let after_seq = match self.graph.get_latest_chat_event_seq(session_id).await {
            Ok(seq) => seq,
            Err(e) => {
                warn!("recovery: could not close {session_id}: {e}");
                return;
            }
        };
        let event = ChatEvent::Error {
            message: reason.to_string(),
            parent_tool_use_id: None,
            code: Some("restart_interrupted".to_string()),
            reason: None,
            index: None,
            // Not a permission refusal: it names no request (P11 field).
            request_id: None,
        };
        let record = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: after_seq + 1,
            event_type: event.event_type().to_string(),
            data: serde_json::to_string(&event).unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        if let Err(e) = self.graph.store_chat_events(session_id, vec![record]).await {
            warn!("recovery: could not close {session_id}: {e}");
            return;
        }
        let sid = session_id.to_string();
        if let Some(nats) = &self.nats {
            nats.publish_chat_event(&sid, event);
        }
        if let Some(emitter) = &self.event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &sid,
                serde_json::json!({ "is_streaming": false }),
                None,
            );
        }
    }

    /// Reopens the queued sessions (step 2), for at most `max_wait`. `poll` is the pause
    /// between two looks.
    ///
    /// A session whose provider reads its key from the vault waits while the vault is
    /// locked; the others resume at once (review of #697, finding 7). A provider that still
    /// answers `credentials_locked` (a key read elsewhere) puts its session back to wait. A
    /// transient provider error is tried again, [`MAX_TRANSIENT_RETRIES`] times. What still
    /// waits at the deadline is closed with an `error` event.
    ///
    /// Note: the session is reopened with `user_claims: None` because the session owner's
    /// identity is not persisted (see module doc); the session says it lacks the PO tools
    /// when the server signs its tokens.
    pub(crate) async fn drive_recovery(
        &self,
        mut pending: Vec<PendingResume>,
        poll: Duration,
        max_wait: Duration,
    ) -> RecoveryReport {
        let mut report = RecoveryReport::default();
        let deadline = tokio::time::Instant::now() + max_wait;
        while !pending.is_empty() {
            let locked = self
                .vault
                .as_ref()
                .is_some_and(|v| !v.is_unlocked(chrono::Utc::now()));
            let mut waiting = Vec::new();
            for mut item in pending {
                match self.recover_one(&mut item, locked).await {
                    Attempt::Resumed => {
                        info!("recovery: session {} resumed after a restart", item.node.id);
                        report.resumed += 1;
                    }
                    Attempt::Skipped => report.skipped += 1,
                    Attempt::Wait => waiting.push(item),
                    Attempt::Retry => {
                        report.retried += 1;
                        waiting.push(item);
                    }
                    Attempt::Failed => report.failed += 1,
                }
            }
            pending = waiting;
            if pending.is_empty() {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                for item in &pending {
                    if self.abandon(item).await {
                        report.abandoned += 1;
                    } else {
                        report.skipped += 1;
                    }
                }
                break;
            }
            tokio::time::sleep(poll).await;
        }
        report
    }

    /// One attempt at a queued session.
    ///
    /// The look and the resume hold the session's resume lock, the lock a person's resume
    /// takes (`resume_agent_session`): a person who resumed the session first is SEEN here
    /// (a live handle, or newer events) and the session is skipped — its "continue" never
    /// lands in the conversation the person took back, and nothing is written into it; a
    /// person who comes second finds the session recovery brought back and talks to it
    /// (follow-up review of #697, R1).
    async fn recover_one(&self, item: &mut PendingResume, vault_locked: bool) -> Attempt {
        let sid = item.node.id.to_string();
        let _resuming = self.resume_locks.lock(&sid).await;
        if self.moved_on(item).await {
            return Attempt::Skipped;
        }
        if vault_locked && self.needs_vault(&item.node).await {
            return Attempt::Wait;
        }
        // The checks of a person's resume (engine still switched on, project slug).
        let mut node = item.node.clone();
        let result = match self.ready_for_resume(&mut node).await {
            Ok(()) => self.reopen_agent_session(&node, &item.message, None).await,
            Err(e) => Err(e),
        };
        let Err(error) = result else {
            return Attempt::Resumed;
        };
        // An attempt that got as far as adopting the session (its first message refused)
        // ends it: a live session is never written into from here, and what this recovery
        // writes next goes after the events the attempt stored (review of #697, finding 9).
        if self.agent_runtime.owns(&sid).await {
            let _ = self.close_session(&sid).await;
        }
        // `credentials_locked` waits only while the vault IS locked. With the vault open it
        // says the key could not be read for another reason (vault unavailable, throttled,
        // storage error: `credentials.rs`): counted like any transient error, not retried
        // every `poll` until the deadline (review of #711, finding 3).
        let locked_out = credentials_locked(&error)
            && self
                .vault
                .as_ref()
                .is_some_and(|v| !v.is_unlocked(chrono::Utc::now()));
        let retryable = transient(&error) || credentials_locked(&error);
        if locked_out || retryable {
            if let Ok(seq) = self.graph.get_latest_chat_event_seq(item.node.id).await {
                item.seen_seq = seq;
            }
        }
        if locked_out {
            return Attempt::Wait;
        }
        if retryable && item.failures < MAX_TRANSIENT_RETRIES {
            item.failures += 1;
            warn!("recovery: session {sid} not resumed yet ({error}), trying again");
            return Attempt::Retry;
        }
        warn!("recovery: session {sid} could not be resumed: {error}");
        self.close_cut_turn(item.node.id, CLOSED_FAILED).await;
        Attempt::Failed
    }

    /// Closes a session still waiting when the recovery gives up, unless it moved on
    /// meanwhile (`false`). Under the resume lock, as an attempt.
    async fn abandon(&self, item: &PendingResume) -> bool {
        let sid = item.node.id.to_string();
        let _resuming = self.resume_locks.lock(&sid).await;
        if self.moved_on(item).await {
            return false;
        }
        self.close_cut_turn(item.node.id, CLOSED_ABANDONED).await;
        true
    }

    /// Whether somebody acted on the session since it was queued: it is live here, or it
    /// has newer events.
    async fn moved_on(&self, item: &PendingResume) -> bool {
        self.is_session_active(&item.node.id.to_string()).await
            || self
                .graph
                .get_latest_chat_event_seq(item.node.id)
                .await
                .map(|seq| seq != item.seen_seq)
                .unwrap_or(false)
    }

    /// Whether the session's provider reads its key from the vault: a registered instance
    /// whose credential is a `vault:` reference (a remote Claude Code always is). A built-in
    /// provider has no stored instance and no key of its own there. Unreadable: assumed so.
    async fn needs_vault(&self, node: &ChatSessionNode) -> bool {
        let provider_id = node
            .provider_id
            .as_deref()
            .unwrap_or(crate::chat::provider::resolver::CLAUDE_CODE);
        match crate::chat::provider::store::instance(self.graph.as_ref(), provider_id).await {
            Ok(Some(record)) => {
                record.credential_ref.starts_with("vault:")
                    || record.kind == crate::chat::provider::settings::KIND_CLAUDE_CODE_REMOTE
            }
            Ok(None) => false,
            Err(_) => true,
        }
    }
}

/// A provider error worth trying again (`ProviderError::retryable`).
fn transient(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<nexus_claude::agent::ProviderError>()
            .is_some_and(|p| p.retryable())
    })
}

/// The provider's key lives in a vault that is locked: try again once it is not.
fn credentials_locked(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<nexus_claude::agent::ProviderError>()
            .is_some_and(|p| matches!(p, nexus_claude::agent::ProviderError::CredentialsLocked))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(seq: i64, event_type: &str, data: &str) -> ChatEventRecord {
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            seq,
            event_type: event_type.into(),
            data: data.into(),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn an_empty_or_finished_tail_is_settled() {
        assert_eq!(judge(&[]), Verdict::Settled);
        let done = [ev(1, "user_message", "{}"), ev(2, "result", "{}")];
        assert_eq!(judge(&done), Verdict::Settled);
        let failed = [ev(1, "user_message", "{}"), ev(2, "error", "{}")];
        assert_eq!(judge(&failed), Verdict::Settled);
    }

    #[test]
    fn bookkeeping_after_the_result_does_not_reopen_a_turn() {
        let tail = [
            ev(1, "user_message", "{}"),
            ev(2, "result", "{}"),
            ev(3, "tool_timing", "{}"),
        ];
        assert_eq!(judge(&tail), Verdict::Settled);
    }

    #[test]
    fn a_closed_session_or_a_settings_change_after_the_result_is_settled() {
        // Settings changed on an idle session, or the announce of a reopened one, do not
        // make a finished turn look cut (closing a CLI session or an out-of-window one at
        // boot would otherwise write a false "restart" error into it).
        for after in [
            "model_changed",
            "permission_mode_changed",
            "auto_continue_state_changed",
            "system_init",
        ] {
            let tail = [
                ev(1, "user_message", "{}"),
                ev(2, "result", "{}"),
                ev(3, after, "{}"),
            ];
            assert_eq!(judge(&tail), Verdict::Settled, "{after}");
        }
        // A session closed on purpose, or ended in a session error, was not cut by a
        // restart, whatever ran before.
        for last in ["session_closed", "session_error"] {
            let tail = [
                ev(1, "user_message", "{}"),
                ev(2, "assistant_text", "{}"),
                ev(3, last, "{}"),
            ];
            assert_eq!(judge(&tail), Verdict::Settled, "{last}");
        }
    }

    #[test]
    fn output_written_on_an_idle_session_does_not_reopen_its_turn() {
        // Review of #711, finding 1: both engines store a background command's output
        // AFTER the turn's result; so are the cancel of a background task and the like.
        for after in [
            "background_output",
            "tool_cancelled",
            "tools_cancelled",
            "active_tasks_update",
            "pending_queue",
            "secret_request_resolved",
            "conversation_relayed",
            "retrying",
            "compact_boundary",
            "workflow",
        ] {
            let tail = [
                ev(1, "user_message", "{}"),
                ev(2, "result", "{}"),
                ev(3, after, "{}"),
            ];
            assert_eq!(judge(&tail), Verdict::Settled, "{after}");
        }
        // A turn that a background output started is judged on its own events.
        let started = [
            ev(1, "result", "{}"),
            ev(2, "background_output", "{}"),
            ev(3, "assistant_text", "{}"),
        ];
        assert!(matches!(judge(&started), Verdict::Resume { .. }));
    }

    #[test]
    fn a_failed_attempt_of_the_recovery_is_not_counted_as_a_restart() {
        // Review of #711, finding 2: an attempt that adopted the session and failed left
        // its marker, then `session_closed`; the attempt tried again is the one resume.
        let resume = resume_message(None, "");
        let marker = serde_json::json!({ "type": "user_message", "content": resume }).to_string();
        let retried = [
            ev(1, "user_message", r#"{"content":"go"}"#),
            ev(2, "user_message", &marker),
            ev(3, "session_closed", "{}"),
            ev(4, "user_message", &marker),
            ev(5, "assistant_text", "{}"),
        ];
        assert_eq!(count_resumes(&retried), 1);
        assert!(matches!(judge(&retried), Verdict::Resume { .. }));
        // Two real resumes still close.
        let twice = [
            ev(1, "user_message", r#"{"content":"go"}"#),
            ev(2, "user_message", &marker),
            ev(3, "session_closed", "{}"),
            ev(4, "user_message", &marker),
            ev(5, "assistant_text", "{}"),
            ev(6, "user_message", &marker),
            ev(7, "assistant_text", "{}"),
        ];
        assert_eq!(count_resumes(&twice), 2);
        assert!(matches!(judge(&twice), Verdict::Close(_)));
    }

    #[test]
    fn a_turn_cut_in_text_or_after_a_user_message_resumes() {
        for last in ["user_message", "assistant_text", "thinking", "tool_result"] {
            let tail = [ev(1, "user_message", "{}"), ev(2, last, "{}")];
            assert_eq!(
                judge(&tail),
                Verdict::Resume {
                    dangling_tool: None
                },
                "cut after {last}"
            );
        }
    }

    #[test]
    fn a_cut_between_a_tool_call_and_its_result_names_the_tool() {
        let tail = [
            ev(1, "user_message", "{}"),
            ev(
                2,
                "tool_use",
                r#"{"type":"tool_use","id":"t1","tool":"Bash","input":{}}"#,
            ),
            ev(3, "tool_timing", "{}"),
        ];
        assert_eq!(
            judge(&tail),
            Verdict::Resume {
                dangling_tool: Some("Bash".into())
            }
        );
        assert!(resume_message(Some("Bash"), "").contains("Bash"));
        assert!(resume_message(None, "").starts_with(RESUME_MARKER));
    }

    #[test]
    fn a_turn_waiting_for_an_unanswered_question_is_closed_not_resumed() {
        let last = "ask_user_question";
        let tail = [ev(1, "user_message", "{}"), ev(2, last, "{}")];
        assert!(matches!(judge(&tail), Verdict::Close(_)), "{last}");
    }

    #[test]
    fn an_unanswered_permission_request_is_closed() {
        let tail = [
            ev(1, "user_message", "{}"),
            ev(2, "permission_request", "{}"),
        ];
        assert!(matches!(judge(&tail), Verdict::Close(_)));
    }

    #[test]
    fn an_approved_permission_request_is_resumed_not_closed() {
        // permission_decision (bookkeeping) follows permission_request: the tool was
        // approved and running when the restart happened — resume, not close.
        // IDs must match, and allow must be true; the tool name from the request becomes
        // dangling_tool so the model is told it may have run fully/partly/not at all.
        let tail = [
            ev(1, "user_message", "{}"),
            ev(
                2,
                "permission_request",
                r#"{"id":"p1","tool":"Bash","input":{"cmd":"ls"}}"#,
            ),
            ev(3, "permission_decision", r#"{"id":"p1","allow":true}"#),
        ];
        assert_eq!(
            judge(&tail),
            Verdict::Resume {
                dangling_tool: Some("Bash".into())
            }
        );
    }

    #[test]
    fn a_denied_permission_request_is_closed() {
        // A denial (allow: false) must close the session just like no decision at all.
        let tail = [
            ev(1, "user_message", "{}"),
            ev(
                2,
                "permission_request",
                r#"{"id":"p1","tool":"Bash","input":{}}"#,
            ),
            ev(3, "permission_decision", r#"{"id":"p1","allow":false}"#),
        ];
        assert!(matches!(judge(&tail), Verdict::Close(_)));
    }

    #[test]
    fn a_mismatched_decision_id_is_treated_as_unanswered() {
        // decision id does not match request id → treat as unanswered → Close.
        let tail = [
            ev(1, "user_message", "{}"),
            ev(
                2,
                "permission_request",
                r#"{"id":"p1","tool":"Bash","input":{}}"#,
            ),
            ev(3, "permission_decision", r#"{"id":"p2","allow":true}"#),
        ];
        assert!(matches!(judge(&tail), Verdict::Close(_)));
    }

    #[test]
    fn a_session_cut_again_and_again_is_closed() {
        let resume = resume_message(None, "");
        let line = serde_json::json!({ "type": "user_message", "content": resume }).to_string();
        let tail = [
            ev(1, "user_message", &line),
            ev(2, "assistant_text", "{}"),
            ev(3, "user_message", &line),
            ev(4, "assistant_text", "{}"),
        ];
        assert!(matches!(judge(&tail), Verdict::Close(_)));
        let once = [ev(1, "user_message", &line), ev(2, "assistant_text", "{}")];
        assert!(matches!(judge(&once), Verdict::Resume { .. }));
    }

    #[test]
    fn orchestrator_child_sessions_are_identified_by_spawned_by() {
        // SpawnedBy uses #[serde(tag = "type", rename_all = "snake_case")]:
        // internally tagged, snake_case discriminant.
        let runner_id = Uuid::nil().to_string();
        assert!(is_orchestrator_child(&format!(
            r#"{{"type":"runner","run_id":"{runner_id}","task_id":"{runner_id}"}}"#
        )));
        assert!(is_orchestrator_child(&format!(
            r#"{{"type":"pipeline","run_id":"{runner_id}","task_id":"{runner_id}","wave":1}}"#
        )));
        assert!(is_orchestrator_child(&format!(
            r#"{{"type":"gate","run_id":"{runner_id}","task_id":"{runner_id}","gate_name":"q","attempt":1}}"#
        )));
        assert!(is_orchestrator_child(&format!(
            r#"{{"type":"delegation","plan_id":"{runner_id}","task_id":"{runner_id}"}}"#
        )));
        // Conversation and Trigger are NOT orchestrator-managed → included in recovery.
        assert!(!is_orchestrator_child(&format!(
            r#"{{"type":"conversation","parent_session_id":"{runner_id}"}}"#
        )));
        assert!(!is_orchestrator_child(&format!(
            r#"{{"type":"trigger","trigger_id":"{runner_id}"}}"#
        )));
        assert!(!is_orchestrator_child("not json"));
        assert!(!is_orchestrator_child("{}"));
    }

    /// P18b, fix 3: a `conversation` child carrying a `tool_use_id` is a sub-agent of the
    /// `Agent` tool. Its parent re-issues the call when it is resumed, so the child must be
    /// left to its owner (closed), never resumed on its own — otherwise a SECOND child runs
    /// the same work and the old one holds one of the parent's four slots.
    #[test]
    fn an_agent_tool_child_is_left_to_its_parent_a_send_message_child_is_not() {
        let parent = Uuid::nil().to_string();
        assert!(is_orchestrator_child(&format!(
            r#"{{"type":"conversation","parent_session_id":"{parent}","tool_use_id":"toolu_1"}}"#
        )));
        // Written by `SpawnedBy` itself, so the test follows the serialisation.
        assert!(is_orchestrator_child(
            &crate::chat::types::SpawnedBy::Conversation {
                parent_session_id: Uuid::nil(),
                tool_use_id: Some("toolu_1".into()),
            }
            .to_json_string()
        ));
        assert!(!is_orchestrator_child(
            &crate::chat::types::SpawnedBy::Conversation {
                parent_session_id: Uuid::nil(),
                tool_use_id: None,
            }
            .to_json_string()
        ));
    }

    #[test]
    fn cut_turn_context_is_reconstructed_from_events() {
        let tail = [
            ev(1, "user_message", r#"{"content":"fix the bug"}"#),
            ev(
                2,
                "tool_use",
                r#"{"type":"tool_use","id":"t1","tool":"Read","input":{}}"#,
            ),
            ev(
                3,
                "tool_result",
                r#"{"type":"tool_result","id":"t1","result":"ok"}"#,
            ),
            ev(
                4,
                "tool_use",
                r#"{"type":"tool_use","id":"t2","tool":"Edit","input":{}}"#,
            ),
            // cut here — no tool_result for Edit
        ];
        let ctx = reconstruct_cut_turn(&tail);
        assert!(ctx.contains("fix the bug"), "user message in context");
        assert!(ctx.contains("Read"), "first tool in context");
        assert!(ctx.contains("result received"), "first tool has result");
        assert!(ctx.contains("Edit"), "second tool in context");
        // Edit was the last event (no tool_result follows): its line must not carry
        // "→ result received".
        let edit_line = ctx.lines().find(|l| l.contains("Edit")).unwrap();
        assert!(
            !edit_line.contains("result received"),
            "Edit line has no result annotation"
        );

        let msg = resume_message(Some("Edit"), &ctx);
        assert!(msg.starts_with(RESUME_MARKER));
        assert!(msg.contains("Edit"));
        assert!(msg.contains("fix the bug"));
    }

    #[test]
    fn cut_turn_context_is_empty_when_no_user_message_stored() {
        // No user_message before the cut: nothing to reconstruct.
        let tail = [ev(
            1,
            "tool_use",
            r#"{"type":"tool_use","id":"t1","tool":"Bash","input":{}}"#,
        )];
        assert_eq!(reconstruct_cut_turn(&tail), "");
        // resume_message still works with an empty context.
        assert!(resume_message(None, "").starts_with(RESUME_MARKER));
        assert!(!resume_message(None, "").contains("Events saved"));
        assert!(resume_message(None, "some context").contains("Events saved"));
    }
}
