//! Agent turns cut by a restart of the server.
//!
//! A session of the agent engine lives in the memory of the server: its provider process, its
//! stream, its turn. A restart ends all of it, and nothing used to notice: the session stayed
//! "streaming" in the interface until the user typed again. The events of a session are
//! persisted, so a cut turn is READ from them: the last event is neither a `result` nor an
//! `error`.
//!
//! What it does, in two steps:
//!
//! 1. [`ChatManager::scan_interrupted_turns`] (fast, at boot): for each recent agent session,
//!    judges its tail ([`judge`]). A turn that cannot be continued by itself (it was waiting for
//!    an unanswered question or approval) is CLOSED with an `error` event, so the interface shows
//!    it and the next boot does not look at it again. A turn that can be continued is queued, with
//!    a context reconstructed from the stored events so the model knows what it was doing.
//! 2. [`ChatManager::drive_recovery`] (in the background): reopens each queued session and tells
//!    the model what happened. ALL sessions wait for the vault to be unlocked before any resume
//!    starts (a global guard: `vault:<name>` credentials are unreadable while locked, and the
//!    guard is conservative — it blocks even sessions that do not need the vault). The startup
//!    never waits for a person.
//!
//! Skipped: plan-runner child sessions. The runner relaunches them on its own at boot; recovery
//! leaving them alone avoids a double-resume.
//!
//! A session cut again and again does not resume for ever: past [`MAX_AUTO_RESUMES`] it is
//! closed with an `error` event.
//!
//! # User identity
//!
//! `ChatSessionNode` does not persist the owner's JWT claims (a schema limitation). Recovery
//! calls `resume_agent_session` with `user_claims: None`, which is accepted by
//! `authorize_provider_use`. The PO MCP token is generated without a user sub, so tools that
//! gate on user identity will refuse. This is a known gap; fixing it requires storing the
//! owner's sub on the session node (a future schema migration).

use super::*;

/// Sessions untouched for longer than this before the restart are left alone: a turn cut an
/// hour ago is not one somebody is still waiting for. Used as both a coarse filter on
/// `updated_at` and a fine filter on the actual last event timestamp (since `updated_at` is
/// not refreshed during a running turn — a long turn would be filtered out by `updated_at`
/// alone).
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

/// Events that say nothing about whether a turn ended (timing, hints, bookkeeping).
fn is_bookkeeping(event_type: &str) -> bool {
    matches!(
        event_type,
        "tool_timing" | "system_hint" | "tool_use_input_resolved" | "permission_decision"
    )
}

/// How many times this session has already been automatically resumed, counting only the
/// resumes that belong to the CURRENT turn (after the last real user message). With TAIL=200
/// earlier turns' resume markers would otherwise accumulate and trigger false closes.
fn count_resumes(tail: &[ChatEventRecord]) -> usize {
    // Find the start of the current turn: the last user_message that is NOT a resume marker.
    let turn_start = tail
        .iter()
        .rposition(|e| e.event_type == "user_message" && !e.data.contains(RESUME_MARKER))
        .map(|i| i + 1)
        .unwrap_or(0);
    tail[turn_start..]
        .iter()
        .filter(|e| e.event_type == "user_message" && e.data.contains(RESUME_MARKER))
        .count()
}

/// Whether the `spawned_by` JSON string identifies a session managed by an orchestrator
/// (plan runner, pipeline engine, gate-retry, or delegation). Those sessions are relaunched
/// by their orchestrator at boot; recovery leaving them alone avoids a double-resume.
///
/// `SpawnedBy::Conversation` and `SpawnedBy::Trigger` are NOT orchestrator-managed:
/// conversation sub-sessions have no orchestrator, and external triggers do not
/// automatically re-fire after a restart.
fn is_orchestrator_child(spawned_by: &str) -> bool {
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
        "result" | "error" => Verdict::Settled,

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
    /// The last event seen when it was queued: anything newer means somebody else acted.
    pub seen_seq: i64,
}

/// What the scan found.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub scanned: usize,
    pub closed: usize,
}

/// How a recovery ended.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    pub resumed: usize,
    /// Left alone because the session moved on by itself (the user wrote, or it is live).
    pub skipped: usize,
    /// Closed with an `error` event because the resume failed for good.
    pub failed: usize,
    /// Still waiting for the vault when the wait ended.
    pub abandoned: usize,
}

impl ChatManager {
    /// Reads the recent agent sessions and queues the cut turns (step 1).
    pub(crate) async fn scan_interrupted_turns(&self) -> Result<(Vec<PendingResume>, ScanReport)> {
        let (sessions, _) = self
            .graph
            .list_chat_sessions(None, None, 200, 0, true)
            .await?;
        let since = chrono::Utc::now() - RECOVERY_WINDOW;
        let mut pending = Vec::new();
        let mut report = ScanReport::default();
        for node in sessions {
            // The agent engine only: a session without a capability snapshot is a Claude Code
            // CLI session, which the CLI itself resumes.
            if node.capabilities.is_none() {
                continue;
            }
            // Orchestrator-managed sessions (runner, pipeline, gate-retry, delegation) are
            // relaunched by their orchestrator at boot. Recovering them here too would
            // double-resume. Conversation sub-sessions and trigger-spawned sessions have no
            // orchestrator and are included.
            if node
                .spawned_by
                .as_deref()
                .is_some_and(is_orchestrator_child)
            {
                continue;
            }
            // Coarse time filter: skip sessions that have been inactive for more than
            // 2× RECOVERY_WINDOW. `updated_at` is not refreshed during a running turn
            // (only at turn start/end), so a long turn would look stale; the factor-of-2
            // margin prevents false exclusions. The fine filter below uses the actual last
            // event timestamp.
            if node.updated_at < since - RECOVERY_WINDOW {
                continue;
            }
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
                .get_chat_events(node.id, (latest - TAIL).max(0), TAIL)
                .await
            {
                Ok(tail) => tail,
                Err(e) => {
                    warn!("recovery: events of {} unreadable: {e}", node.id);
                    continue;
                }
            };
            // Fine time filter: `updated_at` is not refreshed during a turn, so use the
            // actual last event timestamp instead.
            let last_event_at = tail.last().map(|e| e.created_at).unwrap_or(node.updated_at);
            if last_event_at < since {
                continue;
            }
            match judge(&tail) {
                Verdict::Settled => {}
                Verdict::Close(reason) => {
                    self.close_cut_turn(node.id, latest, reason).await;
                    report.closed += 1;
                }
                Verdict::Resume { dangling_tool } => {
                    let cut_context = reconstruct_cut_turn(&tail);
                    pending.push(PendingResume {
                        message: resume_message(dangling_tool.as_deref(), &cut_context),
                        node,
                        seen_seq: latest,
                    });
                }
            }
        }
        Ok((pending, report))
    }

    /// Ends a cut turn on the thread, so it shows and is not judged again.
    async fn close_cut_turn(&self, session_id: Uuid, after_seq: i64, reason: &str) {
        let event = ChatEvent::Error {
            message: reason.to_string(),
            parent_tool_use_id: None,
            code: Some("restart_interrupted".to_string()),
            reason: None,
            index: None,
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
        }
    }

    /// Reopens the queued sessions (step 2), waiting while the vault is locked, for at most
    /// `max_wait`. `poll` is the pause between two looks.
    ///
    /// The vault guard is global: while the vault is locked, ALL pending sessions wait, even
    /// those whose provider may not need it. This is a conservative simplification — tracking
    /// vault-dependency per session requires a schema change (storing the owner's credential
    /// reference on the session node) and is deferred. Sessions that get a `credentials_locked`
    /// error from the provider (after an unlock that doesn't cover all providers) are also
    /// pushed back to `waiting`.
    ///
    /// Note: `resume_agent_session` is called with `user_claims: None` because the session
    /// owner's identity is not persisted (see module doc). PO tools gated on user identity will
    /// refuse; this is a known limitation pending a schema migration to store the owner sub.
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
            for item in pending {
                if locked {
                    waiting.push(item);
                    continue;
                }
                let sid = item.node.id.to_string();
                let moved_on = self.agent_runtime.get(&sid).await.is_some()
                    || self
                        .graph
                        .get_latest_chat_event_seq(item.node.id)
                        .await
                        .map(|seq| seq != item.seen_seq)
                        .unwrap_or(false);
                if moved_on {
                    report.skipped += 1;
                    continue;
                }
                match self
                    .resume_agent_session(&item.node, &item.message, None, None)
                    .await
                {
                    Ok(()) => {
                        info!("recovery: session {sid} resumed after a restart");
                        report.resumed += 1;
                    }
                    Err(e) if credentials_locked(&e) => waiting.push(item),
                    Err(e) => {
                        warn!("recovery: session {sid} could not be resumed: {e}");
                        self.close_cut_turn(
                            item.node.id,
                            item.seen_seq,
                            "The server restarted while this turn was running, and the session could not be resumed; send your message again.",
                        )
                        .await;
                        report.failed += 1;
                    }
                }
            }
            pending = waiting;
            if pending.is_empty() {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                report.abandoned = pending.len();
                break;
            }
            tokio::time::sleep(poll).await;
        }
        report
    }
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
