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
//!    a permission or an answer that the restart lost) is CLOSED with an `error` event, so the
//!    interface shows it and the next boot does not look at it again. A turn that can be
//!    continued is queued.
//! 2. [`ChatManager::drive_recovery`] (in the background): reopens each queued session and tells
//!    the model what happened. It waits while the vault is locked (a provider key is a
//!    `vault:<name>` reference, unreadable until the user types the passphrase): the startup
//!    never waits for a person.
//!
//! Never replayed: a tool call. The model is told that the last one may or may not have run, and
//! checks before it repeats anything.
//!
//! A session cut again and again does not resume for ever: past [`MAX_AUTO_RESUMES`] it is
//! closed with an `error` event.

use super::*;

/// Sessions untouched for longer than this before the restart are left alone: a turn cut an
/// hour ago is not one somebody is still waiting for.
pub(crate) const RECOVERY_WINDOW: chrono::Duration = chrono::Duration::hours(1);

/// How many automatic resumes one session gets before it is closed instead.
pub(crate) const MAX_AUTO_RESUMES: usize = 2;

/// How many of the last events are read to judge a session.
const TAIL: i64 = 40;

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

/// Judges the last events of a session (oldest first).
pub(crate) fn judge(tail: &[ChatEventRecord]) -> Verdict {
    let Some(last) = tail.iter().rev().find(|e| !is_bookkeeping(&e.event_type)) else {
        return Verdict::Settled;
    };
    match last.event_type.as_str() {
        "result" | "error" => Verdict::Settled,
        "permission_request" | "ask_user_question" => Verdict::Close(
            "The server restarted while this session waited for an answer; ask again.",
        ),
        other => {
            let resumes = tail
                .iter()
                .filter(|e| e.event_type == "user_message" && e.data.contains(RESUME_MARKER))
                .count();
            if resumes >= MAX_AUTO_RESUMES {
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
    serde_json::from_str::<serde_json::Value>(data)
        .ok()?
        .get("tool")?
        .as_str()
        .map(str::to_string)
}

/// What the resumed model is told.
pub(crate) fn resume_message(dangling_tool: Option<&str>) -> String {
    let tool = match dangling_tool {
        Some(tool) => format!(
            " Your last tool call ({tool}) has no result: it may have run fully, partly or not \
             at all. Check the state before repeating it."
        ),
        None => String::new(),
    };
    format!(
        "{RESUME_MARKER} The server restarted while you were working, so your turn was cut.{tool} \
         Continue from where you were; do not start over, and do not repeat what is already done."
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
            if node.capabilities.is_none() || node.updated_at < since {
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
            match judge(&tail) {
                Verdict::Settled => {}
                Verdict::Close(reason) => {
                    self.close_cut_turn(node.id, latest, reason).await;
                    report.closed += 1;
                }
                Verdict::Resume { dangling_tool } => pending.push(PendingResume {
                    message: resume_message(dangling_tool.as_deref()),
                    node,
                    seen_seq: latest,
                }),
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
                    .resume_agent_session(&item.node, &item.message, None)
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
        assert!(resume_message(Some("Bash")).contains("Bash"));
        assert!(resume_message(None).starts_with(RESUME_MARKER));
    }

    #[test]
    fn a_turn_waiting_for_an_answer_is_closed_not_resumed() {
        for last in ["permission_request", "ask_user_question"] {
            let tail = [ev(1, "user_message", "{}"), ev(2, last, "{}")];
            assert!(matches!(judge(&tail), Verdict::Close(_)), "{last}");
        }
    }

    #[test]
    fn a_session_cut_again_and_again_is_closed() {
        let resume = resume_message(None);
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
}
