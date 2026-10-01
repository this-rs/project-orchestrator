//! Attention derivation: what is a session waiting on the user for?
//!
//! Pure functions, no I/O. Two sources, each for what it knows:
//! - the STORED chat events (with `seq`) say WHAT is asked and SINCE WHEN;
//! - the live state (is the CLI still there?) says whether the requester is
//!   STILL THERE.
//!
//! Rules:
//! - a permission is pending = a `permission_request` with no
//!   `permission_decision` of the same id;
//! - a question is pending = an `ask_user_question` with no `user_message` of
//!   a HIGHER `seq` (answering is just a `send_message`: there is NO
//!   `input_response` event);
//! - live session -> actionable ([`WaitingRequest`]); dead session -> orphan
//!   ([`OrphanRequest`]), with since when the CLI is stopped.
//!
//! `input_request` is a dead type (never emitted in production): it is
//! ignored on purpose, counting it would invent phantom requests. The grouped
//! read ([`GraphStore::get_attention_events`]) does not even fetch it.
//!
//! [`GraphStore::get_attention_events`]: crate::neo4j::traits::GraphStore::get_attention_events

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::api::attention::{OrphanRequest, QuestionOption, RequestKind, WaitingRequest};
use crate::chat::types::ChatEvent;
use crate::neo4j::models::ChatEventRecord;

/// Everything the derivation needs to know about one session, except its
/// events (passed separately so the grouped read can be split per session).
#[derive(Debug, Clone)]
pub struct SessionAttentionInput {
    pub session_id: Uuid,
    pub workspace: String,
    pub thread_id: Option<Uuid>,
    /// Is the CLI process still there (in-memory state)?
    pub alive: bool,
    /// Since when the CLI is stopped (only meaningful when `!alive`).
    pub cli_stopped_at: Option<DateTime<Utc>>,
}

/// Result of the derivation: both lists are ordered oldest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DerivedAttention {
    pub waiting: Vec<WaitingRequest>,
    pub orphans: Vec<OrphanRequest>,
}

/// A pending request before it is classified live/dead.
struct Pending {
    request_id: String,
    kind: RequestKind,
    tool_name: Option<String>,
    text: String,
    options: Vec<QuestionOption>,
    seq: u64,
    requested_at: DateTime<Utc>,
}

/// Derive the pending requests of ONE session from its events.
/// `events` may be in any order and may contain unrelated event types.
pub fn derive_session_attention(
    input: &SessionAttentionInput,
    events: &[ChatEventRecord],
    now: DateTime<Utc>,
) -> DerivedAttention {
    let mut decided: HashSet<String> = HashSet::new();
    let mut last_user_seq: i64 = i64::MIN;
    for e in events {
        match e.event_type.as_str() {
            "permission_decision" => {
                if let Ok(ChatEvent::PermissionDecision { id, .. }) =
                    serde_json::from_str::<ChatEvent>(&e.data)
                {
                    decided.insert(id);
                }
            }
            "user_message" => last_user_seq = last_user_seq.max(e.seq),
            _ => {}
        }
    }

    let mut ordered: Vec<&ChatEventRecord> = events.iter().collect();
    ordered.sort_by_key(|e| e.seq);

    let mut seen: HashSet<String> = HashSet::new();
    let mut pending: Vec<Pending> = Vec::new();
    for e in ordered {
        let seq = u64::try_from(e.seq).unwrap_or(0);
        match e.event_type.as_str() {
            "permission_request" => {
                let Ok(ChatEvent::PermissionRequest {
                    id, tool, input, ..
                }) = serde_json::from_str::<ChatEvent>(&e.data)
                else {
                    continue;
                };
                if decided.contains(&id) || !seen.insert(format!("p:{id}")) {
                    continue;
                }
                pending.push(Pending {
                    request_id: id,
                    kind: RequestKind::Permission,
                    tool_name: Some(tool),
                    text: permission_text(&input),
                    options: Vec::new(),
                    seq,
                    requested_at: e.created_at,
                });
            }
            "ask_user_question" => {
                // A user_message of a strictly higher seq answers it; an
                // EARLIER one does not.
                if last_user_seq > e.seq {
                    continue;
                }
                let Ok(ChatEvent::AskUserQuestion { id, questions, .. }) =
                    serde_json::from_str::<ChatEvent>(&e.data)
                else {
                    continue;
                };
                if !seen.insert(format!("q:{id}")) {
                    continue;
                }
                let (text, options) = question_text_and_options(&questions);
                pending.push(Pending {
                    request_id: id,
                    kind: RequestKind::Question,
                    tool_name: None,
                    text,
                    options,
                    seq,
                    requested_at: e.created_at,
                });
            }
            // input_request: dead type, never emitted — deliberately ignored.
            _ => {}
        }
    }

    let mut out = DerivedAttention::default();
    for p in pending {
        let age_secs = u64::try_from((now - p.requested_at).num_seconds()).unwrap_or(0);
        if input.alive {
            out.waiting.push(WaitingRequest {
                request_id: p.request_id,
                kind: p.kind,
                session_id: input.session_id,
                thread_id: input.thread_id,
                workspace: input.workspace.clone(),
                tool_name: p.tool_name,
                text: p.text,
                options: p.options,
                seq: p.seq,
                requested_at: p.requested_at,
                age_secs,
            });
        } else {
            out.orphans.push(OrphanRequest {
                request_id: p.request_id,
                kind: p.kind,
                session_id: input.session_id,
                thread_id: input.thread_id,
                workspace: input.workspace.clone(),
                tool_name: p.tool_name,
                text: p.text,
                options: p.options,
                seq: p.seq,
                requested_at: p.requested_at,
                age_secs,
                cli_stopped_at: input.cli_stopped_at,
            });
        }
    }
    out
}

/// Derive for many sessions from ONE grouped read (events of all sessions
/// mixed): the caller does a single query, never one per session. Lists are
/// sorted oldest first across sessions (ties: session id, then seq).
pub fn derive_attention(
    sessions: &[SessionAttentionInput],
    events: Vec<ChatEventRecord>,
    now: DateTime<Utc>,
) -> DerivedAttention {
    let mut by_session: HashMap<Uuid, Vec<ChatEventRecord>> = HashMap::new();
    for e in events {
        by_session.entry(e.session_id).or_default().push(e);
    }
    let mut out = DerivedAttention::default();
    for s in sessions {
        let evs = by_session.remove(&s.session_id).unwrap_or_default();
        let d = derive_session_attention(s, &evs, now);
        out.waiting.extend(d.waiting);
        out.orphans.extend(d.orphans);
    }
    out.waiting
        .sort_by_key(|r| (r.requested_at, r.session_id, r.seq));
    out.orphans
        .sort_by_key(|r| (r.requested_at, r.session_id, r.seq));
    out
}

/// EXACT text of a permission: the command when there is one, else the file
/// path, else the compact JSON input. Never truncated.
fn permission_text(input: &serde_json::Value) -> String {
    for key in ["command", "file_path", "path", "url"] {
        if let Some(s) = input.get(key).and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    match input {
        serde_json::Value::Null => String::new(),
        v => v.to_string(),
    }
}

/// Question text (several questions are joined by a blank line, untruncated)
/// and the offered options. Options are only exposed when there is a single
/// question: with several, a flat option list would be ambiguous.
fn question_text_and_options(questions: &serde_json::Value) -> (String, Vec<QuestionOption>) {
    let Some(arr) = questions.as_array() else {
        return (String::new(), Vec::new());
    };
    let text = arr
        .iter()
        .filter_map(|q| q.get("question").and_then(|v| v.as_str()))
        .collect::<Vec<_>>()
        .join("\n\n");
    let options = if arr.len() == 1 {
        arr[0]
            .get("options")
            .and_then(|v| v.as_array())
            .map(|opts| {
                opts.iter()
                    .filter_map(|o| {
                        let label = o.get("label").and_then(|v| v.as_str())?;
                        Some(QuestionOption {
                            label: label.to_string(),
                            description: o
                                .get("description")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    (text, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn t(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_800_000_000 + secs, 0).unwrap()
    }

    fn rec(session: Uuid, seq: i64, ev: ChatEvent) -> ChatEventRecord {
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: session,
            seq,
            event_type: ev.event_type().to_string(),
            data: serde_json::to_string(&ev).unwrap(),
            created_at: t(seq * 10),
        }
    }

    fn perm(s: Uuid, seq: i64, id: &str) -> ChatEventRecord {
        rec(
            s,
            seq,
            ChatEvent::PermissionRequest {
                id: id.into(),
                tool: "Bash".into(),
                input: json!({"command": "rm -rf target"}),
                parent_tool_use_id: None,
            },
        )
    }

    fn decision(s: Uuid, seq: i64, id: &str) -> ChatEventRecord {
        rec(
            s,
            seq,
            ChatEvent::PermissionDecision {
                id: id.into(),
                allow: true,
            },
        )
    }

    fn question(s: Uuid, seq: i64, id: &str) -> ChatEventRecord {
        rec(
            s,
            seq,
            ChatEvent::AskUserQuestion {
                id: id.into(),
                tool_call_id: "tc".into(),
                questions: json!([{"question": "Which db?", "options": [
                    {"label": "A", "description": "first"}, {"label": "B"}]}]),
                input: json!({}),
                parent_tool_use_id: None,
            },
        )
    }

    fn user(s: Uuid, seq: i64) -> ChatEventRecord {
        rec(
            s,
            seq,
            ChatEvent::UserMessage {
                content: "ok".into(),
            },
        )
    }

    fn input(s: Uuid, alive: bool) -> SessionAttentionInput {
        SessionAttentionInput {
            session_id: s,
            workspace: "ws".into(),
            thread_id: None,
            alive,
            cli_stopped_at: if alive { None } else { Some(t(500)) },
        }
    }

    #[test]
    fn permission_followed_by_decision_is_not_pending() {
        let s = Uuid::new_v4();
        let evs = vec![perm(s, 1, "p1"), decision(s, 2, "p1")];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        assert!(d.waiting.is_empty() && d.orphans.is_empty());
        let d = derive_session_attention(&input(s, false), &evs, t(1000));
        assert!(d.waiting.is_empty() && d.orphans.is_empty());
    }

    #[test]
    fn decision_of_another_id_does_not_answer() {
        let s = Uuid::new_v4();
        let evs = vec![perm(s, 1, "p1"), decision(s, 2, "other")];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        assert_eq!(d.waiting.len(), 1);
    }

    #[test]
    fn question_followed_by_later_user_message_is_not_pending() {
        let s = Uuid::new_v4();
        let evs = vec![question(s, 5, "q1"), user(s, 6)];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        assert!(d.waiting.is_empty() && d.orphans.is_empty());
    }

    #[test]
    fn question_with_only_an_earlier_user_message_is_pending() {
        let s = Uuid::new_v4();
        let evs = vec![user(s, 2), question(s, 5, "q1")];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        assert_eq!(d.waiting.len(), 1);
        let w = &d.waiting[0];
        assert_eq!(w.kind, RequestKind::Question);
        assert_eq!(w.text, "Which db?");
        assert_eq!(w.options.len(), 2);
        assert_eq!(w.options[0].description.as_deref(), Some("first"));
        assert_eq!(w.seq, 5);
    }

    #[test]
    fn unanswered_request_is_actionable_when_alive_orphan_when_dead() {
        let s = Uuid::new_v4();
        let evs = vec![perm(s, 3, "p1")];
        let live = derive_session_attention(&input(s, true), &evs, t(1000));
        assert_eq!(live.waiting.len(), 1);
        assert!(live.orphans.is_empty());
        assert_eq!(live.waiting[0].text, "rm -rf target");
        assert_eq!(live.waiting[0].tool_name.as_deref(), Some("Bash"));
        assert_eq!(live.waiting[0].age_secs, 970);

        let dead = derive_session_attention(&input(s, false), &evs, t(1000));
        assert!(dead.waiting.is_empty());
        assert_eq!(dead.orphans.len(), 1);
        assert_eq!(dead.orphans[0].cli_stopped_at, Some(t(500)));
        assert_eq!(dead.orphans[0].request_id, "p1");
    }

    #[test]
    fn two_pending_in_one_session_are_both_rendered_oldest_first() {
        let s = Uuid::new_v4();
        // Deliberately out of order in the input.
        let evs = vec![question(s, 9, "q1"), perm(s, 4, "p1")];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        let ids: Vec<_> = d.waiting.iter().map(|w| w.request_id.as_str()).collect();
        assert_eq!(ids, vec!["p1", "q1"]);
    }

    #[test]
    fn input_request_is_ignored() {
        let s = Uuid::new_v4();
        let evs = vec![rec(
            s,
            1,
            ChatEvent::InputRequest {
                prompt: "?".into(),
                options: None,
                parent_tool_use_id: None,
            },
        )];
        let d = derive_session_attention(&input(s, true), &evs, t(1000));
        assert!(d.waiting.is_empty() && d.orphans.is_empty());
    }

    #[test]
    fn grouped_derivation_splits_sessions_and_sorts_oldest_first() {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let evs = vec![perm(a, 8, "pa"), perm(b, 2, "pb"), decision(b, 3, "zzz")];
        let d = derive_attention(&[input(a, true), input(b, false)], evs, t(1000));
        assert_eq!(d.waiting.len(), 1);
        assert_eq!(d.waiting[0].session_id, a);
        assert_eq!(d.orphans.len(), 1);
        assert_eq!(d.orphans[0].session_id, b);
    }

    #[tokio::test]
    async fn grouped_store_read_returns_all_sessions_without_input_request_and_blank_user_data() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::traits::GraphStore;
        let store = MockGraphStore::new();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        store
            .store_chat_events(
                a,
                vec![
                    perm(a, 1, "p"),
                    user(a, 2),
                    rec(
                        a,
                        3,
                        ChatEvent::InputRequest {
                            prompt: "x".into(),
                            options: None,
                            parent_tool_use_id: None,
                        },
                    ),
                ],
            )
            .await
            .unwrap();
        store
            .store_chat_events(b, vec![question(b, 1, "q")])
            .await
            .unwrap();
        let got = store.get_attention_events(&[a, b]).await.unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|e| e.event_type != "input_request"));
        assert!(got
            .iter()
            .find(|e| e.event_type == "user_message")
            .unwrap()
            .data
            .is_empty());
    }
}
