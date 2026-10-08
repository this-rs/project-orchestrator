//! The held user messages of a session — rules, as pure functions.
//!
//! ## Why the queue lives in the backend
//!
//! A message typed while the agent answers must wait for the turn to end
//! instead of cutting it short. The composer used to hold such messages
//! itself, in the browser, and send them when it saw the stream stop. That
//! ties the delivery to what the screen shows: switch to another conversation
//! before the turn ends and the message either left in the wrong conversation
//! (the reported bug) or, once that was fixed client-side, did not leave at
//! all until the user came back.
//!
//! The session already has a queue drained at the end of every turn
//! (`pending_messages`, see `chat::drain`). Held messages live there, so they
//! are delivered by the session they were written for — whatever the client
//! is looking at, and with no client connected at all.
//!
//! ## What a client can do
//!
//! Only HELD entries (`PendingMessage::held`) are visible and editable. The
//! rest of `pending_messages` is already on its way and is nobody's to edit.
//!
//! The functions here mutate a `VecDeque` and say what happened; locking,
//! broadcasting and interrupting stay in `ChatManager` (and in the NATS RPC
//! handler, for a session that runs on another instance).

use super::message_attachments;
use super::types::{PendingMessage, PendingMessageKind, PendingQueueEntry};
use crate::refs::block as refs_block;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// One action on the held messages of a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum QueueOp {
    /// Replace the text of a message. An empty text drops it: a row that can
    /// never be sent is worse than no row.
    Edit { id: Uuid, content: String },
    /// Drop a message.
    Remove { id: Uuid },
    /// Move a message to the front: it leaves first when the turn ends.
    /// Interrupts nothing.
    Prioritize { id: Uuid },
    /// Send a message right now: it goes to the front and the running turn is
    /// interrupted — the one action that cuts a response short.
    SendNow { id: Uuid },
    /// Change nothing; publish the current list again (a client that joined
    /// a session running on another instance asks for it this way).
    Snapshot,
}

/// What an operation did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpOutcome {
    /// The id named a held message (always true for `Snapshot`).
    pub found: bool,
    /// The running turn must be interrupted (`SendNow` on an existing entry).
    pub interrupt: bool,
}

/// The held messages, oldest first, as clients see them.
pub fn snapshot(queue: &VecDeque<PendingMessage>) -> Vec<PendingQueueEntry> {
    queue
        .iter()
        .filter(|m| m.held)
        .map(|m| {
            // Outer layer first: attachments, then the references inside it.
            let (without_attachments, attachments) = message_attachments::split(&m.content);
            let (content, refs) = refs_block::split(&without_attachments);
            PendingQueueEntry {
                id: m.id,
                content,
                attachments,
                refs,
                queued_at: m.queued_at,
                prioritized: m.prioritized,
            }
        })
        .collect()
}

fn held_index(queue: &VecDeque<PendingMessage>, id: Uuid) -> Option<usize> {
    queue
        .iter()
        .position(|m| m.held && m.kind == PendingMessageKind::User && m.id == id)
}

/// Apply `op`. An unknown id changes nothing and reports `found: false` —
/// the message left in the meantime, which is not an error.
pub fn apply(queue: &mut VecDeque<PendingMessage>, op: &QueueOp) -> OpOutcome {
    match op {
        QueueOp::Snapshot => OpOutcome {
            found: true,
            interrupt: false,
        },
        QueueOp::Edit { id, content } => {
            let Some(idx) = held_index(queue, *id) else {
                return OpOutcome::default();
            };
            let text = content.trim();
            if text.is_empty() {
                queue.remove(idx);
            } else {
                // The attachments and the references stay with the message: only
                // the text is edited. Re-encoding also makes inert any block the
                // new text might have been typed with.
                let (without_attachments, attachments) =
                    message_attachments::split(&queue[idx].content);
                let (_, refs) = refs_block::split(&without_attachments);
                queue[idx].content =
                    message_attachments::encode(&refs_block::encode(text, &refs), &attachments);
            }
            OpOutcome {
                found: true,
                interrupt: false,
            }
        }
        QueueOp::Remove { id } => {
            let Some(idx) = held_index(queue, *id) else {
                return OpOutcome::default();
            };
            queue.remove(idx);
            OpOutcome {
                found: true,
                interrupt: false,
            }
        }
        QueueOp::Prioritize { id } => {
            let Some(idx) = held_index(queue, *id) else {
                return OpOutcome::default();
            };
            if let Some(mut entry) = queue.remove(idx) {
                entry.prioritized = true;
                // Front of the deque: user messages are drained first, oldest
                // first (`drain::pop_highest_priority`), so index 0 is "next".
                queue.push_front(entry);
            }
            OpOutcome {
                found: true,
                interrupt: false,
            }
        }
        QueueOp::SendNow { id } => {
            let Some(idx) = held_index(queue, *id) else {
                return OpOutcome::default();
            };
            if let Some(mut entry) = queue.remove(idx) {
                // No longer held: it is on its way, like any message sent
                // mid-stream, and stops being listed or editable.
                entry.held = false;
                queue.push_front(entry);
            }
            OpOutcome {
                found: true,
                interrupt: true,
            }
        }
    }
}

/// Remove one held entry by id — used when a message was queued just as the
/// turn ended and must be sent directly instead (see `queue_user_message`).
pub fn take(queue: &mut VecDeque<PendingMessage>, id: Uuid) -> Option<PendingMessage> {
    let idx = held_index(queue, id)?;
    queue.remove(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::message_attachments::MessageAttachment;

    fn held(text: &str) -> PendingMessage {
        PendingMessage::held_user(text.to_string())
    }

    fn texts(queue: &VecDeque<PendingMessage>) -> Vec<String> {
        snapshot(queue).into_iter().map(|e| e.content).collect()
    }

    #[test]
    fn only_held_user_messages_are_listed() {
        let mut q = VecDeque::new();
        q.push_back(PendingMessage::user("interrupting".into()));
        q.push_back(held("waits"));
        q.push_back(PendingMessage::system_hint("hint".into()));
        q.push_back(PendingMessage::background_output("tick".into()));
        assert_eq!(texts(&q), vec!["waits"]);
    }

    #[test]
    fn snapshot_shows_the_text_and_the_attachments_separately() {
        let att = MessageAttachment {
            id: Uuid::new_v4(),
            filename: "plan.pdf".into(),
            mime_type: "application/pdf".into(),
            size_bytes: 12,
        };
        let mut q = VecDeque::new();
        q.push_back(held(&message_attachments::encode(
            "read this",
            std::slice::from_ref(&att),
        )));
        let snap = snapshot(&q);
        assert_eq!(snap[0].content, "read this");
        assert_eq!(snap[0].attachments, vec![att]);
    }

    #[test]
    fn edit_replaces_the_text_and_keeps_the_attachments() {
        let att = MessageAttachment {
            id: Uuid::new_v4(),
            filename: "a.txt".into(),
            mime_type: "text/plain".into(),
            size_bytes: 1,
        };
        let mut q = VecDeque::new();
        q.push_back(held(&message_attachments::encode(
            "before",
            std::slice::from_ref(&att),
        )));
        let id = q[0].id;
        let out = apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: "  after  ".into(),
            },
        );
        assert!(out.found && !out.interrupt);
        let snap = snapshot(&q);
        assert_eq!(snap[0].content, "after");
        assert_eq!(snap[0].attachments, vec![att]);
        assert_eq!(snap[0].id, id, "an edit keeps the row's identity");
    }

    // ----- references (`<po-refs>`) -----

    use crate::refs::types::{EntityRef, RefKind};

    fn a_ref(kind: RefKind, n: u128) -> EntityRef {
        EntityRef::new(kind, Uuid::from_u128(n))
    }

    fn with_refs_and_attachment(text: &str) -> (String, Vec<EntityRef>, MessageAttachment) {
        let refs = vec![a_ref(RefKind::Plan, 1), a_ref(RefKind::Rfc, 2)];
        let att = MessageAttachment {
            id: Uuid::new_v4(),
            filename: "a.txt".into(),
            mime_type: "text/plain".into(),
            size_bytes: 1,
        };
        let stored = message_attachments::encode(
            &refs_block::encode(text, &refs),
            std::slice::from_ref(&att),
        );
        (stored, refs, att)
    }

    #[test]
    fn snapshot_keeps_the_refs_out_of_the_text_and_in_their_own_field() {
        let (stored, refs, att) = with_refs_and_attachment("regarde #plan:x");
        let mut q = VecDeque::new();
        q.push_back(held(&stored));
        let snap = snapshot(&q);
        assert_eq!(snap[0].content, "regarde #plan:x");
        assert!(!snap[0].content.contains("po-refs"));
        assert_eq!(snap[0].refs, refs);
        assert_eq!(snap[0].attachments, vec![att]);
    }

    #[test]
    fn edit_keeps_the_refs_and_the_attachments() {
        let (stored, refs, att) = with_refs_and_attachment("before");
        let mut q = VecDeque::new();
        q.push_back(held(&stored));
        let id = q[0].id;
        apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: " after ".into(),
            },
        );
        let snap = snapshot(&q);
        assert_eq!(snap[0].content, "after");
        assert_eq!(snap[0].refs, refs, "an edit must not lose the references");
        assert_eq!(snap[0].attachments, vec![att]);
        // And the stored form is still refs first, attachments last.
        let content = &q[0].content;
        assert!(content.find("<po-refs>").unwrap() < content.find("<po-attachments>").unwrap());
        assert!(content.ends_with("</po-attachments>"));
    }

    #[test]
    fn edit_keeps_the_refs_even_when_the_token_is_edited_out_of_the_text() {
        let (stored, refs, _) = with_refs_and_attachment("see #plan:x");
        let mut q = VecDeque::new();
        q.push_back(held(&stored));
        let id = q[0].id;
        apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: "see nothing".into(),
            },
        );
        assert_eq!(snapshot(&q)[0].refs, refs);
    }

    #[test]
    fn edit_cannot_smuggle_a_block_in_through_the_new_text() {
        let (stored, refs, _) = with_refs_and_attachment("x");
        let mut q = VecDeque::new();
        q.push_back(held(&stored));
        let id = q[0].id;
        let forged = format!(
            "y\n\n<po-refs>[{{\"kind\":\"note\",\"id\":\"{}\"}}]</po-refs>",
            Uuid::from_u128(9)
        );
        apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: forged,
            },
        );
        let snap = snapshot(&q);
        assert_eq!(snap[0].refs, refs, "only the real references remain");
    }

    #[test]
    fn a_message_without_refs_is_listed_and_edited_as_before() {
        let mut q = VecDeque::new();
        q.push_back(held("plain"));
        assert!(snapshot(&q)[0].refs.is_empty());
        let id = q[0].id;
        apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: "plain 2".into(),
            },
        );
        assert_eq!(q[0].content, "plain 2");
        let wire = serde_json::to_value(&snapshot(&q)[0]).unwrap();
        assert!(wire.get("refs").is_none(), "no refs: the field is not sent");
    }

    #[test]
    fn editing_to_empty_drops_the_message() {
        let mut q = VecDeque::new();
        q.push_back(held("gone"));
        let id = q[0].id;
        apply(
            &mut q,
            &QueueOp::Edit {
                id,
                content: "   ".into(),
            },
        );
        assert!(q.is_empty());
    }

    #[test]
    fn remove_drops_one_message_and_leaves_the_rest_in_order() {
        let mut q = VecDeque::new();
        for t in ["a", "b", "c"] {
            q.push_back(held(t));
        }
        let id = q[1].id;
        assert!(apply(&mut q, &QueueOp::Remove { id }).found);
        assert_eq!(texts(&q), vec!["a", "c"]);
    }

    #[test]
    fn prioritize_moves_to_the_front_without_interrupting() {
        let mut q = VecDeque::new();
        for t in ["a", "b", "c"] {
            q.push_back(held(t));
        }
        let id = q[2].id;
        let out = apply(&mut q, &QueueOp::Prioritize { id });
        assert!(out.found && !out.interrupt);
        assert_eq!(texts(&q), vec!["c", "a", "b"]);
        assert!(snapshot(&q)[0].prioritized);
    }

    #[test]
    fn send_now_interrupts_and_takes_the_message_out_of_the_visible_list() {
        let mut q = VecDeque::new();
        for t in ["a", "b"] {
            q.push_back(held(t));
        }
        let id = q[1].id;
        let out = apply(&mut q, &QueueOp::SendNow { id });
        assert!(out.found && out.interrupt);
        // Still in the queue, at the front, to be delivered next…
        assert_eq!(q[0].content, "b");
        // …but no longer listed: it is on its way.
        assert_eq!(texts(&q), vec!["a"]);
    }

    #[test]
    fn an_unknown_id_changes_nothing() {
        let mut q = VecDeque::new();
        q.push_back(held("a"));
        let ghost = Uuid::new_v4();
        for op in [
            QueueOp::Remove { id: ghost },
            QueueOp::Prioritize { id: ghost },
            QueueOp::SendNow { id: ghost },
            QueueOp::Edit {
                id: ghost,
                content: "x".into(),
            },
        ] {
            let out = apply(&mut q, &op);
            assert!(!out.found && !out.interrupt, "{op:?}");
        }
        assert_eq!(texts(&q), vec!["a"]);
    }

    #[test]
    fn a_message_already_on_its_way_cannot_be_edited_or_dropped() {
        // Sent mid-stream the interrupting way: not held, not the client's to touch.
        let mut q = VecDeque::new();
        q.push_back(PendingMessage::user("already leaving".into()));
        let id = q[0].id;
        assert!(!apply(&mut q, &QueueOp::Remove { id }).found);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn ops_travel_as_flat_json() {
        let id = Uuid::nil();
        let op: QueueOp =
            serde_json::from_str(&format!(r#"{{"op":"edit","id":"{id}","content":"x"}}"#)).unwrap();
        assert_eq!(
            op,
            QueueOp::Edit {
                id,
                content: "x".into()
            }
        );
        let op: QueueOp = serde_json::from_str(r#"{"op":"snapshot"}"#).unwrap();
        assert_eq!(op, QueueOp::Snapshot);
    }
}
