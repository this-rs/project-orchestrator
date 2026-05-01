//! Drain loop — processes queued pending messages after stream ends.
//!
//! When `stream_response` completes, any messages queued in `pending_messages`
//! (from auto-continue, objective tracker, or concurrent user sends) are
//! drained one at a time: each message is persisted, broadcast, then
//! `stream_response` is called recursively to process it.

use super::manager::{ActiveSession, ChatManager};
use super::types::{ChatEvent, PendingMessage, PendingMessageKind};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager},
    InteractiveClient,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

/// Drain the pending_messages queue: pop the next message, persist it,
/// broadcast it, and recursively call `stream_response` to process it.
///
/// If the queue is empty but `has_pending` was true (race), clean up streaming status.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drain_pending_messages(
    has_pending: bool,
    client: Arc<Mutex<InteractiveClient>>,
    events_tx: broadcast::Sender<ChatEvent>,
    session_id: String,
    session_uuid: Option<Uuid>,
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
    search: Arc<dyn SearchStore>,
) {
    // Pop the next message from the queue, prioritising User > SystemHint
    // > BackgroundOutput so a noisy Monitor cannot starve user messages
    // (plan 806c8f2c, T2). FIFO is preserved within each priority class.
    let next_message = {
        let mut queue = pending_messages.lock().await;
        pop_highest_priority(&mut queue)
    };

    if let Some(next_msg) = next_message {
        let kind = next_msg.kind.clone();
        let msg_content = next_msg.content;

        info!(
            "Processing queued {} for session {} (queue was non-empty after stream)",
            match kind {
                PendingMessageKind::SystemHint => "system_hint",
                PendingMessageKind::User => "user_message",
                PendingMessageKind::BackgroundOutput => "background_output",
            },
            session_id
        );

        // Persist the event and update message count.
        // - SystemHint: persist as system_hint, no message_count bump
        // - User: persist as user_message, bump message_count
        // - BackgroundOutput: SKIP persist + SKIP broadcast — the OOB
        //   listener has already done both before pushing to the queue
        //   (see chat::oob_listener). Re-persisting would duplicate.
        let needs_persist_and_broadcast = !matches!(kind, PendingMessageKind::BackgroundOutput);
        if needs_persist_and_broadcast {
            if let Some(uuid) = session_uuid {
                if matches!(kind, PendingMessageKind::User) {
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
                }

                let (event_type, event_data) = match kind {
                    PendingMessageKind::SystemHint => (
                        "system_hint".to_string(),
                        serde_json::to_string(&ChatEvent::SystemHint {
                            content: msg_content.clone(),
                        })
                        .unwrap_or_default(),
                    ),
                    PendingMessageKind::User => (
                        "user_message".to_string(),
                        serde_json::to_string(&ChatEvent::UserMessage {
                            content: msg_content.clone(),
                        })
                        .unwrap_or_default(),
                    ),
                    PendingMessageKind::BackgroundOutput => unreachable!(
                        "BackgroundOutput is filtered out by needs_persist_and_broadcast"
                    ),
                };

                let event_record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type,
                    data: event_data,
                    created_at: chrono::Utc::now(),
                };
                let _ = graph.store_chat_events(uuid, vec![event_record]).await;
            }

            let chat_event = match kind {
                PendingMessageKind::SystemHint => ChatEvent::SystemHint {
                    content: msg_content.clone(),
                },
                PendingMessageKind::User => ChatEvent::UserMessage {
                    content: msg_content.clone(),
                },
                PendingMessageKind::BackgroundOutput => unreachable!(),
            };
            let _ = events_tx.send(chat_event.clone());
            if let Some(ref nats) = nats {
                nats.publish_chat_event(&session_id, chat_event);
            }
        } else {
            debug!(
                session_id = %session_id,
                "Drain: BackgroundOutput already persisted/broadcasted by OOB listener — \
                 forwarding content to stream_response as prompt"
            );
        }

        // Recursive call to process the queued message
        Box::pin(ChatManager::stream_response(
            client,
            events_tx,
            msg_content,
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
            shared_sdk_control_rx,
            auto_continue,
            retry_config,
            enrichment_pipeline,
            search,
        ))
        .await;
    } else if has_pending {
        // Race: has_pending was true but pop_front returned None.
        // Set is_streaming=false now since there's nothing to process.
        is_streaming.store(false, Ordering::SeqCst);
        let _ = events_tx.send(ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        if let Some(ref nats) = nats {
            nats.publish_chat_event(
                &session_id,
                ChatEvent::StreamingStatus {
                    is_streaming: false,
                },
            );
        }
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
        debug!(
            "Stream completed for session {} (pending race resolved)",
            session_id
        );
    }
}

/// Pop the highest-priority `PendingMessage` from the queue, preserving
/// FIFO order within a single priority class.
///
/// Priority order (lowest number = highest priority):
///   0 — `User` : the human's intent always wins
///   1 — `SystemHint` : system instructions (AgentGuard hints, plan-runner reminders)
///   2 — `BackgroundOutput` : passive observations from background tools (Monitor, BashOutput, …)
///
/// ## Why this exists (plan 806c8f2c)
///
/// Without prioritisation, a noisy `Monitor` (or any background-output-emitting
/// tool) can starve user messages: BackgroundOutput entries accumulate in
/// `pending_messages` while a stream is running, and a strict FIFO drain
/// processes them all before reading the user's typed message — the user
/// perceives the chat as "ignoring me" while ticks keep firing.
///
/// ## Complexity
///
/// `O(n)` scan over the queue + `O(n)` `remove(idx)`. `n` is typically < 50;
/// in practice the queue rarely exceeds 5-10 entries between drain cycles.
///
/// ## Implementation
///
/// Scan the queue once, track the index of the first entry of the lowest
/// priority value seen so far (lower number = higher priority). Then
/// `remove(idx)` it. Both ops are O(n) on `VecDeque`; n is small enough
/// that this is fine.
///
/// ## FIFO tie-break
///
/// Within a single priority class, the iteration finds the first match
/// (oldest in the queue), so insertion order is preserved among equals.
fn pop_highest_priority(queue: &mut VecDeque<PendingMessage>) -> Option<PendingMessage> {
    fn priority(kind: &PendingMessageKind) -> u8 {
        match kind {
            PendingMessageKind::User => 0,
            PendingMessageKind::SystemHint => 1,
            PendingMessageKind::BackgroundOutput => 2,
        }
    }

    // NB: must use `min_by` with explicit tie-break on the index, because
    // `min_by_key` returns the LAST element on equal keys (cf std docs) —
    // which would break our FIFO-intra-class invariant. Index is unique so
    // the secondary `cmp` always disambiguates.
    let best_idx = queue
        .iter()
        .enumerate()
        .min_by(|(ia, a), (ib, b)| {
            priority(&a.kind)
                .cmp(&priority(&b.kind))
                .then_with(|| ia.cmp(ib))
        })
        .map(|(i, _)| i)?;

    queue.remove(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::types::PendingMessage;

    fn user(content: &str) -> PendingMessage {
        PendingMessage::user(content.to_string())
    }
    fn system_hint(content: &str) -> PendingMessage {
        PendingMessage::system_hint(content.to_string())
    }
    fn bg(content: &str) -> PendingMessage {
        PendingMessage::background_output(content.to_string())
    }

    /// Helper: assert next pop is `expected_kind` with `expected_content`.
    #[track_caller]
    fn assert_next(
        q: &mut VecDeque<PendingMessage>,
        expected_kind: PendingMessageKind,
        expected_content: &str,
    ) {
        let m = pop_highest_priority(q).expect("queue not empty");
        assert_eq!(
            m.kind, expected_kind,
            "wrong kind: got {:?} content={:?}, expected {:?} content={:?}",
            m.kind, m.content, expected_kind, expected_content
        );
        assert_eq!(m.content, expected_content);
    }

    /// **T1 of plan 806c8f2c — RED test that demonstrates the bug.**
    ///
    /// On main HEAD with `pop_highest_priority` stubbed to `pop_front`, this
    /// test FAILS: the 3 BackgroundOutput entries are popped before the
    /// User message, reproducing the starvation symptom observed in prod.
    ///
    /// After T2 replaces the stub with the real priority-aware impl, this
    /// test PASSES — and serves as a permanent non-regression guard.
    #[test]
    fn test_user_message_drained_before_accumulated_background_outputs() {
        let mut q = VecDeque::new();
        q.push_back(bg("BG1"));
        q.push_back(bg("BG2"));
        q.push_back(bg("BG3"));
        q.push_back(user("U1"));

        // Expected behavior: User wins, then BG1 BG2 BG3 in FIFO order.
        let first = pop_highest_priority(&mut q).expect("queue not empty");
        assert_eq!(
            first.kind,
            PendingMessageKind::User,
            "BUG: User message starved by accumulated BackgroundOutput. \
             pop_highest_priority returned {:?} content={:?}",
            first.kind,
            first.content
        );
        assert_eq!(first.content, "U1");

        // FIFO within BackgroundOutput class is preserved.
        for expected in ["BG1", "BG2", "BG3"] {
            let next = pop_highest_priority(&mut q).expect("queue not empty");
            assert_eq!(next.kind, PendingMessageKind::BackgroundOutput);
            assert_eq!(
                next.content, expected,
                "FIFO intra-class violated: expected {expected}"
            );
        }

        assert!(pop_highest_priority(&mut q).is_none());
    }

    // ========================================================================
    // T3 of plan 806c8f2c — Comprehensive priority + FIFO invariants.
    //
    // The 9 tests below cover the full truth table of `pop_highest_priority`:
    //   - Single-priority comparisons (User > BG, User > SystemHint, SH > BG)
    //   - FIFO intra-class for User and BackgroundOutput (the two dominant
    //     classes in production)
    //   - Edge cases: empty queue, single entry
    //   - The full mixed scenario [BG, U, SH, BG, U, SH, BG] which is the
    //     most valuable single test — exercises priority + FIFO together.
    // ========================================================================

    #[test]
    fn test_priority_user_beats_background() {
        let mut q = VecDeque::new();
        q.push_back(bg("BG"));
        q.push_back(user("U"));
        assert_next(&mut q, PendingMessageKind::User, "U");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG");
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_user_beats_systemhint() {
        let mut q = VecDeque::new();
        q.push_back(system_hint("SH"));
        q.push_back(user("U"));
        assert_next(&mut q, PendingMessageKind::User, "U");
        assert_next(&mut q, PendingMessageKind::SystemHint, "SH");
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_systemhint_beats_background() {
        let mut q = VecDeque::new();
        q.push_back(bg("BG"));
        q.push_back(system_hint("SH"));
        assert_next(&mut q, PendingMessageKind::SystemHint, "SH");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG");
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_user_beats_multiple_backgrounds() {
        // The User entry is in the MIDDLE of a sea of BGs — it must still
        // win, then the remaining BGs come out in their original FIFO order
        // (BG1, BG2, BG3, BG4 — note BG4 was inserted AFTER U so it stays
        // last even though U was extracted before it).
        let mut q = VecDeque::new();
        q.push_back(bg("BG1"));
        q.push_back(bg("BG2"));
        q.push_back(bg("BG3"));
        q.push_back(user("U"));
        q.push_back(bg("BG4"));

        assert_next(&mut q, PendingMessageKind::User, "U");
        for expected in ["BG1", "BG2", "BG3", "BG4"] {
            let next = pop_highest_priority(&mut q).expect("queue not empty");
            assert_eq!(next.kind, PendingMessageKind::BackgroundOutput);
            assert_eq!(
                next.content, expected,
                "FIFO intra-class violated: expected {expected}"
            );
        }
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_fifo_within_user_class() {
        // Three User messages in a row — strict insertion order out.
        // Catches the `min_by_key` trap (returns LAST on equal keys).
        let mut q = VecDeque::new();
        q.push_back(user("U1"));
        q.push_back(user("U2"));
        q.push_back(user("U3"));
        assert_next(&mut q, PendingMessageKind::User, "U1");
        assert_next(&mut q, PendingMessageKind::User, "U2");
        assert_next(&mut q, PendingMessageKind::User, "U3");
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_fifo_within_background_class() {
        // Three BackgroundOutput in a row — strict insertion order out.
        // This is the "Monitor tick" case: agents must see ticks in the
        // order they were produced, otherwise the assistant gets confused
        // ("Tick 5" arriving before "Tick 4" makes no narrative sense).
        let mut q = VecDeque::new();
        q.push_back(bg("BG1"));
        q.push_back(bg("BG2"));
        q.push_back(bg("BG3"));
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG1");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG2");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG3");
        assert!(pop_highest_priority(&mut q).is_none());
    }

    #[test]
    fn test_priority_empty_queue() {
        let mut q: VecDeque<PendingMessage> = VecDeque::new();
        assert!(
            pop_highest_priority(&mut q).is_none(),
            "empty queue must return None, not panic"
        );
    }

    #[test]
    fn test_priority_single_entry() {
        let mut q = VecDeque::new();
        q.push_back(bg("only"));
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "only");
        assert!(
            q.is_empty(),
            "queue must be empty after popping the sole entry"
        );
        assert!(pop_highest_priority(&mut q).is_none());
    }

    /// The most valuable single test of the suite. Mixes 3 priority classes
    /// across 7 entries and asserts the full pop sequence:
    ///
    ///   queue (insertion order): BG1, U1, SH1, BG2, U2, SH2, BG3
    ///   expected pop sequence:   U1, U2, SH1, SH2, BG1, BG2, BG3
    ///
    /// This single test would catch:
    ///   - Any priority inversion (User not first)
    ///   - Any FIFO-intra-class violation (U2 before U1, or SH2 before SH1)
    ///   - The `min_by_key` trap (LAST on equal keys)
    ///   - Off-by-one in the index tie-break
    ///   - Failure to remove from the right index after `enumerate()`
    #[test]
    fn test_priority_mixed_complex() {
        let mut q = VecDeque::new();
        q.push_back(bg("BG1"));
        q.push_back(user("U1"));
        q.push_back(system_hint("SH1"));
        q.push_back(bg("BG2"));
        q.push_back(user("U2"));
        q.push_back(system_hint("SH2"));
        q.push_back(bg("BG3"));

        // Users first, in FIFO order.
        assert_next(&mut q, PendingMessageKind::User, "U1");
        assert_next(&mut q, PendingMessageKind::User, "U2");
        // Then SystemHints, in FIFO order.
        assert_next(&mut q, PendingMessageKind::SystemHint, "SH1");
        assert_next(&mut q, PendingMessageKind::SystemHint, "SH2");
        // Then BackgroundOutputs, in FIFO order.
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG1");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG2");
        assert_next(&mut q, PendingMessageKind::BackgroundOutput, "BG3");

        assert!(pop_highest_priority(&mut q).is_none());
    }
}
