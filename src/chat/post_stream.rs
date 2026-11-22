//! Post-stream processing logic extracted from `stream_response`.
//!
//! After the Claude SDK stream ends, several housekeeping tasks run:
//! - Post-compaction context re-injection
//! - Lock-free interrupt + ToolCancelled emission
//! - Auto-continue with enriched objectives
//! - Objective tracking reminder injection
//! - Streaming status updates
//! - Event persistence to Neo4j
//! - Memory/feedback/RFC recording
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::manager::ActiveSession;
use super::provider::cognitive::segment::{Boundary, Segment};
use super::types::{ChatEvent, PendingMessage};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager},
    InteractiveClient,
};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

#[allow(unused_imports)]
use std::collections::VecDeque;

/// Longest a single post-stream step may hold the turn.
///
/// Every step talks to a store (Neo4j, Meilisearch) through a client that has
/// no timeout of its own. A call that never answered used to pin the turn:
/// `is_streaming` stayed true (`finalize_streaming_status` leaves it true while
/// messages are queued — the objective reminder it just queued, for one), the
/// Stop had nothing left to cancel (the stream loop, the only reader of the
/// cancellation token, was over), the queued messages were never sent.
/// Measured on 2026-10-05 (session 3780028f, note 1a31aadb) and 2026-10-09
/// (session b4c28b9f: the CLI answered at 20:03:04, the turn was still
/// "streaming" at 22:00). Past this budget the turn goes on — see
/// [`StepBudget`].
pub(crate) const POST_STREAM_STEP_BUDGET: Duration = Duration::from_secs(30);

/// Past this, a step or a write that did finish is logged with its duration:
/// the budget above should stay far from what the stores actually take.
const POST_STREAM_SLOW_STEP: Duration = Duration::from_secs(2);

/// Error code of a post-stream step dropped past its budget (`reason`: the step).
pub(crate) const STEP_ABANDONED_CODE: &str = "post_stream_step_abandoned";
/// Error code of a store write still running past its budget (`reason`: the
/// step). The write is not dropped: it lands when the store answers.
pub(crate) const PERSISTENCE_DELAYED_CODE: &str = "persistence_delayed";

/// Run `fut` under `budget`. Past the budget it is dropped — cancelled at
/// whatever await it was sitting on, releasing what it held — and `None`
/// says so; the step is named in the warning so the next hang names its
/// store. No event: [`StepBudget::run`] adds the one the clients see.
pub(crate) async fn bounded<F: Future>(
    session_id: &str,
    step: &'static str,
    budget: Duration,
    fut: F,
) -> Option<F::Output> {
    let started = Instant::now();
    match tokio::time::timeout(budget, fut).await {
        Ok(out) => {
            let elapsed = started.elapsed();
            if elapsed > POST_STREAM_SLOW_STEP {
                info!(
                    session_id = %session_id,
                    step,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "post-stream step was slow"
                );
            }
            Some(out)
        }
        Err(_) => {
            warn!(
                session_id = %session_id,
                step,
                budget_ms = budget.as_millis() as u64,
                "post-stream step did not finish within its budget: abandoned, the turn goes on"
            );
            None
        }
    }
}

/// Run a store write in a task of its own.
///
/// The write outlives whatever waits for it: a wait past its budget
/// ([`StepBudget::wait_for_write`]), a turn abandoned by the interrupt
/// watchdog. A write is never dropped: it lands when the store answers, and
/// the log says so when it was late, or that it failed.
pub(crate) fn spawn_write<F>(session_id: String, step: &'static str, write: F) -> JoinHandle<()>
where
    F: Future<Output = anyhow::Result<()>> + Send + 'static,
{
    tokio::spawn(async move {
        let started = Instant::now();
        let outcome = write.await;
        let elapsed = started.elapsed();
        match outcome {
            Err(e) => warn!(
                session_id = %session_id,
                step,
                elapsed_ms = elapsed.as_millis() as u64,
                "store write failed: {e:#}"
            ),
            Ok(()) if elapsed > POST_STREAM_SLOW_STEP => info!(
                session_id = %session_id,
                step,
                elapsed_ms = elapsed.as_millis() as u64,
                "store write landed late"
            ),
            Ok(()) => {}
        }
    })
}

/// Wait for a write started by [`spawn_write`], up to `budget`. Past it the
/// caller goes on and gets the `error` event (`persistence_delayed`, `reason` =
/// the step) its clients must see; the write itself keeps running and lands when
/// the store answers ([`spawn_write`] logs it, late or failed).
pub(crate) async fn await_write(
    session_id: &str,
    step: &'static str,
    budget: Duration,
    write: JoinHandle<()>,
) -> Option<ChatEvent> {
    match tokio::time::timeout(budget, write).await {
        Ok(Ok(())) => None,
        Ok(Err(e)) => {
            warn!(
                session_id = %session_id,
                step,
                "store write task ended abnormally: {e}"
            );
            None
        }
        Err(_) => {
            warn!(
                session_id = %session_id,
                step,
                budget_ms = budget.as_millis() as u64,
                "store write still running past its budget: the turn goes on, the write too"
            );
            Some(ChatEvent::Error {
                message: format!(
                    "Saving this conversation ({step}) is taking longer than {}s. \
                     It will be saved when the database answers; until then a reload \
                     may not show the latest messages.",
                    budget.as_secs()
                ),
                parent_tool_use_id: None,
                code: Some(PERSISTENCE_DELAYED_CODE.to_string()),
                reason: Some(step.to_string()),
                index: None,
                request_id: None,
            })
        }
    }
}

/// The budget of the post-stream steps of one turn, and where an overrun is
/// reported: the server log (always, with the step) and the clients (an
/// `error` event whose `code` says what happened and whose `reason` names
/// the step). Nothing is lost silently.
pub(crate) struct StepBudget {
    pub session_id: String,
    pub budget: Duration,
    pub events_tx: broadcast::Sender<ChatEvent>,
    pub nats: Option<Arc<crate::events::NatsEmitter>>,
}

impl StepBudget {
    /// Run a step under the budget; past it the step is dropped and reported.
    pub async fn run<F: Future>(&self, step: &'static str, fut: F) -> Option<F::Output> {
        self.run_for(step, self.budget, fut).await
    }

    /// [`Self::run`] with a budget of its own (a step that sleeps on purpose).
    pub async fn run_for<F: Future>(
        &self,
        step: &'static str,
        budget: Duration,
        fut: F,
    ) -> Option<F::Output> {
        let out = bounded(&self.session_id, step, budget, fut).await;
        if out.is_none() {
            self.report(
                step,
                STEP_ABANDONED_CODE,
                format!(
                    "A wrap-up step of this turn ({step}) did not finish within {}s and was skipped.",
                    budget.as_secs()
                ),
            );
        }
        out
    }

    /// Wait for a write started by [`spawn_write`], up to the budget. Past it
    /// the turn goes on and the clients learn that the write is late; the
    /// write itself keeps running.
    pub async fn wait_for_write(&self, step: &'static str, write: JoinHandle<()>) {
        if let Some(ChatEvent::Error { message, .. }) =
            await_write(&self.session_id, step, self.budget, write).await
        {
            self.report(step, PERSISTENCE_DELAYED_CODE, message);
        }
    }

    fn report(&self, step: &'static str, code: &str, message: String) {
        let event = ChatEvent::Error {
            message,
            parent_tool_use_id: None,
            code: Some(code.to_string()),
            reason: Some(step.to_string()),
            index: None,
            request_id: None,
        };
        let _ = self.events_tx.send(event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(&self.session_id, event);
        }
    }
}

/// Persist the events a turn collected, in a task of its own ([`spawn_write`]):
/// started as soon as the stream loop is over, so neither a slow step before
/// it nor an abandoned turn can lose them. `None` when there is nothing to write.
pub(crate) fn spawn_persist_events(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
    session_id: &str,
    events_to_persist: Vec<ChatEventRecord>,
) -> Option<JoinHandle<()>> {
    let uuid = session_uuid?;
    if events_to_persist.is_empty() {
        return None;
    }
    let graph = graph.clone();
    Some(spawn_write(
        session_id.to_string(),
        "persist_events",
        async move { graph.store_chat_events(uuid, events_to_persist).await },
    ))
}

/// The step budget of `session_id` (its `post_stream_budget`), or the default.
pub(crate) async fn step_budget_of(
    active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
    session_id: &str,
) -> Duration {
    active_sessions
        .read()
        .await
        .get(session_id)
        .map(|s| s.post_stream_budget)
        .unwrap_or(POST_STREAM_STEP_BUDGET)
}

/// Resolved project context from a single Neo4j lookup, shared across
/// all post-stream consumers (compaction, auto-continue, objective tracker, feedback).
#[derive(Default)]
pub(crate) struct PostStreamContext {
    pub project_slug: Option<String>,
    pub project_id: Option<Uuid>,
}

impl PostStreamContext {
    /// Build from a single get_chat_session + get_project_by_slug call.
    pub async fn build(
        graph: &Arc<dyn GraphStore>,
        session_uuid: Option<Uuid>,
    ) -> PostStreamContext {
        if let Some(uuid) = session_uuid {
            match graph.get_chat_session(uuid).await {
                Ok(Some(node)) => {
                    let pid = if let Some(ref slug) = node.project_slug {
                        graph
                            .get_project_by_slug(slug)
                            .await
                            .ok()
                            .flatten()
                            .map(|p| p.id)
                    } else {
                        None
                    };
                    PostStreamContext {
                        project_slug: node.project_slug,
                        project_id: pid,
                    }
                }
                _ => PostStreamContext {
                    project_slug: None,
                    project_id: None,
                },
            }
        } else {
            PostStreamContext {
                project_slug: None,
                project_id: None,
            }
        }
    }
}

/// Holds all shared state needed by post-stream processing.
///
/// Created once after the retry loop ends, used to call each post-stream
/// handler method in sequence. All fields are cheap `Arc` clones.
pub(crate) struct PostStreamHandler {
    pub graph: Arc<dyn GraphStore>,
    pub pending_messages: Arc<Mutex<VecDeque<PendingMessage>>>,
    pub session_id: String,
    pub session_uuid: Option<Uuid>,
    pub active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    pub events_tx: broadcast::Sender<ChatEvent>,
    pub nats: Option<Arc<crate::events::NatsEmitter>>,
    pub next_seq: Arc<AtomicI64>,
    pub interrupt_flag: Arc<AtomicBool>,
    pub interrupt_token: CancellationToken,
    pub ctx: PostStreamContext,
    pub is_streaming: Arc<AtomicBool>,
    pub streaming_text: Arc<Mutex<String>>,
    pub streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
    pub event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    pub search: Arc<dyn SearchStore>,
    pub auto_continue: Arc<AtomicBool>,
    pub work_log: Arc<Mutex<super::types::SessionWorkLog>>,
}

impl PostStreamHandler {
    /// Emit a ChatEvent to local broadcast + NATS (if configured).
    fn emit_chat(&self, event: ChatEvent) {
        let _ = self.events_tx.send(event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(&self.session_id, event);
        }
    }

    // ── Post-compaction injection ─────────────────────────────────────────

    /// Re-inject project context after compaction so the LLM regains awareness.
    pub async fn handle_post_compaction(&self, needs_injection: bool) {
        if needs_injection && !self.interrupt_flag.load(Ordering::SeqCst) {
            let snapshot = self.work_log.lock().await.snapshot();
            let window = self
                .active_sessions
                .read()
                .await
                .get(&self.session_id)
                .and_then(|s| s.context_window);
            let (hint, recovery_event) = post_compaction_recovery(
                &self.graph,
                &self.session_id,
                self.ctx.project_slug.as_deref(),
                Some(snapshot),
                window,
            )
            .await;
            if let Some(hint) = hint {
                self.pending_messages
                    .lock()
                    .await
                    .push_back(PendingMessage::system_hint(hint));
            }
            self.emit_chat(recovery_event);
        } else if needs_injection {
            debug!(
                "Skipping post-compaction injection for session {} (interrupted)",
                self.session_id
            );
        }
    }

    // ── Interrupt cleanup ─────────────────────────────────────────────────

    /// Send lock-free interrupt signal and emit ToolCancelled events for pending tools.
    pub async fn handle_interrupt_cleanup(
        &self,
        stdin_tx: &Option<tokio::sync::mpsc::Sender<String>>,
        pending_tool_calls: &HashMap<String, Option<String>>,
    ) {
        if !self.interrupt_flag.load(Ordering::SeqCst) {
            return;
        }

        // Lock-free interrupt via stdin_tx
        if let Some(ref tx) = stdin_tx {
            let json = InteractiveClient::build_interrupt_json();
            if let Err(e) = tx.try_send(json) {
                warn!("Failed to send lock-free interrupt to CLI: {}", e);
            } else {
                debug!(
                    "Lock-free interrupt sent to CLI for session {}",
                    self.session_id
                );
            }
        }

        // Emit ToolCancelled for each pending tool
        if !pending_tool_calls.is_empty() {
            info!(
                "Emitting ToolCancelled for {} pending tool(s) in session {}",
                pending_tool_calls.len(),
                self.session_id
            );
            // Each ToolCancelled, then the timing of the call it ends (`tool_clock`).
            let tool_clock = super::tool_clock::ToolClock::for_session(&self.session_id);
            let mut cancel_events = Vec::new();
            for (tool_id, parent) in pending_tool_calls {
                let event = ChatEvent::ToolCancelled {
                    id: tool_id.clone(),
                    parent_tool_use_id: parent.clone(),
                };
                let timing = tool_clock.observe(&event, chrono::Utc::now());
                cancel_events.push(event);
                cancel_events.extend(timing);
            }

            // Their seqs are taken BEFORE they are sent, in the order they are sent.
            let cancel_records: Vec<ChatEventRecord> = match self.session_uuid {
                Some(uuid) => cancel_events
                    .iter()
                    .map(|event| ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq: self.next_seq.fetch_add(1, Ordering::SeqCst),
                        event_type: event.event_type().to_string(),
                        data: serde_json::to_string(event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    })
                    .collect(),
                None => Vec::new(),
            };
            for event in &cancel_events {
                self.emit_chat(event.clone());
            }

            // Persist ToolCancelled events (and their timings) in Neo4j
            if let Some(uuid) = self.session_uuid {
                if let Err(e) = self.graph.store_chat_events(uuid, cancel_records).await {
                    warn!(
                        "Failed to persist ToolCancelled events for session {}: {}",
                        self.session_id, e
                    );
                }
            }
        }
    }

    // ── Auto-continue ─────────────────────────────────────────────────────

    /// Check auto-continue eligibility and enqueue enriched "Continue" if triggered.
    /// Returns `true` if auto-continue was allowed (needed by objective tracker).
    pub async fn handle_auto_continue(&self, hit_error_max_turns: bool) -> bool {
        let auto_continue_allowed = {
            let sessions = self.active_sessions.read().await;
            match sessions.get(&self.session_id) {
                Some(session) => auto_continue_allowed(
                    &self.session_id,
                    hit_error_max_turns,
                    &self.auto_continue,
                    self.interrupt_flag.load(Ordering::SeqCst),
                    &session.auto_continue_count,
                    session.max_auto_continues,
                ),
                None => false,
            }
        };

        if auto_continue_allowed {
            info!(
                "Auto-continue triggered for session {} (delay={}ms)",
                self.session_id, AUTO_CONTINUE_DELAY_MS
            );

            // Emit AutoContinue event
            let ac_event = ChatEvent::AutoContinue {
                session_id: self.session_id.clone(),
                delay_ms: AUTO_CONTINUE_DELAY_MS,
            };
            self.emit_chat(ac_event.clone());

            // Persist the AutoContinue event
            if let Some(uuid) = self.session_uuid {
                let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq,
                    event_type: ac_event.event_type().to_string(),
                    data: serde_json::to_string(&ac_event).unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                };
                let _ = self.graph.store_chat_events(uuid, vec![record]).await;
            }

            // Interruptible delay
            let sleep_cancelled = tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(AUTO_CONTINUE_DELAY_MS)) => false,
                _ = self.interrupt_token.cancelled() => true,
            };

            if !sleep_cancelled && !self.interrupt_flag.load(Ordering::SeqCst) {
                let work_summary = self.work_log.lock().await.to_summary_markdown();
                let continue_msg = continuation_message(
                    &self.graph,
                    self.ctx.project_slug.as_deref(),
                    &work_summary,
                )
                .await;

                self.pending_messages
                    .lock()
                    .await
                    .push_back(PendingMessage::system_hint(continue_msg));
                debug!(
                    "Auto-continue: enqueued enriched 'Continue' for session {}",
                    self.session_id
                );
            } else {
                info!(
                    "Auto-continue cancelled by interrupt for session {}",
                    self.session_id
                );
            }
        }

        auto_continue_allowed
    }

    // ── Objective tracking ────────────────────────────────────────────────

    /// If the agent concluded without using *productive* tools and auto-continue
    /// didn't fire, check for the pending tasks linked to THIS session and inject a
    /// SystemHint reminder ([`objective_reminder`], shared with the agent engine).
    ///
    /// "Productive" excludes conclusive tools (git commit/push/status) — the agent
    /// may be wrapping up prematurely after a commit without finishing remaining tasks.
    pub async fn handle_objective_tracking(
        &self,
        had_productive_tool_use: bool,
        had_conclusive_tool_use: bool,
        auto_continue_allowed: bool,
        hit_error_max_turns: bool,
    ) {
        let counters = {
            let sessions = self.active_sessions.read().await;
            match sessions.get(&self.session_id) {
                Some(session) => ObjectiveCounters {
                    enabled: session.objective_tracking,
                    turns_since: session.objective_reminder_turns_since.clone(),
                    in_a_row: session.objective_reminders_in_a_row.clone(),
                    baseline: session.objective_baseline.clone(),
                },
                None => return,
            }
        };
        let work_log_summary = self.work_log.lock().await.to_summary_markdown();
        let turn = ObjectiveTurn {
            had_productive_tool_use,
            had_conclusive_tool_use,
            auto_continue_allowed,
            hit_error_max_turns,
            interrupted: self.interrupt_flag.load(Ordering::SeqCst),
        };
        if let Some(reminder) = objective_reminder(
            &self.graph,
            self.session_uuid,
            &counters,
            turn,
            work_log_summary,
        )
        .await
        {
            info!(
                "Objective tracker: injecting reminder for session {}",
                self.session_id
            );
            self.pending_messages
                .lock()
                .await
                .push_back(PendingMessage::system_hint(reminder));
        }
    }

    // ── Streaming status ──────────────────────────────────────────────────

    /// Update streaming status and clear buffers if no pending messages.
    /// Returns whether there are pending messages (caller needs this for drain).
    pub async fn finalize_streaming_status(&self) -> bool {
        // "Nothing pending" and "no longer streaming" are decided under the
        // queue lock, as one step. `queue_user_message` takes the same lock to
        // read `is_streaming` and push: without that, a message queued between
        // the emptiness check and the store would sit in a queue nobody drains
        // until the next turn.
        let has_pending = {
            let queue = self.pending_messages.lock().await;
            let has_pending = !queue.is_empty();
            if !has_pending {
                self.is_streaming.store(false, Ordering::SeqCst);
            }
            has_pending
        };

        if !has_pending {
            self.emit_chat(ChatEvent::StreamingStatus {
                is_streaming: false,
            });
            if let Some(ref emitter) = self.event_emitter {
                emitter.emit_updated(
                    crate::events::EntityType::ChatSession,
                    &self.session_id,
                    serde_json::json!({ "is_streaming": false }),
                    None,
                );
            }
            self.streaming_text.lock().await.clear();
            self.streaming_events.lock().await.clear();
            debug!("Stream completed for session {}", self.session_id);
        }

        has_pending
    }

    // ── Memory / feedback / RFC ───────────────────────────────────────────

    /// Record assistant response in memory, spawn feedback extraction and RFC detection.
    ///
    /// Nothing here waits on a store: the memory write (`store_messages`, an
    /// HTTP call without a timeout of its own) runs in a task of its own
    /// ([`spawn_write`]), after the memory lock is released — it used to be
    /// awaited under that lock, so a store that never answered held the turn
    /// AND the next turn's `record_user_message`.
    pub async fn handle_feedback(
        &self,
        assistant_text_parts: &[String],
        memory_manager: &Option<Arc<Mutex<ConversationMemoryManager>>>,
        context_injector: &Option<Arc<ContextInjector>>,
    ) {
        let rfc_accumulator = {
            let sessions = self.active_sessions.read().await;
            sessions
                .get(&self.session_id)
                .map(|s| s.rfc_accumulator.clone())
        };
        let answer = assistant_text_parts.join("");
        record_turn_feedback(
            TurnFeedback {
                graph: &self.graph,
                search: &self.search,
                event_emitter: &self.event_emitter,
                session_id: &self.session_id,
                session_uuid: self.session_uuid,
                project_id: self.ctx.project_id,
                rfc_accumulator,
            },
            &answer,
            memory_manager,
            context_injector,
        )
        .await;
        // What the answer CITES, validated and announced (`cites_resolved`):
        // the other direction from the `refs_resolved` of the user's message.
        self.announce_answer_cites(&answer).await;
    }

    /// Validate the citations of the answer and announce what they resolved to
    /// (`cites_resolved`), persisted like `refs_resolved` so a replay has it.
    ///
    /// The interface needs it to draw a citation the session may NOT read as
    /// plain text: without the event, `#plan:<uuid>` reads as a chip whose link
    /// points at an entity that does not exist. Additive: nothing of the chat
    /// contract changes, and a client that does not know the event ignores it.
    /// Behind `refs_v1`, like the rest of the feature.
    async fn announce_answer_cites(&self, answer: &str) {
        if !crate::refs::flag::from_env() {
            return;
        }
        let Some(event) =
            super::feedback::cited_refs_event(&self.graph, &self.session_id, answer).await
        else {
            return;
        };
        self.emit_chat(event.clone());
        if let Some(uuid) = self.session_uuid {
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
}

/// Where the feedback of a turn goes ([`record_turn_feedback`]).
pub(crate) struct TurnFeedback<'a> {
    pub graph: &'a Arc<dyn GraphStore>,
    pub search: &'a Arc<dyn SearchStore>,
    pub event_emitter: &'a Option<Arc<dyn crate::events::EventEmitter>>,
    pub session_id: &'a str,
    pub session_uuid: Option<Uuid>,
    pub project_id: Option<Uuid>,
    /// The session's RFC accumulator (`None`: no observation processing).
    pub rfc_accumulator: Option<Arc<Mutex<super::observation_detector::RfcAccumulator>>>,
}

/// Records the assistant's answer in memory, spawns feedback extraction and the
/// observation (RFC…) detection — both engines (the Claude Code engine's
/// [`PostStreamHandler::handle_feedback`], the agent engine's end of turn).
///
/// Nothing here waits on a store: the memory write (`store_messages`, an HTTP call
/// without a timeout of its own) runs in a task of its own ([`spawn_write`]), after
/// the memory lock is released.
pub(crate) async fn record_turn_feedback(
    f: TurnFeedback<'_>,
    assistant_text: &str,
    memory_manager: &Option<Arc<Mutex<ConversationMemoryManager>>>,
    context_injector: &Option<Arc<ContextInjector>>,
) {
    // Le registre de récupération (PR1 du plan 42d49a31) : les récupérations
    // de CE tour sont réglées maintenant que la réponse est connue — « utilisé »
    // = cité par la réponse. Avant tout retour anticipé : le règlement ne dépend
    // pas de la mémoire de conversation, et une réponse vide n'a rien cité.
    if let Some(uuid) = f.session_uuid {
        let settled =
            crate::notes::retrieval_ledger::settle_turn(uuid, &cited_note_ids(assistant_text));
        if settled.events > 0 {
            debug!(
                session_id = %uuid,
                events = settled.events,
                used = settled.used,
                "retrieval ledger: turn settled"
            );
        }
    }
    let Some(ref mm) = memory_manager else {
        return;
    };
    if assistant_text.is_empty() {
        return;
    }
    let pending = {
        let mut mm = mm.lock().await;
        mm.record_assistant_message(assistant_text);
        context_injector
            .as_ref()
            .map(|injector| (injector.clone(), mm.take_pending_messages()))
    };

    if let Some(uuid) = f.session_uuid {
        super::feedback::spawn_feedback(
            f.graph.clone(),
            uuid,
            f.project_id,
            assistant_text.to_string(),
            super::feedback::SessionDiscussedCache::new(),
        );

        // Observation auto-detection (RFC + all categories)
        if let Some(rfc_acc) = f.rfc_accumulator {
            super::feedback::spawn_observation_processing(
                f.graph.clone(),
                f.search.clone(),
                f.project_id,
                assistant_text.to_string(),
                rfc_acc,
                Some(uuid),
                f.event_emitter.clone(),
            );
        }
    }

    // Store pending messages via ContextInjector
    if let Some((injector, pending)) = pending {
        if !pending.is_empty() {
            let count = pending.len();
            let session_id = f.session_id.to_string();
            spawn_write(f.session_id.to_string(), "memory_store", async move {
                injector
                    .store_messages(&pending)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                debug!("Stored {} messages for session {}", count, session_id);
                Ok(())
            });
        }
    }
}

/// Les identifiants de notes que la réponse CITE (`#note:<uuid>`, `#rfc:<uuid>` —
/// un RFC est une note), lus par le lecteur unique des citations
/// ([`crate::refs::cite::cited_refs`] : un jeton dans un bloc de code n'est pas
/// une citation). C'est la définition **mesurée** de « utilisé » du registre de
/// récupération : une note injectée dans le prompt n'est pas pour autant utilisée.
pub(crate) fn cited_note_ids(answer: &str) -> Vec<String> {
    use crate::refs::RefKind;
    // Filtre par sorte AVANT le plafond par message : vingt plans cités ne
    // doivent pas cacher la note citée ensuite.
    crate::refs::cite::cited_refs_where(answer, |k| matches!(k, RefKind::Note | RefKind::Rfc))
        .into_iter()
        .map(|r| r.id.to_string())
        .collect()
}

// ── Shared by both engines ─────────────────────────────────────────────

/// The context re-injected after a compaction (`None`: nothing to say) and the
/// `compaction_recovery` event that reports it. The Claude Code engine queues the
/// hint from [`PostStreamHandler::handle_post_compaction`], the agent engine from
/// its end of turn (`ManagerTurnServices::after_turn`): one logic.
pub(crate) async fn post_compaction_recovery(
    graph: &Arc<dyn GraphStore>,
    session_id: &str,
    project_slug: Option<&str>,
    work_log: Option<super::types::SessionWorkLogSnapshot>,
    context_window: Option<u64>,
) -> (Option<String>, ChatEvent) {
    info!(
        "Post-compaction injection triggered for session {}",
        session_id
    );
    let build_start = std::time::Instant::now();
    let builder = super::compaction_context::CompactionContextBuilder::new(graph.clone());
    let build_result = builder.build_for_session(project_slug).await;
    let build_latency_ms = build_start.elapsed().as_millis() as u64;
    let (hint, recovery_success) = match build_result {
        Ok(mut ctx) => {
            // Inject SessionWorkLog snapshot so to_markdown() includes "Work Already Done"
            if let Some(snapshot) = work_log {
                let has_work = !snapshot.files_modified.is_empty()
                    || !snapshot.steps_completed.is_empty()
                    || snapshot.tool_use_count > 0;
                if has_work {
                    ctx.work_log = Some(snapshot);
                }
            }
            let mut hint = ctx.to_markdown();
            // When the recovered context would itself overflow a known, tight window,
            // cutting it mid-section (the `HeldContext::cap` path) recurses the overflow
            // the compaction just hit. Fall back to a bounded checkpoint assembled from
            // the conversation's persisted segments and boundaries — lossy, but bounded,
            // and never without the user's latest turn. No persisted graph material: the
            // normal recovery is kept (the existing behavior).
            if recovered_context_overflows(context_window, hint.chars().count()) {
                if let Some(checkpoint) = segment_checkpoint_from_store(graph, session_id).await {
                    info!(
                        "Post-compaction context would overflow the window for session {}; \
                         re-injecting a bounded segment checkpoint ({} chars) instead",
                        session_id,
                        checkpoint.chars().count()
                    );
                    hint = checkpoint;
                }
            }
            if !hint.is_empty() {
                info!(
                    "Injecting post-compaction context for session {} ({} chars)",
                    session_id,
                    hint.len()
                );
            }
            (hint, true)
        }
        Err(e) => {
            warn!(
                "Failed to build post-compaction context for session {}: {}. Injecting minimal reminder.",
                session_id, e
            );
            let minimal = match project_slug {
                Some(slug) => format!(
                    "<system-reminder>\n# Post-Compaction Context\nYou are working on project \"{}\". Context rebuild failed — continue based on conversation history.\n</system-reminder>",
                    slug
                ),
                None => "<system-reminder>\n# Post-Compaction Context\nYou are in an interactive session. No project context available.\n</system-reminder>".to_string(),
            };
            (minimal, false)
        }
    };
    let event = ChatEvent::CompactionRecovery {
        hint_tokens: (hint.len() as u32) / 3,
        build_latency_ms,
        recovery_success,
    };
    ((!hint.is_empty()).then_some(hint), event)
}

// ── Bounded segment checkpoint (overflow-recovery fallback) ─────────────

/// The share of a known window the recovered context may take, mirroring the cap
/// applied downstream (`manager::HELD_CONTEXT_WINDOW_SHARE`), four characters a
/// token. The recovery is "tight" when the window is known and this share is
/// smaller than what the builder produced.
const RECOVERY_WINDOW_SHARE: f64 = 0.05;

/// A hard cap on the checkpoint, in characters (~a few hundred tokens). It keeps
/// the fallback bounded whatever the conversation size, but it is NOT derived
/// from the window: a window tighter than ~375 tokens may still overflow it. The
/// checkpoint is a lossy last resort, not a promise it fits every window.
const SEGMENT_CHECKPOINT_MAX_CHARS: usize = 1500;

/// Most segments and boundaries (each recent-first) the checkpoint keeps.
const CHECKPOINT_MAX_SEGMENTS: usize = 10;
const CHECKPOINT_MAX_BOUNDARIES: usize = 10;

/// The user's latest turn keeps at most this many characters (its start).
const CHECKPOINT_LAST_TURN_CHARS: usize = 600;

/// Whether a recovered context of `recovered_chars` would overflow the window:
/// only with a KNOWN window whose [`RECOVERY_WINDOW_SHARE`] is smaller than the
/// context. An unknown window is never overflow — the builder's own 6000-char
/// cap is the bound then.
fn recovered_context_overflows(window_tokens: Option<u64>, recovered_chars: usize) -> bool {
    match window_tokens {
        Some(window) => {
            let budget = ((window as f64) * RECOVERY_WINDOW_SHARE * 4.0) as usize;
            recovered_chars > budget
        }
        None => false,
    }
}

/// A bounded, deterministic, lossy checkpoint assembled from the conversation's
/// persisted segments and boundaries — never the raw messages — plus the user's
/// latest turn, listed first so even a later cap keeps it. The "Latest request"
/// section is absent when no user message appears in the bounded scan
/// ([`last_user_turn_text`]). Used ONLY when the normal post-compaction recovery
/// would overflow.
fn bounded_segment_checkpoint(
    segments: &[Segment],
    boundaries: &[Boundary],
    last_turn_text: Option<&str>,
) -> String {
    let mut out = String::with_capacity(SEGMENT_CHECKPOINT_MAX_CHARS);
    out.push_str("<system-reminder>\n# Conversation Checkpoint (bounded)\n");

    // The user's latest turn first: nothing may cut it off. Absent when the
    // scan found no user message (`last_turn_text` is `None`).
    if let Some(text) = last_turn_text {
        if !text.is_empty() {
            out.push_str("\n## Latest request\n");
            out.push_str(&truncate_chars(text, CHECKPOINT_LAST_TURN_CHARS));
            out.push('\n');
        }
    }

    // Recent segments: most recently active first (highest `last_turn`), then axis
    // (enum order) for a stable result.
    if !segments.is_empty() {
        let mut recent: Vec<&Segment> = segments.iter().collect();
        recent.sort_by(|a, b| {
            b.last_turn
                .cmp(&a.last_turn)
                .then(a.dimension.cmp(&b.dimension))
        });
        out.push_str("\n## Recent direction\n");
        for segment in recent.into_iter().take(CHECKPOINT_MAX_SEGMENTS) {
            let labels: Vec<&str> = segment.fingerprint.labels().collect();
            let _ = writeln!(
                out,
                "- {}: {} (turns {}-{})",
                segment.dimension.as_str(),
                labels.join(", "),
                segment.start_turn,
                segment.last_turn
            );
        }
    }

    // Recent boundaries: most recent first (highest `at_turn`), then axis.
    if !boundaries.is_empty() {
        let mut recent: Vec<&Boundary> = boundaries.iter().collect();
        recent.sort_by(|a, b| {
            b.at_turn
                .cmp(&a.at_turn)
                .then(a.dimension.cmp(&b.dimension))
        });
        out.push_str("\n## Direction changes\n");
        for boundary in recent.into_iter().take(CHECKPOINT_MAX_BOUNDARIES) {
            let _ = writeln!(
                out,
                "- {} @ turn {}: {} -> {}",
                boundary.dimension.as_str(),
                boundary.at_turn,
                boundary.from.as_str(),
                boundary.to.as_str()
            );
        }
    }

    out.push_str("</system-reminder>");
    // Final safety: never past the checkpoint budget, and always a closed tag.
    if out.chars().count() > SEGMENT_CHECKPOINT_MAX_CHARS {
        const TAIL: &str = "\n</system-reminder>";
        let keep = SEGMENT_CHECKPOINT_MAX_CHARS.saturating_sub(TAIL.chars().count());
        let mut cut: String = out.chars().take(keep).collect();
        cut.push_str(TAIL);
        out = cut;
    }
    out
}

/// The user's latest turn text, read from the session's persisted events (the last
/// `user_message` in sequence order). `None` when there is none, when none appears
/// in the bounded scan, or when the store cannot be read — the checkpoint's
/// "Latest request" section is then simply absent. Bounded: only the last
/// [`LAST_USER_MSG_SCAN`] events are read.
async fn last_user_turn_text(graph: &Arc<dyn GraphStore>, session_id: &str) -> Option<String> {
    const LAST_USER_MSG_SCAN: i64 = 64;
    let uuid = Uuid::parse_str(session_id).ok()?;
    let total = graph.count_chat_events(uuid).await.ok()?;
    let offset = (total - LAST_USER_MSG_SCAN).max(0);
    let events = graph
        .get_chat_events_paginated(uuid, offset, LAST_USER_MSG_SCAN)
        .await
        .ok()?;
    for record in events.iter().rev() {
        if record.event_type != "user_message" {
            continue;
        }
        if let Ok(ChatEvent::UserMessage { content }) = serde_json::from_str(&record.data) {
            if !content.is_empty() {
                return Some(content);
            }
        }
    }
    None
}

/// A bounded checkpoint from the session's persisted segments and boundaries, or
/// `None` when there is no graph material to checkpoint (the caller keeps the
/// existing behavior) or the store cannot be read.
async fn segment_checkpoint_from_store(
    graph: &Arc<dyn GraphStore>,
    session_id: &str,
) -> Option<String> {
    let segments = graph.get_segments(session_id).await.ok()?;
    let boundaries = graph.get_boundaries(session_id).await.ok()?;
    if segments.is_empty() && boundaries.is_empty() {
        return None;
    }
    let last_turn = last_user_turn_text(graph, session_id).await;
    Some(bounded_segment_checkpoint(
        &segments,
        &boundaries,
        last_turn.as_deref(),
    ))
}

/// Truncate to `max` characters on a UTF-8 boundary, adding "…" when cut.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// The objective-reminder state of a session (both engines keep one).
///
/// In memory only: it is rebuilt when the session is opened again (restart, resume,
/// in-place switch of engine). [`ObjectiveCounters::opening`] decides what a rebuilt state
/// starts with, so that rebuilding it never hands out a fresh allowance of reminders.
#[derive(Clone, Default)]
pub(crate) struct ObjectiveCounters {
    /// Objective tracking is on for the session.
    pub enabled: bool,
    /// Turns since the last reminder (the cooldown).
    pub turns_since: Arc<std::sync::atomic::AtomicU32>,
    /// Reminders injected back to back without real progress in between
    /// ([`made_progress`]); reading, searching or running a shell command is not progress.
    pub in_a_row: Arc<std::sync::atomic::AtomicU32>,
    /// What the session's linked tasks looked like in the store at the last reminder (or
    /// at the first turn after a reopening): the reference [`made_progress`] compares with.
    pub baseline: ProgressBaseline,
}

/// The shared slot of [`ObjectiveCounters::baseline`] (the Claude Code engine keeps it on
/// its `ActiveSession`).
pub(crate) type ProgressBaseline = Arc<std::sync::Mutex<Option<ProgressSnapshot>>>;

/// The state, IN THE STORE, of the tasks linked to a session: per task its status, its
/// completed steps and its linked commits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProgressSnapshot {
    pub tasks: std::collections::BTreeMap<Uuid, TaskProgress>,
}

/// One linked task in a [`ProgressSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaskProgress {
    pub status: crate::neo4j::models::TaskStatus,
    pub completed_steps: std::collections::BTreeSet<Uuid>,
    pub commits: std::collections::BTreeSet<String>,
}

/// Whether the session made REAL progress between two reads of the store: a linked task
/// changed status, one of its steps became completed, a commit was linked to it, or a
/// task became linked (a task moved to `in_progress` by the session is linked then).
///
/// Judged on the store, never on the tool calls: a `task update` that failed or changed
/// nothing, an update of a task that is not this session's, a `git commit --dry-run`, a
/// commit refused by a hook, a `Read`, a `pgrep` leave the store as it was and rearm
/// nothing. Both engines decide with this function (through [`objective_reminder`]).
pub(crate) fn made_progress(before: &ProgressSnapshot, after: &ProgressSnapshot) -> bool {
    after
        .tasks
        .iter()
        .any(|(id, now)| match before.tasks.get(id) {
            None => true,
            Some(was) => {
                was.status != now.status
                    || now
                        .completed_steps
                        .difference(&was.completed_steps)
                        .next()
                        .is_some()
                    || now.commits.difference(&was.commits).next().is_some()
            }
        })
}

/// The [`ProgressSnapshot`] of a session, `None` when it cannot be read (no id, a store
/// error): then nothing is rearmed.
///
/// EVERY linked task is in it, with no cap and in id order, so the snapshot does not depend
/// on the order the store returns the links in, and no historical link can push a task
/// still to do out of it. A task still to do (`Pending`/`InProgress`, the ones a reminder
/// lists) is read with its steps and commits; any other only by its status (its change is
/// what matters). Cost: one task read per link, plus two per task to do, at the end of
/// each turn while the reminders are spent; past the 2 s budget of the caller the
/// snapshot is dropped and nothing is rearmed.
pub(crate) async fn progress_snapshot(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
) -> Option<ProgressSnapshot> {
    use crate::neo4j::models::{StepStatus, TaskStatus};
    let links = graph.get_session_links(session_uuid?).await.ok()?;
    let mut snapshot = ProgressSnapshot::default();
    let mut ids = linked_task_ids(&links);
    ids.sort();
    for id in ids {
        // A linked task deleted since is skipped; any read error drops the whole
        // snapshot (nothing is rearmed on a partial read).
        let task = match graph.get_task(id).await {
            Ok(Some(task)) => task,
            Ok(None) => continue,
            Err(_) => return None,
        };
        let to_do = matches!(task.status, TaskStatus::Pending | TaskStatus::InProgress);
        let (steps, commits) = if to_do {
            (
                graph.get_task_steps(id).await.ok()?,
                graph.get_task_commits(id).await.ok()?,
            )
        } else {
            (vec![], vec![])
        };
        snapshot.tasks.insert(
            id,
            TaskProgress {
                status: task.status,
                completed_steps: steps
                    .iter()
                    .filter(|s| s.status == StepStatus::Completed)
                    .map(|s| s.id)
                    .collect(),
                commits: commits.into_iter().map(|c| c.hash).collect(),
            },
        );
    }
    Some(snapshot)
}

impl ObjectiveCounters {
    /// The counters of a session as it is opened, on either engine.
    ///
    /// A FRESH session starts with the full allowance ([`OBJECTIVE_REMINDER_MAX_IN_A_ROW`]).
    /// A RESUMED one (restart, resume, engine switch) starts with the allowance spent: it gets
    /// no reminder until it makes real progress again. The counter is not stored, so this is
    /// what bounds the reminders across reopenings: with no progress, a session receives at
    /// most `OBJECTIVE_REMINDER_MAX_IN_A_ROW` reminders over its whole life, however many
    /// times it is reopened (a reopening never rearms it; only progress does).
    pub(crate) fn opening(enabled: bool, resumed: bool) -> Self {
        Self {
            enabled,
            turns_since: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            in_a_row: Arc::new(std::sync::atomic::AtomicU32::new(Self::initial_in_a_row(
                resumed,
            ))),
            baseline: ProgressBaseline::default(),
        }
    }

    /// The reference of a REOPENED session, read in the store at the reopening itself
    /// (before its first turn), so that progress made during that first turn is seen.
    /// Empty when the store cannot be read: then the end of the first turn is the
    /// reference.
    pub(crate) async fn reopening_baseline(
        graph: &Arc<dyn GraphStore>,
        session_uuid: Option<Uuid>,
    ) -> ProgressBaseline {
        let now = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            progress_snapshot(graph, session_uuid),
        )
        .await
        .ok()
        .flatten();
        Arc::new(std::sync::Mutex::new(now))
    }

    /// What `in_a_row` starts at when a session is opened ([`Self::opening`]); the
    /// Claude Code engine, which keeps the counters on its `ActiveSession`, reads it too.
    pub(crate) fn initial_in_a_row(resumed: bool) -> u32 {
        if resumed {
            OBJECTIVE_REMINDER_MAX_IN_A_ROW
        } else {
            0
        }
    }
}

/// What a turn did, for the objective tracker.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ObjectiveTurn {
    pub had_productive_tool_use: bool,
    pub had_conclusive_tool_use: bool,
    pub auto_continue_allowed: bool,
    pub hit_error_max_turns: bool,
    pub interrupted: bool,
}

/// The tasks linked DIRECTLY to a session, once each, in the store's order: the task
/// associations (`ASSOCIATED_WITH`) and the tasks the session executes
/// (`AgentExecution -[:EXECUTES]->`), both read by `GraphStore::get_session_links`. A
/// linked PLAN is not used: being attached to a plan does not make every task of the
/// plan the work of this chat.
///
/// Deduplicated HERE, by the consumer: the store's contract is one entry per link —
/// `collect(DISTINCT {id, title, source})` per path, so a task the session executes
/// (`source: runner`) and is associated with (`auto`/`manual`) comes back twice.
pub(crate) fn linked_task_ids(links: &crate::neo4j::models::LinkedSessionInfo) -> Vec<Uuid> {
    let mut seen = std::collections::HashSet::new();
    links
        .linked_tasks
        .iter()
        .filter(|t| seen.insert(t.id))
        .map(|t| t.id)
        .collect()
}

/// The reminder entry of a linked task, `None` unless it is still to do (typed
/// `Pending` or `InProgress`): a link is historical, so a task completed, failed,
/// blocked or interrupted since it was linked is left out. A task without a title is
/// shown by its description.
pub(crate) fn pending_task_info(
    task: &crate::neo4j::models::TaskNode,
    steps: &[crate::neo4j::models::StepNode],
) -> Option<PendingTaskInfo> {
    use crate::neo4j::models::{StepStatus, TaskStatus};
    let status = match task.status {
        TaskStatus::Pending => "pending",
        TaskStatus::InProgress => "in_progress",
        _ => return None,
    };
    let title = task
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| truncate_chars(task.description.trim(), 120));
    Some(PendingTaskInfo {
        title,
        status: status.to_string(),
        pending_steps: steps
            .iter()
            .filter(|s| !matches!(s.status, StepStatus::Completed | StepStatus::Skipped))
            .map(|s| s.description.clone())
            .collect(),
        affected_files: task.affected_files.clone(),
    })
}

/// Most tasks listed in one reminder.
pub(crate) const OBJECTIVE_REMINDER_MAX_TASKS: usize = 4;

/// The pending tasks of THIS session for its reminder ([`linked_task_ids`],
/// [`pending_task_info`]), at most [`OBJECTIVE_REMINDER_MAX_TASKS`]. Empty — hence no
/// reminder — for a session without an id, without a linked task, or whose links cannot
/// be read: never the project's tasks instead.
async fn session_pending_tasks(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
) -> Vec<PendingTaskInfo> {
    let Some(session_uuid) = session_uuid else {
        return vec![];
    };
    let links = match graph.get_session_links(session_uuid).await {
        Ok(links) => links,
        Err(e) => {
            debug!(%session_uuid, error = %e, "Objective tracker: links unreadable, no reminder");
            return vec![];
        }
    };
    let mut tasks = Vec::new();
    for id in linked_task_ids(&links) {
        if tasks.len() >= OBJECTIVE_REMINDER_MAX_TASKS {
            break;
        }
        let task = match graph.get_task(id).await {
            Ok(Some(task)) => task,
            Ok(None) => continue,
            Err(e) => {
                warn!(%session_uuid, task_id = %id, error = %e, "Objective tracker: task unreadable, left out of the reminder");
                continue;
            }
        };
        // Filtered on the typed status BEFORE the cap, and before reading its steps.
        if pending_task_info(&task, &[]).is_none() {
            continue;
        }
        // Steps that cannot be read are not "no step left": the task is left out of the
        // reminder (named in the log) rather than shown with a list that would be wrong.
        let steps = match graph.get_task_steps(id).await {
            Ok(steps) => steps,
            Err(e) => {
                warn!(%session_uuid, task_id = %id, error = %e, "Objective tracker: steps unreadable, task left out of the reminder");
                continue;
            }
        };
        if let Some(info) = pending_task_info(&task, &steps) {
            tasks.push(info);
        }
    }
    tasks
}

/// If the agent concluded without using *productive* tools and auto-continue
/// didn't fire, the reminder of the pending tasks OF THIS SESSION to queue, and the
/// counters updated — both engines ([`PostStreamHandler::handle_objective_tracking`],
/// the agent engine's end of turn).
///
/// "Productive" excludes conclusive tools (git commit/push/status) — the agent
/// may be wrapping up prematurely after a commit without finishing remaining tasks.
/// Only real progress, read in the store ([`made_progress`]), rearms the reminders: a
/// turn that only read, searched or called a tool that changed nothing resets nothing.
pub(crate) async fn objective_reminder(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
    counters: &ObjectiveCounters,
    turn: ObjectiveTurn,
    work_log_summary: String,
) -> Option<String> {
    // Once reminded (or reopened with the reminders spent), the store is read at the end
    // of each turn: real progress since the reference rearms the cooldown and the cap.
    if counters.enabled && counters.in_a_row.load(Ordering::Relaxed) > 0 {
        let now = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            progress_snapshot(graph, session_uuid),
        )
        .await
        .ok()
        .flatten();
        if let Some(now) = now {
            let mut baseline = counters.baseline.lock().unwrap_or_else(|e| e.into_inner());
            match baseline.as_ref() {
                Some(before) if made_progress(before, &now) => {
                    // The cooldown is over (not 0, which the increment below would turn
                    // into 1, inside the cooldown, silencing the next stall).
                    counters
                        .turns_since
                        .store(OBJECTIVE_REMINDER_COOLDOWN, Ordering::Relaxed);
                    counters.in_a_row.store(0, Ordering::Relaxed);
                    *baseline = None;
                }
                Some(_) => {}
                // Reopened with the reminders spent: what the store shows now is the
                // reference.
                None => *baseline = Some(now),
            }
        }
    }
    let cooldown_turns = counters.turns_since.fetch_add(1, Ordering::Relaxed);
    let reminders_in_a_row = counters.in_a_row.load(Ordering::Relaxed);

    // Should we check for pending objectives?
    // Yes when: (a) no productive tools, OR (b) productive + conclusive (agent wrapping up)
    let should_check = !turn.had_productive_tool_use || turn.had_conclusive_tool_use;

    // Fetch pending objectives from graph (only if we might need them)
    let (pending_tasks, work_log_summary) = if should_check
        && !turn.auto_continue_allowed
        && !turn.hit_error_max_turns
        && !turn.interrupted
        && counters.enabled
        && reminders_in_a_row < OBJECTIVE_REMINDER_MAX_IN_A_ROW
        && (cooldown_turns == 0 || cooldown_turns >= OBJECTIVE_REMINDER_COOLDOWN)
    {
        let tasks = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            session_pending_tasks(graph, session_uuid),
        )
        .await
        .unwrap_or_default();
        (tasks, work_log_summary)
    } else {
        (vec![], String::new())
    };

    // Pure decision
    let input = ObjectiveCheckInput {
        had_productive_tool_use: turn.had_productive_tool_use,
        had_conclusive_tool_use: turn.had_conclusive_tool_use,
        auto_continue_allowed: turn.auto_continue_allowed,
        hit_error_max_turns: turn.hit_error_max_turns,
        interrupted: turn.interrupted,
        tracking_enabled: counters.enabled,
        cooldown_turns,
        reminders_in_a_row,
        pending_tasks,
        work_log_summary,
    };
    let reminder = check_objective_reminder(&input)?;
    // The cooldown starts: the next turn reads 1, below the threshold, and is skipped.
    counters.turns_since.store(1, Ordering::Relaxed);
    counters.in_a_row.fetch_add(1, Ordering::Relaxed);
    // The reference progress is judged against: the store as it is at this reminder.
    let now = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        progress_snapshot(graph, session_uuid),
    )
    .await
    .ok()
    .flatten();
    *counters.baseline.lock().unwrap_or_else(|e| e.into_inner()) = now;
    Some(reminder)
}

// ── Pure logic for objective tracking (testable without Graph) ─────────

/// Turns that must pass after a reminder before the next one. 2: the turn right after a
/// reminder (`turns_since` reads 1) never gets one, the turn after it (2) may.
pub(crate) const OBJECTIVE_REMINDER_COOLDOWN: u32 = 2;

/// Most reminders injected back to back without real progress in between
/// ([`made_progress`]). Reads and searches do not reset it, and neither does reopening
/// the session ([`ObjectiveCounters::opening`]).
pub(crate) const OBJECTIVE_REMINDER_MAX_IN_A_ROW: u32 = 2;

/// First line of every objective reminder. It says what the message is (automatic, not typed by
/// the user) so the agent never mistakes it for a request, and so a client can fold it into a
/// card instead of showing it as a user message.
pub(crate) const OBJECTIVE_REMINDER_MARKER: &str =
    "[AUTOMATIC OBJECTIVE REMINDER — NOT USER INPUT]";

/// Summary of a pending task for the structured objective reminder.
#[derive(Debug, Clone)]
pub(crate) struct PendingTaskInfo {
    pub title: String,
    pub status: String,
    pub pending_steps: Vec<String>, // descriptions of non-completed steps
    pub affected_files: Vec<String>, // files this task modifies
}

/// Inputs to the objective tracking decision (decoupled from async/Arc state).
pub(crate) struct ObjectiveCheckInput {
    pub had_productive_tool_use: bool,
    pub had_conclusive_tool_use: bool,
    pub auto_continue_allowed: bool,
    pub hit_error_max_turns: bool,
    pub interrupted: bool,
    pub tracking_enabled: bool,
    pub cooldown_turns: u32,
    /// Reminders already injected back to back with no productive tool use in between.
    pub reminders_in_a_row: u32,
    /// Structured pending tasks (replaces the old objectives oneliner).
    pub pending_tasks: Vec<PendingTaskInfo>,
    /// Work already done this session (from SessionWorkLog::to_summary_markdown).
    pub work_log_summary: String,
}

/// Pure decision function: given the objective check inputs, return the
/// reminder text to inject as SystemHint, or None if no reminder should fire.
///
/// This function has NO side effects and is fully testable without mocks.
pub(crate) fn check_objective_reminder(input: &ObjectiveCheckInput) -> Option<String> {
    // Guard 1: skip if auto-continue or error/interrupt
    if input.auto_continue_allowed || input.hit_error_max_turns || input.interrupted {
        return None;
    }

    // Guard 1b: skip if productive tools were used WITHOUT conclusive tools.
    // When the agent used productive tools AND committed (conclusive), we still
    // want to check — the agent may have done partial work then wrapped up.
    if input.had_productive_tool_use && !input.had_conclusive_tool_use {
        return None;
    }

    // Guard 2: tracking must be enabled for this session
    if !input.tracking_enabled {
        return None;
    }

    // Guard 3: cooldown — only fire on first turn (0) or after COOLDOWN turns
    if input.cooldown_turns != 0 && input.cooldown_turns < OBJECTIVE_REMINDER_COOLDOWN {
        return None;
    }

    // Guard 3b: a reminder answered by text only, every turn, is a loop — stop at the cap.
    if input.reminders_in_a_row >= OBJECTIVE_REMINDER_MAX_IN_A_ROW {
        return None;
    }

    // Guard 4: must have pending objectives
    if input.pending_tasks.is_empty() {
        return None;
    }

    // Build structured reminder
    let mut parts = Vec::new();
    parts.push(OBJECTIVE_REMINDER_MARKER.to_string());
    parts.push("⚠️ **You have not finished your current task. Here is what remains:**".to_string());

    for task in &input.pending_tasks {
        parts.push(format!("\n### {} [{}]", task.title, task.status));

        if !task.pending_steps.is_empty() {
            parts.push("**Pending steps:**".to_string());
            for step in &task.pending_steps {
                parts.push(format!("- [ ] {}", step));
            }
        }

        if !task.affected_files.is_empty() {
            parts.push(format!(
                "**Remaining files:** {}",
                task.affected_files
                    .iter()
                    .map(|f| format!("`{}`", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    // Include work already done (prevents agent from restarting completed work)
    if !input.work_log_summary.is_empty() {
        parts.push("\n### Already done this session".to_string());
        parts.push(input.work_log_summary.clone());
    }

    parts.push("\n**Do NOT conclude. Continue working on the pending steps above.**".to_string());
    parts.push(format!(
        "\n_Automatic message, not from the user. If these tasks are not yours or you are waiting on the user, say so in ONE line and do not repeat it: this reminder stops after {OBJECTIVE_REMINDER_MAX_IN_A_ROW} in a row, until a task or step is updated or a commit is made._"
    ));

    Some(parts.join("\n"))
}

// ── Auto-continue (shared by both engines) ───────────────────────────────

/// Pause between a turn that stopped on its turn limit and its continuation.
pub(crate) const AUTO_CONTINUE_DELAY_MS: u64 = 500;

/// How many times in a row a turn stopped on its turn limit is continued, for every
/// session (interactive and runner, both engines). Auto-continue is always on, with
/// this cap (decision 7b95342a).
pub(crate) const MAX_AUTO_CONTINUES: u32 = 5;

/// The auto-continue of a session as it opens or resumes, on either engine: `(on, cap)`.
/// A runner session always continues; any other session unless the operator's kill
/// switch (`ChatConfig::auto_continue`, `CHAT_AUTO_CONTINUE=false`) is off. Always
/// capped at [`MAX_AUTO_CONTINUES`] in a row. No per-session toggle is read.
pub(crate) fn session_auto_continue(runner: bool, configured_on: bool) -> (bool, u32) {
    (runner || configured_on, MAX_AUTO_CONTINUES)
}

/// Whether a turn that just ended is continued automatically: it stopped on its
/// turn limit, the session's auto-continue is on (it is, unless the operator's
/// `CHAT_AUTO_CONTINUE=false` turned it off) and nobody stopped it.
///
/// `count` is the current chain of continuations: past `max` (0: no limit) the chain
/// stops — the turn stays stopped — and starts again from zero, as it does when a
/// turn ends without hitting its limit. The next chain therefore starts from a turn
/// somebody asked for, never from the capped one.
/// The Claude Code engine (`PostStreamHandler::handle_auto_continue`) and the
/// agent engine (`AgentSessionHandle`) decide with this one function.
pub(crate) fn auto_continue_allowed(
    session_id: &str,
    hit_turn_limit: bool,
    enabled: &AtomicBool,
    interrupted: bool,
    chain: &std::sync::atomic::AtomicU32,
    max: u32,
) -> bool {
    if !hit_turn_limit || !enabled.load(Ordering::Relaxed) || interrupted {
        chain.store(0, Ordering::Relaxed);
        return false;
    }
    let count = chain.fetch_add(1, Ordering::Relaxed) + 1;
    if max > 0 && count > max {
        warn!(
            "Auto-continue cap reached for session {} ({} in a row): the turn stays stopped",
            session_id, max
        );
        chain.store(0, Ordering::Relaxed);
        return false;
    }
    if max > 0 {
        info!("Auto-continue {}/{} for session {}", count, max, session_id);
    }
    true
}

/// The system hint a continued turn starts with: "continue", the task/step
/// context of the session's project, and the work done so far.
pub(crate) async fn continuation_message(
    graph: &Arc<dyn GraphStore>,
    project_slug: Option<&str>,
    work_summary: &str,
) -> String {
    let mut parts =
        vec!["Continue where you left off. Do NOT restart work already done.".to_string()];

    // Append CompactionContext markdown (task/step context)
    if let Some(slug) = project_slug {
        let builder = super::compaction_context::CompactionContextBuilder::new(graph.clone());
        if let Ok(Ok(ctx)) = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            builder.build_for_session(Some(slug)),
        )
        .await
        {
            let md = ctx.to_markdown();
            if !md.is_empty() {
                parts.push(md);
            }
        }
    }

    // Append SessionWorkLog summary (files modified, steps done)
    if !work_summary.is_empty() {
        parts.push(work_summary.to_string());
    }

    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn base_input() -> ObjectiveCheckInput {
        ObjectiveCheckInput {
            had_productive_tool_use: false,
            had_conclusive_tool_use: false,
            auto_continue_allowed: false,
            hit_error_max_turns: false,
            interrupted: false,
            tracking_enabled: true,
            cooldown_turns: 0, // first turn → should fire
            reminders_in_a_row: 0,
            pending_tasks: vec![PendingTaskInfo {
                title: "Implement feature T1".to_string(),
                status: "in_progress".to_string(),
                pending_steps: vec!["Write the handler".to_string(), "Add tests".to_string()],
                affected_files: vec!["src/handler.rs".to_string()],
            }],
            work_log_summary: String::new(),
        }
    }

    #[test]
    fn test_reminder_injected_when_no_tools_and_pending_objectives() {
        let input = base_input();
        let result = check_objective_reminder(&input);
        assert!(result.is_some());
        let msg = result.unwrap();
        assert!(msg.contains("not finished"));
        assert!(msg.contains("Implement feature T1"));
        assert!(msg.contains("Write the handler"));
    }

    #[test]
    fn test_no_reminder_when_productive_tools_were_used() {
        let mut input = base_input();
        input.had_productive_tool_use = true;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_reminder_fires_when_only_conclusive_tools_used() {
        // When the agent only used git commit/push (conclusive tools),
        // had_productive_tool_use is false → reminder should fire
        let input = base_input(); // had_productive_tool_use = false by default
        let result = check_objective_reminder(&input);
        assert!(
            result.is_some(),
            "reminder should fire after conclusive-only tools"
        );
        let msg = result.unwrap();
        assert!(msg.contains("not finished"));
    }

    #[test]
    fn test_reminder_fires_when_productive_and_conclusive_tools_mixed() {
        // Scenario: agent does Edit + Bash(git commit) in the same turn
        // Both productive AND conclusive → reminder should fire because agent is wrapping up
        let mut input = base_input();
        input.had_productive_tool_use = true;
        input.had_conclusive_tool_use = true;
        let result = check_objective_reminder(&input);
        assert!(
            result.is_some(),
            "reminder should fire when agent does work AND commits"
        );
        let msg = result.unwrap();
        assert!(msg.contains("not finished"));
    }

    #[test]
    fn test_cooldown_respected() {
        // turns=0 is first turn → should fire
        let mut input = base_input();
        input.cooldown_turns = 0;
        assert!(check_objective_reminder(&input).is_some());

        // turns=1 is the turn right after a reminder: inside the cooldown → silent
        input.cooldown_turns = 1;
        assert!(check_objective_reminder(&input).is_none());

        // turns=2: the cooldown is over → fires
        input.cooldown_turns = 2;
        assert!(check_objective_reminder(&input).is_some());

        // turns=0 is first turn → should fire
        input.cooldown_turns = 0;
        assert!(check_objective_reminder(&input).is_some());
    }

    #[test]
    fn test_no_reminder_when_tracking_disabled() {
        let mut input = base_input();
        input.tracking_enabled = false;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_no_reminder_without_pending_objectives() {
        let mut input = base_input();
        input.pending_tasks = vec![];
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_no_reminder_when_auto_continue_allowed() {
        let mut input = base_input();
        input.auto_continue_allowed = true;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_no_reminder_when_hit_error_max_turns() {
        let mut input = base_input();
        input.hit_error_max_turns = true;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_no_reminder_when_interrupted() {
        let mut input = base_input();
        input.interrupted = true;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_objective_reminder_enriched() {
        let input = ObjectiveCheckInput {
            had_productive_tool_use: false,
            had_conclusive_tool_use: false,
            auto_continue_allowed: false,
            hit_error_max_turns: false,
            interrupted: false,
            tracking_enabled: true,
            cooldown_turns: 0,
            reminders_in_a_row: 0,
            pending_tasks: vec![
                PendingTaskInfo {
                    title: "Add REST endpoint".to_string(),
                    status: "in_progress".to_string(),
                    pending_steps: vec![
                        "Implement handler".to_string(),
                        "Register route".to_string(),
                    ],
                    affected_files: vec![
                        "src/api/handlers.rs".to_string(),
                        "src/api/routes.rs".to_string(),
                    ],
                },
                PendingTaskInfo {
                    title: "Write integration tests".to_string(),
                    status: "pending".to_string(),
                    pending_steps: vec!["Add test for GET /api/foo".to_string()],
                    affected_files: vec!["tests/api.rs".to_string()],
                },
            ],
            work_log_summary: "**Files modified (2):** `src/main.rs`, `src/lib.rs`".to_string(),
        };

        let result = check_objective_reminder(&input);
        assert!(result.is_some(), "reminder should fire");
        let msg = result.unwrap();

        // Must contain the "not finished" header
        assert!(
            msg.contains("not finished"),
            "missing 'not finished' in: {}",
            msg
        );

        // Must contain pending steps
        assert!(
            msg.contains("Implement handler"),
            "missing step in: {}",
            msg
        );
        assert!(msg.contains("Register route"), "missing step in: {}", msg);
        assert!(
            msg.contains("Add test for GET /api/foo"),
            "missing step from second task in: {}",
            msg
        );

        // Must contain affected files
        assert!(
            msg.contains("`src/api/handlers.rs`"),
            "missing affected file in: {}",
            msg
        );

        // Must contain "Already done" section with work log
        assert!(
            msg.contains("Already done"),
            "missing 'Already done' section in: {}",
            msg
        );
        assert!(
            msg.contains("src/main.rs"),
            "missing work_log file in: {}",
            msg
        );

        // Reminder should be substantial (>200 chars)
        assert!(
            msg.len() > 200,
            "reminder too short ({} chars): {}",
            msg.len(),
            msg
        );
    }

    #[test]
    fn test_reminder_is_labelled_as_automatic_on_its_first_line() {
        let msg = check_objective_reminder(&base_input()).expect("reminder");
        assert_eq!(msg.lines().next(), Some(OBJECTIVE_REMINDER_MARKER));
        assert!(
            msg.contains("not from the user"),
            "footer must say it is automatic"
        );
    }

    #[test]
    fn test_reminder_silent_once_the_cap_is_reached() {
        let mut input = base_input();
        input.reminders_in_a_row = OBJECTIVE_REMINDER_MAX_IN_A_ROW - 1;
        assert!(check_objective_reminder(&input).is_some());
        input.reminders_in_a_row = OBJECTIVE_REMINDER_MAX_IN_A_ROW;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_objective_reminder_contains_do_not_conclude() {
        let input = base_input();
        let msg = check_objective_reminder(&input).unwrap();
        assert!(
            msg.contains("Do NOT conclude"),
            "Missing 'Do NOT conclude' directive in: {}",
            msg
        );
    }

    #[test]
    fn test_objective_reminder_no_work_log_no_already_done_section() {
        let input = base_input(); // work_log_summary is empty
        let msg = check_objective_reminder(&input).unwrap();
        assert!(
            !msg.contains("Already done"),
            "Should not have 'Already done' section when work_log is empty: {}",
            msg
        );
    }

    #[test]
    fn test_objective_reminder_multiple_tasks_all_shown() {
        let input = ObjectiveCheckInput {
            had_productive_tool_use: false,
            had_conclusive_tool_use: false,
            auto_continue_allowed: false,
            hit_error_max_turns: false,
            interrupted: false,
            tracking_enabled: true,
            cooldown_turns: 0,
            reminders_in_a_row: 0,
            pending_tasks: vec![
                PendingTaskInfo {
                    title: "Task A".to_string(),
                    status: "in_progress".to_string(),
                    pending_steps: vec!["Step A1".to_string()],
                    affected_files: vec!["a.rs".to_string()],
                },
                PendingTaskInfo {
                    title: "Task B".to_string(),
                    status: "pending".to_string(),
                    pending_steps: vec![],
                    affected_files: vec!["b.rs".to_string()],
                },
            ],
            work_log_summary: String::new(),
        };
        let msg = check_objective_reminder(&input).unwrap();
        assert!(msg.contains("Task A"), "Missing Task A: {}", msg);
        assert!(msg.contains("Task B"), "Missing Task B: {}", msg);
        assert!(msg.contains("`a.rs`"), "Missing file a.rs: {}", msg);
        assert!(msg.contains("`b.rs`"), "Missing file b.rs: {}", msg);
    }
}

#[cfg(test)]
mod objective_scope_tests {
    use super::*;
    use crate::neo4j::models::{LinkedSessionInfo, LinkedTaskInfo, StepStatus, TaskStatus};
    use uuid::Uuid;

    #[test]
    fn only_a_task_still_to_do_is_reminded_by_its_typed_status() {
        let mut task = crate::test_helpers::test_task_titled("T");
        task.status = TaskStatus::InProgress;
        assert_eq!(pending_task_info(&task, &[]).unwrap().status, "in_progress");
        task.status = TaskStatus::Pending;
        assert_eq!(pending_task_info(&task, &[]).unwrap().status, "pending");
        for st in [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Blocked,
            TaskStatus::Interrupted,
        ] {
            task.status = st;
            assert!(pending_task_info(&task, &[]).is_none());
        }
    }

    #[test]
    fn a_task_without_title_is_named_by_its_description_and_its_done_steps_are_left_out() {
        let mut task = crate::test_helpers::test_task();
        task.title = None;
        task.description = "Do the thing".into();
        let mut done = crate::test_helpers::test_step(1, "done step");
        done.status = StepStatus::Completed;
        let todo = crate::test_helpers::test_step(2, "todo step");
        let info = pending_task_info(&task, &[done, todo]).unwrap();
        assert_eq!(info.title, "Do the thing");
        assert_eq!(info.pending_steps, vec!["todo step".to_string()]);
    }

    #[test]
    fn linked_tasks_are_deduplicated_and_plans_are_not_used() {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let t = |id| LinkedTaskInfo {
            id,
            title: String::new(),
            source: "auto".into(),
        };
        let links = LinkedSessionInfo {
            linked_tasks: vec![t(a), t(b), t(a)],
            linked_plans: vec![crate::neo4j::models::LinkedPlanInfo {
                id: Uuid::new_v4(),
                title: "plan".into(),
                source: "auto".into(),
            }],
            linked_rfcs: vec![],
        };
        assert_eq!(linked_task_ids(&links), vec![a, b]);
    }

    fn progress(status: TaskStatus, steps: &[Uuid], commits: &[&str]) -> TaskProgress {
        TaskProgress {
            status,
            completed_steps: steps.iter().copied().collect(),
            commits: commits.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn progress_is_a_change_in_the_store_of_a_linked_task() {
        let (t, step) = (Uuid::new_v4(), Uuid::new_v4());
        let snap = |p: TaskProgress| ProgressSnapshot {
            tasks: [(t, p)].into_iter().collect(),
        };
        let before = snap(progress(TaskStatus::InProgress, &[], &["aaa"]));
        // Nothing changed (a failed or no-op update, a dry-run, a refused commit, a Read).
        assert!(!made_progress(&before, &before.clone()));
        // A status change, a step completed, a commit linked, a task newly linked.
        assert!(made_progress(
            &before,
            &snap(progress(TaskStatus::Completed, &[], &["aaa"]))
        ));
        assert!(made_progress(
            &before,
            &snap(progress(TaskStatus::InProgress, &[step], &["aaa"]))
        ));
        assert!(made_progress(
            &before,
            &snap(progress(TaskStatus::InProgress, &[], &["aaa", "bbb"]))
        ));
        let mut more = before.clone();
        more.tasks
            .insert(Uuid::new_v4(), progress(TaskStatus::InProgress, &[], &[]));
        assert!(made_progress(&before, &more));
        // A task no longer linked is not progress.
        assert!(!made_progress(&before, &ProgressSnapshot::default()));
    }

    #[test]
    fn a_reopened_session_starts_with_its_reminders_spent() {
        use std::sync::atomic::Ordering;
        let fresh = ObjectiveCounters::opening(true, false);
        assert_eq!(fresh.in_a_row.load(Ordering::Relaxed), 0);
        let reopened = ObjectiveCounters::opening(true, true);
        assert_eq!(
            reopened.in_a_row.load(Ordering::Relaxed),
            OBJECTIVE_REMINDER_MAX_IN_A_ROW
        );
        assert!(!ObjectiveCounters::opening(false, false).enabled);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::chat::types::PendingMessageKind;
    use crate::meilisearch::mock::MockSearchStore;
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::GraphStore;
    use crate::test_helpers::{test_project_named, test_task_titled};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32};
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::sync::{broadcast, Mutex, RwLock};
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    /// A no-op transport for tests — the InteractiveClient is stored in
    /// ActiveSession but never actually used by `handle_objective_tracking`.
    struct NoopTransport;

    #[async_trait::async_trait]
    impl nexus_claude::transport::Transport for NoopTransport {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        async fn connect(&mut self) -> nexus_claude::Result<()> {
            Ok(())
        }
        async fn send_message(
            &mut self,
            _message: nexus_claude::transport::InputMessage,
        ) -> nexus_claude::Result<()> {
            Ok(())
        }
        fn receive_messages(
            &mut self,
        ) -> std::pin::Pin<
            Box<
                dyn futures::Stream<Item = nexus_claude::Result<nexus_claude::Message>>
                    + Send
                    + 'static,
            >,
        > {
            Box::pin(futures::stream::empty())
        }
        async fn send_control_request(
            &mut self,
            _request: nexus_claude::ControlRequest,
        ) -> nexus_claude::Result<()> {
            Ok(())
        }
        async fn receive_control_response(
            &mut self,
        ) -> nexus_claude::Result<Option<nexus_claude::ControlResponse>> {
            Ok(None)
        }
        async fn send_sdk_control_request(
            &mut self,
            _request: serde_json::Value,
        ) -> nexus_claude::Result<()> {
            Ok(())
        }
        async fn send_sdk_control_response(
            &mut self,
            _response: serde_json::Value,
        ) -> nexus_claude::Result<()> {
            Ok(())
        }
        fn is_connected(&self) -> bool {
            false
        }
        async fn disconnect(&mut self) -> nexus_claude::Result<()> {
            Ok(())
        }
    }

    /// Helper to build a PostStreamHandler with mock backends.
    async fn build_handler(
        graph: Arc<MockGraphStore>,
        project_slug: Option<String>,
        session_id: &str,
        objective_tracking: bool,
    ) -> PostStreamHandler {
        let (events_tx, _rx) = broadcast::channel(16);
        let pending_messages = Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Insert a minimal ActiveSession with objective_tracking.
        // We use a NoopTransport so we don't need the real Claude CLI binary —
        // the client is never actually used by handle_objective_tracking.
        {
            let client = nexus_claude::InteractiveClient::from_transport(Box::new(NoopTransport));

            let session = ActiveSession {
                context_window: None,
                anchor: Default::default(),
                mirror_tap: None,
                events_tx: events_tx.clone(),
                last_activity: Instant::now(),
                cli_session_id: None,
                client: Arc::new(Mutex::new(client)),
                interrupt_flag: Arc::new(AtomicBool::new(false)),
                memory_manager: None,
                next_seq: Arc::new(AtomicI64::new(0)),
                pending_messages: pending_messages.clone(),
                is_streaming: Arc::new(AtomicBool::new(false)),
                streaming_text: Arc::new(Mutex::new(String::new())),
                streaming_events: Arc::new(Mutex::new(Vec::new())),
                permission_mode: None,
                model: None,
                protocol_run_id: None,
                protocol_state: None,
                sdk_control_rx: Arc::new(Mutex::new(None)),
                stdin_tx: None,
                child_pid: None,
                nats_cancel: CancellationToken::new(),
                interrupt_token: CancellationToken::new(),
                turn_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                result_cursor: Default::default(),
                stream_task: Arc::new(std::sync::Mutex::new(None)),
                post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
                pending_permission_inputs: Arc::new(Mutex::new(HashMap::new())),
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
                    super::super::observation_detector::RfcAccumulator::new(),
                )),
                objective_tracking,
                objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
                objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
                objective_baseline: Default::default(),
                reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
                work_log: Arc::new(Mutex::new(crate::chat::types::SessionWorkLog::default())),
                oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
                oob_trigger_cap: crate::chat::manager::OOB_TRIGGER_CAP_INTERACTIVE,
                oob_trigger_window: std::time::Duration::from_secs(
                    crate::chat::manager::OOB_TRIGGER_WINDOW_SECS,
                ),
                oob_capped_warned: Arc::new(AtomicBool::new(false)),
                cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
                cancel_tools_cap: crate::chat::manager::CANCEL_TOOLS_CAP,
                cancel_tools_window: std::time::Duration::from_secs(
                    crate::chat::manager::CANCEL_TOOLS_WINDOW_SECS,
                ),
                active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
                tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
                cli_background_tasks: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
                cancel_task_cap: crate::chat::manager::CANCEL_TASK_CAP,
                cancel_task_window: std::time::Duration::from_secs(
                    crate::chat::manager::CANCEL_TASK_WINDOW_SECS,
                ),
            };
            active_sessions
                .write()
                .await
                .insert(session_id.to_string(), session);
        }

        PostStreamHandler {
            graph: graph.clone(),
            pending_messages,
            session_id: session_id.to_string(),
            session_uuid: Some(Uuid::new_v4()),
            active_sessions,
            events_tx,
            nats: None,
            next_seq: Arc::new(AtomicI64::new(0)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            interrupt_token: CancellationToken::new(),
            ctx: PostStreamContext {
                project_slug,
                project_id: None,
            },
            is_streaming: Arc::new(AtomicBool::new(false)),
            streaming_text: Arc::new(Mutex::new(String::new())),
            streaming_events: Arc::new(Mutex::new(Vec::new())),
            event_emitter: None,
            search: Arc::new(MockSearchStore::new()),
            auto_continue: Arc::new(AtomicBool::new(false)),
            work_log: Arc::new(Mutex::new(crate::chat::types::SessionWorkLog::default())),
        }
    }

    /// Link `tasks` to `session` the way `get_session_links` reads them (the mock's
    /// `create_associated_with` is a no-op: the links are seeded explicitly).
    async fn link_tasks(graph: &MockGraphStore, session: Uuid, tasks: &[Uuid]) {
        use crate::neo4j::models::{LinkedSessionInfo, LinkedTaskInfo};
        let mut links = graph.session_links.write().await;
        let entry = links
            .entry(session)
            .or_insert_with(LinkedSessionInfo::default);
        for id in tasks {
            entry.linked_tasks.push(LinkedTaskInfo {
                id: *id,
                title: String::new(),
                source: "auto".into(),
            });
        }
    }

    /// The tasks of the project seeded by [`seed_graph_with_pending_tasks`].
    async fn project_task_ids(graph: &MockGraphStore) -> Vec<Uuid> {
        graph.tasks.read().await.keys().copied().collect()
    }

    /// A handler whose session is linked to every task seeded so far.
    async fn linked_handler(
        graph: Arc<MockGraphStore>,
        slug: String,
        sid: &str,
    ) -> PostStreamHandler {
        let handler = build_handler(graph.clone(), Some(slug), sid, true).await;
        let tasks = project_task_ids(&graph).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &tasks).await;
        handler
    }

    /// Populate mock graph with a project, an in_progress plan, and pending tasks.
    async fn seed_graph_with_pending_tasks(graph: &MockGraphStore) -> String {
        use crate::neo4j::models::{PlanNode, PlanStatus, TaskStatus};

        let project = test_project_named("test-project");
        let slug = project.slug.clone();
        graph.create_project(&project).await.unwrap();

        let mut plan = PlanNode::new_for_project(
            "Active Plan".to_string(),
            "Plan with pending tasks".to_string(),
            "test".to_string(),
            50,
            project.id,
        );
        plan.status = PlanStatus::InProgress;
        graph.create_plan(&plan).await.unwrap();
        // Also update via trait method so the mock stores it with InProgress status
        graph
            .update_plan_status(plan.id, PlanStatus::InProgress)
            .await
            .unwrap();

        let mut task1 = test_task_titled("Fix the bug");
        task1.status = TaskStatus::Pending;
        graph.create_task(plan.id, &task1).await.unwrap();

        let mut task2 = test_task_titled("Write tests");
        task2.status = TaskStatus::Pending;
        graph.create_task(plan.id, &task2).await.unwrap();

        slug
    }

    #[tokio::test]
    async fn test_integration_objective_reminder_injected_with_pending_tasks() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;

        // objective_tracking enabled, the session linked to the tasks
        let handler = linked_handler(graph, slug, "test-session-1").await;

        // Call with had_tool_use=false → should inject reminder
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;

        let pending = handler.pending_messages.lock().await;
        assert_eq!(pending.len(), 1, "Expected 1 pending SystemHint message");
        let msg = &pending[0];
        assert_eq!(msg.kind, PendingMessageKind::SystemHint);
        assert!(
            msg.content.contains("not finished"),
            "Reminder text missing 'not finished': {}",
            msg.content
        );
        assert!(
            msg.content.contains("Fix the bug") || msg.content.contains("Write tests"),
            "Should contain task titles: {}",
            msg.content
        );
    }

    #[tokio::test]
    async fn test_integration_no_reminder_when_tools_used() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;

        let handler = linked_handler(graph, slug, "test-session-2").await;

        // Call with had_tool_use=true → no reminder
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;

        let pending = handler.pending_messages.lock().await;
        assert!(
            pending.is_empty(),
            "No reminder should be injected when tools were used"
        );
    }

    #[tokio::test]
    async fn test_integration_no_reminder_when_tracking_disabled() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;

        let handler = build_handler(
            graph.clone(),
            Some(slug),
            "test-session-3",
            false, // tracking disabled
        )
        .await;

        let tasks = project_task_ids(&graph).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &tasks).await;
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;

        let pending = handler.pending_messages.lock().await;
        assert!(
            pending.is_empty(),
            "No reminder when objective_tracking is disabled"
        );
    }

    #[tokio::test]
    async fn test_integration_no_reminder_without_plans() {
        let graph = Arc::new(MockGraphStore::new());

        // Create project but NO plans
        let project = test_project_named("empty-project");
        let slug = project.slug.clone();
        graph.create_project(&project).await.unwrap();

        let handler = build_handler(graph, Some(slug), "test-session-4", true).await;

        handler
            .handle_objective_tracking(false, false, false, false)
            .await;

        let pending = handler.pending_messages.lock().await;
        assert!(
            pending.is_empty(),
            "No reminder when there are no in_progress plans"
        );
    }

    #[tokio::test]
    async fn test_integration_cooldown_prevents_consecutive_reminders() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;

        let handler = linked_handler(graph, slug, "test-session-5").await;

        // First call (turn 0) → should fire
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            1,
            "First call should inject reminder"
        );

        // Clear pending for next check
        handler.pending_messages.lock().await.clear();

        // (f) The turn right after a reminder never gets one (cooldown).
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            0,
            "the turn right after a reminder is inside the cooldown"
        );

        // The turn after it may.
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        assert_eq!(handler.pending_messages.lock().await.len(), 1);
    }

    /// The loop that flooded a chat: a text-only answer to a reminder is followed by the same
    /// reminder on every turn. After OBJECTIVE_REMINDER_MAX_IN_A_ROW the tracker goes quiet, and
    /// speaks again only once the agent has worked.
    #[tokio::test]
    async fn test_integration_reminders_stop_at_the_cap_until_the_agent_works() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        let handler = linked_handler(graph.clone(), slug, "test-session-cap").await;

        let mut injected = 0;
        for _ in 0..(OBJECTIVE_REMINDER_MAX_IN_A_ROW * 2 + 3) {
            handler
                .handle_objective_tracking(false, false, false, false)
                .await;
            injected += handler.pending_messages.lock().await.len();
            handler.pending_messages.lock().await.clear();
        }
        assert_eq!(
            injected as u32, OBJECTIVE_REMINDER_MAX_IN_A_ROW,
            "text-only turns must not be reminded past the cap"
        );

        // The agent progresses: a linked task changes status IN THE STORE.
        complete_one_linked_task(&graph, &handler).await;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        assert_eq!(handler.pending_messages.lock().await.len(), 0);

        // Next stall is reminded afresh.
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            1,
            "after real progress, a new stall gets a reminder again"
        );
    }

    // ── Objective reminder scoped to the session (legacy engine's handler) ──────────

    /// A task of its own plan in the mock, with `status`; `title: None` = a task without title.
    async fn seed_task(
        graph: &MockGraphStore,
        status: crate::neo4j::models::TaskStatus,
        title: Option<&str>,
    ) -> Uuid {
        let mut task = test_task_titled(title.unwrap_or("untitled"));
        task.title = title.map(str::to_string);
        task.description = format!("Description of {}", title.unwrap_or("the untitled task"));
        task.status = status;
        graph.create_task(Uuid::new_v4(), &task).await.unwrap();
        task.id
    }

    /// One task linked to the handler's session passes to `completed` in the store.
    async fn complete_one_linked_task(graph: &MockGraphStore, handler: &PostStreamHandler) {
        let links = graph
            .get_session_links(handler.session_uuid.unwrap())
            .await
            .unwrap();
        let id = links.linked_tasks[0].id;
        graph.tasks.write().await.get_mut(&id).unwrap().status =
            crate::neo4j::models::TaskStatus::Completed;
    }

    async fn reminders(handler: &PostStreamHandler) -> Vec<String> {
        handler
            .pending_messages
            .lock()
            .await
            .drain(..)
            .map(|m| m.content)
            .collect()
    }

    /// A text-only turn (the agent concluded).
    async fn text_turn(handler: &PostStreamHandler) -> Vec<String> {
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        reminders(handler).await
    }

    /// (a) A session linked to nothing gets NO reminder, even though its project has
    /// pending and in-progress tasks: never the project's tasks instead.
    #[tokio::test]
    async fn a_session_without_linked_tasks_is_never_reminded() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        seed_task(&graph, TaskStatus::InProgress, Some("Someone else's work")).await;
        let handler = build_handler(graph, Some(slug), "unlinked", true).await;
        for _ in 0..4 {
            assert!(text_turn(&handler).await.is_empty());
        }
    }

    /// (b) Two chats of the same project: each is reminded of ITS task only, never the
    /// other's, nor the project's other tasks (more than the 4 a reminder lists).
    #[tokio::test]
    async fn two_chats_of_one_project_each_see_only_their_own_tasks() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        for i in 0..6 {
            seed_task(&graph, TaskStatus::Pending, Some(&format!("Foreign {i}"))).await;
        }
        let task_a = seed_task(&graph, TaskStatus::InProgress, Some("Task of chat A")).await;
        let task_b = seed_task(&graph, TaskStatus::Pending, Some("Task of chat B")).await;
        let a = build_handler(graph.clone(), Some(slug.clone()), "chat-a", true).await;
        let b = build_handler(graph.clone(), Some(slug), "chat-b", true).await;
        link_tasks(&graph, a.session_uuid.unwrap(), &[task_a]).await;
        link_tasks(&graph, b.session_uuid.unwrap(), &[task_b]).await;

        let ra = text_turn(&a).await;
        let rb = text_turn(&b).await;
        assert_eq!(ra.len(), 1, "{ra:?}");
        assert_eq!(rb.len(), 1, "{rb:?}");
        assert!(ra[0].contains("Task of chat A") && !ra[0].contains("Task of chat B"));
        assert!(rb[0].contains("Task of chat B") && !rb[0].contains("Task of chat A"));
        for r in [&ra[0], &rb[0]] {
            assert!(
                !r.contains("Foreign") && !r.contains("Fix the bug") && !r.contains("Write tests"),
                "no task foreign to the chat: {r}"
            );
        }
    }

    /// (c) An `in_progress` linked task is reminded (the old filter compared with
    /// "inprogress" and dropped it), shown with its snake_case status.
    #[tokio::test]
    async fn an_in_progress_linked_task_is_reminded() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let task = seed_task(&graph, TaskStatus::InProgress, Some("Ongoing work")).await;
        let handler = build_handler(graph.clone(), None, "in-progress", true).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &[task]).await;
        let r = text_turn(&handler).await;
        assert_eq!(r.len(), 1, "{r:?}");
        assert!(r[0].contains("### Ongoing work [in_progress]"), "{}", r[0]);
    }

    /// Links are historical: a linked task completed, failed, blocked or interrupted
    /// since is not reminded.
    #[tokio::test]
    async fn linked_tasks_that_are_no_longer_to_do_are_left_out() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let mut ids = vec![];
        for st in [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Blocked,
            TaskStatus::Interrupted,
        ] {
            ids.push(seed_task(&graph, st, Some("Old work")).await);
        }
        let handler = build_handler(graph.clone(), None, "historical", true).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &ids).await;
        assert!(text_turn(&handler).await.is_empty());
    }

    /// A task linked twice (an association AND an execution) appears once.
    #[tokio::test]
    async fn a_task_linked_twice_appears_once() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let task = seed_task(&graph, TaskStatus::Pending, Some("Twice linked")).await;
        let handler = build_handler(graph.clone(), None, "twice", true).await;
        // One entry per (task, source), as Neo4j's `get_session_links` answers: the
        // runner's execution AND an association of the same task.
        {
            use crate::neo4j::models::LinkedTaskInfo;
            let mut links = graph.session_links.write().await;
            let entry = links.entry(handler.session_uuid.unwrap()).or_default();
            for source in ["runner", "auto"] {
                entry.linked_tasks.push(LinkedTaskInfo {
                    id: task,
                    title: "Twice linked".into(),
                    source: source.into(),
                });
            }
        }
        let r = text_turn(&handler).await;
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].matches("Twice linked").count(), 1, "{}", r[0]);
    }

    /// A linked task without a title is shown by its description, not dropped.
    #[tokio::test]
    async fn a_linked_task_without_title_is_shown_by_its_description() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let task = seed_task(&graph, TaskStatus::InProgress, None).await;
        let handler = build_handler(graph.clone(), None, "untitled", true).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &[task]).await;
        let r = text_turn(&handler).await;
        assert_eq!(r.len(), 1);
        assert!(
            r[0].contains("Description of the untitled task"),
            "{}",
            r[0]
        );
    }

    /// The links cannot be read (a store error), or the session has no id: no
    /// reminder, never a fallback on the project's tasks.
    #[tokio::test]
    async fn unreadable_links_or_no_session_id_mean_no_reminder() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        let task = seed_task(&graph, TaskStatus::InProgress, Some("Mine")).await;
        let handler = build_handler(graph.clone(), Some(slug.clone()), "err", true).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &[task]).await;
        graph
            .fail_session_links
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(text_turn(&handler).await.is_empty());

        graph
            .fail_session_links
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut no_id = build_handler(graph.clone(), Some(slug), "no-id", true).await;
        no_id.session_uuid = None;
        assert!(text_turn(&no_id).await.is_empty());
    }

    /// (d) reminder → Read → end, repeated: a turn that only reads is not progress, so
    /// the session gets at most OBJECTIVE_REMINDER_MAX_IN_A_ROW reminders. (The old
    /// tracker reset its counter on any non-conclusive tool: one reminder per cycle.)
    /// (e) Real progress rearms it.
    #[tokio::test]
    async fn reminder_read_end_repeated_is_bounded_and_only_progress_rearms() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        let handler = linked_handler(graph.clone(), slug, "read-loop").await;
        let mut total = 0;
        for _ in 0..6 {
            total += text_turn(&handler).await.len();
            // The turn answering the reminder: Read / `pgrep` — productive, no progress.
            handler
                .handle_objective_tracking(true, false, false, false)
                .await;
            total += reminders(&handler).await.len();
        }
        assert_eq!(total as u32, OBJECTIVE_REMINDER_MAX_IN_A_ROW);

        // A linked task really changes status in the store: progress.
        complete_one_linked_task(&graph, &handler).await;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        assert!(
            reminders(&handler).await.is_empty(),
            "a working turn is not reminded"
        );
        assert_eq!(text_turn(&handler).await.len(), 1, "rearmed by progress");
    }

    /// Spend the reminders of `handler`'s session: text-only turns until the cap.
    async fn spend_reminders(handler: &PostStreamHandler) {
        let mut n = 0;
        for _ in 0..(OBJECTIVE_REMINDER_MAX_IN_A_ROW * 2 + 1) {
            n += text_turn(handler).await.len();
        }
        assert_eq!(n as u32, OBJECTIVE_REMINDER_MAX_IN_A_ROW);
    }

    /// Calls that change nothing in the store — a `task update` that failed or kept the
    /// same status, a `git commit --dry-run`, a commit refused by a hook (on this engine
    /// the handler never sees the calls: only the store can tell) — and a status change
    /// of a task NOT linked to the session rearm nothing.
    #[tokio::test]
    async fn no_change_of_a_linked_task_in_the_store_rearms_nothing() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        let foreign = seed_task(&graph, TaskStatus::Pending, Some("Not this chat's")).await;
        let handler = build_handler(graph.clone(), Some(slug), "no-rearm", true).await;
        let mine = seed_task(&graph, TaskStatus::InProgress, Some("Mine")).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &[mine]).await;
        spend_reminders(&handler).await;

        // Same status written again (no-op update), a working turn: nothing changes.
        graph.tasks.write().await.get_mut(&mine).unwrap().status = TaskStatus::InProgress;
        handler
            .handle_objective_tracking(true, true, false, false)
            .await;
        // A task of someone else changes status.
        graph.tasks.write().await.get_mut(&foreign).unwrap().status = TaskStatus::Completed;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        for _ in 0..4 {
            assert!(text_turn(&handler).await.is_empty(), "nothing rearmed it");
        }

        // The positive case: the linked task really changes status.
        graph.tasks.write().await.get_mut(&mine).unwrap().status = TaskStatus::Blocked;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        // (Blocked is not to do: link another pending task so a reminder has content.)
        let next = seed_task(&graph, TaskStatus::Pending, Some("Next of mine")).await;
        link_tasks(&graph, handler.session_uuid.unwrap(), &[next]).await;
        assert_eq!(
            text_turn(&handler).await.len(),
            1,
            "rearmed by the real change"
        );
    }

    /// A task whose steps cannot be read is left out of the reminder (never shown as a
    /// task with no step left); a task with NO steps is shown.
    #[tokio::test]
    async fn unreadable_steps_leave_the_task_out_an_empty_list_does_not() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let task = seed_task(&graph, TaskStatus::InProgress, Some("Stepless")).await;
        let empty = build_handler(graph.clone(), None, "steps-empty", true).await;
        link_tasks(&graph, empty.session_uuid.unwrap(), &[task]).await;
        let shown = text_turn(&empty).await;
        assert_eq!(shown.len(), 1, "{shown:?}");
        assert!(shown[0].contains("Stepless"));

        graph.fail_task_steps.lock().unwrap().insert(task);
        let broken = build_handler(graph.clone(), None, "steps-broken", true).await;
        link_tasks(&graph, broken.session_uuid.unwrap(), &[task]).await;
        let r = text_turn(&broken).await;
        assert!(r.is_empty(), "no reminder built on unreadable steps: {r:?}");
    }

    /// A handler of session `sid` reopened: counters spent, reference read NOW (before
    /// its first turn), as the Claude Code engine's reopening site builds them.
    async fn reopened_handler(
        graph: Arc<MockGraphStore>,
        sid: &str,
        uuid: Option<Uuid>,
    ) -> PostStreamHandler {
        let mut h = build_handler(graph.clone(), None, sid, true).await;
        h.session_uuid = uuid;
        let dyn_graph: Arc<dyn GraphStore> = graph;
        let baseline = ObjectiveCounters::reopening_baseline(&dyn_graph, uuid).await;
        let mut sessions = h.active_sessions.write().await;
        let session = sessions.get_mut(sid).unwrap();
        session.objective_reminders_in_a_row.store(
            ObjectiveCounters::initial_in_a_row(true),
            std::sync::atomic::Ordering::Relaxed,
        );
        session.objective_baseline = baseline;
        drop(sessions);
        h
    }

    /// Reopened, then REAL progress during the very first turn (a linked task changes
    /// status): it is seen — the reference was read at the reopening, not at the end of
    /// that turn — and the reminders are rearmed.
    #[tokio::test]
    async fn progress_in_the_first_turn_after_a_reopening_rearms() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let done_soon = seed_task(&graph, TaskStatus::InProgress, Some("Done soon")).await;
        let next = seed_task(&graph, TaskStatus::Pending, Some("Next")).await;
        let uuid = Uuid::new_v4();
        link_tasks(&graph, uuid, &[done_soon, next]).await;
        let h = reopened_handler(graph.clone(), "reopened-progress", Some(uuid)).await;
        // The first turn completes a linked task.
        graph
            .tasks
            .write()
            .await
            .get_mut(&done_soon)
            .unwrap()
            .status = TaskStatus::Completed;
        h.handle_objective_tracking(true, false, false, false).await;
        assert!(reminders(&h).await.is_empty());
        assert_eq!(
            text_turn(&h).await.len(),
            1,
            "the first turn's progress rearmed it"
        );
    }

    /// Links, all of them (here 40), are the reference — not a window: the same links in
    /// another order, a link removed (a disappearance is never progress), and an active
    /// task behind 39 completed historical links that really changes status.
    #[tokio::test]
    async fn the_reference_covers_every_link_in_any_order() {
        use crate::neo4j::models::{LinkedTaskInfo, TaskStatus};
        let graph = Arc::new(MockGraphStore::new());
        let mut ids = vec![];
        for i in 0..39 {
            ids.push(seed_task(&graph, TaskStatus::Completed, Some(&format!("Old {i}"))).await);
        }
        let active = seed_task(&graph, TaskStatus::InProgress, Some("Active")).await;
        ids.push(active);
        let handler = build_handler(graph.clone(), None, "many-links", true).await;
        let uuid = handler.session_uuid.unwrap();
        let set_links = |order: Vec<Uuid>| {
            let graph = graph.clone();
            async move {
                graph.session_links.write().await.insert(
                    uuid,
                    crate::neo4j::models::LinkedSessionInfo {
                        linked_tasks: order
                            .into_iter()
                            .map(|id| LinkedTaskInfo {
                                id,
                                title: String::new(),
                                source: "auto".into(),
                            })
                            .collect(),
                        ..Default::default()
                    },
                );
            }
        };
        set_links(ids.clone()).await;
        spend_reminders(&handler).await;

        // Same links, reversed: no change, no rearm.
        let mut reversed = ids.clone();
        reversed.reverse();
        set_links(reversed).await;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        // A link removed (the smallest id: the first of any sorted window).
        let mut fewer = ids.clone();
        fewer.sort();
        let removed = fewer.remove(0);
        assert_ne!(removed, active);
        set_links(fewer.clone()).await;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        for _ in 0..3 {
            assert!(
                text_turn(&handler).await.is_empty(),
                "no rearm without progress"
            );
        }

        // The active task, behind the historical links, changes status: progress.
        graph.tasks.write().await.get_mut(&active).unwrap().status = TaskStatus::Blocked;
        handler
            .handle_objective_tracking(true, false, false, false)
            .await;
        graph.tasks.write().await.get_mut(&active).unwrap().status = TaskStatus::InProgress;
        // (back to in progress so the next stall has something to remind)
        let r = text_turn(&handler).await;
        assert_eq!(r.len(), 1, "the active task's progress was seen: {r:?}");
        assert!(
            r[0].contains("Active") && !r[0].contains("Old "),
            "{}",
            r[0]
        );
    }

    /// `(turns_since, in_a_row)` of the handler's session.
    async fn counters_of(handler: &PostStreamHandler) -> (u32, u32) {
        let sessions = handler.active_sessions.read().await;
        let s = sessions.get(&handler.session_id).unwrap();
        (
            s.objective_reminder_turns_since
                .load(std::sync::atomic::Ordering::Relaxed),
            s.objective_reminders_in_a_row
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// The whole sequence, counters at each turn: reminder → progress (a working turn)
    /// → text turn (reminded: the cooldown is over and the cap rearmed) → text turn
    /// (inside the cooldown of that reminder: silent).
    #[tokio::test]
    async fn reminder_progress_text_text_counters() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let done_soon = seed_task(&graph, TaskStatus::InProgress, Some("Done soon")).await;
        let next = seed_task(&graph, TaskStatus::Pending, Some("Next")).await;
        let h = build_handler(graph.clone(), None, "sequence", true).await;
        link_tasks(&graph, h.session_uuid.unwrap(), &[done_soon, next]).await;

        assert_eq!(text_turn(&h).await.len(), 1, "turn 1: reminded");
        assert_eq!(counters_of(&h).await, (1, 1));

        graph
            .tasks
            .write()
            .await
            .get_mut(&done_soon)
            .unwrap()
            .status = TaskStatus::Completed;
        h.handle_objective_tracking(true, false, false, false).await;
        assert!(
            reminders(&h).await.is_empty(),
            "turn 2: working, not reminded"
        );
        // Rearmed: in_a_row 0, turns_since set past the cooldown (2) then counted (3).
        assert_eq!(counters_of(&h).await, (OBJECTIVE_REMINDER_COOLDOWN + 1, 0));

        let r = text_turn(&h).await;
        assert_eq!(r.len(), 1, "turn 3: reminded again");
        assert!(
            r[0].contains("Next") && !r[0].contains("Done soon"),
            "{}",
            r[0]
        );
        assert_eq!(counters_of(&h).await, (1, 1));

        assert!(
            text_turn(&h).await.is_empty(),
            "turn 4: inside the cooldown"
        );
        assert_eq!(counters_of(&h).await, (2, 1));
    }

    /// Progress AND a conclusive tool (a commit) in the same turn, after a reminder: the
    /// progress rearms first (in_a_row 0, cooldown over), then the turn — wrapping up while
    /// a linked task is still to do — is reminded at once (in_a_row 1, turns_since 1).
    #[tokio::test]
    async fn progress_and_a_commit_in_one_turn_rearm_then_remind_the_wrap_up() {
        use crate::neo4j::models::TaskStatus;
        let graph = Arc::new(MockGraphStore::new());
        let done_soon = seed_task(&graph, TaskStatus::InProgress, Some("Done soon")).await;
        let next = seed_task(&graph, TaskStatus::Pending, Some("Next")).await;
        let h = build_handler(graph.clone(), None, "progress-commit", true).await;
        link_tasks(&graph, h.session_uuid.unwrap(), &[done_soon, next]).await;
        assert_eq!(text_turn(&h).await.len(), 1);
        assert_eq!(counters_of(&h).await, (1, 1));

        graph
            .tasks
            .write()
            .await
            .get_mut(&done_soon)
            .unwrap()
            .status = TaskStatus::Completed;
        // productive (Edit) + conclusive (git commit)
        h.handle_objective_tracking(true, true, false, false).await;
        let r = reminders(&h).await;
        assert_eq!(r.len(), 1, "the wrap-up is reminded of what remains");
        assert!(r[0].contains("Next"), "{}", r[0]);
        assert_eq!(counters_of(&h).await, (1, 1));
        assert!(text_turn(&h).await.is_empty(), "then the cooldown");
    }

    /// A reopened session (restart, resume) starts with its allowance spent
    /// (`ObjectiveCounters::initial_in_a_row(true)`, what the reopening sites use): three
    /// reopenings in a row without progress add no reminder to the first run's cap.
    #[tokio::test]
    async fn successive_reopenings_never_rearm_the_reminders() {
        let graph = Arc::new(MockGraphStore::new());
        let slug = seed_graph_with_pending_tasks(&graph).await;
        let first = linked_handler(graph.clone(), slug.clone(), "reopened").await;
        let uuid = first.session_uuid;
        let mut total = 0;
        for _ in 0..5 {
            total += text_turn(&first).await.len();
        }
        for _ in 0..3 {
            // The session reopened: new counters, as the reopening site builds them.
            let mut again =
                build_handler(graph.clone(), Some(slug.clone()), "reopened", true).await;
            again.session_uuid = uuid;
            again
                .active_sessions
                .read()
                .await
                .get("reopened")
                .unwrap()
                .objective_reminders_in_a_row
                .store(
                    ObjectiveCounters::initial_in_a_row(true),
                    std::sync::atomic::Ordering::Relaxed,
                );
            for _ in 0..5 {
                total += text_turn(&again).await.len();
            }
        }
        assert_eq!(
            total as u32, OBJECTIVE_REMINDER_MAX_IN_A_ROW,
            "the cap holds across reopenings"
        );
    }

    /// The cap bounds a CHAIN of continuations: the turn past it stays stopped, auto-continue
    /// stays on (no toggle can switch it back on), and the next chain gets the full cap again.
    #[test]
    fn the_cap_stops_a_chain_and_the_next_chain_starts_from_zero() {
        let on = AtomicBool::new(true);
        let chain = AtomicU32::new(0);
        let max = MAX_AUTO_CONTINUES;
        for _ in 0..max {
            assert!(auto_continue_allowed("s", true, &on, false, &chain, max));
        }
        assert!(!auto_continue_allowed("s", true, &on, false, &chain, max));
        assert!(on.load(Ordering::Relaxed), "the cap never switches it off");
        // The next turn (somebody asked for it) starts a new chain.
        for _ in 0..max {
            assert!(auto_continue_allowed("s", true, &on, false, &chain, max));
        }
        // A turn that ends within its limit ends the chain too.
        assert!(!auto_continue_allowed("s", false, &on, false, &chain, max));
        assert!(auto_continue_allowed("s", true, &on, false, &chain, max));
        // Off (the operator's kill switch) or stopped: never.
        assert!(!auto_continue_allowed("s", true, &on, true, &chain, max));
        assert!(!auto_continue_allowed(
            "s",
            true,
            &AtomicBool::new(false),
            false,
            &chain,
            max
        ));
    }

    /// Claude Code engine: a NEW interactive session, configured as the manager opens it
    /// under the default configuration (no toggle), continues a turn stopped on its limit,
    /// `MAX_AUTO_CONTINUES` times in a row, then the cap stops it.
    #[tokio::test(start_paused = true)]
    async fn a_new_legacy_session_continues_up_to_the_cap() {
        let graph = Arc::new(MockGraphStore::new());
        let mut handler = build_handler(graph, None, "legacy-new", false).await;
        let (on, max) =
            session_auto_continue(false, crate::chat::config::auto_continue_from_env(None));
        assert!(on, "on by default, without any toggle");
        assert_eq!(max, MAX_AUTO_CONTINUES);
        let flag = Arc::new(AtomicBool::new(on));
        handler.auto_continue = flag.clone();
        {
            let mut sessions = handler.active_sessions.write().await;
            let session = sessions.get_mut("legacy-new").unwrap();
            session.auto_continue = flag;
            session.max_auto_continues = max;
        }
        for i in 0..max {
            assert!(handler.handle_auto_continue(true).await, "continuation {i}");
        }
        assert!(
            !handler.handle_auto_continue(true).await,
            "the cap stops the chain"
        );
        assert_eq!(handler.pending_messages.lock().await.len(), max as usize);
        // A runner session continues even under the operator's kill switch; an
        // interactive one does not.
        assert_eq!(
            session_auto_continue(true, false),
            (true, MAX_AUTO_CONTINUES)
        );
        assert_eq!(
            session_auto_continue(false, false),
            (false, MAX_AUTO_CONTINUES)
        );
    }

    #[tokio::test]
    async fn test_auto_continue_enriched() {
        use crate::neo4j::models::{PlanNode, PlanStatus, StepNode, StepStatus, TaskStatus};

        let graph = Arc::new(MockGraphStore::new());

        // Seed project + plan + tasks + steps
        let project = test_project_named("enrich-project");
        let slug = project.slug.clone();
        graph.create_project(&project).await.unwrap();

        let mut plan = PlanNode::new_for_project(
            "Enriched Plan".to_string(),
            "Plan with steps".to_string(),
            "test".to_string(),
            50,
            project.id,
        );
        plan.status = PlanStatus::InProgress;
        graph.create_plan(&plan).await.unwrap();
        graph
            .update_plan_status(plan.id, PlanStatus::InProgress)
            .await
            .unwrap();

        let mut task1 = test_task_titled("Implement feature A");
        task1.status = TaskStatus::InProgress;
        graph.create_task(plan.id, &task1).await.unwrap();
        graph
            .update_task_status(task1.id, TaskStatus::InProgress)
            .await
            .unwrap();

        let step1 = StepNode::new(1, "Step one done".to_string(), Some("verify1".to_string()));
        graph.create_step(task1.id, &step1).await.unwrap();
        graph
            .update_step_status(step1.id, StepStatus::Completed)
            .await
            .unwrap();
        let step2 = StepNode::new(
            2,
            "Step two pending".to_string(),
            Some("verify2".to_string()),
        );
        graph.create_step(task1.id, &step2).await.unwrap();

        let mut task2 = test_task_titled("Write unit tests");
        task2.status = TaskStatus::Pending;
        graph.create_task(plan.id, &task2).await.unwrap();

        // Build handler with auto_continue enabled
        let mut handler = build_handler(graph, Some(slug), "enrich-session", false).await;
        handler.auto_continue = Arc::new(AtomicBool::new(true));

        // Configure session for auto_continue
        {
            let mut sessions = handler.active_sessions.write().await;
            let session = sessions.get_mut("enrich-session").unwrap();
            session.auto_continue = Arc::new(AtomicBool::new(true));
            session.max_auto_continues = 5;
        }

        // Populate work_log with 3 modified files
        {
            let mut log = handler.work_log.lock().await;
            log.record_tool_use(
                "Write",
                &serde_json::json!({"file_path": "/src/main.rs", "content": "fn main() {}"}),
            );
            log.record_tool_use(
                "Edit",
                &serde_json::json!({"file_path": "/src/lib.rs", "old_string": "a", "new_string": "b"}),
            );
            log.record_tool_use(
                "Write",
                &serde_json::json!({"file_path": "/src/utils.rs", "content": "pub fn util() {}"}),
            );
            // Mark a step completed
            log.steps_completed.push(Uuid::new_v4());
        }

        // Trigger auto-continue (hit_error_max_turns = true)
        let allowed = handler.handle_auto_continue(true).await;
        assert!(allowed, "Auto-continue should be allowed");

        // Check the enqueued message
        let messages = handler.pending_messages.lock().await;
        assert_eq!(messages.len(), 1, "Should have one pending message");
        let msg = &messages[0];
        assert_eq!(
            msg.kind,
            PendingMessageKind::SystemHint,
            "Should be SystemHint"
        );
        let content = &msg.content;

        // Must contain the "do not restart" instruction
        assert!(
            content.contains("Continue where you left off"),
            "Missing continue instruction in: {}",
            &content[..content.len().min(200)]
        );
        assert!(
            content.contains("Do NOT restart work already done"),
            "Missing do-not-restart instruction"
        );

        // Must contain CompactionContext sections (task/step info)
        assert!(
            content.contains("Task Context")
                || content.contains("Active Plans")
                || content.contains("Pending Objectives"),
            "Missing CompactionContext sections in auto-continue message"
        );

        // Must contain SessionWorkLog data (files modified)
        assert!(
            content.contains("Session Work Log") || content.contains("Files modified"),
            "Missing SessionWorkLog section"
        );
        assert!(content.contains("/src/main.rs"), "Missing file in work log");

        // Must be >500 chars (enriched, not just "Continue")
        assert!(
            content.len() > 500,
            "Auto-continue message too short ({} chars), expected >500",
            content.len()
        );
    }

    // ── Bounded segment checkpoint (overflow-recovery fallback) ──────────

    /// A conversation's segmentation for the checkpoint tests: two turns on one
    /// topic, then a drift confirmed on the fourth turn (two segments, one boundary).
    fn checkpoint_segments() -> (Vec<Segment>, Vec<Boundary>) {
        use crate::chat::provider::cognitive::segment::{
            Dimension, Fingerprint, Observation, SegmentPolicy, Segmenter,
        };
        fn obs(turn: u32, labels: &[&str]) -> Observation {
            Observation {
                turn,
                prints: std::iter::once((
                    Dimension::Topic,
                    Fingerprint::new(labels.iter().copied()),
                ))
                .collect(),
            }
        }
        let history = vec![
            obs(0, &["relay"]),
            obs(1, &["relay"]),
            obs(2, &["docker"]),
            obs(3, &["docker"]),
        ];
        let segmenter = Segmenter::replay(SegmentPolicy::default(), 7, &history).unwrap();
        (segmenter.segments(), segmenter.boundaries().to_vec())
    }

    #[test]
    fn a_checkpoint_is_deterministic() {
        let (segments, boundaries) = checkpoint_segments();
        let first = bounded_segment_checkpoint(&segments, &boundaries, Some("Fix the relay"));
        let second = bounded_segment_checkpoint(&segments, &boundaries, Some("Fix the relay"));
        assert_eq!(
            first, second,
            "the checkpoint must not depend on arrival order"
        );
        assert!(
            first.chars().count() <= SEGMENT_CHECKPOINT_MAX_CHARS,
            "the checkpoint must stay bounded: {}",
            first.chars().count()
        );
    }

    #[test]
    fn a_checkpoint_keeps_the_latest_turn_even_without_segments() {
        let checkpoint = bounded_segment_checkpoint(&[], &[], Some("The latest request"));
        assert!(
            checkpoint.contains("The latest request"),
            "the user's latest turn must always be kept: {checkpoint}"
        );
        assert!(
            checkpoint.contains("# Conversation Checkpoint"),
            "a checkpoint with only a latest turn is still a checkpoint: {checkpoint}"
        );
    }

    #[tokio::test]
    async fn a_session_with_segments_builds_a_bounded_checkpoint_when_the_window_is_tight() {
        const SID: &str = "00000000-0000-0000-0000-000000000001";
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let (segments, boundaries) = checkpoint_segments();
        graph.put_segments(SID, &segments).await.unwrap();
        graph.put_boundaries(SID, &boundaries).await.unwrap();
        // The user's latest turn, persisted like any event.
        let uuid = Uuid::parse_str(SID).unwrap();
        graph
            .store_chat_events(
                uuid,
                vec![ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: 1,
                    event_type: "user_message".to_string(),
                    data: serde_json::to_string(&ChatEvent::UserMessage {
                        content: "Fix the relay timeout please".to_string(),
                    })
                    .unwrap(),
                    created_at: chrono::Utc::now(),
                }],
            )
            .await
            .unwrap();

        // A tight window: the normal recovery (~75 chars) already overflows its 5 % share.
        let (hint, _event) = post_compaction_recovery(&graph, SID, None, None, Some(100)).await;
        let hint = hint.expect("a checkpoint should be injected");

        assert!(
            hint.contains("# Conversation Checkpoint"),
            "the bounded checkpoint should replace the normal recovery: {hint}"
        );
        assert!(hint.contains("relay"), "missing segment label: {hint}");
        assert!(hint.contains("docker"), "missing segment label: {hint}");
        assert!(
            hint.contains("Fix the relay timeout please"),
            "the latest turn was lost: {hint}"
        );
        assert!(
            hint.chars().count() <= SEGMENT_CHECKPOINT_MAX_CHARS,
            "the checkpoint must stay bounded: {}",
            hint.chars().count()
        );
        // The latest turn is listed first, so a later cap still keeps it.
        assert!(
            hint.find("Latest request").unwrap() < hint.find("Recent direction").unwrap(),
            "the latest request must precede the segment labels: {hint}"
        );
    }

    /// RESUME after a compaction under a tight window: the model gets the checkpoint and
    /// nothing else of the conversation. With three user turns persisted, only the LAST one
    /// is in it (`last_user_turn_text` scans back to the first `user_message` and stops;
    /// `bounded_segment_checkpoint` takes segment labels and boundaries, never messages).
    /// This asserts what a resumed model should still see — the three asks, in order — and
    /// is kept as a recorded gap (the suite stays green, the fault stays written).
    #[tokio::test]
    #[ignore = "trouvaille: le checkpoint post-compaction ne contient que le DERNIER message utilisateur (post_stream.rs:1024 `last_user_turn_text` s'arrête au premier user_message trouvé) et les étiquettes de segments (post_stream.rs:945 `bounded_segment_checkpoint`), jamais les messages précédents; avec une fenêtre étroite, les demandes 1 et 2 sont perdues"]
    async fn a_checkpoint_after_three_user_turns_keeps_the_three_asks_in_order() {
        const SID: &str = "00000000-0000-0000-0000-000000000003";
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let (segments, boundaries) = checkpoint_segments();
        graph.put_segments(SID, &segments).await.unwrap();
        graph.put_boundaries(SID, &boundaries).await.unwrap();
        let uuid = Uuid::parse_str(SID).unwrap();
        let record = |seq: i64, text: &str| ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq,
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: text.to_string(),
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        };
        graph
            .store_chat_events(
                uuid,
                vec![
                    record(1, "ASK-ONE migrate the schema"),
                    record(2, "ASK-TWO then add the index"),
                    record(3, "ASK-THREE finally benchmark it"),
                ],
            )
            .await
            .unwrap();
        let (hint, _event) = post_compaction_recovery(&graph, SID, None, None, Some(100)).await;
        let hint = hint.expect("a checkpoint should be injected");
        assert!(hint.contains("# Conversation Checkpoint"), "{hint}");
        let at = |needle: &str| hint.find(needle);
        let (one, two, three) = (at("ASK-ONE"), at("ASK-TWO"), at("ASK-THREE"));
        assert!(
            matches!((one, two, three), (Some(a), Some(b), Some(c)) if a < b && b < c),
            "the checkpoint must carry the three asks in order: {hint}"
        );
    }

    #[tokio::test]
    async fn a_session_without_segments_keeps_the_existing_recovery_when_the_window_is_tight() {
        const SID: &str = "00000000-0000-0000-0000-000000000002";
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        // No segments, no boundaries, no user message: there is no graph material
        // to checkpoint, so the normal recovery is kept.
        let (hint, _event) = post_compaction_recovery(&graph, SID, None, None, Some(100)).await;
        let hint = hint.expect("a hint should be injected");
        assert!(
            hint.contains("# Post-Compaction Context"),
            "the existing behavior should be kept: {hint}"
        );
        assert!(
            !hint.contains("# Conversation Checkpoint"),
            "no checkpoint without segments: {hint}"
        );
    }
}

#[cfg(test)]
mod bounded_step_tests {
    use super::bounded;
    #[allow(unused_imports)]
    use std::time::Duration;

    #[tokio::test]
    async fn a_step_that_finishes_in_time_returns_its_value() {
        let out = bounded("s", "quick", Duration::from_millis(200), async { 7 }).await;
        assert_eq!(out, Some(7));
    }

    #[tokio::test]
    async fn a_step_that_never_answers_is_dropped_past_the_budget() {
        let started = std::time::Instant::now();
        let out = bounded(
            "s",
            "stuck",
            Duration::from_millis(30),
            std::future::pending::<()>(),
        )
        .await;
        assert_eq!(out, None);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the budget bounds the wait, the future does not"
        );
    }

    #[tokio::test]
    async fn a_dropped_step_releases_what_it_held() {
        // What the step holds (here a lock) is released when it is dropped:
        // the next step, and the next turn, are not left waiting on it.
        let lock = std::sync::Arc::new(tokio::sync::Mutex::new(()));
        let held = lock.clone();
        let out = bounded("s", "holding", Duration::from_millis(30), async move {
            let _guard = held.lock().await;
            std::future::pending::<()>().await;
        })
        .await;
        assert_eq!(out, None);
        assert!(lock.try_lock().is_ok(), "the lock is free again");
    }

    fn budget(
        ms: u64,
    ) -> (
        super::StepBudget,
        tokio::sync::broadcast::Receiver<super::ChatEvent>,
    ) {
        let (events_tx, rx) = tokio::sync::broadcast::channel(16);
        (
            super::StepBudget {
                session_id: "s".into(),
                budget: Duration::from_millis(ms),
                events_tx,
                nats: None,
            },
            rx,
        )
    }

    fn error_codes(
        rx: &mut tokio::sync::broadcast::Receiver<super::ChatEvent>,
    ) -> Vec<(String, String)> {
        let mut codes = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let super::ChatEvent::Error {
                code: Some(code),
                reason: Some(reason),
                ..
            } = event
            {
                codes.push((code, reason));
            }
        }
        codes
    }

    #[tokio::test]
    async fn a_step_past_its_budget_is_reported_to_the_clients() {
        let (steps, mut rx) = budget(30);
        assert_eq!(steps.run("quick", async { 1 }).await, Some(1));
        assert!(error_codes(&mut rx).is_empty(), "nothing to report");
        assert_eq!(steps.run("stuck", std::future::pending::<()>()).await, None);
        assert_eq!(
            error_codes(&mut rx),
            vec![(super::STEP_ABANDONED_CODE.to_string(), "stuck".to_string())]
        );
    }

    #[tokio::test]
    async fn a_write_past_its_budget_is_reported_and_still_lands() {
        let (steps, mut rx) = budget(30);
        let landed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let gate = std::sync::Arc::new(tokio::sync::Notify::new());
        let write = super::spawn_write("s".into(), "slow_write", {
            let landed = landed.clone();
            let gate = gate.clone();
            async move {
                gate.notified().await;
                landed.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            }
        });
        steps.wait_for_write("slow_write", write).await;
        assert_eq!(
            error_codes(&mut rx),
            vec![(
                super::PERSISTENCE_DELAYED_CODE.to_string(),
                "slow_write".to_string()
            )],
            "the turn went on and said so"
        );
        // The wait is over, the write is not: it lands when the store answers.
        gate.notify_one();
        for _ in 0..100 {
            if landed.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the write was dropped with the wait");
    }
}

/// Le règlement des récupérations à la fin du tour (registre de récupération).
#[cfg(test)]
mod retrieval_settlement_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use crate::notes::retrieval_ledger::{self, RetrievedItem, Source, Trigger};

    #[test]
    fn cited_note_ids_lit_les_notes_et_les_rfc_pas_les_autres_entites() {
        let note = Uuid::new_v4();
        let rfc = Uuid::new_v4();
        let plan = Uuid::new_v4();
        let text = format!("voir #note:{note} et #rfc:{rfc}, pas #plan:{plan}");
        assert_eq!(
            cited_note_ids(&text),
            vec![note.to_string(), rfc.to_string()]
        );
        assert!(cited_note_ids("rien de cité").is_empty());
    }

    /// Revue 31/10 (6) : le plafond de citations par message (toutes sortes)
    /// ne doit pas faire perdre une note citée après vingt plans.
    #[test]
    fn une_note_citee_apres_vingt_plans_est_lue() {
        let note = Uuid::new_v4();
        let mut text = (0..crate::refs::cite::MAX_CITED_PER_MESSAGE + 1)
            .map(|_| format!("#plan:{}", Uuid::new_v4()))
            .collect::<Vec<_>>()
            .join(" ");
        text.push_str(&format!(" et #note:{note}"));
        assert_eq!(cited_note_ids(&text), vec![note.to_string()]);
    }

    /// La fin de tour règle les récupérations de la session avec ce que la
    /// réponse a cité — sans mémoire de conversation (retour anticipé APRÈS le
    /// règlement), et seulement l'intersection avec ce que l'index a rendu.
    // Le garde sérialise un registre GLOBAL à travers l'appel observé.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn la_fin_de_tour_regle_les_recuperations_de_la_session() {
        let _serial = retrieval_ledger::serial_guard();
        let saved_capacity = retrieval_ledger::journal_capacity();
        retrieval_ledger::set_journal_capacity(4096);

        let session = Uuid::new_v4();
        let cited = Uuid::new_v4();
        let id = retrieval_ledger::record_for_session(
            Source::Bm25,
            Trigger::Injection,
            "q",
            vec![
                RetrievedItem::new(cited.to_string(), "note", Some(0.5)),
                RetrievedItem::new("not-cited", "note", Some(0.1)),
            ],
            true,
            1,
            Some(session),
        )
        .expect("le journal accepte l'événement");

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let search: Arc<dyn SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        record_turn_feedback(
            TurnFeedback {
                graph: &graph,
                search: &search,
                event_emitter: &None,
                session_id: &session.to_string(),
                session_uuid: Some(session),
                project_id: None,
                rfc_accumulator: None,
            },
            &format!(
                "La réponse s'appuie sur #note:{cited} et sur #note:{}.",
                Uuid::new_v4()
            ),
            &None,
            &None,
        )
        .await;

        let event = retrieval_ledger::recent_for_session(session, 10)
            .pop()
            .expect("l'événement de la session");
        retrieval_ledger::set_journal_capacity(saved_capacity);
        assert_eq!(event.id, id);
        let usage = event.usage.expect("réglé à la fin du tour");
        assert_eq!(usage.used_ids, vec![cited.to_string()]);
        assert!(usage.validation.is_none() && usage.benefit.is_none());
    }

    /// Une fin de tour appelée pour une session SANS récupération en attente :
    /// rien n'est réglé, et la fin de tour continue (ici sans mémoire, retour
    /// anticipé). Avec le journal en debug, comme en production verbeuse.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn une_fin_de_tour_sans_recuperation_ne_regle_rien() {
        let _serial = retrieval_ledger::serial_guard();
        let saved_capacity = retrieval_ledger::journal_capacity();
        retrieval_ledger::set_journal_capacity(4096);
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(std::io::sink)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let session = Uuid::new_v4();
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let search: Arc<dyn SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        let feedback = || TurnFeedback {
            graph: &graph,
            search: &search,
            event_emitter: &None,
            session_id: "s",
            session_uuid: Some(session),
            project_id: None,
            rfc_accumulator: None,
        };
        record_turn_feedback(feedback(), "Aucune citation.", &None, &None).await;
        assert!(
            retrieval_ledger::recent_for_session(session, 10).is_empty(),
            "aucun événement inventé"
        );

        // Et avec un événement en attente, sous le même abonné debug.
        retrieval_ledger::record_for_session(
            Source::Bm25,
            Trigger::Injection,
            "q",
            vec![RetrievedItem::new("not-cited", "note", Some(0.1))],
            true,
            1,
            Some(session),
        )
        .expect("le journal accepte l'événement");
        record_turn_feedback(feedback(), "Aucune citation.", &None, &None).await;
        let event = retrieval_ledger::recent_for_session(session, 10)
            .pop()
            .expect("l'événement de la session");
        retrieval_ledger::set_journal_capacity(saved_capacity);
        assert!(event.usage.is_some(), "réglé à la fin du tour");
    }
}
