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

use super::manager::ActiveSession;
use super::types::{ChatEvent, PendingMessage};
use crate::neo4j::models::ChatEventRecord;
use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager},
    InteractiveClient,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

use std::collections::VecDeque;

/// Resolved project context from a single Neo4j lookup, shared across
/// all post-stream consumers (compaction, auto-continue, objective tracker, feedback).
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
            info!(
                "Post-compaction injection triggered for session {}",
                self.session_id
            );

            let build_start = std::time::Instant::now();
            let builder =
                super::compaction_context::CompactionContextBuilder::new(self.graph.clone());
            let build_result = builder
                .build_for_session(self.ctx.project_slug.as_deref())
                .await;
            let build_latency_ms = build_start.elapsed().as_millis() as u64;

            let (hint_len, recovery_success) = match build_result {
                Ok(ctx) => {
                    let hint = ctx.to_markdown();
                    let len = hint.len();
                    if !hint.is_empty() {
                        info!(
                            "Injecting post-compaction context for session {} ({} chars)",
                            self.session_id, len
                        );
                        self.pending_messages
                            .lock()
                            .await
                            .push_back(PendingMessage::system_hint(hint));
                    }
                    (len, true)
                }
                Err(e) => {
                    warn!(
                        "Failed to build post-compaction context for session {}: {}. Injecting minimal reminder.",
                        self.session_id, e
                    );
                    let minimal = if self.ctx.project_slug.is_some() {
                        format!(
                            "<system-reminder>\n# Post-Compaction Context\nYou are working on project \"{}\". Context rebuild failed — continue based on conversation history.\n</system-reminder>",
                            self.ctx.project_slug.as_deref().unwrap_or("unknown")
                        )
                    } else {
                        "<system-reminder>\n# Post-Compaction Context\nYou are in an interactive session. No project context available.\n</system-reminder>".to_string()
                    };
                    let len = minimal.len();
                    self.pending_messages
                        .lock()
                        .await
                        .push_back(PendingMessage::system_hint(minimal));
                    (len, false)
                }
            };

            let hint_tokens = (hint_len as u32) / 3;
            let recovery_event = ChatEvent::CompactionRecovery {
                hint_tokens,
                build_latency_ms,
                recovery_success,
            };
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
            let mut cancel_events = Vec::new();
            for (tool_id, parent) in pending_tool_calls {
                let event = ChatEvent::ToolCancelled {
                    id: tool_id.clone(),
                    parent_tool_use_id: parent.clone(),
                };
                cancel_events.push(event.clone());
                self.emit_chat(event);
            }

            // Persist ToolCancelled events in Neo4j
            if let Some(uuid) = self.session_uuid {
                let mut cancel_records = Vec::new();
                for event in &cancel_events {
                    let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
                    cancel_records.push(ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq,
                        event_type: event.event_type().to_string(),
                        data: serde_json::to_string(event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    });
                }
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
        let auto_continue_allowed = if hit_error_max_turns
            && self.auto_continue.load(Ordering::Relaxed)
            && !self.interrupt_flag.load(Ordering::SeqCst)
        {
            let sessions = self.active_sessions.read().await;
            if let Some(session) = sessions.get(&self.session_id) {
                let count = session
                    .auto_continue_count
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;
                let max = session.max_auto_continues;
                if max > 0 && count > max {
                    warn!(
                        "Auto-continue limit reached for session {} ({}/{}), disabling",
                        self.session_id, count, max
                    );
                    self.auto_continue.store(false, Ordering::Relaxed);
                    false
                } else {
                    if max > 0 {
                        info!(
                            "Auto-continue {}/{} for session {}",
                            count, max, self.session_id
                        );
                    }
                    true
                }
            } else {
                false
            }
        } else {
            false
        };

        if auto_continue_allowed {
            let delay_ms = 500u64;
            info!(
                "Auto-continue triggered for session {} (delay={}ms)",
                self.session_id, delay_ms
            );

            // Emit AutoContinue event
            let ac_event = ChatEvent::AutoContinue {
                session_id: self.session_id.clone(),
                delay_ms,
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
                _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => false,
                _ = self.interrupt_token.cancelled() => true,
            };

            if !sleep_cancelled && !self.interrupt_flag.load(Ordering::SeqCst) {
                let continue_msg = 'build_msg: {
                    if let Some(ref slug) = self.ctx.project_slug {
                        let builder =
                            super::compaction_context::CompactionContextBuilder::new(
                                self.graph.clone(),
                            );
                        if let Ok(Ok(ctx)) = tokio::time::timeout(
                            std::time::Duration::from_secs(2),
                            builder.build_for_session(Some(slug.as_str())),
                        )
                        .await
                        {
                            let objectives = ctx.pending_objectives_oneliner();
                            if !objectives.is_empty() {
                                break 'build_msg format!("Continue.{objectives}");
                            }
                        }
                    }
                    "Continue".to_string()
                };

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

    /// Cooldown: require N stream turns between reminders.
    const OBJECTIVE_REMINDER_COOLDOWN: u32 = 3;

    /// If the agent concluded without using tools and auto-continue didn't fire,
    /// check for pending objectives and inject a SystemHint reminder.
    pub async fn handle_objective_tracking(
        &self,
        had_tool_use: bool,
        auto_continue_allowed: bool,
        hit_error_max_turns: bool,
    ) {
        if !had_tool_use
            && !auto_continue_allowed
            && !hit_error_max_turns
            && !self.interrupt_flag.load(Ordering::SeqCst)
        {
            let should_check = {
                let sessions = self.active_sessions.read().await;
                if let Some(session) = sessions.get(&self.session_id) {
                    if session.objective_tracking {
                        let turns = session
                            .objective_reminder_turns_since
                            .fetch_add(1, Ordering::Relaxed);
                        turns >= Self::OBJECTIVE_REMINDER_COOLDOWN || turns == 0
                    } else {
                        false
                    }
                } else {
                    false
                }
            };

            if should_check {
                if let Some(ref slug) = self.ctx.project_slug {
                    let builder =
                        super::compaction_context::CompactionContextBuilder::new(
                            self.graph.clone(),
                        );
                    if let Ok(Ok(ctx)) = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        builder.build_for_session(Some(slug.as_str())),
                    )
                    .await
                    {
                        let objectives = ctx.pending_objectives_oneliner();
                        if !objectives.is_empty() {
                            let reminder = format!(
                                "You stopped without using any tools. \
                                 Check if you have completed all objectives before concluding.{}",
                                objectives
                            );
                            info!(
                                "Objective tracker: injecting reminder for session {}",
                                self.session_id
                            );
                            self.pending_messages
                                .lock()
                                .await
                                .push_back(PendingMessage::system_hint(reminder));

                            // Reset cooldown counter
                            let sessions = self.active_sessions.read().await;
                            if let Some(session) = sessions.get(&self.session_id) {
                                session
                                    .objective_reminder_turns_since
                                    .store(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        } else if had_tool_use {
            // Agent used tools = still working, reset cooldown
            let sessions = self.active_sessions.read().await;
            if let Some(session) = sessions.get(&self.session_id) {
                session
                    .objective_reminder_turns_since
                    .store(0, Ordering::Relaxed);
            }
        }
    }

    // ── Streaming status ──────────────────────────────────────────────────

    /// Update streaming status and clear buffers if no pending messages.
    /// Returns whether there are pending messages (caller needs this for drain).
    pub async fn finalize_streaming_status(&self) -> bool {
        let has_pending = !self.pending_messages.lock().await.is_empty();

        if !has_pending {
            self.is_streaming.store(false, Ordering::SeqCst);
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

    // ── Event persistence ─────────────────────────────────────────────────

    /// Batch-persist collected events to Neo4j.
    pub async fn persist_events(&self, events_to_persist: Vec<ChatEventRecord>) {
        if let Some(uuid) = self.session_uuid {
            if !events_to_persist.is_empty() {
                if let Err(e) = self.graph.store_chat_events(uuid, events_to_persist).await {
                    warn!(
                        "Failed to persist chat events for session {}: {}",
                        self.session_id, e
                    );
                }
            }
        }
    }

    // ── Memory / feedback / RFC ───────────────────────────────────────────

    /// Record assistant response in memory, spawn feedback extraction and RFC detection.
    pub async fn handle_feedback(
        &self,
        assistant_text_parts: &[String],
        memory_manager: &Option<Arc<Mutex<ConversationMemoryManager>>>,
        context_injector: &Option<Arc<ContextInjector>>,
    ) {
        if let Some(ref mm) = memory_manager {
            let assistant_text = assistant_text_parts.join("");
            if !assistant_text.is_empty() {
                let mut mm = mm.lock().await;
                mm.record_assistant_message(&assistant_text);

                if let Some(uuid) = self.session_uuid {
                    let project_id = self.ctx.project_id;
                    super::feedback::spawn_feedback(
                        self.graph.clone(),
                        uuid,
                        project_id,
                        assistant_text.clone(),
                        super::feedback::SessionDiscussedCache::new(),
                    );

                    // RFC auto-detection
                    let rfc_acc = {
                        let sessions = self.active_sessions.read().await;
                        sessions
                            .get(&self.session_id)
                            .map(|s| s.rfc_accumulator.clone())
                    };
                    if let Some(rfc_acc) = rfc_acc {
                        super::feedback::spawn_rfc_processing(
                            self.graph.clone(),
                            self.search.clone(),
                            project_id,
                            assistant_text.clone(),
                            rfc_acc,
                            Some(uuid),
                            self.event_emitter.clone(),
                        );
                    }
                }

                // Store pending messages via ContextInjector
                if let Some(ref injector) = context_injector {
                    let pending = mm.take_pending_messages();
                    if !pending.is_empty() {
                        if let Err(e) = injector.store_messages(&pending).await {
                            warn!(
                                "Failed to store messages for session {}: {}",
                                self.session_id, e
                            );
                        } else {
                            debug!(
                                "Stored {} messages for session {}",
                                pending.len(),
                                self.session_id
                            );
                        }
                    }
                }
            }
        }
    }
}
