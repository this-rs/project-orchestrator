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
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
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
                let count = session.auto_continue_count.fetch_add(1, Ordering::Relaxed) + 1;
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
                        let builder = super::compaction_context::CompactionContextBuilder::new(
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

    /// If the agent concluded without using tools and auto-continue didn't fire,
    /// check for pending objectives and inject a SystemHint reminder.
    pub async fn handle_objective_tracking(
        &self,
        had_tool_use: bool,
        auto_continue_allowed: bool,
        hit_error_max_turns: bool,
    ) {
        // Gather session-level state for the pure decision function
        let (tracking_enabled, cooldown_turns) = {
            let sessions = self.active_sessions.read().await;
            if let Some(session) = sessions.get(&self.session_id) {
                if had_tool_use {
                    // Agent used tools = still working, reset cooldown
                    session
                        .objective_reminder_turns_since
                        .store(0, Ordering::Relaxed);
                }
                let enabled = session.objective_tracking;
                let turns = session
                    .objective_reminder_turns_since
                    .fetch_add(1, Ordering::Relaxed);
                (enabled, turns)
            } else {
                return;
            }
        };

        // Fetch pending objectives from graph (only if we might need them)
        let objectives = if !had_tool_use
            && !auto_continue_allowed
            && !hit_error_max_turns
            && !self.interrupt_flag.load(Ordering::SeqCst)
            && tracking_enabled
            && (cooldown_turns == 0 || cooldown_turns >= OBJECTIVE_REMINDER_COOLDOWN)
        {
            if let Some(ref slug) = self.ctx.project_slug {
                let builder =
                    super::compaction_context::CompactionContextBuilder::new(self.graph.clone());
                if let Ok(Ok(ctx)) = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    builder.build_for_session(Some(slug.as_str())),
                )
                .await
                {
                    ctx.pending_objectives_oneliner()
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Pure decision
        let input = ObjectiveCheckInput {
            had_tool_use,
            auto_continue_allowed,
            hit_error_max_turns,
            interrupted: self.interrupt_flag.load(Ordering::SeqCst),
            tracking_enabled,
            cooldown_turns,
            objectives,
        };

        if let Some(reminder) = check_objective_reminder(&input) {
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

// ── Pure logic for objective tracking (testable without Graph) ─────────

/// Cooldown threshold for objective reminders (same as PostStreamHandler).
pub(crate) const OBJECTIVE_REMINDER_COOLDOWN: u32 = 3;

/// Inputs to the objective tracking decision (decoupled from async/Arc state).
pub(crate) struct ObjectiveCheckInput {
    pub had_tool_use: bool,
    pub auto_continue_allowed: bool,
    pub hit_error_max_turns: bool,
    pub interrupted: bool,
    pub tracking_enabled: bool,
    pub cooldown_turns: u32,
    pub objectives: String,
}

/// Pure decision function: given the objective check inputs, return the
/// reminder text to inject as SystemHint, or None if no reminder should fire.
///
/// This function has NO side effects and is fully testable without mocks.
pub(crate) fn check_objective_reminder(input: &ObjectiveCheckInput) -> Option<String> {
    // Guard 1: only fire when agent concluded without tools
    if input.had_tool_use
        || input.auto_continue_allowed
        || input.hit_error_max_turns
        || input.interrupted
    {
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

    // Guard 4: must have pending objectives
    if input.objectives.is_empty() {
        return None;
    }

    Some(format!(
        "You stopped without using any tools. \
         Check if you have completed all objectives before concluding.{}",
        input.objectives
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> ObjectiveCheckInput {
        ObjectiveCheckInput {
            had_tool_use: false,
            auto_continue_allowed: false,
            hit_error_max_turns: false,
            interrupted: false,
            tracking_enabled: true,
            cooldown_turns: 0, // first turn → should fire
            objectives: " Remaining objectives: T1, T2".to_string(),
        }
    }

    #[test]
    fn test_reminder_injected_when_no_tools_and_pending_objectives() {
        let input = base_input();
        let result = check_objective_reminder(&input);
        assert!(result.is_some());
        let msg = result.unwrap();
        assert!(msg.contains("You stopped without using any tools"));
        assert!(msg.contains("T1, T2"));
    }

    #[test]
    fn test_no_reminder_when_tools_were_used() {
        let mut input = base_input();
        input.had_tool_use = true;
        assert!(check_objective_reminder(&input).is_none());
    }

    #[test]
    fn test_cooldown_respected() {
        // turns=1 is within cooldown (COOLDOWN=3), should NOT fire
        let mut input = base_input();
        input.cooldown_turns = 1;
        assert!(check_objective_reminder(&input).is_none());

        // turns=2 is still within cooldown
        input.cooldown_turns = 2;
        assert!(check_objective_reminder(&input).is_none());

        // turns=3 meets cooldown threshold → should fire
        input.cooldown_turns = 3;
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
        input.objectives = String::new();
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
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::chat::types::PendingMessageKind;
    use crate::meilisearch::mock::MockSearchStore;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::GraphStore;
    use crate::test_helpers::{test_project_named, test_task_titled};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32};
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::sync::{broadcast, Mutex, RwLock};
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

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
        // We use nexus_claude::InteractiveClient::new with auto_download_cli — the client
        // won't be used by handle_objective_tracking, but we need a valid instance
        // to satisfy the ActiveSession struct. auto_download_cli ensures the CLI
        // is available even in CI environments (downloaded and cached on first run).
        {
            let options = nexus_claude::ClaudeCodeOptions::builder()
                .auto_download_cli(true)
                .build();
            let client =
                nexus_claude::InteractiveClient::new(options).expect("InteractiveClient with auto_download_cli");

            let session = ActiveSession {
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
                pending_permission_inputs: Arc::new(Mutex::new(HashMap::new())),
                auto_continue: Arc::new(AtomicBool::new(false)),
                auto_continue_count: Arc::new(AtomicU32::new(0)),
                max_auto_continues: 0,
                rfc_accumulator: Arc::new(Mutex::new(
                    super::super::observation_detector::RfcAccumulator::new(),
                )),
                objective_tracking,
                objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
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
        }
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

        let handler = build_handler(
            graph,
            Some(slug),
            "test-session-1",
            true, // objective_tracking enabled
        )
        .await;

        // Call with had_tool_use=false → should inject reminder
        handler.handle_objective_tracking(false, false, false).await;

        let pending = handler.pending_messages.lock().await;
        assert_eq!(pending.len(), 1, "Expected 1 pending SystemHint message");
        let msg = &pending[0];
        assert_eq!(msg.kind, PendingMessageKind::SystemHint);
        assert!(
            msg.content.contains("You stopped without using any tools"),
            "Reminder text missing: {}",
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

        let handler = build_handler(graph, Some(slug), "test-session-2", true).await;

        // Call with had_tool_use=true → no reminder
        handler.handle_objective_tracking(true, false, false).await;

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
            graph,
            Some(slug),
            "test-session-3",
            false, // tracking disabled
        )
        .await;

        handler.handle_objective_tracking(false, false, false).await;

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

        handler.handle_objective_tracking(false, false, false).await;

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

        let handler = build_handler(graph, Some(slug), "test-session-5", true).await;

        // First call (turn 0) → should fire
        handler.handle_objective_tracking(false, false, false).await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            1,
            "First call should inject reminder"
        );

        // Clear pending for next check
        handler.pending_messages.lock().await.clear();

        // Second call (turn 1, within cooldown) → should NOT fire
        handler.handle_objective_tracking(false, false, false).await;
        assert!(
            handler.pending_messages.lock().await.is_empty(),
            "Second call should be blocked by cooldown"
        );

        // Third call (turn 2, still within cooldown) → should NOT fire
        handler.handle_objective_tracking(false, false, false).await;
        assert!(
            handler.pending_messages.lock().await.is_empty(),
            "Third call should still be blocked by cooldown"
        );

        // Fourth call (turn 3, cooldown met) → should fire
        handler.handle_objective_tracking(false, false, false).await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            1,
            "Fourth call should inject reminder (cooldown met)"
        );
    }
}
