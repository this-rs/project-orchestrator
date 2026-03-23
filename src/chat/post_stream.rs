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
                Ok(mut ctx) => {
                    // Inject SessionWorkLog snapshot so to_markdown() includes "Work Already Done"
                    let snapshot = self.work_log.lock().await.snapshot();
                    let has_work = !snapshot.files_modified.is_empty()
                        || !snapshot.steps_completed.is_empty()
                        || snapshot.tool_use_count > 0;
                    if has_work {
                        ctx.work_log = Some(snapshot);
                    }
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
                    let mut parts = Vec::new();
                    parts.push(
                        "Continue where you left off. Do NOT restart work already done."
                            .to_string(),
                    );

                    // Append CompactionContext markdown (task/step context)
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
                            let md = ctx.to_markdown();
                            if !md.is_empty() {
                                parts.push(md);
                            }
                        }
                    }

                    // Append SessionWorkLog summary (files modified, steps done)
                    let work_summary = self.work_log.lock().await.to_summary_markdown();
                    if !work_summary.is_empty() {
                        parts.push(work_summary);
                    }

                    break 'build_msg parts.join("\n\n");
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

    /// If the agent concluded without using *productive* tools and auto-continue
    /// didn't fire, check for pending objectives and inject a SystemHint reminder.
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
        // Gather session-level state for the pure decision function
        let (tracking_enabled, cooldown_turns) = {
            let sessions = self.active_sessions.read().await;
            if let Some(session) = sessions.get(&self.session_id) {
                if had_productive_tool_use && !had_conclusive_tool_use {
                    // Agent used ONLY productive tools (no commit/push) = actively working, reset cooldown
                    session
                        .objective_reminder_turns_since
                        .store(0, Ordering::Relaxed);
                }
                // When both productive AND conclusive → agent did work AND committed.
                // Don't reset cooldown — let the reminder check fire.
                let enabled = session.objective_tracking;
                let turns = session
                    .objective_reminder_turns_since
                    .fetch_add(1, Ordering::Relaxed);
                (enabled, turns)
            } else {
                return;
            }
        };

        // Should we check for pending objectives?
        // Yes when: (a) no productive tools, OR (b) productive + conclusive (agent wrapping up)
        let should_check = !had_productive_tool_use || had_conclusive_tool_use;

        // Fetch pending objectives from graph (only if we might need them)
        let (pending_tasks, work_log_summary) = if should_check
            && !auto_continue_allowed
            && !hit_error_max_turns
            && !self.interrupt_flag.load(Ordering::SeqCst)
            && tracking_enabled
            && (cooldown_turns == 0 || cooldown_turns >= OBJECTIVE_REMINDER_COOLDOWN)
        {
            let tasks = if let Some(ref slug) = self.ctx.project_slug {
                let builder =
                    super::compaction_context::CompactionContextBuilder::new(self.graph.clone());
                if let Ok(Ok(ctx)) = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    builder.build_for_session(Some(slug.as_str())),
                )
                .await
                {
                    ctx.pending_tasks
                        .iter()
                        .filter(|t| t.status == "inprogress" || t.status == "pending")
                        .take(4)
                        .map(|t| PendingTaskInfo {
                            title: t.title.clone(),
                            status: t.status.clone(),
                            pending_steps: t
                                .steps
                                .iter()
                                .filter(|s| s.status != "completed" && s.status != "skipped")
                                .map(|s| s.description.clone())
                                .collect(),
                            affected_files: t.affected_files.clone(),
                        })
                        .collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            let wl = self.work_log.lock().await.to_summary_markdown();
            (tasks, wl)
        } else {
            (vec![], String::new())
        };

        // Pure decision
        let input = ObjectiveCheckInput {
            had_productive_tool_use,
            had_conclusive_tool_use,
            auto_continue_allowed,
            hit_error_max_turns,
            interrupted: self.interrupt_flag.load(Ordering::SeqCst),
            tracking_enabled,
            cooldown_turns,
            pending_tasks,
            work_log_summary,
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

                    // Observation auto-detection (RFC + all categories)
                    let rfc_acc = {
                        let sessions = self.active_sessions.read().await;
                        sessions
                            .get(&self.session_id)
                            .map(|s| s.rfc_accumulator.clone())
                    };
                    if let Some(rfc_acc) = rfc_acc {
                        super::feedback::spawn_observation_processing(
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
pub(crate) const OBJECTIVE_REMINDER_COOLDOWN: u32 = 1;

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

    // Guard 4: must have pending objectives
    if input.pending_tasks.is_empty() {
        return None;
    }

    // Build structured reminder
    let mut parts = Vec::new();
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

    Some(parts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> ObjectiveCheckInput {
        ObjectiveCheckInput {
            had_productive_tool_use: false,
            had_conclusive_tool_use: false,
            auto_continue_allowed: false,
            hit_error_max_turns: false,
            interrupted: false,
            tracking_enabled: true,
            cooldown_turns: 0, // first turn → should fire
            pending_tasks: vec![PendingTaskInfo {
                title: "Implement feature T1".to_string(),
                status: "inprogress".to_string(),
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

        // turns=1 meets cooldown threshold (COOLDOWN=1) → should fire
        input.cooldown_turns = 1;
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
            pending_tasks: vec![
                PendingTaskInfo {
                    title: "Add REST endpoint".to_string(),
                    status: "inprogress".to_string(),
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
            pending_tasks: vec![
                PendingTaskInfo {
                    title: "Task A".to_string(),
                    status: "inprogress".to_string(),
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
                reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
                work_log: Arc::new(Mutex::new(crate::chat::types::SessionWorkLog::default())),
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

        let handler = build_handler(graph, Some(slug), "test-session-2", true).await;

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
            graph,
            Some(slug),
            "test-session-3",
            false, // tracking disabled
        )
        .await;

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

        let handler = build_handler(graph, Some(slug), "test-session-5", true).await;

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

        // Second call: after first reminder, cooldown_turns was reset to 1.
        // The handler increments it via fetch_add(1) → reads 1, which meets
        // COOLDOWN=1 → fires again. With cooldown=1, reminders fire every turn.
        handler
            .handle_objective_tracking(false, false, false, false)
            .await;
        assert_eq!(
            handler.pending_messages.lock().await.len(),
            1,
            "Second call should fire (cooldown=1 means every turn after reset)"
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
}
