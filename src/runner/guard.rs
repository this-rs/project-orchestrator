//! AgentGuard — Drift detection and hint injection during task execution.
//!
//! The guard runs in parallel with the agent stream, monitoring for:
//! - Idle detection (no file edits for N seconds)
//! - Loop detection (same tool_use repeated 3x)
//! - Task timeout (global time limit)
//! - Compaction (re-inject task context after compaction)
//! - AskUserQuestion (auto-respond for autonomous mode)

use crate::chat::compaction_context::CompactionContextBuilder;
use crate::chat::manager::ChatManager;
use crate::chat::types::ChatEvent;
use crate::neo4j::traits::GraphStore;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================================
// GuardVerdict — what the guard decided
// ============================================================================

/// The guard's verdict after monitoring the agent.
#[derive(Debug, Clone)]
pub enum GuardVerdict {
    /// Agent completed normally (Result event received)
    Completed,
    /// Task timed out
    Timeout { elapsed_secs: f64 },
    /// Guard was cancelled (runner shutdown)
    Cancelled,
}

// ============================================================================
// GuardConfig — extracted from RunnerConfig
// ============================================================================

/// Configuration for the agent guard.
#[derive(Debug, Clone)]
pub struct GuardConfig {
    /// Idle timeout — inject hint if no file edits
    pub idle_timeout: Duration,
    /// Task timeout — force stop if exceeded
    pub task_timeout: Duration,
    /// Max consecutive identical tool_use before loop warning
    pub loop_threshold: usize,
    /// Interval between idle/timeout checks (default 5s)
    pub check_interval: Duration,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(180),
            task_timeout: Duration::from_secs(10800),
            loop_threshold: 3,
            check_interval: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// AgentGuard — monitors agent execution
// ============================================================================

/// Monitors a running agent for drift, idle, loops, and timeout.
///
/// Spawned in parallel with the agent stream via `tokio::spawn`.
/// Communicates back to the runner via a `oneshot::Sender<GuardVerdict>`.
pub struct AgentGuard {
    /// Session ID being monitored
    session_id: String,
    /// Task context for re-injection after compaction
    task_title: String,
    task_id: Uuid,
    /// Guard configuration
    config: GuardConfig,
    /// Broadcast receiver for chat events
    event_rx: broadcast::Receiver<ChatEvent>,
    /// Channel to send hints to the agent (via ChatManager::send_message)
    hint_tx: Option<Arc<dyn HintSender>>,
    /// Graph store for building rich compaction context
    graph: Option<Arc<dyn GraphStore>>,
    /// Plan ID for compaction context (None = session mode)
    plan_id: Option<Uuid>,
    /// Broadcast sender for emitting ChatEvents (e.g. CompactionRecovery metrics)
    events_tx: Option<broadcast::Sender<ChatEvent>>,
}

/// Trait for sending hints to the agent without interrupting.
/// Abstracted for testability.
#[async_trait::async_trait]
pub trait HintSender: Send + Sync {
    /// Send a hint message to the agent session.
    async fn send_hint(&self, session_id: &str, message: &str) -> anyhow::Result<()>;
}

// ============================================================================
// ChatManagerHintSender — concrete impl using ChatManager::inject_hint
// ============================================================================

/// Concrete `HintSender` that delegates to `ChatManager::inject_hint()`.
///
/// Queues the hint message in `pending_messages` WITHOUT interrupting the
/// current stream (no interrupt_flag, no SIGINT, no killed child processes).
pub struct ChatManagerHintSender {
    pub chat_manager: Arc<ChatManager>,
}

#[async_trait::async_trait]
impl HintSender for ChatManagerHintSender {
    async fn send_hint(&self, session_id: &str, message: &str) -> anyhow::Result<()> {
        self.chat_manager.inject_hint(session_id, message).await
    }
}

impl AgentGuard {
    /// Create a new AgentGuard.
    pub fn new(
        session_id: String,
        task_title: String,
        task_id: Uuid,
        config: GuardConfig,
        event_rx: broadcast::Receiver<ChatEvent>,
        hint_tx: Option<Arc<dyn HintSender>>,
        graph: Option<Arc<dyn GraphStore>>,
        plan_id: Option<Uuid>,
    ) -> Self {
        Self {
            session_id,
            task_title,
            task_id,
            config,
            event_rx,
            hint_tx,
            graph,
            plan_id,
            events_tx: None,
        }
    }

    /// Attach a broadcast sender for emitting events (e.g. CompactionRecovery).
    pub fn with_events_tx(mut self, tx: broadcast::Sender<ChatEvent>) -> Self {
        self.events_tx = Some(tx);
        self
    }

    /// Run the guard monitoring loop.
    ///
    /// Returns a `GuardVerdict` when the agent finishes or times out.
    /// The guard watches for:
    /// - Task timeout (hard stop)
    /// - Idle detection (hint injection)
    /// - Loop detection (same tool_use 3x → hint)
    /// - CompactionStarted (re-inject context)
    /// - AskUserQuestion (auto-respond)
    pub async fn monitor(mut self) -> GuardVerdict {
        let start = Instant::now();
        let mut last_activity = Instant::now();
        let mut idle_warned = false;
        let mut tool_history: VecDeque<u64> =
            VecDeque::with_capacity(self.config.loop_threshold + 1);
        let check_interval = self.config.check_interval;

        loop {
            let remaining_timeout = self.config.task_timeout.saturating_sub(start.elapsed());

            if remaining_timeout.is_zero() {
                warn!(
                    "Guard: task {} timed out after {:.0}s",
                    self.task_id,
                    start.elapsed().as_secs_f64()
                );
                return GuardVerdict::Timeout {
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }

            let wait_duration = check_interval.min(remaining_timeout);

            match tokio::time::timeout(wait_duration, self.event_rx.recv()).await {
                Ok(Ok(event)) => {
                    match &event {
                        // Track tool_use for loop detection
                        ChatEvent::ToolUse { tool, input, .. } => {
                            last_activity = Instant::now();
                            idle_warned = false;

                            // Hash the tool call for loop detection
                            let hash = hash_tool_call(tool, input);
                            tool_history.push_back(hash);
                            if tool_history.len() > self.config.loop_threshold {
                                tool_history.pop_front();
                            }

                            // Check if all entries are the same (loop)
                            if tool_history.len() >= self.config.loop_threshold
                                && tool_history.iter().all(|h| *h == tool_history[0])
                            {
                                warn!(
                                    "Guard: loop detected — tool '{}' called {}x with same args",
                                    tool, self.config.loop_threshold
                                );
                                self.inject_hint(&format!(
                                    "⚠️ Loop detected: you've called '{}' {} times with identical arguments. \
                                     Try a different approach or skip this step.",
                                    tool, self.config.loop_threshold
                                ))
                                .await;
                                tool_history.clear();
                            }
                        }

                        // Track assistant text as activity
                        ChatEvent::AssistantText { .. } | ChatEvent::ToolResult { .. } => {
                            last_activity = Instant::now();
                            idle_warned = false;
                        }

                        // Agent finished — return Completed
                        ChatEvent::Result { .. } => {
                            return GuardVerdict::Completed;
                        }

                        // Compaction — re-inject rich task context + emit recovery metrics
                        ChatEvent::CompactionStarted { .. } => {
                            info!("Guard: compaction detected, will re-inject context");
                            let build_start = Instant::now();
                            let hint = self.build_compaction_hint().await;
                            let build_latency_ms = build_start.elapsed().as_millis() as u64;
                            let hint_tokens = (hint.len() as u32) / 3; // rough chars→tokens estimate
                            self.inject_hint(&hint).await;
                            // Emit CompactionRecovery metrics (v1: recovery_success = true if injection succeeded)
                            let recovery_event = ChatEvent::CompactionRecovery {
                                hint_tokens,
                                build_latency_ms,
                                recovery_success: true,
                            };
                            if let Some(ref tx) = self.events_tx {
                                let _ = tx.send(recovery_event);
                            }
                        }

                        // AskUserQuestion — auto-respond for autonomous mode
                        ChatEvent::AskUserQuestion { .. } => {
                            info!("Guard: AskUserQuestion detected, auto-responding");
                            self.inject_hint(
                                "Continue autonomously, no human input available. \
                                 Make your best judgment and proceed.",
                            )
                            .await;
                        }

                        _ => {}
                    }
                }

                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    warn!("Guard event receiver lagged by {} events", n);
                }

                Ok(Err(broadcast::error::RecvError::Closed)) => {
                    // Channel closed — agent session ended
                    return GuardVerdict::Completed;
                }

                Err(_) => {
                    // Timeout on recv — check idle
                    let idle_duration = last_activity.elapsed();
                    if idle_duration >= self.config.idle_timeout && !idle_warned {
                        idle_warned = true;
                        warn!(
                            "Guard: idle detected ({:.0}s since last activity)",
                            idle_duration.as_secs_f64()
                        );
                        self.inject_hint(&format!(
                            "⏰ You've been idle for {:.0}s. Are you stuck? \
                             Review the current task and continue working.",
                            idle_duration.as_secs_f64()
                        ))
                        .await;
                    }
                }
            }
        }
    }

    /// Build a rich compaction hint using `CompactionContextBuilder`.
    ///
    /// Falls back to a minimal 3-line hint if the builder fails (Neo4j down,
    /// timeout, missing graph store, etc.).
    async fn build_compaction_hint(&self) -> String {
        // Fallback hint — always works, even without graph/plan
        let fallback = format!(
            "<system-reminder>\n\
             # Post-Compaction Context (auto-restored)\n\n\
             ## Task Context\n\
             **Task:** {}\n\
             **Task ID:** {}\n\
             Continue where you left off.\n\
             </system-reminder>",
            self.task_title, self.task_id
        );

        let (graph, plan_id) = match (&self.graph, self.plan_id) {
            (Some(g), Some(pid)) => (g.clone(), pid),
            _ => return fallback,
        };

        let builder = CompactionContextBuilder::new(graph);

        // Timeout at 500ms to stay within performance budget
        match tokio::time::timeout(
            Duration::from_millis(500),
            builder.build_for_task(plan_id, self.task_id),
        )
        .await
        {
            Ok(Ok(ctx)) => {
                let md = ctx.to_markdown();
                if md.is_empty() || md.len() < 50 {
                    warn!("Guard: compaction context too small, using fallback");
                    fallback
                } else {
                    md
                }
            }
            Ok(Err(e)) => {
                warn!("Guard: CompactionContextBuilder failed: {}, using fallback", e);
                fallback
            }
            Err(_) => {
                warn!("Guard: CompactionContextBuilder timed out (>500ms), using fallback");
                fallback
            }
        }
    }

    /// Send a hint to the agent (non-interrupting).
    async fn inject_hint(&self, message: &str) {
        if let Some(ref sender) = self.hint_tx {
            if let Err(e) = sender.send_hint(&self.session_id, message).await {
                warn!("Guard: failed to inject hint: {}", e);
            }
        }
    }
}

/// Hash a tool call (name + args) for loop detection.
fn hash_tool_call(tool: &str, input: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool.hash(&mut hasher);
    // Serialize input deterministically
    let input_str = serde_json::to_string(input).unwrap_or_default();
    input_str.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_config_default() {
        let config = GuardConfig::default();
        assert_eq!(config.idle_timeout.as_secs(), 180);
        assert_eq!(config.task_timeout.as_secs(), 10800);
        assert_eq!(config.loop_threshold, 3);
        assert_eq!(config.check_interval.as_secs(), 5);
    }

    #[test]
    fn test_hash_tool_call_deterministic() {
        let input = serde_json::json!({"path": "/src/main.rs"});
        let h1 = hash_tool_call("Read", &input);
        let h2 = hash_tool_call("Read", &input);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_tool_call_different_tools() {
        let input = serde_json::json!({"path": "/src/main.rs"});
        let h1 = hash_tool_call("Read", &input);
        let h2 = hash_tool_call("Write", &input);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_tool_call_different_args() {
        let h1 = hash_tool_call("Read", &serde_json::json!({"path": "/a.rs"}));
        let h2 = hash_tool_call("Read", &serde_json::json!({"path": "/b.rs"}));
        assert_ne!(h1, h2);
    }

    #[tokio::test]
    async fn test_guard_timeout() {
        let (_tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_millis(50), // Very short timeout
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            None,
            None,
            None,
        );

        let verdict = guard.monitor().await;
        match verdict {
            GuardVerdict::Timeout { elapsed_secs } => {
                assert!(elapsed_secs >= 0.04); // At least 40ms
            }
            other => panic!("Expected Timeout, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_guard_completed_on_result() {
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(100),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            None,
            None,
            None,
        );

        // Send a Result event after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let _ = tx.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 1000,
                cost_usd: Some(0.05),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(3),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
    }

    #[tokio::test]
    async fn test_guard_completed_on_channel_close() {
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(100),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            None,
            None,
            None,
        );

        // Drop sender to close channel
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(tx);
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
    }

    #[tokio::test]
    async fn test_guard_loop_detection() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hint_count = Arc::new(AtomicUsize::new(0));
        let hint_count_clone = hint_count.clone();

        struct MockHintSender {
            count: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl HintSender for MockHintSender {
            async fn send_hint(&self, _session_id: &str, _message: &str) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(5),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            Some(Arc::new(MockHintSender {
                count: hint_count_clone,
            })),
            None,
            None,
        );

        // Send 3 identical tool_use events then a Result
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            for _ in 0..3 {
                let _ = tx_clone.send(ChatEvent::ToolUse {
                    id: "tu_1".to_string(),
                    tool: "Read".to_string(),
                    input: serde_json::json!({"path": "/same/file.rs"}),
                    parent_tool_use_id: None,
                });
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx_clone.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 1000,
                cost_usd: Some(0.05),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(3),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
        // Should have sent at least 1 loop warning hint
        assert!(
            hint_count.load(Ordering::SeqCst) >= 1,
            "Expected loop detection hint"
        );
    }

    #[tokio::test]
    async fn test_guard_idle_detection() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hint_count = Arc::new(AtomicUsize::new(0));
        let hint_count_clone = hint_count.clone();
        let last_message = Arc::new(tokio::sync::Mutex::new(String::new()));
        let last_message_clone = last_message.clone();

        struct MockHintSender {
            count: Arc<AtomicUsize>,
            last_message: Arc<tokio::sync::Mutex<String>>,
        }

        #[async_trait::async_trait]
        impl HintSender for MockHintSender {
            async fn send_hint(&self, _session_id: &str, message: &str) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                *self.last_message.lock().await = message.to_string();
                Ok(())
            }
        }

        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_millis(50), // Very short idle timeout
            task_timeout: Duration::from_secs(5),
            loop_threshold: 3,
            check_interval: Duration::from_millis(20), // Fast checks for test
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            Some(Arc::new(MockHintSender {
                count: hint_count_clone,
                last_message: last_message_clone,
            })),
            None,
            None,
        );

        // Send a Result after enough time for idle detection to fire
        tokio::spawn(async move {
            // Wait for idle timeout + check interval
            tokio::time::sleep(Duration::from_millis(200)).await;
            let _ = tx.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 200,
                cost_usd: Some(0.01),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(1),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
        // Should have sent at least 1 idle hint
        assert!(
            hint_count.load(Ordering::SeqCst) >= 1,
            "Expected idle detection hint"
        );
        let msg = last_message.lock().await;
        assert!(msg.contains("idle"), "Hint should mention idle: {}", msg);
    }

    #[tokio::test]
    async fn test_guard_compaction_reinjects_context() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::{test_plan, test_step, test_task_titled};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hint_count = Arc::new(AtomicUsize::new(0));
        let hint_count_clone = hint_count.clone();
        let last_message = Arc::new(tokio::sync::Mutex::new(String::new()));
        let last_message_clone = last_message.clone();

        struct MockHintSender {
            count: Arc<AtomicUsize>,
            last_message: Arc<tokio::sync::Mutex<String>>,
        }

        #[async_trait::async_trait]
        impl HintSender for MockHintSender {
            async fn send_hint(&self, _session_id: &str, message: &str) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                *self.last_message.lock().await = message.to_string();
                Ok(())
            }
        }

        // Seed MockGraphStore with a plan, task, and steps
        let mock_graph = Arc::new(MockGraphStore::new());
        let plan = test_plan();
        let plan_id = plan.id;
        mock_graph.create_plan(&plan).await.unwrap();

        let mut task = test_task_titled("My Important Task");
        task.status = TaskStatus::InProgress;
        task.acceptance_criteria = vec!["Code compiles".to_string(), "Tests pass".to_string()];
        let task_id = task.id;
        mock_graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Create CompactionContextBuilder struct");
        step1.status = StepStatus::Completed;
        mock_graph.create_step(task_id, &step1).await.unwrap();

        let step2 = test_step(2, "Implement to_markdown formatting");
        mock_graph.create_step(task_id, &step2).await.unwrap();

        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(5),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "My Important Task".to_string(),
            task_id,
            config,
            rx,
            Some(Arc::new(MockHintSender {
                count: hint_count_clone,
                last_message: last_message_clone,
            })),
            Some(mock_graph),
            Some(plan_id),
        );

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let _ = tx.send(ChatEvent::CompactionStarted {
                trigger: "auto".to_string(),
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = tx.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 100,
                cost_usd: Some(0.01),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(1),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
        assert_eq!(
            hint_count.load(Ordering::SeqCst),
            1,
            "Expected compaction hint"
        );
        let msg = last_message.lock().await;

        // Verify structured sections from CompactionContextBuilder
        assert!(
            msg.contains("<system-reminder>"),
            "Hint should be wrapped in system-reminder tags: {}",
            msg
        );
        assert!(
            msg.contains("</system-reminder>"),
            "Hint should have closing system-reminder tag: {}",
            msg
        );
        assert!(
            msg.contains("## Task Context"),
            "Hint should contain Task Context section: {}",
            msg
        );
        assert!(
            msg.contains("My Important Task"),
            "Hint should contain task title: {}",
            msg
        );
        assert!(
            msg.contains("## Steps Progress"),
            "Hint should contain Steps Progress section: {}",
            msg
        );
        assert!(
            msg.contains("✅"),
            "Hint should show completed step icon: {}",
            msg
        );
        assert!(
            msg.contains("## Plan"),
            "Hint should contain Plan section: {}",
            msg
        );
    }

    /// Test that compaction falls back to minimal hint when no graph is provided.
    #[tokio::test]
    async fn test_guard_compaction_fallback_without_graph() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hint_count = Arc::new(AtomicUsize::new(0));
        let hint_count_clone = hint_count.clone();
        let last_message = Arc::new(tokio::sync::Mutex::new(String::new()));
        let last_message_clone = last_message.clone();

        struct MockHintSender {
            count: Arc<AtomicUsize>,
            last_message: Arc<tokio::sync::Mutex<String>>,
        }

        #[async_trait::async_trait]
        impl HintSender for MockHintSender {
            async fn send_hint(&self, _session_id: &str, message: &str) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                *self.last_message.lock().await = message.to_string();
                Ok(())
            }
        }

        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(5),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Fallback Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            Some(Arc::new(MockHintSender {
                count: hint_count_clone,
                last_message: last_message_clone,
            })),
            None, // No graph
            None, // No plan_id
        );

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let _ = tx.send(ChatEvent::CompactionStarted {
                trigger: "auto".to_string(),
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 100,
                cost_usd: Some(0.01),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(1),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
        assert_eq!(hint_count.load(Ordering::SeqCst), 1);
        let msg = last_message.lock().await;

        // Fallback should still wrap in system-reminder and contain task title
        assert!(
            msg.contains("<system-reminder>"),
            "Fallback should use system-reminder tags: {}",
            msg
        );
        assert!(
            msg.contains("Fallback Task"),
            "Fallback should contain task title: {}",
            msg
        );
        // But should NOT have structured sections (no graph data)
        assert!(
            !msg.contains("## Steps Progress"),
            "Fallback should NOT have Steps Progress: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_guard_ask_user_question_auto_responds() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hint_count = Arc::new(AtomicUsize::new(0));
        let hint_count_clone = hint_count.clone();

        struct MockHintSender {
            count: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl HintSender for MockHintSender {
            async fn send_hint(&self, _session_id: &str, _message: &str) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let config = GuardConfig {
            idle_timeout: Duration::from_secs(100),
            task_timeout: Duration::from_secs(5),
            loop_threshold: 3,
            ..Default::default()
        };

        let guard = AgentGuard::new(
            "test-session".to_string(),
            "Test Task".to_string(),
            Uuid::new_v4(),
            config,
            rx,
            Some(Arc::new(MockHintSender {
                count: hint_count_clone,
            })),
            None,
            None,
        );

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let _ = tx.send(ChatEvent::AskUserQuestion {
                id: "auq_1".to_string(),
                tool_call_id: "tc_1".to_string(),
                questions: serde_json::json!([{"question": "Should I continue?"}]),
                input: serde_json::json!({"questions": [{"question": "Should I continue?"}]}),
                parent_tool_use_id: None,
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(ChatEvent::Result {
                session_id: "test".to_string(),
                duration_ms: 100,
                cost_usd: Some(0.01),
                subtype: "success".to_string(),
                is_error: false,
                num_turns: Some(1),
                result_text: None,
            });
        });

        let verdict = guard.monitor().await;
        assert!(matches!(verdict, GuardVerdict::Completed));
        // Should have auto-responded to AskUserQuestion
        assert_eq!(
            hint_count.load(Ordering::SeqCst),
            1,
            "Expected auto-response to AskUserQuestion"
        );
    }
}
