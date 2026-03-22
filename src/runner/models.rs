//! Runner models — types for autonomous plan execution
//!
//! Defines configuration, state machine, events, and result types
//! for the PlanRunner execution engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ============================================================================
// RunnerConfig — configurable execution parameters
// ============================================================================

/// Configuration for the PlanRunner execution engine.
///
/// Deserialized from the `[runner]` section of config.yaml.
/// All fields have sensible defaults for autonomous execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunnerConfig {
    /// Maximum time (seconds) for a single task before timeout. Default: 10800 (3h).
    /// Tasks involve code analysis, writing, compilation, tests, and potential retries.
    /// The guard still injects idle reminders after idle_timeout_secs, so drift is caught early.
    pub task_timeout_secs: u64,
    /// Idle timeout (seconds) — if no file edits detected, inject a reminder. Default: 180 (3 min).
    pub idle_timeout_secs: u64,
    /// Maximum retries per task on failure. Default: 1.
    pub max_retries: u32,
    /// Automatically create a PR when the plan completes. Default: false.
    pub auto_pr: bool,
    /// Run `cargo check` / `npm run build` after each task. Default: true.
    pub build_check: bool,
    /// Run tests after each task (language-dependent). Default: false.
    pub test_runner: bool,
    /// Maximum total cost (USD) for the entire plan run. Abort if exceeded. Default: 10.0.
    pub max_cost_usd: f64,
    /// Spawning timeout (seconds) — max time to wait for create_session() before aborting.
    /// Prevents indefinite hangs when ChatManager/Neo4j is unresponsive. Default: 120 (2 min).
    pub spawning_timeout_secs: u64,
    /// CWD validation mode — controls behavior when cwd doesn't match project.root_path.
    /// - "warn" (default): log a warning and emit CwdMismatch event, but continue
    /// - "strict": return an error if cwd doesn't match root_path
    pub cwd_validation: CwdValidation,
    /// Number of consecutive "I'm done" signals before treating the agent as stuck. Default: 5.
    pub completion_loop_threshold: usize,
    /// Maximum message length (chars) to consider as a completion signal. Default: 200.
    /// Messages longer than this are treated as real explanations, not completion loops.
    pub completion_max_chars: usize,
    /// Maximum number of auto-continue cycles before giving up. Default: 5.
    /// Prevents infinite loops when the agent keeps hitting error_max_turns.
    pub max_auto_continues: u32,
}

/// CWD validation mode for the runner.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CwdValidation {
    /// Log a warning on mismatch but continue execution
    #[default]
    Warn,
    /// Return an error on mismatch — blocks execution
    Strict,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            task_timeout_secs: 10800,
            idle_timeout_secs: 180,
            max_retries: 1,
            auto_pr: false,
            build_check: true,
            test_runner: false,
            max_cost_usd: 10.0,
            spawning_timeout_secs: 480,
            cwd_validation: CwdValidation::default(),
            completion_loop_threshold: 5,
            completion_max_chars: 200,
            max_auto_continues: 5,
        }
    }
}

// ============================================================================
// TaskRunStatus — state machine for individual task execution
// ============================================================================

/// Status of a task within the runner's execution pipeline.
///
/// Transitions are validated by `TaskStateMachine::transition()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRunStatus {
    /// Waiting to be executed
    Pending,
    /// Agent session is being created
    Spawning,
    /// Agent is actively working
    Running,
    /// Post-task verification in progress (build check, step validation)
    Verifying,
    /// Task completed successfully
    Completed,
    /// Task failed after retries exhausted
    Failed,
    /// Task exceeded its time limit
    Timeout,
    /// Task is blocked by an unresolved dependency
    Blocked,
}

impl fmt::Display for TaskRunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Spawning => write!(f, "spawning"),
            Self::Running => write!(f, "running"),
            Self::Verifying => write!(f, "verifying"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Timeout => write!(f, "timeout"),
            Self::Blocked => write!(f, "blocked"),
        }
    }
}

// ============================================================================
// TaskStateMachine — validates state transitions
// ============================================================================

/// Validates task status transitions within the runner.
///
/// The runner owns all status transitions — the LLM agent must NOT
/// update task/step statuses via MCP.
pub struct TaskStateMachine;

impl TaskStateMachine {
    /// Attempt a state transition. Returns Ok(()) if valid, Err with reason if not.
    pub fn transition(from: TaskRunStatus, to: TaskRunStatus) -> Result<(), String> {
        use TaskRunStatus::*;

        let valid = matches!(
            (from, to),
            // Normal flow
            (Pending, Spawning)
                | (Spawning, Running)
                | (Running, Verifying)
                | (Verifying, Completed)
                // Failure paths
                | (Spawning, Failed)
                | (Spawning, Timeout)
                | (Running, Failed)
                | (Running, Timeout)
                | (Verifying, Failed)
                // Retry: failed verification goes back to spawning
                | (Verifying, Spawning)
                // Blocking
                | (Pending, Blocked)
                | (Blocked, Pending)
                // Direct completion (e.g., already done or skipped)
                | (Pending, Completed)
        );

        if valid {
            Ok(())
        } else {
            Err(format!("Invalid task transition: {} → {}", from, to))
        }
    }
}

// ============================================================================
// TaskResult — outcome of a single task execution
// ============================================================================

/// Result of executing a single task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskResult {
    /// Task completed successfully
    Success {
        /// Cost in USD for this task
        cost_usd: f64,
        /// Duration in seconds
        duration_secs: f64,
    },
    /// Task failed after retries
    Failed {
        /// Error message or reason
        reason: String,
        /// Number of attempts made
        attempts: u32,
        /// Cost accumulated before failure
        cost_usd: f64,
    },
    /// Task exceeded its time limit
    Timeout {
        /// How long it ran before being killed
        duration_secs: f64,
        /// Cost accumulated before timeout
        cost_usd: f64,
    },
    /// Task was blocked by dependencies
    Blocked {
        /// IDs of blocking tasks
        blocked_by: Vec<Uuid>,
    },
    /// Plan budget exceeded during this task
    BudgetExceeded {
        /// Cumulative cost at the point of exceeding
        cumulated_cost_usd: f64,
        /// Configured limit
        limit_usd: f64,
    },
}

// ============================================================================
// TaskExecutionReport — structured post-execution feedback
// ============================================================================

/// Breakdown of step statuses for a task — used by the step completion guard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepBreakdown {
    /// Number of steps completed successfully
    pub completed: usize,
    /// Number of steps skipped
    pub skipped: usize,
    /// Number of steps still pending
    pub pending: usize,
    /// Total number of steps defined for this task
    pub total: usize,
}

/// Structured report generated after a sub-agent completes a task.
///
/// Captures tool usage metrics from ChatEvents and git-derived data
/// (commits, files modified) to compute a confidence score. Persisted
/// in `AgentExecution.report_json` for analysis and used by the retry
/// logic to inject error context on re-attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionReport {
    /// Total number of tool_use events the agent emitted
    pub tool_use_count: u32,
    /// Breakdown of tool uses by tool name (e.g. {"Edit": 5, "Bash": 3})
    pub tool_use_breakdown: std::collections::HashMap<String, u32>,
    /// Number of tool_result events that contained errors
    pub error_count: u32,
    /// Last error message (if any) — useful for retry context
    pub last_error: Option<String>,
    /// Files modified (from git diff)
    pub files_modified: Vec<String>,
    /// Commit SHAs produced during this task
    pub commits: Vec<String>,
    /// Whether the task ended with agent-reported success
    pub agent_success: bool,
    /// Cost in USD
    pub cost_usd: f64,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Computed confidence score (0.0 - 1.0)
    ///
    /// Formula: starts at 1.0, penalized by:
    /// - error_ratio (errors / tool_uses): -0.3 * ratio
    /// - zero commits: -0.3
    /// - zero files modified: -0.2
    /// - agent failure: -0.4
    pub confidence_score: f64,
}

impl TaskExecutionReport {
    /// Compute the confidence score based on the report's metrics.
    pub fn compute_confidence(&mut self) {
        let mut score = 1.0_f64;

        // Penalize by error ratio
        if self.tool_use_count > 0 {
            let error_ratio = self.error_count as f64 / self.tool_use_count as f64;
            score -= 0.3 * error_ratio;
        }

        // Penalize for no commits
        if self.commits.is_empty() {
            score -= 0.3;
        }

        // Penalize for no file modifications
        if self.files_modified.is_empty() {
            score -= 0.2;
        }

        // Penalize for agent failure
        if !self.agent_success {
            score -= 0.4;
        }

        self.confidence_score = score.clamp(0.0, 1.0);
    }
}

// ============================================================================
// RunStatus — overall plan run status
// ============================================================================

/// Status of an entire plan run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanRunStatus {
    /// Run is currently executing
    Running,
    /// All tasks completed successfully
    Completed,
    /// One or more tasks failed and the run was aborted
    Failed,
    /// Run was cancelled by the user
    Cancelled,
    /// Run was aborted because the budget was exceeded
    BudgetExceeded,
}

impl fmt::Display for PlanRunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::BudgetExceeded => write!(f, "budget_exceeded"),
        }
    }
}

// ============================================================================
// RunnerEvent — events emitted during plan execution
// ============================================================================

/// Events emitted by the PlanRunner for real-time frontend updates.
///
/// Broadcasted via the existing `tokio::sync::broadcast` channel
/// and filterable by `entity_type=runner` on the WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RunnerEvent {
    /// A plan run has started
    PlanStarted {
        run_id: Uuid,
        plan_id: Uuid,
        plan_title: String,
        total_tasks: usize,
        total_waves: usize,
        /// Prediction based on historical runs (None if first run)
        #[serde(skip_serializing_if = "Option::is_none")]
        prediction: Option<crate::runner::vector::RunPrediction>,
    },
    /// A new wave of tasks has started
    WaveStarted {
        run_id: Uuid,
        wave_number: usize,
        task_count: usize,
    },
    /// A task has started executing
    TaskStarted {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        wave_number: usize,
    },
    /// A task has completed successfully
    TaskCompleted {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        cost_usd: f64,
        duration_secs: f64,
    },
    /// A task has failed
    TaskFailed {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        reason: String,
        attempts: u32,
    },
    /// A task has timed out
    TaskTimeout {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        duration_secs: f64,
    },
    /// A wave of tasks has completed
    WaveCompleted {
        run_id: Uuid,
        wave_number: usize,
        tasks_completed: usize,
        tasks_failed: usize,
    },
    /// The entire plan run has completed
    PlanCompleted {
        run_id: Uuid,
        plan_id: Uuid,
        status: PlanRunStatus,
        total_cost_usd: f64,
        total_duration_secs: f64,
        tasks_completed: usize,
        tasks_failed: usize,
        /// PR URL if auto-PR was created
        pr_url: Option<String>,
    },
    /// Task marked completed but no steps were actually completed (all skipped)
    TaskCompletedWithoutSteps {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        steps_skipped: usize,
        steps_total: usize,
    },
    /// CWD doesn't match the project's root_path
    CwdMismatch {
        run_id: Uuid,
        cwd: String,
        root_path: String,
    },
    /// Agent spawning timed out — create_session() took too long
    TaskSpawningTimeout {
        run_id: Uuid,
        task_id: Uuid,
        task_title: String,
        timeout_secs: u64,
    },
    /// A runner error occurred (non-fatal)
    RunnerError { run_id: Uuid, message: String },
    /// Budget limit exceeded — plan execution aborted
    BudgetExceeded {
        run_id: Uuid,
        plan_id: Uuid,
        cumulated_cost_usd: f64,
        limit_usd: f64,
    },
    /// Worktree commits were recovered after a wave
    WorktreeRecovery {
        run_id: Uuid,
        wave_number: usize,
        commits_recovered: usize,
        conflicts: usize,
        worktrees_cleaned: usize,
    },
    /// Lifecycle protocol FSM transition fired (for FSM Viewer)
    LifecycleTransition {
        run_id: Uuid,
        lifecycle_run_id: Uuid,
        from_state: String,
        to_state: String,
        trigger: String,
    },
}

// ============================================================================
// TriggerSource — tracks how a run was initiated
// ============================================================================

/// How a plan run was triggered.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerSource {
    /// Manually started via API or MCP
    #[default]
    Manual,
    /// Started from a chat conversation (LLM called plan(run))
    Chat { session_id: Option<String> },
    /// Started by a cron schedule
    Schedule { trigger_id: Uuid },
    /// Started by an external webhook (e.g., GitHub)
    Webhook {
        trigger_id: Uuid,
        payload_hash: Option<String>,
    },
    /// Started by an internal event (e.g., another plan completing)
    Event {
        trigger_id: Uuid,
        source_event: String,
    },
}

// ============================================================================
// SpawnedBy — origin tracking for agent sessions
// ============================================================================

/// Describes what spawned an agent session.
///
/// Serialized as a tagged enum in JSON (e.g. `{"type":"pipeline", ...}`).
/// Stored as a JSON string in Neo4j `ChatSession.spawned_by`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpawnedBy {
    /// Spawned by the PlanRunner for a specific task
    Runner { run_id: Uuid, plan_id: Uuid },
    /// Spawned as a sub-conversation from another session
    Conversation { session_id: Uuid },
    /// Spawned by the pipeline executor (wave-based execution)
    Pipeline {
        run_id: Uuid,
        plan_id: Uuid,
        wave: usize,
        task_id: Option<Uuid>,
    },
    /// Spawned by a quality gate retry
    Gate {
        run_id: Uuid,
        gate_type: String,
        retry_count: u32,
    },
    /// Spawned by an external trigger (webhook, cron, event)
    Trigger {
        trigger_id: Uuid,
        event_type: String,
    },
}

// ============================================================================
// ActiveAgent — tracks a currently running agent for a task
// ============================================================================

/// Represents an active agent working on a task within the runner.
///
/// Multiple agents can be active simultaneously when tasks run in parallel waves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAgent {
    /// The task this agent is working on
    pub task_id: Uuid,
    /// Human-readable title of the task
    pub task_title: String,
    /// Chat session ID for this agent (set after spawning)
    pub session_id: Option<Uuid>,
    /// When this agent started working
    pub started_at: DateTime<Utc>,
    /// Accumulated cost in USD for this agent
    pub cost_usd: f64,
    /// Current status of the task
    pub status: TaskRunStatus,
}

// ============================================================================
// ActiveAgentSnapshot — serializable snapshot for status endpoints
// ============================================================================

/// A snapshot of an active agent for the status endpoint.
///
/// Contains derived fields (elapsed_secs) computed at snapshot time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAgentSnapshot {
    pub task_id: Uuid,
    pub task_title: String,
    pub session_id: Option<Uuid>,
    pub elapsed_secs: f64,
    pub cost_usd: f64,
    pub status: TaskRunStatus,
}

impl ActiveAgent {
    /// Create a new ActiveAgent for a task.
    pub fn new(task_id: Uuid, task_title: String) -> Self {
        Self {
            task_id,
            task_title,
            session_id: None,
            started_at: Utc::now(),
            cost_usd: 0.0,
            status: TaskRunStatus::Spawning,
        }
    }

    /// Create a snapshot of this agent at the current point in time.
    pub fn snapshot(&self) -> ActiveAgentSnapshot {
        let elapsed_secs = (Utc::now() - self.started_at).num_milliseconds() as f64 / 1000.0;
        ActiveAgentSnapshot {
            task_id: self.task_id,
            task_title: self.task_title.clone(),
            session_id: self.session_id,
            elapsed_secs,
            cost_usd: self.cost_usd,
            status: self.status,
        }
    }
}

// ============================================================================
// RunSnapshot — status snapshot for the /status endpoint
// ============================================================================

/// A snapshot of the current run status, returned by PlanRunner::status().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub run_id: Uuid,
    pub plan_id: Uuid,
    pub status: PlanRunStatus,
    pub current_wave: usize,
    pub total_waves: usize,
    /// Deprecated: use `active_agents` instead. Kept for backward compatibility.
    /// Returns the task_id of the first active agent (if exactly one).
    pub current_task_id: Option<Uuid>,
    /// Deprecated: use `active_agents` instead. Kept for backward compatibility.
    /// Returns the task_title of the first active agent (if exactly one).
    pub current_task_title: Option<String>,
    /// All currently active agents (may be multiple in parallel waves).
    pub active_agents: Vec<ActiveAgentSnapshot>,
    pub tasks_completed: usize,
    pub tasks_failed: usize,
    pub tasks_total: usize,
    pub cumulated_cost_usd: f64,
    pub elapsed_secs: f64,
    pub started_at: DateTime<Utc>,
    pub triggered_by: TriggerSource,
}

// ============================================================================
// Trigger — automatic plan execution triggers
// ============================================================================

/// Type of trigger that can start a plan run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerType {
    /// Cron-based schedule (e.g., "0 2 * * *")
    Schedule,
    /// External webhook (e.g., GitHub push)
    Webhook,
    /// Internal event (e.g., another plan completed)
    Event,
    /// Chat intent (LLM calls plan(run))
    Chat,
}

impl fmt::Display for TriggerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schedule => write!(f, "schedule"),
            Self::Webhook => write!(f, "webhook"),
            Self::Event => write!(f, "event"),
            Self::Chat => write!(f, "chat"),
        }
    }
}

/// A trigger node persisted in Neo4j, linked to a Plan via (:Trigger)-[:TRIGGERS]->(:Plan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    pub id: Uuid,
    pub plan_id: Uuid,
    pub trigger_type: TriggerType,
    /// Type-specific configuration (cron expression, webhook secret, event filter, etc.)
    pub config: serde_json::Value,
    /// Whether this trigger is active.
    pub enabled: bool,
    /// Minimum seconds between firings (0 = no cooldown).
    pub cooldown_secs: u64,
    /// Last time this trigger fired.
    pub last_fired: Option<DateTime<Utc>>,
    /// Total number of times this trigger has fired.
    pub fire_count: u64,
    pub created_at: DateTime<Utc>,
}

/// A firing record — one entry per trigger activation.
/// Persisted as (:TriggerFiring)-[:FIRED_BY]->(:Trigger).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerFiring {
    pub id: Uuid,
    pub trigger_id: Uuid,
    /// The PlanRun that was created by this firing.
    pub plan_run_id: Option<Uuid>,
    pub fired_at: DateTime<Utc>,
    /// Optional payload from the trigger source (webhook body, event data, etc.)
    pub source_payload: Option<serde_json::Value>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_config_defaults() {
        let config = RunnerConfig::default();
        assert_eq!(config.task_timeout_secs, 10800);
        assert_eq!(config.idle_timeout_secs, 180);
        assert_eq!(config.max_retries, 1);
        assert!(!config.auto_pr);
        assert!(config.build_check);
        assert!(!config.test_runner);
        assert!((config.max_cost_usd - 10.0).abs() < f64::EPSILON);
        assert_eq!(config.spawning_timeout_secs, 480);
    }

    #[test]
    fn test_runner_config_deserialize_with_defaults() {
        let yaml = r#"
            task_timeout_secs: 600
            auto_pr: true
        "#;
        let config: RunnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.task_timeout_secs, 600);
        assert!(config.auto_pr);
        // Defaults applied for missing fields
        assert_eq!(config.idle_timeout_secs, 180);
        assert_eq!(config.max_retries, 1);
    }

    #[test]
    fn test_task_state_machine_valid_transitions() {
        use TaskRunStatus::*;

        // Normal flow
        assert!(TaskStateMachine::transition(Pending, Spawning).is_ok());
        assert!(TaskStateMachine::transition(Spawning, Running).is_ok());
        assert!(TaskStateMachine::transition(Running, Verifying).is_ok());
        assert!(TaskStateMachine::transition(Verifying, Completed).is_ok());

        // Failure paths
        assert!(TaskStateMachine::transition(Spawning, Failed).is_ok());
        assert!(TaskStateMachine::transition(Spawning, Timeout).is_ok());
        assert!(TaskStateMachine::transition(Running, Failed).is_ok());
        assert!(TaskStateMachine::transition(Running, Timeout).is_ok());
        assert!(TaskStateMachine::transition(Verifying, Failed).is_ok());

        // Retry
        assert!(TaskStateMachine::transition(Verifying, Spawning).is_ok());

        // Blocking
        assert!(TaskStateMachine::transition(Pending, Blocked).is_ok());
        assert!(TaskStateMachine::transition(Blocked, Pending).is_ok());

        // Direct completion
        assert!(TaskStateMachine::transition(Pending, Completed).is_ok());
    }

    #[test]
    fn test_task_state_machine_invalid_transitions() {
        use TaskRunStatus::*;

        assert!(TaskStateMachine::transition(Completed, Running).is_err());
        assert!(TaskStateMachine::transition(Failed, Running).is_err());
        assert!(TaskStateMachine::transition(Timeout, Pending).is_err());
        assert!(TaskStateMachine::transition(Running, Pending).is_err());
        assert!(TaskStateMachine::transition(Completed, Pending).is_err());
        assert!(TaskStateMachine::transition(Blocked, Running).is_err());
    }

    #[test]
    fn test_task_result_serialization() {
        let result = TaskResult::Success {
            cost_usd: 0.42,
            duration_secs: 120.5,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"type\":\"success\""));
        assert!(json.contains("0.42"));

        let result = TaskResult::BudgetExceeded {
            cumulated_cost_usd: 10.5,
            limit_usd: 10.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"type\":\"budget_exceeded\""));
    }

    #[test]
    fn test_runner_event_serialization() {
        let event = RunnerEvent::PlanStarted {
            run_id: Uuid::nil(),
            plan_id: Uuid::nil(),
            plan_title: "Test Plan".into(),
            total_tasks: 5,
            total_waves: 3,
            prediction: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event\":\"plan_started\""));
        assert!(json.contains("\"total_tasks\":5"));
    }

    #[test]
    fn test_trigger_source_default() {
        let source = TriggerSource::default();
        assert_eq!(source, TriggerSource::Manual);
    }

    #[test]
    fn test_plan_run_status_display() {
        assert_eq!(PlanRunStatus::Running.to_string(), "running");
        assert_eq!(PlanRunStatus::BudgetExceeded.to_string(), "budget_exceeded");
    }

    // ========================================================================
    // TaskExecutionReport tests
    // ========================================================================

    fn make_report() -> TaskExecutionReport {
        TaskExecutionReport {
            tool_use_count: 10,
            tool_use_breakdown: std::collections::HashMap::from([
                ("Edit".to_string(), 5),
                ("Bash".to_string(), 5),
            ]),
            error_count: 0,
            last_error: None,
            files_modified: vec!["src/main.rs".to_string()],
            commits: vec!["abc123".to_string()],
            agent_success: true,
            cost_usd: 0.5,
            duration_secs: 60.0,
            confidence_score: 0.0,
        }
    }

    #[test]
    fn test_report_perfect_confidence() {
        let mut report = make_report();
        report.compute_confidence();
        assert!(
            (report.confidence_score - 1.0).abs() < f64::EPSILON,
            "Perfect report should have confidence 1.0, got {}",
            report.confidence_score
        );
    }

    #[test]
    fn test_report_error_ratio_penalty() {
        let mut report = make_report();
        report.error_count = 5; // 50% error ratio → -0.15
        report.compute_confidence();
        let expected = 1.0 - 0.3 * 0.5;
        assert!(
            (report.confidence_score - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            report.confidence_score
        );
    }

    #[test]
    fn test_report_no_commits_penalty() {
        let mut report = make_report();
        report.commits = vec![];
        report.compute_confidence();
        let expected = 1.0 - 0.3;
        assert!(
            (report.confidence_score - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            report.confidence_score
        );
    }

    #[test]
    fn test_report_no_files_modified_penalty() {
        let mut report = make_report();
        report.files_modified = vec![];
        report.compute_confidence();
        let expected = 1.0 - 0.2;
        assert!(
            (report.confidence_score - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            report.confidence_score
        );
    }

    #[test]
    fn test_report_agent_failure_penalty() {
        let mut report = make_report();
        report.agent_success = false;
        report.compute_confidence();
        let expected = 1.0 - 0.4;
        assert!(
            (report.confidence_score - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            report.confidence_score
        );
    }

    #[test]
    fn test_report_all_penalties_clamped_to_zero() {
        let mut report = TaskExecutionReport {
            tool_use_count: 10,
            tool_use_breakdown: std::collections::HashMap::new(),
            error_count: 10, // 100% errors → -0.3
            last_error: Some("boom".to_string()),
            files_modified: vec![], // -0.2
            commits: vec![],        // -0.3
            agent_success: false,   // -0.4
            cost_usd: 1.0,
            duration_secs: 300.0,
            confidence_score: 0.0,
        };
        // Total penalty = 0.3 + 0.3 + 0.2 + 0.4 = 1.2 → clamped to 0.0
        report.compute_confidence();
        assert!(
            report.confidence_score.abs() < f64::EPSILON,
            "Should be clamped to 0.0, got {}",
            report.confidence_score
        );
    }

    #[test]
    fn test_report_zero_tool_uses_no_error_penalty() {
        let mut report = make_report();
        report.tool_use_count = 0;
        report.error_count = 5; // should be ignored when tool_use_count == 0
        report.compute_confidence();
        assert!(
            (report.confidence_score - 1.0).abs() < f64::EPSILON,
            "Zero tool uses should skip error ratio penalty, got {}",
            report.confidence_score
        );
    }

    #[test]
    fn test_report_serialization_roundtrip() {
        let mut report = make_report();
        report.compute_confidence();
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: TaskExecutionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.tool_use_count, 10);
        assert_eq!(deserialized.error_count, 0);
        assert!(deserialized.agent_success);
        assert_eq!(deserialized.commits, vec!["abc123"]);
        assert_eq!(deserialized.files_modified, vec!["src/main.rs"]);
        assert!((deserialized.confidence_score - 1.0).abs() < f64::EPSILON,);
    }

    #[test]
    fn test_report_combined_penalties() {
        let mut report = make_report();
        report.error_count = 2; // 20% error ratio → -0.06
        report.commits = vec![]; // -0.3
        report.agent_success = true;
        report.compute_confidence();
        let expected = 1.0 - (0.3 * 0.2) - 0.3;
        assert!(
            (report.confidence_score - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            report.confidence_score
        );
    }

    // ========================================================================
    // SpawnedBy tests
    // ========================================================================

    #[test]
    fn test_spawned_by_runner_roundtrip() {
        let spawned = SpawnedBy::Runner {
            run_id: Uuid::nil(),
            plan_id: Uuid::nil(),
        };
        let json = serde_json::to_string(&spawned).unwrap();
        assert!(json.contains("\"type\":\"runner\""));
        let back: SpawnedBy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spawned);
    }

    #[test]
    fn test_spawned_by_pipeline_roundtrip() {
        let spawned = SpawnedBy::Pipeline {
            run_id: Uuid::nil(),
            plan_id: Uuid::nil(),
            wave: 3,
            task_id: Some(Uuid::nil()),
        };
        let json = serde_json::to_string(&spawned).unwrap();
        assert!(json.contains("\"type\":\"pipeline\""));
        assert!(json.contains("\"wave\":3"));
        let back: SpawnedBy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spawned);
    }

    #[test]
    fn test_spawned_by_gate_roundtrip() {
        let spawned = SpawnedBy::Gate {
            run_id: Uuid::nil(),
            gate_type: "cargo-check".into(),
            retry_count: 2,
        };
        let json = serde_json::to_string(&spawned).unwrap();
        assert!(json.contains("\"type\":\"gate\""));
        assert!(json.contains("\"gate_type\":\"cargo-check\""));
        let back: SpawnedBy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spawned);
    }

    #[test]
    fn test_spawned_by_trigger_roundtrip() {
        let spawned = SpawnedBy::Trigger {
            trigger_id: Uuid::nil(),
            event_type: "webhook".into(),
        };
        let json = serde_json::to_string(&spawned).unwrap();
        assert!(json.contains("\"type\":\"trigger\""));
        let back: SpawnedBy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spawned);
    }

    #[test]
    fn test_spawned_by_conversation_roundtrip() {
        let spawned = SpawnedBy::Conversation {
            session_id: Uuid::nil(),
        };
        let json = serde_json::to_string(&spawned).unwrap();
        assert!(json.contains("\"type\":\"conversation\""));
        let back: SpawnedBy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spawned);
    }

    // ========================================================================
    // Spawning timeout tests
    // ========================================================================

    #[test]
    fn test_spawning_timeout_event_serialization() {
        let event = RunnerEvent::TaskSpawningTimeout {
            run_id: Uuid::nil(),
            task_id: Uuid::nil(),
            task_title: "Test task".into(),
            timeout_secs: 120,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event\":\"task_spawning_timeout\""));
        assert!(json.contains("\"timeout_secs\":120"));
    }

    #[test]
    fn test_spawning_timeout_secs_in_config() {
        // Default
        let config = RunnerConfig::default();
        assert_eq!(config.spawning_timeout_secs, 480);

        // Custom via YAML
        let yaml = r#"
            spawning_timeout_secs: 30
        "#;
        let config: RunnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.spawning_timeout_secs, 30);
        // Other defaults still applied
        assert_eq!(config.task_timeout_secs, 10800);
    }

    #[test]
    fn test_guard_config_spawning_timeout_default() {
        let config = crate::runner::guard::GuardConfig::default();
        assert_eq!(config.spawning_timeout, std::time::Duration::from_secs(480));
    }

    #[test]
    fn test_spawning_to_timeout_transition() {
        use TaskRunStatus::*;
        assert!(
            TaskStateMachine::transition(Spawning, Timeout).is_ok(),
            "Spawning → Timeout should be a valid transition for spawning timeout"
        );
    }

    // ========================================================================
    // CWD validation tests
    // ========================================================================

    #[test]
    fn test_cwd_validation_default_is_warn() {
        let config = RunnerConfig::default();
        assert_eq!(config.cwd_validation, CwdValidation::Warn);
    }

    #[test]
    fn test_cwd_validation_strict_deserialize() {
        let yaml = r#"
            cwd_validation: strict
        "#;
        let config: RunnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.cwd_validation, CwdValidation::Strict);
    }

    #[test]
    fn test_cwd_mismatch_event_serialization() {
        let event = RunnerEvent::CwdMismatch {
            run_id: Uuid::nil(),
            cwd: "/wrong/path".into(),
            root_path: "/correct/path".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event\":\"cwd_mismatch\""));
        assert!(json.contains("/wrong/path"));
        assert!(json.contains("/correct/path"));
    }
}
