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
    /// Maximum time (seconds) for a single task before timeout. Default: 900 (15 min).
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
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            task_timeout_secs: 900,
            idle_timeout_secs: 180,
            max_retries: 1,
            auto_pr: false,
            build_check: true,
            test_runner: false,
            max_cost_usd: 10.0,
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
    /// A runner error occurred (non-fatal)
    RunnerError { run_id: Uuid, message: String },
    /// Budget limit exceeded — plan execution aborted
    BudgetExceeded {
        run_id: Uuid,
        plan_id: Uuid,
        cumulated_cost_usd: f64,
        limit_usd: f64,
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
    pub current_task_id: Option<Uuid>,
    pub current_task_title: Option<String>,
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
        assert_eq!(config.task_timeout_secs, 900);
        assert_eq!(config.idle_timeout_secs, 180);
        assert_eq!(config.max_retries, 1);
        assert!(!config.auto_pr);
        assert!(config.build_check);
        assert!(!config.test_runner);
        assert!((config.max_cost_usd - 10.0).abs() < f64::EPSILON);
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
}
