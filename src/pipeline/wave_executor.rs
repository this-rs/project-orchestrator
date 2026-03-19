//! # Wave Executor — Sub-protocol for parallel task execution
//!
//! Manages the lifecycle of a single wave: dispatch tasks in parallel,
//! collect results, and run a merge-check to verify compatibility.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocol::{ProtocolState, ProtocolTransition};

/// A wave to execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveSpec {
    pub wave_number: usize,
    pub plan_id: Uuid,
    pub task_ids: Vec<Uuid>,
    pub cwd: String,
    pub project_slug: Option<String>,
}

/// Result of a single task execution within a wave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionResult {
    pub task_id: Uuid,
    pub status: TaskExecStatus,
    pub message: String,
    pub duration_ms: u64,
    pub files_modified: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskExecStatus {
    Success,
    Failed,
    Timeout,
    Skipped,
}

/// Result of the merge-check after all tasks complete
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeCheckResult {
    pub success: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Result of a complete wave execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveResult {
    pub wave_number: usize,
    pub task_results: Vec<TaskExecutionResult>,
    pub merge_check: Option<MergeCheckResult>,
    pub overall_status: WaveStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub total_duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveStatus {
    /// All tasks succeeded, merge check passed
    Success,
    /// Some tasks failed
    PartialFailure,
    /// All tasks failed
    TotalFailure,
    /// Merge check failed (tasks ok but conflicts)
    MergeConflict,
}

/// Trait for delegating task execution to an external system.
///
/// Implementations can wire into the ChatManager/PlanRunner to spawn real agents,
/// or provide mock implementations for testing.
#[async_trait::async_trait]
pub trait TaskDelegate: Send + Sync {
    /// Execute a task and return the result.
    async fn execute(&self, task_id: Uuid, cwd: &str) -> TaskExecutionResult;
}

/// Default delegate that returns placeholder success results.
/// Used when no real delegation infrastructure is available (V1).
pub struct PlaceholderDelegate;

#[async_trait::async_trait]
impl TaskDelegate for PlaceholderDelegate {
    async fn execute(&self, task_id: Uuid, _cwd: &str) -> TaskExecutionResult {
        TaskExecutionResult {
            task_id,
            status: TaskExecStatus::Success,
            message: "placeholder: task executed successfully".into(),
            duration_ms: 0,
            files_modified: Vec::new(),
        }
    }
}

/// The Wave Executor
pub struct WaveExecutor {
    timeout_secs: u64,
    delegate: std::sync::Arc<dyn TaskDelegate>,
}

impl WaveExecutor {
    /// Create a new wave executor with the given per-task timeout (default 600s / 10 minutes).
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            delegate: std::sync::Arc::new(PlaceholderDelegate),
        }
    }

    /// Create a wave executor with a custom task delegate.
    pub fn with_delegate(timeout_secs: u64, delegate: std::sync::Arc<dyn TaskDelegate>) -> Self {
        Self {
            timeout_secs,
            delegate,
        }
    }

    /// Execute all tasks in a wave, collect results, and optionally run a merge check.
    pub async fn execute(&self, spec: &WaveSpec) -> WaveResult {
        let started_at = Utc::now();

        // Spawn all tasks in parallel
        let mut handles = Vec::new();
        for &task_id in &spec.task_ids {
            let cwd = spec.cwd.clone();
            let delegate = self.delegate.clone();
            handles.push(tokio::spawn(async move {
                delegate.execute(task_id, &cwd).await
            }));
        }

        // Collect results
        let mut task_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => task_results.push(result),
                Err(e) => {
                    // JoinError — task panicked or was cancelled
                    task_results.push(TaskExecutionResult {
                        task_id: Uuid::new_v4(),
                        status: TaskExecStatus::Failed,
                        message: format!("task join error: {e}"),
                        duration_ms: 0,
                        files_modified: Vec::new(),
                    });
                }
            }
        }

        // Determine status from task results
        let mut overall_status = determine_wave_status(&task_results);

        // If all tasks succeeded, run merge check
        let merge_check = if overall_status == WaveStatus::Success && !task_results.is_empty() {
            let check = self.run_merge_check(&spec.cwd, &task_results).await;
            if !check.success {
                overall_status = WaveStatus::MergeConflict;
            }
            Some(check)
        } else {
            None
        };

        let completed_at = Utc::now();
        let total_duration_ms = (completed_at - started_at).num_milliseconds().max(0) as u64;

        WaveResult {
            wave_number: spec.wave_number,
            task_results,
            merge_check,
            overall_status,
            started_at,
            completed_at,
            total_duration_ms,
        }
    }

    /// Execute a single task within a wave via the configured delegate.
    pub async fn execute_task(&self, task_id: Uuid, cwd: &str) -> TaskExecutionResult {
        self.delegate.execute(task_id, cwd).await
    }

    /// Run `cargo check` in the working directory to verify that all task outputs
    /// are compatible (no compilation errors / merge conflicts).
    pub async fn run_merge_check(
        &self,
        cwd: &str,
        _results: &[TaskExecutionResult],
    ) -> MergeCheckResult {
        let output = tokio::process::Command::new("cargo")
            .args(["check", "--message-format=short"])
            .current_dir(cwd)
            .output()
            .await;

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{stdout}\n{stderr}");

                let errors: Vec<String> = combined
                    .lines()
                    .filter(|line| line.contains("error[") || line.contains("error:"))
                    .map(|line| line.to_string())
                    .collect();

                let warnings: Vec<String> = combined
                    .lines()
                    .filter(|line| line.contains("warning:"))
                    .map(|line| line.to_string())
                    .collect();

                let success = output.status.code().unwrap_or(-1) == 0;

                MergeCheckResult {
                    success,
                    errors,
                    warnings,
                }
            }
            Err(e) => MergeCheckResult {
                success: false,
                errors: vec![format!("failed to spawn cargo check: {e}")],
                warnings: Vec::new(),
            },
        }
    }
}

impl Default for WaveExecutor {
    fn default() -> Self {
        Self {
            timeout_secs: 600,
            delegate: std::sync::Arc::new(PlaceholderDelegate),
        }
    }
}

/// Determine the overall [`WaveStatus`] from individual task results.
pub fn determine_wave_status(results: &[TaskExecutionResult]) -> WaveStatus {
    if results.is_empty() {
        return WaveStatus::Success;
    }

    let total = results.len();
    let succeeded = results
        .iter()
        .filter(|r| r.status == TaskExecStatus::Success)
        .count();

    if succeeded == total {
        WaveStatus::Success
    } else if succeeded == 0 {
        WaveStatus::TotalFailure
    } else {
        WaveStatus::PartialFailure
    }
}

/// Generate the protocol states for a wave executor sub-protocol.
///
/// States: init -> dispatch -> execute -> collect -> merge-check -> done/failed
pub fn wave_protocol_states(protocol_id: Uuid) -> Vec<ProtocolState> {
    let init = ProtocolState::start(protocol_id, "init");
    let dispatch = ProtocolState::new(protocol_id, "dispatch");
    let execute = ProtocolState::new(protocol_id, "execute");
    let collect = ProtocolState::new(protocol_id, "collect");
    let merge_check = ProtocolState::new(protocol_id, "merge-check");
    let done = ProtocolState::terminal(protocol_id, "done");
    let failed = ProtocolState::terminal(protocol_id, "failed");

    vec![init, dispatch, execute, collect, merge_check, done, failed]
}

/// Generate the protocol transitions for a wave executor sub-protocol.
///
/// The `states` parameter must be the output of [`wave_protocol_states`].
pub fn wave_protocol_transitions(
    protocol_id: Uuid,
    states: &[ProtocolState],
) -> Vec<ProtocolTransition> {
    // Build a lookup by name for readability
    let by_name: HashMap<&str, Uuid> = states.iter().map(|s| (s.name.as_str(), s.id)).collect();

    let init = by_name["init"];
    let dispatch = by_name["dispatch"];
    let execute = by_name["execute"];
    let collect = by_name["collect"];
    let merge_check = by_name["merge-check"];
    let done = by_name["done"];
    let failed = by_name["failed"];

    vec![
        ProtocolTransition::new(protocol_id, init, dispatch, "wave_ready"),
        ProtocolTransition::new(protocol_id, dispatch, execute, "tasks_dispatched"),
        ProtocolTransition::new(protocol_id, execute, collect, "tasks_completed"),
        ProtocolTransition::new(protocol_id, collect, merge_check, "results_collected"),
        ProtocolTransition::new(protocol_id, merge_check, done, "merge_passed"),
        ProtocolTransition::new(protocol_id, merge_check, failed, "merge_failed"),
        ProtocolTransition::new(protocol_id, execute, failed, "tasks_failed"),
        ProtocolTransition::new(protocol_id, dispatch, failed, "dispatch_error"),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- WaveStatus determination -------------------------------------------

    #[test]
    fn determine_status_empty_wave() {
        let status = determine_wave_status(&[]);
        assert_eq!(status, WaveStatus::Success);
    }

    #[test]
    fn determine_status_all_success() {
        let results = vec![
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Success,
                message: "ok".into(),
                duration_ms: 10,
                files_modified: vec![],
            },
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Success,
                message: "ok".into(),
                duration_ms: 20,
                files_modified: vec![],
            },
        ];
        assert_eq!(determine_wave_status(&results), WaveStatus::Success);
    }

    #[test]
    fn determine_status_all_failed() {
        let results = vec![
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Failed,
                message: "err".into(),
                duration_ms: 0,
                files_modified: vec![],
            },
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Timeout,
                message: "timeout".into(),
                duration_ms: 0,
                files_modified: vec![],
            },
        ];
        assert_eq!(determine_wave_status(&results), WaveStatus::TotalFailure);
    }

    #[test]
    fn determine_status_partial_failure() {
        let results = vec![
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Success,
                message: "ok".into(),
                duration_ms: 10,
                files_modified: vec![],
            },
            TaskExecutionResult {
                task_id: Uuid::new_v4(),
                status: TaskExecStatus::Failed,
                message: "err".into(),
                duration_ms: 0,
                files_modified: vec![],
            },
        ];
        assert_eq!(determine_wave_status(&results), WaveStatus::PartialFailure);
    }

    #[test]
    fn determine_status_skipped_counts_as_non_success() {
        let results = vec![TaskExecutionResult {
            task_id: Uuid::new_v4(),
            status: TaskExecStatus::Skipped,
            message: "skipped".into(),
            duration_ms: 0,
            files_modified: vec![],
        }];
        assert_eq!(determine_wave_status(&results), WaveStatus::TotalFailure);
    }

    // -- MergeCheckResult construction --------------------------------------

    #[test]
    fn merge_check_result_construction() {
        let result = MergeCheckResult {
            success: true,
            errors: vec![],
            warnings: vec!["warning: unused variable".into()],
        };
        assert!(result.success);
        assert!(result.errors.is_empty());
        assert_eq!(result.warnings.len(), 1);

        // Serialization round-trip
        let json = serde_json::to_string(&result).unwrap();
        let back: MergeCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.success, result.success);
        assert_eq!(back.warnings, result.warnings);
    }

    // -- Protocol state generation ------------------------------------------

    #[test]
    fn wave_protocol_states_generates_correct_states() {
        let protocol_id = Uuid::new_v4();
        let states = wave_protocol_states(protocol_id);

        assert_eq!(states.len(), 7);

        // Verify all states belong to the protocol
        for state in &states {
            assert_eq!(state.protocol_id, protocol_id);
        }

        // Check names
        let names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "init",
                "dispatch",
                "execute",
                "collect",
                "merge-check",
                "done",
                "failed"
            ]
        );

        // Check start state
        assert!(states[0].is_start());
        assert!(!states[0].is_terminal());

        // Check terminal states
        assert!(states[5].is_terminal()); // done
        assert!(states[6].is_terminal()); // failed

        // Check intermediate states
        for state in &states[1..5] {
            assert!(!state.is_start());
            assert!(!state.is_terminal());
        }
    }

    #[test]
    fn wave_protocol_transitions_generates_correct_transitions() {
        let protocol_id = Uuid::new_v4();
        let states = wave_protocol_states(protocol_id);
        let transitions = wave_protocol_transitions(protocol_id, &states);

        assert_eq!(transitions.len(), 8);

        // All transitions belong to the protocol
        for t in &transitions {
            assert_eq!(t.protocol_id, protocol_id);
        }

        // Check triggers
        let triggers: Vec<&str> = transitions.iter().map(|t| t.trigger.as_str()).collect();
        assert!(triggers.contains(&"wave_ready"));
        assert!(triggers.contains(&"tasks_dispatched"));
        assert!(triggers.contains(&"tasks_completed"));
        assert!(triggers.contains(&"results_collected"));
        assert!(triggers.contains(&"merge_passed"));
        assert!(triggers.contains(&"merge_failed"));
        assert!(triggers.contains(&"tasks_failed"));
        assert!(triggers.contains(&"dispatch_error"));
    }

    // -- Execute with empty wave --------------------------------------------

    #[tokio::test]
    async fn execute_empty_wave() {
        let executor = WaveExecutor::new(60);
        let spec = WaveSpec {
            wave_number: 1,
            plan_id: Uuid::new_v4(),
            task_ids: vec![],
            cwd: "/tmp".into(),
            project_slug: None,
        };

        let result = executor.execute(&spec).await;
        assert_eq!(result.wave_number, 1);
        assert!(result.task_results.is_empty());
        assert_eq!(result.overall_status, WaveStatus::Success);
        // No merge check for empty wave (no tasks to merge)
        assert!(result.merge_check.is_none());
        assert!(result.total_duration_ms < 1000); // should be near-instant
    }

    // -- Serialization round-trips ------------------------------------------

    #[test]
    fn wave_status_roundtrip() {
        for status in [
            WaveStatus::Success,
            WaveStatus::PartialFailure,
            WaveStatus::TotalFailure,
            WaveStatus::MergeConflict,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: WaveStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn task_exec_status_roundtrip() {
        for status in [
            TaskExecStatus::Success,
            TaskExecStatus::Failed,
            TaskExecStatus::Timeout,
            TaskExecStatus::Skipped,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: TaskExecStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn wave_spec_serialization() {
        let spec = WaveSpec {
            wave_number: 3,
            plan_id: Uuid::new_v4(),
            task_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            cwd: "/home/user/project".into(),
            project_slug: Some("my-project".into()),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: WaveSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.wave_number, 3);
        assert_eq!(back.task_ids.len(), 2);
        assert_eq!(back.project_slug.as_deref(), Some("my-project"));
    }

    #[test]
    fn default_executor_has_600s_timeout() {
        let executor = WaveExecutor::default();
        assert_eq!(executor.timeout_secs, 600);
    }
}
