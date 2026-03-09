//! Runner state — persisted execution state for crash recovery
//!
//! The RunnerState is persisted as a (:PlanRun) node in Neo4j,
//! linked to the Plan via (:PlanRun)-[:RUNS]->(:Plan).
//! This enables recovery after server crashes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::models::{PlanRunStatus, TriggerSource};

// ============================================================================
// RunnerState — persisted in Neo4j as :PlanRun node
// ============================================================================

/// Persistent state of a plan execution run.
///
/// Saved to Neo4j at each task transition (not at every event)
/// to enable crash recovery without excessive writes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerState {
    /// Unique run identifier
    pub run_id: Uuid,
    /// The plan being executed
    pub plan_id: Uuid,
    /// Current wave index (0-based)
    pub current_wave: usize,
    /// ID of the task currently being executed (None between tasks)
    pub current_task_id: Option<Uuid>,
    /// IDs of tasks already completed in this run
    pub completed_tasks: Vec<Uuid>,
    /// IDs of tasks that failed in this run
    pub failed_tasks: Vec<Uuid>,
    /// Git branch created for this run
    pub git_branch: String,
    /// When the run started
    pub started_at: DateTime<Utc>,
    /// When the run ended (None if still running)
    pub completed_at: Option<DateTime<Utc>>,
    /// Current run status
    pub status: PlanRunStatus,
    /// Cumulative cost in USD
    pub cumulated_cost_usd: f64,
    /// How this run was triggered
    pub triggered_by: TriggerSource,
    /// Project ID (for scoping)
    pub project_id: Option<Uuid>,
}

impl RunnerState {
    /// Create a new RunnerState for a fresh plan run.
    pub fn new(plan_id: Uuid, git_branch: String, triggered_by: TriggerSource) -> Self {
        Self {
            run_id: Uuid::new_v4(),
            plan_id,
            current_wave: 0,
            current_task_id: None,
            completed_tasks: Vec::new(),
            failed_tasks: Vec::new(),
            git_branch,
            started_at: Utc::now(),
            completed_at: None,
            status: PlanRunStatus::Running,
            cumulated_cost_usd: 0.0,
            triggered_by,
            project_id: None,
        }
    }

    /// Mark the run as completed with a final status.
    pub fn finalize(&mut self, status: PlanRunStatus) {
        self.status = status;
        self.completed_at = Some(Utc::now());
        self.current_task_id = None;
    }

    /// Add cost from a completed task.
    pub fn add_cost(&mut self, cost_usd: f64) {
        self.cumulated_cost_usd += cost_usd;
    }

    /// Check if the budget has been exceeded.
    pub fn is_budget_exceeded(&self, max_cost_usd: f64) -> bool {
        self.cumulated_cost_usd > max_cost_usd
    }

    /// Record a task as completed.
    pub fn mark_task_completed(&mut self, task_id: Uuid) {
        self.completed_tasks.push(task_id);
        self.current_task_id = None;
    }

    /// Record a task as failed.
    pub fn mark_task_failed(&mut self, task_id: Uuid) {
        self.failed_tasks.push(task_id);
        self.current_task_id = None;
    }

    /// Total elapsed seconds since run started.
    pub fn elapsed_secs(&self) -> f64 {
        let end = self.completed_at.unwrap_or_else(Utc::now);
        (end - self.started_at).num_milliseconds() as f64 / 1000.0
    }

    /// Progress percentage (based on completed tasks vs total).
    pub fn progress_pct(&self, total_tasks: usize) -> f64 {
        if total_tasks == 0 {
            return 0.0;
        }
        (self.completed_tasks.len() as f64 / total_tasks as f64) * 100.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_state_new() {
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(
            plan_id,
            "feat/plan-abc123".into(),
            TriggerSource::Manual,
        );

        assert_eq!(state.plan_id, plan_id);
        assert_eq!(state.current_wave, 0);
        assert!(state.current_task_id.is_none());
        assert!(state.completed_tasks.is_empty());
        assert!(state.failed_tasks.is_empty());
        assert_eq!(state.status, PlanRunStatus::Running);
        assert!((state.cumulated_cost_usd - 0.0).abs() < f64::EPSILON);
        assert!(state.completed_at.is_none());
    }

    #[test]
    fn test_runner_state_finalize() {
        let mut state = RunnerState::new(
            Uuid::new_v4(),
            "feat/test".into(),
            TriggerSource::Manual,
        );
        state.finalize(PlanRunStatus::Completed);

        assert_eq!(state.status, PlanRunStatus::Completed);
        assert!(state.completed_at.is_some());
        assert!(state.current_task_id.is_none());
    }

    #[test]
    fn test_budget_tracking() {
        let mut state = RunnerState::new(
            Uuid::new_v4(),
            "feat/test".into(),
            TriggerSource::Manual,
        );

        state.add_cost(3.50);
        state.add_cost(4.20);
        assert!((state.cumulated_cost_usd - 7.70).abs() < 0.001);
        assert!(!state.is_budget_exceeded(10.0));

        state.add_cost(3.00);
        assert!(state.is_budget_exceeded(10.0));
    }

    #[test]
    fn test_task_lifecycle() {
        let mut state = RunnerState::new(
            Uuid::new_v4(),
            "feat/test".into(),
            TriggerSource::Manual,
        );

        let task_1 = Uuid::new_v4();
        let task_2 = Uuid::new_v4();
        let task_3 = Uuid::new_v4();

        state.current_task_id = Some(task_1);
        state.mark_task_completed(task_1);
        assert_eq!(state.completed_tasks.len(), 1);
        assert!(state.current_task_id.is_none());

        state.current_task_id = Some(task_2);
        state.mark_task_failed(task_2);
        assert_eq!(state.failed_tasks.len(), 1);

        state.current_task_id = Some(task_3);
        state.mark_task_completed(task_3);

        assert_eq!(state.progress_pct(5), 40.0);
    }

    #[test]
    fn test_progress_pct_zero_tasks() {
        let state = RunnerState::new(
            Uuid::new_v4(),
            "feat/test".into(),
            TriggerSource::Manual,
        );
        assert!((state.progress_pct(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trigger_source_serialization() {
        let source = TriggerSource::Webhook {
            trigger_id: Uuid::nil(),
            payload_hash: Some("abc123".into()),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"webhook\""));
        assert!(json.contains("abc123"));
    }
}
