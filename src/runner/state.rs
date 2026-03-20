//! Runner state — persisted execution state for crash recovery
//!
//! The RunnerState is persisted as a (:PlanRun) node in Neo4j,
//! linked to the Plan via (:PlanRun)-[:RUNS]->(:Plan).
//! This enables recovery after server crashes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::models::{ActiveAgent, PlanRunStatus, TaskRunStatus, TriggerSource};

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
    /// Total number of tasks in the plan
    pub total_tasks: usize,
    /// Current wave index (1-based during execution, 0 before start)
    pub current_wave: usize,
    /// Deprecated: use `active_agents` instead. Kept for backward compatibility
    /// with serialized state in Neo4j. Populated from the first active agent.
    #[serde(default)]
    pub current_task_id: Option<Uuid>,
    /// Deprecated: use `active_agents` instead.
    #[serde(default)]
    pub current_task_title: Option<String>,
    /// Currently active agents (one per concurrently executing task).
    #[serde(default)]
    pub active_agents: Vec<ActiveAgent>,
    /// IDs of tasks already completed in this run
    pub completed_tasks: Vec<Uuid>,
    /// IDs of tasks that failed in this run
    pub failed_tasks: Vec<Uuid>,
    /// Per-task retry counts (task_id → retries used)
    #[serde(default)]
    pub retry_counts: std::collections::HashMap<Uuid, u32>,
    /// Git branch created for this run
    pub git_branch: String,
    /// When the run started
    pub started_at: DateTime<Utc>,
    /// When the run ended (None if still running)
    pub completed_at: Option<DateTime<Utc>>,
    /// Current run status
    pub status: PlanRunStatus,
    /// Cumulative cost in USD
    pub cost_usd: f64,
    /// How this run was triggered
    pub triggered_by: TriggerSource,
    /// Project ID (for scoping)
    pub project_id: Option<Uuid>,
    /// Optional lifecycle ProtocolRun tracking this plan execution.
    /// Set when a lifecycle protocol (plan-runner-*) is routed at startup.
    /// None when no lifecycle protocol is available (fallback mode).
    #[serde(default)]
    pub lifecycle_run_id: Option<Uuid>,
}

impl RunnerState {
    /// Create a new RunnerState for a fresh plan run.
    pub fn new(
        run_id: Uuid,
        plan_id: Uuid,
        total_tasks: usize,
        triggered_by: TriggerSource,
    ) -> Self {
        Self {
            run_id,
            plan_id,
            total_tasks,
            current_wave: 0,
            current_task_id: None,
            current_task_title: None,
            active_agents: Vec::new(),
            completed_tasks: Vec::new(),
            failed_tasks: Vec::new(),
            retry_counts: std::collections::HashMap::new(),
            git_branch: String::new(),
            started_at: Utc::now(),
            completed_at: None,
            status: PlanRunStatus::Running,
            cost_usd: 0.0,
            triggered_by,
            project_id: None,
            lifecycle_run_id: None,
        }
    }

    /// Mark the run as completed with a final status.
    pub fn finalize(&mut self, status: PlanRunStatus) {
        self.status = status;
        self.completed_at = Some(Utc::now());
        self.current_task_id = None;
        self.current_task_title = None;
        self.active_agents.clear();
    }

    /// Add cost from a completed task.
    pub fn add_cost(&mut self, cost_usd: f64) {
        self.cost_usd += cost_usd;
    }

    /// Check if the budget has been exceeded.
    pub fn is_budget_exceeded(&self, max_cost_usd: f64) -> bool {
        self.cost_usd > max_cost_usd
    }

    /// Number of completed tasks.
    pub fn tasks_completed(&self) -> usize {
        self.completed_tasks.len()
    }

    /// Record a task as completed. Also removes it from active_agents.
    pub fn mark_task_completed(&mut self, task_id: Uuid) {
        self.completed_tasks.push(task_id);
        self.remove_agent(&task_id);
        self.sync_current_task_compat();
    }

    /// Record a task as failed. Also removes it from active_agents.
    pub fn mark_task_failed(&mut self, task_id: Uuid) {
        self.failed_tasks.push(task_id);
        self.remove_agent(&task_id);
        self.sync_current_task_compat();
    }

    // ========================================================================
    // Active agent management
    // ========================================================================

    /// Add an active agent tracking a running task.
    pub fn add_agent(&mut self, agent: ActiveAgent) {
        self.active_agents.push(agent);
        self.sync_current_task_compat();
    }

    /// Remove an active agent by task_id.
    pub fn remove_agent(&mut self, task_id: &Uuid) {
        self.active_agents.retain(|a| &a.task_id != task_id);
        self.sync_current_task_compat();
    }

    /// Get a reference to an active agent by task_id.
    pub fn get_agent(&self, task_id: &Uuid) -> Option<&ActiveAgent> {
        self.active_agents.iter().find(|a| &a.task_id == task_id)
    }

    /// Update the status of an active agent.
    pub fn update_agent_status(&mut self, task_id: &Uuid, status: TaskRunStatus) {
        if let Some(agent) = self
            .active_agents
            .iter_mut()
            .find(|a| &a.task_id == task_id)
        {
            agent.status = status;
        }
    }

    /// Set the session_id on an active agent (after create_session completes).
    pub fn set_agent_session(&mut self, task_id: &Uuid, session_id: Option<Uuid>) {
        if let Some(agent) = self
            .active_agents
            .iter_mut()
            .find(|a| &a.task_id == task_id)
        {
            agent.session_id = session_id;
        }
    }

    /// Add cost to an active agent.
    pub fn add_agent_cost(&mut self, task_id: &Uuid, cost: f64) {
        if let Some(agent) = self
            .active_agents
            .iter_mut()
            .find(|a| &a.task_id == task_id)
        {
            agent.cost_usd += cost;
        }
    }

    /// Sync deprecated `current_task_id` / `current_task_title` fields
    /// for backward compatibility. When there's exactly one active agent,
    /// these fields reflect that agent.
    fn sync_current_task_compat(&mut self) {
        if self.active_agents.len() == 1 {
            let agent = &self.active_agents[0];
            self.current_task_id = Some(agent.task_id);
            self.current_task_title = Some(agent.task_title.clone());
        } else {
            self.current_task_id = None;
            self.current_task_title = None;
        }
    }

    /// Total elapsed seconds since run started.
    pub fn elapsed_secs(&self) -> f64 {
        let end = self.completed_at.unwrap_or_else(Utc::now);
        (end - self.started_at).num_milliseconds() as f64 / 1000.0
    }

    /// Progress percentage (based on completed tasks vs total).
    pub fn progress_pct(&self) -> f64 {
        if self.total_tasks == 0 {
            return 0.0;
        }
        (self.completed_tasks.len() as f64 / self.total_tasks as f64) * 100.0
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
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, 5, TriggerSource::Manual);

        assert_eq!(state.run_id, run_id);
        assert_eq!(state.plan_id, plan_id);
        assert_eq!(state.total_tasks, 5);
        assert_eq!(state.current_wave, 0);
        assert!(state.current_task_id.is_none());
        assert!(state.current_task_title.is_none());
        assert!(state.active_agents.is_empty());
        assert!(state.completed_tasks.is_empty());
        assert!(state.failed_tasks.is_empty());
        assert!(state.retry_counts.is_empty());
        assert_eq!(state.status, PlanRunStatus::Running);
        assert!((state.cost_usd - 0.0).abs() < f64::EPSILON);
        assert!(state.completed_at.is_none());
    }

    #[test]
    fn test_runner_state_finalize() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();
        state.add_agent(ActiveAgent::new(task_id, "test".into()));
        assert_eq!(state.active_agents.len(), 1);

        state.finalize(PlanRunStatus::Completed);

        assert_eq!(state.status, PlanRunStatus::Completed);
        assert!(state.completed_at.is_some());
        assert!(state.current_task_id.is_none());
        assert!(state.current_task_title.is_none());
        assert!(state.active_agents.is_empty());
    }

    #[test]
    fn test_budget_tracking() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);

        state.add_cost(3.50);
        state.add_cost(4.20);
        assert!((state.cost_usd - 7.70).abs() < 0.001);
        assert!(!state.is_budget_exceeded(10.0));

        state.add_cost(3.00);
        assert!(state.is_budget_exceeded(10.0));
    }

    #[test]
    fn test_task_lifecycle() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 5, TriggerSource::Manual);

        let task_1 = Uuid::new_v4();
        let task_2 = Uuid::new_v4();
        let task_3 = Uuid::new_v4();

        state.add_agent(ActiveAgent::new(task_1, "Task 1".into()));
        state.mark_task_completed(task_1);
        assert_eq!(state.completed_tasks.len(), 1);
        assert!(state.current_task_id.is_none());
        assert!(state.active_agents.is_empty());

        state.add_agent(ActiveAgent::new(task_2, "Task 2".into()));
        state.mark_task_failed(task_2);
        assert_eq!(state.failed_tasks.len(), 1);
        assert!(state.active_agents.is_empty());

        state.add_agent(ActiveAgent::new(task_3, "Task 3".into()));
        state.mark_task_completed(task_3);

        assert_eq!(state.progress_pct(), 40.0);
    }

    #[test]
    fn test_active_agent_single_compat() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();

        // Single agent: current_task_id should be set for backward compat
        state.add_agent(ActiveAgent::new(task_id, "My Task".into()));
        assert_eq!(state.current_task_id, Some(task_id));
        assert_eq!(state.current_task_title.as_deref(), Some("My Task"));
        assert_eq!(state.active_agents.len(), 1);

        // Remove: current_task_id should be cleared
        state.remove_agent(&task_id);
        assert!(state.current_task_id.is_none());
        assert!(state.current_task_title.is_none());
        assert!(state.active_agents.is_empty());
    }

    #[test]
    fn test_active_agent_multi_no_compat() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_1 = Uuid::new_v4();
        let task_2 = Uuid::new_v4();

        state.add_agent(ActiveAgent::new(task_1, "Task 1".into()));
        assert_eq!(state.current_task_id, Some(task_1));

        // Adding a second agent clears the deprecated fields (ambiguous)
        state.add_agent(ActiveAgent::new(task_2, "Task 2".into()));
        assert!(state.current_task_id.is_none());
        assert!(state.current_task_title.is_none());
        assert_eq!(state.active_agents.len(), 2);

        // Removing one restores compat for the remaining single agent
        state.remove_agent(&task_1);
        assert_eq!(state.current_task_id, Some(task_2));
        assert_eq!(state.active_agents.len(), 1);
    }

    #[test]
    fn test_update_agent_status() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();

        state.add_agent(ActiveAgent::new(task_id, "Task".into()));
        assert_eq!(
            state.get_agent(&task_id).unwrap().status,
            TaskRunStatus::Spawning
        );

        state.update_agent_status(&task_id, TaskRunStatus::Running);
        assert_eq!(
            state.get_agent(&task_id).unwrap().status,
            TaskRunStatus::Running
        );
    }

    #[test]
    fn test_add_agent_cost() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();

        state.add_agent(ActiveAgent::new(task_id, "Task".into()));
        state.add_agent_cost(&task_id, 1.50);
        state.add_agent_cost(&task_id, 0.75);
        assert!((state.get_agent(&task_id).unwrap().cost_usd - 2.25).abs() < 0.001);
    }

    #[test]
    fn test_progress_pct_zero_tasks() {
        let state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 0, TriggerSource::Manual);
        assert!((state.progress_pct() - 0.0).abs() < f64::EPSILON);
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

    #[test]
    fn test_retry_counts() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();

        assert_eq!(state.retry_counts.get(&task_id).copied().unwrap_or(0), 0);
        *state.retry_counts.entry(task_id).or_insert(0) += 1;
        assert_eq!(state.retry_counts[&task_id], 1);
    }
}
