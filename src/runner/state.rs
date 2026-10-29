//! Runner state — persisted execution state for crash recovery
//!
//! The RunnerState is persisted as a (:PlanRun) node in Neo4j,
//! linked to the Plan via (:PlanRun)-[:RUNS]->(:Plan).
//! This enables recovery after server crashes.
use crate::neo4j::traits::*;

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
    /// MARGINAL counter (A21): the real spend of the run in USD, i.e. the costs
    /// whose basis is `reported` or `priced`. The only counter that can stop a
    /// run. Serialized as `cost_usd` (the name the `PlanRun` node has always had).
    pub cost_usd: f64,
    /// NOTIONAL counter (A21): what the run would have cost on a `subscription`
    /// or a `free` endpoint. Informative, never cuts a run.
    #[serde(default)]
    pub notional_usd: f64,
    /// Tokens (input + output) of every attempt, whatever the cost basis: the
    /// budget of a run whose provider has no price.
    #[serde(default)]
    pub tokens_used: u64,
    /// Attempts whose cost is unknown (no price for the model): counted apart,
    /// never as zero.
    #[serde(default)]
    pub unknown_cost_attempts: u32,
    /// How this run was triggered
    pub triggered_by: TriggerSource,
    /// Project ID (for scoping)
    pub project_id: Option<Uuid>,
    /// Optional lifecycle ProtocolRun tracking this plan execution.
    /// Set when a lifecycle protocol (plan-runner-*) is routed at startup.
    /// None when no lifecycle protocol is available (fallback mode).
    #[serde(default)]
    pub lifecycle_run_id: Option<Uuid>,
    /// Absolute working directory the run executes in, persisted so that a run
    /// resumed after a restart goes back to the SAME directory (the server's own
    /// directory is never a valid substitute). `None` for runs written before
    /// this field existed: those cannot be resumed safely.
    #[serde(default)]
    pub cwd: Option<String>,
}

/// What one attempt cost, as the provider reported it on the turn's `Result`
/// event: an amount only when the provider gave one (a model without a price
/// has `None`, never 0), the basis of that amount, and the tokens of the turn.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AttemptCost {
    /// Amount in USD, `None` when the provider stated no price.
    pub usd: Option<f64>,
    /// `reported`, `priced`, `free`, `subscription` or `unknown`; `None` for a
    /// provider that stated an amount without a basis (counted as `reported`).
    pub basis: Option<String>,
    /// Input + output tokens of the attempt, when the provider reported them.
    pub tokens: Option<u64>,
}

impl AttemptCost {
    /// The amount, when it is real money (basis `reported` / `priced`, or none).
    pub fn marginal_usd(&self) -> Option<f64> {
        crate::chat::cost::counts_toward_budget(self.basis.as_deref())
            .then_some(self.usd)
            .flatten()
    }
}

/// The limits of a run: a USD ceiling on the marginal counter and, for a
/// provider without a price, a ceiling on the tokens of every attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunBudget {
    /// USD limit on the marginal counter; `<= 0` disables it.
    pub max_cost_usd: f64,
    /// Token limit on the run; `None` disables it.
    pub max_tokens: Option<u64>,
}

/// The limit a run went past.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetBreach {
    /// The marginal counter went past the USD limit.
    Usd { cumulated_usd: f64, limit_usd: f64 },
    /// The tokens of the run went past the token limit.
    Tokens {
        cumulated_tokens: u64,
        limit_tokens: u64,
    },
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
            notional_usd: 0.0,
            tokens_used: 0,
            unknown_cost_attempts: 0,
            triggered_by,
            project_id: None,
            lifecycle_run_id: None,
            cwd: None,
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

    /// Add a REAL spend (basis `reported` or `priced`) to the marginal counter.
    pub fn add_cost(&mut self, cost_usd: f64) {
        self.cost_usd += cost_usd;
    }

    /// The marginal counter: what the run really spent.
    pub fn marginal_usd(&self) -> f64 {
        self.cost_usd
    }

    /// Account the cost of one attempt on the counter its basis names (A21):
    /// `reported` / `priced` (or no basis, the pre-A21 shape) on the marginal
    /// counter, `subscription` / `free` on the notional one, `unknown` (or a
    /// basis this version does not know) counted apart. Tokens always add up.
    pub fn add_attempt_cost(&mut self, cost: &AttemptCost) {
        match cost.basis.as_deref() {
            None | Some("reported") | Some("priced") => {
                if let Some(usd) = cost.usd {
                    self.cost_usd += usd;
                }
            }
            Some("subscription") | Some("free") => {
                if let Some(usd) = cost.usd {
                    self.notional_usd += usd;
                }
            }
            Some(_) => self.unknown_cost_attempts += 1,
        }
        if let Some(tokens) = cost.tokens {
            self.tokens_used = self.tokens_used.saturating_add(tokens);
        }
    }

    /// Whether one of this run's executors runs `session_id` right now.
    pub fn runs_session(&self, session_id: Uuid) -> bool {
        self.active_agents
            .iter()
            .any(|a| a.session_id == Some(session_id))
    }

    /// Whether a delegation belongs to this run (B36): it was spawned by one of
    /// this run's executors — `owner_run` is the run whose executor was its
    /// parent when it started, `parent_session` the session that delegated. A
    /// manual delegation, or one of an earlier run, is not this run's spend,
    /// whatever its plan.
    pub fn owns_delegation(&self, owner_run: Option<Uuid>, parent_session: Option<Uuid>) -> bool {
        owner_run == Some(self.run_id) || parent_session.is_some_and(|p| self.runs_session(p))
    }

    /// Check if the USD budget has been exceeded (marginal counter only).
    pub fn is_budget_exceeded(&self, max_cost_usd: f64) -> bool {
        self.cost_usd > max_cost_usd
    }

    /// The first limit of `budget` the run has gone past, `None` while it is
    /// within both. Only the marginal counter is held against the USD limit;
    /// the tokens of every attempt, whatever their basis, against the token one.
    pub fn budget_breach(&self, budget: &RunBudget) -> Option<BudgetBreach> {
        if budget.max_cost_usd > 0.0 && self.cost_usd > budget.max_cost_usd {
            return Some(BudgetBreach::Usd {
                cumulated_usd: self.cost_usd,
                limit_usd: budget.max_cost_usd,
            });
        }
        match budget.max_tokens {
            Some(limit) if self.tokens_used > limit => Some(BudgetBreach::Tokens {
                cumulated_tokens: self.tokens_used,
                limit_tokens: limit,
            }),
            _ => None,
        }
    }

    /// Number of completed tasks.
    pub fn tasks_completed(&self) -> usize {
        self.completed_tasks.len()
    }

    /// Record a task as completed. Also removes it from active_agents.
    ///
    /// Idempotent, and a task is never both completed and failed: the latest
    /// outcome wins (e.g. a successful retry of a failed task moves it here).
    pub fn mark_task_completed(&mut self, task_id: Uuid) {
        self.failed_tasks.retain(|t| *t != task_id);
        if !self.completed_tasks.contains(&task_id) {
            self.completed_tasks.push(task_id);
        }
        self.remove_agent(&task_id);
        self.sync_current_task_compat();
    }

    /// Record a task as failed. Also removes it from active_agents.
    ///
    /// Idempotent, and removes the task from `completed_tasks` (post-wave
    /// verification can fail a task that was already marked completed).
    pub fn mark_task_failed(&mut self, task_id: Uuid) {
        self.completed_tasks.retain(|t| *t != task_id);
        if !self.failed_tasks.contains(&task_id) {
            self.failed_tasks.push(task_id);
        }
        self.remove_agent(&task_id);
        self.sync_current_task_compat();
    }

    /// Drop a task from `completed_tasks` without recording it as failed.
    ///
    /// Used when a task's agent run succeeded but its verification failed and a
    /// retry is coming: until that retry lands, the task is neither completed
    /// nor failed, so counting it as completed would overstate the run.
    pub fn unmark_task_completed(&mut self, task_id: Uuid) {
        self.completed_tasks.retain(|t| *t != task_id);
        self.sync_current_task_compat();
    }

    // ========================================================================
    // Active agent management
    // ========================================================================

    /// Add an active agent tracking a running task.
    ///
    /// Defensively removes any existing agent with the same task_id first
    /// to prevent phantom duplicates (e.g., from retries).
    pub fn add_agent(&mut self, agent: ActiveAgent) {
        self.active_agents.retain(|a| a.task_id != agent.task_id);
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
    use crate::neo4j::traits::*;

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

    fn attempt(usd: Option<f64>, basis: Option<&str>, tokens: Option<u64>) -> AttemptCost {
        AttemptCost {
            usd,
            basis: basis.map(str::to_string),
            tokens,
        }
    }

    /// A21: two counters. A reported or priced cost is real money (marginal);
    /// a subscription or free one is notional; an unknown one has no amount and
    /// is counted apart, never as 0. Tokens add up whatever the basis.
    #[test]
    fn an_attempt_cost_lands_on_the_counter_its_basis_names() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        state.add_attempt_cost(&attempt(Some(1.5), Some("reported"), Some(100)));
        state.add_attempt_cost(&attempt(Some(0.5), Some("priced"), Some(50)));
        state.add_attempt_cost(&attempt(Some(9.0), Some("subscription"), Some(10)));
        state.add_attempt_cost(&attempt(Some(0.0), Some("free"), None));
        state.add_attempt_cost(&attempt(None, Some("unknown"), Some(40)));
        // A fictitious amount on a free endpoint never reaches the marginal counter.
        state.add_attempt_cost(&attempt(Some(123.0), Some("free"), None));
        // The pre-A21 shape: an amount without a basis was a reported cost.
        state.add_attempt_cost(&attempt(Some(0.25), None, None));

        assert!((state.marginal_usd() - 2.25).abs() < 1e-9, "{state:?}");
        assert!((state.notional_usd - 132.0).abs() < 1e-9, "{state:?}");
        assert_eq!(state.unknown_cost_attempts, 1);
        assert_eq!(state.tokens_used, 200);
    }

    #[test]
    fn an_unknown_cost_has_no_marginal_amount() {
        assert_eq!(
            attempt(Some(2.0), Some("unknown"), None).marginal_usd(),
            None
        );
        assert_eq!(attempt(None, Some("reported"), None).marginal_usd(), None);
        assert_eq!(attempt(Some(2.0), Some("free"), None).marginal_usd(), None);
        assert_eq!(
            attempt(Some(2.0), Some("priced"), None).marginal_usd(),
            Some(2.0)
        );
        assert_eq!(attempt(Some(2.0), None, None).marginal_usd(), Some(2.0));
    }

    /// Only the marginal counter cuts on USD; a run on a provider without a
    /// price is cut on its tokens, and only when a token limit is set.
    #[test]
    fn only_the_marginal_counter_breaches_the_usd_budget_and_tokens_cut_an_unpriced_run() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let usd_only = RunBudget {
            max_cost_usd: 1.0,
            max_tokens: None,
        };
        state.add_attempt_cost(&attempt(Some(50.0), Some("subscription"), Some(500)));
        assert_eq!(state.budget_breach(&usd_only), None, "notional never cuts");
        state.add_attempt_cost(&attempt(None, Some("unknown"), Some(600)));
        assert_eq!(
            state.budget_breach(&usd_only),
            None,
            "no token limit: tokens do not cut"
        );
        let with_tokens = RunBudget {
            max_cost_usd: 1.0,
            max_tokens: Some(1_000),
        };
        assert_eq!(
            state.budget_breach(&with_tokens),
            Some(BudgetBreach::Tokens {
                cumulated_tokens: 1_100,
                limit_tokens: 1_000,
            })
        );
        state.add_attempt_cost(&attempt(Some(1.5), Some("reported"), None));
        assert_eq!(
            state.budget_breach(&with_tokens),
            Some(BudgetBreach::Usd {
                cumulated_usd: 1.5,
                limit_usd: 1.0,
            }),
            "the USD breach is named first"
        );
        let unlimited = RunBudget {
            max_cost_usd: 0.0,
            max_tokens: None,
        };
        assert_eq!(state.budget_breach(&unlimited), None);
    }

    /// B36: the run's budget holds the SUM of its own attempts and the costs
    /// of the delegations its executor spawned. A parent that is under budget
    /// on its own exceeds it once a child's cost lands on the same counter.
    #[test]
    fn a_childs_cost_can_push_the_run_over_the_parents_budget() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let budget = RunBudget {
            max_cost_usd: 2.0,
            max_tokens: None,
        };
        // The parent's own attempt: still under budget.
        state.add_attempt_cost(&attempt(Some(1.2), Some("reported"), None));
        assert_eq!(state.budget_breach(&budget), None);
        // The delegated child's real spend is the run's spend too.
        state.add_attempt_cost(&attempt(Some(0.9), Some("reported"), None));
        assert_eq!(
            state.budget_breach(&budget),
            Some(BudgetBreach::Usd {
                cumulated_usd: 2.1,
                limit_usd: 2.0,
            }),
            "parent + children exceeds the budget"
        );
    }

    #[test]
    fn a_state_written_before_the_two_counters_reads_with_zeroes() {
        let before = serde_json::json!({
            "run_id": Uuid::new_v4(), "plan_id": Uuid::new_v4(), "total_tasks": 2,
            "current_wave": 1, "completed_tasks": [], "failed_tasks": [],
            "git_branch": "b", "started_at": "2026-01-01T00:00:00Z", "completed_at": null,
            "status": "running", "cost_usd": 1.25, "triggered_by": "manual", "project_id": null
        });
        let state: RunnerState = serde_json::from_value(before).unwrap();
        assert_eq!(state.marginal_usd(), 1.25);
        assert_eq!(state.notional_usd, 0.0);
        assert_eq!(state.tokens_used, 0);
        assert_eq!(state.unknown_cost_attempts, 0);
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

    #[test]
    fn test_add_agent_dedup_prevents_phantom() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_id = Uuid::new_v4();

        // First add
        state.add_agent(ActiveAgent::new(task_id, "Task A".to_string()));
        assert_eq!(state.active_agents.len(), 1);

        // Second add with same task_id (simulates retry) — should replace, not duplicate
        state.add_agent(ActiveAgent::new(task_id, "Task A (retry)".to_string()));
        assert_eq!(
            state.active_agents.len(),
            1,
            "add_agent should dedup by task_id — no phantom agents"
        );
        assert_eq!(state.active_agents[0].task_title, "Task A (retry)");
    }

    #[test]
    fn test_add_agent_different_tasks_not_deduped() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let task_a = Uuid::new_v4();
        let task_b = Uuid::new_v4();

        state.add_agent(ActiveAgent::new(task_a, "Task A".to_string()));
        state.add_agent(ActiveAgent::new(task_b, "Task B".to_string()));
        assert_eq!(
            state.active_agents.len(),
            2,
            "Different task_ids should not be deduped"
        );
    }

    #[test]
    fn test_mark_task_completed_is_idempotent() {
        // A task can reach mark_task_completed twice (e.g. step-coherence
        // auto-complete + post-wave watchdog, or a resumed run re-completing a
        // task). It must be counted once, otherwise progress_pct exceeds 100%.
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 1, TriggerSource::Manual);
        let task_id = Uuid::new_v4();
        state.mark_task_completed(task_id);
        state.mark_task_completed(task_id);
        assert_eq!(state.completed_tasks, vec![task_id]);
        assert!((state.progress_pct() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_completed_then_failed_task_is_only_failed() {
        // Post-wave verification can fail a task that was already marked
        // completed. The task must then leave `completed_tasks`, otherwise it is
        // counted both as completed and failed (tc + tf > total).
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 1, TriggerSource::Manual);
        let task_id = Uuid::new_v4();
        state.mark_task_completed(task_id);
        state.mark_task_failed(task_id);
        assert!(state.completed_tasks.is_empty());
        assert_eq!(state.failed_tasks, vec![task_id]);

        // And a successful retry of a failed task moves it back.
        state.mark_task_completed(task_id);
        assert!(state.failed_tasks.is_empty());
        assert_eq!(state.completed_tasks, vec![task_id]);
    }
}
