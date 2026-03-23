//! PlanRunner — Main execution loop for autonomous plan execution.
//!
//! Orchestrates: get_next_task → spawn agent → monitor → verify → update_status → next.
//! The Runner owns task status transitions — agents update step statuses in real-time via MCP.

use crate::chat::manager::ChatManager;
use crate::chat::types::{ChatEvent, ChatRequest};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::models::{PlanStatus, TaskStatus};
use crate::neo4j::traits::GraphStore;
use crate::orchestrator::context::ContextBuilder;
use crate::runner::enricher::TaskEnricher;
use crate::runner::git;
use crate::runner::guard::{AgentGuard, ChatManagerHintSender, GuardConfig, GuardVerdict};
use crate::runner::lifecycle;
use crate::runner::models::{
    ActiveAgent, ActiveAgentSnapshot, CwdValidation, PlanRunStatus, RunnerConfig, RunnerEvent,
    StepBreakdown, TaskExecutionReport, TaskResult, TaskRunStatus, TriggerSource,
};
use crate::runner::persona::{
    activate_skills_for_task, complexity_directive, profile_task, record_skill_feedback,
};
#[cfg(test)]
use crate::runner::prompt::{build_runner_constraints, RunnerPromptContext};
use crate::runner::state::RunnerState;
use crate::runner::vector::VectorCollector;
use crate::runner::verifier::{TaskVerifier, VerifyResult};

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// Pure helper functions (testable without PlanRunner)
// ============================================================================

/// Result of CWD validation — either the resolved cwd or an error.
#[derive(Debug, Clone, PartialEq)]
pub enum CwdResolution {
    /// CWD was empty/dot, replaced by root_path
    FallbackToRoot { resolved_cwd: String },
    /// CWD matches root_path (or no root_path to compare)
    Match { resolved_cwd: String },
    /// CWD differs from root_path — warn mode allows it
    Mismatch {
        resolved_cwd: String,
        root_path: String,
    },
    /// CWD differs from root_path — strict mode rejects it
    StrictMismatch { cwd: String, root_path: String },
    /// No root_path configured — use cwd as-is
    NoRootPath { resolved_cwd: String },
}

/// Validate and resolve the working directory against the project root_path.
///
/// Pure function — no side effects, fully testable.
/// The caller is responsible for emitting events/logs based on the result.
pub fn validate_cwd(
    cwd: &str,
    root_path: Option<&str>,
    validation: &CwdValidation,
) -> CwdResolution {
    let root_path = match root_path {
        Some(rp) if !rp.is_empty() => rp,
        _ => {
            return CwdResolution::NoRootPath {
                resolved_cwd: cwd.to_string(),
            }
        }
    };

    // Expand ~ in root_path
    let root_path_expanded = if root_path.starts_with('~') {
        root_path.replacen('~', &std::env::var("HOME").unwrap_or_default(), 1)
    } else {
        root_path.to_string()
    };

    // Fallback: empty/dot cwd → use root_path
    if cwd == "." || cwd.is_empty() {
        return CwdResolution::FallbackToRoot {
            resolved_cwd: root_path_expanded,
        };
    }

    // Compare canonicalized paths
    let cwd_canon = std::fs::canonicalize(cwd).unwrap_or_else(|_| std::path::PathBuf::from(cwd));
    let root_canon = std::fs::canonicalize(&root_path_expanded)
        .unwrap_or_else(|_| std::path::PathBuf::from(&root_path_expanded));

    if cwd_canon == root_canon {
        return CwdResolution::Match {
            resolved_cwd: cwd.to_string(),
        };
    }

    // Mismatch detected
    match validation {
        CwdValidation::Strict => CwdResolution::StrictMismatch {
            cwd: cwd.to_string(),
            root_path: root_path_expanded,
        },
        CwdValidation::Warn => CwdResolution::Mismatch {
            resolved_cwd: cwd.to_string(),
            root_path: root_path_expanded,
        },
    }
}

/// Check whether the step completion guard should trigger.
///
/// Returns `true` if the task has steps but none were completed (all skipped/pending),
/// meaning the agent likely didn't do real work.
pub fn should_step_guard_trigger(breakdown: &StepBreakdown) -> bool {
    breakdown.total > 0 && breakdown.completed == 0
}

/// Compute the final run status based on task outcomes.
///
/// Pure function — no side effects. Extracted from execute_wave for testability.
/// - If any task is Failed, Blocked, or Pending → CompletedWithErrors
/// - Otherwise → Completed
pub fn compute_final_status(tasks: &[crate::neo4j::models::TaskNode]) -> PlanRunStatus {
    use crate::neo4j::models::TaskStatus;
    let has_failed = tasks.iter().any(|t| t.status == TaskStatus::Failed);
    let has_blocked = tasks.iter().any(|t| t.status == TaskStatus::Blocked);
    let has_pending = tasks.iter().any(|t| t.status == TaskStatus::Pending);

    if has_failed || has_blocked || has_pending {
        PlanRunStatus::CompletedWithErrors
    } else {
        PlanRunStatus::Completed
    }
}

/// Validate that at least one affected file exists on disk before spawning an agent.
///
/// Returns `(valid, missing)` where:
/// - `valid` = true if at least one file exists (or if affected_files is empty — no constraint)
/// - `missing` = list of files that don't exist on disk
///
/// Pure function — no side effects.
pub fn validate_affected_files(affected_files: &[String], cwd: &str) -> (bool, Vec<String>) {
    if affected_files.is_empty() {
        return (true, vec![]);
    }

    let cwd_path = std::path::Path::new(cwd);
    let mut missing = Vec::new();
    let mut at_least_one_exists = false;

    for file in affected_files {
        let full_path = if std::path::Path::new(file).is_absolute() {
            std::path::PathBuf::from(file)
        } else {
            cwd_path.join(file)
        };

        if full_path.exists() {
            at_least_one_exists = true;
        } else {
            missing.push(file.clone());
        }
    }

    (at_least_one_exists, missing)
}

// ============================================================================
// Global state — LazyLock pattern (no fields on OrchestratorState)
// ============================================================================

/// Global runner state — persisted in Neo4j for crash recovery.
/// Only one run can be active at a time (enforced by start()).
pub static RUNNER_STATE: LazyLock<Arc<RwLock<Option<RunnerState>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(None)));

/// Cancel flag — set to true to request graceful cancellation.
/// Checked between each task in the execution loop.
pub static RUNNER_CANCEL: LazyLock<Arc<AtomicBool>> =
    LazyLock::new(|| Arc::new(AtomicBool::new(false)));

/// Budget override — allows updating max_cost_usd during a running execution.
/// Stores f64 bits in an AtomicU64 (0 = no override, use config default).
/// Set via PATCH /api/plans/{plan_id}/run/budget.
pub static RUNNER_BUDGET: LazyLock<Arc<std::sync::atomic::AtomicU64>> =
    LazyLock::new(|| Arc::new(std::sync::atomic::AtomicU64::new(0)));

/// Vector collector — accumulates drift/knowledge metrics during execution.
/// Reset at the start of each run, finalized at the end.
static VECTOR_COLLECTOR: LazyLock<Arc<RwLock<VectorCollector>>> =
    LazyLock::new(|| Arc::new(RwLock::new(VectorCollector::new())));

// ============================================================================
// PlanRunner — the execution engine
// ============================================================================

/// Autonomous plan execution engine.
///
/// Spawns Claude Code agents for each task, monitors execution,
/// and manages the full lifecycle of a plan run.
pub struct PlanRunner {
    chat_manager: Arc<ChatManager>,
    graph: Arc<dyn GraphStore>,
    context_builder: Arc<ContextBuilder>,
    config: RunnerConfig,
    event_tx: broadcast::Sender<RunnerEvent>,
    /// Optional event emitter to bridge RunnerEvents into the CrudEvent system
    /// for WebSocket delivery. When set, every RunnerEvent is also emitted as a
    /// CrudEvent with entity_type=Runner so WS clients can filter on it.
    event_emitter: Option<Arc<dyn EventEmitter>>,
    /// User claims inherited from the caller who started the plan run.
    /// When set, runner agent sessions authenticate as this user instead of
    /// a synthetic service account — avoids 403 when email domain checks are
    /// configured in the auth middleware.
    user_claims: Option<crate::auth::jwt::Claims>,
}

/// Result of starting a plan run.
#[derive(Debug, Clone)]
pub struct StartResult {
    pub run_id: Uuid,
    pub plan_id: Uuid,
    pub total_waves: usize,
    pub total_tasks: usize,
}

/// Result of executing an entire wave of tasks in parallel.
#[derive(Debug)]
struct WaveResult {
    /// Task IDs that completed successfully
    pub tasks_completed: Vec<Uuid>,
    /// Task IDs that failed, with their error reason
    pub tasks_failed: Vec<(Uuid, String)>,
    /// Total cost in USD for the wave
    pub wave_cost_usd: f64,
    /// Whether the wave was aborted due to budget or cancellation
    pub aborted: bool,
}

/// Summary of a completed task, used for inter-task continuity in a PlanRun.
#[derive(Debug, Clone)]
struct TaskSummary {
    #[allow(dead_code)]
    task_id: Uuid,
    title: String,
    wave_number: usize,
    status: String, // "completed" or "failed"
    commits: Vec<String>,
    files_modified: Vec<String>,
}

/// Cumulative memory across waves in a PlanRun, providing continuity context
/// to subsequent tasks so they know what was accomplished by previous tasks.
#[derive(Debug, Clone, Default)]
struct RunMemory {
    summaries: Vec<TaskSummary>,
}

impl RunMemory {
    /// Render the accumulated memory as a markdown section for prompt injection.
    fn to_markdown(&self) -> String {
        if self.summaries.is_empty() {
            return String::new();
        }
        let mut md = String::from("\n## Previous Tasks (Continuity Context)\n\n");
        md.push_str("The following tasks have already been completed in this plan run. ");
        md.push_str("Build on their work — do NOT redo what is already done.\n\n");
        for s in &self.summaries {
            md.push_str(&format!(
                "### Wave {} — {} ({})\n",
                s.wave_number, s.title, s.status
            ));
            if !s.commits.is_empty() {
                md.push_str("**Commits:**\n");
                for c in &s.commits {
                    md.push_str(&format!("- `{}`\n", c));
                }
            }
            if !s.files_modified.is_empty() {
                md.push_str("**Files modified:**\n");
                for f in &s.files_modified {
                    md.push_str(&format!("- `{}`\n", f));
                }
            }
            md.push('\n');
        }
        md
    }

    /// Add a task summary from a completed task.
    fn record_completed(
        &mut self,
        task_id: Uuid,
        title: String,
        wave_number: usize,
        commits: Vec<String>,
        files_modified: Vec<String>,
    ) {
        self.summaries.push(TaskSummary {
            task_id,
            title,
            wave_number,
            status: "completed".to_string(),
            commits,
            files_modified,
        });
    }

    /// Add a summary for a failed task.
    fn record_failed(&mut self, task_id: Uuid, title: String, wave_number: usize, reason: &str) {
        self.summaries.push(TaskSummary {
            task_id,
            title,
            wave_number,
            status: format!("failed: {}", &reason[..reason.len().min(200)]),
            commits: vec![],
            files_modified: vec![],
        });
    }
}

/// Result of execute_task — wraps TaskResult with the session_id for enricher.
#[derive(Debug)]
struct TaskExecutionResult {
    pub result: TaskResult,
    pub session_id: Option<Uuid>,
    /// Skill IDs activated during this task (for post-task feedback).
    pub activated_skill_ids: Vec<Uuid>,
    /// Persona IDs activated during this task (for post-task feedback).
    pub persona_ids: Vec<Uuid>,
    /// AgentExecution UUID created for this task (for USED_SKILL relations).
    pub agent_execution_id: Uuid,
    /// Persona profile string used for this agent.
    pub persona_profile: String,
    /// Structured execution report with tool usage metrics and confidence score.
    pub report: Option<TaskExecutionReport>,
}

impl TaskExecutionResult {
    fn session_id(&self) -> Option<Uuid> {
        self.session_id
    }
}

/// Snapshot of the current runner status (for the status endpoint).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunStatus {
    pub running: bool,
    pub run_id: Option<Uuid>,
    pub plan_id: Option<Uuid>,
    pub status: Option<PlanRunStatus>,
    pub current_wave: Option<usize>,
    /// Deprecated: use `active_agents` instead. Kept for backward compatibility.
    /// Returns the task_id of the first active agent when there is exactly one.
    pub current_task_id: Option<Uuid>,
    /// Deprecated: use `active_agents` instead.
    pub current_task_title: Option<String>,
    /// All currently active agents (may be multiple in parallel waves).
    pub active_agents: Vec<ActiveAgentSnapshot>,
    pub progress_pct: f64,
    pub tasks_completed: usize,
    pub tasks_total: usize,
    pub elapsed_secs: f64,
    pub cost_usd: f64,
    /// Current effective budget limit (from override or config).
    pub max_cost_usd: f64,
}

impl PlanRunner {
    /// Get the effective max_cost_usd, checking for a runtime override.
    pub fn effective_budget(&self) -> f64 {
        let bits = RUNNER_BUDGET.load(std::sync::atomic::Ordering::Relaxed);
        if bits == 0 {
            self.config.max_cost_usd
        } else {
            f64::from_bits(bits)
        }
    }

    /// Update the budget override for the currently running execution.
    /// The new value takes effect on the next budget check in the execution loop.
    pub async fn update_budget(run_id: Uuid, new_budget: f64) -> Result<()> {
        if new_budget <= 0.0 {
            return Err(anyhow!("Budget must be positive"));
        }
        let global = RUNNER_STATE.read().await;
        match &*global {
            Some(state) if state.run_id == run_id && state.status == PlanRunStatus::Running => {
                let bits = new_budget.to_bits();
                RUNNER_BUDGET.store(bits, std::sync::atomic::Ordering::Relaxed);
                info!("Budget updated to ${:.2} for run {}", new_budget, run_id);
                Ok(())
            }
            Some(state) => Err(anyhow!(
                "Run {} is not running (status: {})",
                run_id,
                state.status
            )),
            None => Err(anyhow!("No active run")),
        }
    }

    /// Get the current effective budget (static, no &self needed).
    pub fn current_budget() -> f64 {
        let bits = RUNNER_BUDGET.load(std::sync::atomic::Ordering::Relaxed);
        if bits == 0 {
            0.0
        } else {
            f64::from_bits(bits)
        }
    }

    /// Create a new PlanRunner.
    pub fn new(
        chat_manager: Arc<ChatManager>,
        graph: Arc<dyn GraphStore>,
        context_builder: Arc<ContextBuilder>,
        config: RunnerConfig,
        event_tx: broadcast::Sender<RunnerEvent>,
    ) -> Self {
        Self {
            chat_manager,
            graph,
            context_builder,
            config,
            event_tx,
            event_emitter: None,
            user_claims: None,
        }
    }

    /// Set user claims inherited from the caller who started the run.
    /// Runner agent sessions will authenticate as this user instead of
    /// using a synthetic service account.
    pub fn with_user_claims(mut self, claims: crate::auth::jwt::Claims) -> Self {
        self.user_claims = Some(claims);
        self
    }

    /// Set the event emitter for WebSocket bridging.
    ///
    /// When set, every RunnerEvent is also emitted as a CrudEvent with
    /// `entity_type: Runner` so WebSocket clients can subscribe with
    /// `?entity_types=runner`.
    pub fn with_event_emitter(mut self, emitter: Arc<dyn EventEmitter>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Update a task's status and emit a CrudEvent so the frontend gets real-time updates.
    async fn update_task_status_with_event(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        // Serialize status before move
        let status_json = serde_json::to_value(&status).unwrap_or_default();
        self.graph.update_task_status(task_id, status).await?;

        // Emit CrudEvent for WebSocket real-time updates
        if let Some(ref emitter) = self.event_emitter {
            emitter.emit(CrudEvent {
                entity_type: EntityType::Task,
                action: CrudAction::Updated,
                entity_id: task_id.to_string(),
                related: None,
                payload: serde_json::json!({ "status": status_json }),
                timestamp: chrono::Utc::now().to_rfc3339(),
                project_id: None,
            });
        }
        Ok(())
    }

    /// Emit a RunnerEvent on both the broadcast channel and the CrudEvent bus.
    fn emit_event(&self, event: RunnerEvent) {
        // 1. Broadcast on the dedicated RunnerEvent channel
        let _ = self.event_tx.send(event.clone());

        // 2. Bridge to CrudEvent for WebSocket delivery
        if let Some(ref emitter) = self.event_emitter {
            let (entity_id, action) = match &event {
                RunnerEvent::PlanStarted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Created)
                }
                RunnerEvent::PlanCompleted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::TaskStarted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::TaskCompleted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::TaskFailed { run_id, .. } => (run_id.to_string(), CrudAction::Updated),
                RunnerEvent::TaskTimeout { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::WaveStarted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::WaveCompleted { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::BudgetExceeded { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::RunnerError { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::WorktreeRecovery { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::TaskCompletedWithoutSteps { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::CwdMismatch { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::TaskSpawningTimeout { run_id, .. } => {
                    (run_id.to_string(), CrudAction::Updated)
                }
                RunnerEvent::LifecycleTransition { run_id, .. } => {
                    (run_id.to_string(), CrudAction::StatusChanged)
                }
            };

            let payload = serde_json::to_value(&event).unwrap_or_default();
            emitter.emit(CrudEvent {
                entity_type: EntityType::Runner,
                action,
                entity_id,
                related: None,
                payload,
                timestamp: chrono::Utc::now().to_rfc3339(),
                project_id: None,
            });
        }
    }

    /// Recover interrupted runs from Neo4j at server boot.
    ///
    /// Scans for PlanRun nodes with status=running, restores global state,
    /// re-computes waves, and resumes execution from the last incomplete task.
    /// Already-completed tasks are skipped by `execute_plan`.
    pub async fn recover_interrupted_runs(self: Arc<Self>, cwd: String) -> Result<usize> {
        let active_runs = self.graph.list_active_plan_runs().await?;
        if active_runs.is_empty() {
            info!("No interrupted runs to recover");
            return Ok(0);
        }

        info!("Found {} interrupted run(s) to recover", active_runs.len());

        let mut recovered = 0;
        for saved_state in active_runs {
            let run_id = saved_state.run_id;
            let plan_id = saved_state.plan_id;
            info!(
                "Recovering run {} for plan {} (wave {}, {}/{} tasks done)",
                run_id,
                plan_id,
                saved_state.current_wave,
                saved_state.completed_tasks.len(),
                saved_state.total_tasks
            );

            // Verify the plan still exists
            let plan = self.graph.get_plan(plan_id).await?;
            if plan.is_none() {
                warn!(
                    "Plan {} no longer exists, marking run {} as failed",
                    plan_id, run_id
                );
                let mut failed_state = saved_state.clone();
                failed_state.finalize(PlanRunStatus::Failed);
                self.graph.update_plan_run(&failed_state).await?;
                // Plan no longer exists, no status to update
                continue;
            }

            // Re-compute waves (task statuses may have changed)
            let waves_result = match self.graph.compute_waves(plan_id).await {
                Ok(w) => w,
                Err(e) => {
                    error!("Failed to compute waves for recovery: {}", e);
                    continue;
                }
            };

            if waves_result.waves.is_empty() {
                info!("Plan {} has no remaining tasks, marking complete", plan_id);
                let mut done_state = saved_state.clone();
                done_state.finalize(PlanRunStatus::Completed);
                self.graph.update_plan_run(&done_state).await?;
                if let Err(e) = self
                    .graph
                    .update_plan_status(plan_id, PlanStatus::Completed)
                    .await
                {
                    warn!(
                        "Failed to set plan {} status to Completed on recovery: {}",
                        plan_id, e
                    );
                }
                continue;
            }

            // Restore global state
            {
                let mut global = RUNNER_STATE.write().await;
                *global = Some(saved_state.clone());
            }
            RUNNER_CANCEL.store(false, Ordering::SeqCst);

            // Emit recovery event
            self.emit_event(RunnerEvent::PlanStarted {
                run_id,
                plan_id,
                plan_title: format!("[Recovery] run {}", run_id),
                total_tasks: saved_state.total_tasks,
                total_waves: waves_result.waves.len(),
                prediction: None,
            });

            // Spawn execution in background
            let runner = self.clone();
            let waves = waves_result.waves;
            let cwd_clone = cwd.clone();
            tokio::spawn(async move {
                if let Err(e) = runner
                    .execute_plan(run_id, plan_id, waves, cwd_clone, None)
                    .await
                {
                    error!("Recovery execution failed for run {}: {}", run_id, e);
                    runner.emit_event(RunnerEvent::RunnerError {
                        run_id,
                        message: format!("Recovery failed: {}", e),
                    });
                    let mut global = RUNNER_STATE.write().await;
                    if let Some(ref mut s) = *global {
                        s.finalize(PlanRunStatus::Failed);
                        let _ = runner.graph.update_plan_run(s).await;
                        if let Err(e) = runner
                            .graph
                            .update_plan_status(s.plan_id, PlanStatus::Cancelled)
                            .await
                        {
                            warn!("Failed to set plan {} status to Cancelled after recovery failure: {}", s.plan_id, e);
                        }
                    }
                }
            });

            recovered += 1;
        }

        Ok(recovered)
    }

    /// Start executing a plan. Returns immediately with the run_id.
    ///
    /// The execution loop runs in a background tokio task.
    /// Returns 409-equivalent error if a run is already active for this plan.
    pub async fn start(
        self: Arc<Self>,
        plan_id: Uuid,
        trigger: TriggerSource,
        cwd: String,
        project_slug: Option<String>,
    ) -> Result<StartResult> {
        // 1. Check no active run exists for this plan
        let active_runs = self.graph.list_active_plan_runs().await?;
        if let Some(existing) = active_runs.iter().find(|r| r.plan_id == plan_id) {
            return Err(anyhow!(
                "Plan {} already has an active run: {} (status: {}). Cancel it first.",
                plan_id,
                existing.run_id,
                existing.status
            ));
        }

        // 2. Compute waves
        let waves_result = self.graph.compute_waves(plan_id).await?;
        if waves_result.waves.is_empty() {
            return Err(anyhow!("Plan {} has no tasks to execute", plan_id));
        }

        let total_waves = waves_result.waves.len();
        let total_tasks: usize = waves_result.waves.iter().map(|w| w.task_count).sum();

        // 3. Create RunnerState and persist
        let run_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, total_tasks, trigger.clone());

        self.graph.create_plan_run(&state).await?;

        // 4. Store in global state
        {
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state.clone());
        }

        // Reset cancel flag, budget override, and vector collector
        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        RUNNER_BUDGET.store(
            self.config.max_cost_usd.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        {
            let mut collector = VECTOR_COLLECTOR.write().await;
            *collector = VectorCollector::new();
        }

        // Transition plan status to InProgress (idempotent — warn if already in_progress)
        if let Err(e) = self
            .graph
            .update_plan_status(plan_id, PlanStatus::InProgress)
            .await
        {
            warn!(
                "Failed to set plan {} status to in_progress (may already be in_progress): {}",
                plan_id, e
            );
        } else {
            info!("Plan {} status set to in_progress", plan_id);
        }

        // Route to lifecycle protocol (if available). Non-fatal — fallback on None.
        match lifecycle::route_lifecycle_protocol(&self.graph, plan_id, total_tasks).await {
            Ok(Some(route_result)) => {
                info!(
                    "Plan {} wrapped by lifecycle protocol {} (run: {}, affinity: {:.2})",
                    plan_id,
                    route_result.protocol_name,
                    route_result.run_id,
                    route_result.affinity_score
                );
                let mut global = RUNNER_STATE.write().await;
                if let Some(ref mut s) = *global {
                    s.lifecycle_run_id = Some(route_result.run_id);
                }
            }
            Ok(None) => {
                debug!(
                    "No lifecycle protocol for plan {} — using default runner flow",
                    plan_id
                );
            }
            Err(e) => {
                warn!(
                    "Lifecycle protocol routing failed for plan {}: {}. Continuing without.",
                    plan_id, e
                );
            }
        }

        let result = StartResult {
            run_id,
            plan_id,
            total_waves,
            total_tasks,
        };

        // 5. Compute prediction from historical runs (with per-agent data when available)
        let prediction = match self.graph.list_plan_runs(plan_id, 50).await {
            Ok(runs) if !runs.is_empty() => {
                let vectors: Vec<_> = runs
                    .iter()
                    .rev()
                    .map(crate::runner::vector::ExecutionVector::from_runner_state)
                    .collect();

                // Try to load per-agent vectors for finer-grained prediction
                let mut agent_vectors = Vec::new();
                for run in runs.iter().rev() {
                    match self.graph.get_agent_executions_for_run(run.run_id).await {
                        Ok(aes) => {
                            let avs: Vec<_> = aes
                                .iter()
                                .filter_map(|ae| {
                                    ae.vector_json.as_ref().and_then(|json| {
                                        serde_json::from_str::<
                                            crate::runner::vector::AgentExecutionVector,
                                        >(json)
                                        .ok()
                                    })
                                })
                                .collect();
                            agent_vectors.push(avs);
                        }
                        Err(_) => agent_vectors.push(Vec::new()),
                    }
                }

                Some(crate::runner::vector::predict_run_per_agent(
                    &vectors,
                    &agent_vectors,
                ))
            }
            _ => None,
        };

        // Emit PlanStarted event with prediction
        self.emit_event(RunnerEvent::PlanStarted {
            run_id,
            plan_id,
            plan_title: String::new(), // Will be enriched by caller
            total_tasks,
            total_waves,
            prediction,
        });

        // 6. Spawn the execution loop in background
        let runner = self.clone();
        let waves = waves_result.waves;
        tokio::spawn(async move {
            if let Err(e) = runner
                .execute_plan(run_id, plan_id, waves, cwd, project_slug)
                .await
            {
                error!("PlanRunner execution failed: {}", e);
                runner.emit_event(RunnerEvent::RunnerError {
                    run_id,
                    message: e.to_string(),
                });
                // Update state to failed
                let mut global = RUNNER_STATE.write().await;
                if let Some(ref mut s) = *global {
                    s.finalize(PlanRunStatus::Failed);
                    let _ = runner.graph.update_plan_run(s).await;
                    // Also transition plan status to Cancelled (failed run)
                    if let Err(e) = runner
                        .graph
                        .update_plan_status(s.plan_id, PlanStatus::Cancelled)
                        .await
                    {
                        warn!(
                            "Failed to set plan {} status to Cancelled after error: {}",
                            s.plan_id, e
                        );
                    }
                }
            }
        });

        Ok(result)
    }

    /// Cancel the currently running plan execution.
    ///
    /// When a `ChatManager` is provided, all active agent sessions are
    /// forcefully closed (interrupt + SIGKILL) so that Claude Code
    /// subprocesses don't keep running after the cancel is requested.
    pub async fn cancel(run_id: Uuid, chat_manager: Option<Arc<ChatManager>>) -> Result<()> {
        let global = RUNNER_STATE.read().await;
        match &*global {
            Some(state) if state.run_id == run_id && state.status == PlanRunStatus::Running => {
                RUNNER_CANCEL.store(true, Ordering::SeqCst);
                info!("Runner cancel requested for run {}", run_id);

                // Kill all active agent sessions so subprocesses stop immediately
                if let Some(cm) = chat_manager {
                    let session_ids: Vec<Uuid> = state
                        .active_agents
                        .iter()
                        .filter(|a| {
                            matches!(a.status, TaskRunStatus::Running | TaskRunStatus::Spawning)
                        })
                        .filter_map(|a| a.session_id)
                        .collect();
                    drop(global); // release lock before async close calls
                    Self::close_agent_sessions(&cm, &session_ids).await;
                }

                Ok(())
            }
            Some(state) => Err(anyhow!(
                "Run {} is not running (status: {})",
                run_id,
                state.status
            )),
            None => Err(anyhow!("No active run")),
        }
    }

    /// Force-cancel: immediately clears the global runner state and persists
    /// the run as cancelled in Neo4j. Use when graceful cancel is stuck
    /// (e.g. agents blocked in spawning state that never respond to the
    /// cancel flag).
    ///
    /// When a `ChatManager` is provided, all active agent sessions are
    /// forcefully closed so that Claude Code subprocesses stop immediately.
    pub async fn force_cancel(
        run_id: Uuid,
        graph: Arc<dyn GraphStore>,
        chat_manager: Option<Arc<ChatManager>>,
    ) -> Result<()> {
        // Set cancel flag so any in-flight code that checks it will stop
        RUNNER_CANCEL.store(true, Ordering::SeqCst);
        // Clear budget override
        RUNNER_BUDGET.store(0, std::sync::atomic::Ordering::Relaxed);

        let mut global = RUNNER_STATE.write().await;
        match &mut *global {
            Some(state) if state.run_id == run_id => {
                let plan_id = state.plan_id;

                // Collect active session IDs before finalizing
                let session_ids: Vec<Uuid> = state
                    .active_agents
                    .iter()
                    .filter_map(|a| a.session_id)
                    .collect();

                state.finalize(PlanRunStatus::Cancelled);
                if let Err(e) = graph.update_plan_run(state).await {
                    error!("Failed to persist force-cancelled run to Neo4j: {}", e);
                }
                // Transition plan status to Cancelled
                if let Err(e) = graph
                    .update_plan_status(plan_id, PlanStatus::Cancelled)
                    .await
                {
                    warn!(
                        "Failed to set plan {} status to Cancelled on force-cancel: {}",
                        plan_id, e
                    );
                }
                info!("Runner force-cancelled run {}", run_id);
                // Clear the global state so a new run can start
                *global = None;
                drop(global); // release lock before async close calls

                // Kill all active agent sessions
                if let Some(cm) = chat_manager {
                    Self::close_agent_sessions(&cm, &session_ids).await;
                }

                Ok(())
            }
            Some(state) => Err(anyhow!(
                "Run {} does not match active run {}",
                run_id,
                state.run_id
            )),
            None => Err(anyhow!("No active run to force-cancel")),
        }
    }

    /// Close all active agent sessions, killing their Claude Code subprocesses.
    /// Errors are logged but not propagated (best-effort cleanup).
    async fn close_agent_sessions(chat_manager: &ChatManager, session_ids: &[Uuid]) {
        for sid in session_ids {
            let sid_str = sid.to_string();
            info!("Closing agent session {} on cancel", sid_str);
            if let Err(e) = chat_manager.close_session(&sid_str).await {
                warn!("Failed to close agent session {} on cancel: {}", sid_str, e);
            }
        }
    }

    /// Get the current run status snapshot.
    pub async fn status() -> RunStatus {
        let global = RUNNER_STATE.read().await;
        match &*global {
            Some(state) => {
                let active_agents: Vec<ActiveAgentSnapshot> =
                    state.active_agents.iter().map(|a| a.snapshot()).collect();
                RunStatus {
                    running: state.status == PlanRunStatus::Running,
                    run_id: Some(state.run_id),
                    plan_id: Some(state.plan_id),
                    status: Some(state.status),
                    current_wave: Some(state.current_wave),
                    current_task_id: state.current_task_id,
                    current_task_title: state.current_task_title.clone(),
                    active_agents,
                    progress_pct: state.progress_pct(),
                    tasks_completed: state.tasks_completed(),
                    tasks_total: state.total_tasks,
                    elapsed_secs: state.elapsed_secs(),
                    cost_usd: state.cost_usd,
                    max_cost_usd: Self::current_budget(),
                }
            }
            None => RunStatus {
                running: false,
                run_id: None,
                plan_id: None,
                status: None,
                current_wave: None,
                current_task_id: None,
                current_task_title: None,
                active_agents: Vec::new(),
                progress_pct: 0.0,
                tasks_completed: 0,
                tasks_total: 0,
                elapsed_secs: 0.0,
                cost_usd: 0.0,
                max_cost_usd: 0.0,
            },
        }
    }

    // ========================================================================
    // Internal execution loop
    // ========================================================================

    /// Main execution loop — iterates over waves and tasks.
    async fn execute_plan(
        &self,
        run_id: Uuid,
        plan_id: Uuid,
        waves: Vec<crate::neo4j::plan::Wave>,
        cwd: String,
        project_slug: Option<String>,
    ) -> Result<()> {
        info!(
            "Starting plan execution: run={}, plan={}, waves={}",
            run_id,
            plan_id,
            waves.len()
        );

        // Resolve project_id and root_path from slug (for enricher + cwd validation)
        let (project_id, project_root_path) = if let Some(ref slug) = project_slug {
            match self.graph.get_project_by_slug(slug).await {
                Ok(Some(project)) => {
                    let rp = if project.root_path.is_empty() {
                        None
                    } else {
                        Some(project.root_path.clone())
                    };
                    (Some(project.id), rp)
                }
                Ok(None) => {
                    warn!(
                        "Project slug '{}' not found, enricher will run without project_id",
                        slug
                    );
                    (None, None)
                }
                Err(e) => {
                    warn!("Failed to resolve project slug '{}': {}, enricher will run without project_id", slug, e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // CWD validation: fallback to root_path if cwd is ".", validate mismatch
        let cwd = {
            let resolution = validate_cwd(
                &cwd,
                project_root_path.as_deref(),
                &self.config.cwd_validation,
            );
            match resolution {
                CwdResolution::FallbackToRoot { resolved_cwd } => {
                    info!(
                        "CWD not specified — using project root_path: {}",
                        resolved_cwd
                    );
                    resolved_cwd
                }
                CwdResolution::Match { resolved_cwd }
                | CwdResolution::NoRootPath { resolved_cwd } => resolved_cwd,
                CwdResolution::Mismatch {
                    resolved_cwd,
                    root_path,
                } => {
                    warn!(
                        "CWD mismatch: cwd='{}' != root_path='{}'. Continuing with provided cwd.",
                        resolved_cwd, root_path
                    );
                    self.emit_event(RunnerEvent::CwdMismatch {
                        run_id,
                        cwd: resolved_cwd.clone(),
                        root_path,
                    });
                    resolved_cwd
                }
                CwdResolution::StrictMismatch { cwd, root_path } => {
                    return Err(anyhow!(
                        "CWD mismatch (strict mode): cwd='{}' != root_path='{}'. \
                         Aborting to prevent execution in wrong directory.",
                        cwd,
                        root_path
                    ));
                }
            }
        };

        // Create a dedicated git branch for this run
        let branch_name = self.create_git_branch(plan_id, run_id, &cwd).await;
        if let Some(ref branch) = branch_name {
            // Store branch name in runner state
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.git_branch = branch.clone();
            }
        }

        let mut run_memory = RunMemory::default();

        for (wave_idx, wave) in waves.iter().enumerate() {
            let wave_number = wave_idx + 1;

            // Skip waves where all tasks are already completed (resume optimization)
            let has_pending_tasks = wave.tasks.iter().any(|t| t.status != "completed");
            if !has_pending_tasks {
                info!(
                    "Skipping wave {} — all {} tasks already completed",
                    wave_number, wave.task_count
                );
                // Still track completed tasks in runner state
                {
                    let mut global = RUNNER_STATE.write().await;
                    if let Some(ref mut s) = *global {
                        for t in &wave.tasks {
                            if !s.completed_tasks.contains(&t.id) {
                                s.completed_tasks.push(t.id);
                            }
                        }
                        s.current_wave = wave_number;
                    }
                }
                continue;
            }

            // Check cancel flag
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                info!("Runner cancelled at wave {}", wave_number);
                self.finalize_run(run_id, PlanRunStatus::Cancelled, Some(&cwd))
                    .await?;
                return Ok(());
            }

            // Update current wave
            {
                let mut global = RUNNER_STATE.write().await;
                if let Some(ref mut s) = *global {
                    s.current_wave = wave_number;
                }
            }

            self.emit_event(RunnerEvent::WaveStarted {
                run_id,
                wave_number,
                task_count: wave.task_count,
            });

            // Capture HEAD SHA before the wave — used to verify agents produced commits
            let wave_base_sha = git::head_sha(&cwd).await.unwrap_or_default();

            // Execute all tasks in this wave in parallel via JoinSet
            let continuity_context = run_memory.to_markdown();
            let wave_result = self
                .execute_wave(
                    run_id,
                    plan_id,
                    wave,
                    &cwd,
                    project_slug.as_deref(),
                    project_id,
                    &continuity_context,
                )
                .await?;

            // If wave was aborted (budget/cancel), finalize accordingly
            if wave_result.aborted {
                // Check which reason caused the abort
                if RUNNER_CANCEL.load(Ordering::SeqCst) {
                    self.finalize_run(run_id, PlanRunStatus::Cancelled, Some(&cwd))
                        .await?;
                } else {
                    // Budget exceeded
                    let (cumulated, limit) = {
                        let global = RUNNER_STATE.read().await;
                        global
                            .as_ref()
                            .map(|s| (s.cost_usd, self.effective_budget()))
                            .unwrap_or((0.0, 0.0))
                    };
                    self.emit_event(RunnerEvent::BudgetExceeded {
                        run_id,
                        plan_id,
                        cumulated_cost_usd: cumulated,
                        limit_usd: limit,
                    });
                    self.finalize_run(run_id, PlanRunStatus::BudgetExceeded, Some(&cwd))
                        .await?;
                }
                return Ok(());
            }

            // Post-wave: collect commits from agent worktrees before verification.
            // Agents may have created worktrees for isolation — their commits need
            // to be cherry-picked back onto the run branch.
            {
                let run_branch = git::current_branch(&cwd).await.unwrap_or_default();
                if !run_branch.is_empty() {
                    match git::collect_worktree_commits(&cwd, &run_branch).await {
                        Ok(wt_result) => {
                            if !wt_result.recovered_commits.is_empty()
                                || !wt_result.conflicts.is_empty()
                            {
                                info!(
                                    "Wave {}: recovered {} commits from worktrees, {} conflicts, {} cleaned up",
                                    wave_number,
                                    wt_result.recovered_commits.len(),
                                    wt_result.conflicts.len(),
                                    wt_result.cleaned_up.len(),
                                );
                                self.emit_event(RunnerEvent::WorktreeRecovery {
                                    run_id,
                                    wave_number,
                                    commits_recovered: wt_result.recovered_commits.len(),
                                    conflicts: wt_result.conflicts.len(),
                                    worktrees_cleaned: wt_result.cleaned_up.len(),
                                });
                            }
                            for conflict in &wt_result.conflicts {
                                warn!(
                                    "Worktree conflict: {} on branch {} (commit {}): {}",
                                    conflict.worktree_path,
                                    conflict.branch,
                                    conflict.commit_sha,
                                    conflict.error
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Worktree collection failed (non-fatal): {}", e);
                        }
                    }
                }
            }

            // Post-wave verification: run TaskVerifier ONCE for all completed tasks.
            // Step completion is checked for every task, but build/git checks run
            // only once (on the last task) since they verify the whole project.
            // The base_commit check runs on the last task to detect ghost completions.
            if !wave_result.tasks_completed.is_empty() {
                for (idx, &task_id) in wave_result.tasks_completed.iter().enumerate() {
                    let is_last = idx == wave_result.tasks_completed.len() - 1;
                    // Use with_base_commit on the LAST task to verify the wave
                    // produced at least one commit (prevents ghost completions)
                    let wave_verifier = if is_last && !wave_base_sha.is_empty() {
                        TaskVerifier::with_base_commit(
                            self.graph.clone(),
                            is_last && self.config.build_check,
                            is_last && self.config.test_runner,
                            wave_base_sha.clone(),
                        )
                    } else {
                        TaskVerifier::new(
                            self.graph.clone(),
                            is_last && self.config.build_check,
                            is_last && self.config.test_runner,
                        )
                    };
                    let verify_result = wave_verifier.verify(task_id, &cwd).await;
                    if let VerifyResult::Fail { reasons } = verify_result {
                        let reason =
                            format!("Post-wave verification failed:\n- {}", reasons.join("\n- "));
                        warn!("Task {} verification failed: {}", task_id, reason);
                        self.on_task_failed(run_id, plan_id, task_id, &reason, 0.0, 0.0)
                            .await?;
                    }
                }
            }

            let (tc, tf) = {
                let global = RUNNER_STATE.read().await;
                global
                    .as_ref()
                    .map(|s| (s.completed_tasks.len(), s.failed_tasks.len()))
                    .unwrap_or((0, 0))
            };
            self.emit_event(RunnerEvent::WaveCompleted {
                run_id,
                wave_number,
                tasks_completed: tc,
                tasks_failed: tf,
            });

            // Accumulate RunMemory from completed/failed tasks in this wave
            for &task_id in &wave_result.tasks_completed {
                let title = wave
                    .tasks
                    .iter()
                    .find(|t| t.id == task_id)
                    .and_then(|t| t.title.clone())
                    .unwrap_or_else(|| format!("Task {}", task_id));
                // Collect commits for this task from the graph (best effort)
                let (commits, files) = match self.graph.get_task_commits(task_id).await {
                    Ok(commit_nodes) => {
                        let commit_msgs: Vec<String> = commit_nodes
                            .iter()
                            .map(|c| {
                                format!(
                                    "{}: {}",
                                    &c.hash[..c.hash.len().min(8)],
                                    c.message.lines().next().unwrap_or("")
                                )
                            })
                            .collect();
                        let mut all_files: Vec<String> = Vec::new();
                        for c in &commit_nodes {
                            if let Ok(files) = self.graph.get_commit_files(&c.hash).await {
                                all_files.extend(files.into_iter().map(|f| f.path));
                            }
                        }
                        all_files.sort();
                        all_files.dedup();
                        (commit_msgs, all_files)
                    }
                    Err(_) => (vec![], vec![]),
                };
                run_memory.record_completed(task_id, title, wave_number, commits, files);
            }
            for (task_id, reason) in &wave_result.tasks_failed {
                let title = wave
                    .tasks
                    .iter()
                    .find(|t| t.id == *task_id)
                    .and_then(|t| t.title.clone())
                    .unwrap_or_else(|| format!("Task {}", task_id));
                run_memory.record_failed(*task_id, title, wave_number, reason);
            }
        }

        // Compute final run status based on task outcomes
        let final_status = {
            let tasks = self.graph.get_plan_tasks(plan_id).await.unwrap_or_default();
            let status = compute_final_status(&tasks);
            if status == PlanRunStatus::CompletedWithErrors {
                info!(
                    "Run {} finishing with errors (some tasks failed/blocked/pending)",
                    run_id
                );
            }
            status
        };

        self.finalize_run(run_id, final_status, Some(&cwd)).await?;
        Ok(())
    }

    /// Execute all tasks in a wave in parallel using a JoinSet.
    ///
    /// Each task is spawned as a separate tokio task. Results are collected as
    /// they complete. Budget and cancellation checks happen after each result,
    /// with `join_set.abort_all()` used to stop remaining tasks if needed.
    #[allow(clippy::too_many_arguments)]
    async fn execute_wave(
        &self,
        run_id: Uuid,
        plan_id: Uuid,
        wave: &crate::neo4j::plan::Wave,
        cwd: &str,
        project_slug: Option<&str>,
        project_id: Option<Uuid>,
        continuity_context: &str,
    ) -> Result<WaveResult> {
        use tokio::task::JoinSet;

        let mut wave_result = WaveResult {
            tasks_completed: Vec::new(),
            tasks_failed: Vec::new(),
            wave_cost_usd: 0.0,
            aborted: false,
        };

        // Filter tasks: skip completed and blocked
        let eligible_tasks: Vec<_> = wave
            .tasks
            .iter()
            .filter(|t| {
                if t.status.eq_ignore_ascii_case("completed") {
                    return false;
                }
                if t.status.eq_ignore_ascii_case("blocked") {
                    warn!(
                        "Skipping blocked task {}: {}",
                        t.id,
                        t.title.as_deref().unwrap_or("untitled")
                    );
                    return false;
                }
                true
            })
            .collect();

        if eligible_tasks.is_empty() {
            return Ok(wave_result);
        }

        // Check cancel flag before spawning any agents in this wave
        if RUNNER_CANCEL.load(Ordering::SeqCst) {
            info!("Cancel detected at wave start — skipping wave");
            wave_result.aborted = true;
            return Ok(wave_result);
        }

        // Build a map of task_id -> start_time for enricher
        let mut task_start_times: std::collections::HashMap<Uuid, chrono::DateTime<chrono::Utc>> =
            std::collections::HashMap::new();

        let mut join_set: JoinSet<(Uuid, String, Result<TaskExecutionResult>)> = JoinSet::new();

        for wave_task in &eligible_tasks {
            let task_id = wave_task.id;
            let task_title = wave_task
                .title
                .clone()
                .unwrap_or_else(|| "untitled".to_string());
            task_start_times.insert(task_id, chrono::Utc::now());

            // Mark the first step of this task as in_progress for live feedback
            if let Ok(steps) = self.graph.get_task_steps(task_id).await {
                if let Some(first_step) = steps
                    .iter()
                    .find(|s| s.status == crate::neo4j::models::StepStatus::Pending)
                {
                    if let Err(e) = self
                        .graph
                        .update_step_status(
                            first_step.id,
                            crate::neo4j::models::StepStatus::InProgress,
                        )
                        .await
                    {
                        warn!(
                            "Failed to mark first step in_progress for task {}: {}",
                            task_id, e
                        );
                    } else {
                        debug!(
                            "Marked step {} as in_progress at task {} launch",
                            first_step.id, task_id
                        );
                    }
                }
            }

            // Pre-validate affected_files before spawning agent (T4)
            if !wave_task.affected_files.is_empty() {
                let (valid, missing) = validate_affected_files(&wave_task.affected_files, cwd);
                if !valid {
                    warn!(
                        "Task {} ({}) blocked: none of the affected files exist on disk: {:?}",
                        task_id, task_title, missing
                    );
                    self.update_task_status_with_event(task_id, TaskStatus::Blocked)
                        .await?;
                    wave_result.tasks_failed.push((task_id, task_title.clone()));
                    continue;
                }
                if !missing.is_empty() {
                    warn!(
                        "Task {} ({}): some affected files missing (non-fatal): {:?}",
                        task_id, task_title, missing
                    );
                }
            }

            let runner = self.clone();
            let cwd = cwd.to_string();
            let project_slug = project_slug.map(|s| s.to_string());
            let title_clone = task_title.clone();
            let continuity = continuity_context.to_string();

            join_set.spawn(async move {
                let result = runner
                    .execute_task(
                        run_id,
                        plan_id,
                        task_id,
                        &title_clone,
                        &cwd,
                        project_slug.as_deref(),
                        None, // no retry context on first attempt
                        &continuity,
                    )
                    .await;
                (task_id, title_clone, result)
            });
        }

        // Collect results as they complete
        while let Some(join_result) = join_set.join_next().await {
            let (task_id, task_title, task_result) = match join_result {
                Ok(tuple) => tuple,
                Err(join_error) => {
                    if join_error.is_cancelled() {
                        // Task was aborted via join_set.abort_all() — not a failure
                        info!("A wave task was cancelled via JoinSet abort");
                        continue;
                    }
                    // JoinError from panic — treat as unexpected error
                    error!("A wave task panicked: {}", join_error);
                    continue;
                }
            };

            // Extract session_id from the result (for enricher)
            let task_session_id = task_result.as_ref().ok().and_then(|r| r.session_id());
            let task_start_time = task_start_times
                .get(&task_id)
                .copied()
                .unwrap_or_else(chrono::Utc::now);

            // Extract activated skill IDs and agent execution ID before consuming the result
            let task_activated_skills = task_result
                .as_ref()
                .ok()
                .map(|r| r.activated_skill_ids.clone())
                .unwrap_or_default();

            let task_agent_execution_id = task_result.as_ref().ok().map(|r| r.agent_execution_id);

            let task_persona_ids = task_result
                .as_ref()
                .ok()
                .map(|r| r.persona_ids.clone())
                .unwrap_or_default();

            let task_persona = task_result
                .as_ref()
                .ok()
                .map(|r| r.persona_profile.clone())
                .unwrap_or_default();

            // Extract execution report before consuming the result
            let task_report = task_result.as_ref().ok().and_then(|r| r.report.clone());
            let task_report_json = task_report
                .as_ref()
                .and_then(|r| serde_json::to_string(r).ok());

            match task_result.map(|r| r.result) {
                Ok(TaskResult::Success {
                    duration_secs,
                    cost_usd,
                }) => {
                    self.on_task_completed(run_id, task_id, duration_secs, cost_usd, Some(cwd))
                        .await?;
                    wave_result.tasks_completed.push(task_id);
                    wave_result.wave_cost_usd += cost_usd;

                    // Fire-and-forget enrichment (commits, auto-notes, AFFECTS)
                    let enricher = TaskEnricher::new(self.graph.clone());
                    let enrich_plan_id = plan_id;
                    let enrich_project_id = project_id;
                    let enrich_session_id = task_session_id;
                    let enrich_cwd = cwd.to_string();
                    let enrich_start = task_start_time;
                    tokio::spawn(async move {
                        let result = enricher
                            .enrich(
                                task_id,
                                enrich_plan_id,
                                enrich_project_id,
                                enrich_session_id,
                                enrich_start,
                                &enrich_cwd,
                            )
                            .await;
                        info!(
                            "Enricher completed for task {}: {} commits, note={}, {} affects, {} discussed",
                            task_id,
                            result.commits_linked,
                            result.note_created,
                            result.affects_added,
                            result.discussed_added
                        );
                    });

                    // Fire-and-forget: finalize AgentExecution node (success)
                    if let Some(ae_id) = task_agent_execution_id {
                        let graph = self.graph.clone();
                        let persona = task_persona.clone();
                        let report_json = task_report_json.clone();
                        let report_ref = task_report.clone();
                        tokio::spawn(async move {
                            use crate::neo4j::agent_execution::{
                                AgentExecutionNode, AgentExecutionStatus,
                            };
                            let (tools_json, files, commits) = if let Some(ref r) = report_ref {
                                (
                                    serde_json::to_string(&r.tool_use_breakdown)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                    r.files_modified.clone(),
                                    r.commits.clone(),
                                )
                            } else {
                                ("{}".to_string(), vec![], vec![])
                            };
                            let ae = AgentExecutionNode {
                                id: ae_id,
                                run_id,
                                task_id,
                                session_id: task_session_id,
                                started_at: task_start_time,
                                completed_at: Some(chrono::Utc::now()),
                                cost_usd,
                                duration_secs,
                                status: AgentExecutionStatus::Completed,
                                tools_used: tools_json,
                                files_modified: files,
                                commits,
                                persona_profile: persona,
                                vector_json: None,
                                report_json,
                                execution_type: Default::default(),
                            };
                            if let Err(e) = graph.update_agent_execution(&ae).await {
                                warn!("Failed to update AgentExecution {}: {}", ae_id, e);
                            }
                        });
                    }

                    // Fire-and-forget skill feedback + USED_SKILL relations (success)
                    if !task_activated_skills.is_empty() {
                        let graph = self.graph.clone();
                        let skills = task_activated_skills;
                        let ae_id = task_agent_execution_id;
                        tokio::spawn(async move {
                            record_skill_feedback(
                                graph.clone(),
                                task_id,
                                run_id,
                                skills.clone(),
                                true,
                                cost_usd,
                                duration_secs,
                            )
                            .await;
                            // Create USED_SKILL relations via AgentExecution
                            if let Some(ae_id) = ae_id {
                                for skill_id in &skills {
                                    if let Err(e) = graph
                                        .create_used_skill_relation(ae_id, *skill_id, "success")
                                        .await
                                    {
                                        warn!("Failed to create USED_SKILL relation: {}", e);
                                    }
                                }
                            }
                        });
                    }

                    // Fire-and-forget persona feedback (success)
                    if !task_persona_ids.is_empty() {
                        let graph = self.graph.clone();
                        let pids = task_persona_ids;
                        tokio::spawn(async move {
                            crate::runner::persona::record_persona_feedback(graph, pids, true)
                                .await;
                        });
                    }
                }
                Ok(TaskResult::Failed {
                    reason,
                    attempts: _,
                    cost_usd,
                }) => {
                    // Step coherence check: if all steps are completed despite
                    // a Failed result, auto-complete the task instead of failing it.
                    if let Some(true) = self.check_step_coherence(task_id).await {
                        warn!(
                            "Task {} failed ({}) but all steps are completed — auto-completing",
                            task_id, reason
                        );
                        self.on_task_completed(run_id, task_id, 0.0, cost_usd, Some(cwd))
                            .await?;
                        wave_result.tasks_completed.push(task_id);
                        wave_result.wave_cost_usd += cost_usd;
                    } else {
                        self.on_task_failed(run_id, plan_id, task_id, &reason, 0.0, cost_usd)
                            .await?;
                        wave_result.tasks_failed.push((task_id, reason));
                        wave_result.wave_cost_usd += cost_usd;

                        // Fire-and-forget: finalize AgentExecution node (failure)
                        if let Some(ae_id) = task_agent_execution_id {
                            let graph = self.graph.clone();
                            let persona = task_persona.clone();
                            tokio::spawn(async move {
                                use crate::neo4j::agent_execution::{
                                    AgentExecutionNode, AgentExecutionStatus,
                                };
                                let ae = AgentExecutionNode {
                                    id: ae_id,
                                    run_id,
                                    task_id,
                                    session_id: task_session_id,
                                    started_at: task_start_time,
                                    completed_at: Some(chrono::Utc::now()),
                                    cost_usd,
                                    duration_secs: 0.0,
                                    status: AgentExecutionStatus::Failed,
                                    tools_used: "{}".to_string(),
                                    files_modified: vec![],
                                    commits: vec![],
                                    persona_profile: persona,
                                    vector_json: None,
                                    report_json: None,
                                    execution_type: Default::default(),
                                };
                                if let Err(e) = graph.update_agent_execution(&ae).await {
                                    warn!("Failed to update AgentExecution {}: {}", ae_id, e);
                                }
                            });
                        }

                        // Fire-and-forget skill feedback + USED_SKILL (failure)
                        if !task_activated_skills.is_empty() {
                            let graph = self.graph.clone();
                            let skills = task_activated_skills;
                            let ae_id = task_agent_execution_id;
                            tokio::spawn(async move {
                                record_skill_feedback(
                                    graph.clone(),
                                    task_id,
                                    run_id,
                                    skills.clone(),
                                    false,
                                    cost_usd,
                                    0.0,
                                )
                                .await;
                                if let Some(ae_id) = ae_id {
                                    for skill_id in &skills {
                                        if let Err(e) = graph
                                            .create_used_skill_relation(ae_id, *skill_id, "failure")
                                            .await
                                        {
                                            warn!("Failed to create USED_SKILL relation: {}", e);
                                        }
                                    }
                                }
                            });
                        }

                        // Fire-and-forget persona feedback (failure)
                        if !task_persona_ids.is_empty() {
                            let graph = self.graph.clone();
                            let pids = task_persona_ids;
                            tokio::spawn(async move {
                                crate::runner::persona::record_persona_feedback(graph, pids, false)
                                    .await;
                            });
                        }
                    } // end else (step coherence check)
                }
                Ok(TaskResult::Timeout {
                    duration_secs,
                    cost_usd,
                }) => {
                    // Step coherence check: if all steps completed despite timeout,
                    // auto-complete the task (the agent finished its work but hit the timeout).
                    if let Some(true) = self.check_step_coherence(task_id).await {
                        warn!(
                            "Task {} timed out but all steps are completed — auto-completing",
                            task_id
                        );
                        self.on_task_completed(run_id, task_id, duration_secs, cost_usd, Some(cwd))
                            .await?;
                        wave_result.tasks_completed.push(task_id);
                        wave_result.wave_cost_usd += cost_usd;
                    } else {
                        self.emit_event(RunnerEvent::TaskTimeout {
                            run_id,
                            task_id,
                            task_title: task_title.clone(),
                            duration_secs,
                        });
                        self.on_task_failed(
                            run_id,
                            plan_id,
                            task_id,
                            "Task timeout",
                            duration_secs,
                            cost_usd,
                        )
                        .await?;
                        wave_result
                            .tasks_failed
                            .push((task_id, "Task timeout".to_string()));
                        wave_result.wave_cost_usd += cost_usd;

                        // Fire-and-forget: finalize AgentExecution (timeout)
                        if let Some(ae_id) = task_agent_execution_id {
                            let graph = self.graph.clone();
                            let persona = task_persona.clone();
                            tokio::spawn(async move {
                                use crate::neo4j::agent_execution::{
                                    AgentExecutionNode, AgentExecutionStatus,
                                };
                                let ae = AgentExecutionNode {
                                    id: ae_id,
                                    run_id,
                                    task_id,
                                    session_id: task_session_id,
                                    started_at: task_start_time,
                                    completed_at: Some(chrono::Utc::now()),
                                    cost_usd,
                                    duration_secs,
                                    status: AgentExecutionStatus::Timeout,
                                    tools_used: "{}".to_string(),
                                    files_modified: vec![],
                                    commits: vec![],
                                    persona_profile: persona,
                                    vector_json: None,
                                    report_json: None,
                                    execution_type: Default::default(),
                                };
                                if let Err(e) = graph.update_agent_execution(&ae).await {
                                    warn!("Failed to update AgentExecution {}: {}", ae_id, e);
                                }
                            });
                        }

                        // Fire-and-forget skill feedback + USED_SKILL (timeout = failure)
                        if !task_activated_skills.is_empty() {
                            let graph = self.graph.clone();
                            let skills = task_activated_skills;
                            let ae_id = task_agent_execution_id;
                            tokio::spawn(async move {
                                record_skill_feedback(
                                    graph.clone(),
                                    task_id,
                                    run_id,
                                    skills.clone(),
                                    false,
                                    cost_usd,
                                    duration_secs,
                                )
                                .await;
                                if let Some(ae_id) = ae_id {
                                    for skill_id in &skills {
                                        if let Err(e) = graph
                                            .create_used_skill_relation(ae_id, *skill_id, "failure")
                                            .await
                                        {
                                            warn!("Failed to create USED_SKILL relation: {}", e);
                                        }
                                    }
                                }
                            });
                        }

                        // Fire-and-forget persona feedback (timeout = failure)
                        if !task_persona_ids.is_empty() {
                            let graph = self.graph.clone();
                            let pids = task_persona_ids;
                            tokio::spawn(async move {
                                crate::runner::persona::record_persona_feedback(graph, pids, false)
                                    .await;
                            });
                        }
                    } // end else (step coherence check for timeout)
                }
                Ok(TaskResult::BudgetExceeded {
                    cumulated_cost_usd,
                    limit_usd,
                }) => {
                    self.emit_event(RunnerEvent::BudgetExceeded {
                        run_id,
                        plan_id,
                        cumulated_cost_usd,
                        limit_usd,
                    });
                    wave_result.aborted = true;
                    join_set.abort_all();
                    // Drain remaining aborted tasks
                    while join_set.join_next().await.is_some() {}
                    return Ok(wave_result);
                }
                Ok(TaskResult::Blocked { blocked_by }) => {
                    warn!("Task {} blocked by {:?}", task_id, blocked_by);

                    // Extract session_id and remove agent from active_agents
                    let agent_session_id = {
                        let mut global = RUNNER_STATE.write().await;
                        let sid = global
                            .as_ref()
                            .and_then(|s| s.get_agent(&task_id))
                            .and_then(|a| a.session_id);
                        if let Some(ref mut s) = *global {
                            s.remove_agent(&task_id);
                        }
                        sid
                    };

                    let _ = self
                        .update_task_status_with_event(task_id, TaskStatus::Blocked)
                        .await;

                    // Close the agent's chat session (fire-and-forget)
                    if let Some(sid) = agent_session_id {
                        let chat_manager = self.chat_manager.clone();
                        tokio::spawn(async move {
                            let sid_str = sid.to_string();
                            if let Err(e) = chat_manager.close_session(&sid_str).await {
                                warn!(
                                    "Failed to close agent session {} after task blocked: {}",
                                    sid_str, e
                                );
                            } else {
                                debug!(
                                    "Closed agent session {} after task {} blocked",
                                    sid_str, task_id
                                );
                            }
                        });
                    }
                }
                Err(e) => {
                    error!("Task {} execution error: {}", task_id, e);
                    self.on_task_failed(run_id, plan_id, task_id, &e.to_string(), 0.0, 0.0)
                        .await?;
                    wave_result.tasks_failed.push((task_id, e.to_string()));
                }
            }

            // Budget check after each agent result
            {
                let global = RUNNER_STATE.read().await;
                if let Some(ref s) = *global {
                    if s.is_budget_exceeded(self.effective_budget()) {
                        wave_result.aborted = true;
                        drop(global);
                        join_set.abort_all();
                        while join_set.join_next().await.is_some() {}
                        return Ok(wave_result);
                    }
                }
            }

            // Cancel check after each agent result
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                info!("Runner cancelled during wave execution");
                wave_result.aborted = true;
                join_set.abort_all();
                // Also close any active agent sessions so Claude Code subprocesses stop
                {
                    let global = RUNNER_STATE.read().await;
                    if let Some(ref s) = *global {
                        let session_ids: Vec<Uuid> = s
                            .active_agents
                            .iter()
                            .filter(|a| {
                                matches!(a.status, TaskRunStatus::Running | TaskRunStatus::Spawning)
                            })
                            .filter_map(|a| a.session_id)
                            .collect();
                        drop(global);
                        Self::close_agent_sessions(&self.chat_manager, &session_ids).await;
                    }
                }
                while join_set.join_next().await.is_some() {}
                return Ok(wave_result);
            }
        }

        // =====================================================================
        // Retry loop — sequentially retry failed tasks with error context
        // =====================================================================
        if !wave_result.tasks_failed.is_empty()
            && self.config.max_retries > 0
            && !wave_result.aborted
        {
            // Build title lookup from wave tasks
            let title_map: std::collections::HashMap<Uuid, String> = wave
                .tasks
                .iter()
                .map(|t| {
                    (
                        t.id,
                        t.title.clone().unwrap_or_else(|| "untitled".to_string()),
                    )
                })
                .collect();

            // Collect retryable tasks (those where on_task_failed returned true / status is Pending)
            let retryable: Vec<(Uuid, String)> = {
                let global = RUNNER_STATE.read().await;
                wave_result
                    .tasks_failed
                    .iter()
                    .filter(|(tid, _)| {
                        // Eligible if retry_count <= max_retries (on_task_failed already incremented)
                        global
                            .as_ref()
                            .map(|s| {
                                let count = s.retry_counts.get(tid).copied().unwrap_or(0);
                                count <= self.config.max_retries
                            })
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            };

            for (task_id, failure_reason) in &retryable {
                // Check cancellation before each retry
                if RUNNER_CANCEL.load(Ordering::SeqCst) {
                    info!("Runner cancelled, skipping remaining retries");
                    break;
                }

                // Check budget before retry
                {
                    let global = RUNNER_STATE.read().await;
                    if let Some(ref s) = *global {
                        if s.is_budget_exceeded(self.effective_budget()) {
                            info!("Budget exceeded, skipping remaining retries");
                            break;
                        }
                    }
                }

                let task_title = title_map
                    .get(task_id)
                    .cloned()
                    .unwrap_or_else(|| "untitled".to_string());

                let retry_count = {
                    let global = RUNNER_STATE.read().await;
                    global
                        .as_ref()
                        .and_then(|s| s.retry_counts.get(task_id).copied())
                        .unwrap_or(1)
                };

                info!(
                    "Retrying task {} ({}) — attempt {}/{}",
                    task_id, task_title, retry_count, self.config.max_retries
                );

                // Build retry context from failure reason
                let retry_context = format!(
                    "Attempt {attempt} failed with error:\n```\n{reason}\n```\n\
                     Analyze the root cause carefully before retrying. \
                     Do NOT repeat the same approach — adapt your strategy.",
                    attempt = retry_count,
                    reason = failure_reason,
                );

                // Execute the retry
                let retry_result = self
                    .execute_task(
                        run_id,
                        plan_id,
                        *task_id,
                        &task_title,
                        cwd,
                        project_slug,
                        Some(retry_context),
                        continuity_context,
                    )
                    .await;

                match retry_result {
                    Ok(exec_result) => match exec_result.result {
                        TaskResult::Success {
                            duration_secs,
                            cost_usd,
                        } => {
                            info!(
                                "Retry succeeded for task {} ({}) in {:.1}s",
                                task_id, task_title, duration_secs
                            );
                            self.on_task_completed(
                                run_id,
                                *task_id,
                                duration_secs,
                                cost_usd,
                                Some(cwd),
                            )
                            .await?;
                            // Move from failed to completed
                            wave_result.tasks_failed.retain(|(tid, _)| tid != task_id);
                            wave_result.tasks_completed.push(*task_id);
                            wave_result.wave_cost_usd += cost_usd;

                            // Fire-and-forget enrichment for retry success
                            let enricher = TaskEnricher::new(self.graph.clone());
                            let enrich_plan_id = plan_id;
                            let enrich_project_id = project_id;
                            let enrich_session_id = exec_result.session_id();
                            let enrich_cwd = cwd.to_string();
                            let enrich_start = chrono::Utc::now();
                            let retry_task_id = *task_id;
                            tokio::spawn(async move {
                                let result = enricher
                                    .enrich(
                                        retry_task_id,
                                        enrich_plan_id,
                                        enrich_project_id,
                                        enrich_session_id,
                                        enrich_start,
                                        &enrich_cwd,
                                    )
                                    .await;
                                info!(
                                    "Enricher (retry) completed for task {}: {} commits, note={}, {} affects",
                                    retry_task_id,
                                    result.commits_linked,
                                    result.note_created,
                                    result.affects_added
                                );
                            });
                        }
                        TaskResult::Failed {
                            reason,
                            attempts: _,
                            cost_usd,
                        } => {
                            warn!(
                                "Retry failed again for task {} ({}): {}",
                                task_id, task_title, reason
                            );
                            // Update the failure reason in wave_result
                            if let Some(entry) = wave_result
                                .tasks_failed
                                .iter_mut()
                                .find(|(tid, _)| tid == task_id)
                            {
                                entry.1 = format!("retry failed: {}", reason);
                            }
                            wave_result.wave_cost_usd += cost_usd;
                            // Mark definitively failed
                            self.update_task_status_with_event(*task_id, TaskStatus::Failed)
                                .await?;
                            {
                                let mut global = RUNNER_STATE.write().await;
                                if let Some(ref mut s) = *global {
                                    s.mark_task_failed(*task_id);
                                    self.graph.update_plan_run(s).await?;
                                }
                            }
                            // Create gotcha note for future learning
                            self.create_failure_gotcha(
                                *task_id,
                                &task_title,
                                &reason,
                                retry_count,
                                project_id,
                            )
                            .await;
                        }
                        other => {
                            warn!(
                                "Retry for task {} returned unexpected result: {:?}",
                                task_id, other
                            );
                        }
                    },
                    Err(e) => {
                        error!("Retry execution error for task {}: {}", task_id, e);
                        // Mark definitively failed
                        self.update_task_status_with_event(*task_id, TaskStatus::Failed)
                            .await?;
                        {
                            let mut global = RUNNER_STATE.write().await;
                            if let Some(ref mut s) = *global {
                                s.mark_task_failed(*task_id);
                                self.graph.update_plan_run(s).await?;
                            }
                        }
                        // Create gotcha note for future learning
                        self.create_failure_gotcha(
                            *task_id,
                            &task_title,
                            &e.to_string(),
                            retry_count,
                            project_id,
                        )
                        .await;
                    }
                }
            }
        }

        Ok(wave_result)
    }

    /// Execute a single task by spawning a Claude Code agent.
    #[allow(clippy::too_many_arguments)]
    async fn execute_task(
        &self,
        run_id: Uuid,
        plan_id: Uuid,
        task_id: Uuid,
        task_title: &str,
        cwd: &str,
        project_slug: Option<&str>,
        retry_context: Option<String>,
        continuity_context: &str,
    ) -> Result<TaskExecutionResult> {
        if retry_context.is_some() {
            info!(
                "Retrying task {}: {} (with error context)",
                task_id, task_title
            );
        } else {
            info!("Executing task {}: {}", task_id, task_title);
        }

        // Update global state — register active agent
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_agent(ActiveAgent::new(task_id, task_title.to_string()));
            }
        }

        // Mark task as in_progress (with CrudEvent for WebSocket)
        self.update_task_status_with_event(task_id, TaskStatus::InProgress)
            .await?;

        // Get current wave number
        let wave_number = {
            let global = RUNNER_STATE.read().await;
            global.as_ref().map(|s| s.current_wave).unwrap_or(0)
        };

        self.emit_event(RunnerEvent::TaskStarted {
            run_id,
            task_id,
            task_title: task_title.to_string(),
            wave_number,
        });

        // --- Step 1: Profile the task ---
        let task_node = self.graph.get_task(task_id).await?;
        let steps = self.graph.get_task_steps(task_id).await.unwrap_or_default();
        let task_profile = task_node
            .as_ref()
            .map(|t| profile_task(t, steps.len()))
            .unwrap_or_else(|| {
                profile_task(
                    &crate::neo4j::models::TaskNode {
                        id: task_id,
                        title: Some(task_title.to_string()),
                        description: String::new(),
                        status: crate::neo4j::models::TaskStatus::InProgress,
                        assigned_to: None,
                        priority: None,
                        tags: vec![],
                        acceptance_criteria: vec![],
                        affected_files: vec![],
                        estimated_complexity: None,
                        actual_complexity: None,
                        created_at: chrono::Utc::now(),
                        updated_at: None,
                        started_at: None,
                        completed_at: None,
                        frustration_score: 0.0,
                        execution_context: None,
                        persona: None,
                        prompt_cache: None,
                    },
                    0,
                )
            });
        info!(
            task_id = %task_id,
            complexity = %task_profile.complexity,
            timeout_secs = task_profile.timeout_secs,
            max_cost_usd = task_profile.max_cost_usd,
            "Task profiled"
        );

        // --- Step 2: Resolve project once (reused for skills + scaffolding) ---
        let project_node = if let Some(slug) = &project_slug {
            self.graph.get_project_by_slug(slug).await.ok().flatten()
        } else {
            None
        };
        let project_id_for_skills = project_node.as_ref().map(|p| p.id);

        let skill_activation =
            if let (Some(pid), Some(ref tn)) = (project_id_for_skills, &task_node) {
                activate_skills_for_task(self.graph.as_ref(), pid, tn).await
            } else {
                None
            };

        let activated_skill_ids = skill_activation
            .as_ref()
            .map(|sa| sa.activated_skill_ids.clone())
            .unwrap_or_default();

        let skill_context = skill_activation
            .map(|sa| sa.context_text)
            .filter(|s| !s.is_empty());

        // --- Step 2b: Load PersonaStack ---
        let persona_stack = if let (Some(pid), Some(ref tn)) = (project_id_for_skills, &task_node) {
            crate::runner::persona::load_persona_stack(self.graph.as_ref(), tn, pid, &steps).await
        } else {
            None
        };

        let persona_ids_for_feedback: Vec<Uuid> = persona_stack
            .as_ref()
            .map(|ps| ps.entries().iter().map(|e| e.persona_id).collect())
            .unwrap_or_default();

        let persona_context = persona_stack
            .as_ref()
            .map(|ps| ps.render_for_prompt(ps.max_context_tokens))
            .filter(|s| !s.is_empty());

        // Build the prompt — use prompt_cache if available (pre-enrichment), else build from scratch
        let cached_prompt = task_node.as_ref().and_then(|t| t.prompt_cache.clone());
        let (mut prompt, affected_files_for_ctx) = if let Some(ref cached) = cached_prompt {
            info!(task_id = %task_id, "Using pre-enriched prompt_cache (skipping build_context)");
            let af = task_node
                .as_ref()
                .map(|t| t.affected_files.clone())
                .unwrap_or_default();
            (cached.clone(), af)
        } else {
            let context = self.context_builder.build_context(task_id, plan_id).await?;
            let p = self.context_builder.generate_prompt(&context);
            let af = context
                .target_files
                .iter()
                .map(|f| f.path.clone())
                .collect();
            (p, af)
        };

        // Build dynamic runner constraints with git branch, wave info, skills, and profile
        let git_branch = git::current_branch(cwd).await.unwrap_or_default();
        let task_tags = task_node
            .as_ref()
            .map(|t| t.tags.clone())
            .unwrap_or_default();
        let frustration_level = task_node
            .as_ref()
            .map(|t| t.frustration_score)
            .unwrap_or(0.0);
        // Resolve scaffolding level from project (best effort, default L0)
        let scaffolding_level = if let Some(ref project) = project_node {
            self.graph
                .compute_scaffolding_level(project.id, project.scaffolding_override)
                .await
                .map(|l| l.level)
                .unwrap_or(0)
        } else {
            0
        };

        // Build RunnerContext — behavioral constraints now go into the SYSTEM PROMPT
        // (via create_session), not the user message. This prevents the generic PO
        // system prompt from conflicting with runner execution instructions.
        let runner_context = crate::chat::types::RunnerContext {
            git_branch,
            task_tags,
            affected_files: affected_files_for_ctx,
            forbidden_files: vec![], // populated by execute_wave for parallel tasks
            skill_context,
            frustration_level,
            wave_number,
            parallel_agents: 1, // default, overridden by execute_wave
            scaffolding_level,
        };

        // --- Step 2b.4: Inject continuity context from previous waves ---
        // (still in user message — this is task-specific, not behavioral)
        if !continuity_context.is_empty() {
            prompt.push_str(continuity_context);
        }

        // --- Step 2b.5: Pre-read affected files so the agent has immediate context ---
        let file_contents = crate::orchestrator::context::pre_read_affected_files(
            cwd,
            &runner_context.affected_files,
            5,
            200,
        )
        .await;
        if !file_contents.is_empty() {
            prompt.push('\n');
            prompt.push_str(&file_contents);
        }

        // --- Step 2b.6: Inject retry context if this is a retry ---
        if let Some(ref ctx) = retry_context {
            prompt.push_str(&format!(
                "\n## ⚠️ Previous Attempt Failed\n\
                 This is a retry. The previous attempt failed with the following context:\n\n\
                 {}\n\n\
                 **Instructions for retry:**\n\
                 - Analyze the error carefully before starting\n\
                 - Try a DIFFERENT approach than the one that failed\n\
                 - If the error was a compilation issue, check types and imports first\n\
                 - If the error was a test failure, read the test expectations carefully\n",
                ctx
            ));
        }

        // --- Step 2c: Inject persona context ---
        if let Some(ref pc) = persona_context {
            prompt.push_str(&format!("\n## Persona Context\n{}\n", pc));
        }

        // --- Step 2d: Inject knowledge context (notes + decisions) for affected files ---
        {
            use crate::notes::EntityType as NoteEntityType;
            let mut knowledge_parts: Vec<String> = Vec::new();
            // Cap at 5 files to avoid prompt bloat
            for file_path in runner_context.affected_files.iter().take(5) {
                let mut file_notes = Vec::new();
                // Fetch notes linked to this file
                if let Ok(notes) = self
                    .graph
                    .get_notes_for_entity(&NoteEntityType::File, file_path)
                    .await
                {
                    for note in notes.iter().take(3) {
                        // Truncate content to ~200 chars
                        let excerpt = if note.content.len() > 200 {
                            format!("{}…", &note.content[..200])
                        } else {
                            note.content.clone()
                        };
                        file_notes.push(format!(
                            "  - **[{:?}]** ({:?}): {}",
                            note.note_type, note.importance, excerpt
                        ));
                    }
                }
                // Fetch decisions affecting this file
                if let Ok(decisions) = self
                    .graph
                    .get_decisions_affecting("File", file_path, Some("accepted"))
                    .await
                {
                    for dec in decisions.iter().take(2) {
                        let rationale_excerpt = if dec.rationale.len() > 150 {
                            format!("{}…", &dec.rationale[..150])
                        } else {
                            dec.rationale.clone()
                        };
                        file_notes.push(format!(
                            "  - **[Decision]** {}: {}",
                            dec.description, rationale_excerpt
                        ));
                    }
                }
                if !file_notes.is_empty() {
                    knowledge_parts.push(format!("### `{}`\n{}", file_path, file_notes.join("\n")));
                }
            }
            if !knowledge_parts.is_empty() {
                prompt.push_str("\n## Knowledge Context\n");
                prompt.push_str("The following notes and decisions are relevant to the files you will modify:\n\n");
                prompt.push_str(&knowledge_parts.join("\n\n"));
                prompt.push('\n');
            }
        }

        // --- Step 2e: Inject learned reflexes (scars, co-change, episode recall) ---
        if let Some(pid) = project_id_for_skills {
            let ref_ctx = crate::reflex::RefContext {
                affected_files: runner_context.affected_files.clone(),
                task_title: Some(task_title.to_string()),
                step_description: None,
                embedding: None,
                project_id: pid,
            };
            let engine = crate::reflex::ReflexEngine::new(self.graph.clone());
            let suggestions = engine.suggest(&ref_ctx).await;
            let reflex_block = crate::reflex::ReflexEngine::format_markdown(&suggestions);
            if !reflex_block.is_empty() {
                prompt.push_str("\n## Learned Reflexes\n");
                prompt.push_str(&reflex_block);
                prompt.push('\n');
            }
        }

        // --- Step 3: Inject complexity directive ---
        prompt.push_str(&format!(
            "\n## Cognitive Profile: {}\n{}\n",
            task_profile.complexity,
            complexity_directive(task_profile.complexity)
        ));

        // --- Step 3b: Prepend runner behavioral instructions into the user message ---
        // The --system-prompt CLI flag is ignored in interactive mode (--input-format stream-json).
        // To ensure the agent receives the autonomous execution instructions, we prepend them
        // directly into the user message. This is the ONLY reliable way to reach the agent.
        {
            let mut prompt_ctx = runner_context.to_prompt_context();
            prompt_ctx.scaffolding_level = 0; // Force FULL runner prompt — higher levels produce minimal instructions that agents dismiss
            let runner_instructions =
                crate::runner::prompt::build_runner_system_prompt(&prompt_ctx);
            let mut full_prompt =
                String::with_capacity(runner_instructions.len() + prompt.len() + 2);
            full_prompt.push_str(&runner_instructions);
            full_prompt.push_str("\n---\n\n");
            full_prompt.push_str(&prompt);
            prompt = full_prompt;
        }

        // Spawn agent: create_session → subscribe → send_message → listen
        let task_context_str = task_node
            .as_ref()
            .map(|t| {
                if t.description.is_empty() {
                    t.title.clone().unwrap_or_default()
                } else {
                    t.description.clone()
                }
            })
            .unwrap_or_default();
        let request = ChatRequest {
            message: prompt, // Send the full prompt directly in create_session — avoids the ghost empty message at seq 1
            session_id: None,
            cwd: cwd.to_string(),
            project_slug: project_slug.map(|s| s.to_string()),
            model: None,
            permission_mode: Some("bypassPermissions".to_string()),
            add_dirs: None,
            workspace_slug: None,
            user_claims: Some(self.user_claims.clone().unwrap_or_else(|| {
                crate::auth::jwt::Claims::service_account(&format!("runner-agent:{}", run_id))
            })),
            spawned_by: Some(
                serde_json::json!({
                    "type": "runner",
                    "run_id": run_id.to_string(),
                    "plan_id": plan_id.to_string(),
                    "task_id": task_id.to_string(),
                    "scaffolding_level": scaffolding_level,
                })
                .to_string(),
            ),
            task_context: Some(task_context_str),
            scaffolding_override: None,
            runner_context: Some(runner_context),
        };

        let spawning_timeout = Duration::from_secs(self.config.spawning_timeout_secs);
        let session = match tokio::time::timeout(
            spawning_timeout,
            self.chat_manager.create_session(&request),
        )
        .await
        {
            Ok(Ok(session)) => session,
            Ok(Err(e)) => {
                // create_session returned an error (not a timeout)
                return Err(e);
            }
            Err(_elapsed) => {
                // Spawning timed out — create_session() hung
                warn!(
                    "Task {} — create_session() timed out after {}s",
                    task_id, self.config.spawning_timeout_secs
                );
                self.emit_event(RunnerEvent::TaskSpawningTimeout {
                    run_id,
                    task_id,
                    task_title: task_title.to_string(),
                    timeout_secs: self.config.spawning_timeout_secs,
                });
                // Transition agent status: Spawning → Timeout
                {
                    let mut global = RUNNER_STATE.write().await;
                    if let Some(ref mut s) = *global {
                        s.update_agent_status(&task_id, TaskRunStatus::Timeout);
                    }
                }
                let result = TaskResult::Failed {
                    reason: format!(
                        "Agent spawning timed out after {}s — create_session() did not respond",
                        self.config.spawning_timeout_secs
                    ),
                    attempts: 0,
                    cost_usd: 0.0,
                };
                self.finalize_steps(task_id, &result, cwd).await;
                return Ok(TaskExecutionResult {
                    result,
                    session_id: None,
                    activated_skill_ids: activated_skill_ids.clone(),
                    persona_ids: persona_ids_for_feedback.clone(),
                    agent_execution_id: Uuid::new_v4(),
                    persona_profile: String::new(),
                    report: None,
                });
            }
        };
        let session_id = session.session_id.clone();
        let session_uuid = session_id.parse::<Uuid>().ok();

        // Transition agent status: Spawning → Running (session is now live)
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.update_agent_status(&task_id, TaskRunStatus::Running);
                s.set_agent_session(&task_id, session_uuid);
            }
        }

        // Create an AgentExecution ID for this task
        let agent_execution_id = Uuid::new_v4();
        let persona_str = persona_stack
            .as_ref()
            .and_then(|ps| ps.get_primary())
            .map(|p| format!("{}:{}", p.persona_name, task_profile.complexity))
            .unwrap_or_else(|| task_profile.complexity.to_string());

        // Create AgentExecution node in Neo4j (fire-and-forget)
        {
            use crate::neo4j::agent_execution::{AgentExecutionNode, AgentExecutionStatus};
            let ae = AgentExecutionNode {
                id: agent_execution_id,
                run_id,
                task_id,
                session_id: session_uuid,
                started_at: chrono::Utc::now(),
                completed_at: None,
                cost_usd: 0.0,
                duration_secs: 0.0,
                status: AgentExecutionStatus::Running,
                tools_used: "{}".to_string(),
                files_modified: vec![],
                commits: vec![],
                persona_profile: persona_str.clone(),
                vector_json: None,
                report_json: None,
                execution_type: Default::default(),
            };
            let graph = self.graph.clone();
            tokio::spawn(async move {
                if let Err(e) = graph.create_agent_execution(&ae).await {
                    warn!("Failed to create AgentExecution node: {}", e);
                }
            });
        }

        // Helper to wrap TaskResult with session_id and activated skills/personas
        let activated_ids = activated_skill_ids.clone();
        let persona_ids_clone = persona_ids_for_feedback.clone();
        let persona_clone = persona_str.clone();
        let wrap = move |result: TaskResult| -> TaskExecutionResult {
            TaskExecutionResult {
                result,
                session_id: session_uuid,
                activated_skill_ids: activated_ids.clone(),
                persona_ids: persona_ids_clone.clone(),
                agent_execution_id,
                persona_profile: persona_clone.clone(),
                report: None,
            }
        };

        // Subscribe to events BEFORE the background task starts streaming
        // (create_session spawns a tokio task that sends the message — subscribe must happen first)
        let rx = self.chat_manager.subscribe(&session_id).await?;
        // Clone a second receiver for the guard
        let guard_rx = self.chat_manager.subscribe(&session_id).await?;

        let start = std::time::Instant::now();

        // Spawn the AgentGuard in parallel (timeout adapted to task profile)
        let guard_config = GuardConfig {
            idle_timeout: Duration::from_secs(self.config.idle_timeout_secs),
            task_timeout: Duration::from_secs(task_profile.timeout_secs),
            spawning_timeout: Duration::from_secs(self.config.spawning_timeout_secs),
            loop_threshold: 3,
            completion_loop_threshold: self.config.completion_loop_threshold,
            completion_max_chars: self.config.completion_max_chars,
            ..Default::default()
        };
        let hint_sender = Arc::new(ChatManagerHintSender {
            chat_manager: self.chat_manager.clone(),
        });
        let guard = AgentGuard::new(
            session_id.clone(),
            task_title.to_string(),
            task_id,
            guard_config,
            guard_rx,
            Some(hint_sender),
            Some(self.graph.clone()),
            Some(plan_id),
        );
        // Attach events_tx so the guard can emit CompactionRecovery metrics
        let guard = match self.chat_manager.get_events_tx(&session_id).await {
            Ok(tx) => guard.with_events_tx(tx),
            Err(_) => guard, // best-effort: if session not found, skip
        };

        let guard_handle = tokio::spawn(async move { guard.monitor().await });

        // Run listener and guard in parallel with tokio::select!
        // Priority: Result event > Guard completion loop > Guard timeout
        //
        // Previously, listen_for_result was awaited sequentially then guard
        // verdict was checked — causing a race where guard timeout would
        // override a successful Result event.
        let (event_result, event_metrics, guard_verdict) = {
            let listen_fut = self.listen_for_result(rx, run_id, Some(task_id));
            tokio::pin!(listen_fut);
            tokio::pin!(guard_handle);

            tokio::select! {
                // Branch 1: listen_for_result completes (Result event received or channel closed)
                (result, metrics) = &mut listen_fut => {
                    // Got an event result — abort the guard (we don't need it anymore)
                    guard_handle.abort();
                    // Guard was aborted — we don't need its verdict since we have a Result event
                    let verdict = GuardVerdict::Completed;
                    (result, metrics, verdict)
                }
                // Branch 2: guard finishes first (timeout or completion loop detected)
                verdict = &mut guard_handle => {
                    let verdict = match verdict {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Guard task panicked: {}", e);
                            GuardVerdict::Completed
                        }
                    };
                    match &verdict {
                        GuardVerdict::Timeout { .. } | GuardVerdict::CompletionLoopDetected => {
                            let is_loop = matches!(verdict, GuardVerdict::CompletionLoopDetected);
                            if is_loop {
                                info!("Guard detected completion loop, waiting 5s for Result event...");
                            }
                            // Give listener a grace period to get a pending Result
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                &mut listen_fut,
                            ).await {
                                Ok((result, metrics)) => {
                                    // Got a Result within grace period — use it
                                    let effective_verdict = if is_loop {
                                        GuardVerdict::Completed
                                    } else {
                                        verdict
                                    };
                                    (result, metrics, effective_verdict)
                                }
                                Err(_) => {
                                    if is_loop {
                                        // No Result but agent said it's done — treat as success
                                        info!("No Result event but agent signaled completion — treating as success");
                                        (EventListenResult::Completed {
                                            cost_usd: 0.0,
                                            is_error: false,
                                            error_text: String::new(),
                                            subtype: "completion_loop".to_string(),
                                        }, EventMetrics::default(), GuardVerdict::Completed)
                                    } else {
                                        // True timeout — no Result within grace period
                                        (EventListenResult::ChannelClosed { cost_usd: 0.0 }, EventMetrics::default(), verdict)
                                    }
                                }
                            }
                        }
                        _ => {
                            // Completed/Cancelled from guard — get listener result
                            let (result, metrics) = listen_fut.await;
                            (result, metrics, verdict.clone())
                        }
                    }
                }
            }
        };

        // Process the event listener result
        let (cost_usd, is_error, error_text, _subtype, _timed_out) = match event_result {
            EventListenResult::Completed {
                cost_usd,
                is_error,
                error_text,
                subtype,
            } => (cost_usd, is_error, error_text, subtype, false),
            EventListenResult::ChannelClosed { cost_usd } => {
                // Channel closed without Result — check guard verdict
                let result = if let GuardVerdict::Timeout { elapsed_secs } = guard_verdict {
                    TaskResult::Timeout {
                        duration_secs: elapsed_secs,
                        cost_usd,
                    }
                } else {
                    TaskResult::Failed {
                        reason: "Chat event channel closed unexpectedly".to_string(),
                        attempts: 0,
                        cost_usd,
                    }
                };
                self.finalize_steps(task_id, &result, cwd).await;
                return Ok(wrap(result));
            }
            EventListenResult::Cancelled { cost_usd } => {
                let result = TaskResult::Failed {
                    reason: "Cancelled by user".to_string(),
                    attempts: 0,
                    cost_usd,
                };
                self.finalize_steps(task_id, &result, cwd).await;
                return Ok(wrap(result));
            }
        };

        // Only honor guard timeout if we did NOT get a successful Result
        // (the select! already gave priority to Results, but double-check)
        if is_error {
            if let GuardVerdict::Timeout { elapsed_secs } = guard_verdict {
                let result = TaskResult::Timeout {
                    duration_secs: elapsed_secs,
                    cost_usd,
                };
                self.finalize_steps(task_id, &result, cwd).await;
                return Ok(wrap(result));
            }
        }

        let duration_secs = start.elapsed().as_secs_f64();

        // Check budget AFTER this task's cost
        // Note: agent cost is already updated in listen_for_result for real-time tracking
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_cost(cost_usd);
                if s.is_budget_exceeded(self.effective_budget()) {
                    let result = TaskResult::BudgetExceeded {
                        cumulated_cost_usd: s.cost_usd,
                        limit_usd: self.effective_budget(),
                    };
                    self.finalize_steps(task_id, &result, cwd).await;
                    return Ok(wrap(result));
                }
            }
        }

        if is_error {
            // error_max_turns means the agent hit the CLI turn limit — NOT a real failure.
            // The agent likely finished its work; proceed to verification.
            if _subtype == "error_max_turns" {
                info!(
                    "Task {} hit max_turns — proceeding to verification (code may be complete)",
                    task_id
                );
            } else {
                let result = TaskResult::Failed {
                    reason: if error_text.is_empty() {
                        format!("Agent returned error (subtype: {})", _subtype)
                    } else {
                        error_text
                    },
                    attempts: 0,
                    cost_usd,
                };
                self.finalize_steps(task_id, &result, cwd).await;
                return Ok(wrap(result));
            }
        }

        // Fallback: auto-complete any steps the agent didn't update during execution.
        // This is the success path — finalize_steps marks remaining pending/in_progress as completed.
        {
            // Check for uncommitted changes — agent might have written code but forgot to commit
            let output = tokio::process::Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(cwd)
                .output()
                .await;
            let has_uncommitted = output
                .map(|o| {
                    o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty()
                })
                .unwrap_or(false);

            if has_uncommitted {
                warn!(
                    "Task {} agent has uncommitted changes — code was written but not committed",
                    task_id
                );
            }

            let success_result = TaskResult::Success {
                cost_usd,
                duration_secs: start.elapsed().as_secs_f64(),
            };
            let breakdown = self.finalize_steps(task_id, &success_result, cwd).await;

            // Step completion guard: if the task has steps but NONE were completed,
            // the agent likely skipped all work — override to Failed.
            if should_step_guard_trigger(&breakdown) {
                warn!(
                    "Task {} — step completion guard triggered: 0/{} steps completed ({} skipped) — overriding to Failed",
                    task_id, breakdown.total, breakdown.skipped
                );
                self.emit_event(RunnerEvent::TaskCompletedWithoutSteps {
                    run_id,
                    task_id,
                    task_title: task_title.to_string(),
                    steps_skipped: breakdown.skipped,
                    steps_total: breakdown.total,
                });
                // Override: return Failed instead of Success
                let failed_result = TaskResult::Failed {
                    reason: format!(
                        "Step completion guard: 0/{} steps completed ({} skipped, {} pending)",
                        breakdown.total, breakdown.skipped, breakdown.pending
                    ),
                    attempts: 0,
                    cost_usd,
                };
                return Ok(wrap(failed_result));
            }
        }

        // Post-execution: persist persona used on the task node (Step 5 — T9)
        {
            let persona_for_persist = persona_str.clone();
            let graph = self.graph.clone();
            let tid = task_id;
            tokio::spawn(async move {
                if let Err(e) = graph
                    .update_task_enrichment(tid, None, Some(&persona_for_persist), None)
                    .await
                {
                    warn!("Failed to persist persona on task {}: {}", tid, e);
                }
            });
        }

        // NOTE: Verification (build check, steps, git sanity) is now performed
        // post-wave by execute_wave, not per-task. This avoids redundant build
        // checks when multiple tasks run in parallel within the same wave.

        // Build TaskExecutionReport from event metrics + git data
        let git_files = {
            let output = tokio::process::Command::new("git")
                .args(["diff", "--name-only", "HEAD~1..HEAD"])
                .current_dir(cwd)
                .output()
                .await;
            output
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .filter(|l| !l.is_empty())
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let git_commits = {
            let output = tokio::process::Command::new("git")
                .args(["log", "--oneline", "--format=%H", "-5"])
                .current_dir(cwd)
                .output()
                .await;
            output
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .filter(|l| !l.is_empty())
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };

        let mut report = TaskExecutionReport {
            tool_use_count: event_metrics.tool_use_count,
            tool_use_breakdown: event_metrics.tool_use_breakdown,
            error_count: event_metrics.error_count,
            last_error: event_metrics.last_error,
            files_modified: git_files,
            commits: git_commits,
            agent_success: !is_error,
            cost_usd,
            duration_secs,
            confidence_score: 0.0,
        };
        report.compute_confidence();

        info!(
            "Task {} report: {} tool_uses, {} errors, {} files, {} commits, confidence={:.2}",
            task_id,
            report.tool_use_count,
            report.error_count,
            report.files_modified.len(),
            report.commits.len(),
            report.confidence_score,
        );

        let mut result = wrap(TaskResult::Success {
            duration_secs,
            cost_usd,
        });
        result.report = Some(report);
        Ok(result)
    }

    // ========================================================================
    // Step coherence check
    // ========================================================================

    /// Check if all steps of a task are completed, indicating the task should
    /// be auto-completed even if the guard didn't detect it (e.g. timeout).
    ///
    /// Returns `Some(true)` if all steps are completed (task should be completed).
    /// Returns `Some(false)` if task is marked completed but no steps were updated (suspicious).
    /// Returns `None` if the check is inconclusive (mixed states or no steps).
    async fn check_step_coherence(&self, task_id: Uuid) -> Option<bool> {
        let steps = self.graph.get_task_steps(task_id).await.ok()?;
        if steps.is_empty() {
            return None;
        }

        let all_completed = steps
            .iter()
            .all(|s| s.status == crate::neo4j::models::StepStatus::Completed);

        if all_completed {
            return Some(true);
        }

        // Check suspicious case: no steps updated at all
        let all_pending_or_skipped = steps.iter().all(|s| {
            s.status == crate::neo4j::models::StepStatus::Pending
                || s.status == crate::neo4j::models::StepStatus::Skipped
        });

        if all_pending_or_skipped {
            return Some(false);
        }

        None
    }

    // ========================================================================
    // Auto-formatting
    // ========================================================================

    /// Run `cargo fmt --all` in the given working directory.
    /// Returns Ok(()) if formatting succeeds, Err if it fails.
    /// This is non-fatal: callers should log a warning on failure.
    async fn run_cargo_fmt(cwd: &str) -> Result<()> {
        let output = tokio::process::Command::new("cargo")
            .args(["fmt", "--all"])
            .current_dir(cwd)
            .output()
            .await?;

        if output.status.success() {
            info!("cargo fmt --all completed successfully in {}", cwd);
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!(
                "cargo fmt --all failed (exit {}): {}",
                output.status,
                stderr.trim()
            ))
        }
    }

    // ========================================================================
    // Task lifecycle callbacks
    // ========================================================================

    async fn on_task_completed(
        &self,
        run_id: Uuid,
        task_id: Uuid,
        duration_secs: f64,
        cost_usd: f64,
        cwd: Option<&str>,
    ) -> Result<()> {
        // Extract session_id BEFORE mark_task_completed removes the agent
        let agent_session_id = {
            let global = RUNNER_STATE.read().await;
            global
                .as_ref()
                .and_then(|s| s.get_agent(&task_id))
                .and_then(|a| a.session_id)
        };

        // Run cargo fmt before marking task as completed (non-fatal)
        if let Some(cwd) = cwd {
            if let Err(e) = Self::run_cargo_fmt(cwd).await {
                warn!("cargo fmt failed for task {} (non-fatal): {}", task_id, e);
            }
        }

        // Update task status (with CrudEvent for WebSocket)
        self.update_task_status_with_event(task_id, TaskStatus::Completed)
            .await?;

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.mark_task_completed(task_id);
                // Persist to Neo4j
                self.graph.update_plan_run(s).await?;
            }
        }

        // Close the agent's chat session (fire-and-forget)
        if let Some(sid) = agent_session_id {
            let chat_manager = self.chat_manager.clone();
            tokio::spawn(async move {
                let sid_str = sid.to_string();
                if let Err(e) = chat_manager.close_session(&sid_str).await {
                    warn!(
                        "Failed to close agent session {} after task completion: {}",
                        sid_str, e
                    );
                } else {
                    debug!(
                        "Closed agent session {} after task {} completed",
                        sid_str, task_id
                    );
                }
            });
        }

        // Record in vector collector
        {
            let mut collector = VECTOR_COLLECTOR.write().await;
            collector.record_task(task_id, duration_secs, cost_usd);
        }

        // Emit event
        self.emit_event(RunnerEvent::TaskCompleted {
            run_id,
            task_id,
            task_title: String::new(), // Title already logged
            cost_usd,
            duration_secs,
        });

        info!(
            "Task {} completed (duration: {:.1}s, cost: ${:.4})",
            task_id, duration_secs, cost_usd
        );

        Ok(())
    }

    /// Handle a task failure. Returns `true` if a retry is eligible (retry count
    /// has been incremented and task is marked Pending, not Failed).
    async fn on_task_failed(
        &self,
        run_id: Uuid,
        _plan_id: Uuid,
        task_id: Uuid,
        reason: &str,
        duration_secs: f64,
        cost_usd: f64,
    ) -> Result<bool> {
        // Check retry eligibility
        let (should_retry, retry_count) = {
            let global = RUNNER_STATE.read().await;
            if let Some(ref s) = *global {
                let count = s.retry_counts.get(&task_id).copied().unwrap_or(0);
                (count < self.config.max_retries, count)
            } else {
                (false, 0)
            }
        };

        if should_retry {
            // Increment retry count
            {
                let mut global = RUNNER_STATE.write().await;
                if let Some(ref mut s) = *global {
                    *s.retry_counts.entry(task_id).or_insert(0) += 1;
                }
            }

            warn!(
                "Task {} failed ({}), retry {}/{} eligible",
                task_id,
                reason,
                retry_count + 1,
                self.config.max_retries
            );

            // Mark task back to Pending so retry can re-enter InProgress
            self.update_task_status_with_event(task_id, TaskStatus::Pending)
                .await?;

            self.emit_event(RunnerEvent::TaskFailed {
                run_id,
                task_id,
                task_title: String::new(),
                reason: format!(
                    "{} (will retry {}/{})",
                    reason,
                    retry_count + 1,
                    self.config.max_retries
                ),
                attempts: retry_count + 1,
            });

            error!(
                "Task {} failed (retryable): {} (duration: {:.1}s, cost: ${:.4})",
                task_id, reason, duration_secs, cost_usd
            );

            return Ok(true);
        }

        // No retries left — mark as definitively Failed

        // Extract session_id BEFORE mark_task_failed removes the agent
        let agent_session_id = {
            let global = RUNNER_STATE.read().await;
            global
                .as_ref()
                .and_then(|s| s.get_agent(&task_id))
                .and_then(|a| a.session_id)
        };

        self.update_task_status_with_event(task_id, TaskStatus::Failed)
            .await?;

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.mark_task_failed(task_id);
                self.graph.update_plan_run(s).await?;
            }
        }

        // Close the agent's chat session (fire-and-forget, definitive failure only)
        if let Some(sid) = agent_session_id {
            let chat_manager = self.chat_manager.clone();
            tokio::spawn(async move {
                let sid_str = sid.to_string();
                if let Err(e) = chat_manager.close_session(&sid_str).await {
                    warn!(
                        "Failed to close agent session {} after task failure: {}",
                        sid_str, e
                    );
                } else {
                    debug!(
                        "Closed agent session {} after task {} failed (final)",
                        sid_str, task_id
                    );
                }
            });
        }

        self.emit_event(RunnerEvent::TaskFailed {
            run_id,
            task_id,
            task_title: String::new(),
            reason: reason.to_string(),
            attempts: retry_count,
        });

        error!(
            "Task {} failed (final): {} (duration: {:.1}s, cost: ${:.4})",
            task_id, reason, duration_secs, cost_usd
        );

        Ok(false)
    }

    /// Create a gotcha note when a task fails definitively after all retries.
    ///
    /// This captures the failure context as a knowledge artifact so future runs
    /// (and humans) can learn from the failure pattern.
    async fn create_failure_gotcha(
        &self,
        task_id: Uuid,
        task_title: &str,
        reason: &str,
        retry_count: u32,
        project_id: Option<Uuid>,
    ) {
        use crate::notes::models::{Note, NoteImportance, NoteType};

        let content = format!(
            "## Task Failed After {retries} Retry(ies)\n\n\
             **Task**: {title} (`{id}`)\n\n\
             **Final error**:\n```\n{reason}\n```\n\n\
             **Retries attempted**: {retries}/{max}\n\n\
             This task could not be completed autonomously. \
             Manual investigation is recommended.",
            title = task_title,
            id = task_id,
            reason = reason,
            retries = retry_count,
            max = self.config.max_retries,
        );

        let mut note = Note::new(
            project_id,
            NoteType::Gotcha,
            content,
            "runner-retry".to_string(),
        );
        note.importance = NoteImportance::High;
        note.tags = vec![
            "auto-generated".to_string(),
            "task-failure".to_string(),
            "retry-exhausted".to_string(),
            format!("task:{}", task_id),
        ];

        let note_id = note.id;
        if let Err(e) = self.graph.create_note(&note).await {
            warn!(
                "Failed to create failure gotcha note for task {}: {}",
                task_id, e
            );
            return;
        }

        // Link note to the task
        use crate::notes::models::EntityType as NoteEntityType;
        let task_id_str = task_id.to_string();
        if let Err(e) = self
            .graph
            .link_note_to_entity(note_id, &NoteEntityType::Task, &task_id_str, None, None)
            .await
        {
            warn!("Failed to link gotcha note to task {}: {}", task_id, e);
        }

        info!(
            "Created failure gotcha note {} for task {} ({})",
            note_id, task_id, task_title
        );
    }

    /// Create a PR automatically via `gh pr create` after plan completion.
    ///
    /// Collects tasks and commits, generates a markdown body, and calls `gh`.
    /// Returns the PR URL on success.
    async fn create_auto_pr(&self, plan_id: Uuid) -> Result<String> {
        // Get plan info
        let plan = self
            .graph
            .get_plan(plan_id)
            .await?
            .ok_or_else(|| anyhow!("Plan {} not found for auto-PR", plan_id))?;

        // Get tasks
        let tasks = self.graph.get_plan_tasks(plan_id).await?;

        // Get commits
        let commits = self.graph.get_plan_commits(plan_id).await?;

        // Build PR body
        let mut body = format!("## {}\n\n", plan.title);
        if !plan.description.is_empty() {
            body.push_str(&format!("{}\n\n", plan.description));
        }

        body.push_str("### Tasks\n\n");
        for task in &tasks {
            use crate::neo4j::models::TaskStatus;
            let emoji = match task.status {
                TaskStatus::Completed => "✅",
                TaskStatus::Failed => "❌",
                TaskStatus::Blocked => "🚫",
                _ => "⬜",
            };
            body.push_str(&format!(
                "- {} {}\n",
                emoji,
                task.title.as_deref().unwrap_or("Untitled")
            ));
        }

        if !commits.is_empty() {
            body.push_str("\n### Commits\n\n");
            for commit in &commits {
                body.push_str(&format!(
                    "- `{}` {}\n",
                    &commit.hash[..7.min(commit.hash.len())],
                    commit.message
                ));
            }
        }

        body.push_str("\n---\n🤖 Generated by PlanRunner\n");

        let pr_title = format!("[PlanRunner] {}", plan.title);

        let output = tokio::process::Command::new("gh")
            .args(["pr", "create", "--title", &pr_title, "--body", &body])
            .output()
            .await
            .map_err(|e| anyhow!("Failed to run gh: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("gh pr create failed: {}", stderr));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Finalize steps for a task on ALL exit paths (success, timeout, error, cancelled).
    ///
    /// - `Success`: pending/in_progress → completed
    /// - `Timeout` with commits: in_progress → completed, pending → skipped
    /// - `Timeout` without commits: all → skipped
    /// - `Error`/`Cancelled`/`BudgetExceeded`: all → skipped
    async fn finalize_steps(
        &self,
        task_id: Uuid,
        outcome: &TaskResult,
        cwd: &str,
    ) -> StepBreakdown {
        // Check if agent made any commits (proof of work)
        let has_commits = match outcome {
            TaskResult::Timeout { .. } => {
                let output = tokio::process::Command::new("git")
                    .args(["log", "--oneline", "-1", "--since=1 hour ago"])
                    .current_dir(cwd)
                    .output()
                    .await;
                output
                    .map(|o| {
                        o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty()
                    })
                    .unwrap_or(false)
            }
            _ => false,
        };

        let steps = match self.graph.get_task_steps(task_id).await {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "finalize_steps: failed to get steps for task {}: {}",
                    task_id, e
                );
                return StepBreakdown::default();
            }
        };

        // Count step statuses BEFORE finalization (reflects what the agent actually did)
        let mut breakdown = StepBreakdown {
            completed: steps
                .iter()
                .filter(|s| s.status == crate::neo4j::models::StepStatus::Completed)
                .count(),
            skipped: steps
                .iter()
                .filter(|s| s.status == crate::neo4j::models::StepStatus::Skipped)
                .count(),
            pending: steps
                .iter()
                .filter(|s| {
                    s.status == crate::neo4j::models::StepStatus::Pending
                        || s.status == crate::neo4j::models::StepStatus::InProgress
                })
                .count(),
            total: steps.len(),
        };

        let remaining: Vec<_> = steps
            .iter()
            .filter(|s| {
                s.status == crate::neo4j::models::StepStatus::Pending
                    || s.status == crate::neo4j::models::StepStatus::InProgress
            })
            .collect();

        if remaining.is_empty() {
            return breakdown;
        }

        // Warn if agent didn't update ANY step (all still pending = agent ignored MCP step updates)
        let all_pending = remaining
            .iter()
            .all(|s| s.status == crate::neo4j::models::StepStatus::Pending);
        if all_pending && remaining.len() == steps.len() {
            warn!(
                "Task {} — agent updated 0/{} steps during execution (all still pending)",
                task_id,
                steps.len()
            );
        }

        for step in &remaining {
            let new_status = match outcome {
                TaskResult::Success { .. } => crate::neo4j::models::StepStatus::Completed,
                TaskResult::Timeout { .. } => {
                    if has_commits && step.status == crate::neo4j::models::StepStatus::InProgress {
                        crate::neo4j::models::StepStatus::Completed
                    } else {
                        crate::neo4j::models::StepStatus::Skipped
                    }
                }
                _ => crate::neo4j::models::StepStatus::Skipped,
            };

            // Update breakdown to reflect finalized statuses
            match new_status {
                crate::neo4j::models::StepStatus::Completed => {
                    breakdown.completed += 1;
                    breakdown.pending -= 1;
                }
                crate::neo4j::models::StepStatus::Skipped => {
                    breakdown.skipped += 1;
                    breakdown.pending -= 1;
                }
                _ => {}
            }

            if let Err(e) = self.graph.update_step_status(step.id, new_status).await {
                warn!("finalize_steps: failed to update step {}: {}", step.id, e);
            }
        }

        info!(
            "Task {} — finalized {} remaining steps (outcome: {}, breakdown: {}/{}/{} completed/skipped/pending)",
            task_id,
            remaining.len(),
            match outcome {
                TaskResult::Success { .. } => "success",
                TaskResult::Timeout { .. } =>
                    if has_commits {
                        "timeout+commits"
                    } else {
                        "timeout"
                    },
                TaskResult::Failed { .. } => "failed",
                TaskResult::BudgetExceeded { .. } => "budget_exceeded",
                TaskResult::Blocked { .. } => "blocked",
            },
            breakdown.completed,
            breakdown.skipped,
            breakdown.pending,
        );

        breakdown
    }

    /// Finalize the run — update state, persist, emit event, cleanup worktrees.
    ///
    /// `cwd` is optional: when provided, remaining agent worktrees are cleaned up.
    async fn finalize_run(
        &self,
        run_id: Uuid,
        status: PlanRunStatus,
        cwd: Option<&str>,
    ) -> Result<()> {
        // Cleanup any remaining agent worktrees before finalizing
        if let Some(cwd) = cwd {
            match git::WorktreeCollector::cleanup_worktrees(cwd).await {
                Ok(count) if count > 0 => {
                    info!("Cleaned up {} worktrees during finalize_run", count);
                }
                Err(e) => {
                    warn!("Worktree cleanup failed (non-fatal): {}", e);
                }
                _ => {}
            }
        }

        // Clear budget override
        RUNNER_BUDGET.store(0, std::sync::atomic::Ordering::Relaxed);

        // ============================================================
        // Phase 1: Acquire write lock, finalize state, extract all
        // needed data, then DROP the lock before any async work.
        // This prevents deadlocks (create_failure_note, fire_lifecycle_transition
        // etc. may also acquire RUNNER_STATE).
        // ============================================================
        struct StateSnapshot {
            plan_id: Uuid,
            project_id: Option<Uuid>,
            lifecycle_run_id: Option<Uuid>,
            total_cost: f64,
            total_duration: f64,
            completed_tasks: Vec<Uuid>,
            failed_tasks: Vec<Uuid>,
            started_at: chrono::DateTime<chrono::Utc>,
        }

        let snapshot = {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.finalize(status);
                self.graph.update_plan_run(s).await?;
            }

            // Finalize the execution vector from the collector
            if let Some(ref state) = *global {
                let collector = VECTOR_COLLECTOR.read().await;
                let vector = collector.finalize(state);
                info!(
                    "Run {} vector: efficiency={:.2}, velocity={:.3}, stability={:.2}",
                    run_id,
                    vector.efficiency(),
                    vector.velocity(),
                    vector.stability()
                );
            }

            // Extract everything we need, then drop the lock
            global.as_ref().map(|s| StateSnapshot {
                plan_id: s.plan_id,
                project_id: s.project_id,
                lifecycle_run_id: s.lifecycle_run_id,
                total_cost: s.cost_usd,
                total_duration: s.elapsed_secs(),
                completed_tasks: s.completed_tasks.clone(),
                failed_tasks: s.failed_tasks.clone(),
                started_at: s.started_at,
            })
        };
        // ============================================================
        // Write lock is now RELEASED — safe to call async methods
        // ============================================================

        let snap = match snapshot {
            Some(s) => s,
            None => return Ok(()),
        };

        let plan_id = snap.plan_id;
        let lifecycle_run_id = snap.lifecycle_run_id;
        let total_cost = snap.total_cost;
        let total_duration = snap.total_duration;
        let tc = snap.completed_tasks.len();
        let tf = snap.failed_tasks.len();

        // Transition plan status to match the run's terminal state
        if plan_id != Uuid::nil() {
            let plan_status = match status {
                PlanRunStatus::Completed => Some(PlanStatus::Completed),
                PlanRunStatus::CompletedWithErrors => Some(PlanStatus::InProgress),
                PlanRunStatus::Failed | PlanRunStatus::BudgetExceeded => {
                    Some(PlanStatus::Cancelled)
                }
                PlanRunStatus::Cancelled => Some(PlanStatus::Cancelled),
                _ => None,
            };
            if let Some(ps) = plan_status {
                if let Err(e) = self.graph.update_plan_status(plan_id, ps.clone()).await {
                    warn!(
                        "Failed to set plan {} status to {:?} (idempotent, continuing): {}",
                        plan_id, ps, e
                    );
                } else {
                    info!("Plan {} status set to {:?}", plan_id, ps);
                }
            }
        }

        match status {
            PlanRunStatus::Completed => {
                // ============================================================
                // Lifecycle FSM: child_completed → post_run
                // ============================================================
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "child_completed")
                        .await;
                }

                // Post-run enricher sweep: catch commits missed by per-task enrichment
                // (e.g., mega-commit at end instead of atomic per-task commits)
                if let Some(cwd) = cwd {
                    let enricher = TaskEnricher::new(self.graph.clone());
                    let sweep_linked = enricher
                        .post_run_sweep(
                            snap.plan_id,
                            &snap.completed_tasks,
                            snap.started_at,
                            snap.project_id,
                            cwd,
                        )
                        .await;
                    if sweep_linked > 0 {
                        info!(
                            "Post-run sweep linked {} commit→task relations",
                            sweep_linked
                        );
                    }
                }

                // ============================================================
                // Lifecycle FSM: commits_linked → pr_decision
                // ============================================================
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "commits_linked")
                        .await;
                }

                // ============================================================
                // PR decision: evaluate guard and create PR or skip
                // ============================================================
                let pr_url = if let Some(lc_run_id) = lifecycle_run_id {
                    // Evaluate guard: has_commits && is_implementation
                    let should_pr = self.evaluate_pr_guard(plan_id).await;
                    if should_pr {
                        // Try to create PR, then fire pr_created
                        let url = match self.create_auto_pr(plan_id).await {
                            Ok(url) => {
                                info!("Auto-PR created: {}", url);
                                Some(url)
                            }
                            Err(e) => {
                                warn!("Auto-PR failed (non-fatal): {}", e);
                                None
                            }
                        };
                        self.fire_lifecycle_transition(run_id, lc_run_id, "pr_created")
                            .await;
                        url
                    } else {
                        info!("PR guard not met — skipping PR creation");
                        self.fire_lifecycle_transition(run_id, lc_run_id, "skip_pr")
                            .await;
                        None
                    }
                } else if self.config.auto_pr {
                    // Fallback: no lifecycle protocol, use config flag
                    match self.create_auto_pr(plan_id).await {
                        Ok(url) => {
                            info!("Auto-PR created: {}", url);
                            Some(url)
                        }
                        Err(e) => {
                            warn!("Auto-PR failed (non-fatal): {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                self.emit_event(RunnerEvent::PlanCompleted {
                    run_id,
                    plan_id,
                    status: PlanRunStatus::Completed,
                    total_cost_usd: total_cost,
                    total_duration_secs: total_duration,
                    tasks_completed: tc,
                    tasks_failed: tf,
                    pr_url,
                });
                info!("Plan run {} completed successfully", run_id);
            }
            PlanRunStatus::CompletedWithErrors => {
                // Lifecycle FSM: child_completed (partial success is still completion)
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "child_completed")
                        .await;
                }

                // Post-run enricher sweep (same as Completed — partial work still has commits)
                if let Some(cwd) = cwd {
                    let enricher = TaskEnricher::new(self.graph.clone());
                    let sweep_linked = enricher
                        .post_run_sweep(
                            snap.plan_id,
                            &snap.completed_tasks,
                            snap.started_at,
                            snap.project_id,
                            cwd,
                        )
                        .await;
                    if sweep_linked > 0 {
                        info!(
                            "Post-run sweep linked {} commit→task relations",
                            sweep_linked
                        );
                    }
                }

                self.emit_event(RunnerEvent::PlanCompleted {
                    run_id,
                    plan_id,
                    status: PlanRunStatus::CompletedWithErrors,
                    total_cost_usd: total_cost,
                    total_duration_secs: total_duration,
                    tasks_completed: tc,
                    tasks_failed: tf,
                    pr_url: None, // No auto-PR for partial runs
                });

                warn!(
                    "Plan run {} completed with errors ({} completed, {} failed)",
                    run_id, tc, tf
                );

                // Create a gotcha note with error context
                if plan_id != Uuid::nil() {
                    self.create_failure_note(plan_id, run_id, &snap.failed_tasks, snap.project_id)
                        .await;
                }
            }
            PlanRunStatus::Failed => {
                error!("Plan run {} failed", run_id);

                // Lifecycle FSM: fire execution_failed → failed terminal state
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "execution_failed")
                        .await;
                }

                // Create a gotcha note with error context for failed runs
                if plan_id != Uuid::nil() {
                    self.create_failure_note(plan_id, run_id, &snap.failed_tasks, snap.project_id)
                        .await;
                }
            }
            PlanRunStatus::Cancelled => {
                info!("Plan run {} cancelled", run_id);

                // Lifecycle FSM: fire execution_failed for cancelled runs too
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "execution_failed")
                        .await;
                }
            }
            PlanRunStatus::BudgetExceeded => {
                warn!("Plan run {} aborted: budget exceeded", run_id);

                // Lifecycle FSM: fire execution_failed for budget exceeded
                if let Some(lc_run_id) = lifecycle_run_id {
                    self.fire_lifecycle_transition(run_id, lc_run_id, "execution_failed")
                        .await;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Fire a lifecycle protocol transition and emit a WebSocket event.
    ///
    /// Non-fatal: logs warnings on failure but never propagates errors.
    /// This ensures the runner completes even if the lifecycle protocol
    /// has issues (graceful degradation).
    async fn fire_lifecycle_transition(&self, run_id: Uuid, lifecycle_run_id: Uuid, trigger: &str) {
        // Capture current state name before transition
        let from_state = match self.graph.get_protocol_run(lifecycle_run_id).await {
            Ok(Some(run)) => run
                .states_visited
                .last()
                .map(|v| v.state_name.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            _ => "unknown".to_string(),
        };

        match crate::protocol::engine::fire_transition(
            self.graph.as_ref(),
            lifecycle_run_id,
            trigger,
        )
        .await
        {
            Ok(result) if result.success => {
                info!(
                    "Lifecycle transition: {} → {} (trigger: {})",
                    from_state, result.current_state_name, trigger
                );
                self.emit_event(RunnerEvent::LifecycleTransition {
                    run_id,
                    lifecycle_run_id,
                    from_state,
                    to_state: result.current_state_name,
                    trigger: trigger.to_string(),
                });
            }
            Ok(result) => {
                warn!(
                    "Lifecycle transition failed (non-fatal): trigger='{}', error={:?}",
                    trigger, result.error
                );
            }
            Err(e) => {
                warn!(
                    "Lifecycle transition error (non-fatal): trigger='{}', error={}",
                    trigger, e
                );
            }
        }
    }

    /// Evaluate the PR guard for the lifecycle protocol's pr_decision state.
    ///
    /// Returns true if:
    /// 1. The plan has commits linked (has_commits > 0)
    /// 2. AND is an implementation plan:
    ///    - Task tags contain feat/fix/refactor, OR
    ///    - Tasks have non-empty affected_files
    async fn evaluate_pr_guard(&self, plan_id: Uuid) -> bool {
        // Check has_commits
        let has_commits = match self.graph.get_plan_commits(plan_id).await {
            Ok(commits) => !commits.is_empty(),
            Err(e) => {
                warn!("Failed to get plan commits for PR guard: {}", e);
                false
            }
        };

        if !has_commits {
            return false;
        }

        // Check is_implementation: task tags contain feat/fix/refactor OR affected_files non-empty
        let is_implementation = match self.graph.get_plan_tasks(plan_id).await {
            Ok(tasks) => {
                let implementation_tags = ["feat", "fix", "refactor"];
                tasks.iter().any(|t| {
                    // Check tags
                    let has_impl_tag = t.tags.iter().any(|tag| {
                        implementation_tags
                            .iter()
                            .any(|it| tag.to_lowercase().contains(it))
                    });
                    // Check affected_files
                    let has_files = !t.affected_files.is_empty();
                    has_impl_tag || has_files
                })
            }
            Err(e) => {
                warn!("Failed to get plan tasks for PR guard: {}", e);
                false
            }
        };

        is_implementation
    }

    /// Create a gotcha note when a plan run fails.
    ///
    /// Captures error context (failed tasks, run_id) for debugging.
    async fn create_failure_note(
        &self,
        plan_id: Uuid,
        run_id: Uuid,
        failed_tasks: &[Uuid],
        project_id: Option<Uuid>,
    ) {
        let failed_summary: Vec<String> = failed_tasks
            .iter()
            .map(|id| format!("- Task {}", id))
            .collect();

        let content = format!(
            "## Plan run failed\n\n\
             **Run ID**: {}\n\
             **Plan ID**: {}\n\n\
             ### Failed tasks\n{}\n",
            run_id,
            plan_id,
            if failed_summary.is_empty() {
                "No task-level failures recorded (runner-level error).".to_string()
            } else {
                failed_summary.join("\n")
            }
        );

        let mut note = crate::notes::models::Note::new(
            project_id,
            crate::notes::models::NoteType::Gotcha,
            content,
            "runner".to_string(),
        );
        note.importance = crate::notes::models::NoteImportance::High;
        note.tags = vec!["runner-failure".to_string(), "auto-generated".to_string()];

        if let Err(e) = self.graph.create_note(&note).await {
            warn!("Failed to create failure note (non-fatal): {}", e);
        } else {
            info!("Created gotcha note {} for failed run {}", note.id, run_id);
        }
    }

    /// Create a dedicated git branch for this plan run.
    ///
    /// Branch format: `runner/<plan-slug>-<short-run-id>`.
    /// Falls back gracefully if git commands fail (e.g., dirty worktree).
    async fn create_git_branch(&self, plan_id: Uuid, run_id: Uuid, cwd: &str) -> Option<String> {
        // Get plan title for branch naming
        let plan_title = match self.graph.get_plan(plan_id).await {
            Ok(Some(plan)) => plan.title,
            _ => format!("plan-{}", &plan_id.to_string()[..8]),
        };

        // Slugify: lowercase, replace non-alphanum with hyphens, trim
        let slug: String = plan_title
            .to_lowercase()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect::<String>()
            .trim_matches('-')
            .to_string();

        // Truncate slug and add short run id
        let slug = if slug.len() > 50 { &slug[..50] } else { &slug };
        let short_run = &run_id.to_string()[..8];
        let branch_name = format!("runner/{}-{}", slug, short_run);

        // Create and checkout the branch
        let output = tokio::process::Command::new("git")
            .args(["checkout", "-b", &branch_name])
            .current_dir(cwd)
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                info!("Created git branch: {}", branch_name);
                Some(branch_name)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                warn!(
                    "Failed to create git branch '{}': {}. Running on current branch.",
                    branch_name, stderr
                );
                None
            }
            Err(e) => {
                warn!("Git command failed: {}. Running on current branch.", e);
                None
            }
        }
    }

    /// Listen for chat events until a Result event is received.
    ///
    /// Extracted from execute_task to allow the guard to run in parallel.
    async fn listen_for_result(
        &self,
        mut rx: broadcast::Receiver<ChatEvent>,
        _run_id: Uuid,
        task_id: Option<Uuid>,
    ) -> (EventListenResult, EventMetrics) {
        let start = std::time::Instant::now();
        // Use a generous timeout here — the guard handles the actual task_timeout
        // with soft hints before hard stop. This is just a safety net.
        let safety_timeout = std::time::Duration::from_secs(self.config.task_timeout_secs + 60);
        let mut cost_usd = 0.0_f64;
        let mut metrics = EventMetrics::default();

        loop {
            let remaining = safety_timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return (
                    EventListenResult::Completed {
                        cost_usd,
                        is_error: true,
                        error_text: "Safety timeout exceeded".to_string(),
                        subtype: "timeout".to_string(),
                    },
                    metrics,
                );
            }

            // Check cancel flag
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                return (EventListenResult::Cancelled { cost_usd }, metrics);
            }

            match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv()).await {
                Ok(Ok(event)) => {
                    match &event {
                        ChatEvent::ToolUse { tool, .. } => {
                            metrics.tool_use_count += 1;
                            *metrics.tool_use_breakdown.entry(tool.clone()).or_insert(0) += 1;
                        }
                        ChatEvent::ToolResult {
                            is_error, result, ..
                        } if *is_error => {
                            metrics.error_count += 1;
                            // Extract error text from the result value
                            let err_text = result
                                .as_str()
                                .map(|s| s.to_string())
                                .or_else(|| {
                                    result
                                        .get("error")
                                        .and_then(|e| e.as_str())
                                        .map(|s| s.to_string())
                                })
                                .unwrap_or_else(|| result.to_string());
                            // Keep last 500 chars to avoid bloat
                            metrics.last_error = Some(if err_text.len() > 500 {
                                err_text[err_text.len() - 500..].to_string()
                            } else {
                                err_text
                            });
                        }
                        ChatEvent::Result {
                            cost_usd: event_cost,
                            subtype,
                            is_error,
                            result_text,
                            ..
                        } => {
                            if let Some(c) = event_cost {
                                cost_usd = *c;
                                // Update agent cost in real-time for run_status
                                if let Some(tid) = task_id {
                                    let mut global = RUNNER_STATE.write().await;
                                    if let Some(ref mut s) = *global {
                                        s.add_agent_cost(&tid, *c);
                                    }
                                }
                            }
                            return (
                                EventListenResult::Completed {
                                    cost_usd,
                                    is_error: *is_error,
                                    error_text: result_text.clone().unwrap_or_default(),
                                    subtype: subtype.clone(),
                                },
                                metrics,
                            );
                        }
                        _ => {}
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    warn!("Runner event receiver lagged by {} events", n);
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => {
                    return (EventListenResult::ChannelClosed { cost_usd }, metrics);
                }
                Err(_) => {
                    // 5s poll timeout — just loop again (check cancel flag)
                }
            }
        }
    }
}

// ============================================================================
// EventListenResult — internal result type for listen_for_result
// ============================================================================

/// Internal result of the event listening loop.
enum EventListenResult {
    /// A Result event was received from the agent.
    Completed {
        cost_usd: f64,
        is_error: bool,
        error_text: String,
        subtype: String,
    },
    /// The broadcast channel was closed unexpectedly.
    ChannelClosed { cost_usd: f64 },
    /// The run was cancelled by the user.
    Cancelled { cost_usd: f64 },
}

/// Metrics collected from ChatEvents during task execution.
///
/// Populated by `listen_for_result` while waiting for the final Result event.
/// Used to build [`TaskExecutionReport`] after task completion.
#[derive(Debug, Default)]
struct EventMetrics {
    /// Total tool_use events
    tool_use_count: u32,
    /// Per-tool breakdown
    tool_use_breakdown: std::collections::HashMap<String, u32>,
    /// Number of tool_result events that indicate errors
    error_count: u32,
    /// Last error text from a tool_result
    last_error: Option<String>,
}

// ============================================================================
// Trait impls
// ============================================================================

impl Clone for PlanRunner {
    fn clone(&self) -> Self {
        Self {
            chat_manager: self.chat_manager.clone(),
            graph: self.graph.clone(),
            context_builder: self.context_builder.clone(),
            config: self.config.clone(),
            event_tx: self.event_tx.clone(),
            event_emitter: self.event_emitter.clone(),
            user_claims: self.user_claims.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::models::TriggerSource;

    /// Mutex to serialize tests that touch global RUNNER_STATE / RUNNER_CANCEL.
    /// tokio::sync::Mutex is used because these are async tests.
    static TEST_MUTEX: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

    /// Reset global state to a clean baseline.
    async fn reset_globals() {
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
        RUNNER_CANCEL.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn test_status_when_no_run() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let status = PlanRunner::status().await;
        assert!(!status.running);
        assert!(status.run_id.is_none());
        assert_eq!(status.progress_pct, 0.0);
    }

    #[tokio::test]
    async fn test_status_when_running() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, 5, TriggerSource::Manual);
        {
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }
        let status = PlanRunner::status().await;
        assert!(status.running);
        assert_eq!(status.run_id, Some(run_id));
        assert_eq!(status.plan_id, Some(plan_id));
        assert_eq!(status.tasks_total, 5);
        assert_eq!(status.tasks_completed, 0);
        reset_globals().await;
    }

    #[tokio::test]
    async fn test_cancel_no_active_run() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let result = PlanRunner::cancel(Uuid::new_v4(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No active run"));
    }

    #[tokio::test]
    async fn test_cancel_sets_flag() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, 3, TriggerSource::Manual);
        {
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }

        let result = PlanRunner::cancel(run_id, None).await;
        assert!(result.is_ok());
        assert!(RUNNER_CANCEL.load(Ordering::SeqCst));
        reset_globals().await;
    }

    #[tokio::test]
    async fn test_cancel_wrong_run_id() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, 3, TriggerSource::Manual);
        {
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }

        let result = PlanRunner::cancel(Uuid::new_v4(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not running"));
        reset_globals().await;
    }

    #[test]
    fn test_runner_constraints_in_prompt() {
        let ctx = RunnerPromptContext::single_agent(String::new());
        let constraints = build_runner_constraints(&ctx);
        assert!(constraints.contains("## Runner Execution Mode"));
        assert!(constraints.contains("autonomous code execution agent"));
        assert!(constraints.contains("DO NOT"));
        assert!(constraints.contains("task(action: \"update\", status"));
        assert!(constraints.contains(".env"));
        assert!(constraints.contains("cargo check"));
    }

    // ---------------------------------------------------------------
    // RunMemory / TaskSummary unit tests
    // ---------------------------------------------------------------

    #[test]
    fn test_run_memory_to_markdown_empty() {
        let mem = RunMemory::default();
        assert_eq!(mem.to_markdown(), "");
    }

    #[test]
    fn test_run_memory_to_markdown_single_completed() {
        let mut mem = RunMemory::default();
        mem.record_completed(
            Uuid::new_v4(),
            "Setup CI".into(),
            1,
            vec!["abc1234".into()],
            vec!["ci.yml".into()],
        );
        let md = mem.to_markdown();
        assert!(md.contains("## Previous Tasks (Continuity Context)"));
        assert!(md.contains("### Wave 1 — Setup CI (completed)"));
        assert!(md.contains("**Commits:**"));
        assert!(md.contains("- `abc1234`"));
        assert!(md.contains("**Files modified:**"));
        assert!(md.contains("- `ci.yml`"));
    }

    #[test]
    fn test_run_memory_to_markdown_multiple_waves() {
        let mut mem = RunMemory::default();
        mem.record_completed(Uuid::new_v4(), "Task A".into(), 1, vec![], vec![]);
        mem.record_completed(
            Uuid::new_v4(),
            "Task B".into(),
            2,
            vec!["def5678".into()],
            vec!["src/main.rs".into(), "Cargo.toml".into()],
        );
        let md = mem.to_markdown();
        assert!(md.contains("### Wave 1 — Task A (completed)"));
        assert!(md.contains("### Wave 2 — Task B (completed)"));
        // Task A has no commits/files — sections should be absent for it
        // but present for Task B
        assert!(md.contains("- `def5678`"));
        assert!(md.contains("- `src/main.rs`"));
        assert!(md.contains("- `Cargo.toml`"));
    }

    #[test]
    fn test_run_memory_to_markdown_failed_task() {
        let mut mem = RunMemory::default();
        mem.record_failed(Uuid::new_v4(), "Broken task".into(), 1, "compilation error");
        let md = mem.to_markdown();
        assert!(md.contains("### Wave 1 — Broken task (failed: compilation error)"));
        // Failed tasks should NOT have commits/files sections
        assert!(!md.contains("**Commits:**"));
        assert!(!md.contains("**Files modified:**"));
    }

    #[test]
    fn test_run_memory_to_markdown_omits_empty_commits_and_files() {
        let mut mem = RunMemory::default();
        mem.record_completed(Uuid::new_v4(), "No artifacts".into(), 1, vec![], vec![]);
        let md = mem.to_markdown();
        assert!(md.contains("### Wave 1 — No artifacts (completed)"));
        assert!(!md.contains("**Commits:**"));
        assert!(!md.contains("**Files modified:**"));
    }

    #[test]
    fn test_run_memory_record_completed_stores_fields() {
        let task_id = Uuid::new_v4();
        let mut mem = RunMemory::default();
        mem.record_completed(
            task_id,
            "My task".into(),
            3,
            vec!["c1".into(), "c2".into()],
            vec!["f1.rs".into()],
        );
        assert_eq!(mem.summaries.len(), 1);
        let s = &mem.summaries[0];
        assert_eq!(s.task_id, task_id);
        assert_eq!(s.title, "My task");
        assert_eq!(s.wave_number, 3);
        assert_eq!(s.status, "completed");
        assert_eq!(s.commits, vec!["c1", "c2"]);
        assert_eq!(s.files_modified, vec!["f1.rs"]);
    }

    #[test]
    fn test_run_memory_record_failed_truncates_reason() {
        let mut mem = RunMemory::default();
        let long_reason = "x".repeat(300);
        mem.record_failed(Uuid::new_v4(), "Fail".into(), 1, &long_reason);
        let s = &mem.summaries[0];
        // status = "failed: " (8 chars) + truncated reason (200 chars) = 208 chars
        assert_eq!(s.status.len(), 208);
        assert!(s.status.starts_with("failed: "));
        assert!(s.status.ends_with("xxxx"));
    }

    #[test]
    fn test_run_memory_record_failed_short_reason_not_truncated() {
        let mut mem = RunMemory::default();
        mem.record_failed(Uuid::new_v4(), "Fail".into(), 1, "short reason");
        let s = &mem.summaries[0];
        assert_eq!(s.status, "failed: short reason");
    }

    #[test]
    fn test_run_status_default() {
        let status = RunStatus {
            running: false,
            run_id: None,
            plan_id: None,
            status: None,
            current_wave: None,
            current_task_id: None,
            current_task_title: None,
            active_agents: Vec::new(),
            progress_pct: 0.0,
            tasks_completed: 0,
            tasks_total: 0,
            elapsed_secs: 0.0,
            cost_usd: 0.0,
            max_cost_usd: 0.0,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"running\":false"));
        assert!(json.contains("\"active_agents\":[]"));
    }

    // ---------------------------------------------------------------
    // Helper: build a PlanRunner with mock stores
    // ---------------------------------------------------------------

    fn test_plan_runner() -> PlanRunner {
        use crate::chat::config::ChatConfig;
        use crate::chat::manager::ChatManager;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::meilisearch::traits::SearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::notes::manager::NoteManager;
        use crate::orchestrator::context::ContextBuilder;
        use crate::plan::manager::PlanManager;

        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let search: Arc<dyn SearchStore> = Arc::new(MockSearchStore::new());
        let chat_config = ChatConfig {
            mcp_server_path: std::path::PathBuf::from("/dev/null"),
            default_model: "test".into(),
            max_sessions: 1,
            session_timeout: std::time::Duration::from_secs(10),
            neo4j_uri: "bolt://mock:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "test".into(),
            meilisearch_url: "http://mock:7700".into(),
            meilisearch_key: "test".into(),
            nats_url: None,
            max_turns: 5,
            permission: Default::default(),
            auto_continue: false,
            retry: Default::default(),
            process_path: None,
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: false,
            jwt_secret: None,
            server_port: 0,
            session_token_expiry_secs: 3600,
        };
        let chat_manager = Arc::new(ChatManager::new_without_memory(
            graph.clone(),
            search.clone(),
            chat_config,
        ));
        let plan_manager = Arc::new(PlanManager::new(graph.clone(), search.clone()));
        let note_manager = Arc::new(NoteManager::new(graph.clone(), search.clone()));
        let context_builder = Arc::new(ContextBuilder::new(
            graph.clone(),
            search.clone(),
            plan_manager,
            note_manager,
        ));
        let (event_tx, _) = broadcast::channel(16);
        PlanRunner::new(
            chat_manager,
            graph,
            context_builder,
            RunnerConfig::default(),
            event_tx,
        )
    }

    // ---------------------------------------------------------------
    // EventMetrics tests
    // ---------------------------------------------------------------

    #[test]
    fn test_event_metrics_default() {
        let m = EventMetrics::default();
        assert_eq!(m.tool_use_count, 0);
        assert!(m.tool_use_breakdown.is_empty());
        assert_eq!(m.error_count, 0);
        assert!(m.last_error.is_none());
    }

    #[test]
    fn test_event_metrics_mutation() {
        let mut breakdown = std::collections::HashMap::new();
        breakdown.insert("Edit".to_string(), 3u32);
        breakdown.insert("Bash".to_string(), 2u32);
        let m = EventMetrics {
            tool_use_count: 5,
            tool_use_breakdown: breakdown,
            error_count: 1,
            last_error: Some("something broke".into()),
        };

        assert_eq!(m.tool_use_count, 5);
        assert_eq!(m.tool_use_breakdown.len(), 2);
        assert_eq!(m.tool_use_breakdown["Edit"], 3);
        assert_eq!(m.tool_use_breakdown["Bash"], 2);
        assert_eq!(m.error_count, 1);
        assert_eq!(m.last_error.as_deref(), Some("something broke"));
    }

    // ---------------------------------------------------------------
    // listen_for_result tests
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_listen_for_result_completed() {
        let runner = test_plan_runner();
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let run_id = Uuid::new_v4();

        // Send a ToolUse, then a Result
        tx.send(ChatEvent::ToolUse {
            id: "tu1".into(),
            tool: "Edit".into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::ToolUse {
            id: "tu2".into(),
            tool: "Bash".into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::ToolUse {
            id: "tu3".into(),
            tool: "Edit".into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::Result {
            session_id: "s1".into(),
            duration_ms: 1000,
            cost_usd: Some(0.05),
            subtype: "success".into(),
            is_error: false,
            num_turns: Some(3),
            result_text: None,
        })
        .unwrap();

        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        let (result, metrics) = runner.listen_for_result(rx, run_id, None).await;

        assert_eq!(metrics.tool_use_count, 3);
        assert_eq!(metrics.tool_use_breakdown["Edit"], 2);
        assert_eq!(metrics.tool_use_breakdown["Bash"], 1);
        assert_eq!(metrics.error_count, 0);
        assert!(metrics.last_error.is_none());

        match result {
            EventListenResult::Completed {
                cost_usd,
                is_error,
                subtype,
                ..
            } => {
                assert!((cost_usd - 0.05).abs() < f64::EPSILON);
                assert!(!is_error);
                assert_eq!(subtype, "success");
            }
            _ => panic!("Expected Completed variant"),
        }
    }

    #[tokio::test]
    async fn test_listen_for_result_tracks_errors() {
        let runner = test_plan_runner();
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let run_id = Uuid::new_v4();

        tx.send(ChatEvent::ToolResult {
            id: "tr1".into(),
            result: serde_json::json!("first error message"),
            is_error: true,
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::ToolResult {
            id: "tr2".into(),
            result: serde_json::json!({"error": "second error"}),
            is_error: true,
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::Result {
            session_id: "s1".into(),
            duration_ms: 500,
            cost_usd: Some(0.01),
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: None,
        })
        .unwrap();

        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        let (_result, metrics) = runner.listen_for_result(rx, run_id, None).await;

        assert_eq!(metrics.error_count, 2);
        // last_error should be from the second error event
        assert_eq!(metrics.last_error.as_deref(), Some("second error"));
    }

    #[tokio::test]
    async fn test_listen_for_result_error_truncation() {
        let runner = test_plan_runner();
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let run_id = Uuid::new_v4();

        let long_error = "x".repeat(1000);
        tx.send(ChatEvent::ToolResult {
            id: "tr1".into(),
            result: serde_json::json!(long_error),
            is_error: true,
            parent_tool_use_id: None,
        })
        .unwrap();

        tx.send(ChatEvent::Result {
            session_id: "s1".into(),
            duration_ms: 100,
            cost_usd: None,
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: None,
        })
        .unwrap();

        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        let (_result, metrics) = runner.listen_for_result(rx, run_id, None).await;

        assert_eq!(metrics.error_count, 1);
        // Error text should be truncated to 500 chars
        assert_eq!(metrics.last_error.as_ref().unwrap().len(), 500);
    }

    #[tokio::test]
    async fn test_listen_for_result_channel_closed() {
        let runner = test_plan_runner();
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let run_id = Uuid::new_v4();

        // Drop sender to close channel
        drop(tx);

        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        let (result, metrics) = runner.listen_for_result(rx, run_id, None).await;

        assert_eq!(metrics.tool_use_count, 0);
        match result {
            EventListenResult::ChannelClosed { cost_usd } => {
                assert!((cost_usd - 0.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected ChannelClosed variant"),
        }
    }

    #[tokio::test]
    async fn test_listen_for_result_cancelled() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let runner = test_plan_runner();
        let (_tx, rx) = broadcast::channel::<ChatEvent>(16);
        let run_id = Uuid::new_v4();

        // Set cancel flag BEFORE listening
        RUNNER_CANCEL.store(true, Ordering::SeqCst);
        let (result, _metrics) = runner.listen_for_result(rx, run_id, None).await;

        match result {
            EventListenResult::Cancelled { cost_usd } => {
                assert!((cost_usd - 0.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Cancelled variant"),
        }
        reset_globals().await;
    }

    // ---------------------------------------------------------------
    // create_failure_gotcha tests
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_create_failure_gotcha_creates_note() {
        let runner = test_plan_runner();
        let task_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();

        runner
            .create_failure_gotcha(
                task_id,
                "Build CI pipeline",
                "compilation failed",
                2,
                Some(project_id),
            )
            .await;

        // The mock graph store should have received the create_note call.
        // Use list_notes to retrieve notes from the mock store.
        let (notes, count) = runner
            .graph
            .list_notes(
                Some(project_id),
                None,
                &crate::notes::NoteFilters::default(),
            )
            .await
            .unwrap();

        assert_eq!(count, 1);
        assert_eq!(notes.len(), 1);
        let note = &notes[0];
        assert!(note.content.contains("Build CI pipeline"));
        assert!(note.content.contains("compilation failed"));
        assert!(note.content.contains("2 Retry(ies)"));
        assert_eq!(note.note_type, crate::notes::NoteType::Gotcha);
        assert_eq!(note.importance, crate::notes::NoteImportance::High);
        assert!(note.tags.contains(&"auto-generated".to_string()));
        assert!(note.tags.contains(&"task-failure".to_string()));
        assert!(note.tags.contains(&"retry-exhausted".to_string()));
        assert!(note.tags.contains(&format!("task:{}", task_id)));
    }

    #[tokio::test]
    async fn test_create_failure_gotcha_without_project() {
        let runner = test_plan_runner();
        let task_id = Uuid::new_v4();

        // Should not panic even with project_id = None
        runner
            .create_failure_gotcha(task_id, "Task title", "some error", 1, None)
            .await;
    }

    // ---------------------------------------------------------------
    // on_task_failed tests
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_on_task_failed_should_retry() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let runner = test_plan_runner();
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Set up RUNNER_STATE with a state that has 0 retries for this task
        {
            let state = RunnerState::new(run_id, plan_id, 3, TriggerSource::Manual);
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }

        // RunnerConfig default max_retries = 1, so retry_count 0 < 1 → should retry
        let result = runner
            .on_task_failed(run_id, plan_id, task_id, "test error", 5.0, 0.01)
            .await
            .unwrap();

        assert!(result, "Should return true when retries are available");

        // Verify retry count was incremented
        {
            let global = RUNNER_STATE.read().await;
            let s = global.as_ref().unwrap();
            assert_eq!(s.retry_counts.get(&task_id).copied(), Some(1));
        }
        reset_globals().await;
    }

    #[tokio::test]
    async fn test_on_task_failed_no_retry_when_exhausted() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let runner = test_plan_runner();
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Set up RUNNER_STATE with retry_count already at max_retries (1)
        {
            let mut state = RunnerState::new(run_id, plan_id, 3, TriggerSource::Manual);
            state.retry_counts.insert(task_id, 1); // already at max
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }

        // retry_count 1 >= max_retries 1 → should NOT retry
        let result = runner
            .on_task_failed(run_id, plan_id, task_id, "final failure", 10.0, 0.05)
            .await
            .unwrap();

        assert!(!result, "Should return false when retries exhausted");
        reset_globals().await;
    }

    #[tokio::test]
    async fn test_on_task_failed_no_retry_without_state() {
        let _lock = TEST_MUTEX.lock().await;
        reset_globals().await;
        let runner = test_plan_runner();
        let result = runner
            .on_task_failed(
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                "err",
                1.0,
                0.0,
            )
            .await
            .unwrap();

        assert!(!result, "Should return false when no runner state");
        reset_globals().await;
    }

    // ---------------------------------------------------------------
    // WaveResult tests
    // ---------------------------------------------------------------

    #[test]
    fn test_wave_result_empty() {
        let wr = WaveResult {
            tasks_completed: vec![],
            tasks_failed: vec![],
            wave_cost_usd: 0.0,
            aborted: false,
        };
        assert!(wr.tasks_completed.is_empty());
        assert!(wr.tasks_failed.is_empty());
        assert!(!wr.aborted);
        assert!((wr.wave_cost_usd - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wave_result_with_completed_and_failed() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let wr = WaveResult {
            tasks_completed: vec![id1, id2],
            tasks_failed: vec![(id3, "timeout".to_string())],
            wave_cost_usd: 1.23,
            aborted: false,
        };
        assert_eq!(wr.tasks_completed.len(), 2);
        assert_eq!(wr.tasks_failed.len(), 1);
        assert_eq!(wr.tasks_failed[0].0, id3);
        assert_eq!(wr.tasks_failed[0].1, "timeout");
        assert!((wr.wave_cost_usd - 1.23).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wave_result_aborted() {
        let wr = WaveResult {
            tasks_completed: vec![Uuid::new_v4()],
            tasks_failed: vec![],
            wave_cost_usd: 5.0,
            aborted: true,
        };
        assert!(wr.aborted);
    }

    // ---------------------------------------------------------------
    // TaskExecutionResult tests
    // ---------------------------------------------------------------

    #[test]
    fn test_task_execution_result_session_id() {
        use crate::runner::models::TaskResult;

        let session = Uuid::new_v4();
        let exec_id = Uuid::new_v4();
        let ter = TaskExecutionResult {
            result: TaskResult::Success {
                cost_usd: 0.05,
                duration_secs: 30.0,
            },
            session_id: Some(session),
            activated_skill_ids: vec![],
            persona_ids: vec![],
            agent_execution_id: exec_id,
            persona_profile: "test-persona".into(),
            report: None,
        };
        assert_eq!(ter.session_id(), Some(session));
    }

    #[test]
    fn test_task_execution_result_no_session() {
        use crate::runner::models::TaskResult;

        let ter = TaskExecutionResult {
            result: TaskResult::Failed {
                reason: "test failure".into(),
                attempts: 2,
                cost_usd: 0.1,
            },
            session_id: None,
            activated_skill_ids: vec![Uuid::new_v4()],
            persona_ids: vec![Uuid::new_v4()],
            agent_execution_id: Uuid::new_v4(),
            persona_profile: String::new(),
            report: None,
        };
        assert_eq!(ter.session_id(), None);
    }

    #[test]
    fn test_task_execution_result_with_report() {
        use crate::runner::models::{TaskExecutionReport, TaskResult};

        let report = TaskExecutionReport {
            tool_use_count: 10,
            tool_use_breakdown: {
                let mut m = std::collections::HashMap::new();
                m.insert("Edit".into(), 7);
                m.insert("Bash".into(), 3);
                m
            },
            error_count: 1,
            last_error: Some("compile error".into()),
            files_modified: vec!["src/main.rs".into()],
            commits: vec!["abc1234".into()],
            agent_success: true,
            cost_usd: 0.05,
            duration_secs: 45.0,
            confidence_score: 0.85,
        };

        let ter = TaskExecutionResult {
            result: TaskResult::Success {
                cost_usd: 0.05,
                duration_secs: 45.0,
            },
            session_id: Some(Uuid::new_v4()),
            activated_skill_ids: vec![],
            persona_ids: vec![],
            agent_execution_id: Uuid::new_v4(),
            persona_profile: "dev".into(),
            report: Some(report),
        };

        assert!(ter.report.is_some());
        let r = ter.report.as_ref().unwrap();
        assert_eq!(r.tool_use_count, 10);
        assert_eq!(r.error_count, 1);
        assert!(r.agent_success);
        assert!((r.confidence_score - 0.85).abs() < f64::EPSILON);
    }

    // ---------------------------------------------------------------
    // finalize_steps tests
    // ---------------------------------------------------------------

    /// Helper: build a PlanRunner that shares a MockGraphStore we can seed
    fn test_plan_runner_with_graph() -> (PlanRunner, Arc<crate::neo4j::mock::MockGraphStore>) {
        use crate::chat::config::ChatConfig;
        use crate::chat::manager::ChatManager;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::meilisearch::traits::SearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::notes::manager::NoteManager;
        use crate::orchestrator::context::ContextBuilder;
        use crate::plan::manager::PlanManager;

        let mock_graph = Arc::new(MockGraphStore::new());
        let graph: Arc<dyn GraphStore> = mock_graph.clone();
        let search: Arc<dyn SearchStore> = Arc::new(MockSearchStore::new());
        let chat_config = ChatConfig {
            mcp_server_path: std::path::PathBuf::from("/dev/null"),
            default_model: "test".into(),
            max_sessions: 1,
            session_timeout: std::time::Duration::from_secs(10),
            neo4j_uri: "bolt://mock:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "test".into(),
            meilisearch_url: "http://mock:7700".into(),
            meilisearch_key: "test".into(),
            nats_url: None,
            max_turns: 5,
            permission: Default::default(),
            auto_continue: false,
            retry: Default::default(),
            process_path: None,
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: false,
            jwt_secret: None,
            server_port: 0,
            session_token_expiry_secs: 3600,
        };
        let chat_manager = Arc::new(ChatManager::new_without_memory(
            graph.clone(),
            search.clone(),
            chat_config,
        ));
        let plan_manager = Arc::new(PlanManager::new(graph.clone(), search.clone()));
        let note_manager = Arc::new(NoteManager::new(graph.clone(), search.clone()));
        let context_builder = Arc::new(ContextBuilder::new(
            graph.clone(),
            search.clone(),
            plan_manager,
            note_manager,
        ));
        let (event_tx, _) = broadcast::channel(16);
        let runner = PlanRunner::new(
            chat_manager,
            graph,
            context_builder,
            RunnerConfig::default(),
            event_tx,
        );
        (runner, mock_graph)
    }

    #[tokio::test]
    async fn test_finalize_steps_on_success() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // 2 pending steps + 1 already completed
        let mut step1 = test_step(1, "Write code");
        step1.status = StepStatus::Pending;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Run tests");
        step2.status = StepStatus::InProgress;
        graph.create_step(task_id, &step2).await.unwrap();

        let mut step3 = test_step(3, "Already done");
        step3.status = StepStatus::Completed;
        graph.create_step(task_id, &step3).await.unwrap();

        // finalize_steps with Success outcome
        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 30.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        // Verify breakdown
        assert_eq!(breakdown.total, 3);
        assert_eq!(breakdown.completed, 3); // 1 already + 2 finalized
        assert_eq!(breakdown.skipped, 0);
        assert_eq!(breakdown.pending, 0);

        // Verify: pending and in_progress → completed, already completed stays
        let steps = graph.get_task_steps(task_id).await.unwrap();
        for step in &steps {
            assert_eq!(
                step.status,
                StepStatus::Completed,
                "Step '{}' should be Completed, got {:?}",
                step.description,
                step.status
            );
        }
    }

    #[tokio::test]
    async fn test_finalize_steps_on_error_marks_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let step1 = test_step(1, "Write code");
        graph.create_step(task_id, &step1).await.unwrap();

        let step2 = test_step(2, "Run tests");
        graph.create_step(task_id, &step2).await.unwrap();

        // finalize_steps with Failed outcome
        let result = TaskResult::Failed {
            reason: "Agent crashed".to_string(),
            attempts: 1,
            cost_usd: 0.05,
        };
        runner.finalize_steps(task_id, &result, "/tmp").await;

        // Verify: all pending → skipped
        let steps = graph.get_task_steps(task_id).await.unwrap();
        for step in &steps {
            assert_eq!(
                step.status,
                StepStatus::Skipped,
                "Step '{}' should be Skipped on error, got {:?}",
                step.description,
                step.status
            );
        }
    }

    #[tokio::test]
    async fn test_finalize_steps_on_budget_exceeded_marks_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let step1 = test_step(1, "Pending step");
        graph.create_step(task_id, &step1).await.unwrap();

        let result = TaskResult::BudgetExceeded {
            cumulated_cost_usd: 10.0,
            limit_usd: 5.0,
        };
        runner.finalize_steps(task_id, &result, "/tmp").await;

        let steps = graph.get_task_steps(task_id).await.unwrap();
        assert_eq!(steps[0].status, StepStatus::Skipped);
    }

    #[tokio::test]
    async fn test_finalize_steps_noop_when_all_completed() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Already done");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        // finalize_steps should be a no-op (all already completed)
        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 10.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        assert_eq!(breakdown.total, 1);
        assert_eq!(breakdown.completed, 1);
        assert_eq!(breakdown.skipped, 0);
        assert_eq!(breakdown.pending, 0);

        let steps = graph.get_task_steps(task_id).await.unwrap();
        assert_eq!(steps[0].status, StepStatus::Completed);
    }

    // === Step Completion Guard Tests ===

    #[tokio::test]
    async fn test_step_guard_some_completed_returns_ok() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // 1 completed + 1 skipped + 1 pending
        let mut step1 = test_step(1, "Completed step");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Skipped step");
        step2.status = StepStatus::Skipped;
        graph.create_step(task_id, &step2).await.unwrap();

        let step3 = test_step(3, "Pending step");
        graph.create_step(task_id, &step3).await.unwrap();

        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 30.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        // Guard should NOT trigger: 1 completed before + 1 finalized = 2 completed
        assert_eq!(breakdown.total, 3);
        assert!(
            breakdown.completed >= 1,
            "At least 1 step should be completed"
        );
        // This means the guard (total > 0 && completed == 0) would be false → task stays Success
    }

    #[tokio::test]
    async fn test_step_guard_all_skipped_triggers() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // All steps already skipped (agent skipped them all)
        let mut step1 = test_step(1, "Skipped step 1");
        step1.status = StepStatus::Skipped;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Skipped step 2");
        step2.status = StepStatus::Skipped;
        graph.create_step(task_id, &step2).await.unwrap();

        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 30.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        // Guard SHOULD trigger: total=2, completed=0, skipped=2
        assert_eq!(breakdown.total, 2);
        assert_eq!(breakdown.completed, 0);
        assert_eq!(breakdown.skipped, 2);
        // In execute_task, this would override Success → Failed
    }

    #[tokio::test]
    async fn test_step_guard_no_steps_returns_ok() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // No steps at all
        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 30.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        // Guard should NOT trigger: total=0, so condition (total > 0 && completed == 0) is false
        assert_eq!(breakdown.total, 0);
        assert_eq!(breakdown.completed, 0);
        // No steps = task with no defined steps → Success is valid
    }

    // === Step coherence tests ===

    #[tokio::test]
    async fn test_check_step_coherence_all_completed_returns_true() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // All steps completed
        let mut step1 = test_step(1, "Write code");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Run tests");
        step2.status = StepStatus::Completed;
        graph.create_step(task_id, &step2).await.unwrap();

        // All steps completed → should return Some(true)
        let result = runner.check_step_coherence(task_id).await;
        assert_eq!(result, Some(true));
    }

    #[tokio::test]
    async fn test_check_step_coherence_all_pending_returns_false() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // All steps still pending (agent did nothing)
        let step1 = test_step(1, "Write code");
        graph.create_step(task_id, &step1).await.unwrap();

        let step2 = test_step(2, "Run tests");
        graph.create_step(task_id, &step2).await.unwrap();

        // All pending → should return Some(false) (suspicious)
        let result = runner.check_step_coherence(task_id).await;
        assert_eq!(result, Some(false));
    }

    #[tokio::test]
    async fn test_check_step_coherence_mixed_returns_none() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Mixed: 1 completed, 1 pending
        let mut step1 = test_step(1, "Write code");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        let step2 = test_step(2, "Run tests");
        graph.create_step(task_id, &step2).await.unwrap();

        // Mixed → should return None (inconclusive)
        let result = runner.check_step_coherence(task_id).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_check_step_coherence_no_steps_returns_none() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // No steps → should return None
        let result = runner.check_step_coherence(task_id).await;
        assert_eq!(result, None);
    }

    // === Pure function tests: validate_cwd ===

    #[test]
    fn test_validate_cwd_no_root_path() {
        let result = validate_cwd("/some/path", None, &CwdValidation::Warn);
        assert_eq!(
            result,
            CwdResolution::NoRootPath {
                resolved_cwd: "/some/path".to_string()
            }
        );
    }

    #[test]
    fn test_validate_cwd_empty_root_path() {
        let result = validate_cwd("/some/path", Some(""), &CwdValidation::Warn);
        assert_eq!(
            result,
            CwdResolution::NoRootPath {
                resolved_cwd: "/some/path".to_string()
            }
        );
    }

    #[test]
    fn test_validate_cwd_dot_fallback_to_root() {
        let result = validate_cwd(".", Some("/tmp"), &CwdValidation::Warn);
        assert!(matches!(result, CwdResolution::FallbackToRoot { .. }));
        if let CwdResolution::FallbackToRoot { resolved_cwd } = result {
            assert_eq!(resolved_cwd, "/tmp");
        }
    }

    #[test]
    fn test_validate_cwd_empty_fallback_to_root() {
        let result = validate_cwd("", Some("/tmp"), &CwdValidation::Warn);
        assert!(matches!(result, CwdResolution::FallbackToRoot { .. }));
    }

    #[test]
    fn test_validate_cwd_matching_paths() {
        // Use /tmp which exists and canonicalizes to itself
        let result = validate_cwd("/tmp", Some("/tmp"), &CwdValidation::Warn);
        assert!(
            matches!(result, CwdResolution::Match { .. }),
            "Expected Match, got {:?}",
            result
        );
    }

    #[test]
    fn test_validate_cwd_mismatch_warn_mode() {
        let result = validate_cwd("/tmp", Some("/var"), &CwdValidation::Warn);
        assert!(
            matches!(result, CwdResolution::Mismatch { .. }),
            "Expected Mismatch, got {:?}",
            result
        );
        if let CwdResolution::Mismatch {
            resolved_cwd,
            root_path,
        } = result
        {
            assert_eq!(resolved_cwd, "/tmp");
            // root_path is the expanded/original value
            assert!(!root_path.is_empty());
        }
    }

    #[test]
    fn test_validate_cwd_mismatch_strict_mode() {
        let result = validate_cwd("/tmp", Some("/var"), &CwdValidation::Strict);
        assert!(
            matches!(result, CwdResolution::StrictMismatch { .. }),
            "Expected StrictMismatch, got {:?}",
            result
        );
        if let CwdResolution::StrictMismatch { cwd, root_path } = result {
            assert_eq!(cwd, "/tmp");
            assert!(!root_path.is_empty());
        }
    }

    #[test]
    fn test_validate_cwd_tilde_expansion() {
        // The ~ should be expanded using $HOME
        let home = std::env::var("HOME").unwrap_or_default();
        if !home.is_empty() {
            let result = validate_cwd(".", Some("~/nonexistent"), &CwdValidation::Warn);
            if let CwdResolution::FallbackToRoot { resolved_cwd } = result {
                assert!(
                    resolved_cwd.starts_with(&home),
                    "Expected path to start with HOME={}, got {}",
                    home,
                    resolved_cwd
                );
                assert!(!resolved_cwd.contains('~'));
            }
        }
    }

    #[test]
    fn test_validate_cwd_nonexistent_paths_still_compare() {
        // Even if paths don't exist, canonicalize falls back to raw comparison
        let result = validate_cwd(
            "/nonexistent/path/a",
            Some("/nonexistent/path/b"),
            &CwdValidation::Warn,
        );
        assert!(
            matches!(result, CwdResolution::Mismatch { .. }),
            "Expected Mismatch for non-existent different paths, got {:?}",
            result
        );
    }

    // === Pure function tests: validate_affected_files ===

    #[test]
    fn test_validate_affected_files_empty_list_is_valid() {
        let (valid, missing) = validate_affected_files(&[], "/tmp");
        assert!(valid);
        assert!(missing.is_empty());
    }

    #[test]
    fn test_validate_affected_files_all_missing() {
        let files = vec![
            "nonexistent/foo.rs".to_string(),
            "nonexistent/bar.rs".to_string(),
        ];
        let (valid, missing) = validate_affected_files(&files, "/tmp");
        assert!(!valid);
        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn test_validate_affected_files_at_least_one_exists() {
        // Use a file that always exists on any system
        let files = vec!["nonexistent/foo.rs".to_string()];
        // /tmp should exist, but none of the relative files do
        let (valid, _) = validate_affected_files(&files, "/tmp");
        assert!(!valid);

        // Now test with an absolute path that exists
        let files_with_existing = vec![
            "/tmp".to_string(), // exists (it's a dir, but Path::exists returns true)
            "nonexistent/bar.rs".to_string(),
        ];
        let (valid2, missing2) = validate_affected_files(&files_with_existing, "/");
        assert!(valid2);
        assert_eq!(missing2.len(), 1);
        assert_eq!(missing2[0], "nonexistent/bar.rs");
    }

    // === Pure function tests: should_step_guard_trigger ===

    #[test]
    fn test_step_guard_trigger_all_skipped() {
        let breakdown = StepBreakdown {
            completed: 0,
            skipped: 3,
            pending: 0,
            total: 3,
        };
        assert!(should_step_guard_trigger(&breakdown));
    }

    #[test]
    fn test_step_guard_trigger_some_completed() {
        let breakdown = StepBreakdown {
            completed: 1,
            skipped: 2,
            pending: 0,
            total: 3,
        };
        assert!(!should_step_guard_trigger(&breakdown));
    }

    #[test]
    fn test_step_guard_trigger_no_steps() {
        let breakdown = StepBreakdown {
            completed: 0,
            skipped: 0,
            pending: 0,
            total: 0,
        };
        assert!(!should_step_guard_trigger(&breakdown));
    }

    #[test]
    fn test_step_guard_trigger_all_pending() {
        let breakdown = StepBreakdown {
            completed: 0,
            skipped: 0,
            pending: 5,
            total: 5,
        };
        assert!(should_step_guard_trigger(&breakdown));
    }

    #[test]
    fn test_step_guard_trigger_default_breakdown() {
        let breakdown = StepBreakdown::default();
        assert!(!should_step_guard_trigger(&breakdown));
    }

    // === emit_event tests for new variants ===

    #[test]
    fn test_emit_event_new_variants_broadcast() {
        let (event_tx, mut rx) = broadcast::channel(16);
        // We just test that new event variants can be sent through the channel
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        let events = vec![
            RunnerEvent::TaskSpawningTimeout {
                run_id,
                task_id,
                task_title: "test task".to_string(),
                timeout_secs: 120,
            },
            RunnerEvent::CwdMismatch {
                run_id,
                cwd: "/wrong/path".to_string(),
                root_path: "/correct/path".to_string(),
            },
            RunnerEvent::TaskCompletedWithoutSteps {
                run_id,
                task_id,
                task_title: "test task".to_string(),
                steps_skipped: 3,
                steps_total: 3,
            },
        ];

        for event in &events {
            event_tx.send(event.clone()).unwrap();
        }

        // Verify all 3 events received
        for _ in 0..3 {
            let received = rx.try_recv();
            assert!(received.is_ok(), "Expected event, got {:?}", received);
        }
    }

    #[test]
    fn test_event_serialization_new_variants() {
        // RunnerEvent uses #[serde(tag = "event", rename_all = "snake_case")]
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // TaskSpawningTimeout → "event": "task_spawning_timeout"
        let event = RunnerEvent::TaskSpawningTimeout {
            run_id,
            task_id,
            task_title: "my task".to_string(),
            timeout_secs: 120,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains("task_spawning_timeout"),
            "Expected 'task_spawning_timeout' in: {}",
            json
        );
        assert!(json.contains("120"));
        // Roundtrip
        let _: RunnerEvent = serde_json::from_str(&json).unwrap();

        // CwdMismatch → "event": "cwd_mismatch"
        let event = RunnerEvent::CwdMismatch {
            run_id,
            cwd: "/a".to_string(),
            root_path: "/b".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains("cwd_mismatch"),
            "Expected 'cwd_mismatch' in: {}",
            json
        );
        let _: RunnerEvent = serde_json::from_str(&json).unwrap();

        // TaskCompletedWithoutSteps → "event": "task_completed_without_steps"
        let event = RunnerEvent::TaskCompletedWithoutSteps {
            run_id,
            task_id,
            task_title: "task".to_string(),
            steps_skipped: 5,
            steps_total: 5,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains("task_completed_without_steps"),
            "Expected 'task_completed_without_steps' in: {}",
            json
        );
        assert!(json.contains("steps_skipped"));
        let _: RunnerEvent = serde_json::from_str(&json).unwrap();
    }

    // === T7 skipped step: test session cleanup after on_task_completed ===

    #[tokio::test]
    async fn test_on_task_completed_removes_agent_from_active_agents() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Set up RUNNER_STATE with an active agent for this task
        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test task".to_string()));
            *global = Some(state);
        }

        // Verify agent is present before
        {
            let global = RUNNER_STATE.read().await;
            let state = global
                .as_ref()
                .expect("RUNNER_STATE should be set by this test");
            assert!(
                state.get_agent(&task_id).is_some(),
                "Agent should be present before on_task_completed"
            );
        }

        // Call on_task_completed
        runner
            .on_task_completed(run_id, task_id, 10.0, 0.5, None)
            .await
            .unwrap();

        // Verify agent is removed after
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref state) = *global {
                assert!(
                    state.get_agent(&task_id).is_none(),
                    "Agent should be removed after on_task_completed"
                );
            }
            // If state is None, mark_task_completed already cleaned up — agent is gone
        }

        // Verify task status is completed
        let updated_task = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated_task.status, TaskStatus::Completed);

        // Cleanup (idempotent)
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === T3 skipped step: test run_cargo_fmt ===

    #[tokio::test]
    async fn test_run_cargo_fmt_success_in_valid_project() {
        // Use the project-orchestrator itself as a valid Cargo project
        let cwd = env!("CARGO_MANIFEST_DIR");
        let result = PlanRunner::run_cargo_fmt(cwd).await;
        assert!(
            result.is_ok(),
            "cargo fmt should succeed on this project: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_run_cargo_fmt_fails_in_non_cargo_dir() {
        let result = PlanRunner::run_cargo_fmt("/tmp").await;
        assert!(
            result.is_err(),
            "cargo fmt should fail in a non-Cargo directory"
        );
    }

    // === T5 skipped step: test step lifecycle (in_progress → completed on success, → failed on failure) ===

    #[tokio::test]
    async fn test_finalize_steps_on_failure_marks_remaining_as_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Step 1: in_progress (simulates step started by runner at launch)
        let mut step1 = test_step(1, "Write code");
        step1.status = StepStatus::InProgress;
        graph.create_step(task_id, &step1).await.unwrap();

        // Step 2: pending (not started yet)
        let mut step2 = test_step(2, "Run tests");
        step2.status = StepStatus::Pending;
        graph.create_step(task_id, &step2).await.unwrap();

        // Step 3: completed (already done before failure)
        let mut step3 = test_step(3, "Setup");
        step3.status = StepStatus::Completed;
        graph.create_step(task_id, &step3).await.unwrap();

        // finalize_steps with Failed outcome — remaining steps become Skipped
        let result = TaskResult::Failed {
            reason: "compilation error".to_string(),
            attempts: 1,
            cost_usd: 0.3,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        // Verify breakdown
        assert_eq!(breakdown.total, 3);
        assert_eq!(breakdown.completed, 1); // step3 was already completed
        assert_eq!(breakdown.skipped, 2); // step1 (in_progress) + step2 (pending) → skipped

        // Verify step statuses:
        let steps = graph.get_task_steps(task_id).await.unwrap();
        let step1_updated = steps
            .iter()
            .find(|s| s.description == "Write code")
            .unwrap();
        let step2_updated = steps.iter().find(|s| s.description == "Run tests").unwrap();
        let step3_updated = steps.iter().find(|s| s.description == "Setup").unwrap();

        // in_progress → skipped (on failure, remaining steps are skipped)
        assert_eq!(
            step1_updated.status,
            StepStatus::Skipped,
            "in_progress step should become skipped on task failure"
        );
        // pending → skipped
        assert_eq!(
            step2_updated.status,
            StepStatus::Skipped,
            "pending step should become skipped on task failure"
        );
        // completed stays completed
        assert_eq!(
            step3_updated.status,
            StepStatus::Completed,
            "already completed step should stay completed"
        );
    }

    #[tokio::test]
    async fn test_step_marked_in_progress_at_task_launch_then_completed() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Pending;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Create 3 pending steps
        let mut step1 = test_step(1, "First step");
        step1.status = StepStatus::Pending;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Second step");
        step2.status = StepStatus::Pending;
        graph.create_step(task_id, &step2).await.unwrap();

        let mut step3 = test_step(3, "Third step");
        step3.status = StepStatus::Pending;
        graph.create_step(task_id, &step3).await.unwrap();

        // Simulate what execute_wave does: mark first pending step as in_progress
        let steps = graph.get_task_steps(task_id).await.unwrap();
        let first_pending = steps
            .iter()
            .find(|s| s.status == StepStatus::Pending)
            .unwrap();
        graph
            .update_step_status(first_pending.id, StepStatus::InProgress)
            .await
            .unwrap();

        // Verify first step is now in_progress
        let steps_after = graph.get_task_steps(task_id).await.unwrap();
        let first = steps_after
            .iter()
            .find(|s| s.description == "First step")
            .unwrap();
        assert_eq!(first.status, StepStatus::InProgress);

        // Now simulate task success: finalize_steps should mark all as completed
        let result = TaskResult::Success {
            cost_usd: 0.1,
            duration_secs: 30.0,
        };
        let breakdown = runner.finalize_steps(task_id, &result, "/tmp").await;

        assert_eq!(breakdown.total, 3);
        assert_eq!(breakdown.completed, 3);

        // All steps should be completed
        let final_steps = graph.get_task_steps(task_id).await.unwrap();
        for step in &final_steps {
            assert_eq!(
                step.status,
                StepStatus::Completed,
                "Step '{}' should be completed after task success, got {:?}",
                step.description,
                step.status
            );
        }
    }

    // === Coverage tests: on_task_completed with cwd (cargo fmt path) ===

    #[tokio::test]
    async fn test_on_task_completed_with_cwd_runs_cargo_fmt() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }

        // Call with cwd = this project (cargo fmt will run and succeed)
        let cwd = env!("CARGO_MANIFEST_DIR");
        runner
            .on_task_completed(run_id, task_id, 10.0, 0.5, Some(cwd))
            .await
            .unwrap();

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Completed);

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage tests: on_task_failed (definitive, with session close) ===

    #[tokio::test]
    async fn test_on_task_failed_definitive_marks_failed_and_removes_agent() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }
        // Override config to 0 retries
        let runner = {
            let (event_tx, _) = broadcast::channel(16);
            let config = RunnerConfig {
                max_retries: 0,
                ..Default::default()
            };
            PlanRunner::new(
                runner.chat_manager.clone(),
                runner.graph.clone(),
                runner.context_builder.clone(),
                config,
                event_tx,
            )
        };

        let should_retry = runner
            .on_task_failed(run_id, plan_id, task_id, "test error", 10.0, 0.5)
            .await
            .unwrap();

        assert!(!should_retry, "Should NOT retry with max_retries=0");

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Failed);

        // Agent should be removed
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref state) = *global {
                assert!(state.get_agent(&task_id).is_none());
            }
        }

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    #[tokio::test]
    async fn test_on_task_failed_retryable_marks_pending() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            *global = Some(state);
        }

        // Default config has max_retries = 1, retry_count starts at 0 → should retry
        let should_retry = runner
            .on_task_failed(run_id, plan_id, task_id, "transient error", 5.0, 0.2)
            .await
            .unwrap();

        assert!(
            should_retry,
            "Should retry with max_retries=1 and 0 attempts"
        );

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(
            updated.status,
            TaskStatus::Pending,
            "Task should be back to Pending for retry"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage tests: finalize_run with CompletedWithErrors ===

    #[tokio::test]
    async fn test_finalize_run_completed_with_errors() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Failed;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::CompletedWithErrors, None)
            .await
            .unwrap();

        // Verify the run state was finalized
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref state) = *global {
                assert_eq!(state.status, PlanRunStatus::CompletedWithErrors);
            }
        }

        // Plan status should be InProgress (not Completed, not Cancelled)
        let updated_plan = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated_plan.status,
            crate::neo4j::models::PlanStatus::InProgress,
            "CompletedWithErrors should map to PlanStatus::InProgress"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    #[tokio::test]
    async fn test_finalize_run_completed_success() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 0, TriggerSource::Manual);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::Completed, None)
            .await
            .unwrap();

        let updated_plan = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated_plan.status,
            crate::neo4j::models::PlanStatus::Completed,
            "Completed should map to PlanStatus::Completed"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage tests: compute_final_status (extracted pure function) ===

    #[test]
    fn test_compute_final_status_all_completed() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        t1.id = Uuid::new_v4();
        let mut t2 = test_task();
        t2.status = TaskStatus::Completed;
        t2.id = Uuid::new_v4();

        let result = compute_final_status(&[t1, t2]);
        assert_eq!(result, PlanRunStatus::Completed);
    }

    #[test]
    fn test_compute_final_status_with_failed_task() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        let mut t2 = test_task();
        t2.status = TaskStatus::Failed;
        t2.id = Uuid::new_v4();

        let result = compute_final_status(&[t1, t2]);
        assert_eq!(result, PlanRunStatus::CompletedWithErrors);
    }

    #[test]
    fn test_compute_final_status_with_blocked_task() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        let mut t2 = test_task();
        t2.status = TaskStatus::Blocked;
        t2.id = Uuid::new_v4();

        let result = compute_final_status(&[t1, t2]);
        assert_eq!(result, PlanRunStatus::CompletedWithErrors);
    }

    #[test]
    fn test_compute_final_status_with_pending_task() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        let mut t2 = test_task();
        t2.status = TaskStatus::Pending;
        t2.id = Uuid::new_v4();

        let result = compute_final_status(&[t1, t2]);
        assert_eq!(result, PlanRunStatus::CompletedWithErrors);
    }

    #[test]
    fn test_compute_final_status_empty_tasks() {
        let result = compute_final_status(&[]);
        assert_eq!(result, PlanRunStatus::Completed);
    }

    // === Coverage: PlanRunStatus::CompletedWithErrors Display ===

    #[test]
    fn test_plan_run_status_completed_with_errors_display() {
        assert_eq!(
            PlanRunStatus::CompletedWithErrors.to_string(),
            "completed_with_errors"
        );
    }

    // === Coverage: validate_affected_files ===

    #[test]
    fn test_validate_affected_files_absolute_existing_path() {
        let files = vec!["/tmp".to_string()];
        let (valid, missing) = validate_affected_files(&files, "/nonexistent");
        assert!(valid, "/tmp exists so validation should pass");
        assert!(missing.is_empty());
    }

    #[test]
    fn test_validate_affected_files_mixed_existing_and_missing() {
        let files = vec![
            "/tmp".to_string(),
            "/nonexistent_file_xyz_12345".to_string(),
        ];
        let (valid, missing) = validate_affected_files(&files, "/");
        assert!(valid, "At least /tmp exists");
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], "/nonexistent_file_xyz_12345");
    }

    #[test]
    fn test_validate_affected_files_relative_paths() {
        // Use Cargo.toml which exists in the project root
        let cwd = env!("CARGO_MANIFEST_DIR");
        let files = vec!["Cargo.toml".to_string()];
        let (valid, missing) = validate_affected_files(&files, cwd);
        assert!(valid);
        assert!(missing.is_empty());
    }

    // === Coverage: finalize_run with Failed status ===

    #[tokio::test]
    async fn test_finalize_run_failed_creates_note() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            let failed_task_id = Uuid::new_v4();
            state.failed_tasks.push(failed_task_id);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::Failed, None)
            .await
            .unwrap();

        // Plan should be cancelled
        let updated = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated.status,
            crate::neo4j::models::PlanStatus::Cancelled,
            "Failed run should map to PlanStatus::Cancelled"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: finalize_run with Cancelled status ===

    #[tokio::test]
    async fn test_finalize_run_cancelled() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 0, TriggerSource::Manual);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::Cancelled, None)
            .await
            .unwrap();

        let updated = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated.status,
            crate::neo4j::models::PlanStatus::Cancelled,
            "Cancelled run should map to PlanStatus::Cancelled"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: finalize_run with BudgetExceeded status ===

    #[tokio::test]
    async fn test_finalize_run_budget_exceeded() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 2, TriggerSource::Manual);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::BudgetExceeded, None)
            .await
            .unwrap();

        let updated = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated.status,
            crate::neo4j::models::PlanStatus::Cancelled,
            "BudgetExceeded should map to PlanStatus::Cancelled"
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: create_failure_note ===

    #[tokio::test]
    async fn test_create_failure_note_creates_gotcha() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        let failed_task = Uuid::new_v4();
        let failed_tasks = vec![failed_task];

        runner
            .create_failure_note(plan_id, run_id, &failed_tasks, None)
            .await;

        // Verify note was created (the mock stores notes)
        let (notes, _count) = graph
            .list_notes(None, None, &crate::notes::models::NoteFilters::default())
            .await
            .unwrap();
        assert!(
            !notes.is_empty(),
            "create_failure_note should create a note"
        );
        let note = &notes[0];
        assert!(note.content.contains("Plan run failed"));
        assert!(note.content.contains(&failed_task.to_string()));

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: create_failure_gotcha ===

    #[tokio::test]
    async fn test_create_failure_gotcha_creates_note_and_links() {
        let (runner, graph) = test_plan_runner_with_graph();

        let task_id = Uuid::new_v4();
        let project_id = Some(Uuid::new_v4());

        runner
            .create_failure_gotcha(
                task_id,
                "My failing task",
                "connection timeout",
                2,
                project_id,
            )
            .await;

        // Verify note was created
        let (notes, _count) = graph
            .list_notes(None, None, &crate::notes::models::NoteFilters::default())
            .await
            .unwrap();
        assert!(
            !notes.is_empty(),
            "create_failure_gotcha should create a note"
        );
        let note = &notes[0];
        assert!(note.content.contains("Task Failed After 2 Retry(ies)"));
        assert!(note.content.contains("connection timeout"));
        assert!(note.content.contains("My failing task"));
        assert!(note.tags.contains(&"retry-exhausted".to_string()));
        assert!(note.tags.contains(&"auto-generated".to_string()));
    }

    // === Coverage: finalize_steps with BudgetExceeded (all → skipped) ===

    #[tokio::test]
    async fn test_finalize_steps_budget_exceeded_marks_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Create steps: one pending, one in_progress
        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::Pending;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Step 2");
        step2.id = Uuid::new_v4();
        step2.status = StepStatus::InProgress;
        graph.create_step(task_id, &step2).await.unwrap();

        let outcome = TaskResult::BudgetExceeded {
            cumulated_cost_usd: 100.0,
            limit_usd: 50.0,
        };
        let cwd = env!("CARGO_MANIFEST_DIR");
        let breakdown = runner.finalize_steps(task_id, &outcome, cwd).await;

        assert_eq!(breakdown.skipped, 2, "Both steps should be marked skipped");
        assert_eq!(breakdown.completed, 0);
        assert_eq!(breakdown.total, 2);
    }

    // === Coverage: finalize_steps with Failed outcome ===

    #[tokio::test]
    async fn test_finalize_steps_failed_marks_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Step 2");
        step2.id = Uuid::new_v4();
        step2.status = StepStatus::Pending;
        graph.create_step(task_id, &step2).await.unwrap();

        let mut step3 = test_step(3, "Step 3");
        step3.id = Uuid::new_v4();
        step3.status = StepStatus::Pending;
        graph.create_step(task_id, &step3).await.unwrap();

        let outcome = TaskResult::Failed {
            reason: "agent crashed".to_string(),
            attempts: 1,
            cost_usd: 0.5,
        };
        let cwd = env!("CARGO_MANIFEST_DIR");
        let breakdown = runner.finalize_steps(task_id, &outcome, cwd).await;

        // 1 already completed + 2 pending → skipped
        assert_eq!(breakdown.completed, 1);
        assert_eq!(breakdown.skipped, 2);
        assert_eq!(breakdown.total, 3);
    }

    // === Coverage: finalize_steps with Timeout (no commits) ===

    #[tokio::test]
    async fn test_finalize_steps_timeout_no_commits_marks_skipped() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::InProgress;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Step 2");
        step2.id = Uuid::new_v4();
        step2.status = StepStatus::Pending;
        graph.create_step(task_id, &step2).await.unwrap();

        let outcome = TaskResult::Timeout {
            duration_secs: 600.0,
            cost_usd: 5.0,
        };
        // Use a temp dir (no git repo) so has_commits = false
        let tmp = std::env::temp_dir();
        let cwd = tmp.to_str().unwrap();
        let breakdown = runner.finalize_steps(task_id, &outcome, cwd).await;

        // Without commits, timeout marks everything as skipped
        assert_eq!(breakdown.skipped, 2);
        assert_eq!(breakdown.completed, 0);
        assert_eq!(breakdown.total, 2);
    }

    // === Coverage: finalize_steps all-pending warning path ===

    #[tokio::test]
    async fn test_finalize_steps_all_pending_warning() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // All steps pending = agent updated 0 steps
        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::Pending;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Step 2");
        step2.id = Uuid::new_v4();
        step2.status = StepStatus::Pending;
        graph.create_step(task_id, &step2).await.unwrap();

        let outcome = TaskResult::Success {
            duration_secs: 30.0,
            cost_usd: 0.5,
        };
        let cwd = env!("CARGO_MANIFEST_DIR");
        let breakdown = runner.finalize_steps(task_id, &outcome, cwd).await;

        // On success, pending steps are auto-completed
        assert_eq!(breakdown.completed, 2);
        assert_eq!(breakdown.pending, 0);
        assert_eq!(breakdown.total, 2);
    }

    // === Coverage: finalize_steps no remaining steps (early return) ===

    #[tokio::test]
    async fn test_finalize_steps_all_already_completed_early_return() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Completed;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::Completed;
        graph.create_step(task_id, &step1).await.unwrap();

        let outcome = TaskResult::Success {
            duration_secs: 10.0,
            cost_usd: 0.2,
        };
        let cwd = env!("CARGO_MANIFEST_DIR");
        let breakdown = runner.finalize_steps(task_id, &outcome, cwd).await;

        assert_eq!(breakdown.completed, 1);
        assert_eq!(breakdown.pending, 0);
        assert_eq!(breakdown.skipped, 0);
        assert_eq!(breakdown.total, 1);
    }

    // === Coverage: check_step_coherence with all_pending_or_skipped ===

    #[tokio::test]
    async fn test_check_step_coherence_all_skipped_returns_false() {
        use crate::neo4j::models::{StepStatus, TaskStatus};
        use crate::test_helpers::{test_plan, test_step, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let mut step1 = test_step(1, "Step 1");
        step1.status = StepStatus::Skipped;
        graph.create_step(task_id, &step1).await.unwrap();

        let mut step2 = test_step(2, "Step 2");
        step2.id = Uuid::new_v4();
        step2.status = StepStatus::Skipped;
        graph.create_step(task_id, &step2).await.unwrap();

        let result = runner.check_step_coherence(task_id).await;
        assert_eq!(result, Some(false), "All skipped should return Some(false)");
    }

    // === Coverage: on_task_completed with cwd=None (no cargo fmt) ===

    #[tokio::test]
    async fn test_on_task_completed_without_cwd_skips_fmt() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }

        // Call with cwd=None — should skip cargo fmt and still succeed
        runner
            .on_task_completed(run_id, task_id, 5.0, 0.3, None)
            .await
            .unwrap();

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Completed);

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: on_task_failed with RUNNER_STATE=None ===

    #[tokio::test]
    async fn test_on_task_failed_no_global_state() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        // Don't set RUNNER_STATE — tests the (false, 0) fallback branch
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }

        let should_retry = runner
            .on_task_failed(run_id, plan_id, task_id, "no state error", 1.0, 0.1)
            .await
            .unwrap();

        assert!(
            !should_retry,
            "No global state should mean no retry (false, 0)"
        );

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Failed);
    }

    // === Coverage: should_step_guard_trigger edge cases ===

    #[test]
    fn test_should_step_guard_trigger_with_completed() {
        let breakdown = StepBreakdown {
            completed: 2,
            skipped: 0,
            pending: 0,
            total: 2,
        };
        assert!(!should_step_guard_trigger(&breakdown));
    }

    #[test]
    fn test_should_step_guard_trigger_zero_completed_with_total() {
        let breakdown = StepBreakdown {
            completed: 0,
            skipped: 3,
            pending: 0,
            total: 3,
        };
        assert!(should_step_guard_trigger(&breakdown));
    }

    // === Coverage: PlanRunStatus from neo4j deserialization ===

    #[test]
    fn test_plan_run_status_all_variants_display() {
        assert_eq!(PlanRunStatus::Running.to_string(), "running");
        assert_eq!(PlanRunStatus::Completed.to_string(), "completed");
        assert_eq!(
            PlanRunStatus::CompletedWithErrors.to_string(),
            "completed_with_errors"
        );
        assert_eq!(PlanRunStatus::Failed.to_string(), "failed");
        assert_eq!(PlanRunStatus::Cancelled.to_string(), "cancelled");
        assert_eq!(PlanRunStatus::BudgetExceeded.to_string(), "budget_exceeded");
    }

    // === Coverage: finalize_run with cwd for worktree cleanup ===

    #[tokio::test]
    async fn test_finalize_run_with_cwd_attempts_worktree_cleanup() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::test_helpers::test_plan;

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let state = RunnerState::new(run_id, plan_id, 0, TriggerSource::Manual);
            *global = Some(state);
        }

        let cwd = env!("CARGO_MANIFEST_DIR");
        // Should not fail even if there are no worktrees to clean
        runner
            .finalize_run(run_id, PlanRunStatus::Completed, Some(cwd))
            .await
            .unwrap();

        let updated_plan = graph.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(
            updated_plan.status,
            crate::neo4j::models::PlanStatus::Completed
        );

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: finalize_run CompletedWithErrors creates failure note ===

    #[tokio::test]
    async fn test_finalize_run_completed_with_errors_creates_failure_note() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Failed;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.failed_tasks.push(task.id);
            *global = Some(state);
        }

        runner
            .finalize_run(run_id, PlanRunStatus::CompletedWithErrors, None)
            .await
            .unwrap();

        // Verify failure note was created
        let (notes, _count) = graph
            .list_notes(None, None, &crate::notes::models::NoteFilters::default())
            .await
            .unwrap();
        assert!(
            !notes.is_empty(),
            "CompletedWithErrors should create a failure note"
        );
        assert!(notes[0].content.contains("Plan run failed"));

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: RunnerState methods used in production code ===

    #[test]
    fn test_runner_state_elapsed_secs() {
        let state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        // elapsed should be very small (just created)
        assert!(state.elapsed_secs() < 1.0);
    }

    #[test]
    fn test_runner_state_progress_pct_partial() {
        let mut state = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 4, TriggerSource::Manual);
        state.completed_tasks.push(Uuid::new_v4());
        state.completed_tasks.push(Uuid::new_v4());
        assert!((state.progress_pct() - 50.0).abs() < 0.01);
    }

    // === Coverage: on_task_completed with session_id (close_session branch) ===

    #[tokio::test]
    async fn test_on_task_completed_with_session_id_closes_session() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        let fake_session_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            // Set a session_id so the close_session branch is exercised
            state.set_agent_session(&task_id, Some(fake_session_id));
            *global = Some(state);
        }

        let cwd = env!("CARGO_MANIFEST_DIR");
        runner
            .on_task_completed(run_id, task_id, 10.0, 0.5, Some(cwd))
            .await
            .unwrap();

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Completed);

        // Agent should be removed from active state
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref state) = *global {
                assert!(
                    state.get_agent(&task_id).is_none(),
                    "Agent should be removed after completion"
                );
            }
        }

        // Give tokio::spawn a moment to run the fire-and-forget close_session
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: on_task_failed definitive with session_id (close_session branch) ===

    #[tokio::test]
    async fn test_on_task_failed_definitive_with_session_id_closes_session() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        let fake_session_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            state.set_agent_session(&task_id, Some(fake_session_id));
            *global = Some(state);
        }

        // Override config to 0 retries for definitive failure path
        let runner = {
            let (event_tx, _) = broadcast::channel(16);
            let config = RunnerConfig {
                max_retries: 0,
                ..Default::default()
            };
            PlanRunner::new(
                runner.chat_manager.clone(),
                runner.graph.clone(),
                runner.context_builder.clone(),
                config,
                event_tx,
            )
        };

        let should_retry = runner
            .on_task_failed(run_id, plan_id, task_id, "fatal error", 10.0, 0.5)
            .await
            .unwrap();

        assert!(!should_retry, "Should NOT retry with max_retries=0");

        let updated = graph.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Failed);

        // Agent should be removed
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref state) = *global {
                assert!(state.get_agent(&task_id).is_none());
            }
        }

        // Give tokio::spawn a moment to run the fire-and-forget close_session
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: on_task_completed exercises vector collector record_task ===

    #[tokio::test]
    async fn test_on_task_completed_records_vector_collector() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (runner, graph) = test_plan_runner_with_graph();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }

        runner
            .on_task_completed(run_id, task_id, 42.0, 1.23, None)
            .await
            .unwrap();

        // Verify vector collector recorded the task
        {
            let collector = VECTOR_COLLECTOR.read().await;
            // per_task_durations and per_task_costs should have at least 1 entry
            assert!(
                !collector.per_task_durations.is_empty(),
                "Vector collector should have recorded at least 1 task duration"
            );
            assert!(
                !collector.per_task_costs.is_empty(),
                "Vector collector should have recorded at least 1 task cost"
            );
            // The last entry should match our task
            let (last_tid, last_dur) = collector.per_task_durations.last().unwrap();
            assert_eq!(*last_tid, task_id);
            assert!((last_dur - 42.0).abs() < 0.001);
            let (last_tid, last_cost) = collector.per_task_costs.last().unwrap();
            assert_eq!(*last_tid, task_id);
            assert!((last_cost - 1.23).abs() < 0.001);
        }

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: add_agent_cost via RUNNER_STATE (real-time cost path) ===

    #[tokio::test]
    async fn test_add_agent_cost_via_runner_state() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        let task_id = Uuid::new_v4();
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();

        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "cost-test".to_string()));
            *global = Some(state);
        }

        // Simulate what listen_for_result does: update agent cost in real-time
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_agent_cost(&task_id, 0.75);
            }
        }
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_agent_cost(&task_id, 1.25);
            }
        }

        // Verify cumulative cost
        {
            let global = RUNNER_STATE.read().await;
            let state = global.as_ref().unwrap();
            let agent = state.get_agent(&task_id).unwrap();
            assert!(
                (agent.cost_usd - 2.0).abs() < 0.001,
                "Agent cost should be 2.0, got {}",
                agent.cost_usd
            );
        }

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: compute_final_status called from execute_wave context ===

    #[test]
    fn test_compute_final_status_mixed_failed_and_pending() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        t1.id = Uuid::new_v4();
        let mut t2 = test_task();
        t2.status = TaskStatus::Failed;
        t2.id = Uuid::new_v4();
        let mut t3 = test_task();
        t3.status = TaskStatus::Pending;
        t3.id = Uuid::new_v4();

        let result = compute_final_status(&[t1, t2, t3]);
        assert_eq!(result, PlanRunStatus::CompletedWithErrors);
    }

    #[test]
    fn test_compute_final_status_single_completed() {
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::test_task;

        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;

        let result = compute_final_status(&[t1]);
        assert_eq!(result, PlanRunStatus::Completed);
    }

    // === Coverage: validate_affected_files edge case — all absolute missing ===

    #[test]
    fn test_validate_affected_files_all_absolute_missing() {
        let files = vec![
            "/nonexistent_a_xyz".to_string(),
            "/nonexistent_b_xyz".to_string(),
        ];
        let (valid, missing) = validate_affected_files(&files, "/tmp");
        assert!(!valid, "No files exist so validation should fail");
        assert_eq!(missing.len(), 2);
    }

    // === Coverage: on_task_completed emits RunnerEvent::TaskCompleted ===

    #[tokio::test]
    async fn test_on_task_completed_emits_event() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (event_tx, _) = broadcast::channel(16);
        let mut event_rx = event_tx.subscribe();

        let (runner_base, graph) = test_plan_runner_with_graph();
        let runner = PlanRunner::new(
            runner_base.chat_manager.clone(),
            runner_base.graph.clone(),
            runner_base.context_builder.clone(),
            RunnerConfig::default(),
            event_tx,
        );

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }

        runner
            .on_task_completed(run_id, task_id, 5.0, 0.3, None)
            .await
            .unwrap();

        // Verify event was emitted
        let event = event_rx.try_recv();
        assert!(event.is_ok(), "Should have received a TaskCompleted event");
        match event.unwrap() {
            RunnerEvent::TaskCompleted {
                run_id: r,
                task_id: t,
                cost_usd,
                duration_secs,
                ..
            } => {
                assert_eq!(r, run_id);
                assert_eq!(t, task_id);
                assert!((cost_usd - 0.3).abs() < 0.001);
                assert!((duration_secs - 5.0).abs() < 0.001);
            }
            other => panic!("Expected TaskCompleted, got {:?}", other),
        }

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    // === Coverage: on_task_failed emits RunnerEvent::TaskFailed ===

    #[tokio::test]
    async fn test_on_task_failed_definitive_emits_event() {
        let _guard = TEST_MUTEX.lock().await;
        reset_globals().await;

        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan, test_task};

        let (event_tx, _) = broadcast::channel(16);
        let mut event_rx = event_tx.subscribe();

        let (runner_base, graph) = test_plan_runner_with_graph();
        let config = RunnerConfig {
            max_retries: 0,
            ..Default::default()
        };
        let runner = PlanRunner::new(
            runner_base.chat_manager.clone(),
            runner_base.graph.clone(),
            runner_base.context_builder.clone(),
            config,
            event_tx,
        );

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let run_id = Uuid::new_v4();
        {
            let mut global = RUNNER_STATE.write().await;
            let mut state = RunnerState::new(run_id, plan_id, 1, TriggerSource::Manual);
            state.add_agent(ActiveAgent::new(task_id, "test".to_string()));
            *global = Some(state);
        }

        let should_retry = runner
            .on_task_failed(run_id, plan_id, task_id, "crash", 8.0, 0.4)
            .await
            .unwrap();

        assert!(!should_retry);

        // Verify event was emitted
        let event = event_rx.try_recv();
        assert!(event.is_ok(), "Should have received a TaskFailed event");
        match event.unwrap() {
            RunnerEvent::TaskFailed {
                run_id: r,
                task_id: t,
                reason,
                ..
            } => {
                assert_eq!(r, run_id);
                assert_eq!(t, task_id);
                assert_eq!(reason, "crash");
            }
            other => panic!("Expected TaskFailed, got {:?}", other),
        }

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }
}
