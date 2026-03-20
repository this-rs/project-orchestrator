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
use crate::runner::models::{
    ActiveAgent, ActiveAgentSnapshot, PlanRunStatus, RunnerConfig, RunnerEvent,
    TaskExecutionReport, TaskResult, TriggerSource,
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
use tracing::{error, info, warn};
use uuid::Uuid;

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
        }
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
                if let Err(e) = self.graph.update_plan_status(plan_id, PlanStatus::Completed).await {
                    warn!("Failed to set plan {} status to Completed on recovery: {}", plan_id, e);
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
                        if let Err(e) = runner.graph.update_plan_status(s.plan_id, PlanStatus::Cancelled).await {
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
        if let Err(e) = self.graph.update_plan_status(plan_id, PlanStatus::InProgress).await {
            warn!(
                "Failed to set plan {} status to in_progress (may already be in_progress): {}",
                plan_id, e
            );
        } else {
            info!("Plan {} status set to in_progress", plan_id);
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
                    if let Err(e) = runner.graph.update_plan_status(s.plan_id, PlanStatus::Cancelled).await {
                        warn!("Failed to set plan {} status to Cancelled after error: {}", s.plan_id, e);
                    }
                }
            }
        });

        Ok(result)
    }

    /// Cancel the currently running plan execution.
    pub async fn cancel(run_id: Uuid) -> Result<()> {
        let global = RUNNER_STATE.read().await;
        match &*global {
            Some(state) if state.run_id == run_id && state.status == PlanRunStatus::Running => {
                RUNNER_CANCEL.store(true, Ordering::SeqCst);
                info!("Runner cancel requested for run {}", run_id);
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
    pub async fn force_cancel(run_id: Uuid, graph: Arc<dyn GraphStore>) -> Result<()> {
        // Set cancel flag so any in-flight code that checks it will stop
        RUNNER_CANCEL.store(true, Ordering::SeqCst);
        // Clear budget override
        RUNNER_BUDGET.store(0, std::sync::atomic::Ordering::Relaxed);

        let mut global = RUNNER_STATE.write().await;
        match &mut *global {
            Some(state) if state.run_id == run_id => {
                let plan_id = state.plan_id;
                state.finalize(PlanRunStatus::Cancelled);
                if let Err(e) = graph.update_plan_run(state).await {
                    error!("Failed to persist force-cancelled run to Neo4j: {}", e);
                }
                // Transition plan status to Cancelled
                if let Err(e) = graph.update_plan_status(plan_id, PlanStatus::Cancelled).await {
                    warn!("Failed to set plan {} status to Cancelled on force-cancel: {}", plan_id, e);
                }
                info!("Runner force-cancelled run {}", run_id);
                // Clear the global state so a new run can start
                *global = None;
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

        // Resolve project_id from slug (for enricher)
        let project_id = if let Some(ref slug) = project_slug {
            match self.graph.get_project_by_slug(slug).await {
                Ok(Some(project)) => Some(project.id),
                Ok(None) => {
                    warn!(
                        "Project slug '{}' not found, enricher will run without project_id",
                        slug
                    );
                    None
                }
                Err(e) => {
                    warn!("Failed to resolve project slug '{}': {}, enricher will run without project_id", slug, e);
                    None
                }
            }
        } else {
            None
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

        // All waves completed successfully
        self.finalize_run(run_id, PlanRunStatus::Completed, Some(&cwd))
            .await?;
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
                if t.status == "completed" {
                    return false;
                }
                if t.status == "blocked" {
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
                    self.on_task_completed(run_id, task_id, duration_secs, cost_usd)
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
                }
                Ok(TaskResult::Timeout {
                    duration_secs,
                    cost_usd,
                }) => {
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
                    let _ = self
                        .update_task_status_with_event(task_id, TaskStatus::Blocked)
                        .await;
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
                            self.on_task_completed(run_id, *task_id, duration_secs, cost_usd)
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
            user_claims: None,
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

        let session = self.chat_manager.create_session(&request).await?;
        let session_id = session.session_id.clone();
        let session_uuid = session_id.parse::<Uuid>().ok();

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
            loop_threshold: 3,
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
            let listen_fut = self.listen_for_result(rx, run_id);
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
            self.finalize_steps(task_id, &success_result, cwd).await;
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
    // Task lifecycle callbacks
    // ========================================================================

    async fn on_task_completed(
        &self,
        run_id: Uuid,
        task_id: Uuid,
        duration_secs: f64,
        cost_usd: f64,
    ) -> Result<()> {
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
    async fn finalize_steps(&self, task_id: Uuid, outcome: &TaskResult, cwd: &str) {
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
                return;
            }
        };

        let remaining: Vec<_> = steps
            .iter()
            .filter(|s| {
                s.status == crate::neo4j::models::StepStatus::Pending
                    || s.status == crate::neo4j::models::StepStatus::InProgress
            })
            .collect();

        if remaining.is_empty() {
            return;
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

            if let Err(e) = self.graph.update_step_status(step.id, new_status).await {
                warn!("finalize_steps: failed to update step {}: {}", step.id, e);
            }
        }

        info!(
            "Task {} — finalized {} remaining steps (outcome: {})",
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
            }
        );
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

        let mut global = RUNNER_STATE.write().await;
        if let Some(ref mut s) = *global {
            s.finalize(status);
            self.graph.update_plan_run(s).await?;
        }

        // Finalize the execution vector from the collector
        if let Some(ref state) = *global {
            let collector = VECTOR_COLLECTOR.read().await;
            let vector = collector.finalize(state);
            // Log the derived metrics
            info!(
                "Run {} vector: efficiency={:.2}, velocity={:.3}, stability={:.2}",
                run_id,
                vector.efficiency(),
                vector.velocity(),
                vector.stability()
            );
            // TODO: persist vector to PlanRun node in Neo4j when schema supports it
        }

        // Transition plan status to match the run's terminal state
        {
            let plan_id = global.as_ref().map(|s| s.plan_id).unwrap_or(Uuid::nil());
            if plan_id != Uuid::nil() {
                let plan_status = match status {
                    PlanRunStatus::Completed => Some(PlanStatus::Completed),
                    PlanRunStatus::Failed | PlanRunStatus::BudgetExceeded => Some(PlanStatus::Cancelled),
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
        }

        match status {
            PlanRunStatus::Completed => {
                let (plan_id, total_cost, total_duration, tc, tf) = global
                    .as_ref()
                    .map(|s| {
                        (
                            s.plan_id,
                            s.cost_usd,
                            s.elapsed_secs(),
                            s.completed_tasks.len(),
                            s.failed_tasks.len(),
                        )
                    })
                    .unwrap_or((Uuid::nil(), 0.0, 0.0, 0, 0));

                // Post-run enricher sweep: catch commits missed by per-task enrichment
                // (e.g., mega-commit at end instead of atomic per-task commits)
                if let Some(cwd) = cwd {
                    if let Some(ref state) = *global {
                        let enricher = TaskEnricher::new(self.graph.clone());
                        let sweep_linked = enricher
                            .post_run_sweep(
                                state.plan_id,
                                &state.completed_tasks,
                                state.started_at,
                                state.project_id,
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
                }

                // Auto-PR: generate and create PR if auto_pr is enabled
                let pr_url = if self.config.auto_pr {
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
            PlanRunStatus::Cancelled => {
                info!("Plan run {} cancelled", run_id);
            }
            PlanRunStatus::BudgetExceeded => {
                warn!("Plan run {} aborted: budget exceeded", run_id);
            }
            PlanRunStatus::Failed => {
                error!("Plan run {} failed", run_id);
            }
            _ => {}
        }

        Ok(())
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
        let result = PlanRunner::cancel(Uuid::new_v4()).await;
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

        let result = PlanRunner::cancel(run_id).await;
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

        let result = PlanRunner::cancel(Uuid::new_v4()).await;
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
        let (result, metrics) = runner.listen_for_result(rx, run_id).await;

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
        let (_result, metrics) = runner.listen_for_result(rx, run_id).await;

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
        let (_result, metrics) = runner.listen_for_result(rx, run_id).await;

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
        let (result, metrics) = runner.listen_for_result(rx, run_id).await;

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
        let (result, _metrics) = runner.listen_for_result(rx, run_id).await;

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
        runner.finalize_steps(task_id, &result, "/tmp").await;

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
        runner.finalize_steps(task_id, &result, "/tmp").await;

        let steps = graph.get_task_steps(task_id).await.unwrap();
        assert_eq!(steps[0].status, StepStatus::Completed);
    }
}
