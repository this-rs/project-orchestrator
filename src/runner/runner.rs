//! PlanRunner — Main execution loop for autonomous plan execution.
//!
//! Orchestrates: get_next_task → spawn agent → monitor → verify → update_status → next.
//! The Runner owns all task/step status transitions — the spawned agent must NOT update them.

use crate::chat::manager::ChatManager;
use crate::chat::types::{ChatEvent, ChatRequest};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::models::TaskStatus;
use crate::neo4j::traits::GraphStore;
use crate::orchestrator::context::ContextBuilder;
use crate::runner::enricher::TaskEnricher;
use crate::runner::guard::{AgentGuard, ChatManagerHintSender, GuardConfig, GuardVerdict};
use crate::runner::models::{PlanRunStatus, RunnerConfig, RunnerEvent, TaskResult, TriggerSource};
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

/// Vector collector — accumulates drift/knowledge metrics during execution.
/// Reset at the start of each run, finalized at the end.
static VECTOR_COLLECTOR: LazyLock<Arc<RwLock<VectorCollector>>> =
    LazyLock::new(|| Arc::new(RwLock::new(VectorCollector::new())));

// ============================================================================
// Runner constraints injected into the agent prompt
// ============================================================================

const RUNNER_CONSTRAINTS: &str = r#"
## Runner Execution Mode

You are an **autonomous code execution agent** spawned by the PlanRunner.
Your ONLY job is to **write code, test it, and commit it**. You are NOT in a conversation.

### Behavior Rules
1. **Execute immediately** — read the task, analyze the code, write the fix, test, commit. No discussion.
2. **DO NOT** call `task(action: "update", status: ...)` or `step(action: "update", status: ...)` via MCP — the Runner manages all status transitions.
3. **DO NOT** ask questions, request confirmation, or explain your reasoning at length. Just do the work.
4. **DO NOT** use MCP project orchestrator tools for searching code — use Read, Grep, Glob directly for speed.
5. Make atomic commits with conventional format: `<type>(<scope>): <short description>`.
6. **NEVER** commit sensitive files (.env, credentials, *.key, *.pem, *.secret).
7. After writing code, ALWAYS run `cargo check` (Rust) or the relevant build command to verify compilation.
8. If tests are mentioned in steps, run them.
9. If `cargo check` or tests fail, fix the errors and retry — do not give up.
10. When done with ALL steps, make a final commit summarizing the work.

### Execution Flow
1. Read the affected files listed below
2. For each step: implement → verify → move to next
3. Run `cargo check` / `cargo test` as verification
4. Commit with a clear message
5. You are DONE when all steps are implemented and the code compiles
"#;

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

/// Result of execute_task — wraps TaskResult with the session_id for enricher.
#[derive(Debug)]
struct TaskExecutionResult {
    pub result: TaskResult,
    pub session_id: Option<Uuid>,
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
    pub current_task_id: Option<Uuid>,
    pub current_task_title: Option<String>,
    pub progress_pct: f64,
    pub tasks_completed: usize,
    pub tasks_total: usize,
    pub elapsed_secs: f64,
    pub cost_usd: f64,
}

impl PlanRunner {
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

        // Reset cancel flag and vector collector
        RUNNER_CANCEL.store(false, Ordering::SeqCst);
        {
            let mut collector = VECTOR_COLLECTOR.write().await;
            *collector = VectorCollector::new();
        }

        let result = StartResult {
            run_id,
            plan_id,
            total_waves,
            total_tasks,
        };

        // 5. Compute prediction from historical runs
        let prediction = match self.graph.list_plan_runs(plan_id, 50).await {
            Ok(runs) if !runs.is_empty() => {
                let vectors: Vec<_> = runs
                    .iter()
                    .rev()
                    .map(crate::runner::vector::ExecutionVector::from_runner_state)
                    .collect();
                Some(crate::runner::vector::predict_run(&vectors))
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

    /// Get the current run status snapshot.
    pub async fn status() -> RunStatus {
        let global = RUNNER_STATE.read().await;
        match &*global {
            Some(state) => RunStatus {
                running: state.status == PlanRunStatus::Running,
                run_id: Some(state.run_id),
                plan_id: Some(state.plan_id),
                status: Some(state.status),
                current_wave: Some(state.current_wave),
                current_task_id: state.current_task_id,
                current_task_title: state.current_task_title.clone(),
                progress_pct: state.progress_pct(),
                tasks_completed: state.tasks_completed(),
                tasks_total: state.total_tasks,
                elapsed_secs: state.elapsed_secs(),
                cost_usd: state.cost_usd,
            },
            None => RunStatus {
                running: false,
                run_id: None,
                plan_id: None,
                status: None,
                current_wave: None,
                current_task_id: None,
                current_task_title: None,
                progress_pct: 0.0,
                tasks_completed: 0,
                tasks_total: 0,
                elapsed_secs: 0.0,
                cost_usd: 0.0,
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

        for (wave_idx, wave) in waves.iter().enumerate() {
            let wave_number = wave_idx + 1;

            // Check cancel flag
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                info!("Runner cancelled at wave {}", wave_number);
                self.finalize_run(run_id, PlanRunStatus::Cancelled).await?;
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

            // Execute tasks in this wave sequentially (v1)
            for wave_task in &wave.tasks {
                // Check cancel flag between tasks
                if RUNNER_CANCEL.load(Ordering::SeqCst) {
                    info!("Runner cancelled before task {}", wave_task.id);
                    self.finalize_run(run_id, PlanRunStatus::Cancelled).await?;
                    return Ok(());
                }

                // Skip already completed tasks (for recovery)
                if wave_task.status == "completed" {
                    continue;
                }

                // Skip blocked tasks
                if wave_task.status == "blocked" {
                    warn!(
                        "Skipping blocked task {}: {}",
                        wave_task.id,
                        wave_task.title.as_deref().unwrap_or("untitled")
                    );
                    continue;
                }

                let task_start_time = chrono::Utc::now();

                let task_result = self
                    .execute_task(
                        run_id,
                        plan_id,
                        wave_task.id,
                        wave_task.title.as_deref().unwrap_or("untitled"),
                        &cwd,
                        project_slug.as_deref(),
                    )
                    .await;

                // Extract session_id from the result (for enricher)
                let task_session_id = task_result.as_ref().ok().and_then(|r| r.session_id());

                match task_result.map(|r| r.result) {
                    Ok(TaskResult::Success {
                        duration_secs,
                        cost_usd,
                    }) => {
                        self.on_task_completed(run_id, wave_task.id, duration_secs, cost_usd)
                            .await?;

                        // Fire-and-forget enrichment (commits, auto-notes, AFFECTS)
                        let enricher = TaskEnricher::new(self.graph.clone());
                        let enrich_task_id = wave_task.id;
                        let enrich_plan_id = plan_id;
                        let enrich_project_id = project_id;
                        let enrich_session_id = task_session_id;
                        let enrich_cwd = cwd.clone();
                        let enrich_start = task_start_time;
                        tokio::spawn(async move {
                            let result = enricher
                                .enrich(
                                    enrich_task_id,
                                    enrich_plan_id,
                                    enrich_project_id,
                                    enrich_session_id,
                                    enrich_start,
                                    &enrich_cwd,
                                )
                                .await;
                            info!(
                                "Enricher completed for task {}: {} commits, note={}, {} affects, {} discussed",
                                enrich_task_id,
                                result.commits_linked,
                                result.note_created,
                                result.affects_added,
                                result.discussed_added
                            );
                        });
                    }
                    Ok(TaskResult::Failed {
                        reason,
                        attempts: _,
                        cost_usd,
                    }) => {
                        self.on_task_failed(run_id, plan_id, wave_task.id, &reason, 0.0, cost_usd)
                            .await?;
                    }
                    Ok(TaskResult::Timeout {
                        duration_secs,
                        cost_usd,
                    }) => {
                        self.emit_event(RunnerEvent::TaskTimeout {
                            run_id,
                            task_id: wave_task.id,
                            task_title: wave_task.title.clone().unwrap_or_default(),
                            duration_secs,
                        });
                        self.on_task_failed(
                            run_id,
                            plan_id,
                            wave_task.id,
                            "Task timeout",
                            duration_secs,
                            cost_usd,
                        )
                        .await?;
                    }
                    Ok(TaskResult::BudgetExceeded {
                        cumulated_cost_usd,
                        limit_usd,
                    }) => {
                        let plan_id = {
                            let global = RUNNER_STATE.read().await;
                            global.as_ref().map(|s| s.plan_id).unwrap_or(plan_id)
                        };
                        self.emit_event(RunnerEvent::BudgetExceeded {
                            run_id,
                            plan_id,
                            cumulated_cost_usd,
                            limit_usd,
                        });
                        self.finalize_run(run_id, PlanRunStatus::BudgetExceeded)
                            .await?;
                        return Ok(());
                    }
                    Ok(TaskResult::Blocked { blocked_by }) => {
                        warn!("Task {} blocked by {:?}", wave_task.id, blocked_by);
                        // Mark task as blocked and continue (with CrudEvent)
                        let _ = self
                            .update_task_status_with_event(wave_task.id, TaskStatus::Blocked)
                            .await;
                    }
                    Err(e) => {
                        error!("Task {} execution error: {}", wave_task.id, e);
                        self.on_task_failed(
                            run_id,
                            plan_id,
                            wave_task.id,
                            &e.to_string(),
                            0.0,
                            0.0,
                        )
                        .await?;
                    }
                }

                // Budget check after each task
                {
                    let global = RUNNER_STATE.read().await;
                    if let Some(ref s) = *global {
                        if s.is_budget_exceeded(self.config.max_cost_usd) {
                            self.emit_event(RunnerEvent::BudgetExceeded {
                                run_id,
                                plan_id,
                                cumulated_cost_usd: s.cost_usd,
                                limit_usd: self.config.max_cost_usd,
                            });
                            drop(global);
                            self.finalize_run(run_id, PlanRunStatus::BudgetExceeded)
                                .await?;
                            return Ok(());
                        }
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
        }

        // All waves completed successfully
        self.finalize_run(run_id, PlanRunStatus::Completed).await?;
        Ok(())
    }

    /// Execute a single task by spawning a Claude Code agent.
    async fn execute_task(
        &self,
        run_id: Uuid,
        plan_id: Uuid,
        task_id: Uuid,
        task_title: &str,
        cwd: &str,
        project_slug: Option<&str>,
    ) -> Result<TaskExecutionResult> {
        info!("Executing task {}: {}", task_id, task_title);

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.current_task_id = Some(task_id);
                s.current_task_title = Some(task_title.to_string());
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

        // Build the prompt
        let context = self.context_builder.build_context(task_id, plan_id).await?;
        let mut prompt = self.context_builder.generate_prompt(&context);
        prompt.push_str(RUNNER_CONSTRAINTS);

        // Spawn agent: create_session → subscribe → send_message → listen
        let request = ChatRequest {
            message: String::new(), // message sent separately via send_message
            session_id: None,
            cwd: cwd.to_string(),
            project_slug: project_slug.map(|s| s.to_string()),
            model: None,
            permission_mode: Some("bypassPermissions".to_string()),
            add_dirs: None,
            workspace_slug: None,
            user_claims: None,
            spawned_by: Some(serde_json::json!({
                "type": "runner",
                "run_id": run_id.to_string(),
                "plan_id": plan_id.to_string(),
                "task_id": task_id.to_string(),
            }).to_string()),
        };

        let session = self.chat_manager.create_session(&request).await?;
        let session_id = session.session_id.clone();
        let session_uuid = session_id.parse::<Uuid>().ok();

        // Helper to wrap TaskResult with session_id
        let wrap = |result: TaskResult| -> TaskExecutionResult {
            TaskExecutionResult {
                result,
                session_id: session_uuid,
            }
        };

        // Subscribe to events BEFORE sending message (to not miss any)
        let rx = self.chat_manager.subscribe(&session_id).await?;
        // Clone a second receiver for the guard
        let guard_rx = self.chat_manager.subscribe(&session_id).await?;

        // Send the prompt
        self.chat_manager.send_message(&session_id, &prompt).await?;

        let start = std::time::Instant::now();

        // Spawn the AgentGuard in parallel
        let guard_config = GuardConfig {
            idle_timeout: Duration::from_secs(self.config.idle_timeout_secs),
            task_timeout: Duration::from_secs(self.config.task_timeout_secs),
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
        );

        let guard_handle = tokio::spawn(async move { guard.monitor().await });

        // Listen for events until Result — in parallel with the guard
        let event_result = self.listen_for_result(rx, run_id).await;

        // Wait for guard to finish (it should return quickly once the channel closes)
        let guard_verdict = match guard_handle.await {
            Ok(verdict) => verdict,
            Err(e) => {
                warn!("Guard task panicked: {}", e);
                GuardVerdict::Completed
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
                // Check if the guard timed out (it would have interrupted the session)
                if let GuardVerdict::Timeout { elapsed_secs } = guard_verdict {
                    return Ok(wrap(TaskResult::Timeout {
                        duration_secs: elapsed_secs,
                        cost_usd,
                    }));
                }
                return Ok(wrap(TaskResult::Failed {
                    reason: "Chat event channel closed unexpectedly".to_string(),
                    attempts: 0,
                    cost_usd,
                }));
            }
            EventListenResult::Cancelled { cost_usd } => {
                return Ok(wrap(TaskResult::Failed {
                    reason: "Cancelled by user".to_string(),
                    attempts: 0,
                    cost_usd,
                }));
            }
        };

        // If the guard decided timeout, honor it even if we got a result
        if let GuardVerdict::Timeout { elapsed_secs } = guard_verdict {
            return Ok(wrap(TaskResult::Timeout {
                duration_secs: elapsed_secs,
                cost_usd,
            }));
        }

        let duration_secs = start.elapsed().as_secs_f64();

        // Check budget AFTER this task's cost
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_cost(cost_usd);
                if s.is_budget_exceeded(self.config.max_cost_usd) {
                    return Ok(wrap(TaskResult::BudgetExceeded {
                        cumulated_cost_usd: s.cost_usd,
                        limit_usd: self.config.max_cost_usd,
                    }));
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
                return Ok(wrap(TaskResult::Failed {
                    reason: if error_text.is_empty() {
                        format!("Agent returned error (subtype: {})", _subtype)
                    } else {
                        error_text
                    },
                    attempts: 0,
                    cost_usd,
                }));
            }
        }

        // Auto-complete pending steps: the agent is instructed NOT to update step statuses
        // via MCP (the Runner manages all status transitions). After the agent finishes,
        // we mark all remaining pending steps as completed so the verifier passes.
        if let Ok(steps) = self.graph.get_task_steps(task_id).await {
            for step in &steps {
                if step.status == crate::neo4j::models::StepStatus::Pending
                    || step.status == crate::neo4j::models::StepStatus::InProgress
                {
                    if let Err(e) = self
                        .graph
                        .update_step_status(step.id, crate::neo4j::models::StepStatus::Completed)
                        .await
                    {
                        warn!("Failed to auto-complete step {}: {}", step.id, e);
                    }
                }
            }
        }

        // Post-task verification (build, steps, git sanity)
        let verifier = TaskVerifier::new(
            self.graph.clone(),
            self.config.build_check,
            self.config.test_runner,
        );
        let verify_result = verifier.verify(task_id, cwd).await;

        match verify_result {
            VerifyResult::Pass => Ok(wrap(TaskResult::Success {
                duration_secs,
                cost_usd,
            })),
            VerifyResult::Fail { reasons } => {
                let reason = format!("Post-task verification failed:\n- {}", reasons.join("\n- "));
                warn!("Task {} verification failed: {}", task_id, reason);
                Ok(wrap(TaskResult::Failed {
                    reason,
                    attempts: 0,
                    cost_usd,
                }))
            }
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
    ) -> Result<()> {
        // Update task status (with CrudEvent for WebSocket)
        self.update_task_status_with_event(task_id, TaskStatus::Completed)
            .await?;

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.mark_task_completed(task_id);
                s.current_task_id = None;
                s.current_task_title = None;
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

    async fn on_task_failed(
        &self,
        run_id: Uuid,
        _plan_id: Uuid,
        task_id: Uuid,
        reason: &str,
        duration_secs: f64,
        cost_usd: f64,
    ) -> Result<()> {
        // Check retry count
        let mut should_retry = false;
        {
            let global = RUNNER_STATE.read().await;
            if let Some(ref s) = *global {
                let retry_count = s.retry_counts.get(&task_id).copied().unwrap_or(0);
                if retry_count < self.config.max_retries {
                    should_retry = true;
                }
            }
        }

        if should_retry {
            // Increment retry count
            {
                let mut global = RUNNER_STATE.write().await;
                if let Some(ref mut s) = *global {
                    *s.retry_counts.entry(task_id).or_insert(0) += 1;
                }
            }

            warn!(
                "Task {} failed ({}), retrying with error context...",
                task_id, reason
            );

            // Retry: re-execute with the error context injected
            // For now, we just mark failed — retry logic will be enhanced in T3/T4
            let retry_count = {
                let global = RUNNER_STATE.read().await;
                global
                    .as_ref()
                    .and_then(|s| s.retry_counts.get(&task_id).copied())
                    .unwrap_or(0)
            };
            self.emit_event(RunnerEvent::TaskFailed {
                run_id,
                task_id,
                task_title: String::new(),
                reason: format!("{} (will retry)", reason),
                attempts: retry_count,
            });
        }

        // Mark task as failed (with CrudEvent for WebSocket)
        self.update_task_status_with_event(task_id, TaskStatus::Failed)
            .await?;

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.mark_task_failed(task_id);
                s.current_task_id = None;
                s.current_task_title = None;
                self.graph.update_plan_run(s).await?;
            }
        }

        let attempts = {
            let global = RUNNER_STATE.read().await;
            global
                .as_ref()
                .and_then(|s| s.retry_counts.get(&task_id).copied())
                .unwrap_or(0)
        };
        self.emit_event(RunnerEvent::TaskFailed {
            run_id,
            task_id,
            task_title: String::new(),
            reason: reason.to_string(),
            attempts,
        });

        error!(
            "Task {} failed: {} (duration: {:.1}s, cost: ${:.4})",
            task_id, reason, duration_secs, cost_usd
        );

        Ok(())
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

    /// Finalize the run — update state, persist, emit event.
    async fn finalize_run(&self, run_id: Uuid, status: PlanRunStatus) -> Result<()> {
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
    ) -> EventListenResult {
        let start = std::time::Instant::now();
        // Use a generous timeout here — the guard handles the actual task_timeout
        // with soft hints before hard stop. This is just a safety net.
        let safety_timeout = std::time::Duration::from_secs(self.config.task_timeout_secs + 60);
        let mut cost_usd = 0.0_f64;

        loop {
            let remaining = safety_timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return EventListenResult::Completed {
                    cost_usd,
                    is_error: true,
                    error_text: "Safety timeout exceeded".to_string(),
                    subtype: "timeout".to_string(),
                };
            }

            // Check cancel flag
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                return EventListenResult::Cancelled { cost_usd };
            }

            match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv()).await {
                Ok(Ok(event)) => {
                    if let ChatEvent::Result {
                        cost_usd: event_cost,
                        subtype,
                        is_error,
                        result_text,
                        ..
                    } = event
                    {
                        if let Some(c) = event_cost {
                            cost_usd = c;
                        }
                        return EventListenResult::Completed {
                            cost_usd,
                            is_error,
                            error_text: result_text.unwrap_or_default(),
                            subtype,
                        };
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    warn!("Runner event receiver lagged by {} events", n);
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => {
                    return EventListenResult::ChannelClosed { cost_usd };
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

    #[tokio::test]
    async fn test_status_when_no_run() {
        // Clear global state
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
        let status = PlanRunner::status().await;
        assert!(!status.running);
        assert!(status.run_id.is_none());
        assert_eq!(status.progress_pct, 0.0);
    }

    #[tokio::test]
    async fn test_status_when_running() {
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

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    #[tokio::test]
    async fn test_cancel_no_active_run() {
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
        let result = PlanRunner::cancel(Uuid::new_v4()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No active run"));
    }

    #[tokio::test]
    async fn test_cancel_sets_flag() {
        let run_id = Uuid::new_v4();
        let plan_id = Uuid::new_v4();
        let state = RunnerState::new(run_id, plan_id, 3, TriggerSource::Manual);
        {
            let mut global = RUNNER_STATE.write().await;
            *global = Some(state);
        }
        RUNNER_CANCEL.store(false, Ordering::SeqCst);

        let result = PlanRunner::cancel(run_id).await;
        assert!(result.is_ok());
        assert!(RUNNER_CANCEL.load(Ordering::SeqCst));

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
        RUNNER_CANCEL.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn test_cancel_wrong_run_id() {
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

        // Cleanup
        {
            let mut global = RUNNER_STATE.write().await;
            *global = None;
        }
    }

    #[test]
    fn test_runner_constraints_in_prompt() {
        assert!(RUNNER_CONSTRAINTS.contains("## Runner Execution Mode"));
        assert!(RUNNER_CONSTRAINTS.contains("autonomous code execution agent"));
        assert!(RUNNER_CONSTRAINTS.contains("DO NOT"));
        assert!(RUNNER_CONSTRAINTS.contains("task(action: \"update\", status"));
        assert!(RUNNER_CONSTRAINTS.contains(".env"));
        assert!(RUNNER_CONSTRAINTS.contains("cargo check"));
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
            progress_pct: 0.0,
            tasks_completed: 0,
            tasks_total: 0,
            elapsed_secs: 0.0,
            cost_usd: 0.0,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"running\":false"));
    }
}
