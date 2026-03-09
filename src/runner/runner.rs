//! PlanRunner — Main execution loop for autonomous plan execution.
//!
//! Orchestrates: get_next_task → spawn agent → monitor → verify → update_status → next.
//! The Runner owns all task/step status transitions — the spawned agent must NOT update them.

use crate::chat::manager::ChatManager;
use crate::chat::types::{ChatEvent, ChatRequest};
use crate::neo4j::models::TaskStatus;
use crate::neo4j::traits::GraphStore;
use crate::orchestrator::context::ContextBuilder;
use crate::runner::models::{
    PlanRunStatus, RunnerConfig, RunnerEvent, TaskResult, TriggerSource,
};
use crate::runner::state::RunnerState;

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
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

// ============================================================================
// Runner constraints injected into the agent prompt
// ============================================================================

const RUNNER_CONSTRAINTS: &str = r#"
## Runner Constraints

You are an autonomous agent spawned by the PlanRunner. Follow these rules strictly:

1. **DO NOT** call `task(action: "update", status: ...)` or `step(action: "update", status: ...)` via MCP — the Runner manages all status transitions.
2. Work autonomously without asking for user confirmation. Execute the task fully.
3. Make atomic commits with conventional format: `<type>(<scope>): <short description>`.
4. **NEVER** commit sensitive files (.env, credentials, *.key, *.pem, *.secret).
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
}

/// Result of starting a plan run.
#[derive(Debug, Clone)]
pub struct StartResult {
    pub run_id: Uuid,
    pub plan_id: Uuid,
    pub total_waves: usize,
    pub total_tasks: usize,
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
        }
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

        // Reset cancel flag
        RUNNER_CANCEL.store(false, Ordering::SeqCst);

        let result = StartResult {
            run_id,
            plan_id,
            total_waves,
            total_tasks,
        };

        // 5. Emit PlanStarted event
        let _ = self.event_tx.send(RunnerEvent::PlanStarted {
            run_id,
            plan_id,
            plan_title: String::new(), // Will be enriched by caller
            total_tasks,
            total_waves,
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
                let _ = runner.event_tx.send(RunnerEvent::RunnerError {
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
                status: Some(state.status.clone()),
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

            let _ = self.event_tx.send(RunnerEvent::WaveStarted {
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
                    warn!("Skipping blocked task {}: {}", wave_task.id, wave_task.title.as_deref().unwrap_or("untitled"));
                    continue;
                }

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

                match task_result {
                    Ok(TaskResult::Success { duration_secs, cost_usd }) => {
                        self.on_task_completed(run_id, wave_task.id, duration_secs, cost_usd)
                            .await?;
                    }
                    Ok(TaskResult::Failed { reason, attempts, cost_usd }) => {
                        self.on_task_failed(run_id, plan_id, wave_task.id, &reason, 0.0, cost_usd)
                            .await?;
                    }
                    Ok(TaskResult::Timeout { duration_secs, cost_usd }) => {
                        let _ = self.event_tx.send(RunnerEvent::TaskTimeout {
                            run_id,
                            task_id: wave_task.id,
                            task_title: wave_task.title.clone().unwrap_or_default(),
                            duration_secs,
                        });
                        self.on_task_failed(run_id, plan_id, wave_task.id, "Task timeout", duration_secs, cost_usd)
                            .await?;
                    }
                    Ok(TaskResult::BudgetExceeded { cumulated_cost_usd, limit_usd }) => {
                        let plan_id = {
                            let global = RUNNER_STATE.read().await;
                            global.as_ref().map(|s| s.plan_id).unwrap_or(plan_id)
                        };
                        let _ = self.event_tx.send(RunnerEvent::BudgetExceeded {
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
                        // Mark task as blocked and continue
                        let _ = self
                            .graph
                            .update_task_status(wave_task.id, TaskStatus::Blocked)
                            .await;
                    }
                    Err(e) => {
                        error!("Task {} execution error: {}", wave_task.id, e);
                        self.on_task_failed(run_id, plan_id, wave_task.id, &e.to_string(), 0.0, 0.0)
                            .await?;
                    }
                }

                // Budget check after each task
                {
                    let global = RUNNER_STATE.read().await;
                    if let Some(ref s) = *global {
                        if s.is_budget_exceeded(self.config.max_cost_usd) {
                            let _ = self.event_tx.send(RunnerEvent::BudgetExceeded {
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
                global.as_ref().map(|s| (s.completed_tasks.len(), s.failed_tasks.len())).unwrap_or((0, 0))
            };
            let _ = self.event_tx.send(RunnerEvent::WaveCompleted {
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
    ) -> Result<TaskResult> {
        info!("Executing task {}: {}", task_id, task_title);

        // Update global state
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.current_task_id = Some(task_id);
                s.current_task_title = Some(task_title.to_string());
            }
        }

        // Mark task as in_progress
        self.graph.update_task_status(task_id, TaskStatus::InProgress).await?;

        // Get current wave number
        let wave_number = {
            let global = RUNNER_STATE.read().await;
            global.as_ref().map(|s| s.current_wave).unwrap_or(0)
        };

        let _ = self.event_tx.send(RunnerEvent::TaskStarted {
            run_id,
            task_id,
            task_title: task_title.to_string(),
            wave_number,
        });

        // Build the prompt
        let context = self
            .context_builder
            .build_context(task_id, plan_id)
            .await?;
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
        };

        let session = self.chat_manager.create_session(&request).await?;
        let session_id = session.session_id.clone();

        // Subscribe to events BEFORE sending message (to not miss any)
        let mut rx = self.chat_manager.subscribe(&session_id).await?;

        // Send the prompt
        self.chat_manager.send_message(&session_id, &prompt).await?;

        // Listen for events until Result
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(self.config.task_timeout_secs);
        let mut cost_usd = 0.0_f64;
        let mut is_error = false;
        let mut error_text = String::new();
        let mut _subtype = String::new();

        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Ok(TaskResult::Timeout {
                    duration_secs: start.elapsed().as_secs_f64(),
                    cost_usd,
                });
            }

            // Check cancel flag
            if RUNNER_CANCEL.load(Ordering::SeqCst) {
                return Ok(TaskResult::Failed {
                    reason: "Cancelled by user".to_string(),
                    attempts: 0,
                    cost_usd,
                });
            }

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Ok(event)) => match event {
                    ChatEvent::Result {
                        cost_usd: event_cost,
                        subtype,
                        is_error: err,
                        result_text,
                        ..
                    } => {
                        if let Some(c) = event_cost {
                            cost_usd = c;
                        }
                        is_error = err;
                        _subtype = subtype.clone();
                        if let Some(ref text) = result_text {
                            error_text = text.clone();
                        }
                        break;
                    }
                    // Accumulate cost if available from other events
                    _ => {} // Ignore intermediate events
                },
                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    warn!("Runner event receiver lagged by {} events", n);
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => {
                    return Ok(TaskResult::Failed {
                        reason: "Chat event channel closed unexpectedly".to_string(),
                        attempts: 0,
                        cost_usd,
                    });
                }
                Err(_) => {
                    // Timeout
                    return Ok(TaskResult::Timeout {
                        duration_secs: start.elapsed().as_secs_f64(),
                        cost_usd,
                    });
                }
            }
        }

        let duration_secs = start.elapsed().as_secs_f64();

        // Check budget AFTER this task's cost
        {
            let mut global = RUNNER_STATE.write().await;
            if let Some(ref mut s) = *global {
                s.add_cost(cost_usd);
                if s.is_budget_exceeded(self.config.max_cost_usd) {
                    return Ok(TaskResult::BudgetExceeded {
                        cumulated_cost_usd: s.cost_usd,
                        limit_usd: self.config.max_cost_usd,
                    });
                }
            }
        }

        if is_error {
            Ok(TaskResult::Failed {
                reason: if error_text.is_empty() {
                    format!("Agent returned error (subtype: {})", _subtype)
                } else {
                    error_text
                },
                attempts: 0,
                cost_usd,
            })
        } else {
            Ok(TaskResult::Success {
                duration_secs,
                cost_usd,
            })
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
        // Update task status
        self.graph.update_task_status(task_id, TaskStatus::Completed).await?;

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

        // Emit event
        let _ = self.event_tx.send(RunnerEvent::TaskCompleted {
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
        plan_id: Uuid,
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
                global.as_ref().and_then(|s| s.retry_counts.get(&task_id).copied()).unwrap_or(0)
            };
            let _ = self.event_tx.send(RunnerEvent::TaskFailed {
                run_id,
                task_id,
                task_title: String::new(),
                reason: format!("{} (will retry)", reason),
                attempts: retry_count,
            });
        }

        // Mark task as failed
        self.graph.update_task_status(task_id, TaskStatus::Failed).await?;

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
            global.as_ref().and_then(|s| s.retry_counts.get(&task_id).copied()).unwrap_or(0)
        };
        let _ = self.event_tx.send(RunnerEvent::TaskFailed {
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

    /// Finalize the run — update state, persist, emit event.
    async fn finalize_run(&self, run_id: Uuid, status: PlanRunStatus) -> Result<()> {
        let mut global = RUNNER_STATE.write().await;
        if let Some(ref mut s) = *global {
            s.finalize(status.clone());
            self.graph.update_plan_run(s).await?;
        }

        match status {
            PlanRunStatus::Completed => {
                let (plan_id, total_cost, total_duration, tc, tf) = global
                    .as_ref()
                    .map(|s| (s.plan_id, s.cost_usd, s.elapsed_secs(), s.completed_tasks.len(), s.failed_tasks.len()))
                    .unwrap_or((Uuid::nil(), 0.0, 0.0, 0, 0));
                let _ = self.event_tx.send(RunnerEvent::PlanCompleted {
                    run_id,
                    plan_id,
                    status: PlanRunStatus::Completed,
                    total_cost_usd: total_cost,
                    total_duration_secs: total_duration,
                    tasks_completed: tc,
                    tasks_failed: tf,
                    pr_url: None, // Auto-PR handled by T7
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
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::models::TriggerSource;

    fn make_runner_config() -> RunnerConfig {
        RunnerConfig {
            task_timeout_secs: 10,
            idle_timeout_secs: 5,
            max_retries: 1,
            auto_pr: false,
            build_check: false,
            test_runner: false,
            max_cost_usd: 1.0,
        }
    }

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
        assert!(RUNNER_CONSTRAINTS.contains("## Runner Constraints"));
        assert!(RUNNER_CONSTRAINTS.contains("DO NOT"));
        assert!(RUNNER_CONSTRAINTS.contains("task(action: \"update\", status"));
        assert!(RUNNER_CONSTRAINTS.contains(".env"));
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
