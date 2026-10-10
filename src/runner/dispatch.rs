//! TriggerDispatcher — turns a trigger that fires into a plan run.
//!
//! The three plan trigger sources (schedule, webhook, event) all go through
//! [`TriggerDispatcher::dispatch`]:
//! 1. the guards of [`TriggerEngine::evaluate_and_prepare`] (enabled, cooldown,
//!    no active run);
//! 2. where the run executes, resolved from the plan's project exactly as a
//!    `plan(action: "run", project_slug)` call does (cwd `.` → the project's
//!    `root_path`);
//! 3. [`PlanRunner::start`], through the same [`PlanRunnerFactory`] the REST
//!    `POST /api/plans/{id}/run` handler uses;
//! 4. one [`TriggerFiring`] per firing, carrying the `plan_run_id` of the run
//!    it started, or the `start_error` that kept it from starting.

use crate::auth::jwt::Claims;
use crate::chat::manager::ChatManager;
use crate::events::EventEmitter;
use crate::neo4j::traits::GraphStore;
use crate::orchestrator::context::ContextBuilder;
use crate::runner::models::{RunnerConfig, Trigger, TriggerFiring, TriggerSource};
use crate::runner::runner::{PlanRunner, StartResult};
use crate::runner::trigger::TriggerEngine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================================
// Building a PlanRunner
// ============================================================================

/// Per-run options of a [`PlanRunner`] (caller identity, budget, routing).
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Claims the run's agents authenticate with; `None`: a service account
    /// per agent (see `PlanRunner::with_user_claims`).
    pub claims: Option<Claims>,
    /// Budget in USD of the run; `None`: the runner's default.
    pub max_cost_usd: Option<f64>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_tokens: Option<u64>,
}

/// What a [`PlanRunner`] needs from the server. The single place a runner for
/// a new run is put together: the REST handler and the trigger dispatcher both
/// build theirs here.
#[derive(Clone)]
pub struct PlanRunnerFactory {
    chat_manager: Arc<ChatManager>,
    graph: Arc<dyn GraphStore>,
    context_builder: Arc<ContextBuilder>,
    config: RunnerConfig,
    event_emitter: Option<Arc<dyn EventEmitter>>,
}

impl std::fmt::Debug for PlanRunnerFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanRunnerFactory").finish()
    }
}

impl PlanRunnerFactory {
    pub fn new(
        chat_manager: Arc<ChatManager>,
        graph: Arc<dyn GraphStore>,
        context_builder: Arc<ContextBuilder>,
        config: RunnerConfig,
        event_emitter: Option<Arc<dyn EventEmitter>>,
    ) -> Self {
        Self {
            chat_manager,
            graph,
            context_builder,
            config,
            event_emitter,
        }
    }

    /// A runner for one new run.
    pub fn build(&self, options: RunOptions) -> Arc<PlanRunner> {
        let mut config = self.config.clone();
        if let Some(budget) = options.max_cost_usd {
            config.max_cost_usd = budget;
        }
        // RunnerEvents of this run (the WebSocket bridge is the event emitter).
        let (event_tx, _) = tokio::sync::broadcast::channel(256);
        let mut runner = PlanRunner::new(
            self.chat_manager.clone(),
            self.graph.clone(),
            self.context_builder.clone(),
            config,
            event_tx,
        );
        if let Some(claims) = options.claims {
            runner = runner.with_user_claims(claims);
        }
        runner = runner.with_run_routing(options.provider, options.model, options.max_tokens);
        // Cognitive routing (B-R7): None until the decider is installed at startup.
        runner = runner.with_routing(crate::runner::routing::installed());
        if let Some(emitter) = &self.event_emitter {
            runner = runner.with_event_emitter(emitter.clone());
        }
        Arc::new(runner)
    }
}

/// Starts a plan run. Implemented by [`PlanRunnerFactory`]; a seam for tests.
#[async_trait]
pub trait PlanRunStarter: Send + Sync {
    async fn start_run(
        &self,
        plan_id: Uuid,
        source: TriggerSource,
        cwd: String,
        project_slug: Option<String>,
    ) -> Result<StartResult>;
}

#[async_trait]
impl PlanRunStarter for PlanRunnerFactory {
    async fn start_run(
        &self,
        plan_id: Uuid,
        source: TriggerSource,
        cwd: String,
        project_slug: Option<String>,
    ) -> Result<StartResult> {
        // No caller: the run's agents use a service account each, the budget
        // and the routing are the defaults.
        self.build(RunOptions::default())
            .start(plan_id, source, cwd, project_slug)
            .await
    }
}

/// The server has no chat manager: no run can start. Every firing records it.
#[derive(Debug, Default)]
pub struct NoPlanRunner;

#[async_trait]
impl PlanRunStarter for NoPlanRunner {
    async fn start_run(
        &self,
        plan_id: Uuid,
        _source: TriggerSource,
        _cwd: String,
        _project_slug: Option<String>,
    ) -> Result<StartResult> {
        Err(anyhow!(
            "Plan {} not started: the chat manager is not initialized, no run can start",
            plan_id
        ))
    }
}

// ============================================================================
// Where a triggered run executes
// ============================================================================

/// Where the run of `plan_id` executes: `(cwd, project_slug)` as a
/// `plan(action: "run")` call passes them — cwd `.` and the slug of the plan's
/// project, which the runner resolves to the project's `root_path`.
///
/// A trigger has no caller to name a directory, so a plan without a project,
/// or whose project has no existing `root_path`, cannot run: an error, never
/// the server's own working directory.
pub async fn resolve_run_location(
    graph: &dyn GraphStore,
    plan_id: Uuid,
) -> Result<(String, Option<String>)> {
    let plan = graph
        .get_plan(plan_id)
        .await?
        .ok_or_else(|| anyhow!("Plan {} not found", plan_id))?;

    let project = match plan.project_id {
        Some(project_id) => graph.get_project(project_id).await?,
        None => match graph.list_plan_project_slugs(plan_id).await?.first() {
            Some(slug) => graph.get_project_by_slug(slug).await?,
            None => None,
        },
    }
    .ok_or_else(|| {
        anyhow!(
            "Plan {} has no project: a triggered run needs one to know where to execute",
            plan_id
        )
    })?;

    if project.root_path.is_empty() {
        return Err(anyhow!(
            "Project '{}' of plan {} has no root_path: a triggered run has no directory to execute in",
            project.slug,
            plan_id
        ));
    }
    let root = expand_home(&project.root_path);
    if !std::path::Path::new(&root).is_dir() {
        return Err(anyhow!(
            "root_path '{}' of project '{}' is not a directory",
            project.root_path,
            project.slug
        ));
    }

    Ok((".".to_string(), Some(project.slug)))
}

/// `~` at the start of a path → `$HOME` (as the runner's cwd resolution does).
fn expand_home(path: &str) -> String {
    if path.starts_with('~') {
        path.replacen('~', &std::env::var("HOME").unwrap_or_default(), 1)
    } else {
        path.to_string()
    }
}

// ============================================================================
// Dispatcher
// ============================================================================

/// What became of a trigger signal.
#[derive(Debug, Clone)]
pub enum DispatchOutcome {
    /// The guards held it back (disabled, cooldown, active run): no firing.
    Skipped,
    /// A run started; the firing carries its id.
    Started {
        firing: TriggerFiring,
        start: StartResult,
    },
    /// The trigger fired but no run started; the firing carries the reason.
    StartFailed {
        firing: TriggerFiring,
        error: String,
    },
}

/// The one way a plan trigger turns into a plan run.
pub struct TriggerDispatcher {
    graph: Arc<dyn GraphStore>,
    engine: Arc<TriggerEngine>,
    starter: Arc<dyn PlanRunStarter>,
}

impl std::fmt::Debug for TriggerDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerDispatcher").finish()
    }
}

impl TriggerDispatcher {
    pub fn new(
        graph: Arc<dyn GraphStore>,
        engine: Arc<TriggerEngine>,
        starter: Arc<dyn PlanRunStarter>,
    ) -> Self {
        Self {
            graph,
            engine,
            starter,
        }
    }

    /// Evaluate the guards of `trigger` and, when it fires, start the plan run
    /// and record the firing (with the run id, or with the start error).
    ///
    /// `Err` only when the guards or the firing record cannot be read/written;
    /// a run that does not start is an `Ok(StartFailed)` with its firing.
    pub async fn dispatch(
        &self,
        trigger: &Trigger,
        source_payload: Option<serde_json::Value>,
    ) -> Result<DispatchOutcome> {
        let Some(source) = self.engine.evaluate_and_prepare(trigger).await? else {
            return Ok(DispatchOutcome::Skipped);
        };

        let started = match resolve_run_location(self.graph.as_ref(), trigger.plan_id).await {
            Ok((cwd, project_slug)) => {
                self.starter
                    .start_run(trigger.plan_id, source, cwd, project_slug)
                    .await
            }
            Err(e) => Err(e),
        };

        match started {
            Ok(start) => {
                let firing = self
                    .engine
                    .record_fire(trigger, Some(start.run_id), source_payload, None)
                    .await?;
                info!(
                    "Trigger {} started run {} of plan {}",
                    trigger.id, start.run_id, trigger.plan_id
                );
                Ok(DispatchOutcome::Started { firing, start })
            }
            Err(e) => {
                let error = format!("{e:#}");
                warn!(
                    "Trigger {} fired but plan {} did not start: {}",
                    trigger.id, trigger.plan_id, error
                );
                let firing = self
                    .engine
                    .record_fire(trigger, None, source_payload, Some(error.clone()))
                    .await?;
                Ok(DispatchOutcome::StartFailed { firing, error })
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::models::TriggerType;
    use crate::runner::RunnerState;
    use crate::test_helpers::{test_plan, test_project, test_task};
    use chrono::Utc;
    use tokio::sync::Mutex;

    /// One `start_run` call: plan, source, cwd, project slug.
    pub(crate) type StartCall = (Uuid, TriggerSource, String, Option<String>);

    /// Records each start and persists a PlanRun, as `PlanRunner::start` does.
    #[derive(Default)]
    pub(crate) struct RecordingStarter {
        pub graph: Option<Arc<dyn GraphStore>>,
        pub calls: Mutex<Vec<StartCall>>,
    }

    impl RecordingStarter {
        pub(crate) fn on(graph: Arc<dyn GraphStore>) -> Self {
            Self {
                graph: Some(graph),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl PlanRunStarter for RecordingStarter {
        async fn start_run(
            &self,
            plan_id: Uuid,
            source: TriggerSource,
            cwd: String,
            project_slug: Option<String>,
        ) -> Result<StartResult> {
            self.calls
                .lock()
                .await
                .push((plan_id, source.clone(), cwd, project_slug));
            let run_id = Uuid::new_v4();
            if let Some(graph) = &self.graph {
                graph
                    .create_plan_run(&RunnerState::new(run_id, plan_id, 1, source))
                    .await?;
            }
            Ok(StartResult {
                run_id,
                plan_id,
                total_waves: 1,
                total_tasks: 1,
            })
        }
    }

    pub(crate) fn trigger_of(plan_id: Uuid, trigger_type: TriggerType) -> Trigger {
        Trigger {
            id: Uuid::new_v4(),
            plan_id,
            trigger_type,
            config: serde_json::json!({}),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        }
    }

    /// A plan with one task in a project whose root_path is `root`.
    pub(crate) async fn runnable_plan(mock: &MockGraphStore, root: &std::path::Path) -> Uuid {
        let mut project = test_project();
        project.root_path = root.to_string_lossy().into_owned();
        mock.create_project(&project).await.unwrap();
        let mut plan = test_plan();
        plan.project_id = Some(project.id);
        mock.create_plan(&plan).await.unwrap();
        mock.create_task(plan.id, &test_task()).await.unwrap();
        plan.id
    }

    /// A plan whose only task is completed, in a project whose root_path is a
    /// fresh git repository at `root`: a real run of it starts, spawns no
    /// agent, and touches no directory outside `root`. `(project_id, plan_id)`.
    pub(crate) async fn completed_plan_in_git_repo(
        graph: &dyn GraphStore,
        root: &std::path::Path,
    ) -> (Uuid, Uuid) {
        use crate::neo4j::models::TaskStatus;
        let git_init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .status()
            .unwrap();
        assert!(git_init.success());
        let mut project = test_project();
        project.root_path = root.to_string_lossy().into_owned();
        graph.create_project(&project).await.unwrap();
        let mut plan = test_plan();
        plan.project_id = Some(project.id);
        graph.create_plan(&plan).await.unwrap();
        let mut task = test_task();
        task.status = TaskStatus::Completed;
        graph.create_task(plan.id, &task).await.unwrap();
        (project.id, plan.id)
    }

    /// Wait (bounded) for the background execution of `run_id` to end, so a
    /// test does not release the runner globals while it still runs.
    pub(crate) async fn wait_until_finished(graph: &dyn GraphStore, run_id: Uuid) {
        use crate::runner::models::PlanRunStatus;
        for _ in 0..400 {
            let run = graph.get_plan_run(run_id).await.unwrap().unwrap();
            if run.status != PlanRunStatus::Running {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        panic!("run {run_id} still running after 10s");
    }

    #[tokio::test]
    async fn dispatch_starts_the_run_and_records_its_id() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = trigger_of(plan_id, TriggerType::Schedule);
        mock.create_trigger(&trigger).await.unwrap();

        let starter = Arc::new(RecordingStarter::on(mock.clone()));
        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            starter.clone(),
        );

        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        let DispatchOutcome::Started { firing, start } = outcome else {
            panic!("expected a started run, got {outcome:?}");
        };
        assert_eq!(firing.plan_run_id, Some(start.run_id));
        assert!(firing.start_error.is_none());

        // Started with the plan's project, cwd resolved by the runner.
        let calls = starter.calls.lock().await;
        assert_eq!(calls.len(), 1);
        let project_slug = test_project().slug;
        assert_eq!(
            calls[0],
            (
                plan_id,
                TriggerSource::Schedule {
                    trigger_id: trigger.id
                },
                ".".to_string(),
                Some(project_slug)
            )
        );

        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].plan_run_id, Some(start.run_id));
    }

    #[tokio::test]
    async fn dispatch_skips_without_firing_when_guards_hold() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let mut trigger = trigger_of(plan_id, TriggerType::Schedule);
        trigger.enabled = false;
        mock.create_trigger(&trigger).await.unwrap();

        let starter = Arc::new(RecordingStarter::on(mock.clone()));
        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            starter.clone(),
        );

        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        assert!(matches!(outcome, DispatchOutcome::Skipped));
        assert!(starter.calls.lock().await.is_empty());
        assert!(mock
            .list_trigger_firings(trigger.id, 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_plan_without_project_is_a_recorded_start_error() {
        let mock = Arc::new(MockGraphStore::new());
        let plan = test_plan();
        mock.create_plan(&plan).await.unwrap();
        mock.create_task(plan.id, &test_task()).await.unwrap();
        let trigger = trigger_of(plan.id, TriggerType::Event);
        mock.create_trigger(&trigger).await.unwrap();

        let starter = Arc::new(RecordingStarter::on(mock.clone()));
        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            starter.clone(),
        );

        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        let DispatchOutcome::StartFailed { firing, error } = outcome else {
            panic!("expected a start failure, got {outcome:?}");
        };
        assert!(error.contains("has no project"), "{error}");
        assert!(firing.plan_run_id.is_none());
        // Never started anywhere (not in the server's own directory).
        assert!(starter.calls.lock().await.is_empty());

        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].start_error.as_deref(), Some(error.as_str()));
    }

    #[tokio::test]
    async fn a_project_without_root_path_is_a_recorded_start_error() {
        let mock = Arc::new(MockGraphStore::new());
        let mut project = test_project();
        project.root_path = String::new();
        mock.create_project(&project).await.unwrap();
        let mut plan = test_plan();
        plan.project_id = Some(project.id);
        mock.create_plan(&plan).await.unwrap();
        let trigger = trigger_of(plan.id, TriggerType::Webhook);
        mock.create_trigger(&trigger).await.unwrap();

        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            Arc::new(RecordingStarter::on(mock.clone())),
        );

        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        let DispatchOutcome::StartFailed { error, .. } = outcome else {
            panic!("expected a start failure, got {outcome:?}");
        };
        assert!(error.contains("no root_path"), "{error}");
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings[0].start_error.as_deref(), Some(error.as_str()));
    }

    #[tokio::test]
    async fn without_a_runner_the_firing_says_why() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = trigger_of(plan_id, TriggerType::Webhook);
        mock.create_trigger(&trigger).await.unwrap();

        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            Arc::new(NoPlanRunner),
        );
        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        assert!(matches!(outcome, DispatchOutcome::StartFailed { .. }));
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert!(firings[0]
            .start_error
            .as_deref()
            .is_some_and(|e| e.contains("chat manager")));
    }

    /// A real `PlanRunner`, built by the factory the REST handler uses: the
    /// PlanRun exists, carries the trigger as its source, and is the run the
    /// firing names. A plan without tasks is refused by the runner itself and
    /// the refusal lands in the firing.
    #[tokio::test]
    async fn the_factory_really_starts_the_run() {
        use crate::chat::config::ChatConfig;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::meilisearch::traits::SearchStore;
        use crate::notes::manager::NoteManager;
        use crate::plan::manager::PlanManager;

        let _guard = crate::runner::runner::RUNNER_GLOBALS_TEST_LOCK.lock().await;
        {
            *crate::runner::RUNNER_STATE.write().await = None;
        }

        let mock = Arc::new(MockGraphStore::new());
        let graph: Arc<dyn GraphStore> = mock.clone();
        let search: Arc<dyn SearchStore> = Arc::new(MockSearchStore::new());
        let chat_manager = Arc::new(ChatManager::new_without_memory(
            graph.clone(),
            search.clone(),
            ChatConfig::default(),
        ));
        let context_builder = Arc::new(ContextBuilder::new(
            graph.clone(),
            search.clone(),
            Arc::new(PlanManager::new(graph.clone(), search.clone())),
            Arc::new(NoteManager::new(graph.clone(), search.clone())),
        ));
        let factory = Arc::new(PlanRunnerFactory::new(
            chat_manager,
            graph.clone(),
            context_builder,
            RunnerConfig::default(),
            None,
        ));

        let dir = tempfile::tempdir().unwrap();
        let (project_id, plan_id) = completed_plan_in_git_repo(graph.as_ref(), dir.path()).await;

        let trigger = trigger_of(plan_id, TriggerType::Schedule);
        mock.create_trigger(&trigger).await.unwrap();
        let dispatcher = TriggerDispatcher::new(
            graph.clone(),
            Arc::new(TriggerEngine::new(graph.clone())),
            factory.clone(),
        );

        let outcome = dispatcher.dispatch(&trigger, None).await.unwrap();
        let DispatchOutcome::Started { firing, start } = outcome else {
            panic!("expected a started run, got {outcome:?}");
        };
        let run = mock
            .get_plan_run(start.run_id)
            .await
            .unwrap()
            .expect("the PlanRun exists in the graph");
        assert_eq!(run.plan_id, plan_id);
        assert_eq!(
            run.triggered_by,
            TriggerSource::Schedule {
                trigger_id: trigger.id
            }
        );
        assert_eq!(firing.plan_run_id, Some(start.run_id));

        wait_until_finished(graph.as_ref(), start.run_id).await;

        // Same factory, a plan without tasks: the runner refuses, the firing says so.
        let mut empty = test_plan();
        empty.project_id = Some(project_id);
        mock.create_plan(&empty).await.unwrap();
        let empty_trigger = trigger_of(empty.id, TriggerType::Event);
        mock.create_trigger(&empty_trigger).await.unwrap();
        let outcome = dispatcher.dispatch(&empty_trigger, None).await.unwrap();
        let DispatchOutcome::StartFailed { firing, error } = outcome else {
            panic!("expected a start failure, got {outcome:?}");
        };
        assert!(error.contains("no tasks"), "{error}");
        assert_eq!(firing.start_error.as_deref(), Some(error.as_str()));

        {
            *crate::runner::RUNNER_STATE.write().await = None;
        }
    }
}
