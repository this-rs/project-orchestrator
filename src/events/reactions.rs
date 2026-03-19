//! # Built-in Reactions — Automatic event-driven behaviors
//!
//! This module defines the built-in reaction handlers that the [`EventReactor`](super::EventReactor)
//! dispatches when matching events are received from the [`EventBus`](super::EventBus).
//!
//! ## Built-in Reactions
//!
//! | Event                              | Reaction                                                   |
//! |------------------------------------|------------------------------------------------------------|
//! | `Project::Synced`                  | Bootstrap knowledge fabric (first sync) or update scores   |
//! | `Task::StatusChanged → Completed`  | Check if all plan tasks completed → auto-complete plan     |
//! | `Plan::StatusChanged → Completed`  | Log completion (future: collect episode if protocol linked)|
//!
//! ## Architecture
//!
//! Each reaction is an async function that receives the [`CrudEvent`] and an
//! `Arc<ServerState>`. They are registered via [`register_builtin_reactions`]
//! which returns a configured [`ReactorBuilder`](super::ReactorBuilder).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::types::{CrudAction, CrudEvent, EntityType, EventEmitter};
use super::ReactorBuilder;
use crate::api::handlers::ServerState;

// ─────────────────────────────────────────────────────────────
// Reaction: Project::Synced → bootstrap or update fabric
// ─────────────────────────────────────────────────────────────

/// When a project is synced, bootstrap the knowledge fabric if it's the first sync,
/// or update fabric scores for subsequent syncs.
///
/// Reads `is_first_sync` from the event payload to decide which path to take.
async fn on_project_synced(event: CrudEvent, state: Arc<ServerState>) {
    let project_id_str = &event.entity_id;
    let project_id = match Uuid::parse_str(project_id_str) {
        Ok(id) => id,
        Err(e) => {
            warn!(
                project_id = %project_id_str,
                error = %e,
                "on_project_synced: invalid project UUID"
            );
            return;
        }
    };

    let is_first_sync = event
        .payload
        .get("is_first_sync")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_first_sync {
        info!(
            project_id = %project_id,
            "on_project_synced: first sync detected, bootstrapping knowledge fabric"
        );

        // Get the project to find root_path for backfill_commit_touches
        let project = match state.orchestrator.neo4j().get_project(project_id).await {
            Ok(Some(p)) => p,
            Ok(None) => {
                warn!(project_id = %project_id, "on_project_synced: project not found");
                return;
            }
            Err(e) => {
                error!(project_id = %project_id, error = %e, "on_project_synced: failed to get project");
                return;
            }
        };

        // Step 1: Backfill commit touches
        let root_path = std::path::PathBuf::from(crate::expand_tilde(&project.root_path));
        match state
            .orchestrator
            .backfill_commit_touches(project_id, &root_path)
            .await
        {
            Ok(result) => {
                info!(
                    project_id = %project_id,
                    commits_parsed = result.commits_parsed,
                    touches_created = result.touches_created,
                    "on_project_synced: backfill_commit_touches completed"
                );
            }
            Err(e) => {
                error!(
                    project_id = %project_id,
                    error = %e,
                    "on_project_synced: backfill_commit_touches failed"
                );
            }
        }

        // Step 2: Reindex decisions
        match state.orchestrator.plan_manager().reindex_decisions().await {
            Ok((total, indexed)) => {
                debug!(
                    project_id = %project_id,
                    total, indexed,
                    "on_project_synced: reindex_decisions completed"
                );
            }
            Err(e) => {
                error!(
                    project_id = %project_id,
                    error = %e,
                    "on_project_synced: reindex_decisions failed"
                );
            }
        }

        // Step 3: Update fabric scores
        let weights = crate::graph::models::FabricWeights::default();
        match tokio::time::timeout(
            std::time::Duration::from_secs(120),
            state
                .orchestrator
                .analytics()
                .analyze_fabric_graph(project_id, &weights),
        )
        .await
        {
            Ok(Ok(analytics)) => {
                info!(
                    project_id = %project_id,
                    nodes = analytics.metrics.len(),
                    communities = analytics.communities.len(),
                    "on_project_synced: bootstrap fabric scores completed"
                );
            }
            Ok(Err(e)) => {
                error!(
                    project_id = %project_id,
                    error = %e,
                    "on_project_synced: analyze_fabric_graph failed"
                );
            }
            Err(_) => {
                error!(
                    project_id = %project_id,
                    "on_project_synced: analyze_fabric_graph timed out (120s)"
                );
            }
        }
    } else {
        debug!(
            project_id = %project_id,
            "on_project_synced: incremental sync, debouncing fabric score update"
        );

        // For incremental syncs, use the debouncer to avoid expensive recomputation
        // on every file change. The debouncer batches multiple syncs into one update.
        state.orchestrator.analytics_debouncer().trigger(project_id);
    }
}

// ─────────────────────────────────────────────────────────────
// Reaction: Task::StatusChanged(Completed) → auto-complete plan
// ─────────────────────────────────────────────────────────────

/// When a task status changes to "Completed", check if all tasks in the parent plan
/// are now completed. If so, automatically transition the plan to "Completed".
///
/// This provides automatic plan lifecycle management — no manual intervention needed
/// to mark a plan as complete when all its tasks are done.
async fn on_task_status_changed(event: CrudEvent, state: Arc<ServerState>) {
    let new_status = event
        .payload
        .get("new_status")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Only react to Completed transitions
    if new_status != "Completed" {
        return;
    }

    let task_id = match Uuid::parse_str(&event.entity_id) {
        Ok(id) => id,
        Err(e) => {
            warn!(
                task_id = %event.entity_id,
                error = %e,
                "on_task_status_changed: invalid task UUID"
            );
            return;
        }
    };

    debug!(
        task_id = %task_id,
        "on_task_status_changed: task completed, checking plan progress"
    );

    // Find the parent plan for this task by querying the graph
    // We need to get plan_id from the task → plan relationship
    let plan_id = match get_plan_id_for_task(&state, task_id).await {
        Some(id) => id,
        None => {
            debug!(
                task_id = %task_id,
                "on_task_status_changed: no parent plan found for task"
            );
            return;
        }
    };

    // Get all tasks in the plan
    let tasks = match state.orchestrator.neo4j().get_plan_tasks(plan_id).await {
        Ok(tasks) => tasks,
        Err(e) => {
            error!(
                plan_id = %plan_id,
                error = %e,
                "on_task_status_changed: failed to get plan tasks"
            );
            return;
        }
    };

    if tasks.is_empty() {
        return;
    }

    // Check if ALL tasks are completed (or failed — those don't block completion)
    let all_done = tasks.iter().all(|t| {
        matches!(
            t.status,
            crate::neo4j::models::TaskStatus::Completed | crate::neo4j::models::TaskStatus::Failed
        )
    });

    if all_done {
        info!(
            plan_id = %plan_id,
            total_tasks = tasks.len(),
            "on_task_status_changed: all tasks completed, auto-completing plan"
        );

        // Check that plan is currently in_progress (valid transition)
        match state.orchestrator.neo4j().get_plan(plan_id).await {
            Ok(Some(plan)) => {
                if plan.status != crate::neo4j::models::PlanStatus::InProgress {
                    debug!(
                        plan_id = %plan_id,
                        current_status = ?plan.status,
                        "on_task_status_changed: plan not in_progress, skipping auto-complete"
                    );
                    return;
                }
            }
            Ok(None) => return,
            Err(e) => {
                error!(plan_id = %plan_id, error = %e, "on_task_status_changed: failed to get plan");
                return;
            }
        }

        if let Err(e) = state
            .orchestrator
            .neo4j()
            .update_plan_status(plan_id, crate::neo4j::models::PlanStatus::Completed)
            .await
        {
            error!(
                plan_id = %plan_id,
                error = %e,
                "on_task_status_changed: failed to auto-complete plan"
            );
        } else {
            // Emit the StatusChanged event for the plan
            state.event_bus.emit_status_changed(
                EntityType::Plan,
                &plan_id.to_string(),
                "InProgress",
                "Completed",
                None,
            );
        }
    } else {
        let completed = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::Completed)
            .count();
        debug!(
            plan_id = %plan_id,
            completed,
            total = tasks.len(),
            "on_task_status_changed: plan not yet complete ({}/{})",
            completed,
            tasks.len()
        );
    }
}

/// Helper: find the plan UUID that owns a given task.
///
/// Uses `GraphStore::get_plan_id_for_task` which traverses the
/// `(Plan)-[:HAS_TASK]->(Task)` relationship in Neo4j.
async fn get_plan_id_for_task(state: &ServerState, task_id: Uuid) -> Option<Uuid> {
    match state
        .orchestrator
        .neo4j()
        .get_plan_id_for_task(task_id)
        .await
    {
        Ok(plan_id) => plan_id,
        Err(e) => {
            error!(task_id = %task_id, error = %e, "get_plan_id_for_task: query failed");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Reaction: Plan::StatusChanged(Completed) → log + future episode
// ─────────────────────────────────────────────────────────────

/// When a plan is completed, log the event. In the future, this will also
/// collect an episode if a protocol run is linked to the plan.
async fn on_plan_status_changed(event: CrudEvent, state: Arc<ServerState>) {
    let new_status = event
        .payload
        .get("new_status")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if new_status != "Completed" {
        return;
    }

    let plan_id = match Uuid::parse_str(&event.entity_id) {
        Ok(id) => id,
        Err(_) => return,
    };

    info!(
        plan_id = %plan_id,
        "on_plan_completed: plan completed"
    );

    // Future: check if a ProtocolRun is linked to this plan and collect an episode
    // For now, just log the completion. Episode collection will be added
    // when the protocol-plan linking is more mature.
    let _ = state; // suppress unused warning
    let _ = plan_id;
}

// ─────────────────────────────────────────────────────────────
// Helpers (visible to tests)
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────

/// Extract `Arc<ServerState>` from the reactor's opaque context.
///
/// The reactor stores context as `Arc<dyn Any + Send + Sync>`. Since we know
/// it's always `Arc<ServerState>` for our built-in reactions, we downcast it.
fn extract_state(ctx: Arc<dyn std::any::Any + Send + Sync>) -> Arc<ServerState> {
    ctx.downcast::<ServerState>()
        .expect("reactor context must be Arc<ServerState>")
}

/// Register all built-in reactions on the given [`ReactorBuilder`].
///
/// This is called during application startup to wire up the automatic
/// event-driven behaviors.
///
/// # Reactions registered
///
/// 1. **project-synced** — `Project::Synced` → bootstrap/update knowledge fabric
/// 2. **task-completed** — `Task::StatusChanged` → check plan auto-completion
/// 3. **plan-completed** — `Plan::StatusChanged` → log + future episode collection
pub fn register_builtin_reactions(
    builder: ReactorBuilder,
    _state: Arc<ServerState>,
) -> ReactorBuilder {
    // Enable persistent EventTrigger support via Neo4j
    let builder = builder.with_trigger_support(
        _state.orchestrator.neo4j_arc(),
        _state.event_bus.clone() as Arc<dyn super::types::EventEmitter>,
    );

    builder
        .on(
            "project-synced",
            Some(EntityType::Project),
            Some(CrudAction::Synced),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_project_synced(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "task-completed",
            Some(EntityType::Task),
            Some(CrudAction::StatusChanged),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_task_status_changed(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "plan-completed",
            Some(EntityType::Plan),
            Some(CrudAction::StatusChanged),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_plan_status_changed(event, extract_state(ctx)).await;
                })
            }),
        )
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::events::{EventBus, HybridEmitter};
    use crate::neo4j::models::{PlanStatus, TaskStatus};
    use crate::orchestrator::{watcher::FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_plan, test_project, test_task};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    // ── helpers ──────────────────────────────────────────────────

    /// Build a [`CrudEvent`] with the given fields.
    fn make_event(
        entity_type: EntityType,
        action: CrudAction,
        entity_id: &str,
        payload: serde_json::Value,
    ) -> CrudEvent {
        CrudEvent::new(entity_type, action, entity_id).with_payload(payload)
    }

    /// Build a minimal mock [`ServerState`] for reaction tests.
    async fn mock_server_state() -> Arc<ServerState> {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(RwLock::new(FileWatcher::new(orchestrator.clone())));
        let event_bus = Arc::new(HybridEmitter::new(Arc::new(EventBus::default())));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        })
    }

    // ── extract_state ───────────────────────────────────────────

    #[tokio::test]
    async fn extract_state_succeeds_with_server_state() {
        let state = mock_server_state().await;
        let any: Arc<dyn std::any::Any + Send + Sync> = state.clone();
        // Should not panic
        let recovered = extract_state(any);
        assert_eq!(recovered.server_port, state.server_port);
    }

    #[test]
    #[should_panic(expected = "reactor context must be Arc<ServerState>")]
    fn extract_state_panics_on_wrong_type() {
        let wrong: Arc<dyn std::any::Any + Send + Sync> = Arc::new(42u32);
        let _ = extract_state(wrong);
    }

    // ── get_plan_id_for_task ────────────────────────────────────

    #[tokio::test]
    async fn get_plan_id_for_task_returns_none_when_no_relationship() {
        let state = mock_server_state().await;
        let task_id = Uuid::new_v4();
        let result = get_plan_id_for_task(&state, task_id).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn get_plan_id_for_task_returns_plan_when_linked() {
        let state = mock_server_state().await;
        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
        let plan_id = plan.id;
        state.orchestrator.neo4j().create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Completed;
        let task_id = task.id;
        state
            .orchestrator
            .neo4j()
            .create_task(plan_id, &task)
            .await
            .unwrap();

        let result = get_plan_id_for_task(&state, task_id).await;
        assert_eq!(result, Some(plan_id));
    }

    // ── on_project_synced ───────────────────────────────────────

    #[tokio::test]
    async fn on_project_synced_returns_on_invalid_uuid() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            "not-a-uuid",
            json!({"is_first_sync": true}),
        );
        // Should return early without panicking
        on_project_synced(event, state).await;
    }

    #[tokio::test]
    async fn on_project_synced_returns_when_project_not_found() {
        let state = mock_server_state().await;
        let id = Uuid::new_v4();
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            &id.to_string(),
            json!({"is_first_sync": true}),
        );
        // Project doesn't exist in the mock store — should return early
        on_project_synced(event, state).await;
    }

    #[tokio::test]
    async fn on_project_synced_incremental_does_not_panic() {
        let state = mock_server_state().await;
        let project = test_project();
        state
            .orchestrator
            .neo4j()
            .create_project(&project)
            .await
            .unwrap();
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            &project.id.to_string(),
            json!({"is_first_sync": false}),
        );
        // Incremental sync path — triggers debouncer, should not panic
        on_project_synced(event, state).await;
    }

    #[tokio::test]
    async fn on_project_synced_missing_payload_defaults_to_incremental() {
        let state = mock_server_state().await;
        let project = test_project();
        state
            .orchestrator
            .neo4j()
            .create_project(&project)
            .await
            .unwrap();
        // No is_first_sync in payload — should default to false
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            &project.id.to_string(),
            json!({}),
        );
        on_project_synced(event, state).await;
    }

    // ── on_task_status_changed ──────────────────────────────────

    #[tokio::test]
    async fn on_task_status_changed_ignores_non_completed() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "InProgress", "old_status": "Pending"}),
        );
        // Should return early (not "Completed")
        on_task_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_task_status_changed_ignores_missing_status() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({}),
        );
        // Empty payload → new_status defaults to "" → not "Completed"
        on_task_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_task_status_changed_returns_on_invalid_uuid() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            "bad-uuid",
            json!({"new_status": "Completed"}),
        );
        on_task_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_task_status_changed_returns_when_no_parent_plan() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "Completed"}),
        );
        // Task not linked to any plan → early return
        on_task_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_task_status_changed_auto_completes_plan_when_all_done() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Create a plan in InProgress state
        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        // Create a single task, already completed
        let mut task = test_task();
        task.status = TaskStatus::Completed;
        let task_id = task.id;
        neo4j.create_task(plan_id, &task).await.unwrap();

        // Fire the event
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &task_id.to_string(),
            json!({"new_status": "Completed", "old_status": "InProgress"}),
        );
        on_task_status_changed(event, state.clone()).await;

        // Verify plan was auto-completed
        let updated_plan = neo4j.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(updated_plan.status, PlanStatus::Completed);
    }

    #[tokio::test]
    async fn on_task_status_changed_does_not_complete_plan_with_pending_tasks() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        // Task 1: completed
        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        let t1_id = t1.id;
        neo4j.create_task(plan_id, &t1).await.unwrap();

        // Task 2: still pending
        let mut t2 = test_task();
        t2.status = TaskStatus::Pending;
        neo4j.create_task(plan_id, &t2).await.unwrap();

        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &t1_id.to_string(),
            json!({"new_status": "Completed"}),
        );
        on_task_status_changed(event, state.clone()).await;

        // Plan should still be InProgress
        let updated_plan = neo4j.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(updated_plan.status, PlanStatus::InProgress);
    }

    #[tokio::test]
    async fn on_task_status_changed_completes_plan_with_failed_and_completed() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        // One completed, one failed
        let mut t1 = test_task();
        t1.status = TaskStatus::Completed;
        let t1_id = t1.id;
        neo4j.create_task(plan_id, &t1).await.unwrap();

        let mut t2 = test_task();
        t2.status = TaskStatus::Failed;
        neo4j.create_task(plan_id, &t2).await.unwrap();

        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &t1_id.to_string(),
            json!({"new_status": "Completed"}),
        );
        on_task_status_changed(event, state.clone()).await;

        // Failed tasks don't block completion
        let updated_plan = neo4j.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(updated_plan.status, PlanStatus::Completed);
    }

    #[tokio::test]
    async fn on_task_status_changed_skips_if_plan_not_in_progress() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Plan in Draft status — auto-complete should be skipped
        let mut plan = test_plan();
        plan.status = PlanStatus::Draft;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Completed;
        let task_id = task.id;
        neo4j.create_task(plan_id, &task).await.unwrap();

        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &task_id.to_string(),
            json!({"new_status": "Completed"}),
        );
        on_task_status_changed(event, state.clone()).await;

        // Plan should remain Draft
        let updated_plan = neo4j.get_plan(plan_id).await.unwrap().unwrap();
        assert_eq!(updated_plan.status, PlanStatus::Draft);
    }

    // ── on_plan_status_changed ──────────────────────────────────

    #[tokio::test]
    async fn on_plan_status_changed_ignores_non_completed() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "InProgress", "old_status": "Draft"}),
        );
        on_plan_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_status_changed_ignores_missing_status() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({}),
        );
        on_plan_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_status_changed_returns_on_invalid_uuid() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            "not-a-uuid",
            json!({"new_status": "Completed"}),
        );
        on_plan_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_status_changed_logs_completed() {
        let state = mock_server_state().await;
        let plan_id = Uuid::new_v4();
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            &plan_id.to_string(),
            json!({"new_status": "Completed", "old_status": "InProgress"}),
        );
        // Should log and return without panicking
        on_plan_status_changed(event, state).await;
    }

    // ── register_builtin_reactions ──────────────────────────────

    #[tokio::test]
    async fn register_builtin_reactions_builds_without_panic() {
        let state = mock_server_state().await;
        let bus = Arc::new(EventBus::default());
        let rx = bus.subscribe();
        let builder =
            ReactorBuilder::new(rx, state.clone() as Arc<dyn std::any::Any + Send + Sync>);
        // Should register 3 reactions without panicking
        let _builder = register_builtin_reactions(builder, state);
    }

    // ── CrudEvent payload parsing (pure logic) ─────────────────

    #[test]
    fn make_event_sets_payload_correctly() {
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            "abc",
            json!({"new_status": "Completed", "old_status": "Pending"}),
        );
        assert_eq!(event.entity_type, EntityType::Task);
        assert_eq!(event.action, CrudAction::StatusChanged);
        assert_eq!(event.entity_id, "abc");
        assert_eq!(event.payload["new_status"], "Completed");
        assert_eq!(event.payload["old_status"], "Pending");
    }

    #[test]
    fn payload_get_new_status_extracts_string() {
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "Completed"}),
        );
        let status = event
            .payload
            .get("new_status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(status, "Completed");
    }

    #[test]
    fn payload_missing_new_status_defaults_to_empty() {
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({}),
        );
        let status = event
            .payload
            .get("new_status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(status, "");
    }

    #[test]
    fn payload_is_first_sync_parsing() {
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            &Uuid::new_v4().to_string(),
            json!({"is_first_sync": true}),
        );
        let is_first = event
            .payload
            .get("is_first_sync")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(is_first);
    }

    #[test]
    fn payload_is_first_sync_missing_defaults_false() {
        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            &Uuid::new_v4().to_string(),
            json!({}),
        );
        let is_first = event
            .payload
            .get("is_first_sync")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(!is_first);
    }

    #[test]
    fn uuid_parse_valid() {
        let id = Uuid::new_v4();
        assert!(Uuid::parse_str(&id.to_string()).is_ok());
    }

    #[test]
    fn uuid_parse_invalid() {
        assert!(Uuid::parse_str("not-a-uuid").is_err());
        assert!(Uuid::parse_str("").is_err());
        assert!(Uuid::parse_str("12345").is_err());
    }
}
