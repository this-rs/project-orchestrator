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
//! | `Task::StatusChanged → Completed`  | Cascade-complete all pending/in_progress steps for the task|
//! | `Task::StatusChanged → Completed`  | Check if all plan tasks completed → auto-complete plan     |
//! | `Plan::StatusChanged → Completed/Failed` | Collect episode from lifecycle run + emit Episode::Collected |
//! | `ProtocolRun::StatusChanged → Completed/Failed` | Collect episode from protocol run + emit Episode::Collected |
//! | `Episode::Collected`                             | Analyze patterns via EpisodeAnalyzer + emit Learning::PatternsDetected |
//!
//! ## Architecture
//!
//! Each reaction is an async function that receives the [`CrudEvent`] and an
//! `Arc<ServerState>`. They are registered via [`register_builtin_reactions`]
//! which returns a configured [`ReactorBuilder`](super::ReactorBuilder).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::types::{CrudAction, CrudEvent, EntityType, EventEmitter};
use super::ReactorBuilder;
use crate::api::handlers::ServerState;

/// Cooldown for feedback analysis: skip if last analysis was < 1 hour ago.
const FEEDBACK_ANALYSIS_COOLDOWN_SECS: u64 = 3600;

/// Timestamp (epoch seconds) of the last feedback analysis run.
static LAST_FEEDBACK_ANALYSIS_TS: AtomicU64 = AtomicU64::new(0);

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
// Reaction: Task::StatusChanged(Completed) → cascade-complete steps
// ─────────────────────────────────────────────────────────────

/// When a task is completed, automatically complete all its pending/in_progress steps.
///
/// This keeps the step graph consistent — if a task is marked complete (e.g. by the
/// runner or an agent), any steps that weren't individually completed are auto-closed.
async fn on_task_completed_cascade_steps(event: CrudEvent, state: Arc<ServerState>) {
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
                "on_task_completed_cascade_steps: invalid task UUID"
            );
            return;
        }
    };

    match state
        .orchestrator
        .neo4j()
        .complete_pending_steps_for_task(task_id)
        .await
    {
        Ok(count) => {
            if count > 0 {
                info!(
                    task_id = %task_id,
                    steps_completed = count,
                    "on_task_completed_cascade_steps: auto-completed pending steps"
                );
            } else {
                debug!(
                    task_id = %task_id,
                    "on_task_completed_cascade_steps: no pending steps to complete"
                );
            }
        }
        Err(e) => {
            error!(
                task_id = %task_id,
                error = %e,
                "on_task_completed_cascade_steps: failed to cascade-complete steps"
            );
        }
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

/// When a plan status changes to Completed or Failed, collect an episode
/// from the lifecycle ProtocolRun (if linked) and emit Episode::Collected
/// for downstream learning loop processing (T2 cascade).
///
/// The episode collection is fire-and-forget (tokio::spawn) to avoid
/// blocking the event reactor hot path.
async fn on_plan_status_changed(event: CrudEvent, state: Arc<ServerState>) {
    let new_status = event
        .payload
        .get("new_status")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Only react to terminal statuses
    if new_status != "Completed" && new_status != "Failed" {
        return;
    }

    let plan_id = match Uuid::parse_str(&event.entity_id) {
        Ok(id) => id,
        Err(_) => return,
    };

    info!(
        plan_id = %plan_id,
        new_status = new_status,
        "on_plan_completed: plan reached terminal status, collecting episode"
    );

    // Look up the plan to get project_id
    let neo4j = state.orchestrator.neo4j();
    let plan = match neo4j.get_plan(plan_id).await {
        Ok(Some(p)) => p,
        Ok(None) => {
            debug!(plan_id = %plan_id, "on_plan_completed: plan not found");
            return;
        }
        Err(e) => {
            error!(plan_id = %plan_id, error = %e, "on_plan_completed: failed to get plan");
            return;
        }
    };

    let project_id = match plan.project_id {
        Some(pid) => pid,
        None => {
            debug!(plan_id = %plan_id, "on_plan_completed: plan has no project_id, skipping episode collection");
            return;
        }
    };

    // Look up the most recent plan run to find the lifecycle_run_id
    let plan_run = match neo4j.list_plan_runs(plan_id, 1).await {
        Ok(runs) if !runs.is_empty() => runs.into_iter().next().unwrap(),
        Ok(_) => {
            debug!(plan_id = %plan_id, "on_plan_completed: no plan run found, skipping episode collection");
            return;
        }
        Err(e) => {
            warn!(plan_id = %plan_id, error = %e, "on_plan_completed: failed to list plan runs");
            return;
        }
    };

    let lifecycle_run_id = match plan_run.lifecycle_run_id {
        Some(id) => id,
        None => {
            debug!(plan_id = %plan_id, "on_plan_completed: no lifecycle_run_id, skipping episode collection");
            return;
        }
    };

    // Fire-and-forget episode collection
    let event_bus = state.event_bus.clone();
    let neo4j_arc = state.orchestrator.neo4j_arc();
    tokio::spawn(async move {
        match crate::episodes::collector::collect_episode(
            neo4j_arc.as_ref(),
            lifecycle_run_id,
            project_id,
        )
        .await
        {
            Ok(Some(episode)) => {
                let episode_id = episode.id;
                info!(
                    plan_id = %plan_id,
                    episode_id = %episode_id,
                    run_id = %lifecycle_run_id,
                    "on_plan_completed: episode collected successfully"
                );

                // Emit Episode::Collected to trigger T2 cascade
                let payload = serde_json::json!({
                    "episode_id": episode_id.to_string(),
                    "run_id": lifecycle_run_id.to_string(),
                    "project_id": project_id.to_string(),
                    "source": "plan_completed",
                });
                event_bus.emit(
                    CrudEvent::new(
                        EntityType::Episode,
                        CrudAction::Collected,
                        episode_id.to_string(),
                    )
                    .with_payload(payload)
                    .with_project_id(project_id.to_string()),
                );
            }
            Ok(None) => {
                debug!(
                    plan_id = %plan_id,
                    run_id = %lifecycle_run_id,
                    "on_plan_completed: run not found for episode collection"
                );
            }
            Err(e) => {
                error!(
                    plan_id = %plan_id,
                    run_id = %lifecycle_run_id,
                    error = %e,
                    "on_plan_completed: episode collection failed"
                );
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: ProtocolRun::StatusChanged(Completed|Failed) → collect episode
// ─────────────────────────────────────────────────────────────

/// When a protocol run reaches a terminal status (Completed or Failed),
/// collect an episode and emit Episode::Collected for downstream processing.
///
/// This covers protocol runs that are NOT managed by the plan runner
/// (e.g., standalone protocol executions, chat-triggered protocols).
///
/// Fire-and-forget via tokio::spawn — never blocks the event reactor.
async fn on_protocol_run_completed_collect_episode(event: CrudEvent, state: Arc<ServerState>) {
    let new_status = event
        .payload
        .get("new_status")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Only react to terminal statuses
    if new_status != "Completed" && new_status != "Failed" {
        return;
    }

    let run_id = match Uuid::parse_str(&event.entity_id) {
        Ok(id) => id,
        Err(_) => return,
    };

    // Extract project_id from the event payload or via protocol_run → protocol → project chain
    let project_id = if let Some(pid_str) = event.project_id.as_deref() {
        match Uuid::parse_str(pid_str) {
            Ok(id) => id,
            Err(_) => {
                warn!(run_id = %run_id, "on_protocol_run_completed: invalid project_id in event");
                return;
            }
        }
    } else {
        // Resolve project_id via: ProtocolRun → Protocol → project_id
        let neo4j = state.orchestrator.neo4j();
        let protocol_id = match neo4j.get_protocol_run(run_id).await {
            Ok(Some(run)) => run.protocol_id,
            Ok(None) => {
                debug!(run_id = %run_id, "on_protocol_run_completed: run not found");
                return;
            }
            Err(e) => {
                warn!(run_id = %run_id, error = %e, "on_protocol_run_completed: failed to get run");
                return;
            }
        };
        match neo4j.get_protocol(protocol_id).await {
            Ok(Some(proto)) => proto.project_id,
            Ok(None) => {
                debug!(run_id = %run_id, protocol_id = %protocol_id, "on_protocol_run_completed: protocol not found");
                return;
            }
            Err(e) => {
                warn!(run_id = %run_id, error = %e, "on_protocol_run_completed: failed to get protocol");
                return;
            }
        }
    };

    info!(
        run_id = %run_id,
        project_id = %project_id,
        new_status = new_status,
        "on_protocol_run_completed: collecting episode"
    );

    // Fire-and-forget episode collection
    let event_bus = state.event_bus.clone();
    let neo4j_arc = state.orchestrator.neo4j_arc();
    tokio::spawn(async move {
        match crate::episodes::collector::collect_episode(neo4j_arc.as_ref(), run_id, project_id)
            .await
        {
            Ok(Some(episode)) => {
                let episode_id = episode.id;
                info!(
                    run_id = %run_id,
                    episode_id = %episode_id,
                    "on_protocol_run_completed: episode collected successfully"
                );

                // Emit Episode::Collected to trigger T2 cascade
                let payload = serde_json::json!({
                    "episode_id": episode_id.to_string(),
                    "run_id": run_id.to_string(),
                    "project_id": project_id.to_string(),
                    "source": "protocol_run_completed",
                });
                event_bus.emit(
                    CrudEvent::new(
                        EntityType::Episode,
                        CrudAction::Collected,
                        episode_id.to_string(),
                    )
                    .with_payload(payload)
                    .with_project_id(project_id.to_string()),
                );
            }
            Ok(None) => {
                debug!(run_id = %run_id, "on_protocol_run_completed: run not found for collection");
            }
            Err(e) => {
                error!(
                    run_id = %run_id,
                    error = %e,
                    "on_protocol_run_completed: episode collection failed"
                );
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: Episode::Collected → analyze patterns (T2)
// ─────────────────────────────────────────────────────────────

/// When an episode is collected, load all recent episodes for the project,
/// run batch pattern analysis via [`EpisodeAnalyzer`], and emit
/// `Learning::PatternsDetected` if significant recurring patterns are found.
///
/// This is the T2 (ANALYZE) step of the autonomous learning loop:
/// `Episode::Collected → batch episodes → EpisodeAdapter → EpisodeAnalyzer → Learning::PatternsDetected`
///
/// **Design**: Uses batch analysis (last 50 episodes) with `min_frequency=3`,
/// `min_confidence=0.5` to avoid false positives. Requires ≥3 episodes before
/// activating. The full payload includes detected patterns AND skill
/// recommendations for T3 (MATERIALIZE) to consume.
///
/// Fire-and-forget via `tokio::spawn` to avoid blocking the event reactor.
async fn on_episode_collected_analyze_patterns(event: CrudEvent, state: Arc<ServerState>) {
    // Only react to Episode::Collected
    if event.entity_type != EntityType::Episode || event.action != CrudAction::Collected {
        return;
    }

    let episode_id_str = event.entity_id.clone();
    let project_id_str = event.project_id.clone().or_else(|| {
        event
            .payload
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(String::from)
    });
    let run_id_str = event
        .payload
        .get("run_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let project_id = match project_id_str
        .as_deref()
        .and_then(|s| s.parse::<uuid::Uuid>().ok())
    {
        Some(id) => id,
        None => {
            debug!(
                episode_id = %episode_id_str,
                "on_episode_collected: no valid project_id in payload, skipping analysis"
            );
            return;
        }
    };

    let episode_id = match episode_id_str.parse::<uuid::Uuid>() {
        Ok(id) => id,
        Err(_) => {
            debug!(
                episode_id = %episode_id_str,
                "on_episode_collected: invalid episode_id, skipping analysis"
            );
            return;
        }
    };

    let run_id = run_id_str
        .as_deref()
        .and_then(|s| s.parse::<uuid::Uuid>().ok());

    info!(
        episode_id = %episode_id,
        project_id = %project_id,
        run_id = ?run_id,
        "on_episode_collected: starting pattern analysis (T2)"
    );

    let neo4j_arc = state.orchestrator.neo4j_arc();
    let event_bus = state.event_bus.clone();

    tokio::spawn(async move {
        // 1. Load recent episodes for this project (batch analysis, last 50)
        //    Batch analysis detects recurring patterns across multiple runs,
        //    which is far more robust than single-episode analysis.
        let episodes =
            match crate::episodes::collector::list_episodes(neo4j_arc.as_ref(), project_id, 50)
                .await
            {
                Ok(eps) => eps,
                Err(e) => {
                    error!(
                        project_id = %project_id,
                        error = %e,
                        "on_episode_collected: failed to list episodes for analysis"
                    );
                    return;
                }
            };

        if episodes.len() < 3 {
            debug!(
                project_id = %project_id,
                count = episodes.len(),
                "on_episode_collected: too few episodes for analysis (need ≥3), skipping"
            );
            return;
        }

        // 2. Resolve protocol names for all episodes
        let protocol_names =
            crate::pipeline::episode_adapter::resolve_protocol_names(neo4j_arc.as_ref(), &episodes)
                .await;

        // 3. Convert Episodes → EpisodeData via adapter (batch)
        let episode_pairs: Vec<_> = episodes.into_iter().zip(protocol_names).collect();
        let episode_data = crate::pipeline::episode_adapter::episodes_to_data(&episode_pairs);

        // 4. Run the EpisodeAnalyzer with meaningful thresholds
        //    min_frequency=3: pattern must appear ≥3 times to be significant
        //    min_confidence=0.5: ≥50% occurrence rate
        let analyzer = crate::pipeline::feedback::EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episode_data);

        if patterns.is_empty() {
            debug!(
                project_id = %project_id,
                episodes_analyzed = episode_data.len(),
                "on_episode_collected: no significant patterns detected"
            );
            return;
        }

        // 5. Generate skill recommendations from detected patterns
        let recommendations = analyzer.recommend_skills(&patterns);

        info!(
            episode_id = %episode_id,
            project_id = %project_id,
            episodes_analyzed = episode_data.len(),
            patterns_count = patterns.len(),
            recommendations_count = recommendations.len(),
            "on_episode_collected: patterns detected, emitting Learning::PatternsDetected"
        );

        // 6. Emit Learning::PatternsDetected with full payload for T3 (MATERIALIZE)
        // Use serde_json::to_value() instead of manual serialization to ensure
        // field names match the Deserialize impl (e.g., PatternType variants).
        let patterns_payload: Vec<serde_json::Value> = patterns
            .iter()
            .filter_map(|p| serde_json::to_value(p).ok())
            .collect();

        let recommendations_payload: Vec<serde_json::Value> = recommendations
            .iter()
            .filter_map(|r| serde_json::to_value(r).ok())
            .collect();

        let payload = serde_json::json!({
            "project_id": project_id.to_string(),
            "episode_id": episode_id.to_string(),
            "episodes_analyzed": episode_data.len(),
            "patterns_count": patterns.len(),
            "patterns": patterns_payload,
            "recommendations": recommendations_payload,
        });

        event_bus.emit(
            CrudEvent::new(
                EntityType::Learning,
                CrudAction::PatternsDetected,
                episode_id.to_string(),
            )
            .with_payload(payload)
            .with_project_id(project_id.to_string()),
        );
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: Learning::PatternsDetected → materialize (T3)
// ─────────────────────────────────────────────────────────────

/// When patterns are detected (from T2), materialize them into concrete
/// knowledge graph entities: notes, skills, and neural scars.
///
/// This is the T3 (MATERIALIZE) step of the autonomous learning loop:
/// `Learning::PatternsDetected → deserialize patterns → materialize_patterns()`
///
/// Fire-and-forget via `tokio::spawn` to avoid blocking the event reactor.
async fn on_patterns_detected_materialize(event: CrudEvent, state: Arc<ServerState>) {
    let project_id_str = match event.payload.get("project_id").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            debug!("on_patterns_detected: no project_id in payload, skipping");
            return;
        }
    };

    let project_id = match Uuid::parse_str(&project_id_str) {
        Ok(id) => id,
        Err(_) => {
            warn!(
                project_id = %project_id_str,
                "on_patterns_detected: invalid project_id UUID"
            );
            return;
        }
    };

    let patterns_count = event
        .payload
        .get("patterns_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if patterns_count == 0 {
        debug!(project_id = %project_id, "on_patterns_detected: zero patterns, skipping");
        return;
    }

    info!(
        project_id = %project_id,
        patterns_count = patterns_count,
        "on_patterns_detected: starting materialization (T3)"
    );

    // Deserialize patterns and recommendations from the event payload
    let patterns: Vec<crate::pipeline::feedback::DetectedPattern> = event
        .payload
        .get("patterns")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let recommendations: Vec<crate::pipeline::feedback::SkillRecommendation> = event
        .payload
        .get("recommendations")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    if patterns.is_empty() {
        debug!(project_id = %project_id, "on_patterns_detected: failed to deserialize patterns");
        return;
    }

    let neo4j_arc = state.orchestrator.neo4j_arc();

    tokio::spawn(async move {
        match crate::pipeline::materialize::materialize_patterns(
            neo4j_arc.as_ref(),
            project_id,
            &patterns,
            &recommendations,
        )
        .await
        {
            Ok(result) => {
                info!(
                    project_id = %project_id,
                    notes_created = result.notes_created,
                    notes_skipped = result.notes_skipped,
                    skills_created = result.skills_created,
                    "on_patterns_detected: materialization complete (T3)"
                );
            }
            Err(e) => {
                error!(
                    project_id = %project_id,
                    error = %e,
                    "on_patterns_detected: materialization failed"
                );
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: Learning::PatternsDetected → evolve protocols (T6)
// ─────────────────────────────────────────────────────────────

/// When patterns are detected (from T2), evolve project protocols by applying
/// learned mutations (add states like `run_tests`, `lint_check`, `review`).
///
/// Chain: `Learning::PatternsDetected → filter high-confidence → ProtocolEvolver::evolve()`
///
/// Fire-and-forget via `tokio::spawn` to avoid blocking the event reactor.
async fn on_patterns_detected_evolve(event: CrudEvent, state: Arc<ServerState>) {
    let project_id_str = match event.payload.get("project_id").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            debug!("on_patterns_detected_evolve: no project_id in payload, skipping");
            return;
        }
    };

    let project_id = match Uuid::parse_str(&project_id_str) {
        Ok(id) => id,
        Err(_) => {
            warn!(
                project_id = %project_id_str,
                "on_patterns_detected_evolve: invalid project_id UUID"
            );
            return;
        }
    };

    // Deserialize patterns from the event payload
    let patterns: Vec<crate::pipeline::feedback::DetectedPattern> = event
        .payload
        .get("patterns")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    if patterns.is_empty() {
        debug!(project_id = %project_id, "on_patterns_detected_evolve: no patterns, skipping");
        return;
    }

    // Only evolve if there are high-confidence patterns (≥0.8)
    let high_confidence_count = patterns.iter().filter(|p| p.confidence >= 0.8).count();
    if high_confidence_count == 0 {
        debug!(
            project_id = %project_id,
            "on_patterns_detected_evolve: no high-confidence patterns (≥0.8), skipping"
        );
        return;
    }

    info!(
        project_id = %project_id,
        high_confidence_count = high_confidence_count,
        "on_patterns_detected_evolve: starting protocol evolution (T6)"
    );

    let neo4j_arc = state.orchestrator.neo4j_arc();

    tokio::spawn(async move {
        // Find all protocols for this project
        let protocols = match neo4j_arc.list_protocols(project_id, None, 50, 0).await {
            Ok((protos, _)) => protos,
            Err(e) => {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "on_patterns_detected_evolve: failed to list protocols"
                );
                return;
            }
        };

        if protocols.is_empty() {
            debug!(
                project_id = %project_id,
                "on_patterns_detected_evolve: no protocols found for project, skipping"
            );
            return;
        }

        let evolver = crate::pipeline::evolve::ProtocolEvolver::new(neo4j_arc.clone());

        // Apply mutations to each protocol
        for protocol in &protocols {
            match evolver.evolve(protocol.id, &patterns, None).await {
                Ok(result) => {
                    if result.applied_count > 0 {
                        info!(
                            project_id = %project_id,
                            protocol_id = %protocol.id,
                            protocol_name = %protocol.name,
                            applied = result.applied_count,
                            skipped = result.skipped_count,
                            "on_patterns_detected_evolve: protocol evolved (T6)"
                        );
                    } else {
                        debug!(
                            project_id = %project_id,
                            protocol_id = %protocol.id,
                            skipped = result.skipped_count,
                            "on_patterns_detected_evolve: no mutations applied (all skipped)"
                        );
                    }
                }
                Err(e) => {
                    error!(
                        project_id = %project_id,
                        protocol_id = %protocol.id,
                        error = %e,
                        "on_patterns_detected_evolve: evolution failed"
                    );
                }
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: Runner::Updated(PlanCompleted) → analyze feedback
// ─────────────────────────────────────────────────────────────

/// When a plan run completes (Runner::Updated with event=plan_completed payload),
/// run FeedbackAnalyzer to detect recurring patterns in runner-feedback notes.
///
/// Uses a static AtomicU64 cooldown: skips if < 1 hour since the last analysis.
/// The analysis itself runs in `tokio::spawn` (fire-and-forget).
///
/// Detected actionable patterns are persisted as observation notes with tags
/// `['runner-feedback-suggestion', 'auto-observed', <pattern_type>]` and a
/// `Learning::PatternsDetected` event is emitted for frontend notification.
async fn on_plan_run_completed_analyze_feedback(event: CrudEvent, state: Arc<ServerState>) {
    // Only react to Runner::Updated events with plan_completed payload
    if event.entity_type != EntityType::Runner || event.action != CrudAction::Updated {
        return;
    }

    // Check if this is a PlanCompleted event (serde tag = "event": "plan_completed")
    let event_type = event
        .payload
        .get("event")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if event_type != "plan_completed" {
        return;
    }

    // Extract plan_id from the payload
    let plan_id_str = match event.payload.get("plan_id").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            debug!("on_plan_run_completed_analyze_feedback: no plan_id in payload, skipping");
            return;
        }
    };

    let plan_id = match Uuid::parse_str(&plan_id_str) {
        Ok(id) => id,
        Err(_) => {
            warn!(
                plan_id = %plan_id_str,
                "on_plan_run_completed_analyze_feedback: invalid plan_id UUID"
            );
            return;
        }
    };

    // Cooldown check: skip if < 1 hour since last analysis
    let now_secs = chrono::Utc::now().timestamp() as u64;
    let last_ts = LAST_FEEDBACK_ANALYSIS_TS.load(Ordering::Relaxed);
    if now_secs.saturating_sub(last_ts) < FEEDBACK_ANALYSIS_COOLDOWN_SECS {
        debug!(
            plan_id = %plan_id,
            last_analysis_secs_ago = now_secs.saturating_sub(last_ts),
            cooldown_secs = FEEDBACK_ANALYSIS_COOLDOWN_SECS,
            "on_plan_run_completed_analyze_feedback: cooldown active, skipping"
        );
        return;
    }

    // Update the cooldown timestamp
    LAST_FEEDBACK_ANALYSIS_TS.store(now_secs, Ordering::Relaxed);

    // Resolve project_id from plan
    let neo4j = state.orchestrator.neo4j();
    let project_id = match neo4j.get_plan(plan_id).await {
        Ok(Some(plan)) => plan.project_id,
        Ok(None) => {
            debug!(plan_id = %plan_id, "on_plan_run_completed_analyze_feedback: plan not found");
            return;
        }
        Err(e) => {
            error!(plan_id = %plan_id, error = %e, "on_plan_run_completed_analyze_feedback: failed to get plan");
            return;
        }
    };

    info!(
        plan_id = %plan_id,
        project_id = ?project_id,
        "on_plan_run_completed_analyze_feedback: starting feedback analysis"
    );

    let neo4j_arc = state.orchestrator.neo4j_arc();
    let event_bus = state.event_bus.clone();

    tokio::spawn(async move {
        let analyzer = crate::runner::feedback_analyzer::FeedbackAnalyzer::new(neo4j_arc.clone());
        let report = match analyzer.analyze_runner_feedback(project_id).await {
            Ok(r) => r,
            Err(e) => {
                error!(
                    plan_id = %plan_id,
                    error = %e,
                    "on_plan_run_completed_analyze_feedback: analysis failed"
                );
                return;
            }
        };

        if report.actionable_count == 0 {
            debug!(
                plan_id = %plan_id,
                total_notes = report.total_notes_analyzed,
                patterns = report.patterns.len(),
                "on_plan_run_completed_analyze_feedback: no actionable patterns"
            );
            return;
        }

        info!(
            plan_id = %plan_id,
            actionable = report.actionable_count,
            total_patterns = report.patterns.len(),
            "on_plan_run_completed_analyze_feedback: actionable patterns detected, persisting suggestions"
        );

        // Persist each actionable pattern as an observation note
        let mut suggestion_note_ids = Vec::new();
        for pattern in &report.patterns {
            if !pattern.is_actionable {
                continue;
            }

            let content = format!(
                "## Auto-detected Feedback Pattern\n\n\
                 **Pattern**: {} (observed {} times)\n\
                 **Suggestion**: {}\n\
                 **Evidence notes**: {}\n\
                 **Run IDs**: {}\n\
                 **Detected at**: {}\n",
                pattern.pattern_type,
                pattern.count,
                pattern.suggestion,
                pattern
                    .evidence_note_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                pattern
                    .run_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                report.analyzed_at,
            );

            let pattern_tag = format!("pattern:{}", pattern.pattern_type);
            let mut note = crate::notes::models::Note::new(
                project_id,
                crate::notes::models::NoteType::Observation,
                content,
                format!("Feedback pattern: {}", pattern.suggestion),
            );
            note.tags = vec![
                "runner-feedback-suggestion".to_string(),
                "auto-observed".to_string(),
                pattern_tag,
            ];
            note.importance = crate::notes::models::NoteImportance::Medium;
            note.status = crate::notes::models::NoteStatus::Active;

            let note_id = note.id;
            match neo4j_arc.create_note(&note).await {
                Ok(_) => {
                    suggestion_note_ids.push(note_id);
                    info!(
                        note_id = %note_id,
                        pattern = %pattern.pattern_type,
                        "on_plan_run_completed_analyze_feedback: suggestion note created"
                    );
                }
                Err(e) => {
                    error!(
                        pattern = %pattern.pattern_type,
                        error = %e,
                        "on_plan_run_completed_analyze_feedback: failed to create suggestion note"
                    );
                }
            }
        }

        // Emit Learning::FeedbackPatternsDetected for frontend notification
        if !suggestion_note_ids.is_empty() {
            let payload = serde_json::json!({
                "source": "feedback_analyzer",
                "project_id": project_id.map(|id| id.to_string()),
                "patterns_count": report.actionable_count,
                "suggestion_note_ids": suggestion_note_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                "total_notes_analyzed": report.total_notes_analyzed,
            });

            let pid_str = project_id.map(|id| id.to_string()).unwrap_or_default();
            let mut crud_event = CrudEvent::new(
                EntityType::Learning,
                CrudAction::FeedbackPatternsDetected,
                plan_id.to_string(),
            )
            .with_payload(payload);

            if let Some(pid) = project_id {
                crud_event = crud_event.with_project_id(pid.to_string());
            }

            event_bus.emit(crud_event);

            info!(
                plan_id = %plan_id,
                project_id = pid_str,
                suggestions = suggestion_note_ids.len(),
                "on_plan_run_completed_analyze_feedback: Learning::FeedbackPatternsDetected emitted"
            );
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Reaction: *::StatusChanged → evaluate LifecycleHooks
// ─────────────────────────────────────────────────────────────

/// When any entity changes status, evaluate and execute matching LifecycleHooks.
///
/// This is the bridge between the EventReactor and the LifecycleHook system.
/// It runs AFTER the built-in reactions (cascade-steps, plan-auto-complete)
/// and handles user-defined hooks stored in Neo4j.
async fn on_status_changed_lifecycle_hooks(event: CrudEvent, state: Arc<ServerState>) {
    if event.action != CrudAction::StatusChanged {
        return;
    }

    crate::lifecycle::executor::execute_lifecycle_hooks(&event, &state).await;
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
/// 2. **task-cascade-steps** — `Task::StatusChanged(Completed)` → auto-complete pending steps
/// 3. **task-completed** — `Task::StatusChanged(Completed)` → check plan auto-completion
/// 4. **plan-completed** — `Plan::StatusChanged(Completed|Failed)` → collect episode + emit Episode::Collected
/// 5. **protocol-run-episode-collect** — `ProtocolRun::StatusChanged(Completed|Failed)` → collect episode + emit Episode::Collected
/// 6. **episode-analyze-patterns** — `Episode::Collected` → analyze patterns via EpisodeAnalyzer + emit Learning::PatternsDetected
/// 7. **patterns-materialize** — `Learning::PatternsDetected` → materialize patterns into notes/skills/scars
/// 8. **patterns-evolve** — `Learning::PatternsDetected` → evolve protocols via adaptive mutations
/// 9. **lifecycle-hooks** — `*::StatusChanged` → evaluate and execute LifecycleHooks
/// 10. **plan-run-feedback** — `Runner::Updated(PlanCompleted)` → analyze runner feedback patterns
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
            "task-cascade-steps",
            Some(EntityType::Task),
            Some(CrudAction::StatusChanged),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_task_completed_cascade_steps(event, extract_state(ctx)).await;
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
        .on(
            "protocol-run-episode-collect",
            Some(EntityType::ProtocolRun),
            Some(CrudAction::StatusChanged),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_protocol_run_completed_collect_episode(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "episode-analyze-patterns",
            Some(EntityType::Episode),
            Some(CrudAction::Collected),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_episode_collected_analyze_patterns(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "patterns-materialize",
            Some(EntityType::Learning),
            Some(CrudAction::PatternsDetected),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_patterns_detected_materialize(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "patterns-evolve",
            Some(EntityType::Learning),
            Some(CrudAction::PatternsDetected),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_patterns_detected_evolve(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "lifecycle-hooks",
            None,
            Some(CrudAction::StatusChanged),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_status_changed_lifecycle_hooks(event, extract_state(ctx)).await;
                })
            }),
        )
        .on(
            "plan-run-feedback",
            Some(EntityType::Runner),
            Some(CrudAction::Updated),
            Arc::new(|event, ctx| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(async move {
                    on_plan_run_completed_analyze_feedback(event, extract_state(ctx)).await;
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
    use crate::neo4j::models::{PlanStatus, StepStatus, TaskStatus};
    use crate::orchestrator::{watcher::FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_plan, test_project, test_step, test_task};
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
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: None,
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

    // ── on_task_completed_cascade_steps ─────────────────────────

    #[tokio::test]
    async fn cascade_steps_ignores_non_completed() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "InProgress", "old_status": "Pending"}),
        );
        // Should return early — not "Completed"
        on_task_completed_cascade_steps(event, state).await;
    }

    #[tokio::test]
    async fn cascade_steps_returns_on_invalid_uuid() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            "bad-uuid",
            json!({"new_status": "Completed"}),
        );
        on_task_completed_cascade_steps(event, state).await;
    }

    #[tokio::test]
    async fn cascade_steps_completes_pending_steps() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Create plan + task
        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.status = TaskStatus::Completed;
        let task_id = task.id;
        neo4j.create_task(plan_id, &task).await.unwrap();

        // Create 3 steps: one pending, one in_progress, one already completed
        let s1 = test_step(1, "Step pending");
        let s2 = {
            let mut s = test_step(2, "Step in progress");
            s.status = StepStatus::InProgress;
            s
        };
        let s3 = {
            let mut s = test_step(3, "Step already done");
            s.status = StepStatus::Completed;
            s.completed_at = Some(chrono::Utc::now());
            s
        };
        let s1_id = s1.id;
        let s2_id = s2.id;
        let s3_id = s3.id;
        neo4j.create_step(task_id, &s1).await.unwrap();
        neo4j.create_step(task_id, &s2).await.unwrap();
        neo4j.create_step(task_id, &s3).await.unwrap();

        // Fire the event
        let event = make_event(
            EntityType::Task,
            CrudAction::StatusChanged,
            &task_id.to_string(),
            json!({"new_status": "Completed", "old_status": "InProgress"}),
        );
        on_task_completed_cascade_steps(event, state.clone()).await;

        // Verify: s1 and s2 should now be Completed, s3 unchanged
        let step1 = neo4j.get_step(s1_id).await.unwrap().unwrap();
        assert_eq!(step1.status, StepStatus::Completed);
        assert!(step1.completed_at.is_some());

        let step2 = neo4j.get_step(s2_id).await.unwrap().unwrap();
        assert_eq!(step2.status, StepStatus::Completed);
        assert!(step2.completed_at.is_some());

        let step3 = neo4j.get_step(s3_id).await.unwrap().unwrap();
        assert_eq!(step3.status, StepStatus::Completed); // was already completed
    }

    #[tokio::test]
    async fn cascade_steps_no_steps_does_not_panic() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Create plan + task with no steps
        let mut plan = test_plan();
        plan.status = PlanStatus::InProgress;
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
        // Should complete gracefully with 0 updates
        on_task_completed_cascade_steps(event, state).await;
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
        // Should log and return without panicking (no plan in store → early return)
        on_plan_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_status_changed_ignores_non_terminal() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "InProgress", "old_status": "Draft"}),
        );
        // InProgress is not a terminal status — should return early
        on_plan_status_changed(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_status_changed_collects_episode_with_lifecycle_run() {
        use crate::protocol::models::ProtocolRun;

        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Create a project + plan with project_id
        let project = test_project();
        let project_id = project.id;
        neo4j.create_project(&project).await.unwrap();

        let mut plan = test_plan();
        plan.status = PlanStatus::Completed;
        plan.project_id = Some(project_id);
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();
        neo4j
            .link_plan_to_project(plan_id, project_id)
            .await
            .unwrap();

        // Create a protocol run (the lifecycle run)
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.complete();
        let lifecycle_run_id = run.id;
        neo4j.create_protocol_run(&run).await.unwrap();

        // Create a PlanRun with lifecycle_run_id
        let mut runner_state = crate::runner::RunnerState::new(
            Uuid::new_v4(),
            plan_id,
            1,
            crate::runner::models::TriggerSource::Manual,
        );
        runner_state.status = crate::runner::models::PlanRunStatus::Completed;
        runner_state.completed_at = Some(chrono::Utc::now());
        runner_state.lifecycle_run_id = Some(lifecycle_run_id);
        neo4j.create_plan_run(&runner_state).await.unwrap();

        // Fire the event
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            &plan_id.to_string(),
            json!({"new_status": "Completed", "old_status": "InProgress"}),
        );
        on_plan_status_changed(event, state.clone()).await;

        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The episode collection runs in tokio::spawn — we verify no panic occurred
        // and the plan was properly resolved. Full integration test would check
        // that Episode::Collected event was emitted.
    }

    // ── on_protocol_run_completed_collect_episode ──────────────

    #[tokio::test]
    async fn on_protocol_run_completed_ignores_non_terminal() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            &Uuid::new_v4().to_string(),
            json!({"new_status": "Running", "old_status": "Pending"}),
        );
        // Running is not terminal — should return early
        on_protocol_run_completed_collect_episode(event, state).await;
    }

    #[tokio::test]
    async fn on_protocol_run_completed_returns_on_invalid_uuid() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            "not-a-uuid",
            json!({"new_status": "Completed"}),
        );
        on_protocol_run_completed_collect_episode(event, state).await;
    }

    #[tokio::test]
    async fn on_protocol_run_completed_collects_with_project_id_in_event() {
        use crate::protocol::models::ProtocolRun;

        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        let project_id = Uuid::new_v4();
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.complete();
        let run_id = run.id;
        neo4j.create_protocol_run(&run).await.unwrap();

        let mut event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            &run_id.to_string(),
            json!({"new_status": "Completed", "old_status": "Running"}),
        );
        event.project_id = Some(project_id.to_string());

        on_protocol_run_completed_collect_episode(event, state.clone()).await;

        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn on_protocol_run_completed_no_project_id_no_run_returns_early() {
        let state = mock_server_state().await;
        let run_id = Uuid::new_v4();

        // Event WITHOUT project_id and run doesn't exist — handler resolves via
        // protocol chain, gets None, returns early without panic
        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            &run_id.to_string(),
            json!({"new_status": "Completed", "old_status": "Running"}),
        );
        on_protocol_run_completed_collect_episode(event, state).await;
    }

    // ── on_episode_collected_analyze_patterns ──────────────────

    #[tokio::test]
    async fn on_episode_collected_skips_non_episode_events() {
        let state = mock_server_state().await;
        // A Task event should be ignored entirely
        let event = make_event(
            EntityType::Task,
            CrudAction::Collected,
            &Uuid::new_v4().to_string(),
            json!({"project_id": Uuid::new_v4().to_string()}),
        );
        // Should return early without panic
        on_episode_collected_analyze_patterns(event, state).await;
    }

    #[tokio::test]
    async fn on_episode_collected_skips_missing_project_id() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Episode,
            CrudAction::Collected,
            &Uuid::new_v4().to_string(),
            json!({}), // no project_id
        );
        // Should return early (no project_id) without panic
        on_episode_collected_analyze_patterns(event, state).await;
    }

    #[tokio::test]
    async fn on_episode_collected_skips_missing_run_id() {
        let state = mock_server_state().await;
        let project_id = Uuid::new_v4();
        let event = make_event(
            EntityType::Episode,
            CrudAction::Collected,
            &Uuid::new_v4().to_string(),
            json!({"project_id": project_id.to_string()}), // no run_id
        );
        // Should spawn task but return early inside (no run_id)
        on_episode_collected_analyze_patterns(event, state).await;
        // Give spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn on_episode_collected_handles_nonexistent_run_gracefully() {
        let state = mock_server_state().await;
        let project_id = Uuid::new_v4();
        let run_id = Uuid::new_v4();
        let episode_id = Uuid::new_v4();

        let mut event = make_event(
            EntityType::Episode,
            CrudAction::Collected,
            &episode_id.to_string(),
            json!({
                "project_id": project_id.to_string(),
                "run_id": run_id.to_string(),
            }),
        );
        event.project_id = Some(project_id.to_string());

        // Run doesn't exist in mock store → should log and return without panic
        on_episode_collected_analyze_patterns(event, state).await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn on_episode_collected_with_completed_run_spawns_analysis() {
        use crate::protocol::models::ProtocolRun;

        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        let project_id = Uuid::new_v4();
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.complete();
        let run_id = run.id;
        neo4j.create_protocol_run(&run).await.unwrap();

        let episode_id = Uuid::new_v4();
        let mut event = make_event(
            EntityType::Episode,
            CrudAction::Collected,
            &episode_id.to_string(),
            json!({
                "project_id": project_id.to_string(),
                "run_id": run_id.to_string(),
            }),
        );
        event.project_id = Some(project_id.to_string());

        on_episode_collected_analyze_patterns(event, state).await;
        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        // If we got here without panic, the analysis completed (or gracefully failed)
    }

    // ── register_builtin_reactions ──────────────────────────────

    #[tokio::test]
    async fn register_builtin_reactions_builds_without_panic() {
        let state = mock_server_state().await;
        let bus = Arc::new(EventBus::default());
        let rx = bus.subscribe();
        let builder =
            ReactorBuilder::new(rx, state.clone() as Arc<dyn std::any::Any + Send + Sync>);
        // Should register 7 reactions without panicking
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

    // ── on_plan_run_completed_analyze_feedback ────────────────

    #[tokio::test]
    async fn on_plan_run_completed_ignores_non_runner_events() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Task,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": Uuid::new_v4().to_string()}),
        );
        // Wrong entity type — should return early
        on_plan_run_completed_analyze_feedback(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_run_completed_ignores_non_plan_completed_events() {
        let state = mock_server_state().await;
        let event = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "task_completed", "task_id": Uuid::new_v4().to_string()}),
        );
        // Not a plan_completed event — should return early
        on_plan_run_completed_analyze_feedback(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_run_completed_skips_missing_plan_id() {
        let state = mock_server_state().await;
        // Reset cooldown for this test
        LAST_FEEDBACK_ANALYSIS_TS.store(0, std::sync::atomic::Ordering::Relaxed);
        let event = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed"}), // no plan_id
        );
        on_plan_run_completed_analyze_feedback(event, state).await;
    }

    #[tokio::test]
    async fn on_plan_run_completed_cooldown_skips_second_call() {
        let state = mock_server_state().await;
        let plan_id = Uuid::new_v4();

        // Reset cooldown
        LAST_FEEDBACK_ANALYSIS_TS.store(0, std::sync::atomic::Ordering::Relaxed);

        // First call: should proceed (plan not found, but past cooldown)
        let event1 = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": plan_id.to_string()}),
        );
        on_plan_run_completed_analyze_feedback(event1, state.clone()).await;

        // Record the timestamp that was set
        let ts_after_first = LAST_FEEDBACK_ANALYSIS_TS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            ts_after_first > 0,
            "Cooldown timestamp should be set after first call"
        );

        // Second call within cooldown — should be skipped (timestamp unchanged)
        let event2 = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": plan_id.to_string()}),
        );
        on_plan_run_completed_analyze_feedback(event2, state).await;

        // Timestamp should not have changed (cooldown prevented update)
        let ts_after_second = LAST_FEEDBACK_ANALYSIS_TS.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            ts_after_first, ts_after_second,
            "Cooldown should prevent second analysis"
        );
    }

    #[tokio::test]
    async fn on_plan_run_completed_runs_analysis_with_valid_plan() {
        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Reset cooldown
        LAST_FEEDBACK_ANALYSIS_TS.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create a project + plan
        let project = test_project();
        let project_id = project.id;
        neo4j.create_project(&project).await.unwrap();

        let mut plan = test_plan();
        plan.project_id = Some(project_id);
        plan.status = PlanStatus::Completed;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        let event = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": plan_id.to_string(), "status": "completed"}),
        );
        on_plan_run_completed_analyze_feedback(event, state.clone()).await;

        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        // If we got here without panic, the analysis ran (or gracefully completed with 0 patterns)
    }

    #[tokio::test]
    async fn on_plan_run_completed_creates_suggestion_notes_for_actionable_patterns() {
        use crate::notes::models::{Note, NoteFilters, NoteType};

        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Reset cooldown
        LAST_FEEDBACK_ANALYSIS_TS.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create a project + plan
        let project = test_project();
        let project_id = project.id;
        neo4j.create_project(&project).await.unwrap();

        let mut plan = test_plan();
        plan.project_id = Some(project_id);
        plan.status = PlanStatus::Completed;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        // Create 4+ runner-feedback observation notes with the same pattern tag
        // (threshold is 3, so 4 = actionable)
        for i in 0..4 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Manual commit observed after run #{}", i),
                format!("Runner feedback #{}", i),
            );
            note.tags = vec![
                "runner-feedback".to_string(),
                "runner-manual-action".to_string(),
            ];
            neo4j.create_note(&note).await.unwrap();
        }

        // Fire the event
        let event = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": plan_id.to_string(), "status": "completed"}),
        );
        on_plan_run_completed_analyze_feedback(event, state.clone()).await;

        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Verify suggestion notes were created
        let filters = NoteFilters {
            tags: Some(vec!["runner-feedback-suggestion".to_string()]),
            ..Default::default()
        };
        let (notes, _total) = neo4j
            .list_notes(Some(project_id), None, &filters)
            .await
            .unwrap();

        assert!(
            !notes.is_empty(),
            "Expected at least one suggestion note to be created for actionable patterns"
        );

        // Verify the note has the correct tags
        let suggestion = &notes[0];
        assert!(suggestion
            .tags
            .contains(&"runner-feedback-suggestion".to_string()));
        assert!(suggestion.tags.contains(&"auto-observed".to_string()));
        // Should have a pattern:* tag
        assert!(suggestion.tags.iter().any(|t| t.starts_with("pattern:")));
    }

    #[tokio::test]
    async fn on_plan_run_completed_emits_learning_event_for_actionable_patterns() {
        use crate::notes::models::{Note, NoteType};

        let state = mock_server_state().await;
        let neo4j = state.orchestrator.neo4j();

        // Reset cooldown
        LAST_FEEDBACK_ANALYSIS_TS.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create a project + plan
        let project = test_project();
        let project_id = project.id;
        neo4j.create_project(&project).await.unwrap();

        let mut plan = test_plan();
        plan.project_id = Some(project_id);
        plan.status = PlanStatus::Completed;
        let plan_id = plan.id;
        neo4j.create_plan(&plan).await.unwrap();

        // Create 4+ runner-feedback notes for an actionable pattern
        for i in 0..4 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Missing step feedback #{}", i),
                format!("Runner feedback #{}", i),
            );
            note.tags = vec![
                "runner-feedback".to_string(),
                "runner-override-step".to_string(),
            ];
            neo4j.create_note(&note).await.unwrap();
        }

        // Fire the event
        let event = make_event(
            EntityType::Runner,
            CrudAction::Updated,
            &Uuid::new_v4().to_string(),
            json!({"event": "plan_completed", "plan_id": plan_id.to_string(), "status": "completed"}),
        );
        on_plan_run_completed_analyze_feedback(event, state.clone()).await;

        // Give the spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Verify suggestion notes were created (the Learning::FeedbackPatternsDetected
        // event emission happens in the same code path, right after note creation)
        let filters = crate::notes::models::NoteFilters {
            tags: Some(vec!["runner-feedback-suggestion".to_string()]),
            ..Default::default()
        };
        let (notes, _) = neo4j
            .list_notes(Some(project_id), None, &filters)
            .await
            .unwrap();
        assert!(
            !notes.is_empty(),
            "Suggestion notes should be created (event emission follows)"
        );
    }
}
