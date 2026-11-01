//! REST snapshot of everything the orchestrator is running right now.
//!
//! `GET /api/activity/snapshot` exists for one reason: a live feed alone cannot
//! paint a first screen. A client that connects to `/ws/activity` and shows only
//! what arrives next displays an empty page until something happens — the
//! "empty screen for five seconds" problem. The snapshot answers in one request
//! what the feed would otherwise take an unbounded time to describe: the plan
//! runs in flight, the protocol runs in flight, and the conversations that are
//! actually doing something.
//!
//! The snapshot is also the **reconciliation** point. An indicator driven by
//! events alone has two failure modes a user actually hits: a page mounted after
//! a run started shows nothing, and a dropped "finished" event leaves a card
//! that never clears. Re-reading the truth fixes both, which is why the response
//! carries `last_event_seq`: the client passes it to `/ws/activity?last_seq=…`
//! and receives the events it missed between the snapshot and the subscription.
//!
//! Scoping: `project_id` wins over `workspace_slug`; with neither, every project
//! is included. Plan runs and protocol runs are filtered on their own project;
//! conversations are filtered on their project slug (resolved from `project_id`).
use super::handlers::{AppError, OrchestratorState};
use axum::{
    extract::{Query, State},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How many runs of each family the snapshot returns by default.
const DEFAULT_LIMIT: usize = 50;
/// Hard ceiling, so a `?limit=` typo cannot turn the page into a full scan.
const MAX_LIMIT: usize = 200;

/// Query parameters of `GET /api/activity/snapshot`.
#[derive(Debug, Deserialize)]
pub struct ActivitySnapshotQuery {
    /// Restrict to one project. Takes precedence over `workspace_slug`.
    pub project_id: Option<Uuid>,
    /// Restrict to the projects of one workspace.
    pub workspace_slug: Option<String>,
    /// Maximum number of entries per family (default 50, capped at 200).
    pub limit: Option<usize>,
}

/// Counts per family, so a client can render the tabs before reading the lists.
#[derive(Debug, Default, Serialize)]
pub struct ActivityCounts {
    pub plan_runs: usize,
    pub protocol_runs: usize,
    pub chat_sessions: usize,
    pub total: usize,
}

/// One task an agent is working on inside a plan run.
#[derive(Debug, Serialize)]
pub struct ActiveTaskActivity {
    pub task_id: Uuid,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    pub status: String,
    pub elapsed_secs: i64,
    pub cost_usd: f64,
}

/// A plan execution in flight.
#[derive(Debug, Serialize)]
pub struct PlanRunActivity {
    pub run_id: Uuid,
    pub plan_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_title: Option<String>,
    pub status: String,
    pub current_wave: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub active_tasks: Vec<ActiveTaskActivity>,
    pub git_branch: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    pub started_at: String,
    pub elapsed_secs: i64,
    pub cost_usd: f64,
    pub tokens_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_run_id: Option<Uuid>,
}

/// An FSM protocol execution in flight.
#[derive(Debug, Serialize)]
pub struct ProtocolRunActivity {
    pub run_id: Uuid,
    pub protocol_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol_name: Option<String>,
    pub current_state: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_state_name: Option<String>,
    pub status: String,
    pub states_visited: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<Uuid>,
    pub triggered_by: String,
    pub started_at: String,
    pub elapsed_secs: i64,
}

/// What a conversation is doing right now (mirror of `SessionActivity`).
#[derive(Debug, Serialize)]
pub struct ChatActivity {
    pub live: bool,
    pub streaming: bool,
    pub pending_permissions: usize,
    pub monitors: usize,
    pub bash_tasks: usize,
}

/// A conversation that is doing something.
#[derive(Debug, Serialize)]
pub struct ChatSessionActivity {
    pub session_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_slug: Option<String>,
    pub message_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    pub updated_at: String,
    pub idle_secs: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
    pub activity: ChatActivity,
}

/// Body of `GET /api/activity/snapshot`.
#[derive(Debug, Serialize)]
pub struct ActivitySnapshot {
    /// When the snapshot was taken, so a client can tell a fresh "nothing is
    /// running" from a response it failed to refresh.
    pub generated_at: String,
    /// Sequence number the client must pass to `/ws/activity?last_seq=…` to
    /// receive only what happened after this snapshot.
    pub last_event_seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_slug: Option<String>,
    pub counts: ActivityCounts,
    pub plan_runs: Vec<PlanRunActivity>,
    pub protocol_runs: Vec<ProtocolRunActivity>,
    pub chat_sessions: Vec<ChatSessionActivity>,
}

/// `snake_case` wire name of a serde-serializable enum, without a panic path.
fn wire_name<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        Ok(other) => other.to_string(),
        Err(_) => "unknown".to_string(),
    }
}

/// Seconds between `started_at` and now, never negative.
fn elapsed_secs(started_at: DateTime<Utc>) -> i64 {
    (Utc::now() - started_at).num_seconds().max(0)
}

/// GET /api/activity/snapshot — what is running now, in one request.
pub async fn get_activity_snapshot(
    State(state): State<OrchestratorState>,
    Query(params): Query<ActivitySnapshotQuery>,
) -> Result<Json<ActivitySnapshot>, AppError> {
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let graph = state.orchestrator.neo4j_arc();

    // Resolve the scope once: the project slug is what conversations are keyed
    // on, and plan runs only carry a project id.
    let project_slug = match params.project_id {
        Some(id) => graph
            .get_project(id)
            .await
            .ok()
            .flatten()
            .map(|p| p.slug)
            .ok_or_else(|| AppError::NotFound(format!("project {id} not found")))?,
        None => String::new(),
    };

    // A workspace is a set of projects, not a field on a run: resolve it once so
    // plan runs and conversations can be filtered on membership instead of being
    // silently included (a workspace page must not show another workspace's work).
    let mut workspace_project_ids: Option<std::collections::HashSet<Uuid>> = None;
    let mut workspace_project_slugs: Option<std::collections::HashSet<String>> = None;
    if params.project_id.is_none() {
        if let Some(slug) = params.workspace_slug.as_deref() {
            match graph.get_workspace_by_slug(slug).await {
                Ok(Some(workspace)) => {
                    if let Ok(projects) = graph.list_workspace_projects(workspace.id).await {
                        workspace_project_ids =
                            Some(projects.iter().map(|project| project.id).collect());
                        workspace_project_slugs = Some(
                            projects
                                .iter()
                                .map(|project| project.slug.clone())
                                .collect(),
                        );
                    }
                }
                Ok(None) => {
                    return Err(AppError::NotFound(format!("workspace `{slug}` not found")));
                }
                Err(error) => return Err(AppError::Internal(error)),
            }
        }
    }

    // ── Plan runs ────────────────────────────────────────────────────────
    let active_runs = graph
        .list_active_plan_runs()
        .await
        .map_err(AppError::Internal)?;
    let mut plan_runs = Vec::new();
    for run in active_runs.into_iter().take(MAX_LIMIT) {
        if let Some(wanted) = params.project_id {
            if run.project_id != Some(wanted) {
                continue;
            }
        }
        if let Some(ids) = &workspace_project_ids {
            if !run.project_id.map(|id| ids.contains(&id)).unwrap_or(false) {
                continue;
            }
        }
        if plan_runs.len() >= limit {
            break;
        }
        let plan_title = match graph.get_plan(run.plan_id).await {
            Ok(Some(plan)) => Some(plan.title),
            _ => None,
        };
        let active_tasks = run
            .active_agents
            .iter()
            .map(|agent| ActiveTaskActivity {
                task_id: agent.task_id,
                title: agent.task_title.clone(),
                session_id: agent.session_id,
                status: wire_name(&agent.status),
                elapsed_secs: elapsed_secs(agent.started_at),
                cost_usd: agent.cost_usd,
            })
            .collect();
        plan_runs.push(PlanRunActivity {
            run_id: run.run_id,
            plan_id: run.plan_id,
            plan_title,
            status: wire_name(&run.status),
            current_wave: run.current_wave,
            total_tasks: run.total_tasks,
            completed_tasks: run.completed_tasks.len(),
            failed_tasks: run.failed_tasks.len(),
            active_tasks,
            git_branch: run.git_branch,
            cwd: run.cwd,
            started_at: run.started_at.to_rfc3339(),
            elapsed_secs: elapsed_secs(run.started_at),
            cost_usd: run.cost_usd,
            tokens_used: run.tokens_used,
            lifecycle_run_id: run.lifecycle_run_id,
        });
    }

    // ── Protocol runs ────────────────────────────────────────────────────
    let (protocol_runs_raw, _total) = graph
        .list_all_protocol_runs(
            Some(crate::protocol::RunStatus::Running),
            params.project_id,
            params.workspace_slug.as_deref(),
            limit,
            0,
        )
        .await
        .map_err(AppError::Internal)?;
    let mut protocol_cache: std::collections::HashMap<Uuid, Option<crate::protocol::Protocol>> =
        std::collections::HashMap::new();
    let mut protocol_runs = Vec::with_capacity(protocol_runs_raw.len());
    for run in protocol_runs_raw {
        let protocol = match protocol_cache.get(&run.protocol_id) {
            Some(cached) => cached.clone(),
            None => {
                let fetched = graph.get_protocol(run.protocol_id).await.ok().flatten();
                protocol_cache.insert(run.protocol_id, fetched.clone());
                fetched
            }
        };
        // The current state's name is already denormalized on the visit history
        // (`StateVisit.state_name`), so no extra query per run: the protocol
        // lookup below is only for the protocol's own name.
        let current_state_name = run
            .states_visited
            .iter()
            .rev()
            .find(|visit| visit.state_id == run.current_state)
            .map(|visit| visit.state_name.clone());
        protocol_runs.push(ProtocolRunActivity {
            run_id: run.id,
            protocol_id: run.protocol_id,
            protocol_name: protocol.map(|p| p.name),
            current_state: run.current_state,
            current_state_name,
            status: wire_name(&run.status),
            states_visited: run.states_visited.len(),
            project_id: run.project_id,
            plan_id: run.plan_id,
            task_id: run.task_id,
            parent_run_id: run.parent_run_id,
            triggered_by: run.triggered_by,
            started_at: run.started_at.to_rfc3339(),
            elapsed_secs: elapsed_secs(run.started_at),
        });
    }

    // ── Conversations doing something ────────────────────────────────────
    let mut chat_sessions = Vec::new();
    if let Some(chat_manager) = state.chat_manager.as_ref() {
        let activities = chat_manager.session_activity_map().await;
        // Deterministic order: the busiest (streaming, then background work)
        // first, so the grid does not reshuffle on every refresh.
        let mut ids: Vec<(Uuid, crate::chat::types::SessionActivity)> =
            activities.into_iter().collect();
        ids.sort_by_key(|(id, activity)| {
            (
                !(activity.streaming || activity.pending_permissions > 0),
                std::cmp::Reverse(activity.background_tasks()),
                *id,
            )
        });
        for (session_id, activity) in ids.into_iter().take(limit) {
            let node = match graph.get_chat_session(session_id).await {
                Ok(Some(node)) => node,
                // A session the graph does not know yet (just opened): still
                // worth showing — it is live, that is the whole point.
                _ => {
                    if params.project_id.is_some() || params.workspace_slug.is_some() {
                        continue;
                    }
                    chat_sessions.push(ChatSessionActivity {
                        session_id,
                        title: None,
                        model: String::new(),
                        provider_id: None,
                        project_slug: None,
                        message_count: 0,
                        cost_usd: None,
                        updated_at: Utc::now().to_rfc3339(),
                        idle_secs: 0,
                        preview: None,
                        activity: ChatActivity {
                            live: activity.live,
                            streaming: activity.streaming,
                            pending_permissions: activity.pending_permissions,
                            monitors: activity.monitors,
                            bash_tasks: activity.bash_tasks,
                        },
                    });
                    continue;
                }
            };
            if params.project_id.is_some()
                && node.project_slug.as_deref() != Some(project_slug.as_str())
            {
                continue;
            }
            if params.workspace_slug.is_some()
                && node.workspace_slug.as_deref() != params.workspace_slug.as_deref()
            {
                // A session can name its workspace directly, or only its project
                // (most do). Fall back to the workspace's project list before
                // dropping it: a workspace page exists to show exactly these.
                let in_workspace = match (&workspace_project_slugs, node.project_slug.as_deref()) {
                    (Some(slugs), Some(slug)) => slugs.contains(slug),
                    _ => false,
                };
                if !in_workspace {
                    continue;
                }
            }
            chat_sessions.push(ChatSessionActivity {
                session_id,
                title: node.title,
                model: node.model,
                provider_id: node.provider_id,
                project_slug: node.project_slug,
                message_count: node.message_count,
                cost_usd: node.total_cost_usd,
                updated_at: node.updated_at.to_rfc3339(),
                idle_secs: elapsed_secs(node.updated_at),
                preview: node.preview,
                activity: ChatActivity {
                    live: activity.live,
                    streaming: activity.streaming,
                    pending_permissions: activity.pending_permissions,
                    monitors: activity.monitors,
                    bash_tasks: activity.bash_tasks,
                },
            });
        }
    }

    let counts = ActivityCounts {
        plan_runs: plan_runs.len(),
        protocol_runs: protocol_runs.len(),
        chat_sessions: chat_sessions.len(),
        total: plan_runs.len() + protocol_runs.len() + chat_sessions.len(),
    };

    Ok(Json(ActivitySnapshot {
        generated_at: Utc::now().to_rfc3339(),
        last_event_seq: state.event_bus.activity().latest_seq(),
        project_id: params.project_id,
        workspace_slug: params.workspace_slug,
        counts,
        plan_runs,
        protocol_runs,
        chat_sessions,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_dtos_serialize_with_stable_names() {
        let snapshot = ActivitySnapshot {
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            last_event_seq: 42,
            project_id: Some(Uuid::nil()),
            workspace_slug: None,
            counts: ActivityCounts {
                plan_runs: 1,
                protocol_runs: 0,
                chat_sessions: 0,
                total: 1,
            },
            plan_runs: vec![PlanRunActivity {
                run_id: Uuid::nil(),
                plan_id: Uuid::nil(),
                plan_title: Some("Plan".to_string()),
                status: "running".to_string(),
                current_wave: 2,
                total_tasks: 5,
                completed_tasks: 3,
                failed_tasks: 0,
                active_tasks: vec![ActiveTaskActivity {
                    task_id: Uuid::nil(),
                    title: "Task".to_string(),
                    session_id: None,
                    status: "running".to_string(),
                    elapsed_secs: 12,
                    cost_usd: 0.5,
                }],
                git_branch: "feat/x".to_string(),
                cwd: None,
                started_at: "2026-01-01T00:00:00Z".to_string(),
                elapsed_secs: 120,
                cost_usd: 1.5,
                tokens_used: 100,
                lifecycle_run_id: None,
            }],
            protocol_runs: Vec::new(),
            chat_sessions: Vec::new(),
        };

        let json = serde_json::to_value(&snapshot).expect("serializes");
        assert_eq!(json["last_event_seq"], 42);
        assert_eq!(json["counts"]["total"], 1);
        assert_eq!(json["plan_runs"][0]["run_id"], Uuid::nil().to_string());
        assert_eq!(json["plan_runs"][0]["completed_tasks"], 3);
        // Absent optional fields must not appear as null: the frontend types are
        // `field?: T`, and a `null` would fail an exactOptionalPropertyTypes check.
        assert!(json["workspace_slug"].is_null());
        assert!(!json["plan_runs"][0]
            .as_object()
            .unwrap()
            .contains_key("cwd"));
    }

    #[test]
    fn limit_is_clamped_to_the_hard_ceiling() {
        // Mirrors the handler's arithmetic without needing a live graph.
        let clamp = |raw: Option<usize>| raw.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
        assert_eq!(clamp(None), DEFAULT_LIMIT);
        assert_eq!(clamp(Some(0)), 1);
        assert_eq!(clamp(Some(10_000)), MAX_LIMIT);
    }

    use crate::neo4j::traits::*;
    use std::sync::Arc;

    /// What the snapshot handler reads: a graph, a chat manager over it, and the
    /// no-auth server state both are served by.
    async fn rig() -> (
        OrchestratorState,
        Arc<crate::neo4j::mock::MockGraphStore>,
        Arc<crate::chat::manager::ChatManager>,
    ) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let app_state = crate::test_helpers::mock_app_state_with_graph(graph.clone());
        let manager = Arc::new(crate::chat::manager::ChatManager::new_without_memory(
            graph.clone(),
            app_state.meili.clone(),
            crate::chat::manager::test_support::chat_config(),
        ));
        let state =
            crate::test_helpers::server_state_with_chat(manager.clone(), graph.clone()).await;
        (state, graph, manager)
    }

    async fn snapshot(
        state: &OrchestratorState,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        limit: Option<usize>,
    ) -> Result<ActivitySnapshot, AppError> {
        let query = ActivitySnapshotQuery {
            project_id,
            workspace_slug: workspace_slug.map(str::to_string),
            limit,
        };
        get_activity_snapshot(State(state.clone()), Query(query))
            .await
            .map(|Json(snapshot)| snapshot)
    }

    /// A running plan run, owned by `project_id`, with one active agent.
    fn running_run(
        plan_id: Uuid,
        project_id: Option<Uuid>,
        title: &str,
    ) -> crate::runner::RunnerState {
        use crate::runner::{ActiveAgent, RunnerState, TaskRunStatus, TriggerSource};
        let mut run = RunnerState::new(Uuid::new_v4(), plan_id, 3, TriggerSource::Manual);
        run.project_id = project_id;
        run.git_branch = "feat/x".to_string();
        run.cwd = Some("/work/x".to_string());
        run.completed_tasks = vec![Uuid::new_v4()];
        run.failed_tasks = vec![Uuid::new_v4()];
        run.started_at = Utc::now() - chrono::Duration::seconds(90);
        run.active_agents = vec![ActiveAgent {
            task_id: Uuid::new_v4(),
            task_title: title.to_string(),
            session_id: Some(Uuid::new_v4()),
            started_at: Utc::now() - chrono::Duration::seconds(30),
            cost_usd: 0.5,
            status: TaskRunStatus::Running,
        }];
        run
    }

    fn visit(state_id: Uuid, name: &str) -> crate::protocol::StateVisit {
        crate::protocol::StateVisit {
            state_id,
            state_name: name.to_string(),
            entered_at: Utc::now(),
            exited_at: None,
            duration_ms: None,
            trigger: None,
            progress_snapshot: None,
        }
    }

    #[tokio::test]
    async fn an_unknown_project_is_a_not_found_not_an_empty_snapshot() {
        let (state, _graph, _manager) = rig().await;
        let refused = snapshot(&state, Some(Uuid::new_v4()), None, None)
            .await
            .expect_err("an unknown project is refused");
        assert!(matches!(refused, AppError::NotFound(_)), "{refused:?}");
    }

    #[tokio::test]
    async fn an_unknown_workspace_is_a_not_found() {
        let (state, _graph, _manager) = rig().await;
        let refused = snapshot(&state, None, Some("no-such-workspace"), None)
            .await
            .expect_err("an unknown workspace is refused");
        match refused {
            AppError::NotFound(message) => assert!(message.contains("no-such-workspace")),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn plan_runs_are_scoped_by_project_and_by_workspace_membership() {
        let (state, graph, _manager) = rig().await;
        let alpha = crate::test_helpers::test_project_named("alpha");
        let beta = crate::test_helpers::test_project_named("beta");
        let gamma = crate::test_helpers::test_project_named("gamma");
        for project in [&alpha, &beta, &gamma] {
            graph.create_project(project).await.unwrap();
        }
        let workspace = crate::test_helpers::test_workspace();
        graph.create_workspace(&workspace).await.unwrap();
        graph
            .add_project_to_workspace(workspace.id, alpha.id)
            .await
            .unwrap();
        graph
            .add_project_to_workspace(workspace.id, beta.id)
            .await
            .unwrap();

        let plan = crate::test_helpers::test_plan_for_project(alpha.id);
        graph.create_plan(&plan).await.unwrap();
        let in_alpha = running_run(plan.id, Some(alpha.id), "Build alpha");
        let in_beta = running_run(Uuid::new_v4(), Some(beta.id), "Build beta");
        let in_gamma = running_run(plan.id, Some(gamma.id), "Build gamma");
        let unowned = running_run(plan.id, None, "Orphan");
        for run in [&in_alpha, &in_beta, &in_gamma, &unowned] {
            graph.create_plan_run(run).await.unwrap();
        }

        // Unscoped: every running run, the plan title resolved where the plan exists.
        let all = snapshot(&state, None, None, None).await.ok().unwrap();
        assert_eq!(all.counts.plan_runs, 4);
        let alpha_entry = all
            .plan_runs
            .iter()
            .find(|r| r.run_id == in_alpha.run_id)
            .expect("alpha run listed");
        assert_eq!(alpha_entry.plan_title.as_deref(), Some(plan.title.as_str()));
        assert_eq!(alpha_entry.status, "running");
        assert_eq!(alpha_entry.completed_tasks, 1);
        assert_eq!(alpha_entry.failed_tasks, 1);
        assert_eq!(alpha_entry.total_tasks, 3);
        assert_eq!(alpha_entry.git_branch, "feat/x");
        assert_eq!(alpha_entry.cwd.as_deref(), Some("/work/x"));
        assert!(
            alpha_entry.elapsed_secs >= 90,
            "{}",
            alpha_entry.elapsed_secs
        );
        assert_eq!(alpha_entry.active_tasks.len(), 1);
        assert_eq!(alpha_entry.active_tasks[0].title, "Build alpha");
        assert_eq!(alpha_entry.active_tasks[0].status, "running");
        assert!(alpha_entry.active_tasks[0].elapsed_secs >= 30);
        let beta_entry = all
            .plan_runs
            .iter()
            .find(|r| r.run_id == in_beta.run_id)
            .expect("beta run listed");
        assert_eq!(
            beta_entry.plan_title, None,
            "the plan of beta does not exist"
        );

        // A project: only its own runs, the unowned one never leaks in.
        let only_alpha = snapshot(&state, Some(alpha.id), None, None)
            .await
            .ok()
            .unwrap();
        let ids: Vec<Uuid> = only_alpha.plan_runs.iter().map(|r| r.run_id).collect();
        assert_eq!(ids, vec![in_alpha.run_id]);
        assert_eq!(only_alpha.project_id, Some(alpha.id));

        // A workspace: the runs of its member projects, not gamma's and not the orphan.
        let in_ws = snapshot(&state, None, Some(&workspace.slug), None)
            .await
            .ok()
            .unwrap();
        let mut ws_ids: Vec<Uuid> = in_ws.plan_runs.iter().map(|r| r.run_id).collect();
        ws_ids.sort();
        let mut expected = vec![in_alpha.run_id, in_beta.run_id];
        expected.sort();
        assert_eq!(ws_ids, expected);
        assert_eq!(
            in_ws.workspace_slug.as_deref(),
            Some(workspace.slug.as_str())
        );

        // The per-family limit caps the list and is reflected in the counts.
        let capped = snapshot(&state, None, None, Some(1)).await.ok().unwrap();
        assert_eq!(capped.plan_runs.len(), 1);
        assert_eq!(capped.counts.plan_runs, 1);
        // A zero limit is clamped to one, never an empty page that hides everything.
        let clamped = snapshot(&state, None, None, Some(0)).await.ok().unwrap();
        assert_eq!(clamped.plan_runs.len(), 1);
    }

    #[tokio::test]
    async fn protocol_runs_name_their_protocol_and_current_state() {
        let (state, graph, _manager) = rig().await;
        let project = crate::test_helpers::test_project_named("ops");
        graph.create_project(&project).await.unwrap();
        let start = Uuid::new_v4();
        let protocol = crate::protocol::Protocol::new(Some(project.id), "deploy", start);
        graph.upsert_protocol(&protocol).await.unwrap();

        // The current state is named by its visit: the name is denormalized there.
        let mut named = crate::protocol::ProtocolRun::new(protocol.id, start, "start");
        named.states_visited = vec![visit(start, "start")];
        // The current state has no visit (never recorded): no name, not a failure.
        // A child run of the same protocol: the store refuses two *root* runs
        // running on one protocol, and a second run of it exercises the cache.
        let mut unnamed = crate::protocol::ProtocolRun::new(protocol.id, Uuid::new_v4(), "ghost");
        unnamed.states_visited = vec![visit(start, "start")];
        unnamed.parent_run_id = Some(named.id);
        // The protocol itself is gone: the run still shows, without a name.
        let orphan = crate::protocol::ProtocolRun::new(Uuid::new_v4(), start, "start");
        for run in [&named, &unnamed, &orphan] {
            graph.create_protocol_run(run).await.unwrap();
        }

        let all = snapshot(&state, None, None, None).await.ok().unwrap();
        assert_eq!(all.counts.protocol_runs, 3);
        let by_id = |id: Uuid| {
            all.protocol_runs
                .iter()
                .find(|r| r.run_id == id)
                .expect("run listed")
        };
        let named_entry = by_id(named.id);
        assert_eq!(named_entry.protocol_name.as_deref(), Some("deploy"));
        assert_eq!(named_entry.current_state_name.as_deref(), Some("start"));
        assert_eq!(named_entry.states_visited, 1);
        assert_eq!(named_entry.status, "running");
        let unnamed_entry = by_id(unnamed.id);
        assert_eq!(unnamed_entry.protocol_name.as_deref(), Some("deploy"));
        assert_eq!(unnamed_entry.current_state_name, None);
        assert_eq!(by_id(orphan.id).protocol_name, None);

        // A project scope keeps the runs whose protocol belongs to it.
        let scoped = snapshot(&state, Some(project.id), None, None)
            .await
            .ok()
            .unwrap();
        assert_eq!(scoped.counts.protocol_runs, 2);
        assert!(scoped.protocol_runs.iter().all(|r| r.run_id != orphan.id));
    }

    #[tokio::test]
    async fn conversations_are_scoped_like_plan_runs_and_limited() {
        let (state, graph, manager) = rig().await;
        let alpha = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&alpha).await.unwrap();
        let workspace = crate::test_helpers::test_workspace();
        graph.create_workspace(&workspace).await.unwrap();
        graph
            .add_project_to_workspace(workspace.id, alpha.id)
            .await
            .unwrap();

        // In the project.
        let mut in_project = crate::test_helpers::test_chat_session(Some(&alpha.slug));
        in_project.title = Some("Chat in alpha".to_string());
        // Elsewhere.
        let elsewhere = crate::test_helpers::test_chat_session(Some("elsewhere"));
        // Names its workspace directly, its project is elsewhere: in.
        let mut named_workspace = crate::test_helpers::test_chat_session(Some("elsewhere"));
        named_workspace.workspace_slug = Some(workspace.slug.clone());
        // Names another workspace, but its project is in this one: in by project.
        let mut other_workspace = crate::test_helpers::test_chat_session(Some(&alpha.slug));
        other_workspace.workspace_slug = Some("another".to_string());
        // Names another workspace and its project is not ours: out.
        let mut foreign = crate::test_helpers::test_chat_session(Some("elsewhere"));
        foreign.workspace_slug = Some("another".to_string());
        // Names another workspace and no project at all: out, nothing to fall back on.
        let mut projectless = crate::test_helpers::test_chat_session(None);
        projectless.workspace_slug = Some("another".to_string());
        let nodes = [
            &in_project,
            &elsewhere,
            &named_workspace,
            &other_workspace,
            &foreign,
            &projectless,
        ];
        for node in nodes {
            graph.create_chat_session(node).await.unwrap();
        }
        // Live but unknown to the graph (just opened): shown when unscoped only.
        let unknown_id = Uuid::new_v4();

        let mut handles = Vec::new();
        for node in nodes {
            handles.push(
                crate::chat::manager::test_support::insert_mock_cli_session(
                    &manager,
                    &node.id.to_string(),
                )
                .await,
            );
        }
        handles.push(
            crate::chat::manager::test_support::insert_mock_cli_session(
                &manager,
                &unknown_id.to_string(),
            )
            .await,
        );

        let all = snapshot(&state, None, None, None).await.ok().unwrap();
        assert_eq!(all.counts.chat_sessions, 7);
        let unknown_entry = all
            .chat_sessions
            .iter()
            .find(|s| s.session_id == unknown_id)
            .expect("the unknown live session is still shown");
        assert_eq!(unknown_entry.title, None);
        assert_eq!(unknown_entry.model, "");
        assert!(unknown_entry.activity.live);
        let alpha_entry = all
            .chat_sessions
            .iter()
            .find(|s| s.session_id == in_project.id)
            .expect("alpha chat listed");
        assert_eq!(alpha_entry.title.as_deref(), Some("Chat in alpha"));
        assert_eq!(
            alpha_entry.project_slug.as_deref(),
            Some(alpha.slug.as_str())
        );

        // A project: its conversations only; the unknown live one is not theirs.
        let by_project = snapshot(&state, Some(alpha.id), None, None)
            .await
            .ok()
            .unwrap();
        let project_ids: Vec<Uuid> = by_project
            .chat_sessions
            .iter()
            .map(|s| s.session_id)
            .collect();
        assert!(project_ids.contains(&in_project.id));
        assert!(project_ids.contains(&other_workspace.id));
        assert!(!project_ids.contains(&unknown_id));
        assert!(!project_ids.contains(&elsewhere.id));

        // A workspace: by its slug, or by the project it belongs to; the foreign one stays out.
        let by_workspace = snapshot(&state, None, Some(&workspace.slug), None)
            .await
            .ok()
            .unwrap();
        let ws_ids: Vec<Uuid> = by_workspace
            .chat_sessions
            .iter()
            .map(|s| s.session_id)
            .collect();
        assert!(ws_ids.contains(&in_project.id));
        assert!(ws_ids.contains(&named_workspace.id));
        assert!(ws_ids.contains(&other_workspace.id));
        assert!(!ws_ids.contains(&elsewhere.id));
        assert!(!ws_ids.contains(&foreign.id));
        assert!(!ws_ids.contains(&projectless.id));
        assert!(!ws_ids.contains(&unknown_id));

        // The limit applies to conversations too.
        let capped = snapshot(&state, None, None, Some(2)).await.ok().unwrap();
        assert_eq!(capped.chat_sessions.len(), 2);
        drop(handles);
    }

    #[test]
    fn wire_name_is_the_snake_case_name_and_never_a_panic() {
        assert_eq!(wire_name(&crate::runner::TaskRunStatus::Running), "running");
        assert_eq!(
            wire_name(&7u32),
            "7",
            "a non-string value keeps its JSON form"
        );
        assert_eq!(wire_name(&Unserializable), "unknown");
    }

    #[test]
    fn elapsed_is_never_negative() {
        let future = Utc::now() + chrono::Duration::seconds(120);
        assert_eq!(elapsed_secs(future), 0);
        let past = Utc::now() - chrono::Duration::seconds(5);
        assert!(elapsed_secs(past) >= 5);
    }

    /// A value whose serialization fails: `wire_name` must degrade, not panic.
    struct Unserializable;

    impl Serialize for Unserializable {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("refused on purpose"))
        }
    }
}
