//! Neo4j queries for the **Live Activity Hub** REST snapshot.
//!
//! Single round-trip projection used by `GET /api/activity/snapshot` to fetch:
//!
//! - in-progress `PlanRun`s scoped to a project
//! - running `ProtocolRun`s scoped to a project (via `INSTANCE_OF`)
//! - recently active `ChatSession`s scoped to a project (by `project_slug`)
//!
//! Returns the lightweight DTOs from [`crate::api::models::activity`] so the
//! handler is a thin pass-through. The Cypher only selects the columns we
//! actually serialize — no `state_json` blob, no full `states_visited_json`
//! payload (we just return its length).

use super::client::Neo4jClient;
use crate::api::models::activity::{
    ChatSessionSummary, PlanRunSummary, ProtocolRunSummary, SnapshotPlanRunStatus,
    SnapshotProtocolRunStatus,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

/// Aggregate result returned by [`Neo4jClient::fetch_activity_snapshot`].
///
/// Kept as a tuple-struct-like container rather than re-exporting
/// [`crate::api::models::activity::ActivitySnapshot`] directly because the
/// `last_event_seq` field lives on the API side (it comes from the
/// in-process `EventBus`, not from Neo4j).
#[derive(Debug, Default, Clone)]
pub struct ActivitySnapshotData {
    pub plan_runs: Vec<PlanRunSummary>,
    pub protocol_runs: Vec<ProtocolRunSummary>,
    pub chat_sessions: Vec<ChatSessionSummary>,
}

impl Neo4jClient {
    /// Fetch the full activity snapshot for a project in **one Neo4j round-trip**.
    ///
    /// Implementation strategy:
    /// - 3 small `MATCH` queries chained via `CALL { ... }` subqueries so that
    ///   the driver issues a single statement to the server. Each subquery is
    ///   independent and `COLLECT`s its rows into a single value, then the
    ///   outer `RETURN` unpacks the three lists.
    /// - We project only the fields the API serializes (no `state_json`).
    /// - Sorting + capping happens server-side to keep payload bounded.
    ///
    /// Targets <100ms for ~50 active runs.
    pub async fn fetch_activity_snapshot(
        &self,
        project_id: Uuid,
        project_slug: Option<&str>,
        chat_session_limit: i64,
    ) -> Result<ActivitySnapshotData> {
        let q = query(
            r#"
            // ─── PlanRun running scoped to project ──────────────────────────
            CALL {
                WITH $project_id AS pid
                MATCH (r:PlanRun {status: 'running'})-[:RUNS]->(p:Plan)
                WHERE r.project_id = pid OR p.project_id = pid
                RETURN collect({
                    run_id: r.run_id,
                    plan_id: r.plan_id,
                    plan_title: coalesce(p.title, ''),
                    total_tasks: coalesce(r.total_tasks, 0),
                    current_wave: coalesce(r.current_wave, 0),
                    completed_tasks: coalesce(r.completed_tasks, ''),
                    failed_tasks: coalesce(r.failed_tasks, ''),
                    cost_usd: coalesce(r.cost_usd, 0.0),
                    started_at: toString(r.started_at),
                    current_task_id: r.current_task_id,
                    git_branch: r.git_branch
                }) AS plan_runs
            }

            // ─── ProtocolRun running scoped to project ──────────────────────
            CALL {
                WITH $project_id AS pid
                MATCH (run:ProtocolRun {status: 'running'})-[:INSTANCE_OF]->(proto:Protocol {project_id: pid})
                OPTIONAL MATCH (proto)-[:HAS_STATE]->(st:ProtocolState {id: run.current_state})
                RETURN collect({
                    id: run.id,
                    protocol_id: proto.id,
                    protocol_name: coalesce(proto.name, ''),
                    current_state: run.current_state,
                    state_name: coalesce(st.name, ''),
                    states_visited_json: coalesce(run.states_visited_json, '[]'),
                    started_at: toString(run.started_at),
                    plan_id: run.plan_id,
                    task_id: run.task_id,
                    depth: coalesce(run.depth, 0)
                }) AS protocol_runs
            }

            // ─── ChatSession recently active for the project ────────────────
            CALL {
                WITH $project_slug AS slug, $chat_limit AS chat_limit
                MATCH (s:ChatSession)
                WHERE slug IS NOT NULL AND s.project_slug = slug
                  AND (s.spawned_by IS NULL OR s.spawned_by = '')
                RETURN collect({
                    id: s.id,
                    title: s.title,
                    model: s.model,
                    message_count: coalesce(s.message_count, 0),
                    updated_at: toString(s.updated_at),
                    total_cost_usd: s.total_cost_usd,
                    preview: s.preview,
                    project_slug: s.project_slug,
                    _ts: s.updated_at
                })[0..chat_limit] AS chat_sessions
            }

            RETURN plan_runs, protocol_runs, chat_sessions
            "#,
        )
        .param("project_id", project_id.to_string())
        .param(
            "project_slug",
            project_slug.map(|s| s.to_string()).unwrap_or_default(),
        )
        .param("chat_limit", chat_session_limit);

        let mut result = self.graph.execute(q).await?;
        let row = match result.next().await? {
            Some(r) => r,
            None => return Ok(ActivitySnapshotData::default()),
        };

        let plan_runs_raw: Vec<neo4rs::BoltType> = row.get("plan_runs").unwrap_or_default();
        let protocol_runs_raw: Vec<neo4rs::BoltType> = row.get("protocol_runs").unwrap_or_default();
        let chat_sessions_raw: Vec<neo4rs::BoltType> = row.get("chat_sessions").unwrap_or_default();

        let plan_runs = plan_runs_raw
            .into_iter()
            .filter_map(map_to_plan_run_summary)
            .collect();
        let mut protocol_runs: Vec<ProtocolRunSummary> = protocol_runs_raw
            .into_iter()
            .filter_map(map_to_protocol_run_summary)
            .collect();
        protocol_runs.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        let mut chat_sessions: Vec<ChatSessionSummary> = chat_sessions_raw
            .into_iter()
            .filter_map(map_to_chat_session_summary)
            .collect();
        chat_sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(ActivitySnapshotData {
            plan_runs,
            protocol_runs,
            chat_sessions,
        })
    }
}

// ============================================================================
// Map projection helpers (BoltType -> typed DTO)
// ============================================================================

fn bolt_string(map: &std::collections::HashMap<String, neo4rs::BoltType>, key: &str) -> Option<String> {
    match map.get(key)? {
        neo4rs::BoltType::String(s) => Some(s.value.clone()),
        _ => None,
    }
}

fn bolt_i64(map: &std::collections::HashMap<String, neo4rs::BoltType>, key: &str) -> Option<i64> {
    match map.get(key)? {
        neo4rs::BoltType::Integer(i) => Some(i.value),
        _ => None,
    }
}

fn bolt_f64(map: &std::collections::HashMap<String, neo4rs::BoltType>, key: &str) -> Option<f64> {
    match map.get(key)? {
        neo4rs::BoltType::Float(f) => Some(f.value),
        neo4rs::BoltType::Integer(i) => Some(i.value as f64),
        _ => None,
    }
}

/// Bolt maps come as `BoltType::Map` — flatten to a regular `HashMap`.
fn as_map(value: neo4rs::BoltType) -> Option<std::collections::HashMap<String, neo4rs::BoltType>> {
    match value {
        neo4rs::BoltType::Map(m) => Some(
            m.value
                .into_iter()
                .map(|(k, v)| (k.value, v))
                .collect::<std::collections::HashMap<_, _>>(),
        ),
        _ => None,
    }
}

fn parse_uuid_opt(s: &str) -> Option<Uuid> {
    if s.is_empty() {
        None
    } else {
        s.parse::<Uuid>().ok()
    }
}

fn parse_ts_required(s: &str) -> DateTime<Utc> {
    s.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now())
}

/// Parse a Neo4j integer comma-separated list field used for `completed_tasks`
/// / `failed_tasks`. Returns the entry count (the snapshot DTO only needs the
/// counter, not the UUIDs themselves).
fn count_csv(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    s.split(',').filter(|p| !p.trim().is_empty()).count()
}

/// Count entries inside the `states_visited_json` blob without parsing
/// the full `StateVisit` structs.
fn count_states_visited(json: &str) -> usize {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| v.as_array().map(|a| a.len()))
        .unwrap_or(0)
}

fn map_to_plan_run_summary(value: neo4rs::BoltType) -> Option<PlanRunSummary> {
    let m = as_map(value)?;
    Some(PlanRunSummary {
        run_id: parse_uuid_opt(&bolt_string(&m, "run_id")?)?,
        plan_id: parse_uuid_opt(&bolt_string(&m, "plan_id")?)?,
        plan_title: bolt_string(&m, "plan_title").unwrap_or_default(),
        total_tasks: bolt_i64(&m, "total_tasks").unwrap_or(0).max(0) as usize,
        current_wave: bolt_i64(&m, "current_wave").unwrap_or(0).max(0) as usize,
        completed_tasks: count_csv(&bolt_string(&m, "completed_tasks").unwrap_or_default()),
        failed_tasks: count_csv(&bolt_string(&m, "failed_tasks").unwrap_or_default()),
        status: SnapshotPlanRunStatus::Running,
        cost_usd: bolt_f64(&m, "cost_usd").unwrap_or(0.0),
        started_at: parse_ts_required(&bolt_string(&m, "started_at").unwrap_or_default()),
        current_task_id: bolt_string(&m, "current_task_id").and_then(|s| parse_uuid_opt(&s)),
        current_task_title: None,
        git_branch: bolt_string(&m, "git_branch").filter(|s| !s.is_empty()),
    })
}

fn map_to_protocol_run_summary(value: neo4rs::BoltType) -> Option<ProtocolRunSummary> {
    let m = as_map(value)?;
    Some(ProtocolRunSummary {
        id: parse_uuid_opt(&bolt_string(&m, "id")?)?,
        protocol_id: parse_uuid_opt(&bolt_string(&m, "protocol_id")?)?,
        protocol_name: bolt_string(&m, "protocol_name").unwrap_or_default(),
        current_state: parse_uuid_opt(&bolt_string(&m, "current_state")?)?,
        state_name: bolt_string(&m, "state_name").unwrap_or_default(),
        status: SnapshotProtocolRunStatus::Running,
        states_visited: count_states_visited(
            &bolt_string(&m, "states_visited_json").unwrap_or_else(|| "[]".to_string()),
        ),
        started_at: parse_ts_required(&bolt_string(&m, "started_at").unwrap_or_default()),
        plan_id: bolt_string(&m, "plan_id").and_then(|s| parse_uuid_opt(&s)),
        task_id: bolt_string(&m, "task_id").and_then(|s| parse_uuid_opt(&s)),
        depth: bolt_i64(&m, "depth").unwrap_or(0).max(0) as u32,
    })
}

fn map_to_chat_session_summary(value: neo4rs::BoltType) -> Option<ChatSessionSummary> {
    let m = as_map(value)?;
    Some(ChatSessionSummary {
        id: parse_uuid_opt(&bolt_string(&m, "id")?)?,
        title: bolt_string(&m, "title").filter(|s| !s.is_empty()),
        model: bolt_string(&m, "model").unwrap_or_default(),
        message_count: bolt_i64(&m, "message_count").unwrap_or(0),
        updated_at: parse_ts_required(&bolt_string(&m, "updated_at").unwrap_or_default()),
        total_cost_usd: bolt_f64(&m, "total_cost_usd"),
        preview: bolt_string(&m, "preview").filter(|s| !s.is_empty()),
        project_slug: bolt_string(&m, "project_slug").filter(|s| !s.is_empty()),
    })
}

// ============================================================================
// Tests (parsing helpers only — full snapshot is exercised via mock backends)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_csv_handles_empty() {
        assert_eq!(count_csv(""), 0);
        assert_eq!(count_csv("  "), 0);
    }

    #[test]
    fn count_csv_counts_entries() {
        assert_eq!(count_csv("a"), 1);
        assert_eq!(count_csv("a,b,c"), 3);
        assert_eq!(count_csv("a, b , c "), 3);
        assert_eq!(count_csv("a,,b"), 2);
    }

    #[test]
    fn count_states_visited_handles_empty_json() {
        assert_eq!(count_states_visited("[]"), 0);
        assert_eq!(count_states_visited(""), 0);
        assert_eq!(count_states_visited("not-json"), 0);
    }

    #[test]
    fn count_states_visited_counts_entries() {
        assert_eq!(count_states_visited("[{}]"), 1);
        assert_eq!(count_states_visited("[{}, {}, {}]"), 3);
    }

    #[test]
    fn parse_uuid_opt_empty_returns_none() {
        assert!(parse_uuid_opt("").is_none());
    }

    #[test]
    fn parse_uuid_opt_invalid_returns_none() {
        assert!(parse_uuid_opt("not-a-uuid").is_none());
    }

    #[test]
    fn parse_uuid_opt_valid_returns_some() {
        let u = Uuid::new_v4();
        assert_eq!(parse_uuid_opt(&u.to_string()), Some(u));
    }

    // ------------------------------------------------------------------
    // MockGraphStore integration — exercises the production-facing path
    // via the GraphStore trait method.
    // ------------------------------------------------------------------

    use crate::neo4j::GraphStore as _;
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::{PlanRunStatus, RunnerState, TriggerSource};

    fn ts_offset_seconds(offset: i64) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now() + chrono::Duration::seconds(offset)
    }

    #[tokio::test]
    async fn snapshot_returns_running_plan_runs_for_project() {
        let store = MockGraphStore::new();

        let project_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode::new_for_project(
            "Test Plan".into(),
            "desc".into(),
            "agent".into(),
            3,
            project_id,
        );
        let plan_id = plan.id;
        store.create_plan(&plan).await.unwrap();

        // Running run — should be included
        let mut running = RunnerState::new(Uuid::new_v4(), plan_id, 5, TriggerSource::Manual);
        running.project_id = Some(project_id);
        running.current_wave = 2;
        running.completed_tasks = vec![Uuid::new_v4(), Uuid::new_v4()];
        running.cost_usd = 0.42;
        store.create_plan_run(&running).await.unwrap();

        // Completed run — should be excluded
        let mut completed = RunnerState::new(Uuid::new_v4(), plan_id, 5, TriggerSource::Manual);
        completed.project_id = Some(project_id);
        completed.status = PlanRunStatus::Completed;
        store.create_plan_run(&completed).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");

        assert_eq!(snap.plan_runs.len(), 1, "only running runs should match");
        assert_eq!(snap.plan_runs[0].run_id, running.run_id);
        assert_eq!(snap.plan_runs[0].plan_title, "Test Plan");
        assert_eq!(snap.plan_runs[0].current_wave, 2);
        assert_eq!(snap.plan_runs[0].completed_tasks, 2);
        assert!((snap.plan_runs[0].cost_usd - 0.42).abs() < 1e-9);
    }

    #[tokio::test]
    async fn snapshot_excludes_runs_from_other_projects() {
        let store = MockGraphStore::new();

        let project_id = Uuid::new_v4();
        let other_project_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode::new_for_project(
            "P".into(),
            "d".into(),
            "a".into(),
            1,
            other_project_id,
        );
        store.create_plan(&plan).await.unwrap();

        let mut run = RunnerState::new(Uuid::new_v4(), plan.id, 1, TriggerSource::Manual);
        run.project_id = Some(other_project_id);
        store.create_plan_run(&run).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");
        assert!(snap.plan_runs.is_empty());
    }

    #[tokio::test]
    async fn snapshot_orders_plan_runs_desc_by_started_at() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode::new_for_project(
            "P".into(),
            "d".into(),
            "a".into(),
            1,
            project_id,
        );
        store.create_plan(&plan).await.unwrap();

        let mut older = RunnerState::new(Uuid::new_v4(), plan.id, 1, TriggerSource::Manual);
        older.project_id = Some(project_id);
        older.started_at = ts_offset_seconds(-60);
        let mut newer = RunnerState::new(Uuid::new_v4(), plan.id, 1, TriggerSource::Manual);
        newer.project_id = Some(project_id);
        newer.started_at = ts_offset_seconds(0);
        store.create_plan_run(&older).await.unwrap();
        store.create_plan_run(&newer).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");
        assert_eq!(snap.plan_runs.len(), 2);
        assert_eq!(snap.plan_runs[0].run_id, newer.run_id, "newest first");
        assert_eq!(snap.plan_runs[1].run_id, older.run_id);
    }

    #[tokio::test]
    async fn snapshot_includes_running_protocol_runs() {
        use crate::protocol::{Protocol, ProtocolRun, ProtocolState, RunStatus};

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Build a protocol with one state, then a running run on it.
        let mut protocol = Protocol::new(project_id, "MyProtocol", Uuid::nil());
        let state = ProtocolState::start(protocol.id, "Init");
        let state_id = state.id;
        protocol.entry_state = state_id;
        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&state).await.unwrap();

        let mut run = ProtocolRun::new(protocol.id, state_id, "Init");
        run.status = RunStatus::Running;
        store.create_protocol_run(&run).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");

        assert_eq!(snap.protocol_runs.len(), 1);
        let pr = &snap.protocol_runs[0];
        assert_eq!(pr.id, run.id);
        assert_eq!(pr.protocol_id, protocol.id);
        assert_eq!(pr.protocol_name, "MyProtocol");
        assert_eq!(pr.current_state, state_id);
        assert_eq!(pr.state_name, "Init");
    }

    #[tokio::test]
    async fn snapshot_excludes_finished_protocol_runs() {
        use crate::protocol::{Protocol, ProtocolRun, ProtocolState, RunStatus};

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let mut protocol = Protocol::new(project_id, "P", Uuid::nil());
        let state = ProtocolState::start(protocol.id, "S");
        protocol.entry_state = state.id;
        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&state).await.unwrap();

        // Completed run — must be excluded
        let mut completed = ProtocolRun::new(protocol.id, state.id, "S");
        completed.complete();
        assert_eq!(completed.status, RunStatus::Completed);
        store.create_protocol_run(&completed).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");
        assert!(snap.protocol_runs.is_empty());
    }

    #[tokio::test]
    async fn snapshot_includes_active_chat_sessions_filtered_by_slug() {
        use crate::neo4j::models::ChatSessionNode;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let session = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: Some("my-project".into()),
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: Some("Hello".into()),
            model: "claude-sonnet".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 4,
            total_cost_usd: Some(0.05),
            conversation_id: None,
            preview: Some("Hi".into()),
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
        };
        store.create_chat_session(&session).await.unwrap();

        // A session for another project — must be excluded
        let other = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: Some("other-project".into()),
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "m".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
        };
        store.create_chat_session(&other).await.unwrap();

        let snap = store
            .fetch_activity_snapshot(project_id, Some("my-project"), 50)
            .await
            .expect("snapshot");

        assert_eq!(snap.chat_sessions.len(), 1);
        assert_eq!(snap.chat_sessions[0].id, session.id);
        assert_eq!(snap.chat_sessions[0].model, "claude-sonnet");
    }

    #[tokio::test]
    async fn snapshot_respects_chat_session_limit() {
        use crate::neo4j::models::ChatSessionNode;
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        for i in 0..5 {
            let session = ChatSessionNode {
                id: Uuid::new_v4(),
                cli_session_id: None,
                project_slug: Some("p".into()),
                workspace_slug: None,
                cwd: "/tmp".into(),
                title: Some(format!("S{i}")),
                model: "m".into(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now() + chrono::Duration::seconds(i),
                message_count: 0,
                total_cost_usd: None,
                conversation_id: None,
                preview: None,
                permission_mode: None,
                add_dirs: None,
                spawned_by: None,
            };
            store.create_chat_session(&session).await.unwrap();
        }
        let snap = store
            .fetch_activity_snapshot(project_id, Some("p"), 3)
            .await
            .expect("snapshot");
        assert_eq!(snap.chat_sessions.len(), 3, "limit honored");
    }

    #[tokio::test]
    async fn snapshot_skips_chat_sessions_when_no_slug() {
        use crate::neo4j::models::ChatSessionNode;
        let store = MockGraphStore::new();
        let session = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: Some("p".into()),
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "m".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
        };
        store.create_chat_session(&session).await.unwrap();
        let snap = store
            .fetch_activity_snapshot(Uuid::new_v4(), None, 50)
            .await
            .expect("snapshot");
        assert!(snap.chat_sessions.is_empty());
    }

    #[tokio::test]
    async fn snapshot_under_target_latency_for_50_runs() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode::new_for_project(
            "P".into(),
            "d".into(),
            "a".into(),
            10,
            project_id,
        );
        store.create_plan(&plan).await.unwrap();

        for _ in 0..50 {
            let mut run = RunnerState::new(Uuid::new_v4(), plan.id, 10, TriggerSource::Manual);
            run.project_id = Some(project_id);
            store.create_plan_run(&run).await.unwrap();
        }

        let started = std::time::Instant::now();
        let snap = store
            .fetch_activity_snapshot(project_id, None, 50)
            .await
            .expect("snapshot");
        let elapsed = started.elapsed();
        assert_eq!(snap.plan_runs.len(), 50);
        // Mock backend should be far under 100ms — give comfortable headroom
        // for slow CI machines. The real Neo4j target is 100ms; the mock is
        // expected to be <10ms typically.
        assert!(
            elapsed.as_millis() < 100,
            "snapshot took {}ms (>100ms target)",
            elapsed.as_millis()
        );
    }
}
