//! Neo4j PlanRun operations — runner execution state persistence

use super::client::Neo4jClient;
use crate::runner::{PlanRunStatus, RunnerState, TriggerSource};
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // PlanRun operations
    // ========================================================================

    /// Create a PlanRun node and link it to the Plan via (:PlanRun)-[:RUNS]->(:Plan).
    pub async fn create_plan_run(&self, state: &RunnerState) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (r:PlanRun {
                run_id: $run_id,
                plan_id: $plan_id,
                total_tasks: $total_tasks,
                current_wave: $current_wave,
                git_branch: $git_branch,
                started_at: datetime($started_at),
                status: $status,
                cost_usd: $cost_usd,
                triggered_by: $triggered_by,
                completed_tasks: $completed_tasks,
                failed_tasks: $failed_tasks,
                state_json: $state_json
            })
            CREATE (r)-[:RUNS]->(p)
            "#,
        )
        .param("run_id", state.run_id.to_string())
        .param("plan_id", state.plan_id.to_string())
        .param("total_tasks", state.total_tasks as i64)
        .param("current_wave", state.current_wave as i64)
        .param("git_branch", state.git_branch.clone())
        .param("started_at", state.started_at.to_rfc3339())
        .param("status", state.status.to_string())
        .param("cost_usd", state.cost_usd)
        .param(
            "triggered_by",
            serde_json::to_string(&state.triggered_by).unwrap_or_default(),
        )
        .param(
            "completed_tasks",
            state
                .completed_tasks
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
        .param(
            "failed_tasks",
            state
                .failed_tasks
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
        .param(
            "state_json",
            serde_json::to_string(state).unwrap_or_default(),
        );

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update an existing PlanRun with current execution state.
    /// Uses state_json for full fidelity (retry_counts, current_task_title, etc.).
    pub async fn update_plan_run(&self, state: &RunnerState) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (r:PlanRun {run_id: $run_id})
            SET r.current_wave = $current_wave,
                r.status = $status,
                r.cost_usd = $cost_usd,
                r.total_tasks = $total_tasks,
                r.completed_tasks = $completed_tasks,
                r.failed_tasks = $failed_tasks,
                r.state_json = $state_json
            "#,
        );

        if let Some(task_id) = state.current_task_id {
            cypher.push_str(&format!(", r.current_task_id = '{}'", task_id));
        } else {
            cypher.push_str(", r.current_task_id = null");
        }

        if let Some(completed_at) = state.completed_at {
            cypher.push_str(&format!(
                ", r.completed_at = datetime('{}')",
                completed_at.to_rfc3339()
            ));
        }

        let q = query(&cypher)
            .param("run_id", state.run_id.to_string())
            .param("current_wave", state.current_wave as i64)
            .param("status", state.status.to_string())
            .param("cost_usd", state.cost_usd)
            .param("total_tasks", state.total_tasks as i64)
            .param(
                "completed_tasks",
                state
                    .completed_tasks
                    .iter()
                    .map(|u| u.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .param(
                "failed_tasks",
                state
                    .failed_tasks
                    .iter()
                    .map(|u| u.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .param(
                "state_json",
                serde_json::to_string(state).unwrap_or_default(),
            );

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a PlanRun by its run_id.
    pub async fn get_plan_run(&self, run_id: Uuid) -> Result<Option<RunnerState>> {
        let q = query(
            r#"
            MATCH (r:PlanRun {run_id: $run_id})
            RETURN r
            "#,
        )
        .param("run_id", run_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(self.node_to_plan_run(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all PlanRuns with status=running (for crash recovery).
    pub async fn list_active_plan_runs_impl(&self) -> Result<Vec<RunnerState>> {
        let q = query(
            r#"
            MATCH (r:PlanRun {status: 'running'})
            RETURN r
            ORDER BY r.started_at DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut runs = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            runs.push(self.node_to_plan_run(&node)?);
        }
        Ok(runs)
    }

    /// List all PlanRuns across all plans, ordered by started_at desc.
    pub async fn list_all_plan_runs_impl(
        &self,
        limit: i64,
        offset: i64,
        status: Option<&str>,
    ) -> Result<Vec<RunnerState>> {
        let cypher = if status.is_some() {
            r#"
                MATCH (r:PlanRun)
                WHERE r.status = $status
                RETURN r
                ORDER BY r.started_at DESC
                SKIP $offset
                LIMIT $limit
                "#
            .to_string()
        } else {
            r#"
            MATCH (r:PlanRun)
            RETURN r
            ORDER BY r.started_at DESC
            SKIP $offset
            LIMIT $limit
            "#
            .to_string()
        };

        let mut q = query(&cypher).param("limit", limit).param("offset", offset);
        if let Some(s) = status {
            q = q.param("status", s.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut runs = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            runs.push(self.node_to_plan_run(&node)?);
        }
        Ok(runs)
    }

    /// List all PlanRuns for a given plan.
    pub async fn list_plan_runs_impl(&self, plan_id: Uuid, limit: i64) -> Result<Vec<RunnerState>> {
        let q = query(
            r#"
            MATCH (r:PlanRun {plan_id: $plan_id})
            RETURN r
            ORDER BY r.started_at DESC
            LIMIT $limit
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut runs = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            runs.push(self.node_to_plan_run(&node)?);
        }
        Ok(runs)
    }

    /// Convert a Neo4j node to RunnerState.
    ///
    /// Prefers `state_json` (full fidelity) but falls back to individual fields
    /// for backward compatibility.
    fn node_to_plan_run(&self, node: &neo4rs::Node) -> Result<RunnerState> {
        // Try state_json first (has all fields including retry_counts, current_task_title)
        if let Ok(json_str) = node.get::<String>("state_json") {
            if !json_str.is_empty() {
                if let Ok(state) = serde_json::from_str::<RunnerState>(&json_str) {
                    return Ok(state);
                }
            }
        }

        // Fallback: reconstruct from individual fields
        let run_id: String = node.get("run_id")?;
        let plan_id: String = node.get("plan_id")?;
        let status: String = node.get("status")?;
        let started_at: String = node.get("started_at")?;
        let completed_at: Option<String> = node.get("completed_at").ok();
        let current_task_id: Option<String> = node.get("current_task_id").ok();
        let triggered_by: String = node
            .get("triggered_by")
            .unwrap_or_else(|_| r#""manual""#.to_string());

        let completed_tasks_str: String = node.get("completed_tasks").unwrap_or_default();
        let failed_tasks_str: String = node.get("failed_tasks").unwrap_or_default();

        let parse_uuid_list = |s: &str| -> Vec<Uuid> {
            if s.is_empty() {
                return Vec::new();
            }
            s.split(',')
                .filter_map(|id| id.trim().parse::<Uuid>().ok())
                .collect()
        };

        let plan_run_status = match status.as_str() {
            "running" => PlanRunStatus::Running,
            "completed" => PlanRunStatus::Completed,
            "failed" => PlanRunStatus::Failed,
            "cancelled" => PlanRunStatus::Cancelled,
            "budget_exceeded" => PlanRunStatus::BudgetExceeded,
            _ => PlanRunStatus::Running,
        };

        Ok(RunnerState {
            run_id: run_id.parse()?,
            plan_id: plan_id.parse()?,
            total_tasks: node.get::<i64>("total_tasks").unwrap_or(0) as usize,
            current_wave: node.get::<i64>("current_wave").unwrap_or(0) as usize,
            current_task_id: current_task_id.and_then(|s| s.parse().ok()),
            current_task_title: None,
            active_agents: Vec::new(),
            completed_tasks: parse_uuid_list(&completed_tasks_str),
            failed_tasks: parse_uuid_list(&failed_tasks_str),
            retry_counts: std::collections::HashMap::new(),
            git_branch: node.get("git_branch").unwrap_or_default(),
            started_at: started_at.parse()?,
            completed_at: completed_at.and_then(|s| s.parse().ok()),
            status: plan_run_status,
            cost_usd: node.get("cost_usd").unwrap_or(0.0),
            triggered_by: serde_json::from_str(&triggered_by).unwrap_or(TriggerSource::Manual),
            project_id: node
                .get::<String>("project_id")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }
}
