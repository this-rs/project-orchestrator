//! Neo4j AgentExecution operations — per-agent execution tracking within a PlanRun.
//!
//! Each spawned agent gets its own AgentExecution node, linked to both the PlanRun
//! and the Task it executes. This enables per-agent vector collection and
//! fine-grained historical analysis.

use super::client::Neo4jClient;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

/// Represents an AgentExecution node in Neo4j.
///
/// Tracks a single agent's execution within a PlanRun, including its cost,
/// status, tools used, files modified, and commits produced.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentExecutionNode {
    pub id: Uuid,
    pub run_id: Uuid,
    pub task_id: Uuid,
    pub session_id: Option<Uuid>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub cost_usd: f64,
    pub duration_secs: f64,
    pub status: AgentExecutionStatus,
    pub tools_used: String,
    pub files_modified: Vec<String>,
    pub commits: Vec<String>,
    pub persona_profile: String,
    /// Per-agent execution vector JSON (serialized AgentExecutionVector)
    pub vector_json: Option<String>,
    /// Structured execution report JSON (serialized TaskExecutionReport)
    pub report_json: Option<String>,
    /// The type of execution (task agent, gate retry, verification).
    #[serde(default)]
    pub execution_type: ExecutionType,
}

/// Type of agent execution — distinguishes regular task runs from gate retries
/// and verification passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionType {
    /// Normal task execution by an agent.
    #[default]
    TaskAgent,
    /// Re-execution triggered by a quality gate failure.
    GateRetry,
    /// Verification pass (e.g., running tests after a fix).
    Verification,
}

impl std::fmt::Display for ExecutionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TaskAgent => write!(f, "task_agent"),
            Self::GateRetry => write!(f, "gate_retry"),
            Self::Verification => write!(f, "verification"),
        }
    }
}

impl ExecutionType {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "gate_retry" => Self::GateRetry,
            "verification" => Self::Verification,
            _ => Self::TaskAgent,
        }
    }
}

/// Status of an agent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentExecutionStatus {
    Running,
    Completed,
    Failed,
    Timeout,
}

impl std::fmt::Display for AgentExecutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

impl AgentExecutionStatus {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "timeout" => Self::Timeout,
            _ => Self::Running,
        }
    }
}

impl Neo4jClient {
    // ========================================================================
    // AgentExecution operations
    // ========================================================================

    /// Create an AgentExecution node and link it to the PlanRun and Task.
    ///
    /// Creates:
    /// - `(:AgentExecution)-[:PART_OF]->(:PlanRun)`
    /// - `(:AgentExecution)-[:EXECUTES]->(:Task)`
    pub async fn create_agent_execution_impl(&self, ae: &AgentExecutionNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:PlanRun {run_id: $run_id})
            MATCH (t:Task {id: $task_id})
            CREATE (ae:AgentExecution {
                id: $id,
                run_id: $run_id,
                task_id: $task_id,
                session_id: $session_id,
                started_at: datetime($started_at),
                cost_usd: $cost_usd,
                duration_secs: $duration_secs,
                status: $status,
                tools_used: $tools_used,
                files_modified: $files_modified,
                commits: $commits,
                persona_profile: $persona,
                execution_type: $execution_type
            })
            CREATE (ae)-[:PART_OF]->(r)
            CREATE (ae)-[:EXECUTES]->(t)
            "#,
        )
        .param("id", ae.id.to_string())
        .param("run_id", ae.run_id.to_string())
        .param("task_id", ae.task_id.to_string())
        .param(
            "session_id",
            ae.session_id.map(|s| s.to_string()).unwrap_or_default(),
        )
        .param("started_at", ae.started_at.to_rfc3339())
        .param("cost_usd", ae.cost_usd)
        .param("duration_secs", ae.duration_secs)
        .param("status", ae.status.to_string())
        .param("tools_used", ae.tools_used.clone())
        .param("files_modified", ae.files_modified.clone())
        .param("commits", ae.commits.clone())
        .param("persona", ae.persona_profile.clone())
        .param("execution_type", ae.execution_type.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update an existing AgentExecution node with final results.
    pub async fn update_agent_execution_impl(&self, ae: &AgentExecutionNode) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (ae:AgentExecution {id: $id})
            SET ae.cost_usd = $cost_usd,
                ae.duration_secs = $duration_secs,
                ae.status = $status,
                ae.tools_used = $tools_used,
                ae.files_modified = $files_modified,
                ae.commits = $commits
            "#,
        );

        if let Some(completed_at) = ae.completed_at {
            cypher.push_str(&format!(
                ", ae.completed_at = datetime('{}')",
                completed_at.to_rfc3339()
            ));
        }

        if ae.vector_json.is_some() {
            // Use parameter for vector_json to avoid injection
            cypher.push_str(", ae.vector_json = $vector_json");
        }

        if ae.report_json.is_some() {
            cypher.push_str(", ae.report_json = $report_json");
        }

        let mut q = query(&cypher)
            .param("id", ae.id.to_string())
            .param("cost_usd", ae.cost_usd)
            .param("duration_secs", ae.duration_secs)
            .param("status", ae.status.to_string())
            .param("tools_used", ae.tools_used.clone())
            .param("files_modified", ae.files_modified.clone())
            .param("commits", ae.commits.clone());

        if let Some(ref vector_json) = ae.vector_json {
            q = q.param("vector_json", vector_json.clone());
        }

        if let Some(ref report_json) = ae.report_json {
            q = q.param("report_json", report_json.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get all AgentExecution nodes for a given PlanRun.
    pub async fn get_agent_executions_for_run_impl(
        &self,
        run_id: Uuid,
    ) -> Result<Vec<AgentExecutionNode>> {
        let q = query(
            r#"
            MATCH (ae:AgentExecution {run_id: $run_id})
            RETURN ae
            ORDER BY ae.started_at ASC
            "#,
        )
        .param("run_id", run_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut executions = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("ae")?;
            executions.push(self.node_to_agent_execution(&node)?);
        }
        Ok(executions)
    }

    /// Create a USED_SKILL relationship from an AgentExecution to a Skill.
    pub async fn create_used_skill_relation_impl(
        &self,
        agent_execution_id: Uuid,
        skill_id: Uuid,
        result: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (ae:AgentExecution {id: $ae_id})
            MATCH (s:Skill {id: $skill_id})
            MERGE (ae)-[r:USED_SKILL]->(s)
            SET r.result = $result,
                r.timestamp = datetime()
            "#,
        )
        .param("ae_id", agent_execution_id.to_string())
        .param("skill_id", skill_id.to_string())
        .param("result", result);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Convert a Neo4j node to AgentExecutionNode.
    fn node_to_agent_execution(&self, node: &neo4rs::Node) -> Result<AgentExecutionNode> {
        let id: String = node.get("id")?;
        let run_id: String = node.get("run_id")?;
        let task_id: String = node.get("task_id")?;
        let session_id: Option<String> = node.get("session_id").ok();
        let started_at: String = node.get("started_at")?;
        let completed_at: Option<String> = node.get("completed_at").ok();
        let status: String = node.get("status")?;

        let files_modified: Vec<String> = node.get("files_modified").unwrap_or_default();
        let commits: Vec<String> = node.get("commits").unwrap_or_default();

        Ok(AgentExecutionNode {
            id: id.parse()?,
            run_id: run_id.parse()?,
            task_id: task_id.parse()?,
            session_id: session_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            started_at: started_at.parse()?,
            completed_at: completed_at.and_then(|s| s.parse().ok()),
            cost_usd: node.get("cost_usd").unwrap_or(0.0),
            duration_secs: node.get("duration_secs").unwrap_or(0.0),
            status: AgentExecutionStatus::from_str_lossy(&status),
            tools_used: node.get("tools_used").unwrap_or_default(),
            files_modified,
            commits,
            persona_profile: node.get("persona_profile").unwrap_or_default(),
            vector_json: node.get("vector_json").ok(),
            report_json: node.get("report_json").ok(),
            execution_type: node
                .get::<String>("execution_type")
                .map(|s| ExecutionType::from_str_lossy(&s))
                .unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_agent_execution(
        report_json: Option<String>,
        vector_json: Option<String>,
    ) -> AgentExecutionNode {
        AgentExecutionNode {
            id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            session_id: Some(Uuid::new_v4()),
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            cost_usd: 0.05,
            duration_secs: 12.5,
            status: AgentExecutionStatus::Completed,
            tools_used: "note,code".to_string(),
            files_modified: vec!["src/main.rs".to_string()],
            commits: vec!["abc123".to_string()],
            persona_profile: "test-profile".to_string(),
            vector_json,
            report_json,
            execution_type: ExecutionType::TaskAgent,
        }
    }

    #[test]
    fn test_agent_execution_node_serialize_with_report_json() {
        let ae = make_agent_execution(
            Some(r#"{"summary":"ok"}"#.to_string()),
            Some(r#"{"vec":[1,2]}"#.to_string()),
        );
        let json = serde_json::to_string(&ae).unwrap();
        assert!(json.contains("report_json"));
        // report_json is a String field, so the inner JSON is escaped in the outer JSON
        assert!(json.contains("summary"));
        assert!(json.contains("vector_json"));
        assert!(json.contains("vec"));
    }

    #[test]
    fn test_agent_execution_node_serialize_without_report_json() {
        let ae = make_agent_execution(None, None);
        let json = serde_json::to_string(&ae).unwrap();
        assert!(json.contains("\"report_json\":null"));
        assert!(json.contains("\"vector_json\":null"));
    }

    #[test]
    fn test_agent_execution_node_roundtrip() {
        let ae = make_agent_execution(
            Some(r#"{"tasks_completed":3}"#.to_string()),
            Some(r#"{"energy":0.9}"#.to_string()),
        );
        let json = serde_json::to_string(&ae).unwrap();
        let deserialized: AgentExecutionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.report_json, ae.report_json);
        assert_eq!(deserialized.vector_json, ae.vector_json);
        assert_eq!(deserialized.id, ae.id);
        assert_eq!(deserialized.cost_usd, ae.cost_usd);
        assert_eq!(deserialized.status, ae.status);
    }

    #[test]
    fn test_agent_execution_status_display() {
        assert_eq!(AgentExecutionStatus::Running.to_string(), "running");
        assert_eq!(AgentExecutionStatus::Completed.to_string(), "completed");
        assert_eq!(AgentExecutionStatus::Failed.to_string(), "failed");
        assert_eq!(AgentExecutionStatus::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_agent_execution_status_from_str_lossy() {
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("completed"),
            AgentExecutionStatus::Completed
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("failed"),
            AgentExecutionStatus::Failed
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("timeout"),
            AgentExecutionStatus::Timeout
        );
        // Unknown strings default to Running
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("running"),
            AgentExecutionStatus::Running
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("unknown"),
            AgentExecutionStatus::Running
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy(""),
            AgentExecutionStatus::Running
        );
    }

    #[test]
    fn test_agent_execution_node_without_session_id() {
        let mut ae = make_agent_execution(None, None);
        ae.session_id = None;
        ae.completed_at = None;
        let json = serde_json::to_string(&ae).unwrap();
        let deserialized: AgentExecutionNode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.session_id.is_none());
        assert!(deserialized.completed_at.is_none());
    }

    #[test]
    fn test_agent_execution_status_serialize_roundtrip() {
        for status in &[
            AgentExecutionStatus::Running,
            AgentExecutionStatus::Completed,
            AgentExecutionStatus::Failed,
            AgentExecutionStatus::Timeout,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let deserialized: AgentExecutionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, deserialized);
        }
    }

    #[test]
    fn test_agent_execution_node_debug() {
        let ae = make_agent_execution(Some("report".to_string()), None);
        let debug = format!("{:?}", ae);
        assert!(debug.contains("AgentExecutionNode"));
        assert!(debug.contains("report_json"));
    }

    #[test]
    fn test_agent_execution_node_clone() {
        let ae = make_agent_execution(Some("report".to_string()), Some("vector".to_string()));
        let cloned = ae.clone();
        assert_eq!(cloned.id, ae.id);
        assert_eq!(cloned.report_json, ae.report_json);
        assert_eq!(cloned.vector_json, ae.vector_json);
    }

    #[test]
    fn test_execution_type_display() {
        assert_eq!(ExecutionType::TaskAgent.to_string(), "task_agent");
        assert_eq!(ExecutionType::GateRetry.to_string(), "gate_retry");
        assert_eq!(ExecutionType::Verification.to_string(), "verification");
    }

    #[test]
    fn test_execution_type_from_str_lossy() {
        assert_eq!(
            ExecutionType::from_str_lossy("gate_retry"),
            ExecutionType::GateRetry
        );
        assert_eq!(
            ExecutionType::from_str_lossy("verification"),
            ExecutionType::Verification
        );
        assert_eq!(
            ExecutionType::from_str_lossy("task_agent"),
            ExecutionType::TaskAgent
        );
        // Unknown defaults to TaskAgent
        assert_eq!(
            ExecutionType::from_str_lossy("unknown"),
            ExecutionType::TaskAgent
        );
        assert_eq!(ExecutionType::from_str_lossy(""), ExecutionType::TaskAgent);
    }

    #[test]
    fn test_execution_type_default() {
        let default: ExecutionType = Default::default();
        assert_eq!(default, ExecutionType::TaskAgent);
    }

    #[test]
    fn test_execution_type_serde_roundtrip() {
        for variant in &[
            ExecutionType::TaskAgent,
            ExecutionType::GateRetry,
            ExecutionType::Verification,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: ExecutionType = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, deserialized);
        }
    }
}
