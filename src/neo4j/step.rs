//! Neo4j Step operations

use super::client::{pascal_to_snake_case, Neo4jClient};
use super::models::*;
use crate::plan::models::UpdateStepRequest;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Step operations
    // ========================================================================

    /// Create a step for a task
    pub async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        let now = step.created_at.to_rfc3339();
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            CREATE (s:Step {
                id: $id,
                order: $order,
                description: $description,
                status: $status,
                verification: $verification,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            CREATE (t)-[:HAS_STEP]->(s)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("id", step.id.to_string())
        .param("order", step.order as i64)
        .param("description", step.description.clone())
        .param("status", format!("{:?}", step.status))
        .param(
            "verification",
            step.verification.clone().unwrap_or_default(),
        )
        .param("created_at", now.clone())
        .param("updated_at", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get steps for a task
    pub async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:HAS_STEP]->(s:Step)
            RETURN s
            ORDER BY s.order
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut steps = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            steps.push(StepNode {
                id: node.get::<String>("id")?.parse()?,
                order: node.get::<i64>("order")? as u32,
                description: node.get("description")?,
                status: serde_json::from_str(&format!(
                    "\"{}\"",
                    pascal_to_snake_case(&node.get::<String>("status")?)
                ))
                .unwrap_or(StepStatus::Pending),
                verification: node
                    .get::<String>("verification")
                    .ok()
                    .filter(|s| !s.is_empty()),
                created_at: node
                    .get::<String>("created_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: node
                    .get::<String>("updated_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                completed_at: node
                    .get::<String>("completed_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                execution_context: None,
                persona: None,
            });
        }

        Ok(steps)
    }

    /// Update step fields (description, verification)
    pub async fn update_step(&self, step_id: Uuid, updates: &UpdateStepRequest) -> Result<()> {
        let mut set_clauses = vec!["s.updated_at = datetime($now)"];

        if updates.description.is_some() {
            set_clauses.push("s.description = $description");
        }
        if updates.verification.is_some() {
            set_clauses.push("s.verification = $verification");
        }

        // If only status (no description/verification), skip — handled by update_step_status
        if set_clauses.len() == 1 && updates.description.is_none() && updates.verification.is_none()
        {
            return Ok(());
        }

        let cypher = format!("MATCH (s:Step {{id: $id}}) SET {}", set_clauses.join(", "));
        let now = chrono::Utc::now().to_rfc3339();
        let mut q = query(&cypher)
            .param("id", step_id.to_string())
            .param("now", now);

        if let Some(description) = &updates.description {
            q = q.param("description", description.clone());
        }
        if let Some(verification) = &updates.verification {
            q = q.param("verification", verification.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update step status
    pub async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let q = match status {
            StepStatus::Completed | StepStatus::Skipped => query(
                r#"
                MATCH (s:Step {id: $id})
                SET s.status = $status,
                    s.completed_at = datetime($now),
                    s.updated_at = datetime($now)
                "#,
            ),
            _ => query(
                r#"
                MATCH (s:Step {id: $id})
                SET s.status = $status,
                    s.updated_at = datetime($now)
                "#,
            ),
        }
        .param("id", step_id.to_string())
        .param("status", format!("{:?}", status))
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get count of completed steps for a task
    pub async fn get_task_step_progress(&self, task_id: Uuid) -> Result<(u32, u32)> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:HAS_STEP]->(s:Step)
            RETURN count(s) AS total,
                   sum(CASE WHEN s.status = 'Completed' THEN 1 ELSE 0 END) AS completed
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total")?;
            let completed: i64 = row.get("completed")?;
            Ok((completed as u32, total as u32))
        } else {
            Ok((0, 0))
        }
    }

    /// Get a single step by ID
    pub async fn get_step(&self, step_id: Uuid) -> Result<Option<StepNode>> {
        let q = query(
            r#"
            MATCH (s:Step {id: $id})
            RETURN s
            "#,
        )
        .param("id", step_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(StepNode {
                id: node.get::<String>("id")?.parse()?,
                order: node.get::<i64>("order")? as u32,
                description: node.get("description")?,
                status: serde_json::from_str(&format!(
                    "\"{}\"",
                    pascal_to_snake_case(&node.get::<String>("status")?)
                ))
                .unwrap_or(StepStatus::Pending),
                verification: node
                    .get::<String>("verification")
                    .ok()
                    .filter(|s| !s.is_empty()),
                created_at: node
                    .get::<String>("created_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: node
                    .get::<String>("updated_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                completed_at: node
                    .get::<String>("completed_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                execution_context: None,
                persona: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Delete a step
    pub async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (s:Step {id: $id})
            DETACH DELETE s
            "#,
        )
        .param("id", step_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }
}
