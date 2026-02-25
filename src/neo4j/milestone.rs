//! Neo4j Milestone operations

use super::client::{pascal_to_snake_case, Neo4jClient, WhereBuilder};
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Milestone operations
    // ========================================================================

    /// Create a milestone
    pub async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            CREATE (m:Milestone {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                closed_at: $closed_at,
                created_at: datetime($created_at),
                project_id: $project_id
            })
            CREATE (p)-[:HAS_MILESTONE]->(m)
            "#,
        )
        .param("id", milestone.id.to_string())
        .param("title", milestone.title.clone())
        .param(
            "description",
            milestone.description.clone().unwrap_or_default(),
        )
        .param(
            "status",
            serde_json::to_value(&milestone.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        )
        .param("project_id", milestone.project_id.to_string())
        .param(
            "target_date",
            milestone
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param(
            "closed_at",
            milestone
                .closed_at
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("created_at", milestone.created_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a milestone by ID
    pub async fn get_milestone(&self, id: Uuid) -> Result<Option<MilestoneNode>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            RETURN m
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            Ok(Some(self.node_to_milestone(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to MilestoneNode
    pub(crate) fn node_to_milestone(&self, node: &neo4rs::Node) -> Result<MilestoneNode> {
        Ok(MilestoneNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get("title")?,
            description: node
                .get::<String>("description")
                .ok()
                .filter(|s| !s.is_empty()),
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(MilestoneStatus::Open),
            target_date: node
                .get::<String>("target_date")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            closed_at: node
                .get::<String>("closed_at")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            project_id: node.get::<String>("project_id")?.parse()?,
        })
    }

    /// List milestones for a project
    pub async fn list_project_milestones(&self, project_id: Uuid) -> Result<Vec<MilestoneNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_MILESTONE]->(m:Milestone)
            RETURN m
            ORDER BY m.target_date ASC, m.created_at ASC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut milestones = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            milestones.push(self.node_to_milestone(&node)?);
        }

        Ok(milestones)
    }

    /// Update a milestone
    pub async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if status.is_some() {
            set_clauses.push("m.status = $status");
        }
        if target_date.is_some() {
            set_clauses.push("m.target_date = $target_date");
        }
        if closed_at.is_some() {
            set_clauses.push("m.closed_at = $closed_at");
        }
        if title.is_some() {
            set_clauses.push("m.title = $title");
        }
        if description.is_some() {
            set_clauses.push("m.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (m:Milestone {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(ref s) = status {
            q = q.param(
                "status",
                serde_json::to_value(s)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
        }
        if let Some(d) = target_date {
            q = q.param("target_date", d.to_rfc3339());
        }
        if let Some(d) = closed_at {
            q = q.param("closed_at", d.to_rfc3339());
        }
        if let Some(ref t) = title {
            q = q.param("title", t.clone());
        }
        if let Some(ref d) = description {
            q = q.param("description", d.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a task to a milestone
    pub async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $milestone_id})
            MATCH (t:Task {id: $task_id})
            MERGE (m)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a plan to a project milestone
    pub async fn link_plan_to_milestone(&self, plan_id: Uuid, milestone_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (m:Milestone {id: $milestone_id})
            MERGE (p)-[:TARGETS_MILESTONE]->(m)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from a project milestone
    pub async fn unlink_plan_from_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[r:TARGETS_MILESTONE]->(m:Milestone {id: $milestone_id})
            DELETE r
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get milestone details with tasks
    pub async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            OPTIONAL MATCH (m)-[:INCLUDES_TASK]->(t:Task)
            RETURN m, collect(DISTINCT t) AS tasks
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let milestone_node: neo4rs::Node = row.get("m")?;
            let milestone = self.node_to_milestone(&milestone_node)?;

            let task_nodes: Vec<neo4rs::Node> = row.get("tasks").unwrap_or_default();
            let tasks: Vec<TaskNode> = task_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            Ok(Some((milestone, tasks)))
        } else {
            Ok(None)
        }
    }

    /// Get milestone progress (total, completed, in_progress, pending)
    pub async fn get_milestone_progress(&self, milestone_id: Uuid) -> Result<(u32, u32, u32, u32)> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN
                count(t) AS total,
                sum(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                sum(CASE WHEN t.status = 'InProgress' THEN 1 ELSE 0 END) AS in_progress,
                sum(CASE WHEN t.status = 'Pending' THEN 1 ELSE 0 END) AS pending
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total").unwrap_or(0);
            let completed: i64 = row.get("completed").unwrap_or(0);
            let in_progress: i64 = row.get("in_progress").unwrap_or(0);
            let pending: i64 = row.get("pending").unwrap_or(0);
            Ok((
                total as u32,
                completed as u32,
                in_progress as u32,
                pending as u32,
            ))
        } else {
            Ok((0, 0, 0, 0))
        }
    }

    /// Get tasks linked to a project milestone (with plan info)
    pub async fn get_milestone_tasks_with_plans(
        &self,
        milestone_id: Uuid,
    ) -> Result<Vec<TaskWithPlan>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $milestone_id})-[:INCLUDES_TASK]->(t:Task)
            OPTIONAL MATCH (p:Plan)-[:HAS_TASK]->(t)
            RETURN t, p.id AS plan_id, COALESCE(p.title, '') AS plan_title,
                   COALESCE(p.status, '') AS plan_status
            ORDER BY t.priority DESC, t.created_at
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let plan_id_str: String = row.get("plan_id").unwrap_or_default();
            let plan_title: String = row.get("plan_title").unwrap_or_default();
            let plan_status: String = row.get("plan_status").unwrap_or_default();
            tasks.push(TaskWithPlan {
                task: self.node_to_task(&node)?,
                plan_id: plan_id_str.parse().unwrap_or_default(),
                plan_title,
                plan_status: if plan_status.is_empty() {
                    None
                } else {
                    Some(pascal_to_snake_case(&plan_status))
                },
            });
        }
        Ok(tasks)
    }

    /// Get all steps for all tasks linked to a project milestone (batch query)
    pub async fn get_milestone_steps_batch(
        &self,
        milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $milestone_id})-[:INCLUDES_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
            RETURN t.id AS task_id, s
            ORDER BY t.id, s.order
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut steps_map: std::collections::HashMap<Uuid, Vec<StepNode>> =
            std::collections::HashMap::new();

        while let Some(row) = result.next().await? {
            let task_id_str: String = row.get("task_id")?;
            let task_id: Uuid = task_id_str.parse()?;
            let node: neo4rs::Node = row.get("s")?;
            let step = StepNode {
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
            };
            steps_map.entry(task_id).or_default().push(step);
        }

        Ok(steps_map)
    }

    /// Delete a milestone
    pub async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            DETACH DELETE m
            "#,
        )
        .param("id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks for a milestone
    pub async fn get_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// List project milestones with filters and pagination
    pub async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<MilestoneNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_status_filter("m", statuses);

        let where_clause = where_builder.build_and();
        let order_field = match sort_by {
            Some("title") => "m.title",
            Some("created_at") => "m.created_at",
            _ => "m.target_date",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!(
            "MATCH (p:Project {{id: $project_id}})-[:HAS_MILESTONE]->(m:Milestone) {} RETURN count(m) AS total",
            if where_clause.is_empty() { "" } else { &where_clause }
        );
        let count_result = self
            .execute_with_params(
                query(&count_cypher).param("project_id", project_id.to_string()),
            )
            .await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Project {{id: $project_id}})-[:HAS_MILESTONE]->(m:Milestone)
            {}
            RETURN m
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self
            .graph
            .execute(query(&cypher).param("project_id", project_id.to_string()))
            .await?;
        let mut milestones = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            milestones.push(self.node_to_milestone(&node)?);
        }

        Ok((milestones, total as usize))
    }
}
