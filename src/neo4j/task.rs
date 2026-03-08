//! Neo4j Task operations

use super::client::{pascal_to_snake_case, Neo4jClient, WhereBuilder};
use super::models::*;
use crate::plan::models::TaskDetails;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Task operations
    // ========================================================================

    /// Create a task for a plan
    pub async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> Result<()> {
        let now = task.created_at.to_rfc3339();
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (t:Task {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                priority: $priority,
                tags: $tags,
                acceptance_criteria: $acceptance_criteria,
                affected_files: $affected_files,
                estimated_complexity: $estimated_complexity,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            CREATE (p)-[:HAS_TASK]->(t)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("id", task.id.to_string())
        .param("title", task.title.clone().unwrap_or_default())
        .param("description", task.description.clone())
        .param("status", format!("{:?}", task.status))
        .param("priority", task.priority.unwrap_or(0) as i64)
        .param("tags", task.tags.clone())
        .param("acceptance_criteria", task.acceptance_criteria.clone())
        .param("affected_files", task.affected_files.clone())
        .param(
            "estimated_complexity",
            task.estimated_complexity.map(|c| c as i64).unwrap_or(0),
        )
        .param("created_at", now.clone())
        .param("updated_at", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks for a plan
    pub async fn get_plan_tasks(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Helper to convert Neo4j node to TaskNode
    pub(crate) fn node_to_task(&self, node: &neo4rs::Node) -> Result<TaskNode> {
        Ok(TaskNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get::<String>("title").ok().filter(|s| !s.is_empty()),
            description: node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(TaskStatus::Pending),
            assigned_to: node.get("assigned_to").ok(),
            priority: node.get::<i64>("priority").ok().map(|v| v as i32),
            tags: node.get("tags").unwrap_or_default(),
            acceptance_criteria: node.get("acceptance_criteria").unwrap_or_default(),
            affected_files: node.get("affected_files").unwrap_or_default(),
            estimated_complexity: node
                .get::<i64>("estimated_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            actual_complexity: node
                .get::<i64>("actual_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            started_at: node
                .get::<String>("started_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            completed_at: node
                .get::<String>("completed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    /// Convert a Neo4j Node to a StepNode
    fn node_to_step(&self, node: &neo4rs::Node) -> Option<StepNode> {
        Some(StepNode {
            id: node.get::<String>("id").ok()?.parse().ok()?,
            order: node.get::<i64>("order").ok()? as u32,
            description: node.get::<String>("description").ok()?,
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| serde_json::from_str(&format!("\"{}\"", s.to_lowercase())).ok())
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
        })
    }

    // node_to_decision is defined in decision.rs as an associated function

    /// Get full task details including steps, decisions, dependencies, and modified files
    pub async fn get_task_with_full_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            OPTIONAL MATCH (t)-[:HAS_STEP]->(s:Step)
            OPTIONAL MATCH (t)-[:INFORMED_BY]->(d:Decision)
            OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
            OPTIONAL MATCH (t)-[:MODIFIES]->(f:File)
            RETURN t,
                   collect(DISTINCT s) AS steps,
                   collect(DISTINCT d) AS decisions,
                   collect(DISTINCT dep.id) AS depends_on,
                   collect(DISTINCT f.path) AS files
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;

        let row = match result.next().await? {
            Some(r) => r,
            None => return Ok(None),
        };

        let task_node: neo4rs::Node = row.get("t")?;
        let task = self.node_to_task(&task_node)?;

        // Parse steps
        let step_nodes: Vec<neo4rs::Node> = row.get("steps").unwrap_or_default();
        let mut steps: Vec<StepNode> = step_nodes
            .iter()
            .filter_map(|n| self.node_to_step(n))
            .collect();
        steps.sort_by_key(|s| s.order);

        // Parse decisions
        let decision_nodes: Vec<neo4rs::Node> = row.get("decisions").unwrap_or_default();
        let decisions: Vec<DecisionNode> = decision_nodes
            .iter()
            .filter_map(|n| Self::node_to_decision(n).ok())
            .collect();

        // Parse dependencies
        let depends_on_strs: Vec<String> = row.get("depends_on").unwrap_or_default();
        let depends_on: Vec<Uuid> = depends_on_strs
            .into_iter()
            .filter_map(|s| s.parse().ok())
            .collect();

        let modifies_files: Vec<String> = row.get("files").unwrap_or_default();

        Ok(Some(TaskDetails {
            task,
            steps,
            decisions,
            depends_on,
            modifies_files,
        }))
    }

    /// Analyze the impact of a task on the codebase (files it modifies + their dependents)
    pub async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:MODIFIES]->(f:File)
            OPTIONAL MATCH (f)<-[:IMPORTS*1..3]-(dependent:File)
            RETURN f.path AS file, collect(DISTINCT dependent.path) AS dependents
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut impacted = Vec::new();

        while let Some(row) = result.next().await? {
            let file: String = row.get("file")?;
            impacted.push(file);
            let dependents: Vec<String> = row.get("dependents").unwrap_or_default();
            impacted.extend(dependents);
        }

        impacted.sort();
        impacted.dedup();
        Ok(impacted)
    }

    /// Find pending tasks in a plan that are blocked by uncompleted dependencies
    pub async fn find_blocked_tasks(
        &self,
        plan_id: Uuid,
    ) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task {status: 'Pending'})
            MATCH (t)-[:DEPENDS_ON]->(blocker:Task)
            WHERE blocker.status <> 'Completed'
            RETURN t, collect(blocker) AS blockers
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut blocked = Vec::new();

        while let Some(row) = result.next().await? {
            let task_node: neo4rs::Node = row.get("t")?;
            let task = self.node_to_task(&task_node)?;

            let blocker_nodes: Vec<neo4rs::Node> = row.get("blockers").unwrap_or_default();
            let blockers: Vec<TaskNode> = blocker_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            blocked.push((task, blockers));
        }

        Ok(blocked)
    }

    /// Update task status
    pub async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let q = match status {
            TaskStatus::InProgress => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.started_at = datetime($now),
                    t.updated_at = datetime($now)
                "#,
            ),
            TaskStatus::Completed | TaskStatus::Failed => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.completed_at = datetime($now),
                    t.updated_at = datetime($now)
                "#,
            ),
            _ => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.updated_at = datetime($now)
                "#,
            ),
        }
        .param("id", task_id.to_string())
        .param("status", format!("{:?}", status))
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Assign task to an agent
    pub async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (a:Agent {id: $agent_id})
            SET t.assigned_to = $agent_id
            MERGE (a)-[:WORKING_ON]->(t)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("agent_id", agent_id);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add task dependency
    pub async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (dep:Task {id: $depends_on_id})
            MERGE (t)-[:DEPENDS_ON]->(dep)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove task dependency
    pub async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[r:DEPENDS_ON]->(dep:Task {id: $depends_on_id})
            DELETE r
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks that block this task (dependencies that are not completed)
    pub async fn get_task_blockers(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:DEPENDS_ON]->(blocker:Task)
            WHERE blocker.status <> 'Completed'
            RETURN blocker
            ORDER BY COALESCE(blocker.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("blocker")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get tasks blocked by this task (tasks depending on this one)
    pub async fn get_tasks_blocked_by(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (blocked:Task)-[:DEPENDS_ON]->(t:Task {id: $task_id})
            RETURN blocked
            ORDER BY COALESCE(blocked.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("blocked")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get all dependencies for a task (all tasks it depends on, regardless of status)
    pub async fn get_task_dependencies(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:DEPENDS_ON]->(dep:Task)
            RETURN dep
            ORDER BY COALESCE(dep.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("dep")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get next available task (no unfinished dependencies)
    pub async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task {status: 'Pending'})
            WHERE NOT EXISTS {
                MATCH (t)-[:DEPENDS_ON]->(dep:Task)
                WHERE dep.status <> 'Completed'
            }
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            LIMIT 1
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_task(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a single task by ID
    pub async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            RETURN t
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_task(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a task with new values
    pub async fn update_task(
        &self,
        task_id: Uuid,
        updates: &crate::plan::models::UpdateTaskRequest,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if updates.title.is_some() {
            set_clauses.push("t.title = $title");
        }
        if updates.description.is_some() {
            set_clauses.push("t.description = $description");
        }
        if updates.priority.is_some() {
            set_clauses.push("t.priority = $priority");
        }
        if updates.tags.is_some() {
            set_clauses.push("t.tags = $tags");
        }
        if updates.acceptance_criteria.is_some() {
            set_clauses.push("t.acceptance_criteria = $acceptance_criteria");
        }
        if updates.affected_files.is_some() {
            set_clauses.push("t.affected_files = $affected_files");
        }
        if updates.actual_complexity.is_some() {
            set_clauses.push("t.actual_complexity = $actual_complexity");
        }
        if updates.estimated_complexity.is_some() {
            set_clauses.push("t.estimated_complexity = $estimated_complexity");
        }
        if updates.assigned_to.is_some() {
            set_clauses.push("t.assigned_to = $assigned_to");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        // Always update updated_at
        set_clauses.push("t.updated_at = datetime($updated_at)");

        let cypher = format!("MATCH (t:Task {{id: $id}}) SET {}", set_clauses.join(", "));

        let mut q = query(&cypher)
            .param("id", task_id.to_string())
            .param("updated_at", chrono::Utc::now().to_rfc3339());

        if let Some(ref title) = updates.title {
            q = q.param("title", title.clone());
        }
        if let Some(ref desc) = updates.description {
            q = q.param("description", desc.clone());
        }
        if let Some(priority) = updates.priority {
            q = q.param("priority", priority as i64);
        }
        if let Some(ref tags) = updates.tags {
            q = q.param("tags", tags.clone());
        }
        if let Some(ref criteria) = updates.acceptance_criteria {
            q = q.param("acceptance_criteria", criteria.clone());
        }
        if let Some(ref files) = updates.affected_files {
            q = q.param("affected_files", files.clone());
        }
        if let Some(complexity) = updates.actual_complexity {
            q = q.param("actual_complexity", complexity as i64);
        }
        if let Some(complexity) = updates.estimated_complexity {
            q = q.param("estimated_complexity", complexity as i64);
        }
        if let Some(ref assigned) = updates.assigned_to {
            q = q.param("assigned_to", assigned.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a task and all its related data (steps, decisions)
    pub async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        // Delete all steps belonging to this task
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", task_id.to_string());
        self.graph.run(q).await?;

        // Delete all decisions belonging to this task
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:INFORMED_BY]->(d:Decision)
            DETACH DELETE d
            "#,
        )
        .param("id", task_id.to_string());
        self.graph.run(q).await?;

        // Delete the task itself
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            DETACH DELETE t
            "#,
        )
        .param("id", task_id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Get the project that owns a task (via Plan←Task, Project←Plan chain).
    ///
    /// Traverses: `(Project)-[:HAS_PLAN]->(Plan)-[:HAS_TASK]->(Task)`
    /// Returns `None` if the task doesn't exist or has no linked project.
    pub async fn get_project_for_task(&self, task_id: Uuid) -> Result<Option<ProjectNode>> {
        let q = query(
            r#"
            MATCH (proj:Project)-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task {id: $task_id})
            RETURN proj
            LIMIT 1
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("proj")?;
            let project = self.node_to_project(&node)?;
            Ok(Some(project))
        } else {
            Ok(None)
        }
    }

    /// Link a task to files it modifies
    pub async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> Result<()> {
        for path in file_paths {
            let q = query(
                r#"
                MATCH (t:Task {id: $task_id})
                MATCH (f:File {path: $path})
                MERGE (t)-[:MODIFIES]->(f)
                "#,
            )
            .param("task_id", task_id.to_string())
            .param("path", path.clone());

            self.graph.run(q).await?;
        }
        Ok(())
    }

    /// List all tasks across all plans with filters and pagination
    ///
    /// Returns (tasks_with_plan_info, total_count)
    #[allow(clippy::too_many_arguments)]
    pub async fn list_all_tasks_filtered(
        &self,
        plan_id: Option<Uuid>,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        tags: Option<Vec<String>>,
        assigned_to: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<TaskWithPlan>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder
            .add_status_filter("t", statuses)
            .add_priority_filter("t", priority_min, priority_max)
            .add_tags_filter("t", tags)
            .add_assigned_to_filter("t", assigned_to);

        // Build plan filter if specified
        let plan_match = if let Some(pid) = plan_id {
            format!("MATCH (p:Plan {{id: '{}'}})-[:HAS_TASK]->(t:Task)", pid)
        } else if let Some(pid) = project_id {
            format!(
                "MATCH (proj:Project {{id: '{}'}})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)",
                pid
            )
        } else if let Some(ws) = workspace_slug {
            format!(
                "MATCH (w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)",
                ws
            )
        } else {
            "MATCH (p:Plan)-[:HAS_TASK]->(t:Task)".to_string()
        };

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("priority") => "COALESCE(t.priority, 0)",
            Some("title") => "t.title",
            Some("status") => "t.status",
            Some("created_at") => "t.created_at",
            Some("updated_at") => "t.updated_at",
            _ => "COALESCE(t.priority, 0) DESC, t.created_at",
        };
        let order_dir = if sort_by.is_some() && sort_order == "asc" {
            "ASC"
        } else if sort_by.is_some() {
            "DESC"
        } else {
            "" // Default ordering already includes direction
        };

        // Count query
        let count_cypher = format!("{} {} RETURN count(t) AS total", plan_match, where_clause);
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            {}
            {}
            RETURN t, p.id AS plan_id, p.title AS plan_title,
                   COALESCE(p.status, '') AS plan_status
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            plan_match, where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut tasks = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let plan_id_str: String = row.get("plan_id")?;
            let plan_title: String = row.get("plan_title")?;
            let plan_status: String = row.get("plan_status").unwrap_or_default();
            tasks.push(TaskWithPlan {
                task: self.node_to_task(&node)?,
                plan_id: plan_id_str.parse()?,
                plan_title,
                plan_status: if plan_status.is_empty() {
                    None
                } else {
                    Some(pascal_to_snake_case(&plan_status))
                },
            });
        }

        Ok((tasks, total as usize))
    }
}
