//! Neo4j Workspace operations

use super::client::{pascal_to_snake_case, snake_to_pascal_case, Neo4jClient};
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Workspace operations
    // ========================================================================

    /// Create a new workspace
    pub async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (w:Workspace {
                id: $id,
                name: $name,
                slug: $slug,
                description: $description,
                created_at: datetime($created_at),
                metadata: $metadata
            })
            "#,
        )
        .param("id", workspace.id.to_string())
        .param("name", workspace.name.clone())
        .param("slug", workspace.slug.clone())
        .param(
            "description",
            workspace.description.clone().unwrap_or_default(),
        )
        .param("created_at", workspace.created_at.to_rfc3339())
        .param("metadata", workspace.metadata.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a workspace by ID
    pub async fn get_workspace(&self, id: Uuid) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})
            RETURN w
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a workspace by slug
    pub async fn get_workspace_by_slug(&self, slug: &str) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {slug: $slug})
            RETURN w
            "#,
        )
        .param("slug", slug);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all workspaces
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace)
            RETURN w
            ORDER BY w.name
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut workspaces = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            workspaces.push(self.node_to_workspace(&node)?);
        }

        Ok(workspaces)
    }

    /// Update a workspace
    pub async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let mut set_clauses = vec!["w.updated_at = datetime($now)".to_string()];

        if name.is_some() {
            set_clauses.push("w.name = $name".to_string());
        }
        if description.is_some() {
            set_clauses.push("w.description = $description".to_string());
        }
        if metadata.is_some() {
            set_clauses.push("w.metadata = $metadata".to_string());
        }

        let cypher = format!(
            r#"
            MATCH (w:Workspace {{id: $id}})
            SET {}
            "#,
            set_clauses.join(", ")
        );

        let mut q = query(&cypher)
            .param("id", id.to_string())
            .param("now", chrono::Utc::now().to_rfc3339());

        if let Some(n) = name {
            q = q.param("name", n);
        }
        if let Some(d) = description {
            q = q.param("description", d);
        }
        if let Some(m) = metadata {
            q = q.param("metadata", m.to_string());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a workspace and all its data
    pub async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        // Delete workspace milestones
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            DETACH DELETE wm
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete resources owned by workspace
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_RESOURCE]->(r:Resource)
            DETACH DELETE r
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete components
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_COMPONENT]->(c:Component)
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Remove workspace association from projects (don't delete projects)
        let q = query(
            r#"
            MATCH (p:Project)-[r:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $id})
            DELETE r
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete the workspace itself
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})
            DETACH DELETE w
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Add a project to a workspace
    pub async fn add_project_to_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            MATCH (p:Project {id: $project_id})
            MERGE (p)-[:BELONGS_TO_WORKSPACE]->(w)
            "#,
        )
        .param("workspace_id", workspace_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a project from a workspace
    pub async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[r:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $workspace_id})
            DELETE r
            "#,
        )
        .param("workspace_id", workspace_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List all projects in a workspace
    pub async fn list_workspace_projects(&self, workspace_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $workspace_id})
            RETURN p
            ORDER BY p.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Get the workspace a project belongs to
    pub async fn get_project_workspace(&self, project_id: Uuid) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:BELONGS_TO_WORKSPACE]->(w:Workspace)
            RETURN w
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to WorkspaceNode
    fn node_to_workspace(&self, node: &neo4rs::Node) -> Result<WorkspaceNode> {
        let metadata_str: String = node.get("metadata").unwrap_or_else(|_| "{}".to_string());
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_str).unwrap_or(serde_json::json!({}));

        Ok(WorkspaceNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            slug: node.get("slug")?,
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            metadata,
        })
    }

    // ========================================================================
    // Workspace Milestone operations
    // ========================================================================

    /// Create a workspace milestone
    pub async fn create_workspace_milestone(
        &self,
        milestone: &WorkspaceMilestoneNode,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            CREATE (wm:WorkspaceMilestone {
                id: $id,
                workspace_id: $workspace_id,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                created_at: datetime($created_at),
                tags: $tags
            })
            CREATE (w)-[:HAS_WORKSPACE_MILESTONE]->(wm)
            "#,
        )
        .param("id", milestone.id.to_string())
        .param("workspace_id", milestone.workspace_id.to_string())
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
        .param(
            "target_date",
            milestone
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("created_at", milestone.created_at.to_rfc3339())
        .param("tags", milestone.tags.clone());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a workspace milestone by ID
    pub async fn get_workspace_milestone(
        &self,
        id: Uuid,
    ) -> Result<Option<WorkspaceMilestoneNode>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $id})
            RETURN wm
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            Ok(Some(self.node_to_workspace_milestone(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List workspace milestones (unpaginated, used internally)
    pub async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<WorkspaceMilestoneNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            RETURN wm
            ORDER BY wm.target_date, wm.title
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut milestones = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            milestones.push(self.node_to_workspace_milestone(&node)?);
        }

        Ok(milestones)
    }

    /// List workspace milestones with pagination and status filter
    ///
    /// Returns (milestones, total_count)
    pub async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<WorkspaceMilestoneNode>, usize)> {
        let status_filter = if let Some(s) = status {
            format!("WHERE toLower(wm.status) = toLower('{}')", s)
        } else {
            String::new()
        };

        let count_cypher = format!(
            "MATCH (w:Workspace {{id: $workspace_id}})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone) {} RETURN count(wm) AS total",
            status_filter
        );
        let mut count_stream = self
            .graph
            .execute(query(&count_cypher).param("workspace_id", workspace_id.to_string()))
            .await?;
        let total: i64 = if let Some(row) = count_stream.next().await? {
            row.get("total")?
        } else {
            0
        };

        let data_cypher = format!(
            r#"
            MATCH (w:Workspace {{id: $workspace_id}})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            {}
            RETURN wm
            ORDER BY wm.target_date, wm.title
            SKIP {}
            LIMIT {}
            "#,
            status_filter, offset, limit
        );

        let mut result = self
            .graph
            .execute(query(&data_cypher).param("workspace_id", workspace_id.to_string()))
            .await?;
        let mut milestones = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            milestones.push(self.node_to_workspace_milestone(&node)?);
        }

        Ok((milestones, total as usize))
    }

    /// List all workspace milestones across all workspaces with filters and pagination
    ///
    /// Returns (milestones_with_workspace_info, total_count)
    pub async fn list_all_workspace_milestones_filtered(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(WorkspaceMilestoneNode, String, String, String)>> {
        let mut conditions = Vec::new();
        if let Some(wid) = workspace_id {
            conditions.push(format!("w.id = '{}'", wid));
        }
        if let Some(s) = status {
            let pascal = snake_to_pascal_case(s);
            conditions.push(format!("wm.status = '{}'", pascal));
        }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let cypher = format!(
            r#"
            MATCH (w:Workspace)-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            {}
            RETURN wm, w.id AS workspace_id, w.name AS workspace_name, w.slug AS workspace_slug
            ORDER BY wm.target_date, wm.title
            SKIP {}
            LIMIT {}
            "#,
            where_clause, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut items = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            let wid: String = row.get("workspace_id")?;
            let wname: String = row.get("workspace_name")?;
            let wslug: String = row.get("workspace_slug")?;
            items.push((self.node_to_workspace_milestone(&node)?, wid, wname, wslug));
        }

        Ok(items)
    }

    /// Count all workspace milestones across workspaces with optional filters
    pub async fn count_all_workspace_milestones(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
    ) -> Result<usize> {
        let mut conditions = Vec::new();
        if let Some(wid) = workspace_id {
            conditions.push(format!("w.id = '{}'", wid));
        }
        if let Some(s) = status {
            let pascal = snake_to_pascal_case(s);
            conditions.push(format!("wm.status = '{}'", pascal));
        }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let cypher = format!(
            "MATCH (w:Workspace)-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone) {} RETURN count(wm) AS total",
            where_clause
        );
        let count_result = self.execute(&cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        Ok(total as usize)
    }

    /// Update a workspace milestone
    pub async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if title.is_some() {
            set_clauses.push("wm.title = $title".to_string());
        }
        if description.is_some() {
            set_clauses.push("wm.description = $description".to_string());
        }
        if status.is_some() {
            set_clauses.push("wm.status = $status".to_string());
        }
        if target_date.is_some() {
            set_clauses.push("wm.target_date = $target_date".to_string());
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            r#"
            MATCH (wm:WorkspaceMilestone {{id: $id}})
            SET {}
            "#,
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(t) = title {
            q = q.param("title", t);
        }
        if let Some(d) = description {
            q = q.param("description", d);
        }
        if let Some(s) = status {
            q = q.param(
                "status",
                serde_json::to_value(&s)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
        }
        if let Some(td) = target_date {
            q = q.param("target_date", td.to_rfc3339());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a workspace milestone
    pub async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $id})
            DETACH DELETE wm
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a task to a workspace milestone
    pub async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            MATCH (t:Task {id: $task_id})
            MERGE (wm)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a task from a workspace milestone
    pub async fn remove_task_from_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})-[r:INCLUDES_TASK]->(t:Task {id: $task_id})
            DELETE r
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a plan to a workspace milestone
    pub async fn link_plan_to_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            MERGE (p)-[:TARGETS_MILESTONE]->(wm)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from a workspace milestone
    pub async fn unlink_plan_from_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[r:TARGETS_MILESTONE]->(wm:WorkspaceMilestone {id: $milestone_id})
            DELETE r
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get workspace milestone progress (direct + plan-based task links)
    pub async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            OPTIONAL MATCH (wm)-[:INCLUDES_TASK]->(t1:Task)
            OPTIONAL MATCH (p:Plan)-[:TARGETS_MILESTONE]->(wm), (p)-[:HAS_TASK]->(t2:Task)
            WITH [x IN collect(DISTINCT t1) + collect(DISTINCT t2) WHERE x IS NOT NULL] AS tasks
            RETURN
                size(tasks) AS total,
                size([t IN tasks WHERE t.status = 'Completed']) AS completed,
                size([t IN tasks WHERE t.status = 'InProgress']) AS in_progress,
                size([t IN tasks WHERE t.status = 'Pending']) AS pending
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

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

    /// Get tasks linked to a workspace milestone (with plan info)
    /// Merges tasks from direct links (INCLUDES_TASK) and
    /// plan-based links (TARGETS_MILESTONE → HAS_TASK)
    pub async fn get_workspace_milestone_tasks(
        &self,
        milestone_id: Uuid,
    ) -> Result<Vec<TaskWithPlan>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            OPTIONAL MATCH (wm)-[:INCLUDES_TASK]->(t1:Task)
            OPTIONAL MATCH (p:Plan)-[:TARGETS_MILESTONE]->(wm), (p)-[:HAS_TASK]->(t2:Task)
            WITH [x IN collect(DISTINCT t1) + collect(DISTINCT t2) WHERE x IS NOT NULL] AS tasks
            UNWIND tasks AS t
            OPTIONAL MATCH (pl:Plan)-[:HAS_TASK]->(t)
            RETURN t, COALESCE(pl.id, '') AS plan_id,
                   COALESCE(pl.title, '') AS plan_title,
                   COALESCE(pl.status, '') AS plan_status
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
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

    /// Get all steps for all tasks linked to a workspace milestone (batch query)
    /// Includes steps from both direct (INCLUDES_TASK) and
    /// plan-based (TARGETS_MILESTONE → HAS_TASK) task links
    pub async fn get_workspace_milestone_steps(
        &self,
        milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            OPTIONAL MATCH (wm)-[:INCLUDES_TASK]->(t1:Task)
            OPTIONAL MATCH (p:Plan)-[:TARGETS_MILESTONE]->(wm), (p)-[:HAS_TASK]->(t2:Task)
            WITH [x IN collect(DISTINCT t1) + collect(DISTINCT t2) WHERE x IS NOT NULL] AS tasks
            UNWIND tasks AS t
            MATCH (t)-[:HAS_STEP]->(s:Step)
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

    /// Helper to convert Neo4j node to WorkspaceMilestoneNode
    fn node_to_workspace_milestone(&self, node: &neo4rs::Node) -> Result<WorkspaceMilestoneNode> {
        let status_str: String = node.get("status").unwrap_or_else(|_| "Open".to_string());
        let status =
            serde_json::from_str::<MilestoneStatus>(&format!("\"{}\"", status_str.to_lowercase()))
                .unwrap_or(MilestoneStatus::Open);

        let tags: Vec<String> = node.get("tags").unwrap_or_else(|_| vec![]);

        Ok(WorkspaceMilestoneNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node.get::<String>("workspace_id")?.parse()?,
            title: node.get("title")?,
            description: node.get("description").ok(),
            status,
            target_date: node
                .get::<String>("target_date")
                .ok()
                .and_then(|s| s.parse().ok()),
            closed_at: node
                .get::<String>("closed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            tags,
        })
    }

    // ========================================================================
    // Resource operations
    // ========================================================================

    /// Create a resource
    pub async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (r:Resource {
                id: $id,
                workspace_id: $workspace_id,
                project_id: $project_id,
                name: $name,
                resource_type: $resource_type,
                file_path: $file_path,
                url: $url,
                format: $format,
                version: $version,
                description: $description,
                created_at: datetime($created_at),
                metadata: $metadata
            })
            "#,
        )
        .param("id", resource.id.to_string())
        .param(
            "workspace_id",
            resource
                .workspace_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        )
        .param(
            "project_id",
            resource
                .project_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        )
        .param("name", resource.name.clone())
        .param("resource_type", format!("{:?}", resource.resource_type))
        .param("file_path", resource.file_path.clone())
        .param("url", resource.url.clone().unwrap_or_default())
        .param("format", resource.format.clone().unwrap_or_default())
        .param("version", resource.version.clone().unwrap_or_default())
        .param(
            "description",
            resource.description.clone().unwrap_or_default(),
        )
        .param("created_at", resource.created_at.to_rfc3339())
        .param("metadata", resource.metadata.to_string());

        self.graph.run(q).await?;

        // Link to workspace if specified
        if let Some(workspace_id) = resource.workspace_id {
            let link_q = query(
                r#"
                MATCH (w:Workspace {id: $workspace_id})
                MATCH (r:Resource {id: $resource_id})
                MERGE (w)-[:HAS_RESOURCE]->(r)
                "#,
            )
            .param("workspace_id", workspace_id.to_string())
            .param("resource_id", resource.id.to_string());
            self.graph.run(link_q).await?;
        }

        // Link to project if specified
        if let Some(project_id) = resource.project_id {
            let link_q = query(
                r#"
                MATCH (p:Project {id: $project_id})
                MATCH (r:Resource {id: $resource_id})
                MERGE (p)-[:HAS_RESOURCE]->(r)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("resource_id", resource.id.to_string());
            self.graph.run(link_q).await?;
        }

        Ok(())
    }

    /// Get a resource by ID
    pub async fn get_resource(&self, id: Uuid) -> Result<Option<ResourceNode>> {
        let q = query(
            r#"
            MATCH (r:Resource {id: $id})
            RETURN r
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(self.node_to_resource(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List workspace resources
    pub async fn list_workspace_resources(&self, workspace_id: Uuid) -> Result<Vec<ResourceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_RESOURCE]->(r:Resource)
            RETURN r
            ORDER BY r.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut resources = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            resources.push(self.node_to_resource(&node)?);
        }

        Ok(resources)
    }

    /// Update a resource
    pub async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if name.is_some() {
            set_clauses.push("r.name = $name");
        }
        if file_path.is_some() {
            set_clauses.push("r.file_path = $file_path");
        }
        if url.is_some() {
            set_clauses.push("r.url = $url");
        }
        if version.is_some() {
            set_clauses.push("r.version = $version");
        }
        if description.is_some() {
            set_clauses.push("r.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (r:Resource {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(file_path) = file_path {
            q = q.param("file_path", file_path);
        }
        if let Some(url) = url {
            q = q.param("url", url);
        }
        if let Some(version) = version {
            q = q.param("version", version);
        }
        if let Some(description) = description {
            q = q.param("description", description);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a resource
    pub async fn delete_resource(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Resource {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a project as implementing a resource
    pub async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (r:Resource {id: $resource_id})
            MERGE (p)-[:IMPLEMENTS_RESOURCE]->(r)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("resource_id", resource_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a project as using a resource
    pub async fn link_project_uses_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (r:Resource {id: $resource_id})
            MERGE (p)-[:USES_RESOURCE]->(r)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("resource_id", resource_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get projects that implement a resource
    pub async fn get_resource_implementers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:IMPLEMENTS_RESOURCE]->(r:Resource {id: $resource_id})
            RETURN p
            "#,
        )
        .param("resource_id", resource_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Get projects that use a resource
    pub async fn get_resource_consumers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:USES_RESOURCE]->(r:Resource {id: $resource_id})
            RETURN p
            "#,
        )
        .param("resource_id", resource_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Helper to convert Neo4j node to ResourceNode
    fn node_to_resource(&self, node: &neo4rs::Node) -> Result<ResourceNode> {
        let type_str: String = node
            .get("resource_type")
            .unwrap_or_else(|_| "Other".to_string());
        let resource_type = match type_str.to_lowercase().as_str() {
            "apicontract" | "api_contract" => ResourceType::ApiContract,
            "protobuf" => ResourceType::Protobuf,
            "graphqlschema" | "graphql_schema" => ResourceType::GraphqlSchema,
            "jsonschema" | "json_schema" => ResourceType::JsonSchema,
            "databaseschema" | "database_schema" => ResourceType::DatabaseSchema,
            "sharedtypes" | "shared_types" => ResourceType::SharedTypes,
            "config" => ResourceType::Config,
            "documentation" => ResourceType::Documentation,
            _ => ResourceType::Other,
        };

        let metadata_str: String = node.get("metadata").unwrap_or_else(|_| "{}".to_string());
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_str).unwrap_or(serde_json::json!({}));

        Ok(ResourceNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node
                .get::<String>("workspace_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            project_id: node
                .get::<String>("project_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            name: node.get("name")?,
            resource_type,
            file_path: node.get("file_path")?,
            url: node.get("url").ok(),
            format: node.get("format").ok(),
            version: node.get("version").ok(),
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            metadata,
        })
    }

    // ========================================================================
    // Component operations (Topology)
    // ========================================================================

    /// Create a component
    pub async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            CREATE (c:Component {
                id: $id,
                workspace_id: $workspace_id,
                name: $name,
                component_type: $component_type,
                description: $description,
                runtime: $runtime,
                config: $config,
                created_at: datetime($created_at),
                tags: $tags
            })
            CREATE (w)-[:HAS_COMPONENT]->(c)
            "#,
        )
        .param("id", component.id.to_string())
        .param("workspace_id", component.workspace_id.to_string())
        .param("name", component.name.clone())
        .param("component_type", format!("{:?}", component.component_type))
        .param(
            "description",
            component.description.clone().unwrap_or_default(),
        )
        .param("runtime", component.runtime.clone().unwrap_or_default())
        .param("config", component.config.to_string())
        .param("created_at", component.created_at.to_rfc3339())
        .param("tags", component.tags.clone());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a component by ID
    pub async fn get_component(&self, id: Uuid) -> Result<Option<ComponentNode>> {
        let q = query(
            r#"
            MATCH (c:Component {id: $id})
            RETURN c
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(self.node_to_component(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List components in a workspace
    pub async fn list_components(&self, workspace_id: Uuid) -> Result<Vec<ComponentNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_COMPONENT]->(c:Component)
            RETURN c
            ORDER BY c.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut components = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            components.push(self.node_to_component(&node)?);
        }

        Ok(components)
    }

    /// Update a component
    pub async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if name.is_some() {
            set_clauses.push("c.name = $name");
        }
        if description.is_some() {
            set_clauses.push("c.description = $description");
        }
        if runtime.is_some() {
            set_clauses.push("c.runtime = $runtime");
        }
        if config.is_some() {
            set_clauses.push("c.config = $config");
        }
        if tags.is_some() {
            set_clauses.push("c.tags = $tags");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (c:Component {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(runtime) = runtime {
            q = q.param("runtime", runtime);
        }
        if let Some(config) = config {
            q = q.param("config", config.to_string());
        }
        if let Some(tags) = tags {
            q = q.param("tags", tags);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a component
    pub async fn delete_component(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Component {id: $id})
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a dependency between components
    pub async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c1:Component {id: $component_id})
            MATCH (c2:Component {id: $depends_on_id})
            MERGE (c1)-[r:DEPENDS_ON_COMPONENT]->(c2)
            SET r.protocol = $protocol, r.required = $required
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("depends_on_id", depends_on_id.to_string())
        .param("protocol", protocol.unwrap_or_default())
        .param("required", required);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a dependency between components
    pub async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c1:Component {id: $component_id})-[r:DEPENDS_ON_COMPONENT]->(c2:Component {id: $depends_on_id})
            DELETE r
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Map a component to a project
    pub async fn map_component_to_project(
        &self,
        component_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Component {id: $component_id})
            MATCH (p:Project {id: $project_id})
            MERGE (c)-[:MAPS_TO_PROJECT]->(p)
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get the workspace topology (all components with their dependencies)
    pub async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_COMPONENT]->(c:Component)
            OPTIONAL MATCH (c)-[:MAPS_TO_PROJECT]->(p:Project)
            OPTIONAL MATCH (c)-[d:DEPENDS_ON_COMPONENT]->(dep:Component)
            RETURN c, p.name AS project_name,
                   collect({dep_id: dep.id, protocol: d.protocol, required: d.required}) AS dependencies
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut topology = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            let component = self.node_to_component(&node)?;
            let project_name: Option<String> = row.get("project_name").ok();

            // Parse dependencies
            let deps_raw: Vec<serde_json::Value> =
                row.get("dependencies").unwrap_or_else(|_| vec![]);
            let mut dependencies = Vec::new();
            for dep in deps_raw {
                if let Some(dep_id_str) = dep.get("dep_id").and_then(|v| v.as_str()) {
                    if let Ok(dep_id) = dep_id_str.parse::<Uuid>() {
                        dependencies.push(ComponentDependency {
                            from_id: component.id,
                            to_id: dep_id,
                            protocol: dep
                                .get("protocol")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            required: dep
                                .get("required")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(true),
                        });
                    }
                }
            }

            topology.push((component, project_name, dependencies));
        }

        Ok(topology)
    }

    /// Helper to convert Neo4j node to ComponentNode
    fn node_to_component(&self, node: &neo4rs::Node) -> Result<ComponentNode> {
        let type_str: String = node
            .get("component_type")
            .unwrap_or_else(|_| "Other".to_string());
        let component_type = match type_str.to_lowercase().as_str() {
            "service" => ComponentType::Service,
            "frontend" => ComponentType::Frontend,
            "worker" => ComponentType::Worker,
            "database" => ComponentType::Database,
            "messagequeue" | "message_queue" => ComponentType::MessageQueue,
            "cache" => ComponentType::Cache,
            "gateway" => ComponentType::Gateway,
            "external" => ComponentType::External,
            _ => ComponentType::Other,
        };

        let config_str: String = node.get("config").unwrap_or_else(|_| "{}".to_string());
        let config: serde_json::Value =
            serde_json::from_str(&config_str).unwrap_or(serde_json::json!({}));

        let tags: Vec<String> = node.get("tags").unwrap_or_else(|_| vec![]);

        Ok(ComponentNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node.get::<String>("workspace_id")?.parse()?,
            name: node.get("name")?,
            component_type,
            description: node.get("description").ok(),
            runtime: node.get("runtime").ok(),
            config,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            tags,
        })
    }
}
