//! Neo4j Plan operations

use super::client::{pascal_to_snake_case, Neo4jClient, WhereBuilder};
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    pub async fn create_plan(&self, plan: &PlanNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Plan {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                created_at: datetime($created_at),
                created_by: $created_by,
                priority: $priority,
                project_id: $project_id
            })
            "#,
        )
        .param("id", plan.id.to_string())
        .param("title", plan.title.clone())
        .param("description", plan.description.clone())
        .param("status", format!("{:?}", plan.status))
        .param("created_at", plan.created_at.to_rfc3339())
        .param("created_by", plan.created_by.clone())
        .param("priority", plan.priority as i64)
        .param(
            "project_id",
            plan.project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project if specified
        if let Some(project_id) = plan.project_id {
            let q = query(
                r#"
                MATCH (project:Project {id: $project_id})
                MATCH (plan:Plan {id: $plan_id})
                MERGE (project)-[:HAS_PLAN]->(plan)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("plan_id", plan.id.to_string());

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get a plan by ID
    pub async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            RETURN p
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_plan(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to PlanNode
    pub(crate) fn node_to_plan(&self, node: &neo4rs::Node) -> Result<PlanNode> {
        Ok(PlanNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get("title")?,
            description: node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(PlanStatus::Draft),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            created_by: node.get("created_by")?,
            priority: node.get::<i64>("priority")? as i32,
            project_id: node.get::<String>("project_id").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse().ok()
                }
            }),
        })
    }

    /// List all active plans
    pub async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// List active plans for a specific project
    pub async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// Count plans for a project (lightweight COUNT query, no data transfer).
    pub async fn count_project_plans(&self, project_id: Uuid) -> Result<i64> {
        let q = query(
            "MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan) WHERE p.status IN ['Draft', 'Approved', 'InProgress'] RETURN count(p) AS cnt",
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<i64>("cnt")?)
        } else {
            Ok(0)
        }
    }

    /// List plans for a project with filters
    pub async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)> {
        // Build status filter
        let status_clause = if let Some(statuses) = &status_filter {
            if !statuses.is_empty() {
                let status_list: Vec<String> = statuses
                    .iter()
                    .map(|s| {
                        // Convert to PascalCase for enum matching
                        let pascal = match s.to_lowercase().as_str() {
                            "draft" => "Draft",
                            "approved" => "Approved",
                            "in_progress" => "InProgress",
                            "completed" => "Completed",
                            "cancelled" => "Cancelled",
                            _ => s.as_str(),
                        };
                        format!("'{}'", pascal)
                    })
                    .collect();
                format!("AND p.status IN [{}]", status_list.join(", "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Count total
        let count_q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN count(p) AS total
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string());

        let count_rows = self.execute_with_params(count_q).await?;
        let total: i64 = count_rows
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Get plans
        let q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            SKIP $offset
            LIMIT $limit
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string())
        .param("offset", offset as i64)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
    }

    /// Update plan status
    pub async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            SET p.status = $status
            "#,
        )
        .param("id", id.to_string())
        .param("status", format!("{:?}", status));

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a plan to a project (creates HAS_PLAN relationship)
    pub async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})
            MATCH (plan:Plan {id: $plan_id})
            SET plan.project_id = $project_id
            MERGE (project)-[:HAS_PLAN]->(plan)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("plan_id", plan_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from its project
    pub async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project)-[r:HAS_PLAN]->(plan:Plan {id: $plan_id})
            DELETE r
            SET plan.project_id = null
            "#,
        )
        .param("plan_id", plan_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a plan and all its related data (tasks, steps, decisions, constraints)
    pub async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        // Delete all steps belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all decisions belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:INFORMED_BY]->(d:Decision)
            DETACH DELETE d
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all tasks belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)
            DETACH DELETE t
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all constraints belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:CONSTRAINED_BY]->(c:Constraint)
            DETACH DELETE c
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete the plan itself
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            DETACH DELETE p
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Get dependency graph for a plan (all tasks and their dependencies)
    pub async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        // Get all tasks in the plan
        let tasks = self.get_plan_tasks(plan_id).await?;

        // Get all DEPENDS_ON edges between tasks in this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task)-[:DEPENDS_ON]->(dep:Task)<-[:HAS_TASK]-(p)
            RETURN t.id AS from_id, dep.id AS to_id
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id")?;
            let to_id: String = row.get("to_id")?;
            if let (Ok(from), Ok(to)) = (from_id.parse::<Uuid>(), to_id.parse::<Uuid>()) {
                edges.push((from, to));
            }
        }

        Ok((tasks, edges))
    }

    /// Find critical path in a plan (longest chain of dependencies)
    pub async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        // Get all paths from tasks with no incoming deps to tasks with no outgoing deps
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(start:Task)
            WHERE NOT EXISTS { MATCH (start)-[:DEPENDS_ON]->(:Task) }
            MATCH (p)-[:HAS_TASK]->(end:Task)
            WHERE NOT EXISTS { MATCH (:Task)-[:DEPENDS_ON]->(end) }
            MATCH path = (start)<-[:DEPENDS_ON*0..]-(end)
            WHERE ALL(node IN nodes(path) WHERE (p)-[:HAS_TASK]->(node))
            WITH path, length(path) AS pathLength
            ORDER BY pathLength DESC
            LIMIT 1
            UNWIND nodes(path) AS task
            RETURN DISTINCT task
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("task")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// List plans with filters and pagination
    ///
    /// Returns (plans, total_count)
    #[allow(clippy::too_many_arguments)]
    pub async fn list_plans_filtered(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder
            .add_status_filter("p", statuses)
            .add_priority_filter("p", priority_min, priority_max)
            .add_search_filter("p", search);

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("priority") => "COALESCE(p.priority, 0)",
            Some("title") => "p.title",
            Some("status") => "p.status",
            _ => "p.created_at",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        let match_clause = if let Some(pid) = project_id {
            format!(
                "MATCH (proj:Project {{id: '{}'}})-[:HAS_PLAN]->(p:Plan)",
                pid
            )
        } else if let Some(ws) = workspace_slug {
            format!(
                "MATCH (w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)-[:HAS_PLAN]->(p:Plan)",
                ws
            )
        } else {
            "MATCH (p:Plan)".to_string()
        };

        // Count query
        let count_cypher = format!("{} {} RETURN count(p) AS total", match_clause, where_clause);
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
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            match_clause, where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut plans = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
    }
}
