//! Neo4j Project operations

use super::client::{Neo4jClient, WhereBuilder};
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Project operations
    // ========================================================================

    /// Create a new project
    pub async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Project {
                id: $id,
                name: $name,
                slug: $slug,
                root_path: $root_path,
                description: $description,
                created_at: datetime($created_at),
                watch_enabled: $watch_enabled
            })
            "#,
        )
        .param("id", project.id.to_string())
        .param("name", project.name.clone())
        .param("slug", project.slug.clone())
        .param("root_path", project.root_path.clone())
        .param(
            "description",
            project.description.clone().unwrap_or_default(),
        )
        .param("created_at", project.created_at.to_rfc3339())
        .param("watch_enabled", project.watch_enabled);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a project by ID
    pub async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            RETURN p
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_project(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a project by slug
    pub async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project {slug: $slug})
            RETURN p
            "#,
        )
        .param("slug", slug);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_project(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all projects
    pub async fn list_projects(&self) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)
            RETURN p
            ORDER BY p.name
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Update project fields (name, description, root_path)
    pub async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];

        if name.is_some() {
            set_clauses.push("p.name = $name");
        }
        if description.is_some() {
            set_clauses.push("p.description = $description");
        }
        if root_path.is_some() {
            set_clauses.push("p.root_path = $root_path");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (p:Project {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(desc) = description {
            q = q.param("description", desc.unwrap_or_default());
        }
        if let Some(root_path) = root_path {
            q = q.param("root_path", root_path);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update project last_synced timestamp
    pub async fn update_project_synced(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.last_synced = datetime($now)
            "#,
        )
        .param("id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update project analytics_computed_at timestamp
    pub async fn update_project_analytics_timestamp(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.analytics_computed_at = datetime($now)
            "#,
        )
        .param("id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update project last_co_change_computed_at timestamp
    pub async fn update_project_co_change_timestamp(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.last_co_change_computed_at = datetime($now)
            "#,
        )
        .param("id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Set or clear the scaffolding level override on a project (biomimicry T8).
    /// Pass None to clear the override and return to auto-computed level.
    pub async fn set_scaffolding_override(&self, id: Uuid, level: Option<u8>) -> Result<()> {
        let q = if let Some(lvl) = level {
            query("MATCH (p:Project {id: $id}) SET p.scaffolding_override = $level")
                .param("id", id.to_string())
                .param("level", lvl.min(4) as i64)
        } else {
            query("MATCH (p:Project {id: $id}) REMOVE p.scaffolding_override")
                .param("id", id.to_string())
        };
        self.graph.run(q).await?;
        Ok(())
    }

    /// Set the watch_enabled flag on a project.
    ///
    /// When false, the file watcher will not auto-start for this project at boot.
    pub async fn set_watch_enabled(&self, id: Uuid, enabled: bool) -> Result<()> {
        let q = query("MATCH (p:Project {id: $id}) SET p.watch_enabled = $enabled")
            .param("id", id.to_string())
            .param("enabled", enabled);
        self.graph.run(q).await?;
        Ok(())
    }

    /// Set or clear default_note_energy on a project (homeostasis).
    pub async fn set_default_note_energy(&self, id: Uuid, energy: Option<f64>) -> Result<()> {
        let q = if let Some(e) = energy {
            query("MATCH (p:Project {id: $id}) SET p.default_note_energy = $energy")
                .param("id", id.to_string())
                .param("energy", e.clamp(0.0, 1.0))
        } else {
            query("MATCH (p:Project {id: $id}) REMOVE p.default_note_energy")
                .param("id", id.to_string())
        };
        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a project and all its data
    pub async fn delete_project(&self, id: Uuid, project_name: &str) -> Result<()> {
        // ================================================================
        // Phase 1: Archive knowledge (Notes + Decisions) — preserve, don't delete
        // ================================================================

        // Archive notes: detach from project + files, set status=archived
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[r:HAS_NOTE]->(n:Note)
            // Remove LINKED_TO relationships to files/symbols of THIS project
            OPTIONAL MATCH (n)-[lt:LINKED_TO]->(target)
                WHERE (target:File OR target:Function OR target:Struct OR target:Trait OR target:Enum)
                AND EXISTS { MATCH (target)<-[:CONTAINS*1..2]-(p) }
            DELETE lt, r
            SET n.status = 'archived',
                n.project_id = null,
                n.archived_from_project = $project_name,
                n.archived_at = datetime()
            "#,
        )
        .param("id", id.to_string())
        .param("project_name", project_name);
        self.graph.run(q).await?;

        // Archive decisions: detach from tasks + files, set status=archived
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_PLAN]->(plan:Plan)-[:HAS_TASK]->(task:Task)-[r_inf:INFORMED_BY]->(d:Decision)
            // Remove AFFECTS relationships to files of THIS project
            OPTIONAL MATCH (d)-[aff:AFFECTS]->(target)
                WHERE (target:File OR target:Function OR target:Struct)
                AND EXISTS { MATCH (target)<-[:CONTAINS*1..2]-(p) }
            DELETE aff, r_inf
            SET d.status = 'archived',
                d.archived_from_project = $project_name,
                d.archived_at = datetime()
            "#,
        )
        .param("id", id.to_string())
        .param("project_name", project_name);
        self.graph.run(q).await?;

        // ================================================================
        // Phase 2: Cascade delete structural entities (leaf-first order)
        // ================================================================

        // Delete steps (must come before tasks)
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_PLAN]->(plan:Plan)-[:HAS_TASK]->(task:Task)-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete constraints (must come before plans)
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_PLAN]->(plan:Plan)-[:CONSTRAINED_BY]->(c:Constraint)
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete plans + tasks (decisions already detached in phase 1)
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_PLAN]->(plan:Plan)
            OPTIONAL MATCH (plan)-[:HAS_TASK]->(task:Task)
            DETACH DELETE task, plan
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete milestones
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_MILESTONE]->(m:Milestone)
            DETACH DELETE m
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete releases
        let q = query(
            r#"
            MATCH (p:Project {id: $id})-[:HAS_RELEASE]->(r:Release)
            DETACH DELETE r
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete feature graphs
        let q = query(
            r#"
            MATCH (fg:FeatureGraph {project_id: $id})
            DETACH DELETE fg
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete commits linked to this project's plans/tasks
        let q = query(
            r#"
            MATCH (c:Commit {project_id: $id})
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // ================================================================
        // Phase 3: Delete code structure (files + symbols)
        // ================================================================

        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            OPTIONAL MATCH (p)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            DETACH DELETE symbol, f
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // ================================================================
        // Phase 4: Delete the project node itself
        // ================================================================

        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            DETACH DELETE p
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Helper to convert Neo4j node to ProjectNode
    pub(crate) fn node_to_project(&self, node: &neo4rs::Node) -> Result<ProjectNode> {
        Ok(ProjectNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            slug: node.get("slug")?,
            root_path: node.get("root_path")?,
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            last_synced: node
                .get::<String>("last_synced")
                .ok()
                .and_then(|s| s.parse().ok()),
            analytics_computed_at: node
                .get::<String>("analytics_computed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            last_co_change_computed_at: node
                .get::<String>("last_co_change_computed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            default_note_energy: node
                .get::<f64>("default_note_energy")
                .ok()
                .filter(|&v| v < 1.0), // None means default (1.0)
            scaffolding_override: node
                .get::<i64>("scaffolding_override")
                .ok()
                .map(|v| v as u8),
            sharing_policy: node
                .get::<String>("sharing_policy")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok()),
            watch_enabled: node.get::<bool>("watch_enabled").unwrap_or(true),
        })
    }

    /// Get project progress stats
    pub async fn get_project_progress(&self, project_id: Uuid) -> Result<(u32, u32, u32, u32)> {
        // Count tasks across all plans for this project
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)
            RETURN count(t) AS total,
                   sum(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                   sum(CASE WHEN t.status = 'InProgress' THEN 1 ELSE 0 END) AS in_progress,
                   sum(CASE WHEN t.status = 'Blocked' THEN 1 ELSE 0 END) AS blocked
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total")?;
            let completed: i64 = row.get("completed")?;
            let in_progress: i64 = row.get("in_progress")?;
            let blocked: i64 = row.get("blocked")?;
            Ok((
                total as u32,
                completed as u32,
                in_progress as u32,
                blocked as u32,
            ))
        } else {
            Ok((0, 0, 0, 0))
        }
    }

    /// Get project task dependencies (all cross-plan deps)
    pub async fn get_project_task_dependencies(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, Uuid)>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)-[:DEPENDS_ON]->(dep:Task)<-[:HAS_TASK]-(p2:Plan)<-[:HAS_PLAN]-(project)
            RETURN t.id AS from_id, dep.id AS to_id
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id")?;
            let to_id: String = row.get("to_id")?;
            if let (Ok(from), Ok(to)) = (from_id.parse::<Uuid>(), to_id.parse::<Uuid>()) {
                edges.push((from, to));
            }
        }

        Ok(edges)
    }

    /// Get all tasks for a project (across all plans)
    pub async fn get_project_tasks(&self, project_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// List projects with search and pagination
    pub async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ProjectNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_search_filter("p", search);

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("created_at") => "p.created_at",
            Some("last_synced") => "p.last_synced",
            _ => "p.name",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!(
            "MATCH (p:Project) {} RETURN count(p) AS total",
            where_clause
        );
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Project)
            {}
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut projects = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok((projects, total as usize))
    }
}
