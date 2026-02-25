//! Neo4j Release operations

use super::client::{pascal_to_snake_case, Neo4jClient, WhereBuilder};
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Release operations
    // ========================================================================

    /// Create a release
    pub async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            CREATE (r:Release {
                id: $id,
                version: $version,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                released_at: $released_at,
                created_at: datetime($created_at),
                project_id: $project_id
            })
            CREATE (p)-[:HAS_RELEASE]->(r)
            "#,
        )
        .param("id", release.id.to_string())
        .param("version", release.version.clone())
        .param("title", release.title.clone().unwrap_or_default())
        .param(
            "description",
            release.description.clone().unwrap_or_default(),
        )
        .param("status", format!("{:?}", release.status))
        .param("project_id", release.project_id.to_string())
        .param(
            "target_date",
            release
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param(
            "released_at",
            release
                .released_at
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("created_at", release.created_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a release by ID
    pub async fn get_release(&self, id: Uuid) -> Result<Option<ReleaseNode>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            RETURN r
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(self.node_to_release(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to ReleaseNode
    fn node_to_release(&self, node: &neo4rs::Node) -> Result<ReleaseNode> {
        Ok(ReleaseNode {
            id: node.get::<String>("id")?.parse()?,
            version: node.get("version")?,
            title: node.get::<String>("title").ok().filter(|s| !s.is_empty()),
            description: node
                .get::<String>("description")
                .ok()
                .filter(|s| !s.is_empty()),
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(ReleaseStatus::Planned),
            target_date: node
                .get::<String>("target_date")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            released_at: node
                .get::<String>("released_at")
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

    /// List releases for a project
    pub async fn list_project_releases(&self, project_id: Uuid) -> Result<Vec<ReleaseNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_RELEASE]->(r:Release)
            RETURN r
            ORDER BY r.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut releases = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            releases.push(self.node_to_release(&node)?);
        }

        Ok(releases)
    }

    /// Update a release
    pub async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if status.is_some() {
            set_clauses.push("r.status = $status");
        }
        if target_date.is_some() {
            set_clauses.push("r.target_date = $target_date");
        }
        if released_at.is_some() {
            set_clauses.push("r.released_at = $released_at");
        }
        if title.is_some() {
            set_clauses.push("r.title = $title");
        }
        if description.is_some() {
            set_clauses.push("r.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (r:Release {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(ref s) = status {
            q = q.param("status", format!("{:?}", s));
        }
        if let Some(d) = target_date {
            q = q.param("target_date", d.to_rfc3339());
        }
        if let Some(d) = released_at {
            q = q.param("released_at", d.to_rfc3339());
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

    /// Add a task to a release
    pub async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})
            MATCH (t:Task {id: $task_id})
            MERGE (r)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("release_id", release_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a commit to a release
    pub async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (r)-[:INCLUDES_COMMIT]->(c)
            "#,
        )
        .param("release_id", release_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a commit from a release
    pub async fn remove_commit_from_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})-[rel:INCLUDES_COMMIT]->(c:Commit {hash: $hash})
            DELETE rel
            "#,
        )
        .param("release_id", release_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get release details with tasks and commits
    pub async fn get_release_details(
        &self,
        release_id: Uuid,
    ) -> Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            OPTIONAL MATCH (r)-[:INCLUDES_TASK]->(t:Task)
            OPTIONAL MATCH (r)-[:INCLUDES_COMMIT]->(c:Commit)
            RETURN r,
                   collect(DISTINCT t) AS tasks,
                   collect(DISTINCT c) AS commits
            "#,
        )
        .param("id", release_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let release_node: neo4rs::Node = row.get("r")?;
            let release = self.node_to_release(&release_node)?;

            let task_nodes: Vec<neo4rs::Node> = row.get("tasks").unwrap_or_default();
            let tasks: Vec<TaskNode> = task_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            let commit_nodes: Vec<neo4rs::Node> = row.get("commits").unwrap_or_default();
            let commits: Vec<CommitNode> = commit_nodes
                .iter()
                .filter_map(|n| self.node_to_commit(n).ok())
                .collect();

            Ok(Some((release, tasks, commits)))
        } else {
            Ok(None)
        }
    }

    /// Delete a release
    pub async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", release_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks for a release
    pub async fn get_release_tasks(&self, release_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("id", release_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// List project releases with filters and pagination
    pub async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ReleaseNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_status_filter("r", statuses);

        let where_clause = where_builder.build_and();
        let order_field = match sort_by {
            Some("version") => "r.version",
            Some("target_date") => "r.target_date",
            Some("released_at") => "r.released_at",
            Some("title") => "r.title",
            _ => "r.created_at",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!(
            "MATCH (p:Project {{id: $project_id}})-[:HAS_RELEASE]->(r:Release) {} RETURN count(r) AS total",
            where_clause
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
            MATCH (p:Project {{id: $project_id}})-[:HAS_RELEASE]->(r:Release)
            {}
            RETURN r
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
        let mut releases = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            releases.push(self.node_to_release(&node)?);
        }

        Ok((releases, total as usize))
    }
}
