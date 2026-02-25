//! Neo4j Commit operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Commit operations
    // ========================================================================

    /// Create a commit node
    pub async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        let q = query(
            r#"
            MERGE (c:Commit {hash: $hash})
            SET c.message = $message,
                c.author = $author,
                c.timestamp = datetime($timestamp)
            "#,
        )
        .param("hash", commit.hash.clone())
        .param("message", commit.message.clone())
        .param("author", commit.author.clone())
        .param("timestamp", commit.timestamp.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a commit by hash
    pub async fn get_commit(&self, hash: &str) -> Result<Option<CommitNode>> {
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})
            RETURN c
            "#,
        )
        .param("hash", hash);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(self.node_to_commit(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to CommitNode
    pub(crate) fn node_to_commit(&self, node: &neo4rs::Node) -> Result<CommitNode> {
        Ok(CommitNode {
            hash: node.get("hash")?,
            message: node.get("message")?,
            author: node.get("author")?,
            timestamp: node
                .get::<String>("timestamp")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }

    /// Link a commit to a task (RESOLVED_BY relationship)
    pub async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (t)-[:RESOLVED_BY]->(c)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a commit to a plan (RESULTED_IN relationship)
    pub async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (p)-[:RESULTED_IN]->(c)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get commits for a task
    pub async fn get_task_commits(&self, task_id: Uuid) -> Result<Vec<CommitNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:RESOLVED_BY]->(c:Commit)
            RETURN c
            ORDER BY c.timestamp DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut commits = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            commits.push(self.node_to_commit(&node)?);
        }

        Ok(commits)
    }

    /// Get commits for a plan
    pub async fn get_plan_commits(&self, plan_id: Uuid) -> Result<Vec<CommitNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:RESULTED_IN]->(c:Commit)
            RETURN c
            ORDER BY c.timestamp DESC
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut commits = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            commits.push(self.node_to_commit(&node)?);
        }

        Ok(commits)
    }

    /// Delete a commit
    pub async fn delete_commit(&self, hash: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})
            DETACH DELETE c
            "#,
        )
        .param("hash", hash);

        self.graph.run(q).await?;
        Ok(())
    }
}
