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

    // ========================================================================
    // TOUCHES operations (Commit → File)
    // ========================================================================

    /// Create TOUCHES relations between a Commit and the Files it modified.
    ///
    /// Uses UNWIND for batch efficiency. Files that don't exist as nodes
    /// are silently skipped (MATCH, not MERGE on File).
    /// Stores optional additions/deletions stats on the relationship.
    pub async fn create_commit_touches(
        &self,
        commit_hash: &str,
        files: &[FileChangedInfo],
    ) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = files
            .iter()
            .map(|f| {
                let mut m = std::collections::HashMap::new();
                m.insert("path".into(), f.path.clone().into());
                m.insert(
                    "additions".into(),
                    f.additions.unwrap_or(-1).into(),
                );
                m.insert(
                    "deletions".into(),
                    f.deletions.unwrap_or(-1).into(),
                );
                m
            })
            .collect();

        let total = items.len();

        // Single UNWIND query: MATCH on both Commit and File (silent skip for missing files)
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})
            UNWIND $items AS f
            MATCH (file:File {path: f.path})
            MERGE (c)-[r:TOUCHES]->(file)
            SET r.file_path = f.path
            WITH r, f
            WHERE f.additions >= 0
            SET r.additions = f.additions
            WITH r, f
            WHERE f.deletions >= 0
            SET r.deletions = f.deletions
            "#,
        )
        .param("hash", commit_hash)
        .param("items", items);

        self.graph.run(q).await?;

        tracing::debug!(
            commit = %commit_hash,
            files_provided = total,
            "Created TOUCHES relations"
        );

        Ok(())
    }

    /// Get all files touched by a commit (via TOUCHES relationships)
    pub async fn get_commit_files(
        &self,
        commit_hash: &str,
    ) -> Result<Vec<CommitFileInfo>> {
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})-[r:TOUCHES]->(f:File)
            RETURN f.path AS path,
                   r.additions AS additions,
                   r.deletions AS deletions
            ORDER BY f.path
            "#,
        )
        .param("hash", commit_hash);

        let mut result = self.graph.execute(q).await?;
        let mut files = Vec::new();

        while let Some(row) = result.next().await? {
            files.push(CommitFileInfo {
                path: row.get("path")?,
                additions: row.get::<Option<i64>>("additions").ok().flatten(),
                deletions: row.get::<Option<i64>>("deletions").ok().flatten(),
            });
        }

        Ok(files)
    }

    /// Get the commit history for a specific file (via TOUCHES relationships)
    pub async fn get_file_history(
        &self,
        file_path: &str,
        limit: Option<i64>,
    ) -> Result<Vec<FileHistoryEntry>> {
        let limit_val = limit.unwrap_or(50);
        let q = query(
            r#"
            MATCH (c:Commit)-[r:TOUCHES]->(f:File {path: $path})
            RETURN c.hash AS hash,
                   c.message AS message,
                   c.author AS author,
                   c.timestamp AS timestamp,
                   r.additions AS additions,
                   r.deletions AS deletions
            ORDER BY c.timestamp DESC
            LIMIT $limit
            "#,
        )
        .param("path", file_path)
        .param("limit", limit_val);

        let mut result = self.graph.execute(q).await?;
        let mut history = Vec::new();

        while let Some(row) = result.next().await? {
            history.push(FileHistoryEntry {
                hash: row.get("hash")?,
                message: row.get("message")?,
                author: row.get("author")?,
                timestamp: row
                    .get::<String>("timestamp")?
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
                additions: row.get::<Option<i64>>("additions").ok().flatten(),
                deletions: row.get::<Option<i64>>("deletions").ok().flatten(),
            });
        }

        Ok(history)
    }

    // ========================================================================
    // CO_CHANGED computation (File ↔ File derived from TOUCHES)
    // ========================================================================

    /// Compute CO_CHANGED relations between files that are frequently modified together.
    ///
    /// Algorithm (incremental):
    /// 1. Find commits since `since` that touch files in this project
    /// 2. For each commit with >1 file, generate all (f1, f2) pairs where f1.path < f2.path
    /// 3. Aggregate: count co-occurrences
    /// 4. Filter: count >= min_count
    /// 5. MERGE CO_CHANGED relations with {count, last_at, project_id}
    /// 6. Cap to max_relations per project (keep strongest)
    ///
    /// Returns the number of CO_CHANGED relations created/updated.
    pub async fn compute_co_changed(
        &self,
        project_id: Uuid,
        since: Option<chrono::DateTime<chrono::Utc>>,
        min_count: i64,
        max_relations: i64,
    ) -> Result<i64> {
        let since_str = since
            .unwrap_or_else(|| chrono::DateTime::UNIX_EPOCH)
            .to_rfc3339();

        // Phase 1: Compute co-changed pairs from TOUCHES and MERGE into CO_CHANGED
        let q = query(
            r#"
            MATCH (c:Commit)-[:TOUCHES]->(f:File {project_id: $project_id})
            WHERE c.timestamp > datetime($since)
            WITH c, collect(f) AS files
            WHERE size(files) > 1 AND size(files) <= 50
            UNWIND range(0, size(files)-2) AS i
            UNWIND range(i+1, size(files)-1) AS j
            WITH files[i] AS f1, files[j] AS f2
            WHERE f1.path < f2.path
            WITH f1, f2, count(*) AS co_count
            WHERE co_count >= $min_count
            MERGE (f1)-[r:CO_CHANGED]-(f2)
            SET r.count = coalesce(r.count, 0) + co_count,
                r.last_at = datetime(),
                r.project_id = $project_id
            RETURN count(r) AS total
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("since", since_str)
        .param("min_count", min_count);

        let mut result = self.graph.execute(q).await?;
        let total = if let Some(row) = result.next().await? {
            row.get::<i64>("total").unwrap_or(0)
        } else {
            0
        };

        // Phase 2: Cap to max_relations — delete weakest if over limit
        let cleanup = query(
            r#"
            MATCH (f1:File)-[r:CO_CHANGED]-(f2:File)
            WHERE r.project_id = $project_id AND f1.path < f2.path
            WITH r ORDER BY r.count DESC
            SKIP $max_relations
            DELETE r
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("max_relations", max_relations);

        if let Err(e) = self.graph.run(cleanup).await {
            tracing::warn!(
                project_id = %project_id,
                error = %e,
                "CO_CHANGED cap cleanup failed (non-fatal)"
            );
        }

        // Phase 3: Update timestamp
        self.update_project_co_change_timestamp(project_id).await?;

        tracing::info!(
            project_id = %project_id,
            relations_upserted = total,
            "CO_CHANGED computation complete"
        );

        Ok(total)
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
