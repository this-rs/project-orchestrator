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
                m.insert("additions".into(), f.additions.unwrap_or(-1).into());
                m.insert("deletions".into(), f.deletions.unwrap_or(-1).into());
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
    pub async fn get_commit_files(&self, commit_hash: &str) -> Result<Vec<CommitFileInfo>> {
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
        let since_str = since.unwrap_or(chrono::DateTime::UNIX_EPOCH).to_rfc3339();

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

    // ========================================================================
    // CO_CHANGED_TRANSITIVE computation (BFS on CO_CHANGED graph)
    // ========================================================================

    /// Compute CO_CHANGED_TRANSITIVE relations via BFS on the CO_CHANGED graph.
    ///
    /// Algorithm (depth-limited BFS, Rolfsnes et al. 2018):
    /// 1. For each file F with CO_CHANGED edges, traverse up to `max_depth` hops
    /// 2. Score = product of normalized edge weights (count/max_count) along the path
    /// 3. Filter: score >= `min_score` to avoid combinatorial explosion
    /// 4. MERGE CO_CHANGED_TRANSITIVE relations with {score, depth, via, project_id}
    /// 5. Skip pairs that already have a direct CO_CHANGED relation
    ///
    /// Performance: bounded by max_depth=2 (default) and min_score threshold,
    /// ensuring < 5s for projects with 500 files.
    ///
    /// # References
    /// - Rolfsnes et al. (2018) — "Detecting Evolutionary Coupling Using Transitive Association Rules"
    /// - Oliva & Gerosa (2015) — transitive co-change correlates with software defects
    ///
    /// Returns the number of CO_CHANGED_TRANSITIVE relations created/updated.
    pub async fn compute_co_changed_transitive(
        &self,
        project_id: Uuid,
        max_depth: i64,
        min_score: f64,
    ) -> Result<i64> {
        let max_depth = max_depth.clamp(1, 3);

        // Phase 1: Compute the max CO_CHANGED count for normalization
        let max_count_q = query(
            r#"
            MATCH ()-[r:CO_CHANGED]-()
            WHERE r.project_id = $project_id
            RETURN max(r.count) AS max_count
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut max_result = self.graph.execute(max_count_q).await?;
        let max_count: f64 = if let Some(row) = max_result.next().await? {
            row.get::<i64>("max_count").unwrap_or(1).max(1) as f64
        } else {
            return Ok(0);
        };

        // Phase 2: Delete existing transitive relations for this project (full recompute)
        let cleanup_q = query(
            r#"
            MATCH ()-[r:CO_CHANGED_TRANSITIVE]-()
            WHERE r.project_id = $project_id
            DELETE r
            "#,
        )
        .param("project_id", project_id.to_string());
        let _ = self.graph.run(cleanup_q).await;

        // Phase 3: BFS depth-limited transitive co-change computation
        // Uses variable-length path traversal over CO_CHANGED edges.
        // Score = product of (count/max_count) for each edge in the path.
        // Only creates relations where no direct CO_CHANGED exists.
        let bfs_q = query(
            r#"
            MATCH (f1:File {project_id: $project_id})-[path:CO_CHANGED*2..2]-(f2:File)
            WHERE f1 <> f2
              AND f1.path < f2.path
              AND NOT EXISTS { MATCH (f1)-[:CO_CHANGED]-(f2) }
            WITH f1, f2, path,
                 reduce(score = 1.0, r IN relationships(path) | score * (toFloat(r.count) / $max_count)) AS transit_score,
                 [n IN nodes(path)[1..-1] | n.path] AS via_nodes
            WHERE transit_score >= $min_score
            WITH f1, f2, max(transit_score) AS best_score,
                 head(collect(via_nodes)) AS best_via
            MERGE (f1)-[r:CO_CHANGED_TRANSITIVE]-(f2)
            SET r.score = best_score,
                r.depth = 2,
                r.via = best_via,
                r.project_id = $project_id,
                r.computed_at = datetime()
            RETURN count(r) AS total
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("max_count", max_count)
        .param("min_score", min_score);

        let mut result = self.graph.execute(bfs_q).await?;
        let mut total: i64 = if let Some(row) = result.next().await? {
            row.get::<i64>("total").unwrap_or(0)
        } else {
            0
        };

        // Phase 4: If max_depth >= 3, also compute depth-3 paths
        if max_depth >= 3 {
            let bfs3_q = query(
                r#"
                MATCH (f1:File {project_id: $project_id})-[path:CO_CHANGED*3..3]-(f2:File)
                WHERE f1 <> f2
                  AND f1.path < f2.path
                  AND NOT EXISTS { MATCH (f1)-[:CO_CHANGED]-(f2) }
                  AND NOT EXISTS { MATCH (f1)-[:CO_CHANGED_TRANSITIVE]-(f2) }
                WITH f1, f2, path,
                     reduce(score = 1.0, r IN relationships(path) | score * (toFloat(r.count) / $max_count)) AS transit_score,
                     [n IN nodes(path)[1..-1] | n.path] AS via_nodes
                WHERE transit_score >= $min_score
                WITH f1, f2, max(transit_score) AS best_score,
                     head(collect(via_nodes)) AS best_via
                MERGE (f1)-[r:CO_CHANGED_TRANSITIVE]-(f2)
                SET r.score = best_score,
                    r.depth = 3,
                    r.via = best_via,
                    r.project_id = $project_id,
                    r.computed_at = datetime()
                RETURN count(r) AS total
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("max_count", max_count)
            .param("min_score", min_score);

            if let Ok(mut r3) = self.graph.execute(bfs3_q).await {
                if let Ok(Some(row)) = r3.next().await {
                    total += row.get::<i64>("total").unwrap_or(0);
                }
            }
        }

        tracing::info!(
            project_id = %project_id,
            max_depth,
            min_score,
            relations_created = total,
            "CO_CHANGED_TRANSITIVE computation complete"
        );

        Ok(total)
    }

    /// Get files that are transitively co-changed with a given file.
    ///
    /// # References
    /// - Rolfsnes et al. (2018) — "Detecting Evolutionary Coupling Using Transitive Association Rules"
    pub async fn get_file_transitive_co_changers(
        &self,
        file_path: &str,
        min_score: f64,
        limit: i64,
    ) -> Result<Vec<TransitiveCoChanger>> {
        let q = query(
            r#"
            MATCH (f1:File {path: $path})-[r:CO_CHANGED_TRANSITIVE]-(f2:File)
            WHERE r.score >= $min_score
            RETURN f2.path AS path,
                   r.score AS score,
                   r.depth AS depth,
                   r.via AS via
            ORDER BY r.score DESC
            LIMIT $limit
            "#,
        )
        .param("path", file_path)
        .param("min_score", min_score)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut changers = Vec::new();

        while let Some(row) = result.next().await? {
            changers.push(TransitiveCoChanger {
                path: row.get("path")?,
                score: row.get("score").unwrap_or(0.0),
                depth: row.get("depth").unwrap_or(2),
                via: row.get::<Vec<String>>("via").unwrap_or_default(),
            });
        }

        Ok(changers)
    }

    // ========================================================================
    // CO_CHANGED query operations
    // ========================================================================

    /// Get the co-change graph for a project: all CO_CHANGED pairs sorted by count desc.
    pub async fn get_co_change_graph(
        &self,
        project_id: Uuid,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChangePair>> {
        let q = query(
            r#"
            MATCH (f1:File)-[r:CO_CHANGED]-(f2:File)
            WHERE r.project_id = $project_id
              AND r.count >= $min_count
              AND f1.path < f2.path
            RETURN f1.path AS file_a,
                   f2.path AS file_b,
                   r.count AS count,
                   toString(r.last_at) AS last_at
            ORDER BY r.count DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("min_count", min_count)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut pairs = Vec::new();

        while let Some(row) = result.next().await? {
            pairs.push(CoChangePair {
                file_a: row.get("file_a")?,
                file_b: row.get("file_b")?,
                count: row.get("count")?,
                last_at: row.get::<String>("last_at").ok(),
            });
        }

        Ok(pairs)
    }

    /// Get files that co-change with a given file (bidirectional).
    pub async fn get_file_co_changers(
        &self,
        file_path: &str,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChanger>> {
        let q = query(
            r#"
            MATCH (f1:File {path: $path})-[r:CO_CHANGED]-(f2:File)
            WHERE r.count >= $min_count
            RETURN f2.path AS path,
                   r.count AS count,
                   toString(r.last_at) AS last_at
            ORDER BY r.count DESC
            LIMIT $limit
            "#,
        )
        .param("path", file_path)
        .param("min_count", min_count)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut changers = Vec::new();

        while let Some(row) = result.next().await? {
            changers.push(CoChanger {
                path: row.get("path")?,
                count: row.get("count")?,
                last_at: row.get::<String>("last_at").ok(),
            });
        }

        Ok(changers)
    }

    /// Ping freshness on all Notes LINKED_TO the given file paths.
    ///
    /// Sets `freshness_pinged_at = datetime()` on each note that has a
    /// LINKED_TO relationship to a File whose path is in `file_paths`.
    /// Returns the number of distinct notes pinged.
    pub async fn ping_freshness_for_files(&self, file_paths: &[String]) -> Result<usize> {
        if file_paths.is_empty() {
            return Ok(0);
        }

        let paths: Vec<String> = file_paths.to_vec();

        let q = query(
            r#"
            UNWIND $paths AS path
            MATCH (n:Note)-[:LINKED_TO]->(f:File {path: path})
            SET n.freshness_pinged_at = datetime()
            RETURN count(DISTINCT n) AS pinged
            "#,
        )
        .param("paths", paths);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let pinged: i64 = row.get("pinged")?;
            Ok(pinged as usize)
        } else {
            Ok(0)
        }
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
