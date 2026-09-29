//! Neo4j Environment & Deployment operations
//!
//! Graph shape:
//! `(Project)-[:HAS_ENVIRONMENT]->(Environment)-[:HAS_DEPLOYMENT]->(Deployment)`
//! plus an optional `(Deployment)-[:DEPLOYS]->(Commit)` when a commit node with the
//! deployed sha exists.
//!
//! Timestamps are persisted as fixed-width RFC 3339 strings (microsecond precision,
//! `Z` suffix) so that lexicographic `ORDER BY` is chronological. Optional strings
//! follow the repo convention of "empty string means absent".

use super::client::Neo4jClient;
use super::models::*;
use anyhow::{bail, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use neo4rs::query;
use uuid::Uuid;

/// Fixed-width, chronologically sortable timestamp representation.
pub(crate) fn ts(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Micros, true)
}

fn opt_str(node: &neo4rs::Node, key: &str) -> Option<String> {
    node.get::<String>(key).ok().filter(|s| !s.is_empty())
}

fn opt_ts(node: &neo4rs::Node, key: &str) -> Option<DateTime<Utc>> {
    opt_str(node, key).and_then(|s| s.parse().ok())
}

impl Neo4jClient {
    // ========================================================================
    // Environment operations
    // ========================================================================

    /// Create an environment. Fails if the project does not exist or if an
    /// environment with the same name already exists in the project.
    pub async fn create_environment(&self, env: &EnvironmentNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            WHERE NOT EXISTS {
                MATCH (p)-[:HAS_ENVIRONMENT]->(x:Environment {name: $name})
            }
            CREATE (e:Environment {
                id: $id,
                project_id: $project_id,
                name: $name,
                kind: $kind,
                url: $url,
                description: $description,
                config: $config,
                created_at: $created_at
            })
            CREATE (p)-[:HAS_ENVIRONMENT]->(e)
            RETURN e.id AS id
            "#,
        )
        .param("id", env.id.to_string())
        .param("project_id", env.project_id.to_string())
        .param("name", env.name.clone())
        .param("kind", env.kind.as_str())
        .param("url", env.url.clone().unwrap_or_default())
        .param("description", env.description.clone().unwrap_or_default())
        .param("config", env.config.clone().unwrap_or_default())
        .param("created_at", ts(env.created_at));

        let rows = self.execute_with_params(q).await?;
        if rows.is_empty() {
            bail!(
                "Cannot create environment '{}': project not found or name already used",
                env.name
            );
        }
        Ok(())
    }

    /// Convert a Neo4j node to an EnvironmentNode
    pub(crate) fn node_to_environment(&self, node: &neo4rs::Node) -> Result<EnvironmentNode> {
        Ok(EnvironmentNode {
            id: node.get::<String>("id")?.parse()?,
            project_id: node.get::<String>("project_id")?.parse()?,
            name: node.get("name")?,
            kind: node
                .get::<String>("kind")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(EnvironmentKind::Other),
            url: opt_str(node, "url"),
            description: opt_str(node, "description"),
            config: opt_str(node, "config"),
            created_at: opt_ts(node, "created_at").unwrap_or_else(Utc::now),
        })
    }

    /// Get an environment by ID
    pub async fn get_environment(&self, id: Uuid) -> Result<Option<EnvironmentNode>> {
        let q = query("MATCH (e:Environment {id: $id}) RETURN e").param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            Ok(Some(self.node_to_environment(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List the environments of a project (oldest first)
    pub async fn list_project_environments(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<EnvironmentNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_ENVIRONMENT]->(e:Environment)
            RETURN e
            ORDER BY e.created_at ASC, e.name ASC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut envs = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            envs.push(self.node_to_environment(&node)?);
        }
        Ok(envs)
    }

    /// Update an environment. An empty string clears `url`, `description` or `config`.
    pub async fn update_environment(
        &self,
        id: Uuid,
        name: Option<String>,
        kind: Option<EnvironmentKind>,
        url: Option<String>,
        description: Option<String>,
        config: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();
        if name.is_some() {
            set_clauses.push("e.name = $name");
        }
        if kind.is_some() {
            set_clauses.push("e.kind = $kind");
        }
        if url.is_some() {
            set_clauses.push("e.url = $url");
        }
        if description.is_some() {
            set_clauses.push("e.description = $description");
        }
        if config.is_some() {
            set_clauses.push("e.config = $config");
        }
        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (e:Environment {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );
        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(n) = name {
            q = q.param("name", n);
        }
        if let Some(k) = kind {
            q = q.param("kind", k.as_str());
        }
        if let Some(u) = url {
            q = q.param("url", u);
        }
        if let Some(d) = description {
            q = q.param("description", d);
        }
        if let Some(c) = config {
            q = q.param("config", c);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete an environment together with its deployments
    pub async fn delete_environment(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (e:Environment {id: $id})
            OPTIONAL MATCH (e)-[:HAS_DEPLOYMENT]->(d:Deployment)
            DETACH DELETE d, e
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Deployment operations
    // ========================================================================

    /// Record a deployment. Also links it to the `Commit` node with the same sha
    /// when one exists (silently skipped otherwise).
    pub async fn create_deployment(&self, dep: &DeploymentNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (e:Environment {id: $environment_id})
            CREATE (d:Deployment {
                id: $id,
                environment_id: $environment_id,
                version: $version,
                commit_sha: $commit_sha,
                status: $status,
                notes: $notes,
                created_by: $created_by,
                started_at: $started_at,
                finished_at: $finished_at
            })
            CREATE (e)-[:HAS_DEPLOYMENT]->(d)
            RETURN d.id AS id
            "#,
        )
        .param("id", dep.id.to_string())
        .param("environment_id", dep.environment_id.to_string())
        .param("version", dep.version.clone().unwrap_or_default())
        .param("commit_sha", dep.commit_sha.clone().unwrap_or_default())
        .param("status", dep.status.as_str())
        .param("notes", dep.notes.clone().unwrap_or_default())
        .param("created_by", dep.created_by.clone())
        .param("started_at", ts(dep.started_at))
        .param("finished_at", dep.finished_at.map(ts).unwrap_or_default());

        let rows = self.execute_with_params(q).await?;
        if rows.is_empty() {
            bail!(
                "Cannot create deployment: environment {} not found",
                dep.environment_id
            );
        }

        if let Some(sha) = dep.commit_sha.as_deref().filter(|s| !s.is_empty()) {
            let q = query(
                r#"
                MATCH (d:Deployment {id: $id})
                MATCH (c:Commit {hash: $hash})
                MERGE (d)-[:DEPLOYS]->(c)
                "#,
            )
            .param("id", dep.id.to_string())
            .param("hash", sha);
            self.graph.run(q).await?;
        }
        Ok(())
    }

    /// Convert a Neo4j node to a DeploymentNode
    pub(crate) fn node_to_deployment(&self, node: &neo4rs::Node) -> Result<DeploymentNode> {
        Ok(DeploymentNode {
            id: node.get::<String>("id")?.parse()?,
            environment_id: node.get::<String>("environment_id")?.parse()?,
            version: opt_str(node, "version"),
            commit_sha: opt_str(node, "commit_sha"),
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DeploymentStatus::Pending),
            notes: opt_str(node, "notes"),
            created_by: node.get::<String>("created_by").unwrap_or_default(),
            started_at: opt_ts(node, "started_at").unwrap_or_else(Utc::now),
            finished_at: opt_ts(node, "finished_at"),
        })
    }

    /// Get a deployment by ID
    pub async fn get_deployment(&self, id: Uuid) -> Result<Option<DeploymentNode>> {
        let q = query("MATCH (d:Deployment {id: $id}) RETURN d").param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            Ok(Some(self.node_to_deployment(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List the deployments of an environment, newest first, with the total count.
    pub async fn list_environment_deployments(
        &self,
        environment_id: Uuid,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DeploymentNode>, usize)> {
        let count_rows = self
            .execute_with_params(
                query(
                    "MATCH (:Environment {id: $id})-[:HAS_DEPLOYMENT]->(d:Deployment) \
                     RETURN count(d) AS total",
                )
                .param("id", environment_id.to_string()),
            )
            .await?;
        let total: i64 = count_rows
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        let cypher = format!(
            r#"
            MATCH (:Environment {{id: $id}})-[:HAS_DEPLOYMENT]->(d:Deployment)
            RETURN d
            ORDER BY d.started_at DESC, d.id DESC
            SKIP {}
            LIMIT {}
            "#,
            offset, limit
        );
        let mut result = self
            .graph
            .execute(query(&cypher).param("id", environment_id.to_string()))
            .await?;
        let mut deployments = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            deployments.push(self.node_to_deployment(&node)?);
        }
        Ok((deployments, total as usize))
    }

    /// Update a deployment (status, finished_at, notes)
    pub async fn update_deployment(
        &self,
        id: Uuid,
        status: Option<DeploymentStatus>,
        finished_at: Option<DateTime<Utc>>,
        notes: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();
        if status.is_some() {
            set_clauses.push("d.status = $status");
        }
        if finished_at.is_some() {
            set_clauses.push("d.finished_at = $finished_at");
        }
        if notes.is_some() {
            set_clauses.push("d.notes = $notes");
        }
        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (d:Deployment {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );
        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(s) = status {
            q = q.param("status", s.as_str());
        }
        if let Some(f) = finished_at {
            q = q.param("finished_at", ts(f));
        }
        if let Some(n) = notes {
            q = q.param("notes", n);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Deployment matrix of a project in a single aggregate query: every
    /// environment with its latest deployment and the statuses of its last
    /// [`DEPLOYMENT_MATRIX_RECENT`] deployments (newest first).
    pub async fn get_deployment_matrix(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<DeploymentMatrixEntry>> {
        let cypher = format!(
            r#"
            MATCH (p:Project {{id: $project_id}})-[:HAS_ENVIRONMENT]->(e:Environment)
            OPTIONAL MATCH (e)-[:HAS_DEPLOYMENT]->(d:Deployment)
            WITH e, d ORDER BY d.started_at DESC, d.id DESC
            WITH e, collect(d) AS deployments
            RETURN e, deployments[0..{}] AS recent
            ORDER BY e.created_at ASC, e.name ASC
            "#,
            DEPLOYMENT_MATRIX_RECENT
        );
        let rows = self
            .execute_with_params(query(&cypher).param("project_id", project_id.to_string()))
            .await?;

        let mut matrix = Vec::with_capacity(rows.len());
        for row in rows {
            let env_node: neo4rs::Node = row.get("e")?;
            let environment = self.node_to_environment(&env_node)?;
            let dep_nodes: Vec<neo4rs::Node> = row.get("recent").unwrap_or_default();
            let recent: Vec<DeploymentNode> = dep_nodes
                .iter()
                .filter_map(|n| self.node_to_deployment(n).ok())
                .collect();
            matrix.push(DeploymentMatrixEntry {
                environment,
                recent_statuses: recent.iter().map(|d| d.status).collect(),
                latest_deployment: recent.into_iter().next(),
            });
        }
        Ok(matrix)
    }
}
