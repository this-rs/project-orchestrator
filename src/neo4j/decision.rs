//! Neo4j Decision operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Decision operations
    // ========================================================================

    /// Deserialize a DecisionNode from a neo4rs::Node.
    pub(crate) fn node_to_decision(node: &neo4rs::Node) -> Result<DecisionNode> {
        Ok(DecisionNode {
            id: node.get::<String>("id")?.parse()?,
            description: node.get("description")?,
            rationale: node.get("rationale")?,
            alternatives: node.get::<Vec<String>>("alternatives").unwrap_or_default(),
            chosen_option: node
                .get::<String>("chosen_option")
                .ok()
                .filter(|s| !s.is_empty()),
            decided_by: node.get::<String>("decided_by").ok().unwrap_or_default(),
            decided_at: node
                .get::<String>("decided_at")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(chrono::Utc::now),
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DecisionStatus::Accepted),
            // Embeddings are not loaded by default (too large for list queries).
            // Use get_decision_embedding() for explicit retrieval.
            embedding: None,
            embedding_model: node.get::<String>("embedding_model").ok(),
            scar_intensity: node.get("scar_intensity").unwrap_or(0.0),
        })
    }

    /// Record a decision
    pub async fn create_decision(&self, task_id: Uuid, decision: &DecisionNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            CREATE (d:Decision {
                id: $id,
                description: $description,
                rationale: $rationale,
                alternatives: $alternatives,
                chosen_option: $chosen_option,
                decided_by: $decided_by,
                decided_at: datetime($decided_at),
                status: $status
            })
            CREATE (t)-[:INFORMED_BY]->(d)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("id", decision.id.to_string())
        .param("description", decision.description.clone())
        .param("rationale", decision.rationale.clone())
        .param("alternatives", decision.alternatives.clone())
        .param(
            "chosen_option",
            decision.chosen_option.clone().unwrap_or_default(),
        )
        .param("decided_by", decision.decided_by.clone())
        .param("decided_at", decision.decided_at.to_rfc3339())
        .param("status", decision.status.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a single decision by ID
    pub async fn get_decision(&self, decision_id: Uuid) -> Result<Option<DecisionNode>> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            RETURN d
            "#,
        )
        .param("id", decision_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            Ok(Some(Self::node_to_decision(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a decision
    pub async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
        status: Option<DecisionStatus>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if description.is_some() {
            set_clauses.push("d.description = $description");
        }
        if rationale.is_some() {
            set_clauses.push("d.rationale = $rationale");
        }
        if chosen_option.is_some() {
            set_clauses.push("d.chosen_option = $chosen_option");
        }
        if status.is_some() {
            set_clauses.push("d.status = $status");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (d:Decision {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", decision_id.to_string());
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(rationale) = rationale {
            q = q.param("rationale", rationale);
        }
        if let Some(chosen_option) = chosen_option {
            q = q.param("chosen_option", chosen_option);
        }
        if let Some(status) = status {
            q = q.param("status", status.to_string());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a decision
    pub async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            DETACH DELETE d
            "#,
        )
        .param("id", decision_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get decisions related to an entity.
    ///
    /// Finds decisions connected via:
    /// 1. Direct (Decision)-[:AFFECTS]->(entity) — if AFFECTS relations exist (P3)
    /// 2. Indirect via task linkage — (Task)-[:INFORMED_BY]->(Decision)
    ///    where the task's affected_files mention the entity path
    ///
    /// Returns up to `limit` decisions ordered by decided_at DESC.
    pub async fn get_decisions_for_entity(
        &self,
        entity_type: &str,
        entity_id: &str,
        limit: u32,
    ) -> Result<Vec<DecisionNode>> {
        // Try both direct AFFECTS and indirect via task affected_files.
        // For File entities with relative paths, use ENDS WITH matching
        // since Neo4j stores absolute paths.
        let is_relative_file = entity_type == "File" && !entity_id.starts_with('/');
        let match_field = if entity_type == "File" { "path" } else { "id" };

        let (target_clause, match_value) = if is_relative_file {
            let suffix = format!("/{}", entity_id);
            (
                format!(
                    "OPTIONAL MATCH (d1:Decision)-[:AFFECTS]->(target:{})\n            WHERE target.{} ENDS WITH $entity_id",
                    entity_type, match_field
                ),
                suffix,
            )
        } else {
            (
                format!(
                    "OPTIONAL MATCH (d1:Decision)-[:AFFECTS]->(target:{} {{{}: $entity_id}})",
                    entity_type, match_field
                ),
                entity_id.to_string(),
            )
        };

        // For affected_files (Path 2), also handle relative paths:
        // affected_files typically stores relative paths, so check both
        // exact match and suffix match to cover all cases.
        let affected_files_clause = if is_relative_file {
            "WHERE $entity_id_raw IN t.affected_files\n               OR ANY(af IN t.affected_files WHERE af ENDS WITH $entity_id)"
        } else {
            "WHERE $entity_id IN t.affected_files"
        };

        let cypher = format!(
            r#"
            // Path 1: Direct AFFECTS relation (P3 — may not exist yet)
            {}
            WITH collect(DISTINCT d1) AS direct_decisions

            // Path 2: Indirect via Task affected_files containing entity_id
            OPTIONAL MATCH (t:Task)-[:INFORMED_BY]->(d2:Decision)
            {}
            WITH direct_decisions, collect(DISTINCT d2) AS indirect_decisions

            // Merge and deduplicate
            WITH [d IN direct_decisions + indirect_decisions WHERE d IS NOT NULL] AS all_decisions
            UNWIND all_decisions AS d
            WITH DISTINCT d
            RETURN d
            ORDER BY d.decided_at DESC
            LIMIT $limit
            "#,
            target_clause, affected_files_clause,
        );

        let mut q = query(&cypher)
            .param("entity_id", match_value)
            .param("limit", limit as i64);

        // Add raw entity_id param for affected_files matching with relative paths
        if is_relative_file {
            q = q.param("entity_id_raw", entity_id.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut decisions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            decisions.push(Self::node_to_decision(&node)?);
        }

        Ok(decisions)
    }

    /// Store a vector embedding on a Decision node.
    ///
    /// Uses `db.create.setNodeVectorProperty` to ensure the correct type
    /// for the HNSW vector index. Also stores the model name for traceability.
    pub async fn set_decision_embedding(
        &self,
        decision_id: Uuid,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            CALL db.create.setNodeVectorProperty(d, 'embedding', $embedding)
            SET d.embedding_model = $model,
                d.embedded_at = datetime()
            "#,
        )
        .param("id", decision_id.to_string())
        .param("embedding", embedding_f64)
        .param("model", model.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Retrieve the stored vector embedding for a Decision node.
    ///
    /// Returns None if the decision has no embedding (not yet embedded).
    pub async fn get_decision_embedding(&self, decision_id: Uuid) -> Result<Option<Vec<f32>>> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            WHERE d.embedding IS NOT NULL
            RETURN d.embedding AS embedding
            "#,
        )
        .param("id", decision_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let embedding_f64: Vec<f64> = row.get("embedding")?;
            let embedding_f32: Vec<f32> = embedding_f64.iter().map(|&x| x as f32).collect();
            Ok(Some(embedding_f32))
        } else {
            Ok(None)
        }
    }

    /// Get all decisions with their linked task_id.
    /// Used by the reindex command to rebuild MeiliSearch from Neo4j.
    pub async fn get_all_decisions_with_task_id(&self) -> Result<Vec<(DecisionNode, Uuid)>> {
        let q = query(
            r#"
            MATCH (t:Task)-[:INFORMED_BY]->(d:Decision)
            RETURN d, t.id AS task_id
            ORDER BY d.decided_at DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut decisions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            let task_id_str: String = row.get("task_id")?;
            if let (Ok(decision), Ok(task_id)) =
                (Self::node_to_decision(&node), task_id_str.parse::<Uuid>())
            {
                decisions.push((decision, task_id));
            }
        }

        Ok(decisions)
    }

    /// Get all Decision IDs that have no embedding yet.
    /// Used by the backfill command.
    pub async fn get_decisions_without_embedding(&self) -> Result<Vec<(Uuid, String, String)>> {
        let q = query(
            r#"
            MATCH (d:Decision)
            WHERE d.embedding IS NULL
            RETURN d.id AS id, d.description AS description, d.rationale AS rationale
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut decisions = Vec::new();

        while let Some(row) = result.next().await? {
            let id: String = row.get("id")?;
            let description: String = row.get::<String>("description").unwrap_or_default();
            let rationale: String = row.get::<String>("rationale").unwrap_or_default();
            if let Ok(uuid) = id.parse::<Uuid>() {
                decisions.push((uuid, description, rationale));
            }
        }

        Ok(decisions)
    }

    /// Semantic search over Decision embeddings using Neo4j vector index.
    ///
    /// Returns decisions ordered by cosine similarity to the query embedding.
    /// When `project_id` is provided, fetches `limit * 3` results from the global
    /// vector index and filters post-query (the index is global, not project-scoped).
    pub async fn search_decisions_by_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_id: Option<&str>,
    ) -> Result<Vec<(DecisionNode, f64)>> {
        let embedding_f64: Vec<f64> = query_embedding.iter().map(|&x| x as f64).collect();

        // When filtering by project, overfetch x3 then filter post-query
        let fetch_limit = if project_id.is_some() {
            limit * 3
        } else {
            limit
        };

        let cypher = if project_id.is_some() {
            r#"
            CALL db.index.vector.queryNodes('decision_embedding', $fetch_limit, $embedding)
            YIELD node AS d, score
            // Filter by project: Decision is linked via Task→Plan→Project
            MATCH (t:Task)-[:INFORMED_BY]->(d)
            MATCH (p:Plan {id: t.plan_id})-[:BELONGS_TO]->(proj:Project {id: $project_id})
            RETURN d, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        } else {
            r#"
            CALL db.index.vector.queryNodes('decision_embedding', $fetch_limit, $embedding)
            YIELD node AS d, score
            RETURN d, score
            ORDER BY score DESC
            "#
        };

        let mut q = query(cypher)
            .param("fetch_limit", fetch_limit as i64)
            .param("limit", limit as i64)
            .param("embedding", embedding_f64);

        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut results = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            let score: f64 = row.get("score")?;
            results.push((Self::node_to_decision(&node)?, score));
        }

        Ok(results)
    }

    /// Get decisions that AFFECT a given entity (reverse AFFECTS lookup).
    ///
    /// Finds decisions connected via `(Decision)-[:AFFECTS]->(entity)`.
    /// Defaults to status_filter = "accepted" when not specified.
    pub async fn get_decisions_affecting(
        &self,
        entity_type: &str,
        entity_id: &str,
        status_filter: Option<&str>,
    ) -> Result<Vec<DecisionNode>> {
        let match_field = if entity_type == "File" { "path" } else { "id" };
        let status = status_filter.unwrap_or("accepted");

        let cypher = format!(
            r#"
            MATCH (d:Decision)-[:AFFECTS]->(e:{} {{{}: $entity_id}})
            WHERE ($status IS NULL OR d.status = $status)
            RETURN d
            ORDER BY d.decided_at DESC
            "#,
            entity_type, match_field
        );

        let q = query(&cypher)
            .param("entity_id", entity_id.to_string())
            .param("status", status.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut decisions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            decisions.push(Self::node_to_decision(&node)?);
        }

        Ok(decisions)
    }

    // ========================================================================
    // AFFECTS relations (Decision → Entity)
    // ========================================================================

    /// Create an AFFECTS relation from a Decision to any entity in the graph.
    ///
    /// The entity is matched by a generic `{id: $entity_id}` or `{path: $entity_id}`
    /// (for File nodes). The `impact_description` is optional free-text.
    pub async fn add_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        impact_description: Option<&str>,
    ) -> Result<()> {
        let match_field = if entity_type == "File" { "path" } else { "id" };
        let cypher = format!(
            r#"
            MATCH (d:Decision {{id: $did}})
            MATCH (e:{} {{{}: $eid}})
            MERGE (d)-[r:AFFECTS]->(e)
            SET r.impact_description = $desc,
                r.created_at = datetime()
            "#,
            entity_type, match_field
        );

        let q = query(&cypher)
            .param("did", decision_id.to_string())
            .param("eid", entity_id.to_string())
            .param("desc", impact_description.unwrap_or("").to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove an AFFECTS relation from a Decision to an entity.
    pub async fn remove_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<()> {
        let match_field = if entity_type == "File" { "path" } else { "id" };
        let cypher = format!(
            r#"
            MATCH (d:Decision {{id: $did}})-[r:AFFECTS]->(e:{} {{{}: $eid}})
            DELETE r
            "#,
            entity_type, match_field
        );

        let q = query(&cypher)
            .param("did", decision_id.to_string())
            .param("eid", entity_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List all entities affected by a Decision.
    ///
    /// Returns tuples of (entity_type, entity_id, impact_description).
    pub async fn list_decision_affects(&self, decision_id: Uuid) -> Result<Vec<AffectsRelation>> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $did})-[r:AFFECTS]->(e)
            RETURN labels(e)[0] AS entity_type,
                   coalesce(e.path, e.id) AS entity_id,
                   coalesce(e.name, e.path, e.id) AS entity_name,
                   r.impact_description AS impact_description
            ORDER BY entity_type, entity_id
            "#,
        )
        .param("did", decision_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut affects = Vec::new();

        while let Some(row) = result.next().await? {
            affects.push(AffectsRelation {
                entity_type: row.get::<String>("entity_type").unwrap_or_default(),
                entity_id: row.get::<String>("entity_id").unwrap_or_default(),
                entity_name: row.get::<String>("entity_name").ok(),
                impact_description: row
                    .get::<String>("impact_description")
                    .ok()
                    .filter(|s| !s.is_empty()),
            });
        }

        Ok(affects)
    }

    // ========================================================================
    // Decision Timeline
    // ========================================================================

    /// Get a timeline of decisions, optionally filtered by task and date range.
    ///
    /// For each decision, resolves the SUPERSEDES chain (decisions it supersedes)
    /// and checks if it has been superseded by a newer decision.
    /// Returns entries ordered by decided_at DESC.
    pub async fn get_decision_timeline(
        &self,
        task_id: Option<Uuid>,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Vec<DecisionTimelineEntry>> {
        let mut where_clauses = Vec::new();
        if task_id.is_some() {
            where_clauses.push("t.id = $task_id".to_string());
        }
        if from.is_some() {
            where_clauses.push("d.decided_at >= datetime($from)".to_string());
        }
        if to.is_some() {
            where_clauses.push("d.decided_at <= datetime($to)".to_string());
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let cypher = format!(
            r#"
            MATCH (t:Task)-[:INFORMED_BY]->(d:Decision)
            {}
            OPTIONAL MATCH chain = (d)-[:SUPERSEDES*]->(old)
            WITH d, [n IN nodes(chain) WHERE n <> d | n.id] AS chain_ids
            OPTIONAL MATCH (newer)-[:SUPERSEDES]->(d)
            RETURN d, chain_ids, newer.id AS superseded_by
            ORDER BY d.decided_at DESC
            "#,
            where_clause
        );

        let mut q = query(&cypher);
        if let Some(tid) = task_id {
            q = q.param("task_id", tid.to_string());
        }
        if let Some(f) = from {
            q = q.param("from", f.to_string());
        }
        if let Some(t) = to {
            q = q.param("to", t.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut entries = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            let decision = Self::node_to_decision(&node)?;

            let chain_ids: Vec<Uuid> = row
                .get::<Vec<String>>("chain_ids")
                .unwrap_or_default()
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();

            let superseded_by: Option<Uuid> = row
                .get::<String>("superseded_by")
                .ok()
                .and_then(|s| s.parse().ok());

            entries.push(DecisionTimelineEntry {
                decision,
                supersedes_chain: chain_ids,
                superseded_by,
            });
        }

        Ok(entries)
    }

    // ========================================================================
    // SUPERSEDES relation (Decision → Decision)
    // ========================================================================

    /// Mark a decision as superseded by a newer decision.
    ///
    /// Sets the old decision's status to 'superseded' and creates a
    /// SUPERSEDES relationship from the new decision to the old one.
    pub async fn supersede_decision(
        &self,
        new_decision_id: Uuid,
        old_decision_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (new:Decision {id: $new_id})
            MATCH (old:Decision {id: $old_id})
            SET old.status = 'superseded'
            MERGE (new)-[:SUPERSEDES {created_at: datetime()}]->(old)
            "#,
        )
        .param("new_id", new_decision_id.to_string())
        .param("old_id", old_decision_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Graph visualization batch queries
    // ========================================================================

    /// Get all decisions scoped to a project (via Plan→Task→Decision chain)
    /// along with their AFFECTS relations. Used by the graph visualization endpoint.
    pub async fn get_project_decisions_for_graph(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(DecisionNode, Vec<AffectsRelation>)>> {
        // Step 1: Get all decisions for this project
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_PLAN]->(plan:Plan)
                  -[:HAS_TASK]->(task:Task)-[:INFORMED_BY]->(d:Decision)
            RETURN d.id AS id, d.description AS description, d.rationale AS rationale,
                   d.alternatives AS alternatives, d.chosen_option AS chosen_option,
                   d.decided_by AS decided_by, d.decided_at AS decided_at,
                   d.status AS status
            "#,
        )
        .param("pid", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut decisions: Vec<DecisionNode> = Vec::new();
        while let Some(row) = result.next().await? {
            let id_str: String = row.get("id").unwrap_or_default();
            let id = id_str.parse::<Uuid>().unwrap_or_default();
            if id.is_nil() {
                continue;
            }
            let alts_raw: Vec<String> = row.get("alternatives").unwrap_or_default();
            let status: DecisionStatus = row
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DecisionStatus::Accepted);
            let decided_at_str: String = row.get("decided_at").unwrap_or_default();

            decisions.push(DecisionNode {
                id,
                description: row.get("description").unwrap_or_default(),
                rationale: row.get("rationale").unwrap_or_default(),
                alternatives: alts_raw,
                chosen_option: row.get("chosen_option").ok(),
                decided_by: row
                    .get("decided_by")
                    .unwrap_or_else(|_| "unknown".to_string()),
                decided_at: decided_at_str
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
                status,
                embedding: None,
                embedding_model: None,
                scar_intensity: row.get("scar_intensity").unwrap_or(0.0),
            });
        }

        // Step 2: For each decision, get AFFECTS relations
        let mut results = Vec::new();
        for decision in decisions {
            let affects = self
                .list_decision_affects(decision.id)
                .await
                .unwrap_or_default();
            results.push((decision, affects));
        }

        Ok(results)
    }
}
