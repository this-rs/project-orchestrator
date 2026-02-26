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
                decided_at: datetime($decided_at)
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
        .param("decided_at", decision.decided_at.to_rfc3339());

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
            Ok(Some(DecisionNode {
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
            }))
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
        // Try both direct AFFECTS and indirect via task affected_files
        let cypher = format!(
            r#"
            // Path 1: Direct AFFECTS relation (P3 — may not exist yet)
            OPTIONAL MATCH (d1:Decision)-[:AFFECTS]->(target:{} {{{}: $entity_id}})
            WITH collect(DISTINCT d1) AS direct_decisions

            // Path 2: Indirect via Task affected_files containing entity_id
            OPTIONAL MATCH (t:Task)-[:INFORMED_BY]->(d2:Decision)
            WHERE $entity_id IN t.affected_files
            WITH direct_decisions, collect(DISTINCT d2) AS indirect_decisions

            // Merge and deduplicate
            WITH [d IN direct_decisions + indirect_decisions WHERE d IS NOT NULL] AS all_decisions
            UNWIND all_decisions AS d
            WITH DISTINCT d
            RETURN d
            ORDER BY d.decided_at DESC
            LIMIT $limit
            "#,
            entity_type,
            if entity_type == "File" { "path" } else { "id" },
        );

        let q = query(&cypher)
            .param("entity_id", entity_id.to_string())
            .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut decisions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            decisions.push(DecisionNode {
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
            });
        }

        Ok(decisions)
    }
}
