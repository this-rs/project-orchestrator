//! Neo4j persistence for ReasoningTrees
//!
//! Implements selective persistence of ReasoningTrees that have proven their
//! value (via positive feedback or correlation with note/decision creation).
//!
//! Trees are stored as `PersistedReasoningTree` nodes with a serialized JSON
//! blob for the tree structure (not individual nodes — too heavy for v1).
//! A `REASONING_FOR` relation links the tree to the ProtocolRun or Note
//! that it was associated with.

use super::client::Neo4jClient;
use crate::reasoning::models::ReasoningTree;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

/// Metadata about a persisted reasoning tree (without the full tree blob).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistedTreeMeta {
    pub id: Uuid,
    pub request: String,
    pub confidence: f64,
    pub node_count: usize,
    pub depth: usize,
    pub created_at: DateTime<Utc>,
    pub persisted_at: DateTime<Utc>,
    pub linked_entity_type: Option<String>,
    pub linked_entity_id: Option<String>,
}

impl Neo4jClient {
    /// Persist a ReasoningTree to Neo4j as a `PersistedReasoningTree` node.
    ///
    /// The tree structure is stored as a JSON blob in `serialized_tree`.
    /// Optionally links it to a ProtocolRun or Note via `REASONING_FOR`.
    ///
    /// Returns the ID of the persisted tree.
    pub async fn persist_reasoning_tree(
        &self,
        tree: &ReasoningTree,
        linked_entity_type: Option<&str>,
        linked_entity_id: Option<Uuid>,
    ) -> Result<Uuid> {
        let serialized = serde_json::to_string(&tree.roots)
            .context("Failed to serialize reasoning tree roots")?;

        let tree_id = tree.id;

        let q = query(
            r#"
            MERGE (t:PersistedReasoningTree {id: $id})
            ON CREATE SET
                t.request = $request,
                t.confidence = $confidence,
                t.node_count = $node_count,
                t.depth = $depth,
                t.serialized_tree = $serialized_tree,
                t.created_at = $created_at,
                t.persisted_at = datetime(),
                t.project_id = $project_id
            RETURN t.id AS id
            "#,
        )
        .param("id", tree_id.to_string())
        .param("request", tree.request.clone())
        .param("confidence", tree.confidence)
        .param("node_count", tree.node_count as i64)
        .param("depth", tree.depth as i64)
        .param("serialized_tree", serialized)
        .param("created_at", tree.created_at.to_rfc3339())
        .param(
            "project_id",
            tree.project_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        );

        self.graph.run(q).await.context("Failed to persist reasoning tree")?;

        // Create REASONING_FOR relation if a linked entity is specified
        if let (Some(entity_type), Some(entity_id)) = (linked_entity_type, linked_entity_id) {
            let label = match entity_type {
                "run" | "ProtocolRun" => "ProtocolRun",
                "note" | "Note" => "Note",
                _ => {
                    tracing::warn!(
                        "Unknown entity type for REASONING_FOR: {entity_type}, skipping link"
                    );
                    return Ok(tree_id);
                }
            };

            let cypher = format!(
                r#"
                MATCH (t:PersistedReasoningTree {{id: $tree_id}})
                MATCH (e:{} {{id: $entity_id}})
                MERGE (t)-[:REASONING_FOR]->(e)
                "#,
                label
            );

            let q = query(&cypher)
                .param("tree_id", tree_id.to_string())
                .param("entity_id", entity_id.to_string());

            if let Err(e) = self.graph.run(q).await {
                tracing::warn!("Failed to create REASONING_FOR link: {e}");
            }
        }

        Ok(tree_id)
    }

    /// Retrieve a persisted reasoning tree by ID.
    ///
    /// Deserializes the JSON blob back into a full `ReasoningTree`.
    pub async fn get_persisted_reasoning_tree(
        &self,
        tree_id: Uuid,
    ) -> Result<Option<ReasoningTree>> {
        let q = query(
            r#"
            MATCH (t:PersistedReasoningTree {id: $id})
            RETURN t
            "#,
        )
        .param("id", tree_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let tree = self.node_to_reasoning_tree(&node)?;
            Ok(Some(tree))
        } else {
            Ok(None)
        }
    }

    /// Get the reasoning tree associated with a protocol run (via REASONING_FOR).
    pub async fn get_run_reasoning_tree(
        &self,
        run_id: Uuid,
    ) -> Result<Option<ReasoningTree>> {
        let q = query(
            r#"
            MATCH (t:PersistedReasoningTree)-[:REASONING_FOR]->(r:ProtocolRun {id: $run_id})
            RETURN t
            ORDER BY t.persisted_at DESC
            LIMIT 1
            "#,
        )
        .param("run_id", run_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let tree = self.node_to_reasoning_tree(&node)?;
            Ok(Some(tree))
        } else {
            Ok(None)
        }
    }

    /// List persisted reasoning trees (metadata only, without the full blob).
    pub async fn list_persisted_reasoning_trees(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
    ) -> Result<Vec<PersistedTreeMeta>> {
        let cypher = if project_id.is_some() {
            r#"
            MATCH (t:PersistedReasoningTree)
            WHERE t.project_id = $project_id
            OPTIONAL MATCH (t)-[:REASONING_FOR]->(e)
            RETURN t, labels(e) AS entity_labels, e.id AS entity_id
            ORDER BY t.persisted_at DESC
            LIMIT $limit
            "#
        } else {
            r#"
            MATCH (t:PersistedReasoningTree)
            OPTIONAL MATCH (t)-[:REASONING_FOR]->(e)
            RETURN t, labels(e) AS entity_labels, e.id AS entity_id
            ORDER BY t.persisted_at DESC
            LIMIT $limit
            "#
        };

        let mut q = query(cypher).param("limit", limit as i64);
        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut metas = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let entity_labels: Vec<String> = row.get("entity_labels").unwrap_or_default();
            let entity_id: Option<String> = row.get::<String>("entity_id").ok();

            let linked_entity_type = entity_labels.first().cloned();

            metas.push(PersistedTreeMeta {
                id: node.get::<String>("id")?.parse()?,
                request: node.get("request")?,
                confidence: node.get("confidence").unwrap_or(0.0),
                node_count: node.get::<i64>("node_count").unwrap_or(0) as usize,
                depth: node.get::<i64>("depth").unwrap_or(0) as usize,
                created_at: node
                    .get::<String>("created_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(Utc::now),
                persisted_at: node
                    .get::<String>("persisted_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(Utc::now),
                linked_entity_type,
                linked_entity_id: entity_id,
            });
        }

        Ok(metas)
    }

    /// Convert a Neo4j node to a [`ReasoningTree`].
    fn node_to_reasoning_tree(&self, node: &neo4rs::Node) -> Result<ReasoningTree> {
        let serialized: String = node.get("serialized_tree")?;
        let roots = serde_json::from_str(&serialized)
            .context("Failed to deserialize reasoning tree roots")?;

        let mut tree = ReasoningTree::new(
            node.get::<String>("request")?,
            node.get::<String>("project_id")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
        );
        tree.id = node.get::<String>("id")?.parse()?;
        tree.roots = roots;
        tree.confidence = node.get("confidence").unwrap_or(0.0);
        tree.node_count = node.get::<i64>("node_count").unwrap_or(0) as usize;
        tree.depth = node.get::<i64>("depth").unwrap_or(0) as usize;
        tree.created_at = node
            .get::<String>("created_at")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(Utc::now);

        Ok(tree)
    }
}
