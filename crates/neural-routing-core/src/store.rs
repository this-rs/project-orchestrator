//! Neo4j implementation of TrajectoryStore.

use async_trait::async_trait;
use neo4rs::{query, Graph};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{NeuralRoutingError, Result};
use crate::models::*;
use crate::traits::TrajectoryStore;

/// Neo4j-backed trajectory storage.
pub struct Neo4jTrajectoryStore {
    graph: Arc<Graph>,
}

impl Neo4jTrajectoryStore {
    pub fn new(graph: Arc<Graph>) -> Self {
        Self { graph }
    }

    /// Ensure indexes exist for trajectory queries.
    pub async fn ensure_indexes(&self) -> Result<()> {
        let queries = vec![
            // Composite index on Trajectory
            "CREATE INDEX trajectory_session_idx IF NOT EXISTS FOR (t:Trajectory) ON (t.session_id, t.created_at)",
            // Index on Trajectory.total_reward for sorting
            "CREATE INDEX trajectory_reward_idx IF NOT EXISTS FOR (t:Trajectory) ON (t.total_reward)",
            // Index on TrajectoryNode order
            "CREATE INDEX tnode_order_idx IF NOT EXISTS FOR (tn:TrajectoryNode) ON (tn.order_idx)",
        ];

        for q in queries {
            self.graph
                .run(query(q))
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        // Vector indexes (separate because they use a different syntax)
        let vector_queries = vec![
            "CREATE VECTOR INDEX trajectory_query_embedding_idx IF NOT EXISTS
             FOR (t:Trajectory) ON (t.query_embedding)
             OPTIONS {indexConfig: {`vector.dimensions`: 256, `vector.similarity_function`: 'cosine'}}",
            "CREATE VECTOR INDEX tnode_context_embedding_idx IF NOT EXISTS
             FOR (tn:TrajectoryNode) ON (tn.context_embedding)
             OPTIONS {indexConfig: {`vector.dimensions`: 256, `vector.similarity_function`: 'cosine'}}",
        ];

        for q in vector_queries {
            self.graph
                .run(query(q))
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        // Relation indexes for USED_TOOL and TOUCHED_ENTITY
        let relation_queries = vec![
            // Index on USED_TOOL relation properties for efficient lookup
            "CREATE INDEX used_tool_name_idx IF NOT EXISTS FOR ()-[r:USED_TOOL]-() ON (r.tool_name)",
            // Index on TOUCHED_ENTITY relation properties
            "CREATE INDEX touched_entity_type_idx IF NOT EXISTS FOR ()-[r:TOUCHED_ENTITY]-() ON (r.entity_type, r.entity_id)",
        ];

        for q in relation_queries {
            self.graph
                .run(query(q))
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        tracing::info!("Neural routing indexes ensured (including USED_TOOL/TOUCHED_ENTITY)");
        Ok(())
    }

    /// Link a trajectory node to a tool it used (USED_TOOL relation).
    pub async fn link_tool_usage(&self, node_id: &Uuid, usage: &ToolUsage) -> Result<()> {
        let q = query(
            "MATCH (tn:TrajectoryNode {id: $node_id})
             MERGE (tool:ToolNode {tool_name: $tool_name, action: $action})
             MERGE (tn)-[r:USED_TOOL]->(tool)
             SET r.params_hash = $params_hash,
                 r.duration_ms = $duration_ms,
                 r.success = $success",
        )
        .param("node_id", node_id.to_string())
        .param("tool_name", usage.tool_name.clone())
        .param("action", usage.action.clone())
        .param("params_hash", usage.params_hash.clone())
        .param("duration_ms", usage.duration_ms.unwrap_or(0) as i64)
        .param("success", usage.success);

        self.graph
            .run(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        tracing::trace!(
            node_id = %node_id,
            tool = %usage.tool_name,
            action = %usage.action,
            "Linked tool usage"
        );
        Ok(())
    }

    /// Link a trajectory node to an entity it touched (TOUCHED_ENTITY relation).
    ///
    /// The entity is matched by type+id in the existing graph (File, Function, Note, Skill, etc.).
    /// If the target entity doesn't exist, a lightweight proxy node is created.
    pub async fn link_touched_entity(&self, node_id: &Uuid, entity: &TouchedEntity) -> Result<()> {
        // Use MERGE on a generic TouchedNode to avoid coupling to PO's exact label scheme.
        // The entity_type+entity_id pair is the unique key.
        let q = query(
            "MATCH (tn:TrajectoryNode {id: $node_id})
             MERGE (e:TouchedNode {entity_type: $entity_type, entity_id: $entity_id})
             MERGE (tn)-[r:TOUCHED_ENTITY]->(e)
             SET r.access_mode = $access_mode,
                 r.relevance = $relevance,
                 r.entity_type = $entity_type,
                 r.entity_id = $entity_id",
        )
        .param("node_id", node_id.to_string())
        .param("entity_type", entity.entity_type.clone())
        .param("entity_id", entity.entity_id.clone())
        .param("access_mode", entity.access_mode.clone())
        .param("relevance", entity.relevance.unwrap_or(0.0));

        self.graph
            .run(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        tracing::trace!(
            node_id = %node_id,
            entity_type = %entity.entity_type,
            entity_id = %entity.entity_id,
            "Linked touched entity"
        );
        Ok(())
    }

    /// Batch-link multiple tool usages for a trajectory node.
    pub async fn link_tool_usages_batch(&self, node_id: &Uuid, usages: &[ToolUsage]) -> Result<()> {
        if usages.is_empty() {
            return Ok(());
        }

        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        for usage in usages {
            let q = query(
                "MATCH (tn:TrajectoryNode {id: $node_id})
                 MERGE (tool:ToolNode {tool_name: $tool_name, action: $action})
             MERGE (tn)-[r:USED_TOOL]->(tool)
                 SET r.params_hash = $params_hash,
                     r.duration_ms = $duration_ms,
                     r.success = $success",
            )
            .param("node_id", node_id.to_string())
            .param("tool_name", usage.tool_name.clone())
            .param("action", usage.action.clone())
            .param("params_hash", usage.params_hash.clone())
            .param("duration_ms", usage.duration_ms.unwrap_or(0) as i64)
            .param("success", usage.success);

            txn.run(q)
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        txn.commit()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        Ok(())
    }

    /// Batch-link multiple touched entities for a trajectory node.
    pub async fn link_touched_entities_batch(
        &self,
        node_id: &Uuid,
        entities: &[TouchedEntity],
    ) -> Result<()> {
        if entities.is_empty() {
            return Ok(());
        }

        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        for entity in entities {
            let q = query(
                "MATCH (tn:TrajectoryNode {id: $node_id})
                 MERGE (e:TouchedNode {entity_type: $entity_type, entity_id: $entity_id})
                 MERGE (tn)-[r:TOUCHED_ENTITY]->(e)
                 SET r.access_mode = $access_mode,
                     r.relevance = $relevance,
                     r.entity_type = $entity_type,
                     r.entity_id = $entity_id",
            )
            .param("node_id", node_id.to_string())
            .param("entity_type", entity.entity_type.clone())
            .param("entity_id", entity.entity_id.clone())
            .param("access_mode", entity.access_mode.clone())
            .param("relevance", entity.relevance.unwrap_or(0.0));

            txn.run(q)
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        txn.commit()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        Ok(())
    }

    /// Get all tool usages for a trajectory node.
    pub async fn get_node_tool_usages(&self, node_id: &Uuid) -> Result<Vec<ToolUsage>> {
        let q = query(
            "MATCH (tn:TrajectoryNode {id: $node_id})-[r:USED_TOOL]->(tool:ToolNode)
             RETURN tool.tool_name AS tool_name, tool.action AS action,
                    r.params_hash AS params_hash, r.duration_ms AS duration_ms,
                    r.success AS success",
        )
        .param("node_id", node_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        let mut usages = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            usages.push(ToolUsage {
                tool_name: row.get("tool_name").unwrap_or_default(),
                action: row.get("action").unwrap_or_default(),
                params_hash: row.get("params_hash").unwrap_or_default(),
                duration_ms: row.get::<i64>("duration_ms").ok().map(|v| v as u64),
                success: row.get("success").unwrap_or(true),
            });
        }

        Ok(usages)
    }

    /// Get all touched entities for a trajectory node.
    pub async fn get_node_touched_entities(&self, node_id: &Uuid) -> Result<Vec<TouchedEntity>> {
        let q = query(
            "MATCH (tn:TrajectoryNode {id: $node_id})-[r:TOUCHED_ENTITY]->(e:TouchedNode)
             RETURN r.entity_type AS entity_type, r.entity_id AS entity_id,
                    r.access_mode AS access_mode, r.relevance AS relevance",
        )
        .param("node_id", node_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        let mut entities = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            entities.push(TouchedEntity {
                entity_type: row.get("entity_type").unwrap_or_default(),
                entity_id: row.get("entity_id").unwrap_or_default(),
                access_mode: row.get("access_mode").unwrap_or_default(),
                relevance: row.get::<f64>("relevance").ok(),
            });
        }

        Ok(entities)
    }
}

#[async_trait]
impl TrajectoryStore for Neo4jTrajectoryStore {
    async fn store_trajectory(&self, trajectory: &Trajectory) -> Result<()> {
        // Step 1: Create Trajectory node
        let create_trajectory = query(
            "CREATE (t:Trajectory {
                id: $id,
                session_id: $session_id,
                query_embedding: $query_embedding,
                total_reward: $total_reward,
                step_count: $step_count,
                duration_ms: $duration_ms,
                created_at: datetime($created_at)
            })",
        )
        .param("id", trajectory.id.to_string())
        .param("session_id", trajectory.session_id.clone())
        .param("query_embedding", trajectory.query_embedding.clone())
        .param("total_reward", trajectory.total_reward)
        .param("step_count", trajectory.step_count as i64)
        .param("duration_ms", trajectory.duration_ms as i64)
        .param("created_at", trajectory.created_at.to_rfc3339());

        self.graph
            .run(create_trajectory)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        // Step 2: Insert nodes individually (neo4rs doesn't support UNWIND with complex maps)
        if !trajectory.nodes.is_empty() {
            let mut txn = self
                .graph
                .start_txn()
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

            for node in &trajectory.nodes {
                let create_node = query(
                    "MATCH (t:Trajectory {id: $trajectory_id})
                     CREATE (tn:TrajectoryNode {
                         id: $id,
                         context_embedding: $context_embedding,
                         action_type: $action_type,
                         action_params: $action_params,
                         alternatives_count: $alternatives_count,
                         chosen_index: $chosen_index,
                         confidence: $confidence,
                         local_reward: $local_reward,
                         cumulative_reward: $cumulative_reward,
                         delta_ms: $delta_ms,
                         order_idx: $order_idx
                     })
                     CREATE (t)-[:CONTAINS {order_idx: $order_idx}]->(tn)",
                )
                .param("trajectory_id", trajectory.id.to_string())
                .param("id", node.id.to_string())
                .param("context_embedding", node.context_embedding.clone())
                .param("action_type", node.action_type.clone())
                .param("action_params", node.action_params.to_string())
                .param("alternatives_count", node.alternatives_count as i64)
                .param("chosen_index", node.chosen_index as i64)
                .param("confidence", node.confidence)
                .param("local_reward", node.local_reward)
                .param("cumulative_reward", node.cumulative_reward)
                .param("delta_ms", node.delta_ms as i64)
                .param("order_idx", node.order as i64);

                txn.run(create_node)
                    .await
                    .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            }

            // Step 3: Create NEXT_DECISION chains between consecutive nodes
            let chain_nodes = query(
                "MATCH (t:Trajectory {id: $trajectory_id})-[:CONTAINS]->(tn:TrajectoryNode)
                 WITH tn ORDER BY tn.order_idx
                 WITH collect(tn) AS nodes
                 UNWIND range(0, size(nodes) - 2) AS i
                 WITH nodes[i] AS current, nodes[i+1] AS next
                 CREATE (current)-[:NEXT_DECISION {delta_ms: next.delta_ms}]->(next)",
            )
            .param("trajectory_id", trajectory.id.to_string());

            txn.run(chain_nodes)
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

            txn.commit()
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
        }

        tracing::debug!(
            trajectory_id = %trajectory.id,
            nodes = trajectory.nodes.len(),
            "Stored trajectory"
        );

        Ok(())
    }

    async fn get_trajectory(&self, id: &Uuid) -> Result<Option<Trajectory>> {
        let q = query(
            "MATCH (t:Trajectory {id: $id})
             OPTIONAL MATCH (t)-[:CONTAINS]->(tn:TrajectoryNode)
             WITH t, tn ORDER BY tn.order_idx
             WITH t, collect(tn) AS nodes
             RETURN t, nodes",
        )
        .param("id", id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        if let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            let t: neo4rs::Node = row
                .get("t")
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            let nodes_raw: Vec<neo4rs::Node> = row
                .get("nodes")
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

            let nodes = nodes_raw
                .into_iter()
                .map(|n| parse_trajectory_node(&n))
                .collect::<Result<Vec<_>>>()?;

            Ok(Some(parse_trajectory(&t, nodes)?))
        } else {
            Ok(None)
        }
    }

    async fn list_trajectories(&self, filter: &TrajectoryFilter) -> Result<Vec<Trajectory>> {
        let mut conditions = vec!["1=1".to_string()];
        let mut cypher_params: Vec<(String, serde_json::Value)> = vec![];

        if let Some(ref session_id) = filter.session_id {
            conditions.push("t.session_id = $session_id".to_string());
            cypher_params.push(("session_id".into(), serde_json::json!(session_id)));
        }
        if let Some(min_r) = filter.min_reward {
            conditions.push("t.total_reward >= $min_reward".to_string());
            cypher_params.push(("min_reward".into(), serde_json::json!(min_r)));
        }
        if let Some(max_r) = filter.max_reward {
            conditions.push("t.total_reward <= $max_reward".to_string());
            cypher_params.push(("max_reward".into(), serde_json::json!(max_r)));
        }
        if let Some(ref from_date) = filter.from_date {
            conditions.push("t.created_at >= datetime($from_date)".to_string());
            cypher_params.push((
                "from_date".into(),
                serde_json::json!(from_date.to_rfc3339()),
            ));
        }
        if let Some(ref to_date) = filter.to_date {
            conditions.push("t.created_at <= datetime($to_date)".to_string());
            cypher_params.push(("to_date".into(), serde_json::json!(to_date.to_rfc3339())));
        }
        if let Some(min_s) = filter.min_steps {
            conditions.push("t.step_count >= $min_steps".to_string());
            cypher_params.push(("min_steps".into(), serde_json::json!(min_s as i64)));
        }
        if let Some(max_s) = filter.max_steps {
            conditions.push("t.step_count <= $max_steps".to_string());
            cypher_params.push(("max_steps".into(), serde_json::json!(max_s as i64)));
        }

        let limit = filter.limit.unwrap_or(50).min(200);
        let offset = filter.offset.unwrap_or(0);

        let cypher = format!(
            "MATCH (t:Trajectory)
             WHERE {}
             RETURN t
             ORDER BY t.created_at DESC
             SKIP $offset LIMIT $limit",
            conditions.join(" AND ")
        );

        let mut q = query(&cypher)
            .param("offset", offset as i64)
            .param("limit", limit as i64);

        for (key, val) in &cypher_params {
            match val {
                serde_json::Value::String(s) => {
                    q = q.param(key.as_str(), s.clone());
                }
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        q = q.param(key.as_str(), f);
                    }
                }
                _ => {}
            }
        }

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        let mut trajectories = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            let t: neo4rs::Node = row
                .get("t")
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            trajectories.push(parse_trajectory(&t, vec![])?);
        }

        Ok(trajectories)
    }

    async fn search_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(Trajectory, f64)>> {
        let q = query(
            "CALL db.index.vector.queryNodes('trajectory_query_embedding_idx', $top_k, $embedding)
             YIELD node AS t, score
             WHERE score >= $min_similarity
             RETURN t, score
             ORDER BY score DESC",
        )
        .param("top_k", top_k as i64)
        .param("embedding", query_embedding.to_vec())
        .param("min_similarity", min_similarity as f64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            let t: neo4rs::Node = row
                .get("t")
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            let score: f64 = row
                .get("score")
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            let trajectory = parse_trajectory(&t, vec![])?;
            results.push((trajectory, score));
        }

        Ok(results)
    }

    async fn get_stats(&self) -> Result<TrajectoryStats> {
        let q = query(
            "MATCH (t:Trajectory)
             WITH count(t) AS cnt,
                  avg(t.total_reward) AS avg_r,
                  avg(t.step_count) AS avg_s,
                  avg(t.duration_ms) AS avg_d,
                  min(t.total_reward) AS min_r,
                  max(t.total_reward) AS max_r,
                  percentileCont(t.total_reward, 0.25) AS p25,
                  percentileCont(t.total_reward, 0.5) AS p50,
                  percentileCont(t.total_reward, 0.75) AS p75,
                  percentileCont(t.total_reward, 0.9) AS p90
             RETURN cnt, avg_r, avg_s, avg_d, min_r, max_r, p25, p50, p75, p90",
        );

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        if let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            Ok(TrajectoryStats {
                total_count: row.get::<i64>("cnt").unwrap_or(0) as usize,
                avg_reward: row.get::<f64>("avg_r").unwrap_or(0.0),
                avg_step_count: row.get::<f64>("avg_s").unwrap_or(0.0),
                avg_duration_ms: row.get::<f64>("avg_d").unwrap_or(0.0),
                reward_distribution: RewardDistribution {
                    min: row.get::<f64>("min_r").unwrap_or(0.0),
                    max: row.get::<f64>("max_r").unwrap_or(0.0),
                    p25: row.get::<f64>("p25").unwrap_or(0.0),
                    p50: row.get::<f64>("p50").unwrap_or(0.0),
                    p75: row.get::<f64>("p75").unwrap_or(0.0),
                    p90: row.get::<f64>("p90").unwrap_or(0.0),
                },
            })
        } else {
            Ok(TrajectoryStats {
                total_count: 0,
                avg_reward: 0.0,
                avg_step_count: 0.0,
                avg_duration_ms: 0.0,
                reward_distribution: RewardDistribution {
                    min: 0.0,
                    max: 0.0,
                    p25: 0.0,
                    p50: 0.0,
                    p75: 0.0,
                    p90: 0.0,
                },
            })
        }
    }

    async fn count(&self) -> Result<usize> {
        let q = query("MATCH (t:Trajectory) RETURN count(t) AS cnt");
        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        if let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            Ok(row.get::<i64>("cnt").unwrap_or(0) as usize)
        } else {
            Ok(0)
        }
    }

    async fn store_trajectories_batch(&self, trajectories: &[Trajectory]) -> Result<usize> {
        if trajectories.is_empty() {
            return Ok(0);
        }

        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        // ---------------------------------------------------------------
        // Query 1: UNWIND to create all Trajectory nodes in one shot.
        // Uses parallel arrays (one per field) indexed together.
        // ---------------------------------------------------------------
        let ids: Vec<String> = trajectories.iter().map(|t| t.id.to_string()).collect();
        let session_ids: Vec<String> = trajectories.iter().map(|t| t.session_id.clone()).collect();
        let embeddings: Vec<Vec<f32>> = trajectories
            .iter()
            .map(|t| t.query_embedding.clone())
            .collect();
        let rewards: Vec<f64> = trajectories.iter().map(|t| t.total_reward).collect();
        let step_counts: Vec<i64> = trajectories.iter().map(|t| t.step_count as i64).collect();
        let durations: Vec<i64> = trajectories.iter().map(|t| t.duration_ms as i64).collect();
        let created_ats: Vec<String> = trajectories
            .iter()
            .map(|t| t.created_at.to_rfc3339())
            .collect();

        let create_trajectories = query(
            "UNWIND range(0, size($ids) - 1) AS i
             CREATE (t:Trajectory {
                 id: $ids[i],
                 session_id: $session_ids[i],
                 query_embedding: $embeddings[i],
                 total_reward: $rewards[i],
                 step_count: $step_counts[i],
                 duration_ms: $durations[i],
                 created_at: datetime($created_ats[i])
             })",
        )
        .param("ids", ids)
        .param("session_ids", session_ids)
        .param("embeddings", embeddings)
        .param("rewards", rewards)
        .param("step_counts", step_counts)
        .param("durations", durations)
        .param("created_ats", created_ats);

        txn.run(create_trajectories)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        // ---------------------------------------------------------------
        // Query 2: UNWIND to create all TrajectoryNodes + CONTAINS edges.
        // Flattens all nodes across all trajectories into parallel arrays.
        // ---------------------------------------------------------------
        let mut all_traj_ids: Vec<String> = Vec::new();
        let mut all_node_ids: Vec<String> = Vec::new();
        let mut all_ctx_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut all_action_types: Vec<String> = Vec::new();
        let mut all_action_params: Vec<String> = Vec::new();
        let mut all_alt_counts: Vec<i64> = Vec::new();
        let mut all_chosen: Vec<i64> = Vec::new();
        let mut all_confidences: Vec<f64> = Vec::new();
        let mut all_local_rewards: Vec<f64> = Vec::new();
        let mut all_cum_rewards: Vec<f64> = Vec::new();
        let mut all_deltas: Vec<i64> = Vec::new();
        let mut all_orders: Vec<i64> = Vec::new();

        for trajectory in trajectories {
            for node in &trajectory.nodes {
                all_traj_ids.push(trajectory.id.to_string());
                all_node_ids.push(node.id.to_string());
                all_ctx_embeddings.push(node.context_embedding.clone());
                all_action_types.push(node.action_type.clone());
                all_action_params.push(node.action_params.to_string());
                all_alt_counts.push(node.alternatives_count as i64);
                all_chosen.push(node.chosen_index as i64);
                all_confidences.push(node.confidence);
                all_local_rewards.push(node.local_reward);
                all_cum_rewards.push(node.cumulative_reward);
                all_deltas.push(node.delta_ms as i64);
                all_orders.push(node.order as i64);
            }
        }

        if !all_node_ids.is_empty() {
            let create_nodes = query(
                "UNWIND range(0, size($node_ids) - 1) AS i
                 MATCH (t:Trajectory {id: $traj_ids[i]})
                 CREATE (tn:TrajectoryNode {
                     id: $node_ids[i],
                     context_embedding: $ctx_embeddings[i],
                     action_type: $action_types[i],
                     action_params: $action_params[i],
                     alternatives_count: $alt_counts[i],
                     chosen_index: $chosen[i],
                     confidence: $confidences[i],
                     local_reward: $local_rewards[i],
                     cumulative_reward: $cum_rewards[i],
                     delta_ms: $deltas[i],
                     order_idx: $orders[i]
                 })
                 CREATE (t)-[:CONTAINS {order_idx: $orders[i]}]->(tn)",
            )
            .param("traj_ids", all_traj_ids)
            .param("node_ids", all_node_ids)
            .param("ctx_embeddings", all_ctx_embeddings)
            .param("action_types", all_action_types)
            .param("action_params", all_action_params)
            .param("alt_counts", all_alt_counts)
            .param("chosen", all_chosen)
            .param("confidences", all_confidences)
            .param("local_rewards", all_local_rewards)
            .param("cum_rewards", all_cum_rewards)
            .param("deltas", all_deltas)
            .param("orders", all_orders);

            txn.run(create_nodes)
                .await
                .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

            // ---------------------------------------------------------------
            // Query 3: UNWIND trajectory IDs to create NEXT_DECISION chains.
            // For each trajectory, orders its nodes and links consecutive pairs.
            // ---------------------------------------------------------------
            let chain_ids: Vec<String> = trajectories
                .iter()
                .filter(|t| t.nodes.len() >= 2)
                .map(|t| t.id.to_string())
                .collect();

            if !chain_ids.is_empty() {
                let chain_query = query(
                    "UNWIND $traj_ids AS tid
                     MATCH (t:Trajectory {id: tid})-[:CONTAINS]->(tn:TrajectoryNode)
                     WITH t, tn ORDER BY tn.order_idx
                     WITH t, collect(tn) AS nodes
                     WHERE size(nodes) >= 2
                     UNWIND range(0, size(nodes) - 2) AS i
                     CREATE (nodes[i])-[:NEXT_DECISION {delta_ms: nodes[i+1].delta_ms}]->(nodes[i+1])",
                )
                .param("traj_ids", chain_ids);

                txn.run(chain_query)
                    .await
                    .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
            }
        }

        txn.commit()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        let count = trajectories.len();
        tracing::debug!(count, "Batch-stored trajectories via UNWIND");
        Ok(count)
    }

    async fn delete_trajectory(&self, id: &Uuid) -> Result<bool> {
        let q = query(
            "MATCH (t:Trajectory {id: $id})
             OPTIONAL MATCH (t)-[:CONTAINS]->(tn:TrajectoryNode)
             OPTIONAL MATCH (tn)-[:USED_TOOL]->(tool:ToolNode)
             OPTIONAL MATCH (tn)-[:TOUCHED_ENTITY]->(touched:TouchedNode)
             DETACH DELETE t, tn
             WITH tool, touched
             WHERE tool IS NOT NULL AND NOT EXISTS { MATCH ()-[:USED_TOOL]->(tool) }
             DETACH DELETE tool
             WITH touched
             WHERE touched IS NOT NULL AND NOT EXISTS { MATCH ()-[:TOUCHED_ENTITY]->(touched) }
             DETACH DELETE touched
             RETURN true AS deleted",
        )
        .param("id", id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

        if let Some(row) = result
            .next()
            .await
            .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
        {
            Ok(row.get::<bool>("deleted").unwrap_or(false))
        } else {
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_trajectory(node: &neo4rs::Node, nodes: Vec<TrajectoryNode>) -> Result<Trajectory> {
    let id_str: String = node
        .get("id")
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
    let id = Uuid::parse_str(&id_str)
        .map_err(|e| NeuralRoutingError::Neo4j(format!("Invalid trajectory UUID: {}", e)))?;

    Ok(Trajectory {
        id,
        session_id: node.get("session_id").unwrap_or_default(),
        query_embedding: node.get("query_embedding").unwrap_or_default(),
        total_reward: node.get("total_reward").unwrap_or(0.0),
        step_count: node.get::<i64>("step_count").unwrap_or(0) as usize,
        duration_ms: node.get::<i64>("duration_ms").unwrap_or(0) as u64,
        nodes,
        created_at: {
            // Parse ISO 8601 datetime string stored via datetime($created_at)
            let dt_str: String = node.get("created_at").unwrap_or_default();
            chrono::DateTime::parse_from_rfc3339(&dt_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now())
        },
    })
}

fn parse_trajectory_node(node: &neo4rs::Node) -> Result<TrajectoryNode> {
    let id_str: String = node
        .get("id")
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;
    let id = Uuid::parse_str(&id_str)
        .map_err(|e| NeuralRoutingError::Neo4j(format!("Invalid node UUID: {}", e)))?;

    let action_params_str: String = node.get("action_params").unwrap_or_default();
    let action_params: serde_json::Value =
        serde_json::from_str(&action_params_str).unwrap_or(serde_json::Value::Null);

    Ok(TrajectoryNode {
        id,
        context_embedding: node.get("context_embedding").unwrap_or_default(),
        action_type: node.get("action_type").unwrap_or_default(),
        action_params,
        alternatives_count: node.get::<i64>("alternatives_count").unwrap_or(0) as usize,
        chosen_index: node.get::<i64>("chosen_index").unwrap_or(0) as usize,
        confidence: node.get("confidence").unwrap_or(0.0),
        local_reward: node.get("local_reward").unwrap_or(0.0),
        cumulative_reward: node.get("cumulative_reward").unwrap_or(0.0),
        delta_ms: node.get::<i64>("delta_ms").unwrap_or(0) as u64,
        order: node.get::<i64>("order_idx").unwrap_or(0) as usize,
    })
}
