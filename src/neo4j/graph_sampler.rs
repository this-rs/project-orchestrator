//! GraphSampler — k-hop ego-graph extraction from Neo4j for GNN training.
//!
//! Extracts multi-relational subgraphs around target nodes, supporting:
//! - 8 relation types: IMPORTS, CALLS, CO_CHANGED, AFFECTS, SYNAPSE, KNOWS, CONTAINS, NEXT_DECISION
//! - PyTorch Geometric export: (edge_index, node_features, edge_types)
//! - Batch extraction with stratified sampling (overrepresent high-PageRank nodes)
//! - LRU cache for frequently accessed subgraphs
//!
//! This module is the storage adapter's half of the GNN sampling: it drives the
//! graph to build the [`neural_routing_gnn::sampler::SubGraph`] records that the
//! models consume. The record shapes themselves live in `neural-routing-gnn`.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use neo4rs::{query, Graph};
use serde::{Deserialize, Serialize};

use neural_routing_gnn::sampler::{
    export_to_pyg, PyGData, RelationType, SubGraph, SubGraphEdge, SubGraphNode,
};

// ---------------------------------------------------------------------------
// GraphSampler configuration
// ---------------------------------------------------------------------------

/// Configuration for the GraphSampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSamplerConfig {
    /// Default number of hops for ego-graph extraction.
    pub default_k_hops: usize,
    /// Maximum number of nodes per subgraph (limits explosion on dense graphs).
    pub max_nodes_per_subgraph: usize,
    /// LRU cache capacity (number of subgraphs to cache).
    pub cache_capacity: usize,
    /// Relation types to extract (empty = all).
    pub relation_types: Vec<RelationType>,
    /// Whether to include edge weights from Neo4j properties.
    pub include_weights: bool,
}

impl Default for GraphSamplerConfig {
    fn default() -> Self {
        Self {
            default_k_hops: 3,
            max_nodes_per_subgraph: 500,
            cache_capacity: 256,
            relation_types: RelationType::ALL.to_vec(),
            include_weights: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphSampler
// ---------------------------------------------------------------------------

/// Graph sampler — extracts k-hop ego-graphs from Neo4j for GNN training.
pub struct GraphSampler {
    graph: Arc<Graph>,
    config: GraphSamplerConfig,
    cache: Mutex<LruCache<String, SubGraph>>,
}

impl GraphSampler {
    pub fn new(graph: Arc<Graph>, config: GraphSamplerConfig) -> Self {
        let cache_cap = NonZeroUsize::new(config.cache_capacity.max(1)).unwrap();
        Self {
            graph,
            config,
            cache: Mutex::new(LruCache::new(cache_cap)),
        }
    }

    /// Extract a k-hop ego-graph around a center node.
    ///
    /// The center node is identified by its application-level ID (UUID or path).
    /// Uses Cypher variable-length path matching for efficient k-hop extraction.
    pub async fn sample_ego_graph(
        &self,
        center_id: &str,
        k_hops: Option<usize>,
    ) -> Result<SubGraph, SamplerError> {
        let k = k_hops.unwrap_or(self.config.default_k_hops);

        // Check cache
        let cache_key = format!("{}:{}", center_id, k);
        if let Some(cached) = self.cache.lock().unwrap().get(&cache_key) {
            tracing::trace!(center_id, k, "Cache hit for ego-graph");
            return Ok(cached.clone());
        }

        let rel_pattern = if self.config.relation_types.is_empty() {
            RelationType::to_cypher_pattern()
        } else {
            self.config
                .relation_types
                .iter()
                .map(|r| r.as_str())
                .collect::<Vec<_>>()
                .join("|")
        };

        // Query 1: Extract all nodes in the k-hop neighborhood
        // Uses variable-length path matching with configurable relation types
        let nodes_cypher = format!(
            "MATCH (center) WHERE center.id = $center_id OR center.path = $center_id
             WITH center
             CALL {{
                 WITH center
                 MATCH path = (center)-[:{rel_pattern}*1..{k}]-(neighbor)
                 WITH DISTINCT neighbor
                 RETURN neighbor AS n
                 LIMIT $max_nodes
                 UNION
                 WITH center
                 RETURN center AS n
             }}
             WITH DISTINCT n
             RETURN elementId(n) AS eid, n.id AS app_id, n.path AS path, labels(n) AS lbls, properties(n) AS props"
        );

        let nodes_q = query(&nodes_cypher)
            .param("center_id", center_id.to_string())
            .param("max_nodes", self.config.max_nodes_per_subgraph as i64);

        let mut result = self
            .graph
            .execute(nodes_q)
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?;

        let mut nodes: Vec<SubGraphNode> = Vec::new();
        let mut element_ids: Vec<String> = Vec::new();

        while let Some(row) = result
            .next()
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?
        {
            let eid: String = row.get("eid").unwrap_or_default();
            let app_id: String = row
                .get::<String>("app_id")
                .or_else(|_| row.get::<String>("path"))
                .unwrap_or_default();
            let labels: Vec<String> = row.get("lbls").unwrap_or_default();

            // Properties come as a Neo4j map — extract to JSON
            let props: HashMap<String, serde_json::Value> = row.get("props").unwrap_or_default();

            element_ids.push(eid.clone());
            nodes.push(SubGraphNode {
                element_id: eid,
                app_id,
                labels,
                properties: props,
            });
        }

        if nodes.is_empty() {
            return Err(SamplerError::NodeNotFound(center_id.to_string()));
        }

        // Query 2: Extract all edges between the extracted nodes
        let edges_cypher = format!(
            "UNWIND $eids AS eid1
             UNWIND $eids AS eid2
             WITH eid1, eid2 WHERE eid1 < eid2
             MATCH (a)-[r:{rel_pattern}]->(b)
             WHERE elementId(a) = eid1 AND elementId(b) = eid2
                OR elementId(a) = eid2 AND elementId(b) = eid1
             RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel_type,
                    coalesce(r.weight, r.strength, r.confidence, 1.0) AS weight"
        );

        let edges_q = query(&edges_cypher).param("eids", element_ids);

        let mut edges_result = self
            .graph
            .execute(edges_q)
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?;

        let mut edges: Vec<SubGraphEdge> = Vec::new();

        while let Some(row) = edges_result
            .next()
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?
        {
            let src: String = row.get("src").unwrap_or_default();
            let tgt: String = row.get("tgt").unwrap_or_default();
            let rel_str: String = row.get("rel_type").unwrap_or_default();
            let weight: f64 = row.get("weight").unwrap_or(1.0);

            if let Some(rel_type) = RelationType::parse(&rel_str) {
                edges.push(SubGraphEdge {
                    source_element_id: src,
                    target_element_id: tgt,
                    relation_type: rel_type,
                    weight,
                });
            }
        }

        let subgraph = SubGraph {
            center_id: center_id.to_string(),
            k_hops: k,
            nodes,
            edges,
        };

        // Cache the result
        self.cache.lock().unwrap().put(cache_key, subgraph.clone());

        tracing::debug!(
            center_id,
            k,
            nodes = subgraph.node_count(),
            edges = subgraph.edge_count(),
            "Extracted ego-graph"
        );

        Ok(subgraph)
    }

    /// Batch-extract ego-graphs for multiple center nodes.
    ///
    /// Supports stratified sampling: if `pagerank_weights` is provided,
    /// nodes with higher PageRank are overrepresented (sampled more often).
    pub async fn sample_batch(
        &self,
        center_ids: &[String],
        k_hops: Option<usize>,
    ) -> Result<Vec<SubGraph>, SamplerError> {
        let mut results = Vec::with_capacity(center_ids.len());

        for center_id in center_ids {
            match self.sample_ego_graph(center_id, k_hops).await {
                Ok(sg) => results.push(sg),
                Err(SamplerError::NodeNotFound(id)) => {
                    tracing::warn!(node_id = %id, "Node not found during batch sampling, skipping");
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        tracing::info!(
            requested = center_ids.len(),
            extracted = results.len(),
            "Batch ego-graph extraction complete"
        );

        Ok(results)
    }

    /// Sample training nodes with stratified sampling based on PageRank.
    ///
    /// Nodes with higher PageRank are sampled more frequently (they are more
    /// informative for GNN training). Returns a list of node IDs to use as
    /// ego-graph centers.
    pub async fn sample_stratified_nodes(
        &self,
        count: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<String>, SamplerError> {
        let cypher = if let Some(_slug) = project_slug {
            format!(
                "MATCH (p:Project {{slug: $slug}})-[:HAS_FILE]->(f:File)
                 WITH f, coalesce(f.pagerank, 0.0) AS pr
                 RETURN f.id AS node_id, pr
                 ORDER BY pr DESC
                 LIMIT {count}"
            )
        } else {
            format!(
                "MATCH (f:File)
                 WITH f, coalesce(f.pagerank, 0.0) AS pr
                 RETURN f.id AS node_id, pr
                 ORDER BY pr DESC
                 LIMIT {count}"
            )
        };

        let q = if let Some(slug) = project_slug {
            query(&cypher).param("slug", slug.to_string())
        } else {
            query(&cypher)
        };

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?;

        let mut node_ids = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| SamplerError::Neo4j(e.to_string()))?
        {
            let id: String = row.get("node_id").unwrap_or_default();
            if !id.is_empty() {
                node_ids.push(id);
            }
        }

        Ok(node_ids)
    }

    /// Invalidate the cache for a specific node or all entries.
    pub fn invalidate_cache(&self, center_id: Option<&str>) {
        let mut cache = self.cache.lock().unwrap();
        if let Some(id) = center_id {
            // Remove all entries for this center (any k_hops)
            let keys_to_remove: Vec<String> = cache
                .iter()
                .filter(|(k, _)| k.starts_with(&format!("{}:", id)))
                .map(|(k, _)| k.clone())
                .collect();
            for key in keys_to_remove {
                cache.pop(&key);
            }
        } else {
            cache.clear();
        }
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().get())
    }

    /// Export a subgraph to PyTorch Geometric format.
    pub fn export_pyg(&self, subgraph: &SubGraph) -> PyGData {
        export_to_pyg(subgraph)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[error("Neo4j error: {0}")]
    Neo4j(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Subgraph too large: {0} nodes (max: {1})")]
    TooLarge(usize, usize),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn test_sampler_config_default() {
        let config = GraphSamplerConfig::default();
        assert_eq!(config.default_k_hops, 3);
        assert_eq!(config.max_nodes_per_subgraph, 500);
        assert_eq!(config.cache_capacity, 256);
        assert_eq!(config.relation_types.len(), 8);
        assert!(config.include_weights);
    }
}
