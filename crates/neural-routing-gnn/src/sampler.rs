//! GraphSampler — k-hop ego-graph extraction from Neo4j for GNN training.
//!
//! Extracts multi-relational subgraphs around target nodes, supporting:
//! - 8 relation types: IMPORTS, CALLS, CO_CHANGED, AFFECTS, SYNAPSE, KNOWS, CONTAINS, NEXT_DECISION
//! - PyTorch Geometric export: (edge_index, node_features, edge_types)
//! - Batch extraction with stratified sampling (overrepresent high-PageRank nodes)
//! - LRU cache for frequently accessed subgraphs

use std::collections::HashMap;
use std::sync::Mutex;

use lru::LruCache;
use neo4rs::{query, Graph};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Relation type encoding
// ---------------------------------------------------------------------------

/// All relation types tracked in the knowledge graph for GNN training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RelationType {
    Imports = 0,
    Calls = 1,
    CoChanged = 2,
    Affects = 3,
    Synapse = 4,
    Knows = 5,
    Contains = 6,
    NextDecision = 7,
}

impl RelationType {
    pub const ALL: [RelationType; 8] = [
        Self::Imports,
        Self::Calls,
        Self::CoChanged,
        Self::Affects,
        Self::Synapse,
        Self::Knows,
        Self::Contains,
        Self::NextDecision,
    ];

    pub fn count() -> usize {
        8
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "IMPORTS" => Some(Self::Imports),
            "CALLS" => Some(Self::Calls),
            "CO_CHANGED" => Some(Self::CoChanged),
            "AFFECTS" => Some(Self::Affects),
            "SYNAPSE" => Some(Self::Synapse),
            "KNOWS" => Some(Self::Knows),
            "CONTAINS" => Some(Self::Contains),
            "NEXT_DECISION" => Some(Self::NextDecision),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Imports => "IMPORTS",
            Self::Calls => "CALLS",
            Self::CoChanged => "CO_CHANGED",
            Self::Affects => "AFFECTS",
            Self::Synapse => "SYNAPSE",
            Self::Knows => "KNOWS",
            Self::Contains => "CONTAINS",
            Self::NextDecision => "NEXT_DECISION",
        }
    }

    pub fn to_cypher_pattern() -> String {
        Self::ALL
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join("|")
    }
}

// ---------------------------------------------------------------------------
// SubGraph data model
// ---------------------------------------------------------------------------

/// A node in the extracted subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGraphNode {
    /// Neo4j internal element ID (used for edge resolution).
    pub element_id: String,
    /// Application-level ID if available (UUID string or file path).
    pub app_id: String,
    /// Node labels from Neo4j (e.g., ["File"], ["Function", "TrajectoryNode"]).
    pub labels: Vec<String>,
    /// Raw properties as key-value pairs (features will be extracted from these).
    pub properties: HashMap<String, serde_json::Value>,
}

/// An edge in the extracted subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGraphEdge {
    /// Source node element ID.
    pub source_element_id: String,
    /// Target node element ID.
    pub target_element_id: String,
    /// Relation type.
    pub relation_type: RelationType,
    /// Edge weight (if available, e.g., CO_CHANGED strength, SYNAPSE weight).
    pub weight: f64,
}

/// An extracted ego-graph subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGraph {
    /// Center node of the ego-graph.
    pub center_id: String,
    /// Number of hops used for extraction.
    pub k_hops: usize,
    /// All nodes in the subgraph.
    pub nodes: Vec<SubGraphNode>,
    /// All edges in the subgraph.
    pub edges: Vec<SubGraphEdge>,
}

impl SubGraph {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ---------------------------------------------------------------------------
// PyTorch Geometric export format
// ---------------------------------------------------------------------------

/// PyTorch Geometric compatible data format.
///
/// Maps directly to `torch_geometric.data.Data`:
/// - `edge_index`: [2, num_edges] — source/target indices
/// - `edge_type`: [num_edges] — relation type IDs (0-7)
/// - `edge_weight`: [num_edges] — edge weights
/// - `node_labels`: label encoding per node
/// - `node_ids`: original IDs for reverse mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyGData {
    /// Edge index as [sources, targets], each of length num_edges.
    pub edge_index: [Vec<usize>; 2],
    /// Edge type IDs (0-7 mapping to RelationType).
    pub edge_type: Vec<u8>,
    /// Edge weights.
    pub edge_weight: Vec<f64>,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of edges.
    pub num_edges: usize,
    /// Node label encoding (index into label vocabulary).
    pub node_label_ids: Vec<u8>,
    /// Original node IDs for reverse mapping.
    pub node_ids: Vec<String>,
    /// Label vocabulary.
    pub label_vocab: Vec<String>,
}

/// Convert a SubGraph to PyTorch Geometric format.
pub fn export_to_pyg(subgraph: &SubGraph) -> PyGData {
    // Build node index mapping: element_id -> sequential index
    let mut node_index: HashMap<&str, usize> = HashMap::new();
    for (i, node) in subgraph.nodes.iter().enumerate() {
        node_index.insert(&node.element_id, i);
    }

    // Build label vocabulary
    let mut label_vocab: Vec<String> = Vec::new();
    let mut label_map: HashMap<String, u8> = HashMap::new();
    let mut node_label_ids: Vec<u8> = Vec::new();

    for node in &subgraph.nodes {
        // Use the primary label (first in the list)
        let primary_label = node.labels.first().cloned().unwrap_or_default();
        let label_id = if let Some(&id) = label_map.get(&primary_label) {
            id
        } else {
            let id = label_vocab.len() as u8;
            label_vocab.push(primary_label.clone());
            label_map.insert(primary_label, id);
            id
        };
        node_label_ids.push(label_id);
    }

    // Build edge index + types
    let mut sources: Vec<usize> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();
    let mut edge_types: Vec<u8> = Vec::new();
    let mut edge_weights: Vec<f64> = Vec::new();

    for edge in &subgraph.edges {
        if let (Some(&src_idx), Some(&tgt_idx)) = (
            node_index.get(edge.source_element_id.as_str()),
            node_index.get(edge.target_element_id.as_str()),
        ) {
            sources.push(src_idx);
            targets.push(tgt_idx);
            edge_types.push(edge.relation_type as u8);
            edge_weights.push(edge.weight);
        }
    }

    let num_edges = sources.len();
    let node_ids: Vec<String> = subgraph.nodes.iter().map(|n| n.app_id.clone()).collect();

    PyGData {
        edge_index: [sources, targets],
        edge_type: edge_types,
        edge_weight: edge_weights,
        num_nodes: subgraph.nodes.len(),
        num_edges,
        node_label_ids,
        node_ids,
        label_vocab,
    }
}

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
            let props: HashMap<String, serde_json::Value> =
                row.get("props").unwrap_or_default();

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
        self.cache
            .lock()
            .unwrap()
            .put(cache_key, subgraph.clone());

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

    fn make_test_subgraph() -> SubGraph {
        SubGraph {
            center_id: "file-1".to_string(),
            k_hops: 2,
            nodes: vec![
                SubGraphNode {
                    element_id: "e:0".to_string(),
                    app_id: "file-1".to_string(),
                    labels: vec!["File".to_string()],
                    properties: HashMap::new(),
                },
                SubGraphNode {
                    element_id: "e:1".to_string(),
                    app_id: "file-2".to_string(),
                    labels: vec!["File".to_string()],
                    properties: HashMap::new(),
                },
                SubGraphNode {
                    element_id: "e:2".to_string(),
                    app_id: "func-1".to_string(),
                    labels: vec!["Function".to_string()],
                    properties: HashMap::new(),
                },
                SubGraphNode {
                    element_id: "e:3".to_string(),
                    app_id: "note-1".to_string(),
                    labels: vec!["Note".to_string()],
                    properties: HashMap::new(),
                },
            ],
            edges: vec![
                SubGraphEdge {
                    source_element_id: "e:0".to_string(),
                    target_element_id: "e:1".to_string(),
                    relation_type: RelationType::Imports,
                    weight: 1.0,
                },
                SubGraphEdge {
                    source_element_id: "e:0".to_string(),
                    target_element_id: "e:2".to_string(),
                    relation_type: RelationType::Contains,
                    weight: 1.0,
                },
                SubGraphEdge {
                    source_element_id: "e:1".to_string(),
                    target_element_id: "e:2".to_string(),
                    relation_type: RelationType::Calls,
                    weight: 0.9,
                },
                SubGraphEdge {
                    source_element_id: "e:2".to_string(),
                    target_element_id: "e:3".to_string(),
                    relation_type: RelationType::Synapse,
                    weight: 0.75,
                },
                SubGraphEdge {
                    source_element_id: "e:0".to_string(),
                    target_element_id: "e:1".to_string(),
                    relation_type: RelationType::CoChanged,
                    weight: 0.6,
                },
            ],
        }
    }

    #[test]
    fn test_subgraph_counts() {
        let sg = make_test_subgraph();
        assert_eq!(sg.node_count(), 4);
        assert_eq!(sg.edge_count(), 5);
    }

    #[test]
    fn test_export_pyg_edge_index() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        assert_eq!(pyg.num_nodes, 4);
        assert_eq!(pyg.num_edges, 5);
        assert_eq!(pyg.edge_index[0].len(), 5);
        assert_eq!(pyg.edge_index[1].len(), 5);
        assert_eq!(pyg.edge_type.len(), 5);
        assert_eq!(pyg.edge_weight.len(), 5);
    }

    #[test]
    fn test_export_pyg_edge_types() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        // First edge: IMPORTS (0)
        assert_eq!(pyg.edge_type[0], RelationType::Imports as u8);
        // Second edge: CONTAINS (6)
        assert_eq!(pyg.edge_type[1], RelationType::Contains as u8);
        // Third edge: CALLS (1)
        assert_eq!(pyg.edge_type[2], RelationType::Calls as u8);
        // Fourth edge: SYNAPSE (4)
        assert_eq!(pyg.edge_type[3], RelationType::Synapse as u8);
        // Fifth edge: CO_CHANGED (2)
        assert_eq!(pyg.edge_type[4], RelationType::CoChanged as u8);
    }

    #[test]
    fn test_export_pyg_node_labels() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        // 3 unique labels: File, Function, Note
        assert_eq!(pyg.label_vocab.len(), 3);
        assert!(pyg.label_vocab.contains(&"File".to_string()));
        assert!(pyg.label_vocab.contains(&"Function".to_string()));
        assert!(pyg.label_vocab.contains(&"Note".to_string()));

        // First two nodes are Files (same label ID)
        assert_eq!(pyg.node_label_ids[0], pyg.node_label_ids[1]);
        // Third node is Function (different label ID)
        assert_ne!(pyg.node_label_ids[0], pyg.node_label_ids[2]);
    }

    #[test]
    fn test_export_pyg_node_ids() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        assert_eq!(pyg.node_ids, vec!["file-1", "file-2", "func-1", "note-1"]);
    }

    #[test]
    fn test_export_pyg_edge_indices_valid() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        // All source/target indices should be within [0, num_nodes)
        for &src in &pyg.edge_index[0] {
            assert!(src < pyg.num_nodes, "Source index {} out of bounds", src);
        }
        for &tgt in &pyg.edge_index[1] {
            assert!(tgt < pyg.num_nodes, "Target index {} out of bounds", tgt);
        }
    }

    #[test]
    fn test_export_pyg_weights() {
        let sg = make_test_subgraph();
        let pyg = export_to_pyg(&sg);

        assert!((pyg.edge_weight[0] - 1.0).abs() < f64::EPSILON); // IMPORTS
        assert!((pyg.edge_weight[2] - 0.9).abs() < f64::EPSILON); // CALLS
        assert!((pyg.edge_weight[3] - 0.75).abs() < f64::EPSILON); // SYNAPSE
    }

    #[test]
    fn test_export_empty_subgraph() {
        let sg = SubGraph {
            center_id: "empty".to_string(),
            k_hops: 1,
            nodes: vec![],
            edges: vec![],
        };
        let pyg = export_to_pyg(&sg);

        assert_eq!(pyg.num_nodes, 0);
        assert_eq!(pyg.num_edges, 0);
        assert!(pyg.edge_index[0].is_empty());
        assert!(pyg.edge_index[1].is_empty());
    }

    #[test]
    fn test_export_pyg_dangling_edge_ignored() {
        // Edge referencing a node not in the subgraph should be ignored
        let sg = SubGraph {
            center_id: "a".to_string(),
            k_hops: 1,
            nodes: vec![SubGraphNode {
                element_id: "e:0".to_string(),
                app_id: "a".to_string(),
                labels: vec!["File".to_string()],
                properties: HashMap::new(),
            }],
            edges: vec![SubGraphEdge {
                source_element_id: "e:0".to_string(),
                target_element_id: "e:999".to_string(), // doesn't exist
                relation_type: RelationType::Imports,
                weight: 1.0,
            }],
        };
        let pyg = export_to_pyg(&sg);

        // Dangling edge should be skipped
        assert_eq!(pyg.num_edges, 0);
        assert_eq!(pyg.num_nodes, 1);
    }

    #[test]
    fn test_relation_type_roundtrip() {
        for rt in RelationType::ALL {
            let s = rt.as_str();
            let parsed = RelationType::parse(s).unwrap();
            assert_eq!(rt, parsed);
        }
    }

    #[test]
    fn test_relation_type_count() {
        assert_eq!(RelationType::count(), 8);
        assert_eq!(RelationType::ALL.len(), 8);
    }

    #[test]
    fn test_cypher_pattern() {
        let pattern = RelationType::to_cypher_pattern();
        assert!(pattern.contains("IMPORTS"));
        assert!(pattern.contains("CALLS"));
        assert!(pattern.contains("NEXT_DECISION"));
        assert_eq!(pattern.matches('|').count(), 7); // 8 types, 7 separators
    }

    #[test]
    fn test_relation_type_from_str_unknown() {
        assert!(RelationType::parse("UNKNOWN").is_none());
        assert!(RelationType::parse("").is_none());
    }

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
