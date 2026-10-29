//! Subgraph data model and PyTorch Geometric export for GNN training.
//!
//! This module carries the *shape* of an extracted subgraph — the relation
//! encoding, the node/edge records, and the conversion to the format the GNN
//! models consume. The actual k-hop graph extraction (the `GraphSampler`)
//! reads the graph driver and therefore lives in the storage adapter, not here.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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
}
