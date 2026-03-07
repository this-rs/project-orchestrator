//! ReasoningTree data model
//!
//! Defines the core types for the ReasoningTree engine — a dynamic decision
//! tree that emerges from the knowledge graph in response to a query.
//!
//! - [`ReasoningTree`]: Root structure containing the full tree + metadata
//! - [`ReasoningNode`]: A node in the tree, representing an activated entity
//! - [`Action`]: A suggested MCP tool call derived from a leaf node
//! - [`EntitySource`]: The type of entity that sourced a reasoning node

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

// ============================================================================
// Enums
// ============================================================================

/// The type of knowledge graph entity that sourced a reasoning node.
///
/// Each variant corresponds to a Neo4j node type in the PO graph.
/// Used to determine which MCP tool action to suggest for leaf nodes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntitySource {
    /// A knowledge note (guideline, gotcha, pattern, etc.)
    Note,
    /// An architectural decision record
    Decision,
    /// A neural skill (emergent knowledge cluster)
    Skill,
    /// A feature graph (named subgraph of related entities)
    FeatureGraph,
    /// A source code file
    File,
    /// A function/method in the code graph
    Function,
    /// A struct/class/type in the code graph
    Struct,
    /// A trait/interface in the code graph
    Trait,
}

impl fmt::Display for EntitySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Note => write!(f, "note"),
            Self::Decision => write!(f, "decision"),
            Self::Skill => write!(f, "skill"),
            Self::FeatureGraph => write!(f, "feature_graph"),
            Self::File => write!(f, "file"),
            Self::Function => write!(f, "function"),
            Self::Struct => write!(f, "struct"),
            Self::Trait => write!(f, "trait"),
        }
    }
}

impl FromStr for EntitySource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "note" => Ok(Self::Note),
            "decision" => Ok(Self::Decision),
            "skill" => Ok(Self::Skill),
            "feature_graph" => Ok(Self::FeatureGraph),
            "file" => Ok(Self::File),
            "function" => Ok(Self::Function),
            "struct" => Ok(Self::Struct),
            "trait" => Ok(Self::Trait),
            _ => Err(format!("Unknown entity source: {s}")),
        }
    }
}

// ============================================================================
// Action
// ============================================================================

/// A suggested MCP tool call derived from a reasoning tree leaf node.
///
/// When the ReasoningTree reaches a leaf entity, it generates a concrete
/// action the agent can take. For example, a Note leaf → `note(action: "get")`,
/// a File leaf → `code(action: "get_file_symbols")`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// The MCP mega-tool name (e.g., "note", "code", "skill", "decision")
    pub tool: String,

    /// The action within the mega-tool (e.g., "get", "get_file_symbols", "activate")
    pub action: String,

    /// Pre-filled parameters for the tool call (e.g., `{"note_id": "..."}`)
    pub params: serde_json::Value,

    /// Confidence score for this suggestion (0.0 - 1.0).
    /// Derived from the leaf node's relevance score.
    pub confidence: f64,
}

// ============================================================================
// ReasoningNode
// ============================================================================

/// A node in the reasoning tree, representing an activated knowledge graph entity.
///
/// Each node carries:
/// - The identity of the source entity (type + ID)
/// - A relevance score computed by multi-factor scoring
/// - An optional suggested action (for leaf nodes)
/// - Children nodes discovered through graph traversal
/// - A human-readable reasoning explaining why this node is relevant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningNode {
    /// Unique identifier for this reasoning node (ephemeral, not persisted)
    pub id: Uuid,

    /// The type of knowledge graph entity this node represents
    pub entity_type: EntitySource,

    /// The identifier of the source entity.
    /// For notes/decisions/skills: UUID string.
    /// For files/functions/structs/traits: the path or qualified name.
    pub entity_id: String,

    /// Display label for the entity (e.g., note content preview, file name, function name)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    /// Relevance score (0.0 - 1.0+) computed by multi-factor scoring:
    /// cosine_similarity × synapse_weight × energy × recency_decay
    pub relevance: f64,

    /// Suggested MCP tool action for this node (typically set on leaf nodes)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<Action>,

    /// Child nodes discovered through graph traversal (SYNAPSE, LINKED_TO, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ReasoningNode>,

    /// Human-readable explanation of why this node is relevant.
    ///
    /// Examples:
    /// - "Note guideline linked to src/chat/manager.rs via strong SYNAPSE (weight=0.85)"
    /// - "Decision affecting src/mcp/handlers.rs with high PageRank"
    /// - "File co-changed with src/api/routes.rs (strength=0.72)"
    pub reasoning: String,

    /// Depth of this node in the tree (0 = direct seed, 1 = first hop, etc.)
    #[serde(default)]
    pub depth: usize,
}

impl ReasoningNode {
    /// Create a new reasoning node.
    pub fn new(
        entity_type: EntitySource,
        entity_id: impl Into<String>,
        relevance: f64,
        reasoning: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity_type,
            entity_id: entity_id.into(),
            label: None,
            relevance,
            action: None,
            children: Vec::new(),
            reasoning: reasoning.into(),
            depth: 0,
        }
    }

    /// Set the display label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set a suggested action.
    pub fn with_action(mut self, action: Action) -> Self {
        self.action = Some(action);
        self
    }

    /// Set the depth.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Add a child node.
    pub fn add_child(&mut self, child: ReasoningNode) {
        self.children.push(child);
    }

    /// Returns true if this node has no children (leaf node).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Count all nodes in this subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth in this subtree.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth)
        }
    }
}

// ============================================================================
// ReasoningTree
// ============================================================================

/// A reasoning tree that emerges dynamically from the knowledge graph.
///
/// The tree is built in 3 phases:
/// 1. **Activation**: Embed the request → vector search → seed nodes
/// 2. **Propagation**: Traverse SYNAPSE, LINKED_TO, AFFECTS, CO_CHANGED → score nodes
/// 3. **Cristallisation**: Transform activated subgraph into a decision tree
///
/// Trees are **ephemeral** — stored in an LRU cache with TTL, not persisted to Neo4j.
/// The feedback loop (via `reason_feedback`) reinforces the underlying synapses,
/// making future trees for similar queries more accurate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTree {
    /// Unique identifier for this tree (used for cache lookup and feedback)
    pub id: Uuid,

    /// The original natural language request that triggered tree construction
    pub request: String,

    /// Embedding vector of the request (for cache key computation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_embedding: Option<Vec<f32>>,

    /// Root nodes of the tree (one per seed from Phase 1)
    pub roots: Vec<ReasoningNode>,

    /// Maximum depth reached during propagation
    pub depth: usize,

    /// Overall confidence score (0.0 - 1.0).
    /// Weighted average of all node relevances, with depth-based weighting
    /// (closer nodes count more).
    pub confidence: f64,

    /// Total number of nodes in the tree
    pub node_count: usize,

    /// When the tree was constructed
    pub created_at: DateTime<Utc>,

    /// Optional project scope
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,

    /// Duration of tree construction in milliseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_time_ms: Option<u64>,
}

impl ReasoningTree {
    /// Create a new empty reasoning tree for a given request.
    pub fn new(request: impl Into<String>, project_id: Option<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            request: request.into(),
            request_embedding: None,
            roots: Vec::new(),
            depth: 0,
            confidence: 0.0,
            node_count: 0,
            created_at: Utc::now(),
            project_id,
            build_time_ms: None,
        }
    }

    /// Add a root node to the tree.
    pub fn add_root(&mut self, node: ReasoningNode) {
        self.roots.push(node);
        self.recompute_stats();
    }

    /// Recompute aggregate stats (depth, node_count, confidence) from the current tree.
    pub fn recompute_stats(&mut self) {
        self.node_count = self.roots.iter().map(|r| r.node_count()).sum();
        self.depth = self.roots.iter().map(|r| r.max_depth()).max().unwrap_or(0);
        self.confidence = self.compute_confidence();
    }

    /// Compute the overall confidence score.
    ///
    /// Uses a weighted average of all node relevances, where nodes closer to
    /// the root (lower depth) have higher weight: `weight = 1.0 / (depth + 1)`.
    fn compute_confidence(&self) -> f64 {
        let mut total_weighted = 0.0;
        let mut total_weight = 0.0;

        fn accumulate(node: &ReasoningNode, weighted: &mut f64, weight: &mut f64) {
            let w = 1.0 / (node.depth as f64 + 1.0);
            *weighted += node.relevance * w;
            *weight += w;
            for child in &node.children {
                accumulate(child, weighted, weight);
            }
        }

        for root in &self.roots {
            accumulate(root, &mut total_weighted, &mut total_weight);
        }

        if total_weight > 0.0 {
            (total_weighted / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Collect all suggested actions from leaf nodes, sorted by confidence descending.
    pub fn suggested_actions(&self) -> Vec<&Action> {
        let mut actions: Vec<&Action> = Vec::new();

        fn collect_actions<'a>(node: &'a ReasoningNode, actions: &mut Vec<&'a Action>) {
            if let Some(ref action) = node.action {
                actions.push(action);
            }
            for child in &node.children {
                collect_actions(child, actions);
            }
        }

        for root in &self.roots {
            collect_actions(root, &mut actions);
        }

        actions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        actions
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the ReasoningTree engine.
///
/// Controls the 3-phase build process with time budgets per phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTreeConfig {
    /// Maximum number of seed nodes from vector search (Phase 1).
    pub max_seeds: usize,

    /// Maximum depth for graph traversal (Phase 2).
    pub max_depth: usize,

    /// Minimum relevance score to keep a node (Phase 2 pruning threshold).
    pub min_relevance: f64,

    /// Maximum total nodes in the tree (prevents explosion on dense graphs).
    pub max_nodes: usize,

    /// Whether to generate Action suggestions on leaf nodes (Phase 3).
    pub include_actions: bool,

    /// Time budget for Phase 1 (activation) in milliseconds.
    pub activation_budget_ms: u64,

    /// Time budget for Phase 2 (propagation) in milliseconds.
    pub propagation_budget_ms: u64,

    /// Time budget for Phase 3 (cristallisation) in milliseconds.
    pub cristallisation_budget_ms: u64,
}

impl Default for ReasoningTreeConfig {
    fn default() -> Self {
        Self {
            max_seeds: 10,
            max_depth: 4,
            min_relevance: 0.3,
            max_nodes: 100,
            include_actions: true,
            activation_budget_ms: 50,
            propagation_budget_ms: 200,
            cristallisation_budget_ms: 100,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_source_display_and_parse() {
        let variants = [
            (EntitySource::Note, "note"),
            (EntitySource::Decision, "decision"),
            (EntitySource::Skill, "skill"),
            (EntitySource::FeatureGraph, "feature_graph"),
            (EntitySource::File, "file"),
            (EntitySource::Function, "function"),
            (EntitySource::Struct, "struct"),
            (EntitySource::Trait, "trait"),
        ];

        for (variant, expected) in &variants {
            assert_eq!(variant.to_string(), *expected);
            assert_eq!(EntitySource::from_str(expected).unwrap(), *variant);
        }

        assert!(EntitySource::from_str("unknown").is_err());
    }

    #[test]
    fn test_reasoning_node_builder() {
        let node = ReasoningNode::new(
            EntitySource::Note,
            "abc-123",
            0.85,
            "High-energy guideline note linked via SYNAPSE",
        )
        .with_label("API rate limiting guideline")
        .with_depth(1);

        assert_eq!(node.entity_type, EntitySource::Note);
        assert_eq!(node.entity_id, "abc-123");
        assert!((node.relevance - 0.85).abs() < f64::EPSILON);
        assert_eq!(node.label.as_deref(), Some("API rate limiting guideline"));
        assert_eq!(node.depth, 1);
        assert!(node.is_leaf());
        assert_eq!(node.node_count(), 1);
    }

    #[test]
    fn test_reasoning_node_with_children() {
        let child1 = ReasoningNode::new(
            EntitySource::File,
            "src/api/routes.rs",
            0.7,
            "Co-changed file",
        )
        .with_depth(1);
        let child2 = ReasoningNode::new(
            EntitySource::Function,
            "handle_request",
            0.6,
            "Called function",
        )
        .with_depth(1);

        let mut root = ReasoningNode::new(EntitySource::Note, "note-1", 0.9, "Root note");
        root.add_child(child1);
        root.add_child(child2);

        assert!(!root.is_leaf());
        assert_eq!(root.node_count(), 3);
        assert_eq!(root.max_depth(), 1);
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn test_reasoning_tree_construction() {
        let mut tree = ReasoningTree::new("How does the chat system work?", None);

        let mut root1 = ReasoningNode::new(
            EntitySource::Note,
            "note-1",
            0.9,
            "Chat architecture pattern note",
        );
        let child = ReasoningNode::new(
            EntitySource::File,
            "src/chat/manager.rs",
            0.7,
            "Linked file via LINKED_TO",
        )
        .with_depth(1);
        root1.add_child(child);

        let root2 = ReasoningNode::new(
            EntitySource::Decision,
            "dec-1",
            0.8,
            "Chat WebSocket decision",
        );

        tree.add_root(root1);
        tree.add_root(root2);

        assert_eq!(tree.node_count, 3);
        assert_eq!(tree.depth, 1);
        assert!(tree.confidence > 0.0);
        assert!(tree.confidence <= 1.0);
        assert_eq!(tree.request, "How does the chat system work?");
    }

    #[test]
    fn test_reasoning_tree_empty() {
        let tree = ReasoningTree::new("empty query", None);
        assert_eq!(tree.node_count, 0);
        assert_eq!(tree.depth, 0);
        assert!((tree.confidence - 0.0).abs() < f64::EPSILON);
        assert!(tree.suggested_actions().is_empty());
    }

    #[test]
    fn test_reasoning_tree_confidence_depth_weighting() {
        let mut tree = ReasoningTree::new("test", None);

        // Root at depth 0 with relevance 1.0 → weight 1.0
        let mut root = ReasoningNode::new(EntitySource::Note, "n1", 1.0, "root");
        // Child at depth 1 with relevance 0.0 → weight 0.5
        let child = ReasoningNode::new(EntitySource::File, "f1", 0.0, "child").with_depth(1);
        root.add_child(child);
        tree.add_root(root);

        // confidence = (1.0 * 1.0 + 0.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 ≈ 0.667
        assert!((tree.confidence - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_action_in_tree() {
        let mut tree = ReasoningTree::new("find gotchas", None);

        let action = Action {
            tool: "note".to_string(),
            action: "get".to_string(),
            params: serde_json::json!({"note_id": "abc-123"}),
            confidence: 0.85,
        };

        let root = ReasoningNode::new(EntitySource::Note, "abc-123", 0.85, "Gotcha note")
            .with_action(action);

        tree.add_root(root);

        let actions = tree.suggested_actions();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].tool, "note");
        assert_eq!(actions[0].action, "get");
        assert!((actions[0].confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serde_round_trip() {
        let mut tree = ReasoningTree::new("serialization test", None);

        let action = Action {
            tool: "code".to_string(),
            action: "get_file_symbols".to_string(),
            params: serde_json::json!({"file_path": "src/main.rs"}),
            confidence: 0.72,
        };

        let mut root = ReasoningNode::new(EntitySource::Note, "note-1", 0.95, "Top note")
            .with_label("Architecture overview");

        let child = ReasoningNode::new(EntitySource::File, "src/main.rs", 0.72, "Entry point")
            .with_depth(1)
            .with_action(action);

        let grandchild =
            ReasoningNode::new(EntitySource::Function, "main", 0.5, "Main function").with_depth(2);

        let mut child_clone = child;
        child_clone.add_child(grandchild);
        root.add_child(child_clone);
        tree.add_root(root);

        // Serialize
        let json = serde_json::to_string_pretty(&tree).expect("serialize");
        // Deserialize
        let restored: ReasoningTree = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.id, tree.id);
        assert_eq!(restored.request, "serialization test");
        assert_eq!(restored.node_count, 3);
        assert_eq!(restored.depth, 2);
        assert_eq!(restored.roots.len(), 1);
        assert_eq!(restored.roots[0].children.len(), 1);
        assert_eq!(restored.roots[0].children[0].children.len(), 1);
        assert!(restored.roots[0].children[0].action.is_some());
    }

    #[test]
    fn test_config_defaults() {
        let config = ReasoningTreeConfig::default();
        assert_eq!(config.max_seeds, 10);
        assert_eq!(config.max_depth, 4);
        assert!((config.min_relevance - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.max_nodes, 100);
        assert!(config.include_actions);
        assert_eq!(config.activation_budget_ms, 50);
        assert_eq!(config.propagation_budget_ms, 200);
        assert_eq!(config.cristallisation_budget_ms, 100);
    }

    #[test]
    fn test_suggested_actions_sorted_by_confidence() {
        let mut tree = ReasoningTree::new("multi-action test", None);

        let low_action = Action {
            tool: "code".to_string(),
            action: "search".to_string(),
            params: serde_json::json!({}),
            confidence: 0.3,
        };
        let high_action = Action {
            tool: "note".to_string(),
            action: "get".to_string(),
            params: serde_json::json!({}),
            confidence: 0.9,
        };
        let mid_action = Action {
            tool: "decision".to_string(),
            action: "get".to_string(),
            params: serde_json::json!({}),
            confidence: 0.6,
        };

        tree.add_root(
            ReasoningNode::new(EntitySource::File, "f1", 0.3, "low").with_action(low_action),
        );
        tree.add_root(
            ReasoningNode::new(EntitySource::Note, "n1", 0.9, "high").with_action(high_action),
        );
        tree.add_root(
            ReasoningNode::new(EntitySource::Decision, "d1", 0.6, "mid").with_action(mid_action),
        );

        let actions = tree.suggested_actions();
        assert_eq!(actions.len(), 3);
        assert!((actions[0].confidence - 0.9).abs() < f64::EPSILON);
        assert!((actions[1].confidence - 0.6).abs() < f64::EPSILON);
        assert!((actions[2].confidence - 0.3).abs() < f64::EPSILON);
    }
}
