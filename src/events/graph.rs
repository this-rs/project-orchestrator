//! Graph mutation events for real-time visualization
//!
//! These events complement `CrudEvent`s by providing fine-grained,
//! graph-specific mutation notifications for the Intelligence Visualization
//! frontend. They flow through the same `HybridEmitter` (via a dedicated
//! broadcast channel) and are multiplexed into the WebSocket stream.
//!
//! Each event carries a `kind: "graph"` field so the frontend can distinguish
//! them from CRUD events in the same WebSocket connection.

use serde::{Deserialize, Serialize};

/// The 6 intelligence layers for graph visualization.
///
/// Used to categorize graph events and allow the frontend to subscribe
/// to a subset of layers (e.g. only "neural" + "knowledge").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphLayer {
    /// Code structure: files, functions, structs, imports, calls
    Code,
    /// Knowledge: notes, decisions, guidelines
    Knowledge,
    /// Fabric: CO_CHANGED, TOUCHES, AFFECTS relations
    Fabric,
    /// Neural: synapses, energy, spreading activation
    Neural,
    /// Skills: emergent knowledge clusters
    Skills,
    /// Project management: plans, tasks, milestones
    Pm,
}

/// Graph mutation event types.
///
/// Each type represents a specific kind of graph mutation that the
/// frontend should react to (animate, add/remove node, update style, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEventType {
    /// A new node was added to the graph (file synced, note created, etc.)
    NodeCreated,
    /// A node's attributes changed (energy, staleness, status, etc.)
    NodeUpdated,
    /// A new edge was created (synapse, LINKED_TO, CO_CHANGED, etc.)
    EdgeCreated,
    /// An edge was removed (synapse pruned, note unlinked, etc.)
    EdgeRemoved,
    /// Hebbian reinforcement event (energy boost on notes/synapses)
    Reinforcement,
    /// Spreading activation event (energy propagation across synapses)
    Activation,
    /// Community structure changed (Louvain recomputation)
    CommunityChanged,
}

/// A graph mutation event for real-time visualization updates.
///
/// Sent to WebSocket clients alongside `CrudEvent`s. The `kind: "graph"` field
/// allows the frontend to distinguish graph events from CRUD events in the same
/// WebSocket stream.
///
/// JSON shape:
/// ```json
/// {
///   "kind": "graph",
///   "type": "node_created",
///   "layer": "knowledge",
///   "node_id": "uuid-...",
///   "delta": {"note_type": "gotcha", "importance": "high"},
///   "project_id": "uuid-...",
///   "timestamp": "2026-03-05T..."
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEvent {
    /// Always `"graph"` — distinguishes from CrudEvent in the WS stream
    pub kind: String,
    /// The graph event type
    #[serde(rename = "type")]
    pub event_type: GraphEventType,
    /// The intelligence layer this event belongs to
    pub layer: GraphLayer,
    /// Primary node ID (node events) or source node ID (edge events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// Target node ID (edge events only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    /// Edge/relation type (for edge events: "SYNAPSE", "LINKED_TO", "CO_CHANGED", etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// Changed attributes or event-specific data
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub delta: serde_json::Value,
    /// Project scope (graph events are always project-scoped)
    pub project_id: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
}

impl GraphEvent {
    /// Create a node event (node_created, node_updated).
    pub fn node(
        event_type: GraphEventType,
        layer: GraphLayer,
        node_id: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: "graph".into(),
            event_type,
            layer,
            node_id: Some(node_id.into()),
            target_id: None,
            edge_type: None,
            delta: serde_json::Value::Null,
            project_id: project_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create an edge event (edge_created, edge_removed).
    pub fn edge(
        event_type: GraphEventType,
        layer: GraphLayer,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        edge_type: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: "graph".into(),
            event_type,
            layer,
            node_id: Some(source_id.into()),
            target_id: Some(target_id.into()),
            edge_type: Some(edge_type.into()),
            delta: serde_json::Value::Null,
            project_id: project_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a reinforcement event (energy boost on a node).
    pub fn reinforcement(
        node_id: impl Into<String>,
        energy_delta: f64,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: "graph".into(),
            event_type: GraphEventType::Reinforcement,
            layer: GraphLayer::Neural,
            node_id: Some(node_id.into()),
            target_id: None,
            edge_type: None,
            delta: serde_json::json!({ "energy_delta": energy_delta }),
            project_id: project_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create an activation event (spreading activation across synapses).
    pub fn activation(
        source_id: impl Into<String>,
        targets: Vec<ActivationTarget>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: "graph".into(),
            event_type: GraphEventType::Activation,
            layer: GraphLayer::Neural,
            node_id: Some(source_id.into()),
            target_id: None,
            edge_type: None,
            delta: serde_json::to_value(&targets).unwrap_or_default(),
            project_id: project_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a community_changed event.
    pub fn community_changed(
        community_id: impl Into<String>,
        member_count: usize,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: "graph".into(),
            event_type: GraphEventType::CommunityChanged,
            layer: GraphLayer::Skills,
            node_id: Some(community_id.into()),
            target_id: None,
            edge_type: None,
            delta: serde_json::json!({ "member_count": member_count }),
            project_id: project_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Set the delta payload.
    pub fn with_delta(mut self, delta: serde_json::Value) -> Self {
        self.delta = delta;
        self
    }
}

/// A target of spreading activation, used in the `delta` of `Activation` events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationTarget {
    /// The target note ID
    pub note_id: String,
    /// Energy received by this target
    pub energy_received: f64,
    /// Synapse weight used for propagation
    pub synapse_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_layer_serde_roundtrip() {
        let variants = vec![
            GraphLayer::Code,
            GraphLayer::Knowledge,
            GraphLayer::Fabric,
            GraphLayer::Neural,
            GraphLayer::Skills,
            GraphLayer::Pm,
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: GraphLayer = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, &deserialized);
        }

        // Verify snake_case
        assert_eq!(
            serde_json::to_string(&GraphLayer::Code).unwrap(),
            "\"code\""
        );
        assert_eq!(serde_json::to_string(&GraphLayer::Pm).unwrap(), "\"pm\"");
    }

    #[test]
    fn test_graph_event_type_serde_roundtrip() {
        let variants = vec![
            GraphEventType::NodeCreated,
            GraphEventType::NodeUpdated,
            GraphEventType::EdgeCreated,
            GraphEventType::EdgeRemoved,
            GraphEventType::Reinforcement,
            GraphEventType::Activation,
            GraphEventType::CommunityChanged,
        ];
        assert_eq!(variants.len(), 7);

        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: GraphEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, &deserialized);
        }

        // Verify snake_case
        assert_eq!(
            serde_json::to_string(&GraphEventType::NodeCreated).unwrap(),
            "\"node_created\""
        );
        assert_eq!(
            serde_json::to_string(&GraphEventType::CommunityChanged).unwrap(),
            "\"community_changed\""
        );
    }

    #[test]
    fn test_graph_event_node_constructor() {
        let event = GraphEvent::node(
            GraphEventType::NodeCreated,
            GraphLayer::Knowledge,
            "note-123",
            "proj-456",
        )
        .with_delta(serde_json::json!({"note_type": "gotcha", "importance": "high"}));

        assert_eq!(event.kind, "graph");
        assert_eq!(event.event_type, GraphEventType::NodeCreated);
        assert_eq!(event.layer, GraphLayer::Knowledge);
        assert_eq!(event.node_id.as_deref(), Some("note-123"));
        assert!(event.target_id.is_none());
        assert!(event.edge_type.is_none());
        assert_eq!(event.delta["note_type"], "gotcha");
        assert_eq!(event.project_id, "proj-456");
        assert!(!event.timestamp.is_empty());
    }

    #[test]
    fn test_graph_event_edge_constructor() {
        let event = GraphEvent::edge(
            GraphEventType::EdgeCreated,
            GraphLayer::Neural,
            "note-1",
            "note-2",
            "SYNAPSE",
            "proj-789",
        )
        .with_delta(serde_json::json!({"weight": 0.85}));

        assert_eq!(event.event_type, GraphEventType::EdgeCreated);
        assert_eq!(event.layer, GraphLayer::Neural);
        assert_eq!(event.node_id.as_deref(), Some("note-1"));
        assert_eq!(event.target_id.as_deref(), Some("note-2"));
        assert_eq!(event.edge_type.as_deref(), Some("SYNAPSE"));
        assert_eq!(event.delta["weight"], 0.85);
    }

    #[test]
    fn test_graph_event_reinforcement_constructor() {
        let event = GraphEvent::reinforcement("note-1", 0.15, "proj-1");

        assert_eq!(event.event_type, GraphEventType::Reinforcement);
        assert_eq!(event.layer, GraphLayer::Neural);
        assert_eq!(event.node_id.as_deref(), Some("note-1"));
        assert_eq!(event.delta["energy_delta"], 0.15);
    }

    #[test]
    fn test_graph_event_activation_constructor() {
        let targets = vec![
            ActivationTarget {
                note_id: "note-2".into(),
                energy_received: 0.05,
                synapse_weight: 0.7,
            },
            ActivationTarget {
                note_id: "note-3".into(),
                energy_received: 0.03,
                synapse_weight: 0.4,
            },
        ];
        let event = GraphEvent::activation("note-1", targets, "proj-1");

        assert_eq!(event.event_type, GraphEventType::Activation);
        assert_eq!(event.layer, GraphLayer::Neural);
        assert_eq!(event.node_id.as_deref(), Some("note-1"));
        // delta should be a JSON array of targets
        let arr = event.delta.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["note_id"], "note-2");
    }

    #[test]
    fn test_graph_event_community_changed_constructor() {
        let event = GraphEvent::community_changed("community-1", 12, "proj-1");

        assert_eq!(event.event_type, GraphEventType::CommunityChanged);
        assert_eq!(event.layer, GraphLayer::Skills);
        assert_eq!(event.delta["member_count"], 12);
    }

    #[test]
    fn test_graph_event_json_shape() {
        let event = GraphEvent::node(
            GraphEventType::NodeUpdated,
            GraphLayer::Neural,
            "note-1",
            "proj-1",
        )
        .with_delta(serde_json::json!({"energy": 0.8}));

        let json = serde_json::to_string(&event).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "graph");
        assert_eq!(parsed["type"], "node_updated");
        assert_eq!(parsed["layer"], "neural");
        assert_eq!(parsed["node_id"], "note-1");
        assert_eq!(parsed["project_id"], "proj-1");
        assert_eq!(parsed["delta"]["energy"], 0.8);
        // target_id and edge_type should be absent (not null)
        assert!(parsed.get("target_id").is_none());
        assert!(parsed.get("edge_type").is_none());
    }

    #[test]
    fn test_graph_event_null_delta_omitted() {
        let event = GraphEvent::node(
            GraphEventType::NodeCreated,
            GraphLayer::Code,
            "file-1",
            "proj-1",
        );

        let json = serde_json::to_string(&event).unwrap();
        // Null delta should be omitted
        assert!(!json.contains("\"delta\""));
    }

    #[test]
    fn test_graph_event_clone_for_broadcast() {
        let event = GraphEvent::reinforcement("note-1", 0.2, "proj-1");
        let cloned = event.clone();

        assert_eq!(cloned.event_type, event.event_type);
        assert_eq!(cloned.node_id, event.node_id);
        assert_eq!(cloned.project_id, event.project_id);
    }
}
