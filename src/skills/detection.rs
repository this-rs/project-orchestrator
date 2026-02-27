//! Skill Detection Engine — Louvain community detection on the SYNAPSE graph.
//!
//! Detects emergent neural skill clusters by analyzing the SYNAPSE (Note↔Note)
//! subgraph using the Louvain algorithm. Each detected community of strongly
//! connected notes becomes a candidate skill.

use crate::graph::algorithms::{compute_cohesion, louvain_communities};
use crate::graph::{
    AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeNode, CodeNodeType, CommunityInfo,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the skill detection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDetectionConfig {
    /// Minimum SYNAPSE weight to include in the graph (default: 0.1)
    pub min_synapse_weight: f64,
    /// Minimum number of notes in a cluster to become a skill (default: 3)
    pub min_cluster_size: usize,
    /// Minimum cohesion score for a cluster to become a skill (default: 0.3)
    pub min_cohesion: f64,
    /// Louvain resolution parameter — higher = smaller communities (default: 1.5)
    pub louvain_resolution: f64,
    /// Overlap threshold for deduplication with existing skills (default: 0.7)
    pub overlap_threshold: f64,
    /// Minimum number of notes with synapses required before detection (default: 15)
    pub min_notes_for_detection: usize,
}

impl Default for SkillDetectionConfig {
    fn default() -> Self {
        Self {
            min_synapse_weight: 0.1,
            min_cluster_size: 3,
            min_cohesion: 0.3,
            louvain_resolution: 1.5,
            overlap_threshold: 0.7,
            min_notes_for_detection: 15,
        }
    }
}

// ============================================================================
// Detection Result
// ============================================================================

/// Status of the detection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterDetectionStatus {
    /// Detection completed successfully.
    Success,
    /// Not enough data to run detection.
    InsufficientData,
}

/// Result of the skill detection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterDetectionResult {
    /// Overall status of detection.
    pub status: ClusterDetectionStatus,
    /// Detected skill candidates (empty if insufficient data).
    pub candidates: Vec<SkillCandidate>,
    /// Total number of notes in the SYNAPSE graph.
    pub total_notes: usize,
    /// Total number of SYNAPSE edges used.
    pub total_synapses: usize,
    /// Global modularity score from Louvain.
    pub modularity: f64,
    /// Human-readable message.
    pub message: String,
}

/// A candidate skill detected by Louvain community detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCandidate {
    /// Community ID from Louvain.
    pub community_id: u32,
    /// Note IDs belonging to this cluster.
    pub member_note_ids: Vec<String>,
    /// Cohesion score (0.0 - 1.0).
    pub cohesion: f64,
    /// Number of notes in the cluster.
    pub size: usize,
    /// Auto-generated label from common path prefix heuristic.
    pub label: String,
}

// ============================================================================
// Graph Construction
// ============================================================================

/// Build a CodeGraph from SYNAPSE edges for Louvain community detection.
///
/// Each note becomes a node (CodeNodeType::Function as proxy), and each
/// SYNAPSE edge becomes a weighted edge. The graph is ready for `louvain_communities()`.
pub fn build_synapse_graph(
    edges: &[(String, String, f64)],
    project_id: &str,
) -> CodeGraph {
    // Collect unique note IDs
    let mut note_ids: Vec<&str> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (from, to, _) in edges {
        if seen.insert(from.as_str()) {
            note_ids.push(from.as_str());
        }
        if seen.insert(to.as_str()) {
            note_ids.push(to.as_str());
        }
    }

    let mut graph = CodeGraph::with_capacity(note_ids.len(), edges.len());

    // Add note nodes (using Function as proxy since CodeNodeType has no Note variant)
    for note_id in &note_ids {
        graph.add_node(CodeNode {
            id: note_id.to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: note_id.to_string(),
            project_id: Some(project_id.to_string()),
        });
    }

    // Add SYNAPSE edges
    for (from, to, weight) in edges {
        graph.add_edge(
            from,
            to,
            CodeEdge {
                edge_type: CodeEdgeType::Synapse,
                weight: *weight,
            },
        );
    }

    graph
}

// ============================================================================
// Detection Pipeline
// ============================================================================

/// Run the Louvain-based skill detection pipeline on SYNAPSE edges.
///
/// Returns detected skill candidates filtered by minimum cluster size and cohesion.
pub fn detect_skill_candidates(
    edges: &[(String, String, f64)],
    project_id: &str,
    config: &SkillDetectionConfig,
) -> ClusterDetectionResult {
    // Collect unique note IDs to check threshold
    let mut unique_notes = std::collections::HashSet::new();
    for (from, to, _) in edges {
        unique_notes.insert(from.as_str());
        unique_notes.insert(to.as_str());
    }
    let total_notes = unique_notes.len();
    let total_synapses = edges.len();

    // Cold start guard: not enough data
    if total_notes < config.min_notes_for_detection {
        let suggestion = if total_notes >= 10 {
            " Try running admin(action: 'backfill_synapses') to accelerate synapse creation."
        } else {
            " Continue creating notes and synapses will form naturally."
        };

        return ClusterDetectionResult {
            status: ClusterDetectionStatus::InsufficientData,
            candidates: Vec::new(),
            total_notes,
            total_synapses,
            modularity: 0.0,
            message: format!(
                "Project has {} notes with synapses. Minimum required: {}.{}",
                total_notes, config.min_notes_for_detection, suggestion
            ),
        };
    }

    // Build the graph
    let graph = build_synapse_graph(edges, project_id);

    if graph.node_count() == 0 {
        return ClusterDetectionResult {
            status: ClusterDetectionStatus::InsufficientData,
            candidates: Vec::new(),
            total_notes: 0,
            total_synapses: 0,
            modularity: 0.0,
            message: "No notes found in SYNAPSE graph.".to_string(),
        };
    }

    // Configure Louvain
    let analytics_config = AnalyticsConfig {
        louvain_resolution: config.louvain_resolution,
        ..Default::default()
    };

    // Run Louvain community detection
    let (node_to_community, communities, modularity) =
        louvain_communities(&graph, &analytics_config);

    // Compute cohesion per community
    let cohesion_map = compute_cohesion(&graph, &communities, &node_to_community);

    // Filter communities: apply min_cluster_size and min_cohesion
    let candidates: Vec<SkillCandidate> = communities
        .into_iter()
        .filter_map(|community| {
            let cohesion = cohesion_map.get(&community.id).copied().unwrap_or(0.0);

            if community.size < config.min_cluster_size {
                return None;
            }
            if cohesion < config.min_cohesion {
                return None;
            }

            Some(SkillCandidate {
                community_id: community.id,
                member_note_ids: community.members.clone(),
                cohesion,
                size: community.size,
                label: community.label.clone(),
            })
        })
        .collect();

    let n_candidates = candidates.len();
    ClusterDetectionResult {
        status: ClusterDetectionStatus::Success,
        candidates,
        total_notes,
        total_synapses,
        modularity,
        message: format!(
            "Detected {} skill candidates from {} notes and {} synapses (modularity: {:.3})",
            n_candidates, total_notes, total_synapses, modularity
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a set of SYNAPSE edges forming 3 distinct clusters.
    fn make_3_clusters() -> Vec<(String, String, f64)> {
        let mut edges = Vec::new();

        // Cluster A: notes 1-5 (fully connected)
        for i in 1..=5 {
            for j in (i + 1)..=5 {
                edges.push((format!("note-{}", i), format!("note-{}", j), 0.9));
            }
        }

        // Cluster B: notes 6-10 (fully connected)
        for i in 6..=10 {
            for j in (i + 1)..=10 {
                edges.push((format!("note-{}", i), format!("note-{}", j), 0.85));
            }
        }

        // Cluster C: notes 11-15 (fully connected)
        for i in 11..=15 {
            for j in (i + 1)..=15 {
                edges.push((format!("note-{}", i), format!("note-{}", j), 0.8));
            }
        }

        // Weak inter-cluster links
        edges.push(("note-1".to_string(), "note-6".to_string(), 0.15));
        edges.push(("note-6".to_string(), "note-11".to_string(), 0.12));

        edges
    }

    #[test]
    fn test_build_synapse_graph() {
        let edges = vec![
            ("a".to_string(), "b".to_string(), 0.5),
            ("b".to_string(), "c".to_string(), 0.7),
            ("a".to_string(), "c".to_string(), 0.3),
        ];

        let graph = build_synapse_graph(&edges, "proj-1");
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_build_synapse_graph_deduplicates_nodes() {
        let edges = vec![
            ("a".to_string(), "b".to_string(), 0.5),
            ("a".to_string(), "c".to_string(), 0.3),
            ("b".to_string(), "c".to_string(), 0.7),
        ];

        let graph = build_synapse_graph(&edges, "proj-1");
        assert_eq!(graph.node_count(), 3); // a, b, c — no duplicates
    }

    #[test]
    fn test_detect_insufficient_data() {
        // Only 5 notes — below threshold of 15
        let edges = vec![
            ("n1".to_string(), "n2".to_string(), 0.5),
            ("n2".to_string(), "n3".to_string(), 0.6),
            ("n3".to_string(), "n4".to_string(), 0.7),
            ("n4".to_string(), "n5".to_string(), 0.8),
        ];
        let config = SkillDetectionConfig::default();

        let result = detect_skill_candidates(&edges, "proj-1", &config);
        assert_eq!(result.status, ClusterDetectionStatus::InsufficientData);
        assert!(result.candidates.is_empty());
        assert!(result.message.contains("Minimum required: 15"));
    }

    #[test]
    fn test_detect_3_clusters() {
        let edges = make_3_clusters();
        let config = SkillDetectionConfig {
            min_notes_for_detection: 3, // lower threshold for test
            min_cluster_size: 3,
            min_cohesion: 0.3,
            louvain_resolution: 1.5,
            ..Default::default()
        };

        let result = detect_skill_candidates(&edges, "proj-1", &config);
        assert_eq!(result.status, ClusterDetectionStatus::Success);
        assert_eq!(result.total_notes, 15);
        // Should detect 3 clusters (Louvain on clearly separated communities)
        assert!(
            result.candidates.len() >= 2,
            "Expected at least 2 clusters, got {}",
            result.candidates.len()
        );
        // All candidates should have size >= 3
        for c in &result.candidates {
            assert!(c.size >= 3, "Cluster too small: {}", c.size);
            assert!(c.cohesion >= 0.3, "Cohesion too low: {}", c.cohesion);
        }
    }

    #[test]
    fn test_detect_filters_small_clusters() {
        // Create a graph with one big cluster and one tiny one
        let mut edges = Vec::new();
        // Big cluster: notes 1-10
        for i in 1..=10 {
            for j in (i + 1)..=10 {
                edges.push((format!("n{}", i), format!("n{}", j), 0.9));
            }
        }
        // Tiny cluster: notes 11-12 (only 2 notes)
        edges.push(("n11".to_string(), "n12".to_string(), 0.8));

        let config = SkillDetectionConfig {
            min_notes_for_detection: 3,
            min_cluster_size: 3,
            ..Default::default()
        };

        let result = detect_skill_candidates(&edges, "proj-1", &config);
        // The tiny cluster (2 notes) should be filtered out
        for c in &result.candidates {
            assert!(c.size >= 3);
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = SkillDetectionConfig::default();
        assert_eq!(config.min_synapse_weight, 0.1);
        assert_eq!(config.min_cluster_size, 3);
        assert_eq!(config.min_cohesion, 0.3);
        assert_eq!(config.louvain_resolution, 1.5);
        assert_eq!(config.overlap_threshold, 0.7);
        assert_eq!(config.min_notes_for_detection, 15);
    }

    #[test]
    fn test_detect_empty_graph() {
        let edges: Vec<(String, String, f64)> = Vec::new();
        let config = SkillDetectionConfig {
            min_notes_for_detection: 0,
            ..Default::default()
        };

        let result = detect_skill_candidates(&edges, "proj-1", &config);
        assert_eq!(result.status, ClusterDetectionStatus::InsufficientData);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn test_detect_suggestion_near_threshold() {
        // 12 notes — between 10 and 15, should suggest backfill_synapses
        let mut edges = Vec::new();
        for i in 1..=12 {
            for j in (i + 1)..=12 {
                edges.push((format!("n{}", i), format!("n{}", j), 0.5));
            }
        }
        let config = SkillDetectionConfig::default(); // threshold = 15

        let result = detect_skill_candidates(&edges, "proj-1", &config);
        assert_eq!(result.status, ClusterDetectionStatus::InsufficientData);
        assert!(result.message.contains("backfill_synapses"));
    }
}
