//! Skill Detection Engine — Louvain community detection on the SYNAPSE graph.
//!
//! Detects emergent neural skill clusters by analyzing the SYNAPSE (Note↔Note)
//! subgraph using the Louvain algorithm. Each detected community of strongly
//! connected notes becomes a candidate skill.

use crate::graph::algorithms::{compute_cohesion, louvain_communities};
use crate::graph::{AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeNode, CodeNodeType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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
// Cluster → Skill Conversion
// ============================================================================

/// Convert a detected cluster into a SkillNode with computed metrics.
///
/// - `name` is generated from member note tags (via `naming::generate_skill_name`)
/// - `energy` = weighted average of note energies (weighted by importance)
/// - `cohesion` = from Louvain community cohesion score
/// - `coverage` = cluster size / total notes in graph
pub fn cluster_to_skill(
    candidate: &SkillCandidate,
    notes: &[crate::notes::Note],
    project_id: uuid::Uuid,
    _total_notes: usize,
) -> crate::skills::SkillNode {
    use super::naming::generate_skill_name;

    // Collect tags from member notes for naming
    let tags_per_note: Vec<Vec<String>> = notes.iter().map(|n| n.tags.clone()).collect();
    let name = generate_skill_name(&tags_per_note, candidate.community_id);

    // Compute weighted energy: sum(energy × importance_weight) / sum(importance_weight)
    let (weighted_sum, weight_sum) = notes.iter().fold((0.0_f64, 0.0_f64), |(ws, wt), note| {
        let w = note.importance.weight();
        (ws + note.energy * w, wt + w)
    });
    let energy = if weight_sum > 0.0 {
        (weighted_sum / weight_sum).clamp(0.0, 1.0)
    } else {
        0.5 // default if no notes
    };

    // note_count = actual member count, coverage = cluster size (note count)
    let note_count = notes.len() as i64;
    let coverage = candidate.size as i64;

    // Collect all unique tags from member notes
    let mut all_tags: Vec<String> = notes.iter().flat_map(|n| n.tags.iter().cloned()).collect();
    all_tags.sort();
    all_tags.dedup();

    let mut skill = crate::skills::SkillNode::new(project_id, name);
    skill.energy = energy;
    skill.cohesion = candidate.cohesion;
    skill.note_count = note_count;
    skill.coverage = coverage;
    skill.tags = all_tags;
    // Description summarizing the cluster
    skill.description = format!(
        "Auto-detected skill from {} notes (cohesion: {:.2}, modularity cluster #{})",
        candidate.size, candidate.cohesion, candidate.community_id
    );

    skill
}

/// Compute the Jaccard similarity between two sets of note IDs.
///
/// Jaccard = |intersection| / |union|
pub fn jaccard_similarity(set_a: &[String], set_b: &[String]) -> f64 {
    let a: std::collections::HashSet<&str> = set_a.iter().map(|s| s.as_str()).collect();
    let b: std::collections::HashSet<&str> = set_b.iter().map(|s| s.as_str()).collect();
    let intersection = a.intersection(&b).count();
    let union = a.union(&b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ============================================================================
// Deduplication
// ============================================================================

/// Outcome of deduplication for a single candidate.
#[derive(Debug, Clone)]
pub enum DeduplicationOutcome {
    /// Candidate is new — should be created as a fresh skill.
    New(SkillCandidate),
    /// Candidate overlaps significantly with an existing skill — merge into it.
    Merge {
        existing_skill_id: Uuid,
        candidate: SkillCandidate,
        jaccard: f64,
    },
}

/// Deduplicate candidates against existing skills.
///
/// For each candidate, compute the Jaccard similarity of its member_note_ids
/// against each existing skill's member set. If Jaccard > overlap_threshold,
/// the candidate is classified as a Merge (update existing skill); otherwise
/// it's classified as New (create fresh skill).
///
/// If a candidate overlaps with multiple existing skills, the one with the
/// highest Jaccard score wins.
pub fn deduplicate_candidates(
    candidates: Vec<SkillCandidate>,
    existing_skills: &[(Uuid, Vec<String>)], // (skill_id, member_note_ids)
    overlap_threshold: f64,
) -> Vec<DeduplicationOutcome> {
    candidates
        .into_iter()
        .map(|candidate| {
            // Find the existing skill with highest Jaccard overlap
            let best_match = existing_skills
                .iter()
                .map(|(skill_id, members)| {
                    let jaccard = jaccard_similarity(&candidate.member_note_ids, members);
                    (*skill_id, jaccard)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            match best_match {
                Some((skill_id, jaccard)) if jaccard >= overlap_threshold => {
                    DeduplicationOutcome::Merge {
                        existing_skill_id: skill_id,
                        candidate,
                        jaccard,
                    }
                }
                _ => DeduplicationOutcome::New(candidate),
            }
        })
        .collect()
}

// ============================================================================
// Persistence
// ============================================================================

/// Persist detected skills into the graph store.
///
/// For each `DeduplicationOutcome`:
/// - `New`: creates a new SkillNode + adds all members
/// - `Merge`: updates the existing skill's energy/cohesion + syncs members
///
/// Returns the list of created or updated SkillNode IDs.
pub async fn persist_detected_skills(
    graph_store: &dyn crate::neo4j::traits::GraphStore,
    outcomes: &[DeduplicationOutcome],
    notes: &HashMap<String, crate::notes::Note>,
    project_id: Uuid,
    total_notes: usize,
) -> anyhow::Result<Vec<Uuid>> {
    let mut result_ids = Vec::new();

    for outcome in outcomes {
        match outcome {
            DeduplicationOutcome::New(candidate) => {
                // Collect member notes for this candidate
                let member_notes: Vec<&crate::notes::Note> = candidate
                    .member_note_ids
                    .iter()
                    .filter_map(|id| notes.get(id))
                    .collect();

                let member_notes_owned: Vec<crate::notes::Note> =
                    member_notes.iter().map(|n| (*n).clone()).collect();

                // Convert cluster to skill
                let skill =
                    cluster_to_skill(candidate, &member_notes_owned, project_id, total_notes);
                let skill_id = skill.id;

                // Create the skill node
                graph_store.create_skill(&skill).await?;

                // Add all members
                for note in &member_notes {
                    graph_store
                        .add_skill_member(skill_id, "note", note.id)
                        .await?;
                }

                result_ids.push(skill_id);
            }
            DeduplicationOutcome::Merge {
                existing_skill_id,
                candidate,
                ..
            } => {
                // Get the existing skill
                if let Some(mut skill) = graph_store.get_skill(*existing_skill_id).await? {
                    // Collect member notes
                    let member_notes: Vec<&crate::notes::Note> = candidate
                        .member_note_ids
                        .iter()
                        .filter_map(|id| notes.get(id))
                        .collect();

                    let member_notes_owned: Vec<crate::notes::Note> =
                        member_notes.iter().map(|n| (*n).clone()).collect();

                    // Recompute energy from current members
                    let (weighted_sum, weight_sum) =
                        member_notes_owned
                            .iter()
                            .fold((0.0_f64, 0.0_f64), |(ws, wt), note| {
                                let w = note.importance.weight();
                                (ws + note.energy * w, wt + w)
                            });
                    if weight_sum > 0.0 {
                        skill.energy = (weighted_sum / weight_sum).clamp(0.0, 1.0);
                    }
                    skill.cohesion = candidate.cohesion;
                    skill.coverage = candidate.size as i64;
                    skill.updated_at = chrono::Utc::now();

                    // Update the skill
                    graph_store.update_skill(&skill).await?;

                    // Add new members (add_skill_member uses MERGE, so duplicates are safe)
                    for note in &member_notes {
                        graph_store
                            .add_skill_member(*existing_skill_id, "note", note.id)
                            .await?;
                    }

                    result_ids.push(*existing_skill_id);
                }
            }
        }
    }

    Ok(result_ids)
}

// ============================================================================
// Graph Construction
// ============================================================================

/// Build a CodeGraph from SYNAPSE edges for Louvain community detection.
///
/// Each note becomes a node (CodeNodeType::Function as proxy), and each
/// SYNAPSE edge becomes a weighted edge. The graph is ready for `louvain_communities()`.
pub fn build_synapse_graph(edges: &[(String, String, f64)], project_id: &str) -> CodeGraph {
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
// Full Pipeline Orchestrator
// ============================================================================

/// Result of the full detect_skills pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectSkillsPipelineResult {
    /// Detection status
    pub status: ClusterDetectionStatus,
    /// Number of skill candidates detected by Louvain
    pub skills_detected: usize,
    /// Number of new skills created
    pub skills_created: usize,
    /// Number of existing skills updated (merged)
    pub skills_updated: usize,
    /// Total notes in the SYNAPSE graph
    pub total_notes: usize,
    /// Total SYNAPSE edges
    pub total_synapses: usize,
    /// Louvain modularity score
    pub modularity: f64,
    /// Human-readable summary message
    pub message: String,
    /// IDs of created/updated skills
    pub skill_ids: Vec<Uuid>,
}

/// Run the full skill detection pipeline:
/// 1. Fetch SYNAPSE graph from store
/// 2. Run Louvain community detection
/// 3. Deduplicate against existing skills
/// 4. Persist new/updated skills
/// 5. Generate triggers and context templates
///
/// This function is idempotent — re-running updates existing skills
/// rather than creating duplicates.
pub async fn detect_skills_pipeline(
    graph_store: &dyn crate::neo4j::traits::GraphStore,
    project_id: Uuid,
    config: &SkillDetectionConfig,
) -> anyhow::Result<DetectSkillsPipelineResult> {
    let project_id_str = project_id.to_string();

    // Step 1: Fetch SYNAPSE edges
    let edges = graph_store
        .get_synapse_graph(project_id, config.min_synapse_weight)
        .await?;

    // Step 2: Run Louvain detection
    let detection = detect_skill_candidates(&edges, &project_id_str, config);

    if detection.status == ClusterDetectionStatus::InsufficientData {
        return Ok(DetectSkillsPipelineResult {
            status: ClusterDetectionStatus::InsufficientData,
            skills_detected: 0,
            skills_created: 0,
            skills_updated: 0,
            total_notes: detection.total_notes,
            total_synapses: detection.total_synapses,
            modularity: 0.0,
            message: detection.message,
            skill_ids: Vec::new(),
        });
    }

    let skills_detected = detection.candidates.len();

    // Step 3: Fetch existing skills for deduplication
    let existing_skills = graph_store.get_skills_for_project(project_id).await?;
    let mut existing_members: Vec<(Uuid, Vec<String>)> = Vec::new();
    for skill in &existing_skills {
        let (notes, _decisions) = graph_store.get_skill_members(skill.id).await?;
        let member_ids: Vec<String> = notes.iter().map(|n| n.id.to_string()).collect();
        existing_members.push((skill.id, member_ids));
    }

    // Step 4: Deduplicate
    let outcomes = deduplicate_candidates(
        detection.candidates,
        &existing_members,
        config.overlap_threshold,
    );

    let skills_created = outcomes
        .iter()
        .filter(|o| matches!(o, DeduplicationOutcome::New(_)))
        .count();
    let skills_updated = outcomes
        .iter()
        .filter(|o| matches!(o, DeduplicationOutcome::Merge { .. }))
        .count();

    // Step 5: Fetch notes for conversion
    let mut notes_map: HashMap<String, crate::notes::Note> = HashMap::new();
    for outcome in &outcomes {
        let note_ids = match outcome {
            DeduplicationOutcome::New(c) => &c.member_note_ids,
            DeduplicationOutcome::Merge { candidate, .. } => &candidate.member_note_ids,
        };
        for note_id_str in note_ids {
            if !notes_map.contains_key(note_id_str) {
                if let Ok(uuid) = uuid::Uuid::parse_str(note_id_str) {
                    if let Ok(Some(note)) = graph_store.get_note(uuid).await {
                        notes_map.insert(note_id_str.clone(), note);
                    }
                }
            }
        }
    }

    // Step 6: Persist skills
    let skill_ids = persist_detected_skills(
        graph_store,
        &outcomes,
        &notes_map,
        project_id,
        detection.total_notes,
    )
    .await?;

    // Step 7: Generate triggers and templates for each skill
    // Fetch all project notes for quality evaluation
    let all_project_notes = {
        let max_notes = 5000;
        let filters = crate::notes::NoteFilters {
            limit: Some(max_notes),
            ..Default::default()
        };
        let (notes, _) = graph_store
            .list_notes(Some(project_id), None, &filters)
            .await?;
        if notes.len() >= max_notes as usize {
            tracing::warn!(
                project_id = %project_id,
                count = notes.len(),
                "Project has ≥{} notes — trigger quality evaluation may be incomplete",
                max_notes
            );
        }
        notes
    };

    for skill_id in &skill_ids {
        if let Ok(Some(mut skill)) = graph_store.get_skill(*skill_id).await {
            // Get member notes for this skill
            let (member_notes, _) = graph_store.get_skill_members(*skill_id).await?;

            // Generate triggers (no embeddings for now)
            let embeddings = HashMap::new();
            let trigger_result = crate::skills::triggers::generate_all_triggers(
                &member_notes,
                &all_project_notes,
                &embeddings,
            );
            skill.trigger_patterns = trigger_result.triggers;

            // Generate context template
            skill.context_template = Some(crate::skills::templates::generate_context_template(
                &skill.name,
                &skill.description,
                &member_notes,
            ));

            skill.updated_at = chrono::Utc::now();
            graph_store.update_skill(&skill).await?;
        }
    }

    let message = format!(
        "Detected {} candidates from {} notes/{} synapses (modularity: {:.3}). Created {} new, updated {} existing skills.",
        skills_detected,
        detection.total_notes,
        detection.total_synapses,
        detection.modularity,
        skills_created,
        skills_updated
    );

    Ok(DetectSkillsPipelineResult {
        status: ClusterDetectionStatus::Success,
        skills_detected,
        skills_created,
        skills_updated,
        total_notes: detection.total_notes,
        total_synapses: detection.total_synapses,
        modularity: detection.modularity,
        message,
        skill_ids,
    })
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

    // ================================================================
    // cluster_to_skill tests
    // ================================================================

    fn make_test_note(
        id: &str,
        energy: f64,
        importance: crate::notes::NoteImportance,
        tags: Vec<&str>,
    ) -> crate::notes::Note {
        use chrono::Utc;
        crate::notes::Note {
            id: uuid::Uuid::parse_str(id).unwrap_or_else(|_| uuid::Uuid::new_v4()),
            project_id: Some(uuid::Uuid::nil()),
            note_type: crate::notes::NoteType::Observation,
            status: crate::notes::NoteStatus::Active,
            importance,
            scope: crate::notes::NoteScope::Project,
            content: format!("Test note {}", id),
            tags: tags.into_iter().map(|t| t.to_string()).collect(),
            anchors: vec![],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy,
            last_activated: None,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
        }
    }

    #[test]
    fn test_cluster_to_skill_energy_weighted() {
        let candidate = SkillCandidate {
            community_id: 1,
            member_note_ids: vec!["a".into(), "b".into()],
            cohesion: 0.8,
            size: 2,
            label: "test".into(),
        };

        let notes = vec![
            make_test_note(
                "00000000-0000-0000-0000-000000000001",
                1.0,
                crate::notes::NoteImportance::Critical,
                vec!["api"],
            ),
            make_test_note(
                "00000000-0000-0000-0000-000000000002",
                0.2,
                crate::notes::NoteImportance::Low,
                vec!["api"],
            ),
        ];

        let skill = cluster_to_skill(&candidate, &notes, uuid::Uuid::nil(), 10);
        // Critical weight=1.0, Low weight=0.3
        // weighted = (1.0*1.0 + 0.2*0.3) / (1.0+0.3) = 1.06/1.3 ≈ 0.815
        assert!(
            skill.energy > 0.7,
            "Energy should be weighted towards critical note, got {}",
            skill.energy
        );
        assert_eq!(skill.cohesion, 0.8);
        assert_eq!(skill.coverage, 2); // 2 notes in cluster
    }

    #[test]
    fn test_cluster_to_skill_name_from_tags() {
        let candidate = SkillCandidate {
            community_id: 0,
            member_note_ids: vec![],
            cohesion: 0.5,
            size: 3,
            label: "test".into(),
        };

        let notes = vec![
            make_test_note(
                "00000000-0000-0000-0000-000000000001",
                0.5,
                crate::notes::NoteImportance::Medium,
                vec!["neo4j", "cypher"],
            ),
            make_test_note(
                "00000000-0000-0000-0000-000000000002",
                0.5,
                crate::notes::NoteImportance::Medium,
                vec!["neo4j", "graph"],
            ),
            make_test_note(
                "00000000-0000-0000-0000-000000000003",
                0.5,
                crate::notes::NoteImportance::Medium,
                vec!["neo4j"],
            ),
        ];

        let skill = cluster_to_skill(&candidate, &notes, uuid::Uuid::nil(), 10);
        assert!(
            skill.name.contains("Neo4j"),
            "Expected 'Neo4j' in skill name '{}'",
            skill.name
        );
    }

    // ================================================================
    // jaccard_similarity tests
    // ================================================================

    #[test]
    fn test_jaccard_identical() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        assert_eq!(jaccard_similarity(&a, &a), 1.0);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec!["x".into(), "y".into()];
        let b = vec!["z".into(), "w".into()];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        let b = vec!["y".into(), "z".into(), "w".into()];
        // intersection={y,z}=2, union={x,y,z,w}=4 → 0.5
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    // ================================================================
    // deduplicate_candidates tests
    // ================================================================

    #[test]
    fn test_deduplicate_no_existing_skills() {
        let candidates = vec![SkillCandidate {
            community_id: 0,
            member_note_ids: vec!["a".into(), "b".into(), "c".into()],
            cohesion: 0.8,
            size: 3,
            label: "test".into(),
        }];
        let existing: Vec<(Uuid, Vec<String>)> = vec![];
        let results = deduplicate_candidates(candidates, &existing, 0.7);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], DeduplicationOutcome::New(_)));
    }

    #[test]
    fn test_deduplicate_high_overlap_merges() {
        let existing_id = Uuid::new_v4();
        let candidates = vec![SkillCandidate {
            community_id: 0,
            // 4 out of 5 overlap with existing → Jaccard = 4/5 = 0.8
            member_note_ids: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            cohesion: 0.8,
            size: 4,
            label: "test".into(),
        }];
        let existing = vec![(
            existing_id,
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
        )];
        let results = deduplicate_candidates(candidates, &existing, 0.7);
        assert_eq!(results.len(), 1);
        match &results[0] {
            DeduplicationOutcome::Merge {
                existing_skill_id,
                jaccard,
                ..
            } => {
                assert_eq!(*existing_skill_id, existing_id);
                assert!(*jaccard >= 0.7);
            }
            _ => panic!("Expected Merge outcome"),
        }
    }

    #[test]
    fn test_deduplicate_low_overlap_creates_new() {
        let existing_id = Uuid::new_v4();
        let candidates = vec![SkillCandidate {
            community_id: 0,
            // Only 1 overlap out of 5 → Jaccard = 1/5 = 0.2
            member_note_ids: vec!["x".into(), "y".into(), "z".into()],
            cohesion: 0.8,
            size: 3,
            label: "test".into(),
        }];
        let existing = vec![(existing_id, vec!["a".into(), "b".into(), "x".into()])];
        let results = deduplicate_candidates(candidates, &existing, 0.7);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], DeduplicationOutcome::New(_)));
    }

    #[test]
    fn test_deduplicate_best_match_wins() {
        let skill_a = Uuid::new_v4();
        let skill_b = Uuid::new_v4();
        let candidates = vec![SkillCandidate {
            community_id: 0,
            member_note_ids: vec!["a".into(), "b".into(), "c".into()],
            cohesion: 0.8,
            size: 3,
            label: "test".into(),
        }];
        // skill_a has 1/4 overlap (0.25), skill_b has 3/3 overlap (1.0)
        let existing = vec![
            (
                skill_a,
                vec!["a".into(), "x".into(), "y".into(), "z".into()],
            ),
            (skill_b, vec!["a".into(), "b".into(), "c".into()]),
        ];
        let results = deduplicate_candidates(candidates, &existing, 0.7);
        match &results[0] {
            DeduplicationOutcome::Merge {
                existing_skill_id, ..
            } => {
                assert_eq!(*existing_skill_id, skill_b);
            }
            _ => panic!("Expected Merge with skill_b"),
        }
    }
}
