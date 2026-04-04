//! Skill Evolution Engine — Merge, Split, Grow, Shrink
//!
//! When `detect_skills_pipeline` is re-run (new notes, evolved synapses),
//! skills must evolve to match the new cluster structure:
//!
//! - **Merge**: 2 skills with >70% member overlap → fuse into one
//! - **Split**: 1 skill that developed 2+ sub-clusters → divide
//! - **Grow**: New notes appeared in the cluster → add as members
//! - **Shrink**: Notes left the cluster (synapses broken) → remove members
//!
//! Evolutions are traced as observation Notes for auditability.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::notes::{MemoryHorizon, Note, NoteImportance, NoteScope, NoteStatus, NoteType};
use crate::skills::detection::{jaccard_similarity, SkillCandidate};
use crate::skills::models::SkillNode;

// ============================================================================
// Evolution Result
// ============================================================================

/// Summary of all evolution actions performed during a re-detection cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionResult {
    /// Skills that were merged (absorbed into another)
    pub merged: Vec<MergeEvent>,
    /// Skills that were split into multiple new skills
    pub split: Vec<SplitEvent>,
    /// Skills that gained new members
    pub grown: Vec<GrowEvent>,
    /// Skills that lost members
    pub shrunk: Vec<ShrinkEvent>,
    /// Skills that are unchanged
    pub unchanged: usize,
}

/// A merge event: two skills fused into one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEvent {
    /// The surviving skill (highest energy)
    pub survivor_id: Uuid,
    /// The absorbed skill (archived)
    pub absorbed_id: Uuid,
    /// Jaccard overlap between the two
    pub overlap: f64,
}

/// A split event: one skill divided into N new skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitEvent {
    /// The original skill (archived)
    pub original_id: Uuid,
    /// The new skills created from sub-clusters
    pub new_skill_ids: Vec<Uuid>,
}

/// A grow event: a skill gained new members.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowEvent {
    /// The skill that grew
    pub skill_id: Uuid,
    /// Number of new notes added
    pub notes_added: usize,
}

/// A shrink event: a skill lost members.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkEvent {
    /// The skill that shrunk
    pub skill_id: Uuid,
    /// Number of notes removed
    pub notes_removed: usize,
}

// ============================================================================
// Evolution Analysis (pure function — no side effects)
// ============================================================================

/// Mapping from an existing skill to its detected cluster.
#[derive(Debug, Clone)]
pub enum SkillEvolution {
    /// Skill matches a single cluster — update members
    Stable {
        skill_id: Uuid,
        candidate: SkillCandidate,
        notes_to_add: Vec<String>,
        notes_to_remove: Vec<String>,
    },
    /// Two or more skills share a single cluster — merge
    Merge {
        skill_ids: Vec<Uuid>,
        candidate: SkillCandidate,
    },
    /// One skill split into multiple clusters
    Split {
        skill_id: Uuid,
        candidates: Vec<SkillCandidate>,
    },
    /// A cluster with no matching existing skill — create new
    New { candidate: SkillCandidate },
    /// An existing skill with no matching cluster — orphan (shrunk to nothing)
    Orphan { skill_id: Uuid },
}

// ── Guardrails (Biomimicry — Skill Evolution Guards) ──
/// Minimum members for a skill to be eligible for split.
/// Below this threshold, the skill is too small to meaningfully subdivide.
pub const MIN_MEMBERS_FOR_SPLIT: usize = 4;

/// Maximum combined members after a merge.
/// Above this threshold, the merged skill would be too large and lose cohesion.
pub const MAX_MEMBERS_AFTER_MERGE: usize = 20;

/// Minimum cohesion for a skill to remain active.
/// Below this threshold, the skill is considered an orphan and auto-archived.
pub const MIN_COHESION_THRESHOLD: f64 = 0.15;

/// Analyze how skills should evolve given new cluster detection results.
///
/// Compares existing skill membership with newly detected clusters using
/// Jaccard similarity. Returns a list of evolution actions to take.
///
/// Guardrails:
/// - Split is skipped for skills with fewer than [`MIN_MEMBERS_FOR_SPLIT`] members
/// - Merge is skipped if the combined member count exceeds [`MAX_MEMBERS_AFTER_MERGE`]
/// - Stable skills with cohesion below [`MIN_COHESION_THRESHOLD`] are demoted to Orphan
///
/// # Arguments
/// * `existing_skills` - Current skills with their member note IDs
/// * `new_candidates` - Newly detected clusters from Louvain
/// * `overlap_threshold` - Minimum Jaccard for considering two sets as the same (default: 0.7)
pub fn analyze_evolution(
    existing_skills: &[(Uuid, Vec<String>)],
    new_candidates: &[SkillCandidate],
    overlap_threshold: f64,
) -> Vec<SkillEvolution> {
    let mut evolutions = Vec::new();

    // Build a similarity matrix: candidate × existing_skill → jaccard
    let mut candidate_best_matches: Vec<Vec<(Uuid, f64)>> = Vec::new();
    for candidate in new_candidates {
        let mut matches: Vec<(Uuid, f64)> = existing_skills
            .iter()
            .map(|(skill_id, members)| {
                let jaccard = jaccard_similarity(&candidate.member_note_ids, members);
                (*skill_id, jaccard)
            })
            .filter(|(_, j)| *j >= overlap_threshold)
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidate_best_matches.push(matches);
    }

    // Track which skills and candidates have been assigned
    let mut assigned_skills: HashSet<Uuid> = HashSet::new();
    let mut assigned_candidates: HashSet<usize> = HashSet::new();

    // Pass 1: Find 1-to-1 matches (stable or grow/shrink)
    for (cand_idx, matches) in candidate_best_matches.iter().enumerate() {
        if matches.len() == 1 {
            let (skill_id, _jaccard) = matches[0];
            // Check if this skill is only matched by this one candidate
            let other_candidates_matching_this_skill: Vec<usize> = candidate_best_matches
                .iter()
                .enumerate()
                .filter(|(idx, m)| *idx != cand_idx && m.iter().any(|(sid, _)| *sid == skill_id))
                .map(|(idx, _)| idx)
                .collect();

            if other_candidates_matching_this_skill.is_empty() {
                // 1-to-1 match — stable with possible grow/shrink
                let existing_members: &Vec<String> = &existing_skills
                    .iter()
                    .find(|(sid, _)| *sid == skill_id)
                    .unwrap()
                    .1;
                let new_members: HashSet<&str> = new_candidates[cand_idx]
                    .member_note_ids
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let old_members: HashSet<&str> =
                    existing_members.iter().map(|s| s.as_str()).collect();

                let notes_to_add: Vec<String> = new_members
                    .difference(&old_members)
                    .map(|s| s.to_string())
                    .collect();
                let notes_to_remove: Vec<String> = old_members
                    .difference(&new_members)
                    .map(|s| s.to_string())
                    .collect();

                evolutions.push(SkillEvolution::Stable {
                    skill_id,
                    candidate: new_candidates[cand_idx].clone(),
                    notes_to_add,
                    notes_to_remove,
                });
                assigned_skills.insert(skill_id);
                assigned_candidates.insert(cand_idx);
            }
        }
    }

    // Pass 2: Find merges (multiple skills → one candidate)
    for (cand_idx, matches) in candidate_best_matches.iter().enumerate() {
        if assigned_candidates.contains(&cand_idx) {
            continue;
        }
        if matches.len() >= 2 {
            let skill_ids: Vec<Uuid> = matches
                .iter()
                .filter(|(sid, _)| !assigned_skills.contains(sid))
                .map(|(sid, _)| *sid)
                .collect();
            if skill_ids.len() >= 2 {
                // Guardrail: skip merge if combined member count would exceed max
                let total_members: usize = skill_ids
                    .iter()
                    .filter_map(|sid| existing_skills.iter().find(|(id, _)| id == sid))
                    .map(|(_, members)| members.len())
                    .sum();
                if total_members > MAX_MEMBERS_AFTER_MERGE {
                    debug!(
                        total_members,
                        max = MAX_MEMBERS_AFTER_MERGE,
                        "Skipping merge: combined members would exceed max"
                    );
                    continue;
                }

                for sid in &skill_ids {
                    assigned_skills.insert(*sid);
                }
                assigned_candidates.insert(cand_idx);
                evolutions.push(SkillEvolution::Merge {
                    skill_ids,
                    candidate: new_candidates[cand_idx].clone(),
                });
            }
        }
    }

    // Pass 3: Find splits (one skill matched by multiple candidates)
    for (skill_id, members) in existing_skills {
        if assigned_skills.contains(skill_id) {
            continue;
        }
        // Guardrail: skip split if skill has too few members to meaningfully subdivide
        if members.len() < MIN_MEMBERS_FOR_SPLIT {
            debug!(
                skill_id = %skill_id,
                members = members.len(),
                min = MIN_MEMBERS_FOR_SPLIT,
                "Skipping split: skill has too few members"
            );
            continue;
        }
        let matching_candidates: Vec<usize> = candidate_best_matches
            .iter()
            .enumerate()
            .filter(|(idx, matches)| {
                !assigned_candidates.contains(idx) && matches.iter().any(|(sid, _)| sid == skill_id)
            })
            .map(|(idx, _)| idx)
            .collect();

        if matching_candidates.len() >= 2 {
            let candidates: Vec<SkillCandidate> = matching_candidates
                .iter()
                .map(|idx| new_candidates[*idx].clone())
                .collect();
            for idx in &matching_candidates {
                assigned_candidates.insert(*idx);
            }
            assigned_skills.insert(*skill_id);
            evolutions.push(SkillEvolution::Split {
                skill_id: *skill_id,
                candidates,
            });
        }
    }

    // Pass 3b: Demote low-cohesion stable skills to Orphan (auto-archive)
    let mut demoted_skills: HashSet<Uuid> = HashSet::new();
    evolutions.retain(|ev| {
        if let SkillEvolution::Stable {
            skill_id,
            candidate,
            ..
        } = ev
        {
            if candidate.cohesion < MIN_COHESION_THRESHOLD {
                debug!(
                    skill_id = %skill_id,
                    cohesion = candidate.cohesion,
                    threshold = MIN_COHESION_THRESHOLD,
                    "Demoting low-cohesion skill to Orphan"
                );
                demoted_skills.insert(*skill_id);
                return false; // remove from evolutions, will be re-added as Orphan
            }
        }
        true
    });
    for skill_id in &demoted_skills {
        assigned_skills.remove(skill_id);
        evolutions.push(SkillEvolution::Orphan {
            skill_id: *skill_id,
        });
    }

    // Pass 4: Remaining unmatched candidates → New skills
    for (cand_idx, _) in new_candidates.iter().enumerate() {
        if !assigned_candidates.contains(&cand_idx) {
            evolutions.push(SkillEvolution::New {
                candidate: new_candidates[cand_idx].clone(),
            });
        }
    }

    // Pass 5: Remaining unmatched skills → Orphans
    for (skill_id, _) in existing_skills {
        if !assigned_skills.contains(skill_id) {
            evolutions.push(SkillEvolution::Orphan {
                skill_id: *skill_id,
            });
        }
    }

    evolutions
}

// ============================================================================
// Standalone Detection — Fission & Fusion (read-only inspection)
// ============================================================================

/// A candidate for skill fission (split).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionCandidate {
    /// The skill that could be split.
    pub skill_id: Uuid,
    /// The skill name.
    pub skill_name: String,
    /// Current member count.
    pub member_count: usize,
    /// Current cohesion score.
    pub cohesion: f64,
    /// Number of sub-clusters detected.
    pub sub_cluster_count: usize,
    /// Sub-cluster sizes.
    pub sub_cluster_sizes: Vec<usize>,
    /// Whether the guardrail would block this split.
    pub blocked_by_guardrail: bool,
    /// Reason for guardrail block, if any.
    pub guardrail_reason: Option<String>,
}

/// A candidate for skill fusion (merge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCandidate {
    /// The skills that could be merged.
    pub skill_ids: Vec<Uuid>,
    /// The skill names.
    pub skill_names: Vec<String>,
    /// Jaccard overlap between the skills.
    pub overlap: f64,
    /// Combined member count after merge.
    pub combined_member_count: usize,
    /// Whether the guardrail would block this merge.
    pub blocked_by_guardrail: bool,
    /// Reason for guardrail block, if any.
    pub guardrail_reason: Option<String>,
}

/// Detect skills that are candidates for fission (splitting).
///
/// Re-runs Louvain cluster detection on the SYNAPSE graph (read-only),
/// then checks if any existing skill maps to multiple new clusters.
/// Returns candidates without modifying the graph.
pub async fn detect_skill_fission(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    overlap_threshold: f64,
) -> anyhow::Result<Vec<FissionCandidate>> {
    let config = crate::skills::detection::SkillDetectionConfig::default();

    // Step 1: Get current SYNAPSE edges and run Louvain (read-only)
    let edges = graph_store
        .get_synapse_graph(project_id, config.min_synapse_weight)
        .await?;
    let detection =
        crate::skills::detection::detect_skill_candidates(&edges, &project_id.to_string(), &config);

    if detection.status == crate::skills::detection::ClusterDetectionStatus::InsufficientData {
        return Ok(Vec::new());
    }

    // Step 2: Get existing skills with their members
    let existing_skills = graph_store.get_skills_for_project(project_id).await?;
    let mut candidates = Vec::new();

    for skill in &existing_skills {
        if skill.status == crate::skills::SkillStatus::Archived {
            continue;
        }

        let (member_notes, _) = graph_store.get_skill_members(skill.id).await?;
        let member_ids: Vec<String> = member_notes.iter().map(|n| n.id.to_string()).collect();

        if member_ids.len() < 2 {
            continue;
        }

        // Find new clusters that overlap with this skill
        let matching: Vec<&SkillCandidate> = detection
            .candidates
            .iter()
            .filter(|c| {
                let jaccard = jaccard_similarity(&c.member_note_ids, &member_ids);
                jaccard >= overlap_threshold
            })
            .collect();

        if matching.len() >= 2 {
            let sub_cluster_sizes: Vec<usize> = matching.iter().map(|c| c.size).collect();
            let blocked = member_ids.len() < MIN_MEMBERS_FOR_SPLIT;
            candidates.push(FissionCandidate {
                skill_id: skill.id,
                skill_name: skill.name.clone(),
                member_count: member_ids.len(),
                cohesion: skill.cohesion,
                sub_cluster_count: matching.len(),
                sub_cluster_sizes,
                blocked_by_guardrail: blocked,
                guardrail_reason: if blocked {
                    Some(format!(
                        "Skill has {} members, minimum {} required for split",
                        member_ids.len(),
                        MIN_MEMBERS_FOR_SPLIT
                    ))
                } else {
                    None
                },
            });
        }
    }

    Ok(candidates)
}

/// Detect pairs of skills that are candidates for fusion (merging).
///
/// Compares all active skills pairwise by Jaccard similarity of their members.
/// Returns candidates without modifying the graph.
pub async fn detect_skill_fusion(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    overlap_threshold: f64,
) -> anyhow::Result<Vec<FusionCandidate>> {
    let existing_skills = graph_store.get_skills_for_project(project_id).await?;
    let mut candidates = Vec::new();

    // Collect active skills with their members
    let mut active_skills: Vec<(Uuid, String, Vec<String>)> = Vec::new();
    for skill in &existing_skills {
        if skill.status == crate::skills::SkillStatus::Archived {
            continue;
        }
        let (member_notes, _) = graph_store.get_skill_members(skill.id).await?;
        let member_ids: Vec<String> = member_notes.iter().map(|n| n.id.to_string()).collect();
        active_skills.push((skill.id, skill.name.clone(), member_ids));
    }

    // Pairwise comparison
    let mut seen_pairs: HashSet<(Uuid, Uuid)> = HashSet::new();
    for i in 0..active_skills.len() {
        for j in (i + 1)..active_skills.len() {
            let (id_a, name_a, members_a) = &active_skills[i];
            let (id_b, name_b, members_b) = &active_skills[j];

            let pair_key = if id_a < id_b {
                (*id_a, *id_b)
            } else {
                (*id_b, *id_a)
            };
            if seen_pairs.contains(&pair_key) {
                continue;
            }
            seen_pairs.insert(pair_key);

            let overlap = jaccard_similarity(members_a, members_b);
            if overlap >= overlap_threshold {
                let combined = {
                    let mut all: HashSet<&str> = members_a.iter().map(|s| s.as_str()).collect();
                    all.extend(members_b.iter().map(|s| s.as_str()));
                    all.len()
                };
                let blocked = combined > MAX_MEMBERS_AFTER_MERGE;
                candidates.push(FusionCandidate {
                    skill_ids: vec![*id_a, *id_b],
                    skill_names: vec![name_a.clone(), name_b.clone()],
                    overlap,
                    combined_member_count: combined,
                    blocked_by_guardrail: blocked,
                    guardrail_reason: if blocked {
                        Some(format!(
                            "Combined {} members exceeds maximum {}",
                            combined, MAX_MEMBERS_AFTER_MERGE
                        ))
                    } else {
                        None
                    },
                });
            }
        }
    }

    // Sort by overlap descending
    candidates.sort_by(|a, b| {
        b.overlap
            .partial_cmp(&a.overlap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(candidates)
}

// ============================================================================
// Evolution Execution (applies changes to graph store)
// ============================================================================

/// Create an observation note to trace an evolution event.
async fn trace_evolution_note(graph_store: &dyn GraphStore, project_id: Uuid, content: String) {
    let note = Note {
        id: Uuid::new_v4(),
        project_id: Some(project_id),
        note_type: NoteType::Observation,
        status: NoteStatus::Active,
        importance: NoteImportance::Low,
        scope: NoteScope::Project,
        content,
        tags: vec!["skill-evolution".to_string(), "auto-generated".to_string()],
        anchors: vec![],
        created_at: Utc::now(),
        created_by: "skill-evolution".to_string(),
        last_confirmed_at: None,
        last_confirmed_by: None,
        staleness_score: 0.0,
        energy: 0.5, // Moderate initial energy — not critical knowledge
        last_activated: Some(Utc::now()),
        reactivation_count: 0,
        last_reactivated: None,
        freshness_pinged_at: None,
        activation_count: 0,
        scar_intensity: 0.0,
        memory_horizon: MemoryHorizon::Ephemeral, // Auto-generated trace notes are ephemeral
        supersedes: None,
        superseded_by: None,
        changes: vec![],
        assertion_rule: None,
        last_assertion_result: None,
        sharing_consent: Default::default(),
    };
    if let Err(e) = graph_store.create_note(&note).await {
        warn!(error = %e, "Failed to create evolution observation note");
    }
}

/// Execute evolution actions on the graph store.
///
/// For each `SkillEvolution`:
/// - **Stable**: add/remove member notes, update metrics
/// - **Merge**: keep highest-energy skill, transfer members, archive others
/// - **Split**: archive original skill, create N new skills
/// - **New**: create a fresh skill
/// - **Orphan**: archive the skill (no matching cluster)
pub async fn execute_evolution(
    graph_store: &dyn GraphStore,
    evolutions: &[SkillEvolution],
    notes_map: &HashMap<String, crate::notes::Note>,
    project_id: Uuid,
) -> anyhow::Result<EvolutionResult> {
    let mut result = EvolutionResult::default();

    for evolution in evolutions {
        match evolution {
            SkillEvolution::Stable {
                skill_id,
                candidate,
                notes_to_add,
                notes_to_remove,
            } => {
                if notes_to_add.is_empty() && notes_to_remove.is_empty() {
                    result.unchanged += 1;
                    continue;
                }

                // Add new members
                for note_id_str in notes_to_add {
                    if let Ok(uuid) = Uuid::parse_str(note_id_str) {
                        if let Err(e) = graph_store.add_skill_member(*skill_id, "note", uuid).await
                        {
                            warn!(skill_id = %skill_id, note_id = %uuid, error = %e, "Failed to add member during grow");
                        }
                    }
                }

                // Remove departed members
                for note_id_str in notes_to_remove {
                    if let Ok(uuid) = Uuid::parse_str(note_id_str) {
                        if let Err(e) = graph_store
                            .remove_skill_member(*skill_id, "note", uuid)
                            .await
                        {
                            warn!(skill_id = %skill_id, note_id = %uuid, error = %e, "Failed to remove member during shrink");
                        }
                    }
                }

                // Update skill metrics
                if let Ok(Some(mut skill)) = graph_store.get_skill(*skill_id).await {
                    let (members, _) = graph_store.get_skill_members(*skill_id).await?;
                    skill.note_count = members.len() as i64;
                    skill.cohesion = candidate.cohesion.clamp(0.0, 1.0);
                    skill.coverage = candidate.size as i64;
                    skill.version += 1;
                    skill.updated_at = Utc::now();
                    graph_store.update_skill(&skill).await?;
                }

                if !notes_to_add.is_empty() {
                    trace_evolution_note(
                        graph_store,
                        project_id,
                        format!(
                            "Skill {} grew: +{} notes added",
                            skill_id,
                            notes_to_add.len()
                        ),
                    )
                    .await;
                    result.grown.push(GrowEvent {
                        skill_id: *skill_id,
                        notes_added: notes_to_add.len(),
                    });
                }
                if !notes_to_remove.is_empty() {
                    trace_evolution_note(
                        graph_store,
                        project_id,
                        format!(
                            "Skill {} shrunk: -{} notes removed",
                            skill_id,
                            notes_to_remove.len()
                        ),
                    )
                    .await;
                    result.shrunk.push(ShrinkEvent {
                        skill_id: *skill_id,
                        notes_removed: notes_to_remove.len(),
                    });
                }
            }

            SkillEvolution::Merge {
                skill_ids,
                candidate,
            } => {
                // Find the skill with highest energy to keep
                let mut best_skill: Option<SkillNode> = None;
                let mut all_skills: Vec<SkillNode> = Vec::new();

                for sid in skill_ids {
                    if let Ok(Some(s)) = graph_store.get_skill(*sid).await {
                        all_skills.push(s);
                    }
                }

                all_skills.sort_by(|a, b| {
                    b.energy
                        .partial_cmp(&a.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some(survivor) = all_skills.first() {
                    best_skill = Some(survivor.clone());
                }

                if let Some(mut survivor) = best_skill {
                    let survivor_id = survivor.id;

                    // Transfer all members from absorbed skills to survivor
                    for absorbed in all_skills.iter().skip(1) {
                        let (notes, decisions) = graph_store.get_skill_members(absorbed.id).await?;

                        // Compute actual overlap between absorbed skill's members and candidate
                        let absorbed_member_ids: Vec<String> =
                            notes.iter().map(|n| n.id.to_string()).collect();
                        let overlap =
                            jaccard_similarity(&candidate.member_note_ids, &absorbed_member_ids);

                        for note in &notes {
                            if let Err(e) = graph_store
                                .add_skill_member(survivor_id, "note", note.id)
                                .await
                            {
                                warn!(survivor = %survivor_id, note_id = %note.id, error = %e, "Failed to transfer note during merge");
                            }
                        }
                        for decision in &decisions {
                            if let Err(e) = graph_store
                                .add_skill_member(survivor_id, "decision", decision.id)
                                .await
                            {
                                warn!(survivor = %survivor_id, decision_id = %decision.id, error = %e, "Failed to transfer decision during merge");
                            }
                        }

                        // Clean up absorbed skill's member relationships before archiving
                        if let Err(e) = graph_store.remove_all_skill_members(absorbed.id).await {
                            warn!(absorbed = %absorbed.id, error = %e, "Failed to remove members from absorbed skill during merge");
                        }

                        // Archive the absorbed skill
                        let mut archived = absorbed.clone();
                        archived.status = crate::skills::SkillStatus::Archived;
                        archived.updated_at = Utc::now();
                        graph_store.update_skill(&archived).await?;

                        result.merged.push(MergeEvent {
                            survivor_id,
                            absorbed_id: absorbed.id,
                            overlap,
                        });

                        trace_evolution_note(
                            graph_store,
                            project_id,
                            format!(
                                "Skill merge: {} absorbed into survivor {}",
                                absorbed.id, survivor_id
                            ),
                        )
                        .await;

                        info!(
                            survivor = %survivor_id,
                            absorbed = %absorbed.id,
                            "Merged skill into survivor"
                        );
                    }

                    // Update survivor metrics
                    let (members, _) = graph_store.get_skill_members(survivor_id).await?;
                    survivor.note_count = members.len() as i64;
                    survivor.cohesion = candidate.cohesion.clamp(0.0, 1.0);
                    survivor.coverage = candidate.size as i64;
                    survivor.version += 1;
                    survivor.updated_at = Utc::now();
                    graph_store.update_skill(&survivor).await?;
                }
            }

            SkillEvolution::Split {
                skill_id,
                candidates,
            } => {
                let mut new_skill_ids = Vec::new();

                for candidate in candidates {
                    // Collect member notes
                    let member_notes: Vec<crate::notes::Note> = candidate
                        .member_note_ids
                        .iter()
                        .filter_map(|id| notes_map.get(id))
                        .cloned()
                        .collect();

                    let new_skill = crate::skills::detection::cluster_to_skill(
                        candidate,
                        &member_notes,
                        project_id,
                        0,
                    );
                    let new_id = new_skill.id;
                    graph_store.create_skill(&new_skill).await?;

                    for note in &member_notes {
                        if let Err(e) = graph_store.add_skill_member(new_id, "note", note.id).await
                        {
                            warn!(skill_id = %new_id, note_id = %note.id, error = %e, "Failed to add member during split");
                        }
                    }

                    new_skill_ids.push(new_id);
                    debug!(new_skill_id = %new_id, members = member_notes.len(), "Created split skill");
                }

                // Transfer decisions from original skill to new sub-skills
                if let Ok((_, decisions)) = graph_store.get_skill_members(*skill_id).await {
                    for decision in &decisions {
                        // Add decision to all new sub-skills (they share the lineage)
                        for &new_id in &new_skill_ids {
                            if let Err(e) = graph_store
                                .add_skill_member(new_id, "decision", decision.id)
                                .await
                            {
                                warn!(skill_id = %new_id, decision_id = %decision.id, error = %e, "Failed to transfer decision during split");
                            }
                        }
                    }
                }

                // Clean up original skill's member relationships before archiving
                if let Err(e) = graph_store.remove_all_skill_members(*skill_id).await {
                    warn!(skill_id = %skill_id, error = %e, "Failed to remove members from original skill during split");
                }

                // Archive the original skill
                if let Ok(Some(mut original)) = graph_store.get_skill(*skill_id).await {
                    original.status = crate::skills::SkillStatus::Archived;
                    original.updated_at = Utc::now();
                    graph_store.update_skill(&original).await?;
                }

                result.split.push(SplitEvent {
                    original_id: *skill_id,
                    new_skill_ids,
                });

                trace_evolution_note(
                    graph_store,
                    project_id,
                    format!(
                        "Skill {} split into {} sub-clusters",
                        skill_id,
                        candidates.len()
                    ),
                )
                .await;

                info!(original = %skill_id, "Split skill into sub-clusters");
            }

            SkillEvolution::New { candidate } => {
                let member_notes: Vec<crate::notes::Note> = candidate
                    .member_note_ids
                    .iter()
                    .filter_map(|id| notes_map.get(id))
                    .cloned()
                    .collect();

                let new_skill = crate::skills::detection::cluster_to_skill(
                    candidate,
                    &member_notes,
                    project_id,
                    0,
                );
                let new_id = new_skill.id;
                graph_store.create_skill(&new_skill).await?;

                for note in &member_notes {
                    if let Err(e) = graph_store.add_skill_member(new_id, "note", note.id).await {
                        warn!(skill_id = %new_id, note_id = %note.id, error = %e, "Failed to add member to new skill");
                    }
                }

                debug!(skill_id = %new_id, members = member_notes.len(), "Created new skill from unmatched cluster");
            }

            SkillEvolution::Orphan { skill_id } => {
                // Orphan skill — no matching cluster. Archive it.
                if let Ok(Some(mut skill)) = graph_store.get_skill(*skill_id).await {
                    if skill.status != crate::skills::SkillStatus::Archived {
                        // Clean up member relationships before archiving
                        if let Err(e) = graph_store.remove_all_skill_members(*skill_id).await {
                            warn!(skill_id = %skill_id, error = %e, "Failed to remove members from orphaned skill");
                        }

                        skill.status = crate::skills::SkillStatus::Archived;
                        skill.updated_at = Utc::now();
                        graph_store.update_skill(&skill).await?;
                        trace_evolution_note(
                            graph_store,
                            project_id,
                            format!(
                                "Skill {} ({}) orphaned and archived — no matching cluster",
                                skill_id, skill.name
                            ),
                        )
                        .await;
                        warn!(skill_id = %skill_id, name = %skill.name, "Orphaned skill archived — no matching cluster");
                    }
                }
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: u32, members: Vec<&str>) -> SkillCandidate {
        SkillCandidate {
            community_id: id,
            member_note_ids: members.into_iter().map(|s| s.to_string()).collect(),
            cohesion: 0.7,
            size: 0, // will be set from members
            label: format!("cluster-{}", id),
        }
    }

    // ================================================================
    // analyze_evolution tests
    // ================================================================

    #[test]
    fn test_evolution_stable_unchanged() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        let candidates = vec![make_candidate(0, vec!["a", "b", "c"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 1);
        match &evolutions[0] {
            SkillEvolution::Stable {
                skill_id: sid,
                notes_to_add,
                notes_to_remove,
                ..
            } => {
                assert_eq!(*sid, skill_id);
                assert!(notes_to_add.is_empty());
                assert!(notes_to_remove.is_empty());
            }
            other => panic!("Expected Stable, got {:?}", other),
        }
    }

    #[test]
    fn test_evolution_stable_grow() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // New cluster has an extra note "d"
        let candidates = vec![make_candidate(0, vec!["a", "b", "c", "d"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 1);
        match &evolutions[0] {
            SkillEvolution::Stable {
                notes_to_add,
                notes_to_remove,
                ..
            } => {
                assert_eq!(notes_to_add, &vec!["d".to_string()]);
                assert!(notes_to_remove.is_empty());
            }
            other => panic!("Expected Stable, got {:?}", other),
        }
    }

    #[test]
    fn test_evolution_stable_shrink() {
        let skill_id = Uuid::new_v4();
        // 4 members → 3 members: Jaccard = 3/4 = 0.75 ≥ 0.7 threshold
        let existing = vec![(
            skill_id,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        )];
        // Note "d" left the cluster
        let candidates = vec![make_candidate(0, vec!["a", "b", "c"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 1);
        match &evolutions[0] {
            SkillEvolution::Stable {
                notes_to_add,
                notes_to_remove,
                ..
            } => {
                assert!(notes_to_add.is_empty());
                assert_eq!(notes_to_remove, &vec!["d".to_string()]);
            }
            other => panic!("Expected Stable, got {:?}", other),
        }
    }

    #[test]
    fn test_evolution_new_cluster() {
        let existing: Vec<(Uuid, Vec<String>)> = vec![];
        let candidates = vec![make_candidate(0, vec!["x", "y", "z"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 1);
        assert!(matches!(&evolutions[0], SkillEvolution::New { .. }));
    }

    #[test]
    fn test_evolution_orphan_skill() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // No matching candidates at all
        let candidates: Vec<SkillCandidate> = vec![];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 1);
        match &evolutions[0] {
            SkillEvolution::Orphan { skill_id: sid } => {
                assert_eq!(*sid, skill_id);
            }
            other => panic!("Expected Orphan, got {:?}", other),
        }
    }

    #[test]
    fn test_evolution_merge_two_skills() {
        let skill_a = Uuid::new_v4();
        let skill_b = Uuid::new_v4();
        let existing = vec![
            (skill_a, vec!["a".into(), "b".into(), "c".into()]),
            (skill_b, vec!["b".into(), "c".into(), "d".into()]),
        ];
        // Single candidate that overlaps heavily with both
        let candidates = vec![make_candidate(0, vec!["a", "b", "c", "d"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        let merge_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Merge { .. }))
            .count();
        assert!(
            merge_count >= 1,
            "Expected at least one merge, got evolutions: {:?}",
            evolutions
                .iter()
                .map(|e| match e {
                    SkillEvolution::Stable { skill_id, .. } => format!("Stable({})", skill_id),
                    SkillEvolution::Merge { skill_ids, .. } => format!("Merge({:?})", skill_ids),
                    SkillEvolution::Split { skill_id, .. } => format!("Split({})", skill_id),
                    SkillEvolution::New { .. } => "New".to_string(),
                    SkillEvolution::Orphan { skill_id } => format!("Orphan({})", skill_id),
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_evolution_no_match_below_threshold() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // Completely different cluster
        let candidates = vec![make_candidate(0, vec!["x", "y", "z"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        // Should see an Orphan for the existing skill and a New for the candidate
        let has_orphan = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::Orphan { .. }));
        let has_new = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::New { .. }));
        assert!(has_orphan, "Expected orphan for unmatched skill");
        assert!(has_new, "Expected new for unmatched candidate");
    }

    #[test]
    fn test_evolution_multiple_stable() {
        let skill_a = Uuid::new_v4();
        let skill_b = Uuid::new_v4();
        let existing = vec![
            (skill_a, vec!["a".into(), "b".into(), "c".into()]),
            (skill_b, vec!["x".into(), "y".into(), "z".into()]),
        ];
        // Two distinct clusters matching each skill
        let candidates = vec![
            make_candidate(0, vec!["a", "b", "c"]),
            make_candidate(1, vec!["x", "y", "z"]),
        ];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        let stable_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Stable { .. }))
            .count();
        assert_eq!(stable_count, 2, "Expected 2 stable evolutions");
    }

    // ================================================================
    // Guardrail tests (Biomimicry — Skill Evolution Guards)
    // ================================================================

    #[test]
    fn test_guardrail_skip_split_too_few_members() {
        let skill_id = Uuid::new_v4();
        // Skill with only 3 members (< MIN_MEMBERS_FOR_SPLIT = 4)
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // Two candidates that both overlap with the skill
        let candidates = vec![
            make_candidate(0, vec!["a", "b"]),
            make_candidate(1, vec!["b", "c"]),
        ];

        let evolutions = analyze_evolution(&existing, &candidates, 0.5);
        // Should NOT produce a Split — skill is too small
        let split_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Split { .. }))
            .count();
        assert_eq!(split_count, 0, "Should not split skill with < 4 members");
    }

    #[test]
    fn test_guardrail_skip_merge_too_many_members() {
        // Create skills whose combined membership exceeds MAX_MEMBERS_AFTER_MERGE (20)
        let skill_a = Uuid::new_v4();
        let skill_b = Uuid::new_v4();
        let members_a: Vec<String> = (0..12).map(|i| format!("a{}", i)).collect();
        let members_b: Vec<String> = (0..12).map(|i| format!("b{}", i)).collect();
        let existing = vec![(skill_a, members_a.clone()), (skill_b, members_b.clone())];
        // Single candidate overlapping heavily with both
        let mut all_members: Vec<&str> = members_a.iter().map(|s| s.as_str()).collect();
        all_members.extend(members_b.iter().map(|s| s.as_str()));
        let candidates = vec![make_candidate(0, all_members)];

        let evolutions = analyze_evolution(&existing, &candidates, 0.5);
        // Should NOT produce a Merge — combined 24 > 20
        let merge_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Merge { .. }))
            .count();
        assert_eq!(
            merge_count, 0,
            "Should not merge when combined members > 20"
        );
    }

    #[test]
    fn test_guardrail_low_cohesion_demoted_to_orphan() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // Candidate matches but has very low cohesion
        let mut candidate = make_candidate(0, vec!["a", "b", "c"]);
        candidate.cohesion = 0.1; // < MIN_COHESION_THRESHOLD (0.15)
        let candidates = vec![candidate];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        // Should produce Orphan, NOT Stable
        let has_orphan = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::Orphan { skill_id: sid } if *sid == skill_id));
        assert!(has_orphan, "Low-cohesion skill should be demoted to Orphan");
        let has_stable = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::Stable { .. }));
        assert!(!has_stable, "Low-cohesion skill should NOT remain Stable");
    }

    // ================================================================
    // Struct serialization / construction tests
    // ================================================================

    #[test]
    fn test_evolution_result_default() {
        let result = EvolutionResult::default();
        assert!(result.merged.is_empty());
        assert!(result.split.is_empty());
        assert!(result.grown.is_empty());
        assert!(result.shrunk.is_empty());
        assert_eq!(result.unchanged, 0);
    }

    #[test]
    fn test_evolution_result_serialization_roundtrip() {
        let result = EvolutionResult {
            merged: vec![MergeEvent {
                survivor_id: Uuid::new_v4(),
                absorbed_id: Uuid::new_v4(),
                overlap: 0.85,
            }],
            split: vec![SplitEvent {
                original_id: Uuid::new_v4(),
                new_skill_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            }],
            grown: vec![GrowEvent {
                skill_id: Uuid::new_v4(),
                notes_added: 3,
            }],
            shrunk: vec![ShrinkEvent {
                skill_id: Uuid::new_v4(),
                notes_removed: 2,
            }],
            unchanged: 5,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: EvolutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.merged.len(), 1);
        assert_eq!(deserialized.split.len(), 1);
        assert_eq!(deserialized.grown.len(), 1);
        assert_eq!(deserialized.shrunk.len(), 1);
        assert_eq!(deserialized.unchanged, 5);
        assert!((deserialized.merged[0].overlap - 0.85).abs() < f64::EPSILON);
        assert_eq!(deserialized.grown[0].notes_added, 3);
        assert_eq!(deserialized.shrunk[0].notes_removed, 2);
        assert_eq!(deserialized.split[0].new_skill_ids.len(), 2);
    }

    #[test]
    fn test_merge_event_serialization() {
        let event = MergeEvent {
            survivor_id: Uuid::nil(),
            absorbed_id: Uuid::nil(),
            overlap: 0.75,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("survivor_id"));
        assert!(json.contains("absorbed_id"));
        assert!(json.contains("0.75"));
    }

    #[test]
    fn test_split_event_serialization() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let event = SplitEvent {
            original_id: Uuid::nil(),
            new_skill_ids: vec![id1, id2],
        };
        let json = serde_json::to_string(&event).unwrap();
        let de: SplitEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(de.new_skill_ids.len(), 2);
        assert_eq!(de.original_id, Uuid::nil());
    }

    #[test]
    fn test_grow_event_serialization() {
        let event = GrowEvent {
            skill_id: Uuid::nil(),
            notes_added: 10,
        };
        let json = serde_json::to_string(&event).unwrap();
        let de: GrowEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(de.notes_added, 10);
    }

    #[test]
    fn test_shrink_event_serialization() {
        let event = ShrinkEvent {
            skill_id: Uuid::nil(),
            notes_removed: 7,
        };
        let json = serde_json::to_string(&event).unwrap();
        let de: ShrinkEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(de.notes_removed, 7);
    }

    #[test]
    fn test_fission_candidate_serialization() {
        let fc = FissionCandidate {
            skill_id: Uuid::nil(),
            skill_name: "test-skill".to_string(),
            member_count: 8,
            cohesion: 0.6,
            sub_cluster_count: 3,
            sub_cluster_sizes: vec![3, 3, 2],
            blocked_by_guardrail: false,
            guardrail_reason: None,
        };
        let json = serde_json::to_string(&fc).unwrap();
        let de: FissionCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(de.skill_name, "test-skill");
        assert_eq!(de.member_count, 8);
        assert_eq!(de.sub_cluster_count, 3);
        assert_eq!(de.sub_cluster_sizes, vec![3, 3, 2]);
        assert!(!de.blocked_by_guardrail);
        assert!(de.guardrail_reason.is_none());
    }

    #[test]
    fn test_fission_candidate_with_guardrail_block() {
        let fc = FissionCandidate {
            skill_id: Uuid::nil(),
            skill_name: "tiny-skill".to_string(),
            member_count: 3,
            cohesion: 0.5,
            sub_cluster_count: 2,
            sub_cluster_sizes: vec![2, 1],
            blocked_by_guardrail: true,
            guardrail_reason: Some(format!(
                "Skill has {} members, minimum {} required for split",
                3, MIN_MEMBERS_FOR_SPLIT
            )),
        };
        let json = serde_json::to_string(&fc).unwrap();
        let de: FissionCandidate = serde_json::from_str(&json).unwrap();
        assert!(de.blocked_by_guardrail);
        assert!(de.guardrail_reason.unwrap().contains("minimum"));
    }

    #[test]
    fn test_fusion_candidate_serialization() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let fc = FusionCandidate {
            skill_ids: vec![id_a, id_b],
            skill_names: vec!["skill-a".to_string(), "skill-b".to_string()],
            overlap: 0.82,
            combined_member_count: 12,
            blocked_by_guardrail: false,
            guardrail_reason: None,
        };
        let json = serde_json::to_string(&fc).unwrap();
        let de: FusionCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(de.skill_ids.len(), 2);
        assert_eq!(de.skill_names, vec!["skill-a", "skill-b"]);
        assert!((de.overlap - 0.82).abs() < f64::EPSILON);
        assert_eq!(de.combined_member_count, 12);
    }

    #[test]
    fn test_fusion_candidate_with_guardrail_block() {
        let fc = FusionCandidate {
            skill_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            skill_names: vec!["big-a".to_string(), "big-b".to_string()],
            overlap: 0.9,
            combined_member_count: 25,
            blocked_by_guardrail: true,
            guardrail_reason: Some(format!(
                "Combined {} members exceeds maximum {}",
                25, MAX_MEMBERS_AFTER_MERGE
            )),
        };
        assert!(fc.blocked_by_guardrail);
        assert!(fc
            .guardrail_reason
            .as_ref()
            .unwrap()
            .contains("exceeds maximum"));
    }

    // ================================================================
    // Constants tests
    // ================================================================

    #[test]
    fn test_guardrail_constants() {
        assert_eq!(MIN_MEMBERS_FOR_SPLIT, 4);
        assert_eq!(MAX_MEMBERS_AFTER_MERGE, 20);
        assert!((MIN_COHESION_THRESHOLD - 0.15).abs() < f64::EPSILON);
    }

    // ================================================================
    // SkillEvolution enum coverage
    // ================================================================

    #[test]
    fn test_skill_evolution_enum_variants_debug() {
        let stable = SkillEvolution::Stable {
            skill_id: Uuid::nil(),
            candidate: make_candidate(0, vec!["a"]),
            notes_to_add: vec!["b".into()],
            notes_to_remove: vec!["c".into()],
        };
        let debug_str = format!("{:?}", stable);
        assert!(debug_str.contains("Stable"));

        let merge = SkillEvolution::Merge {
            skill_ids: vec![Uuid::nil()],
            candidate: make_candidate(0, vec!["a"]),
        };
        assert!(format!("{:?}", merge).contains("Merge"));

        let split = SkillEvolution::Split {
            skill_id: Uuid::nil(),
            candidates: vec![make_candidate(0, vec!["a"]), make_candidate(1, vec!["b"])],
        };
        assert!(format!("{:?}", split).contains("Split"));

        let new = SkillEvolution::New {
            candidate: make_candidate(0, vec!["a"]),
        };
        assert!(format!("{:?}", new).contains("New"));

        let orphan = SkillEvolution::Orphan {
            skill_id: Uuid::nil(),
        };
        assert!(format!("{:?}", orphan).contains("Orphan"));
    }

    // ================================================================
    // Additional analyze_evolution edge case tests
    // ================================================================

    #[test]
    fn test_evolution_empty_inputs() {
        let existing: Vec<(Uuid, Vec<String>)> = vec![];
        let candidates: Vec<SkillCandidate> = vec![];
        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert!(evolutions.is_empty());
    }

    #[test]
    fn test_evolution_multiple_new_clusters_no_existing() {
        let existing: Vec<(Uuid, Vec<String>)> = vec![];
        let candidates = vec![
            make_candidate(0, vec!["a", "b"]),
            make_candidate(1, vec!["c", "d"]),
            make_candidate(2, vec!["e", "f"]),
        ];
        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 3);
        for ev in &evolutions {
            assert!(matches!(ev, SkillEvolution::New { .. }));
        }
    }

    #[test]
    fn test_evolution_multiple_orphans_no_candidates() {
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let existing = vec![
            (s1, vec!["a".into(), "b".into()]),
            (s2, vec!["c".into(), "d".into()]),
        ];
        let candidates: Vec<SkillCandidate> = vec![];
        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        assert_eq!(evolutions.len(), 2);
        for ev in &evolutions {
            assert!(matches!(ev, SkillEvolution::Orphan { .. }));
        }
    }

    #[test]
    fn test_evolution_grow_and_shrink_same_skill() {
        let skill_id = Uuid::new_v4();
        // Old: {a, b, c, d} → New: {a, b, c, e}
        // Jaccard = 3/5 = 0.6 — use 0.5 threshold so it matches
        let existing = vec![(
            skill_id,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        )];
        let candidates = vec![make_candidate(0, vec!["a", "b", "c", "e"])];

        let evolutions = analyze_evolution(&existing, &candidates, 0.5);
        assert_eq!(evolutions.len(), 1);
        match &evolutions[0] {
            SkillEvolution::Stable {
                notes_to_add,
                notes_to_remove,
                ..
            } => {
                assert_eq!(notes_to_add, &vec!["e".to_string()]);
                assert_eq!(notes_to_remove, &vec!["d".to_string()]);
            }
            other => panic!("Expected Stable, got {:?}", other),
        }
    }

    #[test]
    fn test_evolution_low_threshold_matches_more() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];
        // Candidate: {a, b, x} → Jaccard vs {a, b, c} = 2/4 = 0.5
        let candidates = vec![make_candidate(0, vec!["a", "b", "x"])];

        // At 0.7 threshold → orphan + new (0.5 < 0.7)
        let evolutions_high = analyze_evolution(&existing, &candidates, 0.7);
        let has_orphan = evolutions_high
            .iter()
            .any(|e| matches!(e, SkillEvolution::Orphan { .. }));
        assert!(has_orphan);

        // At 0.4 threshold → stable (0.5 >= 0.4)
        let evolutions_low = analyze_evolution(&existing, &candidates, 0.4);
        let has_stable = evolutions_low
            .iter()
            .any(|e| matches!(e, SkillEvolution::Stable { .. }));
        assert!(has_stable);
    }

    #[test]
    fn test_evolution_cohesion_at_threshold_boundary() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];

        // Cohesion exactly at threshold — should remain Stable
        let mut candidate = make_candidate(0, vec!["a", "b", "c"]);
        candidate.cohesion = MIN_COHESION_THRESHOLD; // 0.15 — exactly at boundary
        let candidates = vec![candidate];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        let has_stable = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::Stable { .. }));
        assert!(has_stable, "Cohesion at threshold should stay Stable");
    }

    #[test]
    fn test_evolution_cohesion_just_below_threshold() {
        let skill_id = Uuid::new_v4();
        let existing = vec![(skill_id, vec!["a".into(), "b".into(), "c".into()])];

        let mut candidate = make_candidate(0, vec!["a", "b", "c"]);
        candidate.cohesion = MIN_COHESION_THRESHOLD - 0.01; // just below
        let candidates = vec![candidate];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        let has_orphan = evolutions
            .iter()
            .any(|e| matches!(e, SkillEvolution::Orphan { .. }));
        assert!(has_orphan, "Cohesion just below threshold should be Orphan");
    }

    #[test]
    fn test_evolution_merge_exactly_at_member_limit() {
        // Two skills whose combined members = exactly MAX_MEMBERS_AFTER_MERGE (20)
        let skill_a = Uuid::new_v4();
        let skill_b = Uuid::new_v4();
        let members_a: Vec<String> = (0..10).map(|i| format!("a{}", i)).collect();
        let members_b: Vec<String> = (0..10).map(|i| format!("b{}", i)).collect();
        let existing = vec![(skill_a, members_a.clone()), (skill_b, members_b.clone())];

        let mut all_members: Vec<&str> = members_a.iter().map(|s| s.as_str()).collect();
        all_members.extend(members_b.iter().map(|s| s.as_str()));
        let candidates = vec![make_candidate(0, all_members)];

        let evolutions = analyze_evolution(&existing, &candidates, 0.5);
        // Combined = 20, not > 20, so merge should proceed
        let merge_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Merge { .. }))
            .count();
        assert_eq!(merge_count, 1, "Merge at exact limit should be allowed");
    }

    #[test]
    fn test_evolution_split_exactly_at_member_threshold() {
        let skill_id = Uuid::new_v4();
        // Skill with exactly MIN_MEMBERS_FOR_SPLIT (4) members — should allow split
        let existing = vec![(
            skill_id,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        )];
        // Two candidates both overlapping with the skill
        // Candidate 0: {a, b, c} → Jaccard vs {a,b,c,d} = 3/4 = 0.75
        // Candidate 1: {b, c, d} → Jaccard vs {a,b,c,d} = 3/4 = 0.75
        let candidates = vec![
            make_candidate(0, vec!["a", "b", "c"]),
            make_candidate(1, vec!["b", "c", "d"]),
        ];

        let evolutions = analyze_evolution(&existing, &candidates, 0.7);
        let split_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Split { .. }))
            .count();
        assert_eq!(
            split_count, 1,
            "Split at exact member threshold should be allowed"
        );
    }

    // ================================================================
    // execute_evolution tests (async with MockGraphStore)
    // ================================================================

    use crate::neo4j::mock::MockGraphStore;
    use crate::skills::SkillNode;
    use std::sync::Arc;

    fn make_test_skill(project_id: Uuid) -> SkillNode {
        SkillNode::new(project_id, "test-skill")
    }

    fn make_test_note(project_id: Uuid, id: Uuid) -> crate::notes::Note {
        crate::notes::Note {
            id,
            project_id: Some(project_id),
            note_type: NoteType::Observation,
            status: NoteStatus::Active,
            importance: NoteImportance::Medium,
            scope: NoteScope::Project,
            content: format!("Test note {}", id),
            tags: vec!["test".to_string()],
            anchors: vec![],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy: 0.5,
            last_activated: None,
            reactivation_count: 0,
            last_reactivated: None,
            freshness_pinged_at: None,
            activation_count: 0,
            scar_intensity: 0.0,
            memory_horizon: MemoryHorizon::Operational,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
            sharing_consent: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_execute_evolution_stable_unchanged() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create project so skill creation works
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        let candidate = make_candidate(0, vec!["a", "b", "c"]);
        let evolutions = vec![SkillEvolution::Stable {
            skill_id: skill.id,
            candidate,
            notes_to_add: vec![],
            notes_to_remove: vec![],
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        assert_eq!(result.unchanged, 1);
        assert!(result.grown.is_empty());
        assert!(result.shrunk.is_empty());
    }

    #[tokio::test]
    async fn test_execute_evolution_stable_grow() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        // Create a note to add
        let note_id = Uuid::new_v4();
        let note = make_test_note(project_id, note_id);
        store.create_note(&note).await.unwrap();

        let candidate = make_candidate(0, vec!["a", "b"]);
        let evolutions = vec![SkillEvolution::Stable {
            skill_id: skill.id,
            candidate,
            notes_to_add: vec![note_id.to_string()],
            notes_to_remove: vec![],
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        assert_eq!(result.grown.len(), 1);
        assert_eq!(result.grown[0].skill_id, skill.id);
        assert_eq!(result.grown[0].notes_added, 1);
    }

    #[tokio::test]
    async fn test_execute_evolution_stable_shrink() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        let note_id = Uuid::new_v4();
        let note = make_test_note(project_id, note_id);
        store.create_note(&note).await.unwrap();
        store
            .add_skill_member(skill.id, "note", note_id)
            .await
            .unwrap();

        let candidate = make_candidate(0, vec!["a"]);
        let evolutions = vec![SkillEvolution::Stable {
            skill_id: skill.id,
            candidate,
            notes_to_add: vec![],
            notes_to_remove: vec![note_id.to_string()],
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        assert_eq!(result.shrunk.len(), 1);
        assert_eq!(result.shrunk[0].skill_id, skill.id);
        assert_eq!(result.shrunk[0].notes_removed, 1);
    }

    #[tokio::test]
    async fn test_execute_evolution_stable_grow_and_shrink() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        let old_note_id = Uuid::new_v4();
        let old_note = make_test_note(project_id, old_note_id);
        store.create_note(&old_note).await.unwrap();
        store
            .add_skill_member(skill.id, "note", old_note_id)
            .await
            .unwrap();

        let new_note_id = Uuid::new_v4();
        let new_note = make_test_note(project_id, new_note_id);
        store.create_note(&new_note).await.unwrap();

        let candidate = make_candidate(0, vec!["a"]);
        let evolutions = vec![SkillEvolution::Stable {
            skill_id: skill.id,
            candidate,
            notes_to_add: vec![new_note_id.to_string()],
            notes_to_remove: vec![old_note_id.to_string()],
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        assert_eq!(result.grown.len(), 1);
        assert_eq!(result.shrunk.len(), 1);
    }

    #[tokio::test]
    async fn test_execute_evolution_orphan() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let mut skill = make_test_skill(project_id);
        skill.status = crate::skills::SkillStatus::Active;
        store.create_skill(&skill).await.unwrap();

        let evolutions = vec![SkillEvolution::Orphan { skill_id: skill.id }];

        let _result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();

        // Verify skill is archived
        let updated = store.get_skill(skill.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::skills::SkillStatus::Archived);
    }

    #[tokio::test]
    async fn test_execute_evolution_orphan_already_archived() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let mut skill = make_test_skill(project_id);
        skill.status = crate::skills::SkillStatus::Archived;
        store.create_skill(&skill).await.unwrap();

        let evolutions = vec![SkillEvolution::Orphan { skill_id: skill.id }];

        // Should not fail on already-archived skill
        let _result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        // No error — gracefully skips already-archived skills
    }

    #[tokio::test]
    async fn test_execute_evolution_new_skill() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let note_id = Uuid::new_v4();
        let note = make_test_note(project_id, note_id);
        store.create_note(&note).await.unwrap();

        let mut notes_map = HashMap::new();
        notes_map.insert(note_id.to_string(), note);

        let candidate = SkillCandidate {
            community_id: 0,
            member_note_ids: vec![note_id.to_string()],
            cohesion: 0.8,
            size: 1,
            label: "new-cluster".to_string(),
        };

        let evolutions = vec![SkillEvolution::New { candidate }];

        let _result = execute_evolution(&*store, &evolutions, &notes_map, project_id)
            .await
            .unwrap();

        // Verify a new skill was created in the store
        let skills = store.get_skills_for_project(project_id).await.unwrap();
        assert_eq!(skills.len(), 1);
    }

    #[tokio::test]
    async fn test_execute_evolution_merge() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Create two skills — skill_a has higher energy
        let mut skill_a = make_test_skill(project_id);
        skill_a.energy = 0.9;
        skill_a.name = "skill-a".to_string();
        store.create_skill(&skill_a).await.unwrap();

        let mut skill_b = make_test_skill(project_id);
        skill_b.energy = 0.3;
        skill_b.name = "skill-b".to_string();
        store.create_skill(&skill_b).await.unwrap();

        // Add notes to skill_b (to be transferred)
        let note_id = Uuid::new_v4();
        let note = make_test_note(project_id, note_id);
        store.create_note(&note).await.unwrap();
        store
            .add_skill_member(skill_b.id, "note", note_id)
            .await
            .unwrap();

        let candidate = SkillCandidate {
            community_id: 0,
            member_note_ids: vec![note_id.to_string()],
            cohesion: 0.75,
            size: 3,
            label: "merged-cluster".to_string(),
        };

        let evolutions = vec![SkillEvolution::Merge {
            skill_ids: vec![skill_a.id, skill_b.id],
            candidate,
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();

        assert_eq!(result.merged.len(), 1);
        assert_eq!(result.merged[0].survivor_id, skill_a.id);
        assert_eq!(result.merged[0].absorbed_id, skill_b.id);

        // Verify skill_b is archived
        let updated_b = store.get_skill(skill_b.id).await.unwrap().unwrap();
        assert_eq!(updated_b.status, crate::skills::SkillStatus::Archived);

        // Verify survivor's version was incremented
        let updated_a = store.get_skill(skill_a.id).await.unwrap().unwrap();
        assert!(updated_a.version > skill_a.version);
    }

    #[tokio::test]
    async fn test_execute_evolution_split() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        // Create notes for sub-clusters
        let note_a = Uuid::new_v4();
        let note_b = Uuid::new_v4();
        let na = make_test_note(project_id, note_a);
        let nb = make_test_note(project_id, note_b);
        store.create_note(&na).await.unwrap();
        store.create_note(&nb).await.unwrap();
        store
            .add_skill_member(skill.id, "note", note_a)
            .await
            .unwrap();
        store
            .add_skill_member(skill.id, "note", note_b)
            .await
            .unwrap();

        let mut notes_map = HashMap::new();
        notes_map.insert(note_a.to_string(), na);
        notes_map.insert(note_b.to_string(), nb);

        let cand1 = SkillCandidate {
            community_id: 0,
            member_note_ids: vec![note_a.to_string()],
            cohesion: 0.8,
            size: 1,
            label: "sub-a".to_string(),
        };
        let cand2 = SkillCandidate {
            community_id: 1,
            member_note_ids: vec![note_b.to_string()],
            cohesion: 0.8,
            size: 1,
            label: "sub-b".to_string(),
        };

        let evolutions = vec![SkillEvolution::Split {
            skill_id: skill.id,
            candidates: vec![cand1, cand2],
        }];

        let result = execute_evolution(&*store, &evolutions, &notes_map, project_id)
            .await
            .unwrap();

        assert_eq!(result.split.len(), 1);
        assert_eq!(result.split[0].original_id, skill.id);
        assert_eq!(result.split[0].new_skill_ids.len(), 2);

        // Original skill should be archived
        let original = store.get_skill(skill.id).await.unwrap().unwrap();
        assert_eq!(original.status, crate::skills::SkillStatus::Archived);

        // New skills should exist
        let all_skills = store.get_skills_for_project(project_id).await.unwrap();
        assert_eq!(all_skills.len(), 3); // original (archived) + 2 new
    }

    #[tokio::test]
    async fn test_execute_evolution_mixed() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Stable unchanged skill
        let skill_stable = make_test_skill(project_id);
        store.create_skill(&skill_stable).await.unwrap();

        // Orphan skill
        let mut skill_orphan = make_test_skill(project_id);
        skill_orphan.status = crate::skills::SkillStatus::Active;
        store.create_skill(&skill_orphan).await.unwrap();

        let candidate = make_candidate(0, vec!["a"]);
        let evolutions = vec![
            SkillEvolution::Stable {
                skill_id: skill_stable.id,
                candidate,
                notes_to_add: vec![],
                notes_to_remove: vec![],
            },
            SkillEvolution::Orphan {
                skill_id: skill_orphan.id,
            },
        ];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();

        assert_eq!(result.unchanged, 1);
        let orphan_archived = store.get_skill(skill_orphan.id).await.unwrap().unwrap();
        assert_eq!(orphan_archived.status, crate::skills::SkillStatus::Archived);
    }

    #[tokio::test]
    async fn test_execute_evolution_stable_with_invalid_uuid_in_notes() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill = make_test_skill(project_id);
        store.create_skill(&skill).await.unwrap();

        let candidate = make_candidate(0, vec!["a"]);
        // "not-a-uuid" will fail Uuid::parse_str — should not crash
        let evolutions = vec![SkillEvolution::Stable {
            skill_id: skill.id,
            candidate,
            notes_to_add: vec!["not-a-uuid".to_string()],
            notes_to_remove: vec!["also-not-uuid".to_string()],
        }];

        let result = execute_evolution(&*store, &evolutions, &HashMap::new(), project_id)
            .await
            .unwrap();
        // Still reports grow/shrink based on list lengths
        assert_eq!(result.grown.len(), 1);
        assert_eq!(result.shrunk.len(), 1);
    }

    // ================================================================
    // detect_skill_fusion tests (async with MockGraphStore)
    // ================================================================

    #[tokio::test]
    async fn test_detect_skill_fusion_no_skills() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let result = detect_skill_fusion(&*store, project_id, 0.7).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_detect_skill_fusion_no_overlap() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Two skills with completely different members
        let skill_a = make_test_skill(project_id);
        store.create_skill(&skill_a).await.unwrap();

        let mut skill_b = make_test_skill(project_id);
        skill_b.name = "skill-b".to_string();
        store.create_skill(&skill_b).await.unwrap();

        let note_a = Uuid::new_v4();
        let note_b = Uuid::new_v4();
        let na = make_test_note(project_id, note_a);
        let nb = make_test_note(project_id, note_b);
        store.create_note(&na).await.unwrap();
        store.create_note(&nb).await.unwrap();
        store
            .add_skill_member(skill_a.id, "note", note_a)
            .await
            .unwrap();
        store
            .add_skill_member(skill_b.id, "note", note_b)
            .await
            .unwrap();

        let result = detect_skill_fusion(&*store, project_id, 0.7).await.unwrap();
        assert!(
            result.is_empty(),
            "No overlap should produce no fusion candidates"
        );
    }

    #[tokio::test]
    async fn test_detect_skill_fusion_with_overlap() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Two skills sharing the same member notes
        let skill_a = make_test_skill(project_id);
        store.create_skill(&skill_a).await.unwrap();

        let mut skill_b = make_test_skill(project_id);
        skill_b.name = "skill-b".to_string();
        store.create_skill(&skill_b).await.unwrap();

        // Both skills share the same notes → Jaccard = 1.0
        let note1 = Uuid::new_v4();
        let note2 = Uuid::new_v4();
        let n1 = make_test_note(project_id, note1);
        let n2 = make_test_note(project_id, note2);
        store.create_note(&n1).await.unwrap();
        store.create_note(&n2).await.unwrap();

        store
            .add_skill_member(skill_a.id, "note", note1)
            .await
            .unwrap();
        store
            .add_skill_member(skill_a.id, "note", note2)
            .await
            .unwrap();
        store
            .add_skill_member(skill_b.id, "note", note1)
            .await
            .unwrap();
        store
            .add_skill_member(skill_b.id, "note", note2)
            .await
            .unwrap();

        let result = detect_skill_fusion(&*store, project_id, 0.7).await.unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0].overlap - 1.0).abs() < f64::EPSILON);
        assert_eq!(result[0].combined_member_count, 2);
        assert!(!result[0].blocked_by_guardrail);
    }

    #[tokio::test]
    async fn test_detect_skill_fusion_skips_archived() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let skill_a = make_test_skill(project_id);
        store.create_skill(&skill_a).await.unwrap();

        let mut skill_b = make_test_skill(project_id);
        skill_b.status = crate::skills::SkillStatus::Archived;
        store.create_skill(&skill_b).await.unwrap();

        // Even though they share members, archived skill should be skipped
        let note1 = Uuid::new_v4();
        let n1 = make_test_note(project_id, note1);
        store.create_note(&n1).await.unwrap();
        store
            .add_skill_member(skill_a.id, "note", note1)
            .await
            .unwrap();
        store
            .add_skill_member(skill_b.id, "note", note1)
            .await
            .unwrap();

        let result = detect_skill_fusion(&*store, project_id, 0.7).await.unwrap();
        assert!(
            result.is_empty(),
            "Archived skills should be excluded from fusion detection"
        );
    }

    // ================================================================
    // detect_skill_fission tests (async with MockGraphStore)
    // ================================================================

    #[tokio::test]
    async fn test_detect_skill_fission_empty_graph() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Mock store returns empty synapse graph → InsufficientData → empty result
        let result = detect_skill_fission(&*store, project_id, 0.5)
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_evolution_split_one_skill_into_two_clusters() {
        let skill_id = Uuid::new_v4();
        // Skill has 6 members that will split into 2 distinct clusters
        let existing = vec![(
            skill_id,
            vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "x".into(),
                "y".into(),
                "z".into(),
            ],
        )];

        // Two new clusters, each overlapping >= 0.5 with the original
        // but each representing a sub-cluster
        // Candidate 0: {a, b, c, x} → Jaccard vs {a,b,c,x,y,z} = 4/6 = 0.67 (≥ 0.5)
        // Candidate 1: {x, y, z, a} → Jaccard vs {a,b,c,x,y,z} = 4/6 = 0.67 (≥ 0.5)
        let candidates = vec![
            make_candidate(0, vec!["a", "b", "c", "x"]),
            make_candidate(1, vec!["x", "y", "z", "a"]),
        ];

        // Use a low threshold (0.5) so both candidates match the single skill
        let evolutions = analyze_evolution(&existing, &candidates, 0.5);

        let split_count = evolutions
            .iter()
            .filter(|e| matches!(e, SkillEvolution::Split { .. }))
            .count();
        assert_eq!(
            split_count,
            1,
            "Expected 1 split, got evolutions: {:?}",
            evolutions
                .iter()
                .map(|e| match e {
                    SkillEvolution::Stable { skill_id, .. } => format!("Stable({})", skill_id),
                    SkillEvolution::Merge { skill_ids, .. } => format!("Merge({:?})", skill_ids),
                    SkillEvolution::Split { skill_id, .. } => format!("Split({})", skill_id),
                    SkillEvolution::New { .. } => "New".to_string(),
                    SkillEvolution::Orphan { skill_id } => format!("Orphan({})", skill_id),
                })
                .collect::<Vec<_>>()
        );

        // Verify the split details
        match &evolutions
            .iter()
            .find(|e| matches!(e, SkillEvolution::Split { .. }))
            .unwrap()
        {
            SkillEvolution::Split {
                skill_id: sid,
                candidates: split_candidates,
            } => {
                assert_eq!(*sid, skill_id);
                assert_eq!(split_candidates.len(), 2);
            }
            _ => unreachable!(),
        }
    }
}
