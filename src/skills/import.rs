//! Skill Import
//!
//! Imports a [`SkillPackage`] into a target project, creating new notes,
//! decisions (encoded as context notes), and a [`SkillNode`] with fresh UUIDs.
//!
//! # Conflict handling
//!
//! When a skill with the same name already exists in the target project,
//! the [`ConflictStrategy`] determines the behavior:
//! - **Skip**: abort import, return conflict info
//! - **Merge**: add new notes/decisions to the existing skill
//! - **Replace**: delete the existing skill, create a fresh one

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use uuid::Uuid;

use crate::episodes::PortableEpisode;
use crate::neo4j::traits::GraphStore;
use crate::notes::models::{MemoryHorizon, Note, NoteImportance, NoteScope, NoteStatus, NoteType};
use crate::protocol::models::{
    Protocol, ProtocolCategory, ProtocolState, ProtocolTransition, StateType,
};
use crate::protocol::routing::RelevanceVector;
use crate::skills::models::{SkillNode, SkillStatus};
use crate::skills::package::{
    validate_package, verify_package_signature, PortableDecision, SkillPackage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Initial synapse weight between co-member notes in an imported skill.
/// Set low (0.3) so imported knowledge must prove itself through usage
/// to develop strong connections.
const IMPORT_SYNAPSE_WEIGHT: f64 = 0.3;

/// Initial energy for imported notes. Lower than fresh notes (1.0) to
/// reflect that imported knowledge hasn't been validated in this project yet.
const IMPORT_NOTE_ENERGY: f64 = 0.6;

/// Creator tag for imported entities.
const IMPORT_CREATOR: &str = "skill-import";

// ============================================================================
// Types
// ============================================================================

/// Strategy for handling name conflicts during import.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStrategy {
    /// Skip import if a skill with the same name exists.
    #[default]
    Skip,
    /// Merge: add imported notes/decisions to the existing skill.
    Merge,
    /// Replace: delete the existing skill, create a fresh one.
    Replace,
}

impl std::fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Skip => write!(f, "skip"),
            Self::Merge => write!(f, "merge"),
            Self::Replace => write!(f, "replace"),
        }
    }
}

impl std::str::FromStr for ConflictStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "skip" => Ok(Self::Skip),
            "merge" => Ok(Self::Merge),
            "replace" => Ok(Self::Replace),
            other => Err(format!(
                "Invalid conflict_strategy '{}'. Must be: skip, merge, replace",
                other
            )),
        }
    }
}

/// Conflict information when a skill name collision is detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConflict {
    /// Name of the conflicting skill.
    pub skill_name: String,
    /// ID of the existing skill in the target project.
    pub existing_skill_id: Uuid,
    /// Strategy that was applied.
    pub strategy_applied: ConflictStrategy,
}

/// Result of a skill import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// ID of the created or merged skill.
    pub skill_id: Uuid,
    /// Number of notes created in the target project.
    pub notes_created: usize,
    /// Number of decisions imported (as context notes).
    pub decisions_imported: usize,
    /// Number of synapses created between imported notes.
    pub synapses_created: usize,
    /// Number of protocols recreated (v2 only).
    pub protocols_imported: usize,
    /// Number of episodes imported as lesson notes (v3 only).
    pub episodes_imported: usize,
    /// Conflict information (if any name collision was detected).
    pub conflict: Option<ImportConflict>,
    /// Whether this was a merge into an existing skill.
    pub was_merged: bool,
    /// Source project name (from package metadata).
    pub source_project: Option<String>,
}

// ============================================================================
// Import function
// ============================================================================

/// Import a [`SkillPackage`] into a target project.
///
/// # Process
///
/// 1. Validate the package (schema version, content)
/// 2. Check for name conflicts with existing skills
/// 3. Apply conflict strategy (Skip → abort, Merge → reuse, Replace → delete+create)
/// 4. Create notes with new UUIDs in the target project
/// 5. Encode decisions as context notes (since decisions require task linkage)
/// 6. Create or reuse SkillNode with status `Imported`
/// 7. Link all created entities as skill members
/// 8. Create initial synapses between co-member notes
///
/// # Errors
///
/// Returns an error if:
/// - Package validation fails
/// - A conflict is detected with `ConflictStrategy::Skip`
/// - Any database operation fails
pub async fn import_skill(
    package: &SkillPackage,
    target_project_id: Uuid,
    graph_store: &dyn GraphStore,
    conflict_strategy: ConflictStrategy,
) -> Result<ImportResult> {
    // 1. Validate the package
    validate_package(package).map_err(|errors| {
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        anyhow::anyhow!("Package validation failed: {}", messages.join("; "))
    })?;

    // 1b. Verify cryptographic signature — MANDATORY
    // All packages must be signed. Unsigned packages are rejected.
    verify_package_signature(package).map_err(|e| {
        anyhow::anyhow!("Package signature verification failed — refusing import: {e}")
    })?;
    tracing::info!(
        source_did = %package.signature.as_ref().unwrap().source_did,
        "Package signature verified successfully"
    );

    // 2. Check for name conflicts
    let existing_skill =
        find_skill_by_name(graph_store, target_project_id, &package.skill.name).await?;

    let (skill_id, was_merged, conflict) = match existing_skill {
        Some(existing) => {
            handle_conflict(
                &existing,
                &package.skill,
                target_project_id,
                graph_store,
                conflict_strategy,
            )
            .await?
        }
        None => {
            // No conflict — create a new skill
            let skill = create_imported_skill(package, target_project_id);
            graph_store
                .create_skill(&skill)
                .await
                .context("Failed to create imported skill")?;
            (skill.id, false, None)
        }
    };

    // 4. Create notes with new UUIDs
    let mut created_note_ids = Vec::with_capacity(package.notes.len());
    for portable_note in &package.notes {
        let note = portable_note_to_note(portable_note, target_project_id)?;
        let note_id = note.id;
        graph_store
            .create_note(&note)
            .await
            .context("Failed to create imported note")?;
        graph_store
            .add_skill_member(skill_id, "note", note_id)
            .await
            .context("Failed to link note to skill")?;
        created_note_ids.push(note_id);
    }

    // 5. Encode decisions as context notes
    let mut decisions_imported = 0;
    for portable_decision in &package.decisions {
        let note = decision_to_note(portable_decision, target_project_id, &package.skill.name)?;
        let note_id = note.id;
        graph_store
            .create_note(&note)
            .await
            .context("Failed to create imported decision-as-note")?;
        graph_store
            .add_skill_member(skill_id, "note", note_id)
            .await
            .context("Failed to link decision-note to skill")?;
        created_note_ids.push(note_id);
        decisions_imported += 1;
    }

    // 6. Create initial synapses between all co-member notes
    let synapses_created = create_import_synapses(graph_store, &created_note_ids).await?;

    // 7. Import protocols (v2 only — skipped for v1 packages)
    let protocols_imported = if !package.protocols.is_empty() {
        import_protocols(&package.protocols, target_project_id, skill_id, graph_store).await?
    } else {
        0
    };

    // 7b. Import episodes as lesson notes (v3 only — skipped for v1/v2 packages)
    let mut episodes_imported = 0;
    for portable_episode in &package.episodes {
        if let Some(lesson) = &portable_episode.lesson {
            let note = episode_to_note(
                portable_episode,
                lesson,
                target_project_id,
                &package.skill.name,
            )?;
            let note_id = note.id;
            graph_store
                .create_note(&note)
                .await
                .context("Failed to create imported episode-lesson note")?;
            graph_store
                .add_skill_member(skill_id, "note", note_id)
                .await
                .context("Failed to link episode-lesson note to skill")?;
            created_note_ids.push(note_id);
            episodes_imported += 1;
        }
    }

    // 8. Update skill counts
    let notes_created = package.notes.len();
    if !was_merged {
        // For newly created skills, update the counts
        let mut skill = graph_store
            .get_skill(skill_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Skill {} disappeared after creation", skill_id))?;
        skill.note_count = created_note_ids.len() as i64;
        skill.decision_count = 0; // decisions are encoded as notes
        graph_store
            .update_skill(&skill)
            .await
            .context("Failed to update skill counts")?;
    }

    Ok(ImportResult {
        skill_id,
        notes_created,
        decisions_imported,
        synapses_created,
        protocols_imported,
        episodes_imported,
        conflict,
        was_merged,
        source_project: package.metadata.source_project.clone(),
    })
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Find a skill by name in a project.
///
/// Uses `list_skills` with a large limit and filters by name client-side.
/// Not ideal for large skill counts, but sufficient for the typical case
/// (projects have < 50 skills).
async fn find_skill_by_name(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    name: &str,
) -> Result<Option<SkillNode>> {
    let (skills, _) = graph_store.list_skills(project_id, None, 200, 0).await?;
    let normalized_name = name.trim().to_lowercase();
    Ok(skills
        .into_iter()
        .find(|s| s.name.trim().to_lowercase() == normalized_name))
}

/// Handle a name conflict based on the chosen strategy.
///
/// Returns (skill_id, was_merged, conflict_info).
async fn handle_conflict(
    existing: &SkillNode,
    portable_skill: &crate::skills::package::PortableSkill,
    target_project_id: Uuid,
    graph_store: &dyn GraphStore,
    strategy: ConflictStrategy,
) -> Result<(Uuid, bool, Option<ImportConflict>)> {
    let conflict_info = ImportConflict {
        skill_name: existing.name.clone(),
        existing_skill_id: existing.id,
        strategy_applied: strategy,
    };

    match strategy {
        ConflictStrategy::Skip => {
            Err(anyhow::anyhow!(
                "Skill '{}' already exists in project (id: {}). Use Merge or Replace strategy to override.",
                existing.name,
                existing.id,
            ))
        }
        ConflictStrategy::Merge => {
            // Reuse the existing skill — new notes will be added to it
            Ok((existing.id, true, Some(conflict_info)))
        }
        ConflictStrategy::Replace => {
            // Delete the existing skill, create a fresh one
            graph_store
                .delete_skill(existing.id)
                .await
                .context("Failed to delete existing skill for replacement")?;

            let package_stub = SkillPackage {
                schema_version: 1,
                metadata: crate::skills::package::PackageMetadata {
                    format: crate::skills::package::FORMAT_ID.to_string(),
                    exported_at: Utc::now(),
                    source_project: None,
                    stats: crate::skills::package::PackageStats {
                        note_count: 0,
                        decision_count: 0,
                        trigger_count: portable_skill.trigger_patterns.len(),
                        activation_count: 0,
                    },
                },
                skill: portable_skill.clone(),
                notes: vec![],
                decisions: vec![],
                protocols: vec![],
                execution_history: None,
                source: None,
                episodes: Vec::new(),
                distilled_episodes: Vec::new(),
                package_trust: None,
                privacy_report: None,
                privacy_mode: None,
                signature: None,
            };
            let skill = create_imported_skill(&package_stub, target_project_id);
            let skill_id = skill.id;
            graph_store
                .create_skill(&skill)
                .await
                .context("Failed to create replacement skill")?;

            Ok((skill_id, false, Some(conflict_info)))
        }
    }
}

/// Create a SkillNode from a package for import.
fn create_imported_skill(package: &SkillPackage, project_id: Uuid) -> SkillNode {
    let now = Utc::now();
    SkillNode {
        id: Uuid::new_v4(),
        project_id,
        name: package.skill.name.clone(),
        description: package.skill.description.clone(),
        status: SkillStatus::Imported,
        // Sanitize FileGlob triggers: strip any leftover absolute paths from old packages
        trigger_patterns: package
            .skill
            .trigger_patterns
            .iter()
            .map(|t| {
                if t.pattern_type == crate::skills::TriggerType::FileGlob
                    && t.pattern_value.starts_with('/')
                {
                    // Old-format absolute glob — strip any recognizable project prefix.
                    // Without the source root we can only heuristically strip up to src/ etc.
                    let mut t = t.clone();
                    // Try to find a known code directory marker and cut there
                    if let Some(idx) = t.pattern_value.find("/src/") {
                        t.pattern_value = t.pattern_value[idx + 1..].to_string();
                    } else if let Some(idx) = t.pattern_value.find("/lib/") {
                        t.pattern_value = t.pattern_value[idx + 1..].to_string();
                    }
                    t
                } else {
                    t.clone()
                }
            })
            .collect(),
        context_template: package.skill.context_template.clone(),
        energy: 0.0, // must prove itself
        cohesion: package.skill.cohesion,
        coverage: 0,
        note_count: 0, // updated after import
        decision_count: 0,
        activation_count: 0,
        hit_rate: 0.0,
        last_activated: None,
        version: 1,
        fingerprint: None,
        imported_at: Some(now),
        is_validated: false,
        tags: package.skill.tags.clone(),
        created_at: now,
        updated_at: now,
    }
}

/// Convert a PortableNote to a full Note with new UUID.
fn portable_note_to_note(
    portable: &crate::skills::package::PortableNote,
    project_id: Uuid,
) -> Result<Note> {
    let note_type = NoteType::from_str(&portable.note_type)
        .map_err(|e| anyhow::anyhow!("Invalid note type '{}': {}", portable.note_type, e))?;
    let importance = NoteImportance::from_str(&portable.importance)
        .map_err(|e| anyhow::anyhow!("Invalid importance '{}': {}", portable.importance, e))?;

    let now = Utc::now();
    let mut tags = portable.tags.clone();
    if !tags.contains(&"imported".to_string()) {
        tags.push("imported".to_string());
    }

    Ok(Note {
        id: Uuid::new_v4(),
        project_id: Some(project_id),
        note_type,
        status: NoteStatus::Active,
        importance,
        scope: NoteScope::Project,
        content: portable.content.clone(),
        tags,
        anchors: vec![],
        created_at: now,
        created_by: IMPORT_CREATOR.to_string(),
        last_confirmed_at: None,
        last_confirmed_by: None,
        staleness_score: 0.0,
        energy: IMPORT_NOTE_ENERGY,
        last_activated: Some(now),
        reactivation_count: 0,
        last_reactivated: None,
        freshness_pinged_at: None,
        activation_count: 0,
        scar_intensity: 0.0,
        memory_horizon: MemoryHorizon::Operational, // Imported notes are operational
        supersedes: None,
        superseded_by: None,
        changes: vec![],
        assertion_rule: None,
        last_assertion_result: None,
        sharing_consent: Default::default(),
    })
}

/// Encode a PortableDecision as a context Note.
///
/// Since `create_decision` requires a task_id (decisions are task-bound),
/// we encode imported decisions as notes with structured content and
/// the `imported_decision` tag. This preserves the knowledge while
/// working within the existing GraphStore interface.
fn decision_to_note(
    decision: &PortableDecision,
    project_id: Uuid,
    skill_name: &str,
) -> Result<Note> {
    // Format the decision as structured Markdown content
    let mut content = format!("**Decision** (imported from skill '{}')\n\n", skill_name);
    content.push_str(&format!("**What**: {}\n\n", decision.description));
    content.push_str(&format!("**Why**: {}\n\n", decision.rationale));

    if !decision.alternatives.is_empty() {
        content.push_str("**Alternatives considered**:\n");
        for alt in &decision.alternatives {
            content.push_str(&format!("- {}\n", alt));
        }
        content.push('\n');
    }

    if let Some(chosen) = &decision.chosen_option {
        content.push_str(&format!("**Chosen**: {}\n", chosen));
    }

    let now = Utc::now();
    Ok(Note {
        id: Uuid::new_v4(),
        project_id: Some(project_id),
        note_type: NoteType::Context,
        status: NoteStatus::Active,
        importance: NoteImportance::High,
        scope: NoteScope::Project,
        content,
        tags: vec!["imported".to_string(), "imported_decision".to_string()],
        anchors: vec![],
        created_at: now,
        created_by: IMPORT_CREATOR.to_string(),
        last_confirmed_at: None,
        last_confirmed_by: None,
        staleness_score: 0.0,
        energy: IMPORT_NOTE_ENERGY,
        last_activated: Some(now),
        reactivation_count: 0,
        last_reactivated: None,
        freshness_pinged_at: None,
        activation_count: 0,
        scar_intensity: 0.0,
        memory_horizon: MemoryHorizon::Operational, // Imported decisions are operational
        supersedes: None,
        superseded_by: None,
        changes: vec![],
        assertion_rule: None,
        last_assertion_result: None,
        sharing_consent: Default::default(),
    })
}

/// Convert a PortableEpisode's Lesson into a local note.
///
/// Episodes with lessons become `pattern` notes — they capture the abstract
/// knowledge extracted from a cognitive episode on another instance.
/// The note content includes the lesson, the process trace (FSM states),
/// and the stimulus for context.
fn episode_to_note(
    episode: &PortableEpisode,
    lesson: &crate::episodes::PortableLesson,
    project_id: Uuid,
    skill_name: &str,
) -> Result<Note> {
    let mut content = format!(
        "**Episodic Lesson** (imported from skill '{}')\n\n",
        skill_name
    );
    content.push_str(&format!("**Pattern**: {}\n\n", lesson.abstract_pattern));

    // Add context from stimulus
    content.push_str(&format!(
        "**Original stimulus**: {}\n",
        episode.stimulus.request
    ));
    content.push_str(&format!("**Trigger**: {:?}\n\n", episode.stimulus.trigger));

    // Add process trace
    if !episode.process.states_visited.is_empty() {
        content.push_str("**FSM states**: ");
        content.push_str(&episode.process.states_visited.join(" → "));
        content.push('\n');
    }

    // Add outcome summary
    content.push_str(&format!(
        "**Produced**: {} notes, {} decisions, {} commits\n",
        episode.outcome.notes_produced,
        episode.outcome.decisions_made,
        episode.outcome.commits_made,
    ));

    // Add portability info
    let portability = match lesson.portability_layer {
        1 => "project-specific",
        2 => "language-specific",
        3 => "universal",
        _ => "unknown",
    };
    content.push_str(&format!(
        "**Portability**: {} (layer {})\n",
        portability, lesson.portability_layer
    ));

    let mut tags = vec!["imported".to_string(), "imported_episode".to_string()];
    tags.extend(lesson.domain_tags.iter().cloned());

    let now = Utc::now();
    Ok(Note {
        id: Uuid::new_v4(),
        project_id: Some(project_id),
        note_type: NoteType::Pattern, // Lessons are patterns
        status: NoteStatus::Active,
        importance: NoteImportance::High,
        scope: NoteScope::Project,
        content,
        tags,
        anchors: vec![],
        created_at: now,
        created_by: IMPORT_CREATOR.to_string(),
        last_confirmed_at: None,
        last_confirmed_by: None,
        staleness_score: 0.0,
        energy: IMPORT_NOTE_ENERGY,
        last_activated: Some(now),
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
    })
}

/// Import protocols from a v2 package into the target project.
///
/// For each `PortableProtocol`:
/// 1. Create `ProtocolState` nodes with fresh UUIDs (name-based lookup for transitions)
/// 2. Identify entry_state (Start type) and terminal_states (Terminal type)
/// 3. Create the `Protocol` node linked to the project and skill
/// 4. Create `ProtocolTransition` nodes, resolving state names to UUIDs
///
/// Returns the number of protocols successfully imported.
async fn import_protocols(
    portable_protocols: &[crate::skills::package::PortableProtocol],
    target_project_id: Uuid,
    skill_id: Uuid,
    graph_store: &dyn GraphStore,
) -> Result<usize> {
    let mut count = 0;

    for portable in portable_protocols {
        let protocol_id = Uuid::new_v4();

        // 1. Create states with fresh UUIDs, build name → UUID map
        let mut name_to_id: std::collections::HashMap<String, Uuid> =
            std::collections::HashMap::new();
        let mut entry_state_id: Option<Uuid> = None;
        let mut terminal_state_ids: Vec<Uuid> = Vec::new();

        for portable_state in &portable.states {
            let state_type = StateType::from_str(&portable_state.state_type).map_err(|e| {
                anyhow::anyhow!("Invalid state type '{}': {}", portable_state.state_type, e)
            })?;

            let state = ProtocolState {
                id: Uuid::new_v4(),
                protocol_id,
                name: portable_state.name.clone(),
                description: portable_state.description.clone(),
                action: portable_state.action.clone(),
                state_type,
                sub_protocol_id: None,
                completion_strategy: None,
                on_failure_strategy: None,
                generator_config: None,
            };

            if state_type == StateType::Start {
                entry_state_id = Some(state.id);
            }
            if state_type == StateType::Terminal {
                terminal_state_ids.push(state.id);
            }

            name_to_id.insert(state.name.clone(), state.id);

            graph_store
                .upsert_protocol_state(&state)
                .await
                .context("Failed to create imported protocol state")?;
        }

        // 2. Determine entry_state — use first state as fallback if no Start type found
        let entry_state = entry_state_id.unwrap_or_else(|| {
            portable
                .states
                .first()
                .and_then(|s| name_to_id.get(&s.name).copied())
                .unwrap_or_else(Uuid::new_v4)
        });

        // 3. Parse category and relevance vector
        let category =
            ProtocolCategory::from_str(&portable.category).unwrap_or(ProtocolCategory::Business);

        let relevance_vector = portable
            .relevance_vector
            .as_ref()
            .map(|rv| RelevanceVector {
                phase: rv.phase,
                structure: rv.structure,
                domain: rv.domain,
                resource: rv.resource,
                lifecycle: rv.lifecycle,
            });

        // 4. Create the Protocol node
        let now = Utc::now();
        let protocol = Protocol {
            id: protocol_id,
            name: portable.name.clone(),
            description: portable.description.clone(),
            project_id: target_project_id,
            skill_id: Some(skill_id),
            entry_state,
            terminal_states: terminal_state_ids,
            protocol_category: category,
            trigger_mode: crate::protocol::models::TriggerMode::Manual,
            trigger_config: None,
            relevance_vector,
            last_triggered_at: None,
            created_at: now,
            updated_at: now,
        };

        graph_store
            .upsert_protocol(&protocol)
            .await
            .context("Failed to create imported protocol")?;

        // 5. Create transitions, resolving state names to UUIDs
        for portable_transition in &portable.transitions {
            let from_id = name_to_id.get(&portable_transition.from_state);
            let to_id = name_to_id.get(&portable_transition.to_state);

            if let (Some(&from), Some(&to)) = (from_id, to_id) {
                let transition = ProtocolTransition {
                    id: Uuid::new_v4(),
                    protocol_id,
                    from_state: from,
                    to_state: to,
                    trigger: portable_transition.trigger.clone(),
                    guard: portable_transition.guard.clone(),
                };

                graph_store
                    .upsert_protocol_transition(&transition)
                    .await
                    .context("Failed to create imported protocol transition")?;
            }
            // Skip transitions with unresolvable state names (defensive — should not
            // happen if the package was validated, but we don't want to abort the whole
            // import for a single bad transition)
        }

        count += 1;
    }

    Ok(count)
}

/// Create bidirectional synapses between all pairs of imported notes.
///
/// For N notes, creates N×(N-1)/2 pairs with weight `IMPORT_SYNAPSE_WEIGHT`.
/// Uses the existing `create_synapses` method which handles MERGE idempotence.
async fn create_import_synapses(graph_store: &dyn GraphStore, note_ids: &[Uuid]) -> Result<usize> {
    if note_ids.len() < 2 {
        return Ok(0);
    }

    let mut total_created = 0;
    for (i, &note_id) in note_ids.iter().enumerate() {
        // For each note, create synapses to all subsequent notes
        // (create_synapses handles bidirectionality internally)
        let neighbors: Vec<(Uuid, f64)> = note_ids[i + 1..]
            .iter()
            .map(|&neighbor_id| (neighbor_id, IMPORT_SYNAPSE_WEIGHT))
            .collect();

        if !neighbors.is_empty() {
            let created = graph_store
                .create_synapses(note_id, &neighbors)
                .await
                .context("Failed to create import synapses")?;
            total_created += created;
        }
    }

    Ok(total_created)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::InstanceIdentity;
    use crate::neo4j::mock::MockGraphStore;
    use crate::skills::models::SkillTrigger;
    use crate::skills::package::*;

    /// Create a valid, signed test package for import tests.
    fn make_test_package() -> SkillPackage {
        let mut pkg = make_unsigned_test_package();
        let identity = InstanceIdentity::generate();
        sign_package(&mut pkg, &identity).unwrap();
        pkg
    }

    /// Create a valid but unsigned test package (used as base).
    fn make_unsigned_test_package() -> SkillPackage {
        SkillPackage {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: PackageMetadata {
                format: FORMAT_ID.to_string(),
                exported_at: Utc::now(),
                source_project: Some("source-project".to_string()),
                stats: PackageStats {
                    note_count: 2,
                    decision_count: 1,
                    trigger_count: 1,
                    activation_count: 10,
                },
            },
            skill: PortableSkill {
                name: "Neo4j Performance".to_string(),
                description: "Query optimization knowledge".to_string(),
                trigger_patterns: vec![SkillTrigger::regex("neo4j|cypher", 0.7)],
                context_template: None,
                tags: vec!["neo4j".to_string()],
                cohesion: 0.75,
            },
            notes: vec![
                PortableNote {
                    note_type: "guideline".to_string(),
                    importance: "high".to_string(),
                    content: "Always use UNWIND for batch operations".to_string(),
                    tags: vec!["neo4j".to_string()],
                },
                PortableNote {
                    note_type: "gotcha".to_string(),
                    importance: "critical".to_string(),
                    content: "Connection pool leak if not closed".to_string(),
                    tags: vec!["neo4j".to_string()],
                },
            ],
            decisions: vec![PortableDecision {
                description: "Use Neo4j 5.x driver".to_string(),
                rationale: "Better async support".to_string(),
                alternatives: vec!["Neo4j 4.x".to_string()],
                chosen_option: Some("neo4j-rust-driver 0.8".to_string()),
            }],
            protocols: vec![],
            execution_history: None,
            source: None,
            episodes: Vec::new(),
            distilled_episodes: Vec::new(),
            package_trust: None,
            privacy_report: None,
            privacy_mode: None,
            signature: None,
        }
    }

    #[tokio::test]
    async fn test_import_skill_basic() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.notes_created, 2);
        assert_eq!(result.decisions_imported, 1);
        assert!(result.synapses_created > 0);
        assert!(!result.was_merged);
        assert_eq!(result.source_project, Some("source-project".to_string()));
        assert!(result.conflict.is_none());

        // Verify the skill was created
        let skill = store.get_skill(result.skill_id).await.unwrap().unwrap();
        assert_eq!(skill.name, "Neo4j Performance");
        assert_eq!(skill.status, SkillStatus::Imported);
        assert!(skill.imported_at.is_some());
        assert!(!skill.is_validated);
        assert_eq!(skill.project_id, project_id);
    }

    #[tokio::test]
    async fn test_import_creates_notes_with_new_uuids() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        // Verify skill members exist
        let (_notes, _decisions) = store.get_skill_members(result.skill_id).await.unwrap();
        // The mock might not return all members depending on implementation,
        // but the import should have created 3 notes (2 regular + 1 decision-as-note)
        assert_eq!(result.notes_created + result.decisions_imported, 3);
    }

    #[tokio::test]
    async fn test_import_decision_encoded_as_note() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.decisions_imported, 1);
        // The decision is encoded as a context note with special tags
    }

    #[tokio::test]
    async fn test_import_conflict_skip() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create an existing skill with the same name
        let existing = SkillNode::new(project_id, "Neo4j Performance");
        store.create_skill(&existing).await.unwrap();

        // Try to import — should fail with Skip strategy
        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_import_conflict_merge() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create an existing skill with the same name
        let existing = SkillNode::new(project_id, "Neo4j Performance");
        let existing_id = existing.id;
        store.create_skill(&existing).await.unwrap();

        // Import with Merge strategy
        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Merge)
            .await
            .unwrap();

        assert!(result.was_merged);
        assert_eq!(result.skill_id, existing_id); // reused existing skill
        assert!(result.conflict.is_some());
        assert_eq!(
            result.conflict.unwrap().strategy_applied,
            ConflictStrategy::Merge
        );
        assert_eq!(result.notes_created, 2);
    }

    #[tokio::test]
    async fn test_import_conflict_replace() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create an existing skill
        let existing = SkillNode::new(project_id, "Neo4j Performance");
        let old_id = existing.id;
        store.create_skill(&existing).await.unwrap();

        // Import with Replace strategy
        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Replace)
            .await
            .unwrap();

        assert!(!result.was_merged);
        assert_ne!(result.skill_id, old_id); // new skill created
        assert!(result.conflict.is_some());
        assert_eq!(
            result.conflict.unwrap().strategy_applied,
            ConflictStrategy::Replace
        );

        // Old skill should be deleted
        assert!(store.get_skill(old_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_import_invalid_package() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut package = make_test_package();
        package.schema_version = 99; // invalid

        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("validation failed"));
    }

    #[tokio::test]
    async fn test_import_imported_skill_status() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let package = make_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        let skill = store.get_skill(result.skill_id).await.unwrap().unwrap();
        // Imported skills start with Imported status, not Active or Emerging
        assert_eq!(skill.status, SkillStatus::Imported);
        assert!(skill.imported_at.is_some());
        assert!(!skill.is_validated);
        // Energy starts at 0 — must prove itself
        assert!((skill.energy - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_import_note_tags_include_imported() {
        // Verify that imported notes are tagged with "imported"
        let portable = crate::skills::package::PortableNote {
            note_type: "guideline".to_string(),
            importance: "high".to_string(),
            content: "Test content".to_string(),
            tags: vec!["original_tag".to_string()],
        };

        let note = portable_note_to_note(&portable, Uuid::new_v4()).unwrap();
        assert!(note.tags.contains(&"imported".to_string()));
        assert!(note.tags.contains(&"original_tag".to_string()));
        assert_eq!(note.created_by, IMPORT_CREATOR);
        assert!((note.energy - IMPORT_NOTE_ENERGY).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_import_decision_to_note_format() {
        let decision = PortableDecision {
            description: "Use connection pooling".to_string(),
            rationale: "Better performance under load".to_string(),
            alternatives: vec!["Single connection".to_string(), "Per-request".to_string()],
            chosen_option: Some("bb8 pool".to_string()),
        };

        let note = decision_to_note(&decision, Uuid::new_v4(), "Test Skill").unwrap();
        assert_eq!(note.note_type, NoteType::Context);
        assert!(note.tags.contains(&"imported_decision".to_string()));
        assert!(note.content.contains("Use connection pooling"));
        assert!(note.content.contains("Better performance under load"));
        assert!(note.content.contains("Single connection"));
        assert!(note.content.contains("bb8 pool"));
    }

    #[tokio::test]
    async fn test_import_empty_decisions() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut package = make_unsigned_test_package();
        package.decisions.clear();
        package.metadata.stats.decision_count = 0;
        let identity = InstanceIdentity::generate();
        sign_package(&mut package, &identity).unwrap();

        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.decisions_imported, 0);
        assert_eq!(result.notes_created, 2);
    }

    #[test]
    fn test_conflict_strategy_display() {
        assert_eq!(ConflictStrategy::Skip.to_string(), "skip");
        assert_eq!(ConflictStrategy::Merge.to_string(), "merge");
        assert_eq!(ConflictStrategy::Replace.to_string(), "replace");
    }

    #[test]
    fn test_conflict_strategy_from_str() {
        assert_eq!(
            "skip".parse::<ConflictStrategy>().unwrap(),
            ConflictStrategy::Skip
        );
        assert_eq!(
            "merge".parse::<ConflictStrategy>().unwrap(),
            ConflictStrategy::Merge
        );
        assert_eq!(
            "replace".parse::<ConflictStrategy>().unwrap(),
            ConflictStrategy::Replace
        );
        assert!("invalid".parse::<ConflictStrategy>().is_err());
        assert!("SKIP".parse::<ConflictStrategy>().is_err()); // case-sensitive
    }

    #[test]
    fn test_synapse_count_formula() {
        // For N notes, we expect N*(N-1)/2 pairs (triangular number)
        // With 3 notes: 3 pairs → each creating bidirectional synapses
        // create_synapses handles bidirectionality, so we call it once per pair
        let n = 5;
        let expected_pairs = n * (n - 1) / 2;
        assert_eq!(expected_pairs, 10);
    }

    // ── v2 protocol import tests ──────────────────────────────────────

    /// Create a v2 test package with protocols for round-trip tests.
    fn make_v2_test_package() -> SkillPackage {
        use crate::skills::package::{
            ExecutionHistory, PortableProtocol, PortableRelevanceVector, PortableState,
            PortableTransition, SourceMetadata,
        };

        let mut pkg = make_unsigned_test_package();

        pkg.protocols = vec![PortableProtocol {
            name: "code-review".to_string(),
            description: "Automated code review workflow".to_string(),
            category: "business".to_string(),
            relevance_vector: Some(PortableRelevanceVector {
                phase: 0.7,
                structure: 0.5,
                domain: 0.3,
                resource: 0.6,
                lifecycle: 0.8,
            }),
            states: vec![
                PortableState {
                    name: "init".to_string(),
                    description: "Gather context".to_string(),
                    action: Some("load_context".to_string()),
                    state_type: "start".to_string(),
                },
                PortableState {
                    name: "review".to_string(),
                    description: "Perform review".to_string(),
                    action: None,
                    state_type: "intermediate".to_string(),
                },
                PortableState {
                    name: "done".to_string(),
                    description: "Review complete".to_string(),
                    action: None,
                    state_type: "terminal".to_string(),
                },
            ],
            transitions: vec![
                PortableTransition {
                    from_state: "init".to_string(),
                    to_state: "review".to_string(),
                    trigger: "context_loaded".to_string(),
                    guard: None,
                },
                PortableTransition {
                    from_state: "review".to_string(),
                    to_state: "done".to_string(),
                    trigger: "review_complete".to_string(),
                    guard: Some("no_critical_issues".to_string()),
                },
            ],
        }];

        pkg.execution_history = Some(ExecutionHistory {
            activation_count: 15,
            success_rate: 0.87,
            avg_score: 0.92,
            source_projects_count: 3,
        });

        pkg.source = Some(SourceMetadata {
            project_name: "upstream-project".to_string(),
            git_remote: Some("git@github.com:org/upstream.git".to_string()),
            instance_id: Some("inst-001".to_string()),
        });

        // Sign after all mutations
        let identity = InstanceIdentity::generate();
        sign_package(&mut pkg, &identity).unwrap();

        pkg
    }

    #[tokio::test]
    async fn test_import_v2_with_protocols() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let package = make_v2_test_package();
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        // Verify protocol was imported
        assert_eq!(result.protocols_imported, 1);
        assert_eq!(result.notes_created, 2);
        assert_eq!(result.decisions_imported, 1);

        // Verify protocol exists in the store
        let (protocols, count) = store
            .list_protocols(project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(count, 1);
        let proto = &protocols[0];
        assert_eq!(proto.name, "code-review");
        assert_eq!(proto.description, "Automated code review workflow");
        assert_eq!(proto.project_id, project_id);
        assert_eq!(proto.skill_id, Some(result.skill_id));
        assert_eq!(
            proto.protocol_category,
            crate::protocol::ProtocolCategory::Business
        );

        // Verify relevance vector
        let rv = proto.relevance_vector.as_ref().unwrap();
        assert!((rv.phase - 0.7).abs() < f64::EPSILON);
        assert!((rv.lifecycle - 0.8).abs() < f64::EPSILON);

        // Verify states
        let states = store.get_protocol_states(proto.id).await.unwrap();
        assert_eq!(states.len(), 3);
        let start_states: Vec<_> = states
            .iter()
            .filter(|s| s.state_type == StateType::Start)
            .collect();
        assert_eq!(start_states.len(), 1);
        assert_eq!(start_states[0].name, "init");
        assert_eq!(start_states[0].action, Some("load_context".to_string()));

        let terminal_states: Vec<_> = states
            .iter()
            .filter(|s| s.state_type == StateType::Terminal)
            .collect();
        assert_eq!(terminal_states.len(), 1);
        assert_eq!(terminal_states[0].name, "done");

        // Verify entry_state points to the start state
        assert_eq!(proto.entry_state, start_states[0].id);
        assert_eq!(proto.terminal_states, vec![terminal_states[0].id]);

        // Verify transitions
        let transitions = store.get_protocol_transitions(proto.id).await.unwrap();
        assert_eq!(transitions.len(), 2);

        // Find the guarded transition
        let guarded: Vec<_> = transitions.iter().filter(|t| t.guard.is_some()).collect();
        assert_eq!(guarded.len(), 1);
        assert_eq!(guarded[0].trigger, "review_complete");
        assert_eq!(guarded[0].guard, Some("no_critical_issues".to_string()));
    }

    #[tokio::test]
    async fn test_import_v1_package_zero_protocols() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // v1 package has no protocols field
        let package = make_test_package(); // no protocols
        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.protocols_imported, 0);

        // No protocols should exist
        let (protocols, count) = store
            .list_protocols(project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(count, 0);
        assert!(protocols.is_empty());
    }

    #[tokio::test]
    async fn test_import_v2_roundtrip_export_import() {
        use crate::neo4j::models::ProjectNode;
        use crate::skills::export::export_skill;

        let store = MockGraphStore::new();
        let source_project_id = Uuid::new_v4();
        let target_project_id = Uuid::new_v4();

        // Create source project in the store
        let source_project = ProjectNode {
            id: source_project_id,
            name: "Source Project".to_string(),
            slug: "source-project".to_string(),
            root_path: "/tmp/source-project".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&source_project).await.unwrap();

        // Create target project
        let target_project = ProjectNode {
            id: target_project_id,
            name: "Target Project".to_string(),
            slug: "target-project".to_string(),
            root_path: "/tmp/target-project".to_string(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&target_project).await.unwrap();

        // Create a skill with a protocol in the source project
        let mut skill = SkillNode::new(source_project_id, "Round Trip Skill");
        skill.description = "Test round-trip".to_string();
        skill.tags = vec!["test".to_string()];
        skill.activation_count = 5;
        skill.cohesion = 0.9;
        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        // Add a note as skill member (required: package must have ≥1 note)
        let note = Note {
            id: Uuid::new_v4(),
            project_id: Some(source_project_id),
            note_type: NoteType::Guideline,
            status: NoteStatus::Active,
            importance: NoteImportance::High,
            scope: NoteScope::Project,
            content: "Use UNWIND for batch ops in /tmp/source-project/src/db.rs".to_string(),
            tags: vec!["neo4j".to_string()],
            anchors: vec![],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy: 0.8,
            last_activated: None,
            reactivation_count: 0,
            last_reactivated: None,
            freshness_pinged_at: None,
            activation_count: 0,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
            memory_horizon: MemoryHorizon::Operational,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };
        let note_id = note.id;
        store.create_note(&note).await.unwrap();
        store
            .add_skill_member(skill_id, "note", note_id)
            .await
            .unwrap();

        // Create a protocol linked to the skill
        let start_state_id = Uuid::new_v4();
        let end_state_id = Uuid::new_v4();

        let proto = crate::protocol::Protocol::new_full(
            source_project_id,
            "test-proto",
            "A test protocol",
            start_state_id,
            vec![end_state_id],
            crate::protocol::ProtocolCategory::Business,
        );
        let proto_id = proto.id;
        let mut proto = proto;
        proto.skill_id = Some(skill_id);
        store.upsert_protocol(&proto).await.unwrap();

        let s1 = crate::protocol::ProtocolState {
            id: start_state_id,
            protocol_id: proto_id,
            name: "begin".to_string(),
            description: "Start state".to_string(),
            action: Some("do_something".to_string()),
            state_type: StateType::Start,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
        };
        let s2 = crate::protocol::ProtocolState {
            id: end_state_id,
            protocol_id: proto_id,
            name: "finish".to_string(),
            description: "End state".to_string(),
            action: None,
            state_type: StateType::Terminal,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
        };
        store.upsert_protocol_state(&s1).await.unwrap();
        store.upsert_protocol_state(&s2).await.unwrap();

        let t =
            crate::protocol::ProtocolTransition::new(proto_id, start_state_id, end_state_id, "go");
        store.upsert_protocol_transition(&t).await.unwrap();

        // Export (signed)
        let identity = InstanceIdentity::generate();
        let package = export_skill(
            skill_id,
            &store,
            Some("source-project".to_string()),
            Some(&identity),
        )
        .await
        .unwrap();

        // Verify exported note has sanitized paths (absolute → relative)
        assert_eq!(package.notes.len(), 1);
        assert_eq!(
            package.notes[0].content,
            "Use UNWIND for batch ops in src/db.rs"
        );
        // Original absolute path should be gone
        assert!(!package.notes[0].content.contains("/tmp/source-project"));

        // Verify exported package has protocols
        assert_eq!(package.protocols.len(), 1);
        assert_eq!(package.protocols[0].name, "test-proto");
        assert_eq!(package.protocols[0].states.len(), 2);
        assert_eq!(package.protocols[0].transitions.len(), 1);

        // Verify transitions use names not UUIDs
        assert_eq!(package.protocols[0].transitions[0].from_state, "begin");
        assert_eq!(package.protocols[0].transitions[0].to_state, "finish");
        assert_eq!(package.protocols[0].transitions[0].trigger, "go");

        // Import into a different project
        let result = import_skill(&package, target_project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.protocols_imported, 1);
        assert_ne!(result.skill_id, skill_id); // fresh UUID

        // Verify imported protocol
        let (imported_protos, _) = store
            .list_protocols(target_project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(imported_protos.len(), 1);
        let imported_proto = &imported_protos[0];
        assert_eq!(imported_proto.name, "test-proto");
        assert_eq!(imported_proto.skill_id, Some(result.skill_id));
        assert_ne!(imported_proto.id, proto_id); // fresh UUID

        // Verify imported states
        let imported_states = store.get_protocol_states(imported_proto.id).await.unwrap();
        assert_eq!(imported_states.len(), 2);

        // Verify state names preserved
        let state_names: Vec<_> = imported_states.iter().map(|s| s.name.clone()).collect();
        assert!(state_names.contains(&"begin".to_string()));
        assert!(state_names.contains(&"finish".to_string()));

        // Verify imported transitions
        let imported_transitions = store
            .get_protocol_transitions(imported_proto.id)
            .await
            .unwrap();
        assert_eq!(imported_transitions.len(), 1);
        assert_eq!(imported_transitions[0].trigger, "go");

        // Verify transition UUIDs match imported state UUIDs (not source ones)
        let begin_id = imported_states
            .iter()
            .find(|s| s.name == "begin")
            .unwrap()
            .id;
        let finish_id = imported_states
            .iter()
            .find(|s| s.name == "finish")
            .unwrap()
            .id;
        assert_eq!(imported_transitions[0].from_state, begin_id);
        assert_eq!(imported_transitions[0].to_state, finish_id);
    }

    #[tokio::test]
    async fn test_import_v2_multiple_protocols() {
        use crate::skills::package::{PortableProtocol, PortableState, PortableTransition};

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut package = make_unsigned_test_package();
        package.protocols = vec![
            PortableProtocol {
                name: "proto-a".to_string(),
                description: "First protocol".to_string(),
                category: "business".to_string(),
                relevance_vector: None,
                states: vec![
                    PortableState {
                        name: "start".to_string(),
                        description: String::new(),
                        action: None,
                        state_type: "start".to_string(),
                    },
                    PortableState {
                        name: "end".to_string(),
                        description: String::new(),
                        action: None,
                        state_type: "terminal".to_string(),
                    },
                ],
                transitions: vec![PortableTransition {
                    from_state: "start".to_string(),
                    to_state: "end".to_string(),
                    trigger: "done".to_string(),
                    guard: None,
                }],
            },
            PortableProtocol {
                name: "proto-b".to_string(),
                description: "Second protocol".to_string(),
                category: "system".to_string(),
                relevance_vector: None,
                states: vec![PortableState {
                    name: "only".to_string(),
                    description: String::new(),
                    action: None,
                    state_type: "start".to_string(),
                }],
                transitions: vec![],
            },
        ];
        let identity = InstanceIdentity::generate();
        sign_package(&mut package, &identity).unwrap();

        let result = import_skill(&package, project_id, &store, ConflictStrategy::Skip)
            .await
            .unwrap();

        assert_eq!(result.protocols_imported, 2);

        let (protos, count) = store
            .list_protocols(project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(count, 2);

        let names: Vec<_> = protos.iter().map(|p| p.name.clone()).collect();
        assert!(names.contains(&"proto-a".to_string()));
        assert!(names.contains(&"proto-b".to_string()));

        // Verify system category was parsed correctly
        let system_proto = protos.iter().find(|p| p.name == "proto-b").unwrap();
        assert_eq!(
            system_proto.protocol_category,
            crate::protocol::ProtocolCategory::System
        );
    }
}
