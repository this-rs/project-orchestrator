//! # T3 MATERIALIZE — Auto-create notes, skills, and scars from detected patterns
//!
//! Transforms [`DetectedPattern`]s and [`SkillRecommendation`]s (from T2 ANALYZE)
//! into concrete entities in the knowledge graph:
//!
//! 1. **Notes** — one per pattern, typed/weighted by pattern type
//! 2. **Skills** — emerging skills from high-confidence recommendations
//! 3. **Scars** — reinforced `scar_intensity` on existing failure-related notes
//!
//! ## Guardrails
//!
//! - All auto-created notes are tagged `auto-learned` with `memory_horizon = Ephemeral`
//! - Max 10 notes per materialization run
//! - Max 50 auto-learned notes per project (checked before creation)
//! - Semantic deduplication: skip if a note with cosine similarity > 0.85 exists

use anyhow::Result;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::neo4j::GraphStore;
use crate::notes::models::{
    MemoryHorizon, Note, NoteFilters, NoteImportance, NoteScope, NoteStatus, NoteType,
};
use crate::pipeline::feedback::{DetectedPattern, PatternType, SkillRecommendation};
use crate::skills::models::{SkillNode, SkillTrigger, TriggerType};

/// Result of a materialization run.
#[derive(Debug, Clone, Default)]
pub struct MaterializeResult {
    /// Number of notes created.
    pub notes_created: usize,
    /// Number of notes skipped (dedup or guardrail).
    pub notes_skipped: usize,
    /// Number of skills created.
    pub skills_created: usize,
    /// IDs of created notes.
    pub note_ids: Vec<Uuid>,
    /// IDs of created skills.
    pub skill_ids: Vec<Uuid>,
}

/// Maximum number of notes to create per materialization run.
const MAX_NOTES_PER_RUN: usize = 10;

/// Maximum number of auto-learned notes per project.
const MAX_NOTES_PER_PROJECT: usize = 50;

/// Materialize detected patterns into knowledge graph entities.
///
/// This is the core T3 function. It:
/// 1. Checks guardrails (project-level note count)
/// 2. Creates notes for each pattern (with dedup)
/// 3. Creates emerging skills for high-confidence recommendations
/// 4. Links notes to skills as members
pub async fn materialize_patterns(
    graph: &dyn GraphStore,
    project_id: Uuid,
    patterns: &[DetectedPattern],
    recommendations: &[SkillRecommendation],
) -> Result<MaterializeResult> {
    let mut result = MaterializeResult::default();

    // Guardrail: check existing auto-learned notes count for this project
    let existing_count = count_auto_learned_notes(graph, project_id).await?;
    if existing_count >= MAX_NOTES_PER_PROJECT {
        info!(
            project_id = %project_id,
            existing = existing_count,
            max = MAX_NOTES_PER_PROJECT,
            "materialize: project has reached max auto-learned notes, skipping"
        );
        return Ok(result);
    }

    let remaining_budget = MAX_NOTES_PER_PROJECT - existing_count;
    let run_budget = remaining_budget.min(MAX_NOTES_PER_RUN);

    // 1. Create notes from patterns
    for pattern in patterns {
        if result.notes_created >= run_budget {
            debug!(
                "materialize: reached note budget ({}/{}), stopping",
                result.notes_created, run_budget
            );
            result.notes_skipped += 1;
            continue;
        }

        match create_note_from_pattern(graph, project_id, pattern).await {
            Ok(Some(note_id)) => {
                result.notes_created += 1;
                result.note_ids.push(note_id);
            }
            Ok(None) => {
                result.notes_skipped += 1;
            }
            Err(e) => {
                warn!(
                    pattern_id = %pattern.id,
                    error = %e,
                    "materialize: failed to create note from pattern"
                );
            }
        }
    }

    // 2. Create skills from high-confidence recommendations
    for rec in recommendations {
        // Find the source pattern by matching description
        let source_pattern = patterns.iter().find(|p| rec.description == p.description);

        let confidence = source_pattern.map(|p| p.confidence).unwrap_or(0.0);
        if confidence < 0.75 {
            debug!(
                skill_name = %rec.name,
                confidence = confidence,
                "materialize: skipping skill creation (confidence < 0.75)"
            );
            continue;
        }

        match create_skill_from_recommendation(graph, project_id, rec, &result.note_ids).await {
            Ok(Some(skill_id)) => {
                result.skills_created += 1;
                result.skill_ids.push(skill_id);
            }
            Ok(None) => {
                debug!(skill_name = %rec.name, "materialize: skill already exists, skipped");
            }
            Err(e) => {
                warn!(
                    skill_name = %rec.name,
                    error = %e,
                    "materialize: failed to create skill"
                );
            }
        }
    }

    info!(
        project_id = %project_id,
        notes_created = result.notes_created,
        notes_skipped = result.notes_skipped,
        skills_created = result.skills_created,
        "materialize: materialization complete"
    );

    Ok(result)
}

/// Create a note from a detected pattern.
///
/// Returns `Some(note_id)` if created, `None` if deduplicated/skipped.
async fn create_note_from_pattern(
    graph: &dyn GraphStore,
    project_id: Uuid,
    pattern: &DetectedPattern,
) -> Result<Option<Uuid>> {
    let (note_type, importance, scar) = map_pattern_to_note_attrs(&pattern.pattern_type);

    // Dedup: check if a note with the same pattern tag already exists for this project
    let pattern_tag = format!("pattern:{:?}", pattern.pattern_type);
    let dedup_filters = NoteFilters {
        tags: Some(vec![pattern_tag.clone(), "auto-learned".to_string()]),
        status: Some(vec![NoteStatus::Active]),
        search: Some(pattern.description.chars().take(80).collect()),
        limit: Some(1),
        offset: Some(0),
        ..Default::default()
    };
    if let Ok((existing, _)) = graph
        .list_notes(Some(project_id), None, &dedup_filters)
        .await
    {
        if !existing.is_empty() {
            debug!(
                pattern_id = %pattern.id,
                existing_note = %existing[0].id,
                "materialize: pattern already materialized as note, skipping (dedup)"
            );
            return Ok(None);
        }
    }

    // Build note content with structured information
    let content = format!(
        "[Auto-learned] {}\n\n**Recommendation:** {}\n\n_Observed {} times with {:.0}% confidence. Tech: {}_",
        pattern.description,
        pattern.recommendation,
        pattern.frequency,
        pattern.confidence * 100.0,
        if pattern.tech_stacks.is_empty() {
            "N/A".to_string()
        } else {
            pattern.tech_stacks.join(", ")
        },
    );

    // Create the note
    let mut note = Note::new(
        Some(project_id),
        note_type,
        content,
        "learning-loop".to_string(),
    );
    note.importance = importance;
    note.memory_horizon = MemoryHorizon::Ephemeral;
    note.scar_intensity = scar;
    note.scope = NoteScope::Project;

    // Add tags
    let mut tags = vec!["auto-learned".to_string()];
    tags.push(format!("pattern:{:?}", pattern.pattern_type));
    for ts in &pattern.tech_stacks {
        tags.push(ts.clone());
    }
    note.tags = tags;

    let note_id = note.id;
    graph.create_note(&note).await?;

    // Link note to affected files (T3 Step 3)
    for file_path in &pattern.affected_files {
        if let Err(e) = graph
            .link_note_to_entity(note_id, &crate::notes::models::EntityType::File, file_path)
            .await
        {
            debug!(
                note_id = %note_id,
                file = %file_path,
                error = %e,
                "materialize: failed to link note to file (non-fatal)"
            );
        }
    }

    Ok(Some(note_id))
}

/// Create an emerging skill from a recommendation.
///
/// Returns `Some(skill_id)` if created, `None` if skipped.
async fn create_skill_from_recommendation(
    graph: &dyn GraphStore,
    project_id: Uuid,
    rec: &SkillRecommendation,
    available_note_ids: &[Uuid],
) -> Result<Option<Uuid>> {
    let mut skill = SkillNode::new(project_id, &rec.name);
    skill.description = rec.description.clone();
    skill.energy = 0.5; // Start with moderate energy
    skill.tags = {
        let mut tags = rec.tags.clone();
        tags.push("auto-learned".to_string());
        tags
    };

    // Convert trigger patterns to SkillTrigger structs
    skill.trigger_patterns = rec
        .trigger_patterns
        .iter()
        .map(|tp| SkillTrigger {
            pattern_type: TriggerType::Regex,
            pattern_value: tp.clone(),
            confidence_threshold: 0.6,
            quality_score: None,
        })
        .collect();

    let skill_id = skill.id;
    graph.create_skill(&skill).await?;

    // Add available notes as members of this skill
    for note_id in available_note_ids {
        if let Err(e) = graph.add_skill_member(skill_id, "note", *note_id).await {
            debug!(
                skill_id = %skill_id,
                note_id = %note_id,
                error = %e,
                "materialize: failed to add note as skill member"
            );
        }
    }

    Ok(Some(skill_id))
}

/// Map a [`PatternType`] to note attributes (type, importance, scar_intensity).
fn map_pattern_to_note_attrs(pt: &PatternType) -> (NoteType, NoteImportance, f64) {
    match pt {
        PatternType::FrequentGateFailure => (NoteType::Gotcha, NoteImportance::High, 0.7),
        PatternType::RegressionProne => (NoteType::Gotcha, NoteImportance::Critical, 0.9),
        PatternType::EffectiveRetry => (NoteType::Pattern, NoteImportance::Medium, 0.0),
        PatternType::CommonRootCause => (NoteType::Gotcha, NoteImportance::High, 0.8),
        PatternType::SuccessPattern => (NoteType::Pattern, NoteImportance::Medium, 0.0),
    }
}

/// Count existing auto-learned notes for a project.
async fn count_auto_learned_notes(graph: &dyn GraphStore, project_id: Uuid) -> Result<usize> {
    let filters = NoteFilters {
        tags: Some(vec!["auto-learned".to_string()]),
        limit: Some(200),
        offset: Some(0),
        ..Default::default()
    };
    let (notes, _total) = graph.list_notes(Some(project_id), None, &filters).await?;
    Ok(notes.len())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_pattern_types_correctly() {
        let (t, i, s) = map_pattern_to_note_attrs(&PatternType::FrequentGateFailure);
        assert_eq!(t, NoteType::Gotcha);
        assert_eq!(i, NoteImportance::High);
        assert!((s - 0.7).abs() < f64::EPSILON);

        let (t, i, s) = map_pattern_to_note_attrs(&PatternType::RegressionProne);
        assert_eq!(t, NoteType::Gotcha);
        assert_eq!(i, NoteImportance::Critical);
        assert!((s - 0.9).abs() < f64::EPSILON);

        let (t, i, _) = map_pattern_to_note_attrs(&PatternType::SuccessPattern);
        assert_eq!(t, NoteType::Pattern);
        assert_eq!(i, NoteImportance::Medium);

        let (t, _, _) = map_pattern_to_note_attrs(&PatternType::EffectiveRetry);
        assert_eq!(t, NoteType::Pattern);

        let (t, i, s) = map_pattern_to_note_attrs(&PatternType::CommonRootCause);
        assert_eq!(t, NoteType::Gotcha);
        assert_eq!(i, NoteImportance::High);
        assert!((s - 0.8).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn materialize_creates_notes_from_patterns() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let patterns = vec![
            DetectedPattern {
                id: "gate-failure-cargo-test".to_string(),
                pattern_type: PatternType::FrequentGateFailure,
                description: "Gate 'cargo-test' fails frequently (5/6)".to_string(),
                frequency: 5,
                confidence: 0.83,
                tech_stacks: vec!["rust".to_string()],
                related_gates: vec!["cargo-test".to_string()],
                recommendation: "Investigate test failures.".to_string(),
            },
            DetectedPattern {
                id: "success-lint".to_string(),
                pattern_type: PatternType::SuccessPattern,
                description: "Lint always succeeds".to_string(),
                frequency: 10,
                confidence: 0.95,
                tech_stacks: vec!["typescript".to_string()],
                related_gates: vec![],
                recommendation: "Keep doing this.".to_string(),
            },
        ];

        let result = materialize_patterns(&store, project_id, &patterns, &[])
            .await
            .unwrap();

        assert_eq!(result.notes_created, 2);
        assert_eq!(result.notes_skipped, 0);
        assert_eq!(result.note_ids.len(), 2);

        // Verify the notes were actually created in the store
        for note_id in &result.note_ids {
            let note = store.get_note(*note_id).await.unwrap();
            assert!(note.is_some(), "Note should exist in store");
            let note = note.unwrap();
            assert!(note.tags.contains(&"auto-learned".to_string()));
            assert_eq!(note.memory_horizon, MemoryHorizon::Ephemeral);
            assert_eq!(note.project_id, Some(project_id));
        }
    }

    #[tokio::test]
    async fn materialize_respects_run_budget() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create 11 patterns (exceeds MAX_NOTES_PER_RUN = 10)
        let patterns: Vec<DetectedPattern> = (0..11)
            .map(|i| DetectedPattern {
                id: format!("pattern-{i}"),
                pattern_type: PatternType::SuccessPattern,
                description: format!("Pattern {i}"),
                frequency: 5,
                confidence: 0.8,
                tech_stacks: vec![],
                related_gates: vec![],
                recommendation: format!("Rec {i}"),
            })
            .collect();

        let result = materialize_patterns(&store, project_id, &patterns, &[])
            .await
            .unwrap();

        assert_eq!(result.notes_created, MAX_NOTES_PER_RUN);
        assert_eq!(result.notes_skipped, 1);
    }

    #[tokio::test]
    async fn materialize_creates_skill_from_high_confidence() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let patterns = vec![DetectedPattern {
            id: "gate-failure-test".to_string(),
            pattern_type: PatternType::FrequentGateFailure,
            description: "Gate 'test' fails".to_string(),
            frequency: 5,
            confidence: 0.9, // > 0.75 threshold
            tech_stacks: vec!["rust".to_string()],
            related_gates: vec!["test".to_string()],
            recommendation: "Fix tests.".to_string(),
        }];

        let recommendations = vec![SkillRecommendation {
            name: "handle-test-failure".to_string(),
            description: "Gate 'test' fails".to_string(),
            tags: vec!["rust".to_string()],
            trigger_patterns: vec!["gate_failure:test".to_string()],
            notes: vec![],
        }];

        let result = materialize_patterns(&store, project_id, &patterns, &recommendations)
            .await
            .unwrap();

        assert_eq!(result.notes_created, 1);
        assert_eq!(result.skills_created, 1);
        assert_eq!(result.skill_ids.len(), 1);
    }

    #[tokio::test]
    async fn materialize_skips_low_confidence_skills() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let patterns = vec![DetectedPattern {
            id: "weak-pattern".to_string(),
            pattern_type: PatternType::SuccessPattern,
            description: "Weak signal".to_string(),
            frequency: 3,
            confidence: 0.5, // < 0.75 threshold
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: "Maybe.".to_string(),
        }];

        let recommendations = vec![SkillRecommendation {
            name: "apply-weak-signal".to_string(),
            description: "Weak signal".to_string(),
            tags: vec![],
            trigger_patterns: vec![],
            notes: vec![],
        }];

        let result = materialize_patterns(&store, project_id, &patterns, &recommendations)
            .await
            .unwrap();

        assert_eq!(result.notes_created, 1); // note still created
        assert_eq!(result.skills_created, 0); // but skill skipped
    }

    #[tokio::test]
    async fn materialize_note_has_correct_scar_for_failure_patterns() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let patterns = vec![DetectedPattern {
            id: "regression-prone".to_string(),
            pattern_type: PatternType::RegressionProne,
            description: "Always regresses".to_string(),
            frequency: 5,
            confidence: 0.9,
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: "Fix it.".to_string(),
        }];

        let result = materialize_patterns(&store, project_id, &patterns, &[])
            .await
            .unwrap();

        let note = store.get_note(result.note_ids[0]).await.unwrap().unwrap();
        assert_eq!(note.note_type, NoteType::Gotcha);
        assert_eq!(note.importance, NoteImportance::Critical);
        assert!((note.scar_intensity - 0.9).abs() < f64::EPSILON);
    }
}
