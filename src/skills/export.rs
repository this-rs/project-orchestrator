//! Skill Export
//!
//! Exports a skill and its members as a portable [`SkillPackage`] that can
//! be imported into another project. Internal IDs are stripped — they are
//! regenerated on import.

use anyhow::{Context, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::skills::package::{
    PackageMetadata, PackageStats, PortableDecision, PortableNote, PortableSkill, SkillPackage,
    CURRENT_SCHEMA_VERSION, FORMAT_ID,
};

/// Export a skill and its members as a portable package.
///
/// Steps:
/// 1. Load the skill by ID
/// 2. Load member notes and decisions via `get_skill_members`
/// 3. Convert to portable format (strip internal IDs)
/// 4. Assemble the SkillPackage with metadata
///
/// # Errors
///
/// Returns an error if the skill is not found or if DB operations fail.
pub async fn export_skill(
    skill_id: Uuid,
    graph_store: &dyn GraphStore,
    source_project_name: Option<String>,
) -> Result<SkillPackage> {
    // 1. Load the skill
    let skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", skill_id))?;

    // 2. Load members
    let (notes, decisions) = graph_store
        .get_skill_members(skill_id)
        .await
        .context("Failed to load skill members")?;

    // 3. Convert to portable format
    let portable_notes: Vec<PortableNote> = notes
        .into_iter()
        .map(|n| PortableNote {
            note_type: n.note_type.to_string(),
            importance: n.importance.to_string(),
            content: n.content,
            tags: n.tags,
        })
        .collect();

    let portable_decisions: Vec<PortableDecision> = decisions
        .into_iter()
        .map(|d| PortableDecision {
            description: d.description,
            rationale: d.rationale,
            alternatives: d.alternatives,
            chosen_option: d.chosen_option,
        })
        .collect();

    let portable_skill = PortableSkill {
        name: skill.name.clone(),
        description: skill.description.clone(),
        trigger_patterns: skill.trigger_patterns.clone(),
        context_template: skill.context_template.clone(),
        tags: skill.tags.clone(),
        cohesion: skill.cohesion,
    };

    // 4. Assemble package
    let package = SkillPackage {
        schema_version: CURRENT_SCHEMA_VERSION,
        metadata: PackageMetadata {
            format: FORMAT_ID.to_string(),
            exported_at: Utc::now(),
            source_project: source_project_name,
            stats: PackageStats {
                note_count: portable_notes.len(),
                decision_count: portable_decisions.len(),
                trigger_count: skill.trigger_patterns.len(),
                activation_count: skill.activation_count,
            },
        },
        skill: portable_skill,
        notes: portable_notes,
        decisions: portable_decisions,
    };

    Ok(package)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;

    #[tokio::test]
    async fn test_export_skill_basic() {
        let store = MockGraphStore::new();

        // Create a skill with members
        let project_id = Uuid::new_v4();
        let mut skill = crate::skills::SkillNode::new(project_id, "Test Skill");
        skill.trigger_patterns = vec![
            crate::skills::SkillTrigger::regex("test|mock", 0.7),
            crate::skills::SkillTrigger::file_glob("src/test/**", 0.8),
        ];
        skill.description = "A test skill".to_string();
        skill.tags = vec!["test".to_string()];
        skill.activation_count = 5;
        skill.cohesion = 0.8;

        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        // Add notes
        let note1 = crate::notes::models::Note {
            id: Uuid::new_v4(),
            project_id: Some(project_id),
            note_type: crate::notes::models::NoteType::Guideline,
            status: crate::notes::models::NoteStatus::Active,
            importance: crate::notes::models::NoteImportance::High,
            scope: crate::notes::models::NoteScope::Project,
            content: "Always use UNWIND for batches".to_string(),
            tags: vec!["neo4j".to_string()],
            anchors: vec![],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy: 0.8,
            last_activated: None,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
        };
        store
            .add_skill_member(skill_id, "note", note1.id)
            .await
            .unwrap();
        // Manually insert note into the store for get_skill_members to find
        // (The mock store returns mock data, so we verify the export logic)

        // Export
        let package = export_skill(skill_id, &store, Some("test-project".to_string()))
            .await
            .unwrap();

        // Verify structure
        assert_eq!(package.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(package.metadata.format, FORMAT_ID);
        assert_eq!(
            package.metadata.source_project,
            Some("test-project".to_string())
        );
        assert_eq!(package.skill.name, "Test Skill");
        assert_eq!(package.skill.description, "A test skill");
        assert_eq!(package.skill.trigger_patterns.len(), 2);
        assert_eq!(package.skill.tags, vec!["test".to_string()]);
        assert!((package.skill.cohesion - 0.8).abs() < f64::EPSILON);

        // Package should NOT contain internal IDs
        let json = serde_json::to_string(&package).unwrap();
        assert!(!json.contains(&project_id.to_string()));
        assert!(!json.contains(&skill_id.to_string()));
    }

    #[tokio::test]
    async fn test_export_skill_not_found() {
        let store = MockGraphStore::new();
        let result = export_skill(Uuid::new_v4(), &store, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_export_produces_valid_package() {
        let store = MockGraphStore::new();

        let project_id = Uuid::new_v4();
        let mut skill = crate::skills::SkillNode::new(project_id, "Valid Skill");
        skill.trigger_patterns = vec![crate::skills::SkillTrigger::regex("test", 0.7)];
        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        let package = export_skill(skill_id, &store, None).await.unwrap();

        // The mock returns empty members, so notes will be empty.
        // For a real integration test, we'd populate the mock.
        // Here we just verify the structure is well-formed.
        assert_eq!(package.schema_version, CURRENT_SCHEMA_VERSION);
        assert!(!package.skill.name.is_empty());
    }

    #[tokio::test]
    async fn test_export_json_size() {
        let store = MockGraphStore::new();

        let project_id = Uuid::new_v4();
        let skill = crate::skills::SkillNode::new(project_id, "Small Skill");
        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        let package = export_skill(skill_id, &store, None).await.unwrap();
        let json = serde_json::to_string(&package).unwrap();

        // A typical skill package should be well under 100KB
        assert!(
            json.len() < 100_000,
            "Package JSON too large: {} bytes",
            json.len()
        );
    }
}
