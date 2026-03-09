//! Skill Export
//!
//! Exports a skill and its members as a portable [`SkillPackage`] that can
//! be imported into another project. Internal IDs are stripped — they are
//! regenerated on import.

use anyhow::{Context, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::protocol::models::RunStatus;
use crate::skills::package::{
    ExecutionHistory, PackageMetadata, PackageStats, PortableDecision, PortableNote,
    PortableProtocol, PortableRelevanceVector, PortableSkill, PortableState, PortableTransition,
    SkillPackage, SourceMetadata, CURRENT_SCHEMA_VERSION, FORMAT_ID,
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

    // 2b. Load project root_path for path sanitization
    let project_root = graph_store
        .get_project(skill.project_id)
        .await
        .ok()
        .flatten()
        .map(|p| p.root_path);

    // 3. Convert to portable format
    let portable_notes: Vec<PortableNote> = notes
        .into_iter()
        .map(|n| PortableNote {
            note_type: n.note_type.to_string(),
            importance: n.importance.to_string(),
            content: sanitize_paths(&n.content, project_root.as_deref()),
            tags: n.tags,
        })
        .collect();

    let portable_decisions: Vec<PortableDecision> = decisions
        .into_iter()
        .map(|d| PortableDecision {
            description: sanitize_paths(&d.description, project_root.as_deref()),
            rationale: sanitize_paths(&d.rationale, project_root.as_deref()),
            alternatives: d
                .alternatives
                .iter()
                .map(|a| sanitize_paths(a, project_root.as_deref()))
                .collect(),
            chosen_option: d
                .chosen_option
                .as_deref()
                .map(|o| sanitize_paths(o, project_root.as_deref())),
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
    // v2 — Load protocols linked to this skill
    let mut portable_protocols = Vec::new();
    let mut total_runs: usize = 0;
    let mut completed_runs: usize = 0;
    let _total_score: f64 = 0.0;

    // Find protocols linked to this skill via skill_id
    let project_id = skill.project_id;
    let (all_protocols, _) = graph_store
        .list_protocols(project_id, None, 100, 0)
        .await
        .unwrap_or_default();

    for proto in &all_protocols {
        if proto.skill_id != Some(skill_id) {
            continue;
        }

        // Load states and transitions
        let states = graph_store
            .get_protocol_states(proto.id)
            .await
            .unwrap_or_default();
        let transitions = graph_store
            .get_protocol_transitions(proto.id)
            .await
            .unwrap_or_default();

        // Build name lookup for transitions (UUID → name)
        let state_name_map: std::collections::HashMap<Uuid, String> =
            states.iter().map(|s| (s.id, s.name.clone())).collect();

        let portable_states: Vec<PortableState> = states
            .iter()
            .map(|s| PortableState {
                name: s.name.clone(),
                description: s.description.clone(),
                action: s.action.clone(),
                state_type: format!("{}", s.state_type).to_lowercase(),
            })
            .collect();

        let portable_transitions: Vec<PortableTransition> = transitions
            .iter()
            .filter_map(|t| {
                let from_name = state_name_map.get(&t.from_state)?.clone();
                let to_name = state_name_map.get(&t.to_state)?.clone();
                Some(PortableTransition {
                    from_state: from_name,
                    to_state: to_name,
                    trigger: t.trigger.clone(),
                    guard: t.guard.clone(),
                })
            })
            .collect();

        let relevance_vector = proto
            .relevance_vector
            .as_ref()
            .map(|rv| PortableRelevanceVector {
                phase: rv.phase,
                structure: rv.structure,
                domain: rv.domain,
                resource: rv.resource,
                lifecycle: rv.lifecycle,
            });

        portable_protocols.push(PortableProtocol {
            name: proto.name.clone(),
            description: proto.description.clone(),
            category: format!("{}", proto.protocol_category).to_lowercase(),
            relevance_vector,
            states: portable_states,
            transitions: portable_transitions,
        });

        // Aggregate execution history from runs
        let (runs, run_total) = graph_store
            .list_protocol_runs(proto.id, None, 1000, 0)
            .await
            .unwrap_or_default();
        total_runs += run_total;
        for run in &runs {
            if run.status == RunStatus::Completed {
                completed_runs += 1;
            }
        }
    }

    let execution_history = if total_runs > 0 || skill.activation_count > 0 {
        let success_rate = if total_runs > 0 {
            completed_runs as f64 / total_runs as f64
        } else {
            0.0
        };
        // Use hit_rate as avg_score proxy
        let avg_score = if skill.hit_rate > 0.0 {
            skill.hit_rate
        } else {
            success_rate
        };
        Some(ExecutionHistory {
            activation_count: skill.activation_count,
            success_rate,
            avg_score,
            source_projects_count: 1, // current project is 1
        })
    } else {
        None
    };

    let source = source_project_name.clone().map(|name| SourceMetadata {
        project_name: name,
        git_remote: None,
        instance_id: None,
    });

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
        protocols: portable_protocols,
        execution_history,
        source,
    };

    Ok(package)
}

// ============================================================================
// Helpers
// ============================================================================

/// Replace absolute project paths with relative paths in text content.
///
/// If `project_root` is provided, replaces occurrences of the root path
/// (with or without trailing slash) with an empty string, making paths
/// relative. This ensures packages are portable across machines.
///
/// Examples:
/// - `/Users/john/myproject/src/main.rs` → `src/main.rs`
/// - `file at /Users/john/myproject/lib/foo.rs` → `file at lib/foo.rs`
fn sanitize_paths(content: &str, project_root: Option<&str>) -> String {
    let Some(root) = project_root else {
        return content.to_string();
    };

    if root.is_empty() {
        return content.to_string();
    }

    // Normalize: ensure root ends with /
    let root_with_slash = if root.ends_with('/') {
        root.to_string()
    } else {
        format!("{}/", root)
    };

    // Replace root_with_slash first (more specific), then bare root
    let result = content.replace(&root_with_slash, "");
    result.replace(root, "")
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
            memory_horizon: crate::notes::MemoryHorizon::Operational,
            scar_intensity: 0.0,
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

    // ── sanitize_paths tests ──────────────────────────────────────────

    #[test]
    fn test_sanitize_paths_replaces_absolute() {
        let content = "See file /Users/john/project/src/main.rs for details";
        let result = sanitize_paths(content, Some("/Users/john/project"));
        assert_eq!(result, "See file src/main.rs for details");
    }

    #[test]
    fn test_sanitize_paths_root_with_trailing_slash() {
        let content = "Check /home/ci/app/lib/foo.rs";
        let result = sanitize_paths(content, Some("/home/ci/app/"));
        assert_eq!(result, "Check lib/foo.rs");
    }

    #[test]
    fn test_sanitize_paths_multiple_occurrences() {
        let content = "Compare /tmp/proj/a.rs and /tmp/proj/b.rs";
        let result = sanitize_paths(content, Some("/tmp/proj"));
        assert_eq!(result, "Compare a.rs and b.rs");
    }

    #[test]
    fn test_sanitize_paths_no_root() {
        let content = "Path /absolute/stays.rs intact";
        let result = sanitize_paths(content, None);
        assert_eq!(result, content);
    }

    #[test]
    fn test_sanitize_paths_empty_root() {
        let content = "No change /foo/bar.rs";
        let result = sanitize_paths(content, Some(""));
        assert_eq!(result, content);
    }

    #[test]
    fn test_sanitize_paths_no_match() {
        let content = "Path /other/project/src/lib.rs";
        let result = sanitize_paths(content, Some("/my/project"));
        assert_eq!(result, content);
    }
}
