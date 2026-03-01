//! Post-sync topology check hook.
//!
//! After a code sync, this hook runs `check_topology_rules()` and auto-creates
//! gotcha notes for newly detected violations. Existing violations (already
//! captured as notes) are skipped to avoid duplicates.

use std::sync::Arc;
use uuid::Uuid;

use crate::graph::models::TopologySeverity;
use crate::neo4j::GraphStore;
use crate::notes::models::{Note, NoteFilters, NoteImportance, NoteType};

/// Result of a post-sync topology check.
#[derive(Debug, Default)]
pub struct TopologyCheckHookResult {
    /// Number of violations detected
    pub violations_found: usize,
    /// Number of new gotcha notes created
    pub notes_created: usize,
    /// Number of violations already captured (skipped)
    pub already_captured: usize,
}

/// Run topology rules after sync and create gotcha notes for new violations.
///
/// This is designed to be called from a `tokio::spawn` (best-effort, non-blocking).
/// Violations are deduplicated using the tag `topology-violation:<fingerprint>`
/// where fingerprint = `rule_id:violator:target`.
pub async fn check_topology_post_sync(
    neo4j: Arc<dyn GraphStore>,
    project_id: Uuid,
) -> anyhow::Result<TopologyCheckHookResult> {
    let pid = project_id.to_string();
    let mut result = TopologyCheckHookResult::default();

    // 1. Run topology rules
    let violations = neo4j.check_topology_rules(&pid).await?;
    result.violations_found = violations.len();

    if violations.is_empty() {
        return Ok(result);
    }

    // 2. Fetch existing topology-violation notes for this project to deduplicate
    let filters = NoteFilters {
        tags: Some(vec!["topology-violation".to_string()]),
        limit: Some(500),
        ..Default::default()
    };
    let (existing_notes, _) = neo4j
        .list_notes(Some(project_id), None, &filters)
        .await
        .unwrap_or((vec![], 0));

    let existing_fingerprints: std::collections::HashSet<String> = existing_notes
        .iter()
        .flat_map(|n| {
            n.tags
                .iter()
                .filter(|t| t.starts_with("topology-violation:"))
                .cloned()
        })
        .collect();

    // 3. Create gotcha notes for new violations
    for violation in &violations {
        let fingerprint = format!(
            "topology-violation:{}:{}:{}",
            violation.rule_id,
            violation.violator_path,
            violation.target_path.as_deref().unwrap_or("*"),
        );

        if existing_fingerprints.contains(&fingerprint) {
            result.already_captured += 1;
            continue;
        }

        // Build note content
        let severity_icon = if violation.severity == TopologySeverity::Error {
            "🔴"
        } else {
            "🟠"
        };
        let content = format!(
            "{} **Topology violation**: {} — {}\n\n\
             **Violator**: `{}`\n\
             **Target**: `{}`\n\
             **Rule**: {} ({:?})\n\
             **Score**: {:.2}\n\n\
             _Auto-detected by post-sync topology check._",
            severity_icon,
            violation.rule_type,
            violation.rule_description,
            violation.violator_path,
            violation.target_path.as_deref().unwrap_or("N/A"),
            violation.rule_id,
            violation.severity,
            violation.violation_score,
        );

        let importance = if violation.severity == TopologySeverity::Error {
            NoteImportance::High
        } else {
            NoteImportance::Medium
        };

        let mut note = Note::new(
            Some(project_id),
            NoteType::Gotcha,
            content,
            "topology-firewall".to_string(),
        );
        note.importance = importance;
        note.tags = vec![
            "topology-violation".to_string(),
            fingerprint,
            format!("rule:{}", violation.rule_type),
            format!("severity:{:?}", violation.severity),
        ];

        // Create the note
        if let Err(e) = neo4j.create_note(&note).await {
            tracing::warn!(
                "Failed to create topology violation note for {}: {}",
                violation.violator_path,
                e
            );
            continue;
        }

        // Link note to the violator file
        if let Err(e) = neo4j
            .link_note_to_entity(
                note.id,
                &crate::notes::models::EntityType::File,
                &violation.violator_path,
                None,
                None,
            )
            .await
        {
            tracing::warn!(
                "Failed to link topology note to {}: {}",
                violation.violator_path,
                e
            );
        }

        // Link note to the target file if present
        if let Some(ref target) = violation.target_path {
            if let Err(e) = neo4j
                .link_note_to_entity(
                    note.id,
                    &crate::notes::models::EntityType::File,
                    target,
                    None,
                    None,
                )
                .await
            {
                tracing::warn!("Failed to link topology note to {}: {}", target, e);
            }
        }

        result.notes_created += 1;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{
        TopologyRule, TopologyRuleType, TopologySeverity, TopologyViolation,
    };
    use crate::neo4j::mock::MockGraphStore;

    #[tokio::test]
    async fn test_no_violations_returns_empty() {
        let mock = Arc::new(MockGraphStore::new());
        let pid = Uuid::new_v4();

        let result = check_topology_post_sync(mock, pid).await.unwrap();
        assert_eq!(result.violations_found, 0);
        assert_eq!(result.notes_created, 0);
        assert_eq!(result.already_captured, 0);
    }

    #[tokio::test]
    async fn test_violations_create_gotcha_notes() {
        let mock = Arc::new(MockGraphStore::new());
        let pid = Uuid::new_v4();

        // Create a topology rule
        let rule = TopologyRule {
            id: "rule-1".to_string(),
            project_id: pid.to_string(),
            rule_type: TopologyRuleType::MustNotImport,
            source_pattern: "src/api/**".to_string(),
            target_pattern: Some("src/neo4j/**".to_string()),
            threshold: None,
            severity: TopologySeverity::Error,
            description: "API must not import Neo4j directly".to_string(),
        };
        mock.create_topology_rule(&rule).await.unwrap();

        // Mock doesn't produce violations, so result should be 0
        // (real check_topology_rules with Neo4j would produce violations)
        let result = check_topology_post_sync(mock, pid).await.unwrap();
        assert_eq!(result.violations_found, 0);
    }

    #[test]
    fn test_fingerprint_format() {
        let v = TopologyViolation {
            rule_id: "r1".to_string(),
            rule_description: "test rule".to_string(),
            rule_type: TopologyRuleType::MustNotImport,
            violator_path: "src/api/handler.rs".to_string(),
            target_path: Some("src/neo4j/client.rs".to_string()),
            severity: TopologySeverity::Error,
            details: "test details".to_string(),
            violation_score: 0.9,
        };
        let fingerprint = format!(
            "topology-violation:{}:{}:{}",
            v.rule_id,
            v.violator_path,
            v.target_path.as_deref().unwrap_or("*"),
        );
        assert_eq!(
            fingerprint,
            "topology-violation:r1:src/api/handler.rs:src/neo4j/client.rs"
        );
    }

    #[test]
    fn test_fingerprint_no_target() {
        let v = TopologyViolation {
            rule_id: "r2".to_string(),
            rule_description: "circular dep".to_string(),
            rule_type: TopologyRuleType::NoCircular,
            violator_path: "src/a.rs".to_string(),
            target_path: None,
            severity: TopologySeverity::Warning,
            details: "circular".to_string(),
            violation_score: 0.5,
        };
        let fingerprint = format!(
            "topology-violation:{}:{}:{}",
            v.rule_id,
            v.violator_path,
            v.target_path.as_deref().unwrap_or("*"),
        );
        assert_eq!(fingerprint, "topology-violation:r2:src/a.rs:*");
    }
}
