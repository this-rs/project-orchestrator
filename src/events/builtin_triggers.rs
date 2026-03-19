//! # Built-in Trigger Definitions
//!
//! Pre-configured [`EventTrigger`] specs for common use-cases.
//! These are **not** auto-created — they serve as templates that the
//! reactor or admin can install via the trigger API.

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use super::trigger::EventTrigger;

/// Helper to build a trigger with common defaults.
fn base_trigger(name: &str, protocol_id: Uuid) -> EventTrigger {
    let now = Utc::now();
    EventTrigger {
        id: Uuid::new_v4(),
        name: name.to_string(),
        protocol_id,
        entity_type_pattern: None,
        action_pattern: None,
        payload_conditions: None,
        cooldown_secs: 0,
        enabled: true,
        project_scope: None,
        created_at: now,
        updated_at: now,
    }
}

/// Trigger for the first project sync.
///
/// Matches: `Project::Synced` with payload `is_first_sync = true`.
/// Typical protocol: bootstrap knowledge fabric, run initial analysis.
pub fn project_first_sync_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("project-first-sync", protocol_id);
    t.entity_type_pattern = Some("project".to_string());
    t.action_pattern = Some("synced".to_string());
    t.payload_conditions = Some(json!({ "is_first_sync": true }));
    t
}

/// Trigger for RFC note creation.
///
/// Matches: `Note::Created` with payload `note_type = "rfc"`.
/// Typical protocol: auto-create a plan from the RFC, notify reviewers.
pub fn rfc_created_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("rfc-created", protocol_id);
    t.entity_type_pattern = Some("note".to_string());
    t.action_pattern = Some("created".to_string());
    t.payload_conditions = Some(json!({ "note_type": "rfc" }));
    t
}

/// Trigger for protocol run completion.
///
/// Matches: `ProtocolRun::StatusChanged` with payload `new_status = "completed"`.
/// Typical protocol: collect episode, update skill energies.
pub fn protocol_run_completed_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("protocol-run-completed", protocol_id);
    t.entity_type_pattern = Some("protocol_run".to_string());
    t.action_pattern = Some("status_changed".to_string());
    t.payload_conditions = Some(json!({ "new_status": "completed" }));
    t
}

/// Trigger for plan completion.
///
/// Matches: `Plan::StatusChanged` with payload `new_status = "completed"`.
/// Typical protocol: collect review episode, update milestones.
pub fn plan_completed_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("plan-completed", protocol_id);
    t.entity_type_pattern = Some("plan".to_string());
    t.action_pattern = Some("status_changed".to_string());
    t.payload_conditions = Some(json!({ "new_status": "completed" }));
    t
}

/// Trigger for commit creation with a cooldown.
///
/// Matches: `Commit::Created` — intended for topology-check protocols
/// that should not fire more than once per 60 seconds.
pub fn commit_topology_check_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("commit-topology-check", protocol_id);
    t.entity_type_pattern = Some("commit".to_string());
    t.action_pattern = Some("created".to_string());
    t.cooldown_secs = 60;
    t
}

/// Trigger for skill emergence or activation.
///
/// Matches: `Skill::StatusChanged` with payload `new_status` in
/// `["emerging", "active"]`.
/// Typical protocol: notify persona, update skill graph.
pub fn skill_emerged_trigger(protocol_id: Uuid) -> EventTrigger {
    let mut t = base_trigger("skill-emerged", protocol_id);
    t.entity_type_pattern = Some("skill".to_string());
    t.action_pattern = Some("status_changed".to_string());
    t.payload_conditions = Some(json!({ "new_status": ["emerging", "active"] }));
    t
}

// ────────────────────────────────────────────────────────────────────
// Install builtin triggers (idempotent)
// ────────────────────────────────────────────────────────────────────

use crate::neo4j::GraphStore;

/// Install all 6 builtin triggers into Neo4j for a given project, idempotently.
///
/// Each trigger is identified by its `name` — if a trigger with the same name
/// already exists (for this project scope), it is skipped.
///
/// `protocol_id` is the protocol UUID that each trigger will fire. Pass a
/// placeholder UUID if no protocol is configured yet; it can be updated later.
///
/// Returns the number of triggers actually created (0..=6).
pub async fn install_builtin_triggers(
    store: &dyn GraphStore,
    project_id: Uuid,
    protocol_id: Uuid,
) -> anyhow::Result<usize> {
    // Build all 6 builtin trigger templates, scoped to this project
    let mut templates = vec![
        project_first_sync_trigger(protocol_id),
        rfc_created_trigger(protocol_id),
        protocol_run_completed_trigger(protocol_id),
        plan_completed_trigger(protocol_id),
        commit_topology_check_trigger(protocol_id),
        skill_emerged_trigger(protocol_id),
    ];
    for t in &mut templates {
        t.project_scope = Some(project_id);
    }

    // Fetch existing triggers for this project to avoid duplicates
    let existing = store
        .list_event_triggers(Some(project_id), false)
        .await?;
    let existing_names: std::collections::HashSet<&str> =
        existing.iter().map(|t| t.name.as_str()).collect();

    let mut created = 0usize;
    for trigger in &templates {
        if existing_names.contains(trigger.name.as_str()) {
            continue;
        }
        store.create_event_trigger(trigger).await?;
        created += 1;
    }

    Ok(created)
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::types::{CrudAction, CrudEvent, EntityType};

    fn make_event(
        entity_type: EntityType,
        action: CrudAction,
        payload: serde_json::Value,
    ) -> CrudEvent {
        CrudEvent::new(entity_type, action, "test-id").with_payload(payload)
    }

    // ================================================================
    // project_first_sync_trigger
    // ================================================================

    #[test]
    fn test_project_first_sync_matches() {
        let protocol_id = Uuid::new_v4();
        let trigger = project_first_sync_trigger(protocol_id);

        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": true, "files_parsed": 42}),
        );
        assert!(trigger.matches(&event));
        assert_eq!(trigger.protocol_id, protocol_id);
    }

    #[test]
    fn test_project_first_sync_rejects_subsequent() {
        let trigger = project_first_sync_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": false}),
        );
        assert!(!trigger.matches(&event));
    }

    #[test]
    fn test_project_first_sync_rejects_wrong_entity() {
        let trigger = project_first_sync_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Note,
            CrudAction::Synced,
            json!({"is_first_sync": true}),
        );
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // rfc_created_trigger
    // ================================================================

    #[test]
    fn test_rfc_created_matches() {
        let trigger = rfc_created_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc", "title": "Design Proposal"}),
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_rfc_created_rejects_guideline() {
        let trigger = rfc_created_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "guideline"}),
        );
        assert!(!trigger.matches(&event));
    }

    #[test]
    fn test_rfc_created_rejects_update() {
        let trigger = rfc_created_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Note,
            CrudAction::Updated,
            json!({"note_type": "rfc"}),
        );
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // protocol_run_completed_trigger
    // ================================================================

    #[test]
    fn test_protocol_run_completed_matches() {
        let trigger = protocol_run_completed_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            json!({"old_status": "running", "new_status": "completed"}),
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_protocol_run_completed_rejects_running() {
        let trigger = protocol_run_completed_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            json!({"old_status": "pending", "new_status": "running"}),
        );
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // plan_completed_trigger
    // ================================================================

    #[test]
    fn test_plan_completed_matches() {
        let trigger = plan_completed_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            json!({"old_status": "in_progress", "new_status": "completed"}),
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_plan_completed_rejects_approved() {
        let trigger = plan_completed_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            json!({"old_status": "draft", "new_status": "approved"}),
        );
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // commit_topology_check_trigger
    // ================================================================

    #[test]
    fn test_commit_topology_check_matches() {
        let trigger = commit_topology_check_trigger(Uuid::new_v4());

        let event = make_event(EntityType::Commit, CrudAction::Created, json!({}));
        assert!(trigger.matches(&event));
        assert_eq!(trigger.cooldown_secs, 60);
    }

    #[test]
    fn test_commit_topology_check_rejects_wrong_entity() {
        let trigger = commit_topology_check_trigger(Uuid::new_v4());

        let event = make_event(EntityType::Task, CrudAction::Created, json!({}));
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // skill_emerged_trigger
    // ================================================================

    #[test]
    fn test_skill_emerged_matches_emerging() {
        let trigger = skill_emerged_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Skill,
            CrudAction::StatusChanged,
            json!({"new_status": "emerging"}),
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_skill_emerged_matches_active() {
        let trigger = skill_emerged_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Skill,
            CrudAction::StatusChanged,
            json!({"new_status": "active"}),
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_skill_emerged_rejects_dormant() {
        let trigger = skill_emerged_trigger(Uuid::new_v4());

        let event = make_event(
            EntityType::Skill,
            CrudAction::StatusChanged,
            json!({"new_status": "dormant"}),
        );
        assert!(!trigger.matches(&event));
    }

    // ================================================================
    // All triggers have correct defaults
    // ================================================================

    #[test]
    fn test_all_builtin_triggers_enabled_by_default() {
        let pid = Uuid::new_v4();
        let triggers = vec![
            project_first_sync_trigger(pid),
            rfc_created_trigger(pid),
            protocol_run_completed_trigger(pid),
            plan_completed_trigger(pid),
            commit_topology_check_trigger(pid),
            skill_emerged_trigger(pid),
        ];

        for t in &triggers {
            assert!(t.enabled, "Trigger '{}' should be enabled", t.name);
            assert!(t.project_scope.is_none(), "Trigger '{}' should have no project scope", t.name);
            assert_eq!(t.protocol_id, pid);
        }
    }
}
