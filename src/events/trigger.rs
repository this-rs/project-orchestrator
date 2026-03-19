//! Persistent event triggers stored in Neo4j.
//!
//! An [`EventTrigger`] describes a pattern-based condition that, when matched
//! by an incoming [`CrudEvent`], can automatically activate a Protocol.
//! This is the persistent counterpart to the in-memory [`ReactionRule`](super::reactor::ReactionRule)
//! used by the EventReactor.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::CrudEvent;

/// Persistent event trigger stored in Neo4j.
/// Links an event pattern to a Protocol for automatic activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTrigger {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name for the trigger
    pub name: String,
    /// The Protocol to activate when the trigger fires
    pub protocol_id: Uuid,
    /// Entity type pattern — `None` = wildcard (matches any), `Some("project")` = exact match.
    /// Compared against `CrudEvent.entity_type` serialized to snake_case.
    pub entity_type_pattern: Option<String>,
    /// Action pattern — `None` = wildcard (matches any), `Some("synced")` = exact match.
    /// Compared against `CrudEvent.action` serialized to snake_case.
    pub action_pattern: Option<String>,
    /// JSONPath-like payload conditions.
    /// Each key-value pair is checked against the event payload:
    /// - String values: exact match
    /// - Bool values: exact match
    /// - Number values: exact match
    /// - Array values: "any of" — event value must match one element in the array
    pub payload_conditions: Option<serde_json::Value>,
    /// Minimum seconds between consecutive firings (debounce)
    pub cooldown_secs: u32,
    /// Whether the trigger is active
    pub enabled: bool,
    /// Optional: limit this trigger to events from a specific project
    pub project_scope: Option<Uuid>,
    /// When the trigger was created
    pub created_at: DateTime<Utc>,
    /// When the trigger was last modified
    pub updated_at: DateTime<Utc>,
}

impl EventTrigger {
    /// Check whether a [`CrudEvent`] matches this trigger's patterns.
    ///
    /// Matching logic:
    /// 1. If `entity_type_pattern` is set, the event's entity_type (snake_case) must match exactly.
    /// 2. If `action_pattern` is set, the event's action (snake_case) must match exactly.
    /// 3. If `payload_conditions` is set, each key-value pair must be present in the event payload.
    /// 4. If `project_scope` is set, the event's `project_id` must match.
    pub fn matches(&self, event: &CrudEvent) -> bool {
        // 1. Check entity_type_pattern
        if let Some(ref pattern) = self.entity_type_pattern {
            let entity_type_str = serde_json::to_string(&event.entity_type)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string();
            if entity_type_str != *pattern {
                return false;
            }
        }

        // 2. Check action_pattern
        if let Some(ref pattern) = self.action_pattern {
            let action_str = serde_json::to_string(&event.action)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string();
            if action_str != *pattern {
                return false;
            }
        }

        // 3. Check project_scope
        if let Some(scope) = self.project_scope {
            match &event.project_id {
                Some(pid) => {
                    if pid != &scope.to_string() {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // 4. Check payload_conditions
        if let Some(ref conditions) = self.payload_conditions {
            if let Some(cond_obj) = conditions.as_object() {
                let payload = &event.payload;
                for (key, expected) in cond_obj {
                    let actual = &payload[key];
                    if actual.is_null() {
                        return false;
                    }
                    if !value_matches(expected, actual) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Check if an expected condition value matches an actual payload value.
///
/// - Array expected: "any of" semantics — actual must equal one element.
/// - Scalar expected: exact equality (string, bool, number).
fn value_matches(expected: &serde_json::Value, actual: &serde_json::Value) -> bool {
    match expected {
        serde_json::Value::Array(options) => {
            // "any of" — actual must match at least one element
            options.iter().any(|opt| scalar_eq(opt, actual))
        }
        _ => scalar_eq(expected, actual),
    }
}

/// Scalar equality: compares two JSON values (string, bool, number).
fn scalar_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a == b,
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a == b,
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_trigger(
        entity_type_pattern: Option<&str>,
        action_pattern: Option<&str>,
        payload_conditions: Option<serde_json::Value>,
        project_scope: Option<Uuid>,
    ) -> EventTrigger {
        let now = Utc::now();
        EventTrigger {
            id: Uuid::new_v4(),
            name: "test-trigger".to_string(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: entity_type_pattern.map(|s| s.to_string()),
            action_pattern: action_pattern.map(|s| s.to_string()),
            payload_conditions,
            cooldown_secs: 0,
            enabled: true,
            project_scope,
            created_at: now,
            updated_at: now,
        }
    }

    fn make_event(
        entity_type: EntityType,
        action: CrudAction,
        payload: serde_json::Value,
        project_id: Option<&str>,
    ) -> CrudEvent {
        let mut event = CrudEvent::new(entity_type, action, "test-entity-id");
        event.payload = payload;
        if let Some(pid) = project_id {
            event.project_id = Some(pid.to_string());
        }
        event
    }

    // ================================================================
    // Wildcard matching (None patterns)
    // ================================================================

    #[test]
    fn test_wildcard_matches_everything() {
        let trigger = make_trigger(None, None, None, None);
        let event = make_event(
            EntityType::Project,
            CrudAction::Created,
            json!({}),
            None,
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_wildcard_entity_with_action_pattern() {
        let trigger = make_trigger(None, Some("synced"), None, None);
        let event = make_event(EntityType::Project, CrudAction::Synced, json!({}), None);
        assert!(trigger.matches(&event));

        let event2 = make_event(EntityType::Note, CrudAction::Synced, json!({}), None);
        assert!(trigger.matches(&event2));

        let event3 = make_event(EntityType::Project, CrudAction::Created, json!({}), None);
        assert!(!trigger.matches(&event3));
    }

    // ================================================================
    // Entity type matching
    // ================================================================

    #[test]
    fn test_entity_type_exact_match() {
        let trigger = make_trigger(Some("project"), None, None, None);

        let event_ok = make_event(EntityType::Project, CrudAction::Synced, json!({}), None);
        assert!(trigger.matches(&event_ok));

        let event_fail = make_event(EntityType::Note, CrudAction::Synced, json!({}), None);
        assert!(!trigger.matches(&event_fail));
    }

    #[test]
    fn test_entity_type_snake_case() {
        let trigger = make_trigger(Some("feature_graph"), None, None, None);

        let event = make_event(
            EntityType::FeatureGraph,
            CrudAction::Created,
            json!({}),
            None,
        );
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_entity_type_workspace_milestone() {
        let trigger = make_trigger(Some("workspace_milestone"), None, None, None);

        let event = make_event(
            EntityType::WorkspaceMilestone,
            CrudAction::Updated,
            json!({}),
            None,
        );
        assert!(trigger.matches(&event));
    }

    // ================================================================
    // Action matching
    // ================================================================

    #[test]
    fn test_action_exact_match() {
        let trigger = make_trigger(Some("note"), Some("created"), None, None);

        let event_ok = make_event(EntityType::Note, CrudAction::Created, json!({}), None);
        assert!(trigger.matches(&event_ok));

        let event_fail = make_event(EntityType::Note, CrudAction::Deleted, json!({}), None);
        assert!(!trigger.matches(&event_fail));
    }

    #[test]
    fn test_action_status_changed() {
        let trigger = make_trigger(Some("plan"), Some("status_changed"), None, None);

        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            json!({"old_status": "draft", "new_status": "approved"}),
            None,
        );
        assert!(trigger.matches(&event));
    }

    // ================================================================
    // Project scope
    // ================================================================

    #[test]
    fn test_project_scope_match() {
        let project_id = Uuid::new_v4();
        let trigger = make_trigger(None, None, None, Some(project_id));

        let event_ok = make_event(
            EntityType::Task,
            CrudAction::Created,
            json!({}),
            Some(&project_id.to_string()),
        );
        assert!(trigger.matches(&event_ok));

        let event_wrong_project = make_event(
            EntityType::Task,
            CrudAction::Created,
            json!({}),
            Some(&Uuid::new_v4().to_string()),
        );
        assert!(!trigger.matches(&event_wrong_project));

        let event_no_project = make_event(EntityType::Task, CrudAction::Created, json!({}), None);
        assert!(!trigger.matches(&event_no_project));
    }

    // ================================================================
    // Payload conditions
    // ================================================================

    #[test]
    fn test_payload_string_condition() {
        let trigger = make_trigger(
            Some("note"),
            Some("created"),
            Some(json!({"note_type": "rfc"})),
            None,
        );

        let event_ok = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc", "title": "My RFC"}),
            None,
        );
        assert!(trigger.matches(&event_ok));

        let event_fail = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "guideline"}),
            None,
        );
        assert!(!trigger.matches(&event_fail));
    }

    #[test]
    fn test_payload_bool_condition() {
        let trigger = make_trigger(
            Some("project"),
            Some("synced"),
            Some(json!({"is_first_sync": true})),
            None,
        );

        let event_ok = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": true, "files_parsed": 42}),
            None,
        );
        assert!(trigger.matches(&event_ok));

        let event_fail = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": false}),
            None,
        );
        assert!(!trigger.matches(&event_fail));
    }

    #[test]
    fn test_payload_number_condition() {
        let trigger = make_trigger(
            None,
            None,
            Some(json!({"priority": 1})),
            None,
        );

        let event_ok = make_event(
            EntityType::Task,
            CrudAction::Created,
            json!({"priority": 1}),
            None,
        );
        assert!(trigger.matches(&event_ok));

        let event_fail = make_event(
            EntityType::Task,
            CrudAction::Created,
            json!({"priority": 2}),
            None,
        );
        assert!(!trigger.matches(&event_fail));
    }

    #[test]
    fn test_payload_array_any_of_condition() {
        let trigger = make_trigger(
            Some("note"),
            Some("created"),
            Some(json!({"note_type": ["rfc", "guideline", "pattern"]})),
            None,
        );

        let event_rfc = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc"}),
            None,
        );
        assert!(trigger.matches(&event_rfc));

        let event_guideline = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "guideline"}),
            None,
        );
        assert!(trigger.matches(&event_guideline));

        let event_gotcha = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "gotcha"}),
            None,
        );
        assert!(!trigger.matches(&event_gotcha));
    }

    #[test]
    fn test_payload_missing_key_fails() {
        let trigger = make_trigger(
            None,
            None,
            Some(json!({"note_type": "rfc"})),
            None,
        );

        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"title": "No note_type field"}),
            None,
        );
        assert!(!trigger.matches(&event));
    }

    #[test]
    fn test_payload_multiple_conditions_all_must_match() {
        let trigger = make_trigger(
            Some("project"),
            Some("synced"),
            Some(json!({"is_first_sync": true, "files_parsed": 42})),
            None,
        );

        // Both match
        let event_ok = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": true, "files_parsed": 42}),
            None,
        );
        assert!(trigger.matches(&event_ok));

        // Only one matches
        let event_partial = make_event(
            EntityType::Project,
            CrudAction::Synced,
            json!({"is_first_sync": true, "files_parsed": 100}),
            None,
        );
        assert!(!trigger.matches(&event_partial));
    }

    // ================================================================
    // Combined matching
    // ================================================================

    #[test]
    fn test_full_trigger_all_fields() {
        let project_id = Uuid::new_v4();
        let trigger = make_trigger(
            Some("note"),
            Some("created"),
            Some(json!({"note_type": "rfc"})),
            Some(project_id),
        );

        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc", "title": "Design doc"}),
            Some(&project_id.to_string()),
        );
        assert!(trigger.matches(&event));
    }

    // ================================================================
    // Serialization roundtrip
    // ================================================================

    #[test]
    fn test_event_trigger_serde_roundtrip() {
        let trigger = make_trigger(
            Some("project"),
            Some("synced"),
            Some(json!({"is_first_sync": true})),
            Some(Uuid::new_v4()),
        );

        let json = serde_json::to_string(&trigger).unwrap();
        let deserialized: EventTrigger = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, trigger.id);
        assert_eq!(deserialized.name, trigger.name);
        assert_eq!(deserialized.protocol_id, trigger.protocol_id);
        assert_eq!(deserialized.entity_type_pattern, trigger.entity_type_pattern);
        assert_eq!(deserialized.action_pattern, trigger.action_pattern);
        assert_eq!(deserialized.payload_conditions, trigger.payload_conditions);
        assert_eq!(deserialized.cooldown_secs, trigger.cooldown_secs);
        assert_eq!(deserialized.enabled, trigger.enabled);
        assert_eq!(deserialized.project_scope, trigger.project_scope);
    }
}
