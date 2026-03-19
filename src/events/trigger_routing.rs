//! # Trigger Routing — Contextual protocol selection
//!
//! When multiple EventTriggers match an event, the router evaluates
//! context (phase, structure, domain) and selects the best protocol.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::trigger::EventTrigger;
use super::types::{CrudAction, CrudEvent, EntityType};

// ────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────

/// Context built from an event and project state.
///
/// Each dimension is a `[0.0, 1.0]` scalar that captures one aspect of
/// the current situation.  The router uses these to compute an affinity
/// score against each candidate trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    /// Project lifecycle phase.
    /// 0.0 = warmup, 0.25 = planning, 0.5 = execution, 0.75 = review, 1.0 = closure.
    pub phase: f64,
    /// Structural complexity.
    /// 0.0 = simple / flat, 1.0 = complex / deeply nested.
    pub structure: f64,
    /// Domain specificity.
    /// 0.5 = domain-agnostic (default), higher = more domain-specific.
    pub domain: f64,
    /// Resource availability.
    /// 0.0 = scarce, 1.0 = abundant.
    pub resource: f64,
    /// Entity lifecycle position.
    /// 0.0 = just started, 1.0 = nearing end.
    pub lifecycle: f64,
}

impl Default for RoutingContext {
    fn default() -> Self {
        Self {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        }
    }
}

/// Result of routing evaluation for a single trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The trigger that was evaluated.
    pub trigger_id: Uuid,
    /// The protocol the trigger would activate.
    pub protocol_id: Uuid,
    /// Affinity score in `[0.0, 1.0]`.
    pub score: f64,
    /// The context used for scoring.
    pub context: RoutingContext,
    /// Human-readable explanation of why this score was assigned.
    pub explanation: String,
}

// ────────────────────────────────────────────────────────────────────
// Router
// ────────────────────────────────────────────────────────────────────

/// The Trigger Router — stateless helper that ranks triggers by
/// contextual affinity.
pub struct TriggerRouter;

impl TriggerRouter {
    /// Infer a [`RoutingContext`] from a [`CrudEvent`].
    ///
    /// The heuristics are intentionally simple — they look at
    /// `entity_type`, `action`, and selected payload keys.
    pub fn build_context_from_event(event: &CrudEvent) -> RoutingContext {
        let mut ctx = RoutingContext::default();

        match (&event.entity_type, &event.action) {
            // Project::Synced → warmup phase
            (EntityType::Project, CrudAction::Synced) => {
                ctx.phase = 0.0;
                ctx.structure = 0.5;
                ctx.lifecycle = 0.0;
            }

            // Note::Created with note_type=rfc → planning phase, domain-specific
            (EntityType::Note, CrudAction::Created) => {
                ctx.phase = 0.25;
                ctx.lifecycle = 0.1;
                // If the payload hints at an RFC, bump domain specificity
                if event.payload.get("note_type").and_then(|v| v.as_str()) == Some("rfc") {
                    ctx.domain = 0.7;
                }
            }

            // ProtocolRun::StatusChanged → check for completion (closure)
            (EntityType::ProtocolRun, CrudAction::StatusChanged) => {
                let is_completed = event
                    .payload
                    .get("new_status")
                    .and_then(|v| v.as_str())
                    == Some("completed");
                if is_completed {
                    ctx.phase = 1.0;
                    ctx.lifecycle = 1.0;
                } else {
                    ctx.phase = 0.5;
                    ctx.lifecycle = 0.5;
                }
            }

            // Plan::StatusChanged → review phase when completed
            (EntityType::Plan, CrudAction::StatusChanged) => {
                let is_completed = event
                    .payload
                    .get("new_status")
                    .and_then(|v| v.as_str())
                    == Some("completed");
                if is_completed {
                    ctx.phase = 0.75;
                    ctx.lifecycle = 0.9;
                } else {
                    ctx.phase = 0.5;
                    ctx.lifecycle = 0.5;
                }
            }

            // Commit::Created → execution phase
            (EntityType::Commit, CrudAction::Created) => {
                ctx.phase = 0.5;
                ctx.structure = 0.5;
                ctx.lifecycle = 0.5;
            }

            // Skill::StatusChanged → execution phase, higher structure
            (EntityType::Skill, CrudAction::StatusChanged) => {
                ctx.phase = 0.5;
                ctx.structure = 0.7;
                ctx.lifecycle = 0.5;
            }

            // Fallback — keep defaults
            _ => {}
        }

        ctx
    }

    /// Rank a set of candidate triggers against a routing context.
    ///
    /// The affinity score is computed as a weighted dot-product between the
    /// trigger's "ideal context" (inferred from its patterns) and the actual
    /// context.  Returns decisions sorted **descending** by score.
    pub fn rank_triggers(
        triggers: &[&EventTrigger],
        context: &RoutingContext,
    ) -> Vec<RoutingDecision> {
        let mut decisions: Vec<RoutingDecision> = triggers
            .iter()
            .map(|trigger| {
                let (score, explanation) = Self::compute_affinity(trigger, context);
                RoutingDecision {
                    trigger_id: trigger.id,
                    protocol_id: trigger.protocol_id,
                    score,
                    context: context.clone(),
                    explanation,
                }
            })
            .collect();

        // Sort descending by score (stable sort keeps insertion order for ties)
        decisions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        decisions
    }

    /// Select the best decision that exceeds `min_score`.
    ///
    /// The input slice is expected to be sorted descending (as returned by
    /// [`rank_triggers`]).  Returns `None` if no decision meets the threshold.
    pub fn select_best(decisions: &[RoutingDecision], min_score: f64) -> Option<&RoutingDecision> {
        decisions.first().filter(|d| d.score >= min_score)
    }

    // ── Private helpers ────────────────────────────────────────────

    /// Compute an affinity score for a trigger given the routing context.
    ///
    /// The idea: each trigger pattern implies a preferred context range.
    /// We compute a similarity between that preferred range and the actual
    /// context, yielding a score in `[0.0, 1.0]`.
    fn compute_affinity(trigger: &EventTrigger, context: &RoutingContext) -> (f64, String) {
        let ideal = Self::infer_ideal_context(trigger);

        // Weighted dimensions — phase matters most, then domain, then structure
        const W_PHASE: f64 = 0.35;
        const W_STRUCTURE: f64 = 0.15;
        const W_DOMAIN: f64 = 0.20;
        const W_RESOURCE: f64 = 0.10;
        const W_LIFECYCLE: f64 = 0.20;

        let phase_sim = 1.0 - (context.phase - ideal.phase).abs();
        let struct_sim = 1.0 - (context.structure - ideal.structure).abs();
        let domain_sim = 1.0 - (context.domain - ideal.domain).abs();
        let resource_sim = 1.0 - (context.resource - ideal.resource).abs();
        let lifecycle_sim = 1.0 - (context.lifecycle - ideal.lifecycle).abs();

        let score = W_PHASE * phase_sim
            + W_STRUCTURE * struct_sim
            + W_DOMAIN * domain_sim
            + W_RESOURCE * resource_sim
            + W_LIFECYCLE * lifecycle_sim;

        let explanation = format!(
            "phase={:.2}(w{:.0}%) struct={:.2}(w{:.0}%) domain={:.2}(w{:.0}%) \
             resource={:.2}(w{:.0}%) lifecycle={:.2}(w{:.0}%) → {:.3}",
            phase_sim,
            W_PHASE * 100.0,
            struct_sim,
            W_STRUCTURE * 100.0,
            domain_sim,
            W_DOMAIN * 100.0,
            resource_sim,
            W_RESOURCE * 100.0,
            lifecycle_sim,
            W_LIFECYCLE * 100.0,
            score,
        );

        (score, explanation)
    }

    /// Infer an "ideal" routing context from a trigger's patterns.
    ///
    /// This maps known entity-type + action combinations to the context
    /// they most naturally belong to.
    fn infer_ideal_context(trigger: &EventTrigger) -> RoutingContext {
        let entity = trigger.entity_type_pattern.as_deref().unwrap_or("");
        let action = trigger.action_pattern.as_deref().unwrap_or("");

        match (entity, action) {
            ("project", "synced") => RoutingContext {
                phase: 0.0,
                structure: 0.5,
                domain: 0.5,
                resource: 0.5,
                lifecycle: 0.0,
            },
            ("note", "created") => RoutingContext {
                phase: 0.25,
                structure: 0.5,
                domain: 0.7,
                resource: 0.5,
                lifecycle: 0.1,
            },
            ("protocol_run", "status_changed") => RoutingContext {
                phase: 1.0,
                structure: 0.5,
                domain: 0.5,
                resource: 0.5,
                lifecycle: 1.0,
            },
            ("plan", "status_changed") => RoutingContext {
                phase: 0.75,
                structure: 0.5,
                domain: 0.5,
                resource: 0.5,
                lifecycle: 0.9,
            },
            ("commit", "created") => RoutingContext {
                phase: 0.5,
                structure: 0.5,
                domain: 0.5,
                resource: 0.5,
                lifecycle: 0.5,
            },
            ("skill", "status_changed") => RoutingContext {
                phase: 0.5,
                structure: 0.7,
                domain: 0.5,
                resource: 0.5,
                lifecycle: 0.5,
            },
            // Unknown pattern → neutral context (will score ~1.0 against default context)
            _ => RoutingContext::default(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    fn make_trigger(
        entity_type_pattern: Option<&str>,
        action_pattern: Option<&str>,
        payload_conditions: Option<serde_json::Value>,
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
            project_scope: None,
            created_at: now,
            updated_at: now,
        }
    }

    fn make_event(
        entity_type: EntityType,
        action: CrudAction,
        payload: serde_json::Value,
    ) -> CrudEvent {
        CrudEvent::new(entity_type, action, "test-entity-id").with_payload(payload)
    }

    // ================================================================
    // build_context_from_event
    // ================================================================

    #[test]
    fn test_context_project_synced() {
        let event = make_event(EntityType::Project, CrudAction::Synced, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.0).abs() < f64::EPSILON);
        assert!((ctx.structure - 0.5).abs() < f64::EPSILON);
        assert!((ctx.lifecycle - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_note_created_rfc() {
        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.25).abs() < f64::EPSILON);
        assert!((ctx.domain - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_note_created_non_rfc() {
        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "guideline"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.25).abs() < f64::EPSILON);
        // domain stays at default for non-RFC notes
        assert!((ctx.domain - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_protocol_run_completed() {
        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            json!({"new_status": "completed"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 1.0).abs() < f64::EPSILON);
        assert!((ctx.lifecycle - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_protocol_run_non_completed() {
        let event = make_event(
            EntityType::ProtocolRun,
            CrudAction::StatusChanged,
            json!({"new_status": "running"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_plan_completed() {
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            json!({"new_status": "completed"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.75).abs() < f64::EPSILON);
        assert!((ctx.lifecycle - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_commit_created() {
        let event = make_event(EntityType::Commit, CrudAction::Created, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_skill_status_changed() {
        let event = make_event(
            EntityType::Skill,
            CrudAction::StatusChanged,
            json!({"new_status": "active"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.5).abs() < f64::EPSILON);
        assert!((ctx.structure - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_unknown_event_defaults() {
        let event = make_event(EntityType::Task, CrudAction::Updated, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);
        assert!((ctx.phase - 0.5).abs() < f64::EPSILON);
        assert!((ctx.structure - 0.5).abs() < f64::EPSILON);
        assert!((ctx.domain - 0.5).abs() < f64::EPSILON);
    }

    // ================================================================
    // rank_triggers
    // ================================================================

    #[test]
    fn test_rank_triggers_single() {
        let trigger = make_trigger(Some("project"), Some("synced"), None);
        let event = make_event(EntityType::Project, CrudAction::Synced, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);

        let decisions = TriggerRouter::rank_triggers(&[&trigger], &ctx);
        assert_eq!(decisions.len(), 1);
        // Perfect context match → score should be 1.0
        assert!(
            (decisions[0].score - 1.0).abs() < 0.01,
            "Expected ~1.0, got {}",
            decisions[0].score
        );
    }

    #[test]
    fn test_rank_triggers_best_match_first() {
        // The "project synced" trigger should rank highest for a Project::Synced event
        let trigger_project = make_trigger(Some("project"), Some("synced"), None);
        let trigger_commit = make_trigger(Some("commit"), Some("created"), None);
        let trigger_plan = make_trigger(Some("plan"), Some("status_changed"), None);

        let event = make_event(EntityType::Project, CrudAction::Synced, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);

        let decisions = TriggerRouter::rank_triggers(
            &[&trigger_commit, &trigger_plan, &trigger_project],
            &ctx,
        );

        assert_eq!(decisions.len(), 3);
        assert_eq!(decisions[0].trigger_id, trigger_project.id);
    }

    #[test]
    fn test_rank_triggers_multiple_with_scores() {
        let trigger_a = make_trigger(Some("note"), Some("created"), None);
        let trigger_b = make_trigger(Some("project"), Some("synced"), None);

        // An RFC-note-created event: trigger_a should score much higher
        let event = make_event(
            EntityType::Note,
            CrudAction::Created,
            json!({"note_type": "rfc"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        let decisions = TriggerRouter::rank_triggers(&[&trigger_a, &trigger_b], &ctx);

        assert_eq!(decisions[0].trigger_id, trigger_a.id);
        assert!(decisions[0].score > decisions[1].score);
    }

    // ================================================================
    // select_best
    // ================================================================

    #[test]
    fn test_select_best_above_threshold() {
        let trigger = make_trigger(Some("project"), Some("synced"), None);
        let event = make_event(EntityType::Project, CrudAction::Synced, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);
        let decisions = TriggerRouter::rank_triggers(&[&trigger], &ctx);

        let best = TriggerRouter::select_best(&decisions, 0.6);
        assert!(best.is_some());
        assert!(best.unwrap().score >= 0.6);
    }

    #[test]
    fn test_select_best_below_threshold() {
        // A project-synced trigger evaluated against a plan-completed context
        let trigger = make_trigger(Some("project"), Some("synced"), None);
        let event = make_event(
            EntityType::Plan,
            CrudAction::StatusChanged,
            json!({"new_status": "completed"}),
        );
        let ctx = TriggerRouter::build_context_from_event(&event);
        let decisions = TriggerRouter::rank_triggers(&[&trigger], &ctx);

        // With a very high threshold, should not select
        let best = TriggerRouter::select_best(&decisions, 0.99);
        assert!(best.is_none(), "Expected None but got score {}", decisions[0].score);
    }

    #[test]
    fn test_select_best_empty_decisions() {
        let best = TriggerRouter::select_best(&[], 0.6);
        assert!(best.is_none());
    }

    #[test]
    fn test_select_best_default_threshold() {
        // Verify the recommended 0.6 threshold works for matching triggers
        let trigger = make_trigger(Some("commit"), Some("created"), None);
        let event = make_event(EntityType::Commit, CrudAction::Created, json!({}));
        let ctx = TriggerRouter::build_context_from_event(&event);
        let decisions = TriggerRouter::rank_triggers(&[&trigger], &ctx);

        let best = TriggerRouter::select_best(&decisions, 0.6);
        assert!(best.is_some());
    }
}
