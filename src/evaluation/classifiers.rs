//! Adapters: each production classifier, seen as `Classifier`.
//!
//! An adapter calls the production code path as it is wired today, with no
//! parameter of its own, so the bench measures what the backend does.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use super::{Case, Classifier, Fixture};
use crate::chat::prompt_sections::ToolRefGroupId;
use crate::chat::provider::cognitive::signature::{ContextHints, TaskSignature};
use crate::chat::routing::{HeuristicRouter, RoutingContext, RoutingProvider};
use crate::events::{CrudAction, CrudEvent, EntityType, EventTrigger, TriggerRouter};
use crate::neurons::intent::{IntentDetector, QueryIntentMode};
use crate::skills::activation::evaluate_skill_match;
use crate::skills::models::{SkillNode, SkillTrigger, TriggerType};

/// Scaffolding level used for the tool-group fixture (the level of the
/// metamorphic corpus, where every group is reachable).
const TOOL_GROUPS_LEVEL: u8 = 4;

/// Minimum affinity for a trigger to count as fired. Same threshold as the
/// trigger routing tests (`select_best(_, 0.6)`).
const TRIGGER_MIN_SCORE: f64 = 0.6;

/// Every classifier of the bench, in report order.
pub fn all() -> Vec<Box<dyn Classifier>> {
    vec![
        Box::new(ToolGroups),
        Box::new(Intent),
        Box::new(TaskClass),
        Box::new(Skills),
        Box::new(Triggers),
    ]
}

fn message(input: &Value) -> Result<&str> {
    input
        .get("message")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("input without `message`"))
}

/// Which tool groups the heuristic router adds on top of the two groups that
/// are always included (Core, Knowledge). Labels are the sorted extra groups
/// joined by `+`, or `none`.
pub struct ToolGroups;

impl Classifier for ToolGroups {
    fn name(&self) -> &'static str {
        "tool_groups"
    }

    fn predict(&self, _fixture: &Fixture, case: &Case) -> Result<String> {
        let ctx = RoutingContext {
            scaffolding_level: TOOL_GROUPS_LEVEL,
            user_message: message(&case.input)?.to_string(),
            ..Default::default()
        };
        let mut extra: Vec<String> = HeuristicRouter
            .route(&ctx)
            .tool_groups
            .iter()
            .filter(|group| !matches!(group, ToolRefGroupId::Core | ToolRefGroupId::Knowledge))
            .map(|group| format!("{group:?}"))
            .collect();
        extra.sort();
        Ok(if extra.is_empty() {
            "none".to_string()
        } else {
            extra.join("+")
        })
    }
}

/// The intent detector's mode, lowercased.
pub struct Intent;

impl Classifier for Intent {
    fn name(&self) -> &'static str {
        "intent"
    }

    fn predict(&self, _fixture: &Fixture, case: &Case) -> Result<String> {
        Ok(match IntentDetector::detect(message(&case.input)?) {
            QueryIntentMode::Debug => "debug",
            QueryIntentMode::Explore => "explore",
            QueryIntentMode::Impact => "impact",
            QueryIntentMode::Plan => "plan",
            QueryIntentMode::Default => "default",
        }
        .to_string())
    }
}

/// The class of a chat turn, as `TaskSignature::from_chat_request` sets it.
pub struct TaskClass;

impl Classifier for TaskClass {
    fn name(&self) -> &'static str {
        "task_class"
    }

    fn predict(&self, _fixture: &Fixture, case: &Case) -> Result<String> {
        let signature = TaskSignature::from_chat_request(
            message(&case.input)?,
            false,
            None,
            ContextHints::default(),
        );
        Ok(signature.class.key())
    }
}

#[derive(Deserialize)]
struct SkillSpec {
    name: String,
    triggers: Vec<TriggerSpec>,
}

#[derive(Deserialize)]
struct TriggerSpec {
    #[serde(rename = "type")]
    kind: TriggerType,
    value: String,
    threshold: f64,
    quality: Option<f64>,
}

/// The skill whose triggers score highest on the message (and file, when given),
/// through `evaluate_skill_match`. `none` when no skill scores above zero.
pub struct Skills;

impl Classifier for Skills {
    fn name(&self) -> &'static str {
        "skills"
    }

    fn predict(&self, fixture: &Fixture, case: &Case) -> Result<String> {
        let specs: Vec<SkillSpec> = serde_json::from_value(fixture.catalog["skills"].clone())?;
        let text = message(&case.input)?;
        let file = case.input.get("file").and_then(Value::as_str);
        let mut best: Option<(f64, String)> = None;
        for spec in specs {
            let mut node = SkillNode::new(Uuid::nil(), spec.name.clone());
            node.trigger_patterns = spec
                .triggers
                .into_iter()
                .map(|t| SkillTrigger {
                    pattern_type: t.kind,
                    pattern_value: t.value,
                    confidence_threshold: t.threshold,
                    quality_score: t.quality,
                })
                .collect();
            let score = evaluate_skill_match(&node, Some(text), file);
            if score > 0.0 && best.as_ref().is_none_or(|(top, _)| score > *top) {
                best = Some((score, spec.name));
            }
        }
        Ok(best.map_or_else(|| "none".to_string(), |(_, name)| name))
    }
}

#[derive(Deserialize)]
struct TriggerSpecEvent {
    name: String,
    entity_type: String,
    action: String,
    payload_conditions: Option<Value>,
}

/// The event trigger the router ranks first for an event, when its affinity
/// reaches `TRIGGER_MIN_SCORE`. `none` otherwise.
pub struct Triggers;

impl Classifier for Triggers {
    fn name(&self) -> &'static str {
        "triggers"
    }

    fn predict(&self, fixture: &Fixture, case: &Case) -> Result<String> {
        let specs: Vec<TriggerSpecEvent> =
            serde_json::from_value(fixture.catalog["triggers"].clone())?;
        // Fixed instant: the ranking does not read the clock.
        let epoch: DateTime<Utc> = DateTime::UNIX_EPOCH;
        let mut names: Vec<(Uuid, String)> = Vec::with_capacity(specs.len());
        let mut triggers: Vec<EventTrigger> = Vec::with_capacity(specs.len());
        for (index, spec) in specs.into_iter().enumerate() {
            let id = Uuid::from_u128(index as u128 + 1);
            names.push((id, spec.name.clone()));
            triggers.push(EventTrigger {
                id,
                name: spec.name,
                protocol_id: Uuid::nil(),
                entity_type_pattern: Some(spec.entity_type),
                action_pattern: Some(spec.action),
                payload_conditions: spec.payload_conditions,
                cooldown_secs: 0,
                enabled: true,
                project_scope: None,
                created_at: epoch,
                updated_at: epoch,
            });
        }

        let entity: EntityType = serde_json::from_value(Value::String(
            case.input["entity_type"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        ))?;
        let action: CrudAction = serde_json::from_value(Value::String(
            case.input["action"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        ))?;
        let mut event = CrudEvent::new(entity, action, case.id.clone());
        event.payload = case.input.get("payload").cloned().unwrap_or(Value::Null);

        let ctx = TriggerRouter::build_context_from_event(&event);
        let refs: Vec<&EventTrigger> = triggers.iter().collect();
        let decisions = TriggerRouter::rank_triggers(&refs, &ctx);
        let Some(top) = TriggerRouter::select_best(&decisions, TRIGGER_MIN_SCORE) else {
            return Ok("none".to_string());
        };
        names
            .iter()
            .find(|(id, _)| *id == top.trigger_id)
            .map(|(_, name)| name.clone())
            .ok_or_else(|| anyhow!("ranked trigger {} not in catalog", top.trigger_id))
    }
}
