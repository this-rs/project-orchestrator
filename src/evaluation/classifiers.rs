//! Adapters: each production classifier, seen as `Classifier`.
//!
//! An adapter calls the production code path as it is wired today, with no
//! parameter of its own, so the bench measures what the backend does.

use std::collections::{BTreeMap, BTreeSet};

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
use crate::neurons::activation::{
    select_top, spread_and_rank, Neighbourhood, RankedCandidate, SpreadEdge, SpreadNode, SpreadRule,
};
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
        Box::new(RecallRanking),
        Box::new(RecallSelection),
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
            let mut node = SkillNode::new(Some(Uuid::nil()), spec.name.clone());
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

/// Defaults of the recall fixture's rule, matching `SpreadingActivationConfig`.
const RECALL_MAX_HOPS: usize = 1;
const RECALL_MIN_ACTIVATION: f64 = 0.01;
const RECALL_DECAY_PER_HOP: f64 = 0.5;
const RECALL_MIN_ENERGY: f64 = 0.05;

fn default_energy() -> f64 {
    1.0
}

fn default_reachable() -> bool {
    true
}

fn default_max_hops() -> usize {
    RECALL_MAX_HOPS
}

fn default_min_activation() -> f64 {
    RECALL_MIN_ACTIVATION
}

fn default_decay_per_hop() -> f64 {
    RECALL_DECAY_PER_HOP
}

fn default_min_energy() -> f64 {
    RECALL_MIN_ENERGY
}

#[derive(Deserialize)]
struct RecallNodeSpec {
    #[serde(default = "default_energy")]
    energy: f64,
    #[serde(default)]
    scar_intensity: f64,
    #[serde(default = "default_reachable")]
    reachable: bool,
}

#[derive(Deserialize)]
struct RecallEdgeSpec {
    from: String,
    to: String,
    weight: f64,
}

#[derive(Deserialize)]
struct RecallRuleSpec {
    #[serde(default = "default_max_hops")]
    max_hops: usize,
    #[serde(default = "default_min_activation")]
    min_activation: f64,
    #[serde(default = "default_decay_per_hop")]
    decay_per_hop: f64,
    #[serde(default = "default_min_energy")]
    min_energy: f64,
}

#[derive(Deserialize)]
struct RecallGraphSpec {
    #[serde(default)]
    nodes: BTreeMap<String, RecallNodeSpec>,
    #[serde(default)]
    seeds: BTreeMap<String, f64>,
    #[serde(default)]
    edges: Vec<RecallEdgeSpec>,
    #[serde(default)]
    config: Option<RecallRuleSpec>,
}

/// The best *propagated* node of a graph declared in the case.
///
/// The case gives a small graph (seeds, weighted edges, node properties); the
/// adapter runs the production rule — `spread_and_rank`, the same function
/// `activate_with_profile` drives hop by hop — and names the node that
/// received the most activation from propagation. Seeds are excluded on
/// purpose: with a decay below 1 a propagated score can never beat its own
/// seed, so the question the fixture asks is what propagation surfaces, not
/// what vector search found.
pub struct RecallRanking;

impl Classifier for RecallRanking {
    fn name(&self) -> &'static str {
        "recall_ranking"
    }

    fn predict(&self, _fixture: &Fixture, case: &Case) -> Result<String> {
        let spec: RecallGraphSpec = serde_json::from_value(case.input.clone())?;

        // Deterministic ids: sorted names, so the parent order (and therefore
        // every tie) does not depend on the JSON key order.
        let mut names: BTreeSet<&str> = BTreeSet::new();
        names.extend(spec.nodes.keys().map(String::as_str));
        names.extend(spec.seeds.keys().map(String::as_str));
        for edge in &spec.edges {
            names.insert(edge.from.as_str());
            names.insert(edge.to.as_str());
        }
        let ids: BTreeMap<&str, Uuid> = names
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name, Uuid::from_u128(index as u128 + 1)))
            .collect();
        let reverse: BTreeMap<Uuid, &str> = ids.iter().map(|(name, id)| (*id, *name)).collect();
        let id_of = |name: &str| -> Result<Uuid> {
            ids.get(name)
                .copied()
                .ok_or_else(|| anyhow!("unknown node `{name}`"))
        };

        let mut hood = Neighbourhood::default();
        for (name, node) in &spec.nodes {
            hood.nodes.insert(
                id_of(name)?,
                SpreadNode {
                    energy: node.energy,
                    scar_intensity: node.scar_intensity,
                    reachable: node.reachable,
                    project_id: None,
                },
            );
        }
        // Names referenced only by seeds or edges default to a live node.
        for id in ids.values() {
            hood.nodes.entry(*id).or_insert(SpreadNode {
                energy: 1.0,
                scar_intensity: 0.0,
                reachable: true,
                project_id: None,
            });
        }
        for edge in &spec.edges {
            hood.edges
                .entry(id_of(&edge.from)?)
                .or_default()
                .push(SpreadEdge {
                    neighbour: id_of(&edge.to)?,
                    weight: edge.weight,
                    is_decision: false,
                });
        }
        let mut seeds: Vec<(Uuid, f64)> = Vec::with_capacity(spec.seeds.len());
        for (name, activation) in &spec.seeds {
            seeds.push((id_of(name)?, *activation));
        }
        let config = spec.config.as_ref();
        let rule = SpreadRule {
            max_hops: config.map_or(RECALL_MAX_HOPS, |c| c.max_hops),
            min_activation: config.map_or(RECALL_MIN_ACTIVATION, |c| c.min_activation),
            decay_per_hop: config.map_or(RECALL_DECAY_PER_HOP, |c| c.decay_per_hop),
            min_energy: config.map_or(RECALL_MIN_ENERGY, |c| c.min_energy),
        };

        let hits = spread_and_rank(&seeds, &hood, &rule, None, 1.0);
        let mut best: Option<(f64, &str)> = None;
        for hit in &hits {
            let name = reverse.get(&hit.node).copied().unwrap_or("none");
            let better = match best {
                None => true,
                Some((score, current)) => {
                    hit.score > score || (hit.score == score && name < current)
                }
            };
            if better {
                best = Some((hit.score, name));
            }
        }

        Ok(best.map_or_else(|| "none".to_string(), |(_, name)| name.to_string()))
    }
}

#[derive(Deserialize)]
struct SelectionCandidate {
    id: String,
    score: f64,
}

#[derive(Deserialize)]
struct SelectionSpec {
    max_results: usize,
    candidates: Vec<SelectionCandidate>,
}

/// Which nodes the ranking rule keeps: the top `max_results` by score, ties
/// broken by node id.
///
/// `recall_ranking` scores the *rule* (what propagation surfaces); this one
/// scores the *selection* (what survives the truncation, and in which order).
/// The case declares the candidates and the slot count; the adapter runs the
/// production `select_top`, the same function `activate_with_profile` calls.
pub struct RecallSelection;

impl Classifier for RecallSelection {
    fn name(&self) -> &'static str {
        "recall_selection"
    }

    fn predict(&self, _fixture: &Fixture, case: &Case) -> Result<String> {
        let spec: SelectionSpec = serde_json::from_value(case.input.clone())?;

        // Deterministic ids: sorted names, so a tie is broken by the fixture's
        // own ordering and never by the JSON key order.
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for candidate in &spec.candidates {
            names.insert(candidate.id.as_str());
        }
        let ids: BTreeMap<&str, Uuid> = names
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name, Uuid::from_u128(index as u128 + 1)))
            .collect();
        let reverse: BTreeMap<Uuid, &str> = ids.iter().map(|(name, id)| (*id, *name)).collect();

        let candidates: Vec<RankedCandidate> = spec
            .candidates
            .iter()
            .map(|candidate| {
                let id = ids
                    .get(candidate.id.as_str())
                    .copied()
                    .ok_or_else(|| anyhow!("unknown node `{}`", candidate.id))?;
                Ok(RankedCandidate {
                    id,
                    score: candidate.score,
                })
            })
            .collect::<Result<_>>()?;

        let selected = select_top(candidates, spec.max_results);
        let names: Vec<&str> = selected
            .iter()
            .map(|candidate| reverse.get(&candidate.id).copied().unwrap_or("none"))
            .collect();
        Ok(if names.is_empty() {
            "none".to_string()
        } else {
            names.join("+")
        })
    }
}
