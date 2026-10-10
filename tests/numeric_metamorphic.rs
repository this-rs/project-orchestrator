//! Metamorphic invariants of the numeric classifiers, offline.
//!
//! These classifiers take no text, so the FR/EN harness does not apply. Two
//! invariants do:
//!  - permutation: the order in which candidates are handed in must not change
//!    the ranking (ties included), because the ranking is a set-level decision;
//!  - monotonicity: moving the input in the direction the score is meant to
//!    follow must never lower it.
//!
//! Covers `TriggerRouter::rank_triggers` (events/trigger_routing.rs) and
//! `rank_protocols` / `compute_affinity` / `ContextVector` (protocol/routing.rs).

use chrono::{DateTime, Utc};
use project_orchestrator::evaluation::embedded_fixtures;
use project_orchestrator::events::{
    CrudAction, CrudEvent, EntityType, EventTrigger, RoutingDecision, TriggerRouter,
};
use project_orchestrator::protocol::models::Protocol;
use project_orchestrator::protocol::routing::{
    compute_affinity, rank_protocols, ContextVector, DimensionWeights, RelevanceVector,
};
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

/// Every ordering of `n` candidates that the test exercises: all rotations of
/// the input and of its reverse. Enough to move every tied candidate.
fn orders(n: usize) -> Vec<Vec<usize>> {
    let base: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    for reversed in [false, true] {
        let seq: Vec<usize> = if reversed {
            base.iter().rev().copied().collect()
        } else {
            base.clone()
        };
        for shift in 0..n.max(1) {
            let mut rotated = seq.clone();
            rotated.rotate_left(shift);
            out.push(rotated);
        }
    }
    out
}

fn trigger(
    index: u128,
    name: &str,
    entity: &str,
    action: &str,
    payload: Option<Value>,
) -> EventTrigger {
    let epoch: DateTime<Utc> = DateTime::UNIX_EPOCH;
    EventTrigger {
        id: Uuid::from_u128(index + 1),
        name: name.to_string(),
        protocol_id: Uuid::nil(),
        entity_type_pattern: Some(entity.to_string()),
        action_pattern: Some(action.to_string()),
        payload_conditions: payload,
        cooldown_secs: 0,
        enabled: true,
        project_scope: None,
        created_at: epoch,
        updated_at: epoch,
    }
}

fn event(entity: &str, action: &str, payload: Value) -> CrudEvent {
    let entity: EntityType =
        serde_json::from_value(Value::String(entity.to_string())).expect("known entity");
    let action: CrudAction =
        serde_json::from_value(Value::String(action.to_string())).expect("known action");
    let mut event = CrudEvent::new(entity, action, "metamorphic");
    event.payload = payload;
    event
}

/// The ranking as `(trigger id, score bits)`: exact, so a float tie is visible.
fn ranking(
    triggers: &[EventTrigger],
    order: &[usize],
    ctx: &project_orchestrator::events::RoutingContext,
) -> Vec<(Uuid, u64)> {
    let refs: Vec<&EventTrigger> = order.iter().map(|&i| &triggers[i]).collect();
    TriggerRouter::rank_triggers(&refs, ctx)
        .iter()
        .map(|d: &RoutingDecision| (d.trigger_id, d.score.to_bits()))
        .collect()
}

#[derive(Deserialize)]
struct TriggerSpec {
    name: String,
    entity_type: String,
    action: String,
    payload_conditions: Option<Value>,
}

#[test]
fn trigger_ranking_is_invariant_to_candidate_order_on_the_bench_fixture() {
    let fixtures = embedded_fixtures().expect("the embedded fixtures parse");
    let fixture = fixtures
        .iter()
        .find(|f| f.classifier == "triggers")
        .expect("the triggers fixture");
    let specs: Vec<TriggerSpec> =
        serde_json::from_value(fixture.catalog["triggers"].clone()).expect("trigger catalog");
    let triggers: Vec<EventTrigger> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            trigger(
                i as u128,
                &s.name,
                &s.entity_type,
                &s.action,
                s.payload_conditions.clone(),
            )
        })
        .collect();

    let mut checked = 0usize;
    for case in &fixture.cases {
        let entity = case.input["entity_type"].as_str().unwrap_or_default();
        let action = case.input["action"].as_str().unwrap_or_default();
        let payload = case.input.get("payload").cloned().unwrap_or(Value::Null);
        let ctx = TriggerRouter::build_context_from_event(&event(entity, action, payload));
        let reference = ranking(&triggers, &(0..triggers.len()).collect::<Vec<_>>(), &ctx);
        for order in orders(triggers.len()) {
            assert_eq!(
                ranking(&triggers, &order, &ctx),
                reference,
                "case {}: the ranking depends on the candidate order {order:?}",
                case.id
            );
            checked += 1;
        }
    }
    println!("trigger permutation: {checked} orderings checked");
}

#[test]
fn trigger_ties_are_ordered_by_id_not_by_input_position() {
    // Three triggers with the same patterns: identical scores, a genuine tie.
    let triggers = vec![
        trigger(7, "tie_c", "note", "created", None),
        trigger(2, "tie_a", "note", "created", None),
        trigger(5, "tie_b", "note", "created", None),
    ];
    let ctx = TriggerRouter::build_context_from_event(&event("note", "created", Value::Null));
    let reference = ranking(&triggers, &[0, 1, 2], &ctx);
    for order in orders(triggers.len()) {
        assert_eq!(
            ranking(&triggers, &order, &ctx),
            reference,
            "ties: the ranking depends on the input order {order:?}"
        );
    }
}

#[test]
fn a_candidate_scoring_below_the_best_never_changes_the_best() {
    let triggers = [
        trigger(1, "exact", "note", "created", None),
        trigger(2, "other_entity", "plan", "status_changed", None),
    ];
    let ctx = TriggerRouter::build_context_from_event(&event("note", "created", Value::Null));
    let alone = TriggerRouter::rank_triggers(&[&triggers[0]], &ctx);
    let both = TriggerRouter::rank_triggers(&[&triggers[0], &triggers[1]], &ctx);
    assert!(
        both[0].score >= alone[0].score,
        "adding a candidate lowered the best score"
    );
    assert_eq!(both[0].trigger_id, alone[0].trigger_id);
}

fn protocol(index: u128, relevance: RelevanceVector) -> Protocol {
    let mut p = Protocol::new(Uuid::nil(), format!("proto-{index}"), Uuid::nil());
    p.id = Uuid::from_u128(index + 100);
    p.relevance_vector = Some(relevance);
    p
}

#[test]
fn protocol_ranking_is_invariant_to_candidate_order_including_ties() {
    let ctx = ContextVector::from_plan_context("execution", 6, 4, 8, 0.4);
    // Two protocols with the default (neutral) relevance: an exact tie.
    let protocols = [
        protocol(3, RelevanceVector::default()),
        protocol(1, RelevanceVector::default()),
        protocol(
            2,
            RelevanceVector {
                phase: 0.9,
                ..RelevanceVector::default()
            },
        ),
    ];
    let weights = DimensionWeights::default();
    let rank = |order: &[usize]| -> Vec<(Uuid, u64)> {
        let ordered: Vec<Protocol> = order.iter().map(|&i| protocols[i].clone()).collect();
        rank_protocols(&ctx, &ordered, &weights)
            .results
            .iter()
            .map(|r| (r.protocol_id, r.affinity.score.to_bits()))
            .collect()
    };
    let reference = rank(&[0, 1, 2]);
    for order in orders(protocols.len()) {
        assert_eq!(
            rank(&order),
            reference,
            "protocol ranking depends on the input order {order:?}"
        );
    }
}

#[test]
fn affinity_does_not_decrease_as_the_context_moves_toward_the_relevance() {
    // Phase distance from 1.0 down to 0.0 in 10 steps; the relevance sits at 0.0.
    let relevance = RelevanceVector {
        phase: 0.0,
        ..RelevanceVector::default()
    };
    let weights = DimensionWeights::default();
    let mut previous = f64::NEG_INFINITY;
    for step in 0..=10 {
        let ctx = ContextVector {
            phase: 1.0 - step as f64 / 10.0,
            ..ContextVector::from_plan_context("execution", 6, 4, 8, 0.4)
        };
        let score = compute_affinity(&ctx, &relevance, &weights).score;
        assert!(
            score + 1e-12 >= previous,
            "affinity dropped at step {step}: {previous} -> {score}"
        );
        previous = score;
    }
}

#[test]
fn structure_is_monotone_in_each_count_it_is_meant_to_follow() {
    // More affected files, or more dependencies at a fixed task count, never
    // lower the structure score.
    let mut previous = f64::NEG_INFINITY;
    for files in 0..=30 {
        let s = ContextVector::from_plan_context("execution", 5, 3, files, 0.0).structure;
        assert!(
            s + 1e-12 >= previous,
            "structure dropped with files={files}"
        );
        previous = s;
    }
    let mut previous = f64::NEG_INFINITY;
    for deps in 0..=10 {
        let s = ContextVector::from_plan_context("execution", 5, deps, 5, 0.0).structure;
        assert!(s + 1e-12 >= previous, "structure dropped with deps={deps}");
        previous = s;
    }
    // More tasks, with no dependency: no dependency density to dilute.
    let mut previous = f64::NEG_INFINITY;
    for tasks in 0..=20 {
        let s = ContextVector::from_plan_context("execution", tasks, 0, 5, 0.0).structure;
        assert!(
            s + 1e-12 >= previous,
            "structure dropped with tasks={tasks}"
        );
        previous = s;
    }
}
