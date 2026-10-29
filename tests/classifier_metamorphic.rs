//! Metamorphic invariants of the classifiers, offline (no network, no embedding).
//!
//! No gold labels are needed: each test states a property that must hold for
//! every input it draws, and fails when the code breaks it.
//!
//! Text classifiers (inputs are strings):
//!  - `IntentDetector::detect`: appending or prepending text can only raise the
//!    intent's priority (keywords are only ever added); case and surrounding
//!    whitespace change nothing.
//!  - `TaskSignature::from_chat_request`: same intent class under case and
//!    whitespace changes; complexity and context need grow with the length of
//!    a message that keeps its class; bounds and passthroughs hold.
//!  - `profile_task`: tag order and case change nothing; the profile's rank
//!    never drops when steps, affected files or tags are added.
//!  - skill matching on a message (`evaluate_skill_match`): case-insensitive;
//!    unanchored patterns stay matched when text is appended; adding a trigger
//!    or reordering triggers never lowers or changes the score.
//!
//! Numeric classifiers (inputs are contexts and candidate lists):
//!  - `TriggerRouter::rank_triggers` and `rank_protocols`: the sorted score
//!    sequence is invariant under input permutation, scores stay in `[0, 1]`,
//!    and an affinity moves monotonically toward its relevance vector.
//!
//! Where the code breaks a property, the test is kept with `#[ignore]` and the
//! counterexample written above it. The code is not changed in this file.
use project_orchestrator::neo4j::traits::*;

use chrono::{DateTime, Utc};
use project_orchestrator::chat::provider::cognitive::signature::{
    ContextHints, TaskSignature, MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS,
};
use project_orchestrator::evaluation::embedded_fixtures;
use project_orchestrator::events::trigger::EventTrigger;
use project_orchestrator::events::trigger_routing::{RoutingContext, TriggerRouter};
use project_orchestrator::neo4j::models::{TaskNode, TaskStatus};
use project_orchestrator::neurons::intent::{IntentDetector, QueryIntentMode};
use project_orchestrator::protocol::models::Protocol;
use project_orchestrator::protocol::routing::{
    rank_protocols, ContextVector, DimensionWeights, RelevanceVector,
};
use project_orchestrator::runner::persona::{profile_task, Complexity};
use project_orchestrator::skills::activation::evaluate_skill_match;
use project_orchestrator::skills::models::{SkillNode, SkillTrigger};
use uuid::Uuid;

// ============================================================================
// Corpora
// ============================================================================

/// Messages from the intent bench fixture, plus a few written by hand. Mixed
/// FR and EN, with and without keywords, and with an empty message.
fn messages() -> Vec<String> {
    let mut out: Vec<String> = embedded_fixtures()
        .expect("the embedded fixtures parse")
        .into_iter()
        .filter(|f| f.classifier == "intent")
        .flat_map(|f| f.cases)
        .filter_map(|c| {
            c.input
                .get("message")
                .and_then(|m| m.as_str())
                .map(String::from)
        })
        .collect();
    out.extend(
        [
            "",
            "payment service",
            "Neo4j client",
            "pourquoi le paiement échoue",
            "COMMENT FONCTIONNE LE CACHE",
            "ajouter un middleware de rate limiting",
            "why and how does this work",
            "change and implement the auth flow",
        ]
        .map(String::from),
    );
    out
}

/// Suffixes and prefixes that include keywords of every intent, to exercise
/// the priority cascade.
const EXTRA_TEXT: &[&str] = &[
    "",
    " pourquoi",
    " bug",
    " impact",
    " change",
    " planifie",
    " implement",
    " comment",
    " how does",
    "!!!",
    " zzz",
];

/// Rank of an intent in the priority cascade (Debug > Impact > Plan > Explore > Default).
fn priority(intent: QueryIntentMode) -> u8 {
    match intent {
        QueryIntentMode::Debug => 4,
        QueryIntentMode::Impact => 3,
        QueryIntentMode::Plan => 2,
        QueryIntentMode::Explore => 1,
        QueryIntentMode::Default => 0,
    }
}

// ============================================================================
// IntentDetector
// ============================================================================

#[test]
fn intent_priority_never_drops_when_text_is_appended_or_prepended() {
    for message in messages() {
        let base = priority(IntentDetector::detect(&message));
        for extra in EXTRA_TEXT {
            let after = format!("{message}{extra}");
            let before = format!("{extra}{message}");
            assert!(
                priority(IntentDetector::detect(&after)) >= base,
                "appending {extra:?} lowered the intent of {message:?}"
            );
            assert!(
                priority(IntentDetector::detect(&before)) >= base,
                "prepending {extra:?} lowered the intent of {message:?}"
            );
        }
    }
}

#[test]
fn intent_is_invariant_to_case() {
    for message in messages() {
        assert_eq!(
            IntentDetector::detect(&message),
            IntentDetector::detect(&message.to_uppercase()),
            "upper-casing changed the intent of {message:?}"
        );
    }
}

#[test]
fn intent_is_invariant_to_surrounding_whitespace() {
    for message in messages() {
        let padded = format!("  \t{message}\n ");
        assert_eq!(
            IntentDetector::detect(&message),
            IntentDetector::detect(&padded),
            "surrounding whitespace changed the intent of {message:?}"
        );
    }
}

// ============================================================================
// TaskSignature::from_chat_request
// ============================================================================

#[test]
fn chat_signature_class_is_invariant_to_case_and_whitespace() {
    for message in messages() {
        let base = TaskSignature::from_chat_request(&message, false, None, ContextHints::default());
        let upper = TaskSignature::from_chat_request(
            &message.to_uppercase(),
            false,
            None,
            ContextHints::default(),
        );
        let padded = TaskSignature::from_chat_request(
            &format!(" {message} \n"),
            false,
            None,
            ContextHints::default(),
        );
        assert_eq!(
            base.arm_key(),
            upper.arm_key(),
            "case changed the class of {message:?}"
        );
        assert_eq!(
            base.arm_key(),
            padded.arm_key(),
            "whitespace changed the class of {message:?}"
        );
    }
}

#[test]
fn chat_signature_grows_with_length_when_the_class_is_kept() {
    // `z` is in no intent keyword, so appending it cannot create a match: the
    // class stays fixed and only the length-driven fields may move.
    for message in messages() {
        let mut last: Option<(u8, u64)> = None;
        for padding in (0..=1800).step_by(50) {
            let text = format!("{message}{}", "z".repeat(padding));
            let sig = TaskSignature::from_chat_request(&text, false, None, ContextHints::default());
            let current = (sig.complexity, sig.context_need_tokens);
            if let Some((complexity, need)) = last {
                assert!(
                    current.0 >= complexity,
                    "complexity fell from {complexity} to {} when {padding} chars were added to {message:?}",
                    current.0
                );
                assert!(
                    current.1 >= need,
                    "context need fell from {need} to {} when {padding} chars were added to {message:?}",
                    current.1
                );
            }
            last = Some(current);
        }
    }
}

#[test]
fn chat_signature_stays_within_its_bounds_and_passes_inputs_through() {
    for message in messages() {
        let long = format!("{message}{}", "z".repeat(5_000));
        for text in [message.as_str(), long.as_str()] {
            let hints = ContextHints {
                affected_bytes: None,
                mcp_schema_tokens: 0,
                budget_remaining_usd: Some(1.5),
            };
            let sig = TaskSignature::from_chat_request(text, true, Some("p"), hints);
            assert!(
                (1..=10).contains(&sig.complexity),
                "complexity {} out of 1..=10",
                sig.complexity
            );
            assert!(
                (MIN_CONTEXT_TOKENS..=MAX_CONTEXT_TOKENS).contains(&sig.context_need_tokens),
                "context need {} out of bounds",
                sig.context_need_tokens
            );
            assert!(sig.needs_images, "has_images=true must set needs_images");
            assert_eq!(sig.budget_remaining_usd, Some(1.5));
            assert_eq!(sig.project_slug.as_deref(), Some("p"));
        }
    }
}

// ============================================================================
// profile_task
// ============================================================================

fn task(tags: &[&str], files: usize) -> TaskNode {
    TaskNode {
        id: Uuid::new_v4(),
        title: Some("Test task".to_string()),
        description: "A test task description".to_string(),
        status: TaskStatus::Pending,
        assigned_to: None,
        interrupted_at: None,
        interrupted_reason: None,
        priority: None,
        tags: tags.iter().map(|t| t.to_string()).collect(),
        acceptance_criteria: vec![],
        affected_files: (0..files).map(|i| format!("src/f{i}.rs")).collect(),
        estimated_complexity: None,
        due_date: None,
        estimated_minutes: None,
        actual_complexity: None,
        created_at: DateTime::<Utc>::UNIX_EPOCH,
        updated_at: None,
        started_at: None,
        completed_at: None,
        frustration_score: 0.0,
        execution_context: None,
        persona: None,
        prompt_cache: None,
    }
}

/// Order of the complexity classes: Simple < Complex < Creative.
fn rank(c: Complexity) -> u8 {
    match c {
        Complexity::Simple => 0,
        Complexity::Complex => 1,
        Complexity::Creative => 2,
    }
}

/// Tag pool covering every branch of `profile_task`.
const TAG_POOL: &[&str] = &[
    "bugfix",
    "refactor",
    "architecture",
    "design",
    "new-feature",
    "docs",
];

/// Every subset of up to three tags from the pool.
fn tag_sets() -> Vec<Vec<&'static str>> {
    let mut out = vec![vec![]];
    for (i, &a) in TAG_POOL.iter().enumerate() {
        out.push(vec![a]);
        for (j, &b) in TAG_POOL.iter().enumerate().skip(i + 1) {
            out.push(vec![a, b]);
            for &c in TAG_POOL.iter().skip(j + 1) {
                out.push(vec![a, b, c]);
            }
        }
    }
    out
}

#[test]
fn profile_is_invariant_to_tag_order_and_case() {
    for tags in tag_sets() {
        let shuffled: Vec<String> = tags.iter().rev().map(|t| t.to_uppercase()).collect();
        let shuffled: Vec<&str> = shuffled.iter().map(String::as_str).collect();
        for files in [0, 1, 3] {
            for steps in [0, 3, 8] {
                let a = profile_task(&task(&tags, files), steps);
                let b = profile_task(&task(&shuffled, files), steps);
                assert_eq!(a.complexity, b.complexity, "tags {tags:?}");
                assert_eq!(a.timeout_secs, b.timeout_secs, "tags {tags:?}");
                assert_eq!(a.max_cost_usd, b.max_cost_usd, "tags {tags:?}");
                assert_eq!(a.search_temperature, b.search_temperature, "tags {tags:?}");
            }
        }
    }
}

#[test]
fn profile_rank_never_drops_when_steps_grow() {
    for tags in tag_sets() {
        for files in [0, 1, 2, 6] {
            let mut last = 0;
            for steps in 0..=12 {
                let r = rank(profile_task(&task(&tags, files), steps).complexity);
                assert!(
                    r >= last,
                    "rank fell at steps={steps} for tags {tags:?}, files {files}"
                );
                last = r;
            }
        }
    }
}

#[test]
fn profile_rank_never_drops_when_files_grow() {
    for tags in tag_sets() {
        for steps in [0, 3, 8] {
            let mut last = 0;
            for files in 0..=6 {
                let r = rank(profile_task(&task(&tags, files), steps).complexity);
                assert!(
                    r >= last,
                    "rank fell at files={files} for tags {tags:?}, steps {steps}"
                );
                last = r;
            }
        }
    }
}

#[test]
fn profile_rank_never_drops_when_a_tag_is_added() {
    for base in tag_sets() {
        for files in [0, 2] {
            for steps in [0, 6] {
                let before = rank(profile_task(&task(&base, files), steps).complexity);
                for &extra in TAG_POOL {
                    if base.contains(&extra) {
                        continue;
                    }
                    let mut more = base.clone();
                    more.push(extra);
                    let after = rank(profile_task(&task(&more, files), steps).complexity);
                    assert!(
                        after >= before,
                        "adding {extra:?} lowered the rank for {base:?} (files {files}, steps {steps})"
                    );
                }
            }
        }
    }
}

// ============================================================================
// Skill selection on a message
// ============================================================================

/// Unanchored, literal-ish patterns: a match survives any text appended to the
/// message. Anchors and lookarounds are excluded on purpose.
const UNANCHORED: &[&str] = &["cache", "pourquoi|why", "bug", "impact", "cach\\w+", "fail"];

fn skill_with(patterns: &[&str]) -> SkillNode {
    let mut skill = SkillNode::new(Some(Uuid::new_v4()), "skill");
    skill.trigger_patterns = patterns
        .iter()
        .map(|p| SkillTrigger::regex(*p, 0.8))
        .collect();
    skill
}

fn score(skill: &SkillNode, message: &str) -> f64 {
    evaluate_skill_match(skill, Some(message), None)
}

#[test]
fn skill_match_is_invariant_to_case() {
    for &pattern in UNANCHORED {
        let skill = skill_with(&[pattern]);
        for message in messages() {
            assert_eq!(
                score(&skill, &message),
                score(&skill, &message.to_uppercase()),
                "case changed the match of {pattern:?} on {message:?}"
            );
        }
    }
}

#[test]
fn skill_match_survives_appended_text() {
    for &pattern in UNANCHORED {
        let skill = skill_with(&[pattern]);
        for message in messages() {
            let base = score(&skill, &message);
            for extra in EXTRA_TEXT {
                let after = score(&skill, &format!("{message}{extra}"));
                assert!(
                    after >= base,
                    "appending {extra:?} lowered the match of {pattern:?} on {message:?}"
                );
            }
        }
    }
}

#[test]
fn skill_match_never_drops_when_a_trigger_is_added() {
    for message in messages() {
        let one = skill_with(&["cache"]);
        let two = skill_with(&["cache", "bug"]);
        assert!(
            score(&two, &message) >= score(&one, &message),
            "adding a trigger lowered the match on {message:?}"
        );
    }
}

#[test]
fn skill_match_is_invariant_to_trigger_order() {
    for message in messages() {
        let forward = skill_with(&["cache", "bug", "impact"]);
        let backward = skill_with(&["impact", "bug", "cache"]);
        assert_eq!(
            score(&forward, &message),
            score(&backward, &message),
            "trigger order changed the match on {message:?}"
        );
    }
}

// ============================================================================
// Numeric: trigger ranking
// ============================================================================

fn trigger(id: u128, entity: &str, action: &str) -> EventTrigger {
    let epoch = DateTime::<Utc>::UNIX_EPOCH;
    EventTrigger {
        id: Uuid::from_u128(id),
        name: format!("t{id}"),
        protocol_id: Uuid::nil(),
        entity_type_pattern: Some(entity.to_string()),
        action_pattern: Some(action.to_string()),
        payload_conditions: None,
        cooldown_secs: 0,
        enabled: true,
        project_scope: None,
        created_at: epoch,
        updated_at: epoch,
    }
}

/// A grid of routing contexts covering the corners and the centre of the unit cube.
fn contexts() -> Vec<RoutingContext> {
    let levels = [0.0, 0.5, 1.0];
    let mut out = Vec::new();
    for &phase in &levels {
        for &structure in &levels {
            out.push(RoutingContext {
                phase,
                structure,
                domain: 0.5,
                resource: 0.25,
                lifecycle: 1.0 - phase,
            });
        }
    }
    out
}

#[test]
fn trigger_scores_are_bounded_and_sorted_descending() {
    let triggers = [
        trigger(1, "task", "created"),
        trigger(2, "task", "updated"),
        trigger(3, "commit", "created"),
        trigger(4, "skill", "status_changed"),
    ];
    for ctx in contexts() {
        let refs: Vec<&EventTrigger> = triggers.iter().collect();
        let ranked = TriggerRouter::rank_triggers(&refs, &ctx);
        assert_eq!(ranked.len(), triggers.len());
        assert!(ranked.iter().all(|d| (0.0..=1.0).contains(&d.score)));
        assert!(ranked.windows(2).all(|w| w[0].score >= w[1].score));
    }
}

#[test]
fn trigger_score_sequence_is_invariant_to_input_order() {
    let triggers = [
        trigger(1, "task", "created"),
        trigger(2, "task", "updated"),
        trigger(3, "commit", "created"),
        trigger(4, "skill", "status_changed"),
    ];
    for ctx in contexts() {
        let forward: Vec<&EventTrigger> = triggers.iter().collect();
        let backward: Vec<&EventTrigger> = triggers.iter().rev().collect();
        let a: Vec<f64> = TriggerRouter::rank_triggers(&forward, &ctx)
            .iter()
            .map(|d| d.score)
            .collect();
        let b: Vec<f64> = TriggerRouter::rank_triggers(&backward, &ctx)
            .iter()
            .map(|d| d.score)
            .collect();
        assert_eq!(a, b, "input order changed the score sequence");
    }
}

/// Counterexample: two triggers with the same patterns have the same score, a
/// tie. The ranking keeps the input order for ties, so `select_best` picks a
/// different trigger depending on how the candidates were loaded. The fix
/// (ties ordered by id) was once in `9e880c65` and is absent from main.
#[test]
#[ignore = "counterexample: exact score ties keep input order (no id tie-break)"]
fn trigger_ties_are_ordered_by_id_not_input_order() {
    let a = trigger(1, "task", "created");
    let b = trigger(2, "task", "created");
    let ctx = RoutingContext::default();
    let first = TriggerRouter::rank_triggers(&[&a, &b], &ctx)[0].trigger_id;
    let swapped = TriggerRouter::rank_triggers(&[&b, &a], &ctx)[0].trigger_id;
    assert_eq!(first, swapped, "a tie's winner depends on the input order");
}

// ============================================================================
// Numeric: protocol ranking
// ============================================================================

fn protocol(relevance: RelevanceVector) -> Protocol {
    let mut p = Protocol::new(Some(Uuid::new_v4()), "p", Uuid::new_v4());
    p.relevance_vector = Some(relevance);
    p
}

fn rel(phase: f64, structure: f64, domain: f64, resource: f64, lifecycle: f64) -> RelevanceVector {
    RelevanceVector {
        phase,
        structure,
        domain,
        resource,
        lifecycle,
    }
}

fn ctx(phase: f64, structure: f64, domain: f64, resource: f64, lifecycle: f64) -> ContextVector {
    ContextVector {
        phase,
        structure,
        domain,
        resource,
        lifecycle,
    }
}

#[test]
fn protocol_score_sequence_is_invariant_to_input_order_and_bounded() {
    let protocols = [
        protocol(rel(0.0, 0.2, 0.5, 0.5, 0.1)),
        protocol(rel(0.5, 0.8, 0.7, 1.0, 0.3)),
        protocol(rel(1.0, 0.5, 0.2, 0.4, 0.9)),
    ];
    let weights = DimensionWeights::default();
    for c in [
        ctx(0.0, 0.0, 0.0, 0.0, 0.0),
        ctx(0.5, 0.5, 0.5, 0.5, 0.5),
        ctx(1.0, 1.0, 1.0, 1.0, 1.0),
    ] {
        let forward: Vec<Protocol> = protocols.to_vec();
        let backward: Vec<Protocol> = protocols.iter().rev().cloned().collect::<Vec<_>>();
        let a = rank_protocols(&c, &forward, &weights);
        let b = rank_protocols(&c, &backward, &weights);
        assert_eq!(a.total_evaluated, protocols.len());
        let sa: Vec<f64> = a.results.iter().map(|r| r.affinity.score).collect();
        let sb: Vec<f64> = b.results.iter().map(|r| r.affinity.score).collect();
        assert_eq!(sa, sb, "input order changed the score sequence");
        assert!(sa.iter().all(|s| (0.0..=1.0).contains(s)));
        assert!(sa.windows(2).all(|w| w[0] >= w[1]));
    }
}

/// Moving the context along a straight line toward the relevance vector can
/// only raise the affinity: each dimension's similarity `1 - |c - r|` grows
/// and the weights are non-negative.
#[test]
fn protocol_affinity_grows_as_context_moves_toward_relevance() {
    let relevance = rel(0.5, 0.8, 0.7, 1.0, 0.3);
    let p = protocol(relevance.clone());
    let start = ctx(0.0, 0.1, 0.2, 0.0, 0.9);
    let target = [0.5, 0.8, 0.7, 1.0, 0.3];
    let origin = [
        start.phase,
        start.structure,
        start.domain,
        start.resource,
        start.lifecycle,
    ];
    let weights = DimensionWeights::default();
    let mut last = f64::NEG_INFINITY;
    for step in 0..=20 {
        let t = step as f64 / 20.0;
        let at = |i: usize| origin[i] + t * (target[i] - origin[i]);
        let c = ctx(at(0), at(1), at(2), at(3), at(4));
        let score = rank_protocols(&c, std::slice::from_ref(&p), &weights).results[0]
            .affinity
            .score;
        assert!(
            score + 1e-12 >= last,
            "affinity fell at t={t}: {last} -> {score}"
        );
        last = score;
    }
}

/// Counterexample: two protocols with the same relevance vector tie. Ties keep
/// the input order (no id tie-break), so the top protocol depends on the input
/// order. Same root cause as the trigger ties above.
#[test]
#[ignore = "counterexample: exact score ties keep input order (no id tie-break)"]
fn protocol_ties_are_ordered_by_id_not_input_order() {
    let a = protocol(rel(0.5, 0.5, 0.5, 0.5, 0.5));
    let b = protocol(rel(0.5, 0.5, 0.5, 0.5, 0.5));
    let c = ctx(0.5, 0.5, 0.5, 0.5, 0.5);
    let weights = DimensionWeights::default();
    let first = rank_protocols(&c, &[a.clone(), b.clone()], &weights).results[0].protocol_id;
    let swapped = rank_protocols(&c, &[b, a], &weights).results[0].protocol_id;
    assert_eq!(first, swapped, "a tie's winner depends on the input order");
}
