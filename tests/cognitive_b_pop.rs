//! `B_pop` — a decision must not move when the context carries no evidence.
//!
//! There is no gold label for "which model should have answered this turn", so
//! accuracy is unmeasurable here. Invariance is not: a context that states no
//! fact about the task must not change the decision. `B_pop` is the size of that
//! change, measured as the total-variation distance between the normalised score
//! vectors of two such contexts:
//!
//! ```text
//! B_pop = E_x d( pi(.|x, c1), pi(.|x, c2) )   with c1, c2 evidentially empty
//! ```
//!
//! Two checks, and neither means anything without the other:
//!
//!  - `invariance` — once every arm has observations, a declared preference
//!    changes NOTHING. `B_pop` must be exactly 0, not merely small: the decision
//!    is a pure function of `(id, request, arms, hints)` on a fixed seed, so any
//!    non-zero value is a leak, not noise.
//!  - `control` — on FRESH arms the same preference is an honest prior and DOES
//!    move the decision. This is the non-vacuity guard: if the control ever
//!    reaches 100 % agreement, the invariance above is measuring nothing, and
//!    this file would pass while telling no one anything.
//!
//! Both directions are therefore guarded. A prior made permanent (applied at
//! `n > 0`) turns `invariance` red; a prior removed entirely turns `control` red.
//!
//! What is NOT here, deliberately: no inverse-popularity sampling, no entropy
//! bonus on the utility. `B_pop` is a measurement, never a term of the score.
//! Exploration stays bounded by `exploration_epsilon` and the Beta forgetting.
//!
//! Offline, deterministic, no network, no embedding.

mod common;

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use common::metamorphic::Report;
use nexus_claude::agent::{CostBasis, ModelPrice};

use project_orchestrator::chat::provider::cognitive::candidates::ModelFacts;
use project_orchestrator::chat::provider::cognitive::decider::decide_with;
use project_orchestrator::chat::provider::cognitive::decision::{CognitiveDecision, DecideRequest};
use project_orchestrator::chat::provider::cognitive::mode::RoutingSettings;
use project_orchestrator::chat::provider::cognitive::scorer::{PriorHints, ALIAS_DEEP};
use project_orchestrator::chat::provider::cognitive::signature::{
    ContextHints, TaskClass, TaskSignature,
};
use project_orchestrator::chat::provider::cognitive::store::{ArmKey, ArmStats};

/// How many decisions each check draws its seeds from. Fixed, so the numbers in
/// the PR body are reproducible on any machine.
const SEEDS: u128 = 256;

fn facts(provider: &str, model: &str) -> ModelFacts {
    ModelFacts {
        provider_id: provider.into(),
        model: model.into(),
        supports_tools: true,
        supports_images: true,
        context_window: Some(128_000),
        window_unknown: None,
        price: Some(ModelPrice {
            input_per_mtok: 1.0,
            output_per_mtok: 1.0,
            cache_read_per_mtok: None,
            cache_write_per_mtok: None,
        }),
        cost_basis: CostBasis::Priced,
        healthy: Some(true),
        allowed_for_project: true,
        sandboxed: false,
        quota_exhausted: None,
        quota_utilization: None,
    }
}

fn pool() -> Vec<ModelFacts> {
    // Two arms identical in everything the scorer reads except their name, so
    // the ONLY thing that can separate them is the posterior — never a price.
    vec![facts("alpha", "m-a"), facts("beta", "m-b")]
}

fn signature() -> TaskSignature {
    // `Complex` is a class `deep` suits, so a hint is able to move a fresh arm:
    // that is what makes the control below a real control.
    let mut signature = TaskSignature::from_chat_request(
        "refactor the cognitive router",
        true,
        Some("po"),
        ContextHints::default(),
    );
    signature.class = TaskClass::Complex;
    signature
}

fn settings() -> RoutingSettings {
    RoutingSettings {
        // Exploration would add coin flips to a measurement of determinism.
        exploration_epsilon: 0.0,
        ..RoutingSettings::default()
    }
}

/// Two arms that have both been observed: the prior is out of the picture.
fn observed_arms(signature: &TaskSignature) -> Vec<ArmStats> {
    let class = signature.arm_key();
    let mut alpha = ArmStats::fresh(ArmKey::new(class.clone(), "alpha", "m-a"), 1.0, 1.0);
    alpha.alpha = 4.0;
    alpha.beta = 3.0;
    alpha.n = 7;
    let mut beta = ArmStats::fresh(ArmKey::new(class, "beta", "m-b"), 1.0, 1.0);
    beta.alpha = 3.0;
    beta.beta = 4.0;
    beta.n = 7;
    vec![alpha, beta]
}

fn seed(i: u128) -> Uuid {
    Uuid::from_u128(i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(52))
}

fn decide(id: Uuid, arms: &[ArmStats], hints: &PriorHints) -> CognitiveDecision {
    decide_with(
        id,
        DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_default(),
        &DecideRequest::new(signature(), settings(), pool()),
        arms,
        hints,
    )
}

fn chosen(decision: &CognitiveDecision) -> String {
    match &decision.chosen {
        Some(pick) => format!("{}/{}", pick.provider_id, pick.model),
        None => format!("none ({})", decision.reason),
    }
}

/// `1 - total variation` between the two normalised score vectors, in `[0, 1]`.
/// 1.0 when the two decisions distribute their mass identically.
fn similarity(a: &CognitiveDecision, b: &CognitiveDecision) -> f64 {
    fn distribution(decision: &CognitiveDecision) -> BTreeMap<String, f64> {
        let raw: BTreeMap<String, f64> = decision
            .alternatives
            .iter()
            .filter_map(|alt| {
                alt.score
                    .map(|s| (format!("{}/{}", alt.pick.provider_id, alt.pick.model), s))
            })
            .collect();
        let total: f64 = raw
            .values()
            .filter(|s| s.is_finite())
            .map(|s| s.max(0.0))
            .sum();
        if total <= 0.0 {
            return BTreeMap::new();
        }
        raw.into_iter()
            .map(|(k, s)| {
                let value = if s.is_finite() { s.max(0.0) } else { 0.0 };
                (k, value / total)
            })
            .collect()
    }

    let (da, db) = (distribution(a), distribution(b));
    let mut distance = 0.0;
    for k in da.keys().chain(db.keys()) {
        let x = da.get(k).copied().unwrap_or(0.0);
        let y = db.get(k).copied().unwrap_or(0.0);
        distance += (x - y).abs();
    }
    (1.0 - distance / 2.0).clamp(0.0, 1.0)
}

fn hints_deep_on(provider: &str, model: &str) -> PriorHints {
    PriorHints::new().with(provider, model, ALIAS_DEEP)
}

#[test]
fn b_pop_is_exactly_zero_once_every_arm_has_observations() {
    let arms = observed_arms(&signature());
    let contexts: [(&str, PriorHints); 3] = [
        ("none", PriorHints::new()),
        ("deep=alpha", hints_deep_on("alpha", "m-a")),
        ("deep=beta", hints_deep_on("beta", "m-b")),
    ];
    let mut report = Report::new("B_pop (arms observed)");
    let mut worst = 0.0_f64;

    for i in 0..SEEDS {
        let id = seed(i);
        let decisions: Vec<(&str, CognitiveDecision)> = contexts
            .iter()
            .map(|(name, hints)| (*name, decide(id, &arms, hints)))
            .collect();
        // Every pair of contexts must agree, not just the first with the others.
        for (a, (name_a, left)) in decisions.iter().enumerate() {
            for (name_b, right) in decisions.iter().skip(a + 1) {
                let s = similarity(left, right);
                worst = worst.max(1.0 - s);
                report.observe(
                    &format!("seed {i}: {name_a} vs {name_b}"),
                    &chosen(left),
                    &chosen(right),
                    s,
                );
            }
        }
    }

    report.print();
    println!(
        "B_pop (arms observed, {} contexts, {SEEDS} seeds) = {worst:.6}  [expected 0.000000]",
        contexts.len()
    );
    assert_eq!(
        report.violations().len(),
        0,
        "a declared preference moved a decision taken on observed arms:\n{}",
        report.violations().join("\n")
    );
    assert_eq!(worst, 0.0, "B_pop must be exactly 0 on observed arms");
}

#[test]
fn the_prior_does_move_a_fresh_arm_so_the_invariant_is_not_vacuous() {
    let mut report = Report::new("B_pop control (fresh arms)");
    let plain = PriorHints::new();
    let deep_alpha = hints_deep_on("alpha", "m-a");
    let deep_beta = hints_deep_on("beta", "m-b");
    let mut moved = 0_u32;

    for i in 0..SEEDS {
        let id = seed(i);
        // No stored arm at all: `scorer::score` falls back to `prior_for`.
        let a = decide(id, &[], &plain);
        let b = decide(id, &[], &deep_alpha);
        let c = decide(id, &[], &deep_beta);
        if chosen(&a) != chosen(&b) || chosen(&a) != chosen(&c) {
            moved += 1;
        }
        report.observe(
            &format!("seed {i}: none vs deep=alpha"),
            &chosen(&a),
            &chosen(&b),
            similarity(&a, &b),
        );
        report.observe(
            &format!("seed {i}: none vs deep=beta"),
            &chosen(&a),
            &chosen(&c),
            similarity(&a, &c),
        );
    }

    report.print();
    println!(
        "control: the hint moved the chosen pair on {moved}/{SEEDS} seeds ({:.1}%)",
        100.0 * f64::from(moved) / SEEDS as f64
    );
    assert!(
        report.agreement() < 1.0,
        "the alias prior changed nothing even on fresh arms: the invariance test above \
         measures nothing and must not be trusted"
    );
}

#[test]
fn candidate_order_does_not_move_the_decision() {
    let arms = observed_arms(&signature());
    let hints = hints_deep_on("alpha", "m-a");
    let mut report = Report::new("candidate order");

    for i in 0..SEEDS {
        let id = seed(i);
        let direct = decide(id, &arms, &hints);
        let mut reversed_request = DecideRequest::new(signature(), settings(), pool());
        reversed_request.pool.reverse();
        let reversed = decide_with(
            id,
            DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_default(),
            &reversed_request,
            &arms,
            &hints,
        );
        report.observe(
            &format!("seed {i}: input order vs reversed"),
            &chosen(&direct),
            &chosen(&reversed),
            similarity(&direct, &reversed),
        );
    }

    report.print();
    assert_eq!(
        report.violations().len(),
        0,
        "{}",
        report.violations().join("\n")
    );
}

#[test]
fn the_same_id_replays_the_same_decision() {
    let arms = observed_arms(&signature());
    let hints = hints_deep_on("beta", "m-b");
    let id = seed(7);
    let first = decide(id, &arms, &hints);
    let second = decide(id, &arms, &hints);
    assert_eq!(chosen(&first), chosen(&second));
    assert_eq!(similarity(&first, &second), 1.0);
}
