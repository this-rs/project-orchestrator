//! Metamorphic harness shared by the offline classifier tests.
//!
//! Ported from Laya's `research/eval/metamorphic.py` (Convai Innovations, Apache-2.0).
//!
//! We rarely have gold labels for what a classifier should answer. Invariance
//! needs none: two inputs that mean the same thing must get the same answer,
//! and a violation is a bug even when the right answer is unknown.
//!
//! The harness only compares outcomes. Each test chooses its equivalence
//! (translation, paraphrase, noise, permutation, monotonicity) and renders its
//! outcomes as strings; `Report` counts agreement and mean similarity.
//!
//! Nothing here touches the network or an embedding model.
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use std::collections::BTreeSet;

/// Jaccard similarity of two sets. Two empty sets are identical.
pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// Similarity of two single labels: `1.0` when equal, `0.0` otherwise.
pub fn label_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        1.0
    } else {
        0.0
    }
}

/// Renders a set as `a+b+c`, sorted, `-` when empty.
pub fn fmt_set(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        return "-".to_string();
    }
    set.iter().map(String::as_str).collect::<Vec<_>>().join("+")
}

/// Counts equivalence checks: how many agree exactly, the mean similarity, and
/// one line per violation.
pub struct Report {
    kind: &'static str,
    n: usize,
    agree: usize,
    similarity_sum: f64,
    violations: Vec<String>,
}

impl Report {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            n: 0,
            agree: 0,
            similarity_sum: 0.0,
            violations: Vec::new(),
        }
    }

    /// Records one check. `a` and `b` are the rendered outcomes; `similarity`
    /// is in `[0, 1]` (Jaccard for sets, `label_similarity` for labels).
    pub fn observe(&mut self, label: &str, a: &str, b: &str, similarity: f64) {
        self.n += 1;
        self.similarity_sum += similarity;
        if a == b {
            self.agree += 1;
        } else {
            self.violations
                .push(format!("    {label}\n      A = {a}\n      B = {b}"));
        }
    }

    pub fn print(&self) {
        println!(
            "\n── invariant `{}` ───────────────────────────────────────────\n\
             cases: {}   agreement: {}/{} = {:.1}%   mean similarity: {:.3}",
            self.kind,
            self.n,
            self.agree,
            self.n,
            self.agreement() * 100.0,
            self.similarity_sum / self.n.max(1) as f64
        );
        if self.violations.is_empty() {
            println!("  no violations");
        } else {
            println!("  VIOLATIONS ({}):", self.violations.len());
            for v in &self.violations {
                println!("{v}");
            }
        }
    }

    /// Share of checks that agree exactly. `0.0` with no check.
    pub fn agreement(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.agree as f64 / self.n as f64
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn violations(&self) -> &[String] {
        &self.violations
    }
}

/// Sentences that carry no intent and must not change a decision.
pub const NOISE: &[&str] = &[
    " Merci d'avance.",
    " On en reparle à la prochaine session.",
    " C'est urgent.",
];

/// Surface punctuation noise: the same message with its punctuation changed.
/// Each variant keeps every word and only moves or repeats the marks.
pub fn punctuation_noise(message: &str) -> Vec<String> {
    vec![
        format!("{message} !!!"),
        format!("{message}..."),
        message.replace(['?', '!'], "").trim().to_string(),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// `B_pop` — a decision must not move when the context carries no evidence.
// ─────────────────────────────────────────────────────────────────────────────
//
// There is no gold label for "which model should have answered this turn", so
// accuracy is unmeasurable on the router. Invariance is not: a context that
// states no fact about the task must not change the decision. `B_pop` is the
// total-variation distance between the normalised score vectors of two such
// contexts, and it must be EXACTLY zero — the decision is a pure function of
// `(id, request, arms, hints)` on a fixed seed, so any non-zero value is a leak,
// not noise.
//
// A measurement with no control is a decoration: the control below runs the same
// three contexts on FRESH arms, where the declared preference is an honest prior
// and MUST move the decision. If it ever stops moving it, `B_pop = 0` above
// measures nothing.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

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

/// How many decisions a `B_pop` measurement draws its seeds from. Fixed, so the
/// published numbers are reproducible on any machine.
pub const B_POP_SEEDS: u128 = 256;

/// Whether the two fixture arms have been observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arms {
    /// No stored `ArmStats`: `scorer::score` falls back to `prior_for`, so the
    /// alias hint is live. An honest prior, and the control uses it.
    Fresh,
    /// Both arms observed: the prior is out of the picture, so the hint must do
    /// nothing at all.
    Observed,
}

/// The fixture pair: two arms identical in everything the scorer reads except
/// their name, so the ONLY thing that can separate them is the posterior.
pub fn b_pop_facts() -> Vec<ModelFacts> {
    vec![facts("alpha", "m-a"), facts("beta", "m-b")]
}

fn facts(provider: &str, model: &str) -> ModelFacts {
    ModelFacts {
        provider_id: provider.into(),
        model: model.into(),
        supports_tools: true,
        // The signature below needs tools and images: a pair rejected by
        // `candidates::apply` would make every context say `no_candidate` and
        // the whole measurement would be vacuously green.
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

/// A `Complex` signature: the class `deep` suits, so a hint is able to move a
/// fresh arm — which is what makes the control a real control.
pub fn b_pop_signature() -> TaskSignature {
    let mut signature = TaskSignature::from_chat_request(
        "refactor the cognitive router",
        true,
        Some("po"),
        ContextHints::default(),
    );
    signature.class = TaskClass::Complex;
    signature
}

fn b_pop_settings() -> RoutingSettings {
    RoutingSettings {
        // Exploration would add coin flips to a measurement of determinism.
        exploration_epsilon: 0.0,
        ..RoutingSettings::default()
    }
}

/// Two arms that have both been observed: the prior is out of the picture.
pub fn b_pop_observed_arms(signature: &TaskSignature) -> Vec<ArmStats> {
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

/// The seed of the `i`-th decision: derived, never drawn, so a run replays.
pub fn b_pop_seed(i: u128) -> Uuid {
    Uuid::from_u128(i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(52))
}

fn b_pop_at() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_default()
}

/// The three contexts of the measurement: no hint, and the same alias declared
/// on each arm in turn. None of them states anything about the task.
pub fn b_pop_hints() -> Vec<(&'static str, PriorHints)> {
    vec![
        ("none", PriorHints::new()),
        (
            "deep=alpha",
            PriorHints::new().with("alpha", "m-a", ALIAS_DEEP),
        ),
        (
            "deep=beta",
            PriorHints::new().with("beta", "m-b", ALIAS_DEEP),
        ),
    ]
}

/// One decision of the fixture, with the given arms state and hint context.
pub fn b_pop_decide(id: Uuid, arms: Arms, hints: &PriorHints) -> CognitiveDecision {
    let stored = match arms {
        Arms::Fresh => Vec::new(),
        Arms::Observed => b_pop_observed_arms(&b_pop_signature()),
    };
    decide_with(
        id,
        b_pop_at(),
        &DecideRequest::new(b_pop_signature(), b_pop_settings(), b_pop_facts()),
        &stored,
        hints,
    )
}

/// The pair a decision chose, rendered: what an invariance check compares.
pub fn chosen(decision: &CognitiveDecision) -> String {
    match &decision.chosen {
        Some(pick) => format!("{}/{}", pick.provider_id, pick.model),
        None => format!("none ({})", decision.reason),
    }
}

/// `1 - total variation` between the two normalised score vectors, in `[0, 1]`.
pub fn similarity(a: &CognitiveDecision, b: &CognitiveDecision) -> f64 {
    let distribution = |decision: &CognitiveDecision| -> BTreeMap<String, f64> {
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
    };

    let (da, db) = (distribution(a), distribution(b));
    let mut distance = 0.0;
    for k in da.keys().chain(db.keys()) {
        distance += (da.get(k).copied().unwrap_or(0.0) - db.get(k).copied().unwrap_or(0.0)).abs();
    }
    (1.0 - distance / 2.0).clamp(0.0, 1.0)
}

/// `B_pop` over a fixed set of seeds, with its control.
#[derive(Debug, Clone, Default)]
pub struct BPop {
    /// Seeds drawn.
    pub seeds: u128,
    /// Invariance on OBSERVED arms: pairs of evidentially-empty contexts compared.
    pub observed_cases: usize,
    /// How many of those agreed exactly.
    pub observed_agree: usize,
    /// Worst `B_pop` seen over those pairs. Must be `0.0`.
    pub observed_b_pop: f64,
    /// Order invariance: input order against reversed input.
    pub order_cases: usize,
    pub order_agree: usize,
    /// Control on FRESH arms: the same contexts, where the prior is live.
    pub control_cases: usize,
    pub control_agree: usize,
    /// Seeds where the declared preference changed the chosen pair.
    pub control_moved_seeds: usize,
}

impl BPop {
    /// Share of observed-arm pairs that agreed. `1.0` is the only acceptable value.
    pub fn observed_agreement(&self) -> f64 {
        if self.observed_cases == 0 {
            return 0.0;
        }
        self.observed_agree as f64 / self.observed_cases as f64
    }

    /// Share of control pairs that agreed. Strictly below `1.0`, or the
    /// invariance above measures nothing.
    pub fn control_agreement(&self) -> f64 {
        if self.control_cases == 0 {
            return 1.0;
        }
        self.control_agree as f64 / self.control_cases as f64
    }

    /// The whole measurement as one printable line.
    pub fn line(&self) -> String {
        format!(
            "B_pop: invariance {}/{} = {:.1}% (B_pop = {:.6}), order {}/{} = {:.1}%, \
             control {}/{} = {:.1}% ({} seeds moved)",
            self.observed_agree,
            self.observed_cases,
            self.observed_agreement() * 100.0,
            self.observed_b_pop,
            self.order_agree,
            self.order_cases,
            if self.order_cases == 0 {
                0.0
            } else {
                self.order_agree as f64 / self.order_cases as f64 * 100.0
            },
            self.control_agree,
            self.control_cases,
            self.control_agreement() * 100.0,
            self.control_moved_seeds,
        )
    }
}

/// Measures `B_pop` and its control over `B_POP_SEEDS` seeds.
pub fn b_pop() -> BPop {
    b_pop_over(B_POP_SEEDS)
}

/// Measures `B_pop` over `seeds` seeds.
pub fn b_pop_over(seeds: u128) -> BPop {
    let mut measured = BPop {
        seeds,
        ..BPop::default()
    };
    let contexts = b_pop_hints();
    let arms = b_pop_observed_arms(&b_pop_signature());

    for i in 0..seeds {
        let id = b_pop_seed(i);

        // Invariance — every pair of evidentially-empty contexts must agree once
        // both arms are observed.
        let observed: Vec<CognitiveDecision> = contexts
            .iter()
            .map(|(_, hints)| b_pop_decide(id, Arms::Observed, hints))
            .collect();
        for a in 0..observed.len() {
            for b in (a + 1)..observed.len() {
                measured.observed_cases += 1;
                let s = similarity(&observed[a], &observed[b]);
                measured.observed_b_pop = measured.observed_b_pop.max(1.0 - s);
                if chosen(&observed[a]) == chosen(&observed[b]) {
                    measured.observed_agree += 1;
                }
            }
        }

        // Order — the pool's input order must not matter either.
        let direct = b_pop_decide(id, Arms::Observed, &contexts[1].1);
        let mut reversed_pool = b_pop_facts();
        reversed_pool.reverse();
        let reversed_request =
            DecideRequest::new(b_pop_signature(), b_pop_settings(), reversed_pool);
        let reversed = decide_with(id, b_pop_at(), &reversed_request, &arms, &contexts[1].1);
        measured.order_cases += 1;
        if chosen(&direct) == chosen(&reversed) {
            measured.order_agree += 1;
        }

        // Control — fresh arms, where the prior is live and MUST move the choice.
        let fresh: Vec<CognitiveDecision> = contexts
            .iter()
            .map(|(_, hints)| b_pop_decide(id, Arms::Fresh, hints))
            .collect();
        for a in 0..fresh.len() {
            for b in (a + 1)..fresh.len() {
                measured.control_cases += 1;
                if chosen(&fresh[a]) == chosen(&fresh[b]) {
                    measured.control_agree += 1;
                }
            }
        }
        if chosen(&fresh[0]) != chosen(&fresh[1]) || chosen(&fresh[0]) != chosen(&fresh[2]) {
            measured.control_moved_seeds += 1;
        }
    }

    measured
}
