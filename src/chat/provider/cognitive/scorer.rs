//! Scoring: one Thompson draw per eligible pair, minus a normalised cost and a
//! normalised latency, with a bounded epsilon-greedy exploration.
//!
//! Pure and deterministic: for a given `(candidates, arms, settings, seed)` the
//! result is always the same, so a decision replays (its id is the seed). The
//! sampler is a small seeded Marsaglia-Tsang gamma; nothing here reads a clock
//! or a global random source.
//!
//! The declared aliases (`fast`, `deep`) are a PRIOR on a fresh arm, never a
//! rule: one extra success on the classes the alias suits, and the evidence
//! overrides it as soon as the arm has observations.

use std::collections::HashMap;

use uuid::Uuid;

use super::candidates::ModelFacts;
use super::mode::{RoutingSettings, MAX_EXPLORATION_EPSILON};
use super::signature::{ChatIntent, TaskClass, TaskSignature};
use super::store::{ArmKey, ArmStats};
use crate::chat::provider::settings::ModelAlias;

/// Alias that marks a model as the strong, slow one.
pub const ALIAS_DEEP: &str = "deep";
/// Alias that marks a model as the quick, cheap one.
pub const ALIAS_FAST: &str = "fast";

/// Which alias (if any) points at each `(provider, model)`: the declarative
/// configuration, read as a prior.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PriorHints {
    aliases: HashMap<(String, String), String>,
}

impl PriorHints {
    /// No hint.
    pub fn new() -> Self {
        Self::default()
    }

    /// The hints of an alias table.
    pub fn from_aliases(aliases: &[ModelAlias]) -> Self {
        Self {
            aliases: aliases
                .iter()
                .map(|a| ((a.provider.clone(), a.model.clone()), a.alias.clone()))
                .collect(),
        }
    }

    /// Adds one hint.
    pub fn with(mut self, provider: &str, model: &str, alias: &str) -> Self {
        self.aliases
            .insert((provider.to_owned(), model.to_owned()), alias.to_owned());
        self
    }

    /// The alias pointing at a pair.
    pub fn alias_of(&self, provider: &str, model: &str) -> Option<&str> {
        self.aliases
            .get(&(provider.to_owned(), model.to_owned()))
            .map(String::as_str)
    }
}

/// The Beta prior of a fresh arm: Beta(1, 1), plus one success when the alias
/// suits the class (`deep` on complex, creative and retry work, `fast` on
/// simple work and plain conversation).
pub fn prior_for(class: TaskClass, alias: Option<&str>) -> (f64, f64) {
    let bonus = match (alias, class) {
        (Some(ALIAS_DEEP), TaskClass::Complex | TaskClass::Creative | TaskClass::Retry) => 1.0,
        (Some(ALIAS_FAST), TaskClass::Simple | TaskClass::Chat(ChatIntent::General)) => 1.0,
        _ => 0.0,
    };
    (1.0 + bonus, 1.0)
}

/// A scored candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    /// The pair and its facts.
    pub candidate: ModelFacts,
    /// Beta successes used (stored arm, or the prior of a fresh one).
    pub alpha: f64,
    /// Beta failures used.
    pub beta: f64,
    /// Observations behind the arm (0 for a fresh one).
    pub n: u64,
    /// The Thompson draw.
    pub draw: f64,
    /// Cost penalty term, in `[0, 1]`.
    pub cost_norm: f64,
    /// Latency penalty term, in `[0, 1]`.
    pub latency_norm: f64,
    /// `draw - cost_weight * cost_norm - latency_weight * latency_norm`.
    pub utility: f64,
    /// Mean known cost of the arm, when it has one.
    pub arm_mean_cost_usd: Option<f64>,
    /// Estimated cost of one turn, when the price is known.
    pub est_turn_cost_usd: Option<f64>,
    /// This pair was picked by the exploration draw, not by its utility.
    pub explored: bool,
}

/// The seed of a decision: a stable hash of its id.
pub fn seed_of(id: Uuid) -> u64 {
    let bits = id.as_u128();
    let mut state = (bits as u64) ^ ((bits >> 64) as u64).rotate_left(17);
    splitmix64(&mut state)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a: a hash whose value never changes between builds, so a persisted
/// decision replays on a later binary.
fn stable_hash(parts: &[&str]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.bytes().chain(std::iter::once(0u8)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        let bits = splitmix64(&mut self.0) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape, 1), Marsaglia-Tsang.
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

/// A draw from Beta(alpha, beta), deterministic for a seed.
pub fn beta_draw(alpha: f64, beta: f64, seed: u64) -> f64 {
    let (alpha, beta) = (alpha.max(1e-3), beta.max(1e-3));
    let mut rng = Rng::new(seed);
    let x = rng.gamma(alpha);
    let y = rng.gamma(beta);
    let total = x + y;
    if total > 0.0 {
        (x / total).clamp(0.0, 1.0)
    } else {
        alpha / (alpha + beta)
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

/// Each value divided by the maximum; unknown values take the median of the
/// known ones; everything is 0 when none is known (or the maximum is 0).
fn normalise(values: &[Option<f64>]) -> Vec<f64> {
    let known: Vec<f64> = values.iter().flatten().copied().collect();
    let Some(fill) = median(known.clone()) else {
        return vec![0.0; values.len()];
    };
    let filled: Vec<f64> = values.iter().map(|v| v.unwrap_or(fill)).collect();
    let max = filled.iter().copied().fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![0.0; values.len()];
    }
    filled.iter().map(|v| (v / max).clamp(0.0, 1.0)).collect()
}

/// Scores `candidates` for `signature`.
///
/// `arms` are the stored arms (any class: only those of the signature's class
/// count). The result is sorted best first; when the exploration draw fires,
/// the explored pair comes first and carries `explored = true`. Ties break on
/// the provider and model so the order never depends on input order.
pub fn score(
    signature: &TaskSignature,
    candidates: &[ModelFacts],
    arms: &[ArmStats],
    hints: &PriorHints,
    settings: &RoutingSettings,
    seed: u64,
) -> Vec<Scored> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let class = signature.arm_key();
    let stored = |facts: &ModelFacts| {
        arms.iter().find(|arm| {
            arm.key.class == class
                && arm.key.provider_id == facts.provider_id
                && arm.key.model == facts.model
        })
    };

    let costs: Vec<Option<f64>> = candidates
        .iter()
        .map(|facts| {
            stored(facts)
                .and_then(ArmStats::mean_cost_usd)
                .or_else(|| facts.estimated_turn_cost_usd(signature))
        })
        .collect();
    let latencies: Vec<Option<f64>> = candidates
        .iter()
        .map(|facts| stored(facts).and_then(ArmStats::mean_latency_ms))
        .collect();
    let cost_norm = normalise(&costs);
    let latency_norm = normalise(&latencies);

    let mut scored: Vec<Scored> = candidates
        .iter()
        .enumerate()
        .map(|(i, facts)| {
            let arm = stored(facts);
            let (alpha, beta, n) = match arm {
                Some(arm) => (arm.alpha, arm.beta, arm.n),
                None => {
                    let (a, b) = prior_for(
                        signature.class,
                        hints.alias_of(&facts.provider_id, &facts.model),
                    );
                    (a, b, 0)
                }
            };
            let arm_seed = seed ^ stable_hash(&[&class, &facts.provider_id, &facts.model]);
            let draw = beta_draw(alpha, beta, arm_seed);
            let utility = draw
                - settings.cost_weight * cost_norm[i]
                - settings.latency_weight * latency_norm[i];
            Scored {
                candidate: facts.clone(),
                alpha,
                beta,
                n,
                draw,
                cost_norm: cost_norm[i],
                latency_norm: latency_norm[i],
                utility,
                arm_mean_cost_usd: arm.and_then(ArmStats::mean_cost_usd),
                est_turn_cost_usd: facts.estimated_turn_cost_usd(signature),
                explored: false,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.utility
            .total_cmp(&a.utility)
            .then_with(|| a.candidate.provider_id.cmp(&b.candidate.provider_id))
            .then_with(|| a.candidate.model.cmp(&b.candidate.model))
    });

    // Exploration: with probability epsilon (never above the hard bound) take a
    // uniformly chosen pair instead of the best one.
    let epsilon = settings
        .exploration_epsilon
        .clamp(0.0, MAX_EXPLORATION_EPSILON);
    let mut rng = Rng::new(seed ^ 0xA5A5_A5A5_5A5A_5A5A);
    let fires = rng.uniform() < epsilon;
    let pick = (rng.uniform() * scored.len() as f64) as usize;
    if fires && scored.len() > 1 {
        let mut explored = scored.remove(pick.min(scored.len() - 1));
        explored.explored = true;
        scored.insert(0, explored);
    }
    scored
}

/// The key of the arm a scored pair belongs to.
pub fn arm_key_of(signature: &TaskSignature, candidate: &ModelFacts) -> ArmKey {
    ArmKey::new(
        signature.arm_key(),
        &candidate.provider_id,
        &candidate.model,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::signature::ContextHints;
    use nexus_claude::agent::{CostBasis, ModelPrice};

    fn facts(provider: &str, model: &str, input_price: Option<f64>) -> ModelFacts {
        ModelFacts {
            provider_id: provider.into(),
            model: model.into(),
            supports_tools: true,
            supports_images: false,
            context_window: Some(200_000),
            price: input_price.map(|p| ModelPrice {
                input_per_mtok: p,
                output_per_mtok: p,
                cache_read_per_mtok: None,
                cache_write_per_mtok: None,
            }),
            cost_basis: if input_price.is_some() {
                CostBasis::Priced
            } else {
                CostBasis::Subscription
            },
            healthy: Some(true),
            allowed_for_project: true,
            sandboxed: false,
        }
    }

    fn signature(class: TaskClass) -> TaskSignature {
        let mut s = TaskSignature::from_chat_request("hello", false, None, ContextHints::default());
        s.class = class;
        s
    }

    fn arm(sig: &TaskSignature, f: &ModelFacts, alpha: f64, beta: f64) -> ArmStats {
        let mut a = ArmStats::fresh(arm_key_of(sig, f), alpha, beta);
        a.n = (alpha + beta) as u64;
        a
    }

    fn settings(epsilon: f64) -> RoutingSettings {
        RoutingSettings {
            exploration_epsilon: epsilon,
            ..RoutingSettings::default()
        }
    }

    #[test]
    fn a_seed_gives_the_same_scores_and_other_seeds_differ() {
        let sig = signature(TaskClass::Simple);
        let pool = vec![facts("a", "m", Some(1.0)), facts("b", "m", Some(1.0))];
        let hints = PriorHints::new();
        let first = score(&sig, &pool, &[], &hints, &settings(0.1), 42);
        assert_eq!(first, score(&sig, &pool, &[], &hints, &settings(0.1), 42));
        let reversed: Vec<_> = pool.iter().rev().cloned().collect();
        assert_eq!(
            first,
            score(&sig, &reversed, &[], &hints, &settings(0.1), 42)
        );
        let differs = (0..20).any(|s| {
            score(&sig, &pool, &[], &hints, &settings(0.1), s)[0]
                .candidate
                .provider_id
                != first[0].candidate.provider_id
        });
        assert!(differs, "the draw depends on the seed");
        let id = Uuid::new_v4();
        assert_eq!(seed_of(id), seed_of(id));
    }

    #[test]
    fn utility_is_the_draw_minus_the_weighted_normalised_cost_and_latency() {
        let sig = signature(TaskClass::Simple);
        let cheap = facts("cheap", "m", Some(1.0));
        let dear = facts("dear", "m", Some(10.0));
        let mut slow_arm = arm(&sig, &dear, 500.0, 500.0);
        slow_arm.latency_sum_ms = 20_000;
        slow_arm.latency_n = 2;
        let mut fast_arm = arm(&sig, &cheap, 500.0, 500.0);
        fast_arm.latency_sum_ms = 2_000;
        fast_arm.latency_n = 2;
        let s = RoutingSettings {
            cost_weight: 0.5,
            latency_weight: 0.2,
            exploration_epsilon: 0.0,
            ..RoutingSettings::default()
        };
        let out = score(
            &sig,
            &[dear.clone(), cheap.clone()],
            &[slow_arm, fast_arm],
            &PriorHints::new(),
            &s,
            7,
        );
        let by = |p: &str| out.iter().find(|x| x.candidate.provider_id == p).unwrap();
        assert!((by("dear").cost_norm - 1.0).abs() < 1e-9);
        assert!((by("cheap").cost_norm - 0.1).abs() < 1e-9);
        assert!((by("dear").latency_norm - 1.0).abs() < 1e-9);
        assert!((by("cheap").latency_norm - 0.1).abs() < 1e-9);
        for x in &out {
            let expected = x.draw - 0.5 * x.cost_norm - 0.2 * x.latency_norm;
            assert!((x.utility - expected).abs() < 1e-12);
        }
        assert_eq!(out[0].candidate.provider_id, "cheap");
    }

    #[test]
    fn an_unknown_cost_takes_the_median_and_none_known_means_no_penalty() {
        let sig = signature(TaskClass::Simple);
        let pool = vec![
            facts("a", "m", Some(1.0)),
            facts("b", "m", Some(3.0)),
            facts("c", "m", Some(9.0)),
            facts("sub", "m", None),
        ];
        let out = score(&sig, &pool, &[], &PriorHints::new(), &settings(0.0), 1);
        let by = |p: &str| out.iter().find(|x| x.candidate.provider_id == p).unwrap();
        assert!(
            (by("sub").cost_norm - by("b").cost_norm).abs() < 1e-9,
            "median"
        );
        assert!((by("c").cost_norm - 1.0).abs() < 1e-9);

        let free = vec![facts("x", "m", None), facts("y", "m", None)];
        let out = score(&sig, &free, &[], &PriorHints::new(), &settings(0.0), 1);
        assert!(out
            .iter()
            .all(|x| x.cost_norm == 0.0 && x.latency_norm == 0.0));
        assert!(score(&sig, &[], &[], &PriorHints::new(), &settings(0.0), 1).is_empty());
    }

    #[test]
    fn a_stored_arm_cost_wins_over_the_estimate() {
        let sig = signature(TaskClass::Simple);
        let a = facts("a", "m", Some(1.0));
        let b = facts("b", "m", Some(1.0));
        let mut measured = arm(&sig, &a, 10.0, 10.0);
        measured.cost_sum_usd = 4.0;
        measured.cost_n = 2;
        let out = score(
            &sig,
            &[a, b],
            &[measured],
            &PriorHints::new(),
            &settings(0.0),
            1,
        );
        let a = out.iter().find(|x| x.candidate.provider_id == "a").unwrap();
        assert_eq!(a.arm_mean_cost_usd, Some(2.0));
        assert!(
            (a.cost_norm - 1.0).abs() < 1e-9,
            "2 USD measured vs a tiny estimate"
        );
    }

    #[test]
    fn the_alias_is_a_prior_on_a_fresh_arm_only_and_evidence_overrides_it() {
        assert_eq!(prior_for(TaskClass::Complex, Some("deep")), (2.0, 1.0));
        assert_eq!(prior_for(TaskClass::Creative, Some("deep")), (2.0, 1.0));
        assert_eq!(prior_for(TaskClass::Retry, Some("deep")), (2.0, 1.0));
        assert_eq!(prior_for(TaskClass::Simple, Some("deep")), (1.0, 1.0));
        assert_eq!(prior_for(TaskClass::Simple, Some("fast")), (2.0, 1.0));
        assert_eq!(
            prior_for(TaskClass::Chat(ChatIntent::General), Some("fast")),
            (2.0, 1.0)
        );
        assert_eq!(prior_for(TaskClass::Complex, Some("fast")), (1.0, 1.0));
        assert_eq!(prior_for(TaskClass::Complex, None), (1.0, 1.0));

        let sig = signature(TaskClass::Complex);
        let deep = facts("p", "big", None);
        let fast = facts("p", "small", None);
        let hints = PriorHints::new()
            .with("p", "big", "deep")
            .with("p", "small", "fast");
        let pool = [deep.clone(), fast.clone()];
        let out = score(&sig, &pool, &[], &hints, &settings(0.0), 3);
        let big = out.iter().find(|x| x.candidate.model == "big").unwrap();
        assert_eq!((big.alpha, big.beta, big.n), (2.0, 1.0, 0));

        // The prior moves the mean of the draw...
        let mean_draw = |model: &str| {
            (0..400)
                .map(|s| {
                    score(&sig, &pool, &[], &hints, &settings(0.0), s)
                        .into_iter()
                        .find(|x| x.candidate.model == model)
                        .unwrap()
                        .draw
                })
                .sum::<f64>()
                / 400.0
        };
        assert!(mean_draw("big") > mean_draw("small") + 0.08);

        // ...but a stored arm ignores it.
        let stored = arm(&sig, &deep, 1.0, 9.0);
        let out = score(&sig, &pool, &[stored], &hints, &settings(0.0), 3);
        let big = out.iter().find(|x| x.candidate.model == "big").unwrap();
        assert_eq!((big.alpha, big.beta), (1.0, 9.0));
    }

    #[test]
    fn arms_of_another_class_do_not_count() {
        let sig = signature(TaskClass::Simple);
        let f = facts("a", "m", None);
        let other = signature(TaskClass::Complex);
        let wrong = arm(&other, &f, 50.0, 1.0);
        let out = score(&sig, &[f], &[wrong], &PriorHints::new(), &settings(0.0), 1);
        assert_eq!((out[0].alpha, out[0].beta, out[0].n), (1.0, 1.0, 0));
    }

    #[test]
    fn exploration_never_fires_at_zero_and_is_bounded_by_the_hard_cap() {
        let sig = signature(TaskClass::Simple);
        let pool = vec![
            facts("a", "m", None),
            facts("b", "m", None),
            facts("c", "m", None),
        ];
        let rate = |epsilon: f64| {
            let fired = (0..3_000u64)
                .filter(|s| {
                    score(&sig, &pool, &[], &PriorHints::new(), &settings(epsilon), *s)
                        .iter()
                        .any(|x| x.explored)
                })
                .count();
            fired as f64 / 3_000.0
        };
        assert_eq!(rate(0.0), 0.0);
        let at_cap = rate(0.25);
        assert!((0.20..=0.30).contains(&at_cap), "{at_cap}");
        let above = rate(0.9);
        assert!(above <= 0.30, "epsilon is clamped to 0.25, got {above}");
        // A single candidate has nothing to explore.
        let one = score(
            &sig,
            &pool[..1],
            &[],
            &PriorHints::new(),
            &settings(0.25),
            1,
        );
        assert!(!one[0].explored);
        // The explored pair leads.
        let seed = (0..3_000u64)
            .find(|s| score(&sig, &pool, &[], &PriorHints::new(), &settings(0.25), *s)[0].explored)
            .unwrap();
        let out = score(&sig, &pool, &[], &PriorHints::new(), &settings(0.25), seed);
        assert_eq!(out.iter().filter(|x| x.explored).count(), 1);
    }

    #[test]
    fn the_beta_draw_stays_in_range_and_follows_its_mean() {
        for (a, b) in [(0.5, 0.5), (1.0, 1.0), (30.0, 3.0), (2.0, 40.0)] {
            let draws: Vec<f64> = (0..2_000).map(|s| beta_draw(a, b, s)).collect();
            assert!(draws.iter().all(|d| (0.0..=1.0).contains(d)));
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            assert!((mean - a / (a + b)).abs() < 0.03, "{a},{b}: {mean}");
        }
        assert_eq!(beta_draw(2.0, 3.0, 9), beta_draw(2.0, 3.0, 9));
    }
}
