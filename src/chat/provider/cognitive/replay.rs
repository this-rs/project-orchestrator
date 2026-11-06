//! Offline replay of the routing journal (PYSR-5): *on this history, would a Beta with
//! forgetting and/or a cost-aware reward have chosen better than the cumulative Beta?*
//!
//! **Pure**: no I/O, no clock, no randomness. It reads a slice of [`CognitiveDecision`]s and
//! returns a [`ReplayReport`]. Nothing in the hot path calls it, and it activates nothing
//! (`decay` stays `1.0` in production).
//!
//! # Method
//!
//! Decisions are replayed in the order of `at` (ties broken by `id`). Each variant keeps its
//! OWN posterior per arm, seeded from the first `arms_snapshot` that shows the arm (what the
//! router really knew then) and then folded, decision after decision, with the observation the
//! journal recorded for the arm that served the turn. A later snapshot never overwrites a
//! variant's state: that would erase the very forgetting being measured.
//!
//! At each decision a variant picks, among the arms of the snapshot that share the served
//! arm's class, the one with the highest posterior MEAN (ties: smallest provider/model).
//! Greedy on the mean, not Thompson: a replay must be deterministic, and the point is to compare
//! what the *posteriors* say.
//!
//! # Formulas
//!
//! * Update (the one of the store, [`beta_update`]): `alpha ← γ·alpha + r`,
//!   `beta ← γ·beta + (1 − r)`.
//! * **Cumulative**: `γ = 1`.
//! * **Forgetting**: `γ = decay_for_window(window) = 1 − 1/window`.
//! * **Cost in the reward**: `r' = clamp(r − w · min(cost_usd / ref_usd, 1), 0, 1)`, with
//!   `w = CostWeighting::weight` (default 0.2) and `ref_usd = CostWeighting::ref_usd`
//!   (default 0.05): a turn at or above `ref_usd` loses at most `w` of reward. A turn whose
//!   cost is unknown keeps `r` unchanged and is COUNTED (`cost_unknown`) — never priced at zero
//!   as if it were free.
//!
//! # Valuing a pick (a replay has no counterfactual)
//!
//! The journal only knows the outcome of the arm that served each turn. A variant's pick is
//! therefore valued by the **next observed outcome of that arm** at or after the decision
//! (direct method). It follows a changing arm (non-stationary), is exactly the logged outcome
//! when the variant agrees with the router, and is `unvalued` (counted, excluded) when the arm
//! is never served again. `regret` = best valued candidate − the pick's value, over decisions
//! where both exist.
//!
//! # No invented numbers
//!
//! A decision without `arms_snapshot`, without a closed outcome (`outcome.reward`), or without
//! a served arm is SKIPPED and counted in [`Skipped`]; it never contributes a zero.
//!
//! # What this does not prove
//!
//! A journal written under Thompson sampling is not uniform: arms the router stopped trying
//! have few observations, so their "next outcome" is old or missing. The report states the
//! sample size and the unvalued count; read the difference between variants against them.
use std::collections::HashMap;

use super::decision::CognitiveDecision;
use super::store::{beta_update, decay_for_window, ArmKey, ArmStats};

/// How cost enters the reward of the cost variant (and the cost-adjusted score of every variant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostWeighting {
    /// Largest reward a turn can lose to its cost, in `[0, 1]`.
    pub weight: f64,
    /// Cost (USD) at which the loss reaches `weight`.
    pub ref_usd: f64,
}

impl Default for CostWeighting {
    fn default() -> Self {
        Self {
            weight: 0.2,
            ref_usd: 0.05,
        }
    }
}

impl CostWeighting {
    /// `r' = clamp(r − w·min(cost/ref, 1), 0, 1)`; `r` unchanged when the cost is unknown.
    pub fn adjust(&self, reward: f64, cost_usd: Option<f64>) -> f64 {
        match cost_usd {
            Some(cost) if self.ref_usd > 0.0 => {
                (reward - self.weight * (cost.max(0.0) / self.ref_usd).min(1.0)).clamp(0.0, 1.0)
            }
            _ => reward.clamp(0.0, 1.0),
        }
    }
}

/// One variant to replay.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// Name shown in the report.
    pub name: String,
    /// Forgetting factor `γ` (`1.0` = cumulative).
    pub decay: f64,
    /// Whether the cost enters the reward the posterior learns from.
    pub cost_in_reward: bool,
}

/// The three variants of PYSR-5.
pub fn standard_variants(window: usize) -> Vec<Variant> {
    vec![
        Variant {
            name: "cumulative".into(),
            decay: 1.0,
            cost_in_reward: false,
        },
        Variant {
            name: format!("forgetting(window={window})"),
            decay: decay_for_window(window),
            cost_in_reward: false,
        },
        Variant {
            name: "cost_in_reward".into(),
            decay: 1.0,
            cost_in_reward: true,
        },
    ]
}

/// Decisions left out, by reason. A decision lands in the first reason that applies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Skipped {
    /// Written before `arms_snapshot` existed (or without arms).
    pub no_snapshot: usize,
    /// No outcome, or an outcome without a reward (still open).
    pub no_outcome: usize,
    /// No arm served the turn (not applied and no `used`).
    pub no_served_arm: usize,
}

impl Skipped {
    /// All of them.
    pub fn total(&self) -> usize {
        self.no_snapshot + self.no_outcome + self.no_served_arm
    }
}

/// What one variant did on the history.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantReport {
    /// The variant.
    pub name: String,
    /// Its pick at every replayed decision, in replay order.
    pub picks: Vec<ArmKey>,
    /// Decisions where the variant picked something other than the arm that served.
    pub moved: usize,
    /// Picks with a valued outcome (the sample the means below are over).
    pub valued: usize,
    /// Picks whose arm is never observed again: left out, not zero.
    pub unvalued: usize,
    /// Mean raw reward of the valued picks.
    pub mean_reward: Option<f64>,
    /// Mean cost-adjusted reward of the valued picks (same weighting for every variant).
    pub mean_adjusted_reward: Option<f64>,
    /// Mean USD of the valued picks whose cost is known.
    pub mean_cost_usd: Option<f64>,
    /// Valued picks whose cost was unknown (not priced at zero).
    pub cost_unknown: usize,
    /// Mean regret (raw reward) against the best valued candidate, `None` without a sample.
    pub mean_regret: Option<f64>,
    /// Mean |posterior mean − mean the router logged| for the arm served, a fidelity check:
    /// near 0 for the cumulative variant when the journal is complete.
    pub mean_gap_to_logged: Option<f64>,
}

/// The comparable result.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Decisions given.
    pub total: usize,
    /// Decisions replayed.
    pub replayed: usize,
    /// Decisions left out, and why.
    pub skipped: Skipped,
    /// One entry per variant, in the order asked.
    pub variants: Vec<VariantReport>,
}

impl ReplayReport {
    /// A readable table, for a test or a log.
    pub fn render(&self) -> String {
        let f = |v: Option<f64>| v.map_or("n/a".to_owned(), |x| format!("{x:.4}"));
        let mut out = format!(
            "replayed {}/{} (skipped: {} no_snapshot, {} no_outcome, {} no_served_arm)\n",
            self.replayed,
            self.total,
            self.skipped.no_snapshot,
            self.skipped.no_outcome,
            self.skipped.no_served_arm
        );
        for v in &self.variants {
            out.push_str(&format!(
                "{:<24} reward={} adjusted={} cost={} regret={} moved={} valued={} unvalued={} cost_unknown={} gap={}\n",
                v.name,
                f(v.mean_reward),
                f(v.mean_adjusted_reward),
                f(v.mean_cost_usd),
                f(v.mean_regret),
                v.moved,
                v.valued,
                v.unvalued,
                v.cost_unknown,
                f(v.mean_gap_to_logged),
            ));
        }
        out
    }
}

/// One observation of an arm: step index, raw reward, cost.
type Observation = (usize, f64, Option<f64>);
/// The next outcome of an arm: raw reward, cost.
type Outcome = (f64, Option<f64>);

/// One decision that can be replayed.
struct Step<'a> {
    snapshot: &'a [ArmStats],
    served: ArmKey,
    reward: f64,
    cost_usd: Option<f64>,
}

/// Replays `decisions` under each of `variants`. `cost` also defines the cost-adjusted score.
pub fn replay(
    decisions: &[CognitiveDecision],
    variants: &[Variant],
    cost: &CostWeighting,
) -> ReplayReport {
    let mut ordered: Vec<&CognitiveDecision> = decisions.iter().collect();
    ordered.sort_by_key(|d| (d.at, d.id));

    let mut skipped = Skipped::default();
    let mut steps: Vec<Step> = Vec::new();
    for d in ordered {
        if d.arms_snapshot.is_empty() {
            skipped.no_snapshot += 1;
            continue;
        }
        let Some((reward, cost_usd)) = d
            .outcome
            .as_ref()
            .and_then(|o| o.reward.map(|r| (r, o.cost_usd)))
        else {
            skipped.no_outcome += 1;
            continue;
        };
        let Some(served) = d.served_arm() else {
            skipped.no_served_arm += 1;
            continue;
        };
        steps.push(Step {
            snapshot: &d.arms_snapshot,
            served,
            reward,
            cost_usd,
        });
    }

    // Per arm: (step index, raw reward, cost) of every observation, in order.
    let mut observed: HashMap<&ArmKey, Vec<Observation>> = HashMap::new();
    for (i, s) in steps.iter().enumerate() {
        observed
            .entry(&s.served)
            .or_default()
            .push((i, s.reward, s.cost_usd));
    }
    let next_outcome = |key: &ArmKey, from: usize| -> Option<Outcome> {
        let list = observed.get(key)?;
        let at = list.partition_point(|(i, _, _)| *i < from);
        list.get(at).map(|(_, r, c)| (*r, *c))
    };

    let variants = variants
        .iter()
        .map(|variant| run_variant(variant, &steps, cost, &next_outcome))
        .collect();
    ReplayReport {
        total: decisions.len(),
        replayed: steps.len(),
        skipped,
        variants,
    }
}

fn run_variant(
    variant: &Variant,
    steps: &[Step],
    cost: &CostWeighting,
    next_outcome: &dyn Fn(&ArmKey, usize) -> Option<Outcome>,
) -> VariantReport {
    let mut state: HashMap<ArmKey, (f64, f64)> = HashMap::new();
    let mut picks = Vec::with_capacity(steps.len());
    let (mut moved, mut valued, mut unvalued, mut cost_unknown) = (0, 0, 0, 0);
    let (mut reward_sum, mut adjusted_sum, mut cost_sum, mut cost_n) = (0.0, 0.0, 0.0, 0usize);
    let (mut regret_sum, mut regret_n) = (0.0, 0usize);
    let (mut gap_sum, mut gap_n) = (0.0, 0usize);

    for (i, step) in steps.iter().enumerate() {
        // Seed arms never seen by this variant from what the router knew right now.
        for arm in step.snapshot {
            state
                .entry(arm.key.clone())
                .or_insert((arm.alpha, arm.beta));
        }
        let mean = |key: &ArmKey| state.get(key).map(|(a, b)| a / (a + b));
        let candidates: Vec<&ArmKey> = step
            .snapshot
            .iter()
            .map(|a| &a.key)
            .filter(|k| k.class == step.served.class)
            .collect();
        let pick = candidates
            .iter()
            .copied()
            .max_by(|a, b| {
                let (ma, mb) = (mean(a).unwrap_or(0.0), mean(b).unwrap_or(0.0));
                ma.total_cmp(&mb)
                    .then_with(|| (&b.provider_id, &b.model).cmp(&(&a.provider_id, &a.model)))
            })
            .unwrap_or(&step.served)
            .clone();

        if let (Some(m), Some(logged)) = (
            mean(&step.served),
            step.snapshot.iter().find(|a| a.key == step.served),
        ) {
            gap_sum += (m - logged.mean()).abs();
            gap_n += 1;
        }
        if pick != step.served {
            moved += 1;
        }
        match next_outcome(&pick, i) {
            Some((r, c)) => {
                valued += 1;
                reward_sum += r;
                adjusted_sum += cost.adjust(r, c);
                match c {
                    Some(c) => {
                        cost_sum += c;
                        cost_n += 1;
                    }
                    None => cost_unknown += 1,
                }
                let best = candidates
                    .iter()
                    .filter_map(|k| next_outcome(k, i).map(|(r, _)| r))
                    .fold(f64::MIN, f64::max);
                regret_sum += best - r;
                regret_n += 1;
            }
            None => unvalued += 1,
        }
        picks.push(pick);

        // Learn from what the journal recorded for the arm that served.
        let learned = if variant.cost_in_reward {
            cost.adjust(step.reward, step.cost_usd)
        } else {
            step.reward
        };
        let entry = state.entry(step.served.clone()).or_insert((1.0, 1.0));
        *entry = beta_update(entry.0, entry.1, variant.decay, learned);
    }

    let avg = |sum: f64, n: usize| (n > 0).then(|| sum / n as f64);
    VariantReport {
        name: variant.name.clone(),
        picks,
        moved,
        valued,
        unvalued,
        mean_reward: avg(reward_sum, valued),
        mean_adjusted_reward: avg(adjusted_sum, valued),
        mean_cost_usd: avg(cost_sum, cost_n),
        cost_unknown,
        mean_regret: avg(regret_sum, regret_n),
        mean_gap_to_logged: avg(gap_sum, gap_n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::candidates::ModelFacts;
    use crate::chat::provider::cognitive::decider::decide_with;
    use crate::chat::provider::cognitive::decision::{DecideRequest, DecisionOutcome, Pick};
    use crate::chat::provider::cognitive::mode::{
        LearningStage, ProviderRoutingMode, RoutingSettings,
    };
    use crate::chat::provider::cognitive::scorer::PriorHints;
    use crate::chat::provider::cognitive::signature::{ContextHints, TaskSignature};
    use crate::chat::provider::cognitive::store::ArmObservation;
    use chrono::{Duration, TimeZone, Utc};
    use nexus_claude::agent::{CostBasis, ModelPrice};
    use uuid::Uuid;

    fn facts(model: &str) -> ModelFacts {
        ModelFacts {
            provider_id: "local".into(),
            model: model.into(),
            supports_tools: true,
            supports_images: false,
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

    /// Deterministic noise in `[-0.05, 0.05]`.
    fn noise(i: u64) -> f64 {
        let x = i
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 33;
        (x % 1000) as f64 / 10_000.0 - 0.05
    }

    /// A journal the way production writes it: the live router draws from CUMULATIVE
    /// posteriors (`decide_with`), the turn runs on what it chose, the outcome is closed.
    /// Arm `a` is good for `drop_at` turns, then collapses; `b` stays mediocre and cheap.
    fn synthetic_journal(turns: u64, drop_at: u64) -> Vec<CognitiveDecision> {
        let sig =
            TaskSignature::from_chat_request("hello", false, Some("po"), ContextHints::default());
        let mut req = DecideRequest::new(
            sig.clone(),
            RoutingSettings {
                mode: ProviderRoutingMode::Full,
                stage: LearningStage::Auto,
                exploration_epsilon: 0.25,
                ..RoutingSettings::default()
            },
            vec![facts("a"), facts("b")],
        );
        req.pool = vec![facts("a"), facts("b")];
        let key = |m: &str| ArmKey::new(sig.arm_key(), "local", m);
        let mut arms = vec![
            ArmStats::fresh(key("a"), 1.0, 1.0),
            ArmStats::fresh(key("b"), 1.0, 1.0),
        ];
        let t0 = Utc.with_ymd_and_hms(2026, 11, 7, 0, 0, 0).unwrap();
        let mut journal = Vec::new();
        for i in 0..turns {
            let mut d = decide_with(
                Uuid::from_u128(i as u128 + 1),
                t0 + Duration::minutes(i as i64),
                &req,
                &arms,
                &PriorHints::new(),
            );
            let chosen = d.chosen.clone().expect("a pair is chosen");
            d.applied = true;
            d.used = Some(Pick::new(&chosen.provider_id, &chosen.model));
            let (base, cost) = match (chosen.model.as_str(), i < drop_at) {
                ("a", true) => (0.9, 0.04),
                ("a", false) => (0.2, 0.04),
                _ => (0.5, 0.005),
            };
            let reward = (base + noise(i)).clamp(0.0, 1.0);
            d.outcome = Some(DecisionOutcome {
                success: Some(reward > 0.5),
                reward: Some(reward),
                cost_usd: Some(cost),
                ..DecisionOutcome::default()
            });
            let served = d.served_arm().unwrap();
            let arm = arms.iter_mut().find(|a| a.key == served).unwrap();
            let (alpha, beta) = beta_update(arm.alpha, arm.beta, 1.0, reward);
            arm.alpha = alpha;
            arm.beta = beta;
            arm.n += 1;
            journal.push(d);
        }
        journal
    }

    fn by_name<'a>(r: &'a ReplayReport, prefix: &str) -> &'a VariantReport {
        r.variants
            .iter()
            .find(|v| v.name.starts_with(prefix))
            .unwrap()
    }

    /// Arm `a` collapses at turn 150. With forgetting the replay stops picking it sooner than
    /// the cumulative Beta, which still carries 150 good turns: it picks `a` fewer times after
    /// the drop, earns more, and regrets less.
    #[test]
    fn forgetting_demotes_an_arm_whose_quality_drops_sooner() {
        let journal = synthetic_journal(400, 150);
        let report = replay(&journal, &standard_variants(30), &CostWeighting::default());
        println!("{}", report.render());

        assert_eq!(report.replayed, 400);
        assert_eq!(report.skipped, Skipped::default());
        let cumulative = by_name(&report, "cumulative");
        let forgetting = by_name(&report, "forgetting");
        let a_after_drop =
            |v: &VariantReport| v.picks[150..].iter().filter(|k| k.model == "a").count();
        assert!(
            a_after_drop(forgetting) + 20 < a_after_drop(cumulative),
            "forgetting picks the collapsed arm far less often: {} vs {}",
            a_after_drop(forgetting),
            a_after_drop(cumulative)
        );
        assert!(forgetting.mean_reward > cumulative.mean_reward);
        assert!(forgetting.mean_regret < cumulative.mean_regret);
        // The cumulative replay follows the router's own posterior closely: the tool is faithful.
        assert!(cumulative.mean_gap_to_logged.unwrap() < 0.02);
    }

    /// The cost variant learns that `a` (0.04 USD) is dearer than `b` (0.005 USD): it spends less
    /// per pick than the cumulative variant while the weighting is the same for both.
    #[test]
    fn cost_in_the_reward_prefers_the_cheaper_arm_when_quality_is_close() {
        // No drop: a ≈ 0.9 − 0.2·0.8 = 0.74 adjusted, b ≈ 0.5: cost alone does not flip it, so
        // use a heavy weighting that makes a (0.04 → 0.8 of ref) lose 0.7 of its reward.
        let journal = synthetic_journal(300, 10_000);
        let heavy = CostWeighting {
            weight: 0.9,
            ref_usd: 0.05,
        };
        let report = replay(&journal, &standard_variants(30), &heavy);
        println!("{}", report.render());
        let cumulative = by_name(&report, "cumulative");
        let cost = by_name(&report, "cost_in_reward");
        assert!(
            cost.mean_cost_usd < cumulative.mean_cost_usd,
            "cost-aware picks are cheaper: {:?} vs {:?}",
            cost.mean_cost_usd,
            cumulative.mean_cost_usd
        );
        assert!(cost.mean_adjusted_reward > cumulative.mean_adjusted_reward);
    }

    /// Nothing is invented: decisions with no snapshot, no closed outcome, or no served arm are
    /// skipped and counted, and the replay is over the rest only.
    #[test]
    fn incomplete_decisions_are_skipped_and_counted_never_zero() {
        let mut journal = synthetic_journal(20, 10_000);
        journal[0].arms_snapshot.clear();
        journal[1].arms_snapshot.clear();
        journal[2].outcome = None;
        journal[3].outcome.as_mut().unwrap().reward = None;
        journal[4].applied = false;
        journal[4].used = None;
        let report = replay(&journal, &standard_variants(10), &CostWeighting::default());
        assert_eq!(report.total, 20);
        assert_eq!(
            report.skipped,
            Skipped {
                no_snapshot: 2,
                no_outcome: 2,
                no_served_arm: 1
            }
        );
        assert_eq!(report.replayed, 15);
        assert!(report.variants.iter().all(|v| v.picks.len() == 15));
        // Fully empty history: no number at all, rather than a zero.
        let empty = replay(&[], &standard_variants(10), &CostWeighting::default());
        assert!(empty.variants.iter().all(|v| v.mean_reward.is_none()));
    }

    /// The replay folds observations with the store's own update.
    #[tokio::test]
    async fn the_fold_is_the_stores_own() {
        use crate::chat::provider::cognitive::store::{InMemoryRoutingStore, RoutingArmStore};
        let store = InMemoryRoutingStore::new();
        let key = ArmKey::new("c", "p", "m");
        let (mut alpha, mut beta) = (1.0, 1.0);
        for r in [0.9, 0.1, 0.7] {
            let arm = store
                .observe(
                    &key,
                    &ArmObservation {
                        reward: r,
                        success_threshold: 0.5,
                        decay: 0.9,
                        cost_usd: None,
                        latency_ms: None,
                    },
                )
                .await
                .unwrap();
            (alpha, beta) = beta_update(alpha, beta, 0.9, r);
            assert_eq!((arm.alpha, arm.beta), (alpha, beta));
        }
    }

    #[test]
    fn an_unknown_cost_is_not_priced_at_zero() {
        let w = CostWeighting::default();
        assert_eq!(w.adjust(0.8, None), 0.8);
        assert!((w.adjust(0.8, Some(1.0)) - 0.6).abs() < 1e-12);
    }
}
