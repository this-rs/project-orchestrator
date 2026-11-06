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
//! # Comparing variants: paired, then IPS (never a raw difference)
//!
//! Variants do not value the same decisions (a variant's pick may have no later outcome while
//! another's has). So the headline comparison is **paired**: [`VariantReport::paired`] is
//! computed on the decisions that EVERY variant of the report can value
//! ([`ReplayReport::paired_n`]). The full-sample numbers (`valued`, `mean_reward`, …) remain as
//! a secondary table: they are over different samples and must not be subtracted.
//!
//! # Bias: the journal comes from the CUMULATIVE policy
//!
//! The journal was written by the live router (cumulative Beta, ε-greedy Thompson). Replaying
//! any other variant on it is **off-policy**, with two distinct biases:
//!
//! 1. **Learning (information advantage).** A variant learns from the arm that SERVED. If the
//!    journal keeps serving an arm that has collapsed, a forgetting variant receives a steady
//!    stream of failures of that arm that, in production, it would have stopped sampling and so
//!    never seen. Its gain is then an **upper bound**, not an estimate. Two mitigations exist:
//!    * the variants suffixed `[on-policy]` ([`Variant::on_policy_learning`]) learn ONLY on the
//!      decisions where they would have picked the arm that served, which removes the
//!      advantage at the price of sample size (a variant that never agrees with the journal
//!      never learns anything and stays on its seed);
//!    * the posterior of every variant is seeded from the first `arms_snapshot` that shows an
//!      arm, i.e. from the cumulative router's knowledge: a residual, unmeasured leak.
//! 2. **Valuation (direct method).** "Next outcome of the picked arm" is a model, not a
//!    counterfactual.
//!
//! **IPS** ([`IpsEstimate`]) corrects (2) on the SAME population of decisions for every
//! variant, from the propensities logged for that purpose (`DecisionAlternative::propensity`,
//! `(1−ε)·p_top1 + ε/k`). The target policy of a variant is deterministic (greedy on its
//! posterior mean), so the estimate is the self-normalised one:
//! `V = Σ wᵢ·1[pickᵢ = servedᵢ]·rᵢ / Σ wᵢ·1[pickᵢ = servedᵢ]`, `wᵢ = 1 / max(μᵢ, IPS_PROPENSITY_FLOOR)`,
//! `μᵢ` = logged propensity of the served arm. The propensity is **clipped below at
//! [`IPS_PROPENSITY_FLOOR`] = 0.05** (weights ≤ 20): with ε small an explored arm has
//! `μ ≈ ε/k` (e.g. 0.005, weight 200) and one such decision would dominate the estimate; the
//! clip trades a downward-biased weight on rare picks for a bounded variance, and the number of
//! clipped decisions is reported (`clipped`). Decisions whose served arm is not the logged
//! `chosen` (user override, declarative choice kept) or that carry no propensity are NOT
//! usable and are counted (`unusable`), never given a made-up weight.
//!
//! IPS does **not** fix (1): it values the pick fairly, but the posterior that made the pick
//! was trained on data a different policy would not have collected. Also the logged propensity
//! is "before the hysteresis" of the live router, so it is approximate when the hysteresis
//! overrode the draw. A variant's gain is therefore an upper bound on its real gain, until it
//! is confirmed by an `[on-policy]` run and by IPS together. [`BIAS_NOTICE`] is printed with
//! every report; no output of this module states that a variant "beats" the cumulative one
//! without it.
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
    /// Learn ONLY from the decisions where this variant would have picked the arm that served
    /// (its own on-policy subset). Removes the information advantage of learning from arms the
    /// journal kept serving, at the cost of sample size. See the module doc.
    pub on_policy_learning: bool,
}

/// The variants of PYSR-5: the three of the plan, then their `[on-policy]` twins (the
/// cumulative one included: greedy-on-mean is not the router's Thompson, so the baseline must
/// learn under the same rule to be comparable).
pub fn standard_variants(window: usize) -> Vec<Variant> {
    let base = |name: String, decay: f64, cost_in_reward: bool| Variant {
        name,
        decay,
        cost_in_reward,
        on_policy_learning: false,
    };
    let three = vec![
        base("cumulative".into(), 1.0, false),
        base(
            format!("forgetting(window={window})"),
            decay_for_window(window),
            false,
        ),
        base("cost_in_reward".into(), 1.0, true),
    ];
    let twins: Vec<Variant> = three
        .iter()
        .map(|v| Variant {
            name: format!("{} [on-policy]", v.name),
            on_policy_learning: true,
            ..v.clone()
        })
        .collect();
    three.into_iter().chain(twins).collect()
}

/// Said with every report: the journal comes from the cumulative policy.
pub const BIAS_NOTICE: &str = "BIAS: the journal was written by the CUMULATIVE policy. A non-cumulative variant's gain is an UPPER BOUND (it learns from arms the journal kept serving, which it would have stopped sampling) unless the [on-policy] and IPS columns agree. Do not read any difference as \"X beats cumulative\".";

/// Floor of the logged propensity in the IPS weights: weights are at most `1/0.05 = 20`.
pub const IPS_PROPENSITY_FLOOR: f64 = 0.05;

/// Decisions left out, by reason. A decision lands in the first reason that applies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Skipped {
    /// Written before `arms_snapshot` existed (or without arms).
    pub no_snapshot: usize,
    /// No outcome, or an outcome without a reward (still open).
    pub no_outcome: usize,
    /// No arm served the turn (not applied and no `used`).
    pub no_served_arm: usize,
    /// The snapshot holds no arm of the served arm's class: there is nothing to choose
    /// among, and a regret against "the best of nothing" would be a made-up number.
    pub no_class_candidate: usize,
}

impl Skipped {
    /// All of them.
    pub fn total(&self) -> usize {
        self.no_snapshot + self.no_outcome + self.no_served_arm + self.no_class_candidate
    }
}

/// The value of one pick on one decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Valuation {
    /// Raw reward of the next outcome of the picked arm.
    pub reward: f64,
    /// Same, cost-adjusted.
    pub adjusted: f64,
    /// Its cost, when known.
    pub cost_usd: Option<f64>,
    /// Best valued candidate − `reward`; `None` when no candidate is valued.
    pub regret: Option<f64>,
}

/// Means over a set of valued picks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics {
    /// Picks the means are over.
    pub valued: usize,
    /// Mean raw reward.
    pub mean_reward: Option<f64>,
    /// Mean cost-adjusted reward (same weighting for every variant).
    pub mean_adjusted_reward: Option<f64>,
    /// Mean USD of the picks whose cost is known.
    pub mean_cost_usd: Option<f64>,
    /// Picks whose cost was unknown (not priced at zero).
    pub cost_unknown: usize,
    /// Mean regret against the best valued candidate.
    pub mean_regret: Option<f64>,
}

impl Metrics {
    /// Over the decisions where `keep(i)` and the pick is valued.
    fn over(valuations: &[Option<Valuation>], keep: impl Fn(usize) -> bool) -> Self {
        let (mut n, mut rs, mut ad, mut cs, mut cn, mut cu) = (0usize, 0.0, 0.0, 0.0, 0usize, 0);
        let (mut gs, mut gn) = (0.0, 0usize);
        for (i, v) in valuations.iter().enumerate() {
            let Some(v) = v.filter(|_| keep(i)) else {
                continue;
            };
            n += 1;
            rs += v.reward;
            ad += v.adjusted;
            match v.cost_usd {
                Some(c) => {
                    cs += c;
                    cn += 1;
                }
                None => cu += 1,
            }
            if let Some(g) = v.regret {
                gs += g;
                gn += 1;
            }
        }
        let avg = |sum: f64, n: usize| (n > 0).then(|| sum / n as f64);
        Self {
            valued: n,
            mean_reward: avg(rs, n),
            mean_adjusted_reward: avg(ad, n),
            mean_cost_usd: avg(cs, cn),
            cost_unknown: cu,
            mean_regret: avg(gs, gn),
        }
    }
}

/// Self-normalised, clipped inverse-propensity estimate of a variant's reward, on the decisions
/// whose logged propensity is usable (the same ones for every variant).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IpsEstimate {
    /// Decisions with a usable propensity (the population, identical across variants).
    pub usable: usize,
    /// Decisions without one (served ≠ logged chosen, or no propensity): counted, not weighted.
    pub unusable: usize,
    /// Usable decisions where the variant's pick is the served arm (the only informative ones).
    pub matched: usize,
    /// Matched decisions whose propensity was raised to [`IPS_PROPENSITY_FLOOR`].
    pub clipped: usize,
    /// `Σ w·1[match] / usable`: near 1 when the logged propensities support the variant's
    /// policy; far from 1 means the estimate rests on little overlap.
    pub coverage: Option<f64>,
    /// Effective sample size `(Σw)² / Σw²` of the matched decisions.
    pub ess: Option<f64>,
    /// Estimated mean raw reward.
    pub value: Option<f64>,
    /// Estimated mean cost-adjusted reward.
    pub value_adjusted: Option<f64>,
}

/// What one variant did on the history.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantReport {
    /// The variant.
    pub name: String,
    /// Its pick at every replayed decision, in replay order.
    pub picks: Vec<ArmKey>,
    /// The value of each pick (`None` = unvalued), aligned with `picks`.
    pub valuations: Vec<Option<Valuation>>,
    /// Decisions where the variant picked something other than the arm that served.
    pub moved: usize,
    /// Decisions on which the variant learned (all, unless `on_policy_learning`).
    pub learned_on: usize,
    /// Picks with a valued outcome (the sample the means below are over).
    pub valued: usize,
    /// Picks whose arm is never observed again: left out, not zero.
    pub unvalued: usize,
    /// **Secondary** (own sample, not comparable across variants): mean raw reward.
    pub mean_reward: Option<f64>,
    /// **Secondary**: mean cost-adjusted reward of the valued picks.
    pub mean_adjusted_reward: Option<f64>,
    /// **Secondary**: mean USD of the valued picks whose cost is known.
    pub mean_cost_usd: Option<f64>,
    /// Valued picks whose cost was unknown (not priced at zero).
    pub cost_unknown: usize,
    /// **Secondary**: mean regret (raw reward) against the best valued candidate.
    pub mean_regret: Option<f64>,
    /// **Headline (direct method)**: the same metrics over the decisions EVERY variant valued.
    /// Still an upper bound for a non-cumulative variant: see [`BIAS_NOTICE`].
    pub paired: Metrics,
    /// IPS-corrected value of the variant's policy.
    pub ips: IpsEstimate,
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
    /// Decisions valued by every variant: the sample of the `paired` column.
    pub paired_n: usize,
    /// Decisions left out, and why.
    pub skipped: Skipped,
    /// One entry per variant, in the order asked.
    pub variants: Vec<VariantReport>,
}

impl ReplayReport {
    /// A readable table, for a test or a log. Always opens with [`BIAS_NOTICE`].
    pub fn render(&self) -> String {
        let f = |v: Option<f64>| v.map_or("n/a".to_owned(), |x| format!("{x:.4}"));
        let mut out = format!(
            "{BIAS_NOTICE}\nreplayed {}/{} (skipped: {} no_snapshot, {} no_outcome, {} no_served_arm, {} no_class_candidate)\n",
            self.replayed,
            self.total,
            self.skipped.no_snapshot,
            self.skipped.no_outcome,
            self.skipped.no_served_arm,
            self.skipped.no_class_candidate
        );
        out.push_str(&format!(
            "-- PAIRED (direct method, same {} decisions for every variant; upper bound for non-cumulative) --\n",
            self.paired_n
        ));
        for v in &self.variants {
            out.push_str(&format!(
                "{:<36} reward={} adjusted={} cost={} regret={}\n",
                v.name,
                f(v.paired.mean_reward),
                f(v.paired.mean_adjusted_reward),
                f(v.paired.mean_cost_usd),
                f(v.paired.mean_regret),
            ));
        }
        out.push_str(&format!(
            "-- IPS (clipped at propensity {IPS_PROPENSITY_FLOOR}, self-normalised; same population for every variant) --\n"
        ));
        for v in &self.variants {
            out.push_str(&format!(
                "{:<36} value={} adjusted={} matched={}/{} clipped={} unusable={} coverage={} ess={}\n",
                v.name,
                f(v.ips.value),
                f(v.ips.value_adjusted),
                v.ips.matched,
                v.ips.usable,
                v.ips.clipped,
                v.ips.unusable,
                f(v.ips.coverage),
                f(v.ips.ess),
            ));
        }
        out.push_str("-- FULL SAMPLE (secondary: each variant on its OWN valued picks; do not subtract) --\n");
        for v in &self.variants {
            out.push_str(&format!(
                "{:<36} reward={} adjusted={} cost={} regret={} moved={} learned_on={} valued={} unvalued={} cost_unknown={} gap={}\n",
                v.name,
                f(v.mean_reward),
                f(v.mean_adjusted_reward),
                f(v.mean_cost_usd),
                f(v.mean_regret),
                v.moved,
                v.learned_on,
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
    /// Logged propensity of the served arm, only when the served arm IS the logged `chosen`.
    propensity: Option<f64>,
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
        if !d.arms_snapshot.iter().any(|a| a.key.class == served.class) {
            skipped.no_class_candidate += 1;
            continue;
        }
        let propensity = d
            .chosen
            .as_ref()
            .filter(|c| c.provider_id == served.provider_id && c.model == served.model)
            .and_then(|c| d.alternatives.iter().find(|a| &a.pick == c))
            .and_then(|a| a.propensity)
            .filter(|p| p.is_finite() && *p > 0.0);
        steps.push(Step {
            snapshot: &d.arms_snapshot,
            served,
            reward,
            cost_usd,
            propensity,
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

    let mut reports: Vec<VariantReport> = variants
        .iter()
        .map(|variant| run_variant(variant, &steps, cost, &next_outcome))
        .collect();

    // Paired set: decisions every variant valued. With no variant, nothing is paired.
    let paired: Vec<bool> = (0..steps.len())
        .map(|i| !reports.is_empty() && reports.iter().all(|r| r.valuations[i].is_some()))
        .collect();
    for r in &mut reports {
        r.paired = Metrics::over(&r.valuations, |i| paired[i]);
    }
    ReplayReport {
        total: decisions.len(),
        replayed: steps.len(),
        paired_n: paired.iter().filter(|p| **p).count(),
        skipped,
        variants: reports,
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
    let mut valuations: Vec<Option<Valuation>> = Vec::with_capacity(steps.len());
    let (mut moved, mut learned_on) = (0, 0);
    let (mut gap_sum, mut gap_n) = (0.0, 0usize);
    let mut ips = IpsEstimate::default();
    let (mut w_sum, mut w2_sum, mut wr_sum, mut wa_sum) = (0.0, 0.0, 0.0, 0.0);

    for (i, step) in steps.iter().enumerate() {
        // Seed arms never seen by this variant from what the router knew right now.
        for arm in step.snapshot {
            state
                .entry(arm.key.clone())
                .or_insert((arm.alpha, arm.beta));
        }
        let mean = |key: &ArmKey| state.get(key).map(|(a, b)| a / (a + b));
        // Never empty: a decision with no arm of the served class was skipped upstream.
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
        let matches = pick == step.served;
        if !matches {
            moved += 1;
        }
        valuations.push(next_outcome(&pick, i).map(|(r, c)| {
            let best = candidates
                .iter()
                .filter_map(|k| next_outcome(k, i).map(|(r, _)| r))
                .reduce(f64::max);
            Valuation {
                reward: r,
                adjusted: cost.adjust(r, c),
                cost_usd: c,
                regret: best.map(|b| b - r),
            }
        }));

        // IPS: a deterministic target policy contributes only where it agrees with the journal.
        match step.propensity {
            None => ips.unusable += 1,
            Some(mu) => {
                ips.usable += 1;
                if matches {
                    let clipped = mu < IPS_PROPENSITY_FLOOR;
                    let w = 1.0 / mu.max(IPS_PROPENSITY_FLOOR);
                    ips.matched += 1;
                    ips.clipped += usize::from(clipped);
                    w_sum += w;
                    w2_sum += w * w;
                    wr_sum += w * step.reward;
                    wa_sum += w * cost.adjust(step.reward, step.cost_usd);
                }
            }
        }
        picks.push(pick);

        // Learn from what the journal recorded for the arm that served — all of it, or only
        // where this variant agrees with the journal (its own on-policy subset).
        if matches || !variant.on_policy_learning {
            let learned = if variant.cost_in_reward {
                cost.adjust(step.reward, step.cost_usd)
            } else {
                step.reward
            };
            let entry = state.entry(step.served.clone()).or_insert((1.0, 1.0));
            *entry = beta_update(entry.0, entry.1, variant.decay, learned);
            learned_on += 1;
        }
    }

    if ips.matched > 0 {
        ips.value = Some(wr_sum / w_sum);
        ips.value_adjusted = Some(wa_sum / w_sum);
        ips.ess = Some(w_sum * w_sum / w2_sum);
        ips.coverage = Some(w_sum / ips.usable as f64);
    }
    let all = Metrics::over(&valuations, |_| true);
    VariantReport {
        name: variant.name.clone(),
        picks,
        moved,
        learned_on,
        valued: all.valued,
        unvalued: valuations.len() - all.valued,
        mean_reward: all.mean_reward,
        mean_adjusted_reward: all.mean_adjusted_reward,
        mean_cost_usd: all.mean_cost_usd,
        cost_unknown: all.cost_unknown,
        mean_regret: all.mean_regret,
        paired: Metrics::default(),
        ips,
        valuations,
        mean_gap_to_logged: (gap_n > 0).then(|| gap_sum / gap_n as f64),
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
                no_served_arm: 1,
                ..Skipped::default()
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

    /// A hand-built journal: every decision serves `a` (class `c`), snapshots `a`=(3,1) and
    /// `b`=(1,1)... overridable by the caller; `rewards[i]` / `propensities[i]` per decision.
    fn hand_journal(
        seeds: [(f64, f64); 2],
        rewards: &[f64],
        propensities: &[Option<f64>],
    ) -> Vec<CognitiveDecision> {
        let mut base = synthetic_journal(rewards.len() as u64, 10_000);
        for (i, d) in base.iter_mut().enumerate() {
            d.signature = TaskSignature::from_chat_request(
                "hello",
                false,
                Some("po"),
                ContextHints::default(),
            );
            let class = d.signature.arm_key();
            let k = |m: &str| ArmKey::new(class.clone(), "local", m);
            d.arms_snapshot = vec![
                ArmStats::fresh(k("a"), seeds[0].0, seeds[0].1),
                ArmStats::fresh(k("b"), seeds[1].0, seeds[1].1),
            ];
            d.chosen = Some(Pick::new("local", "a"));
            d.used = Some(Pick::new("local", "a"));
            d.applied = true;
            d.alternatives = vec![
                crate::chat::provider::cognitive::decision::DecisionAlternative {
                    pick: Pick::new("local", "a"),
                    score: Some(0.5),
                    rejected: None,
                    p_top1: Some(0.5),
                    propensity: propensities[i],
                },
            ];
            d.outcome = Some(DecisionOutcome {
                success: Some(true),
                reward: Some(rewards[i]),
                cost_usd: Some(0.0),
                ..DecisionOutcome::default()
            });
        }
        base
    }

    fn cumulative_only() -> Vec<Variant> {
        standard_variants(30).into_iter().take(1).collect()
    }

    /// C3: a snapshot with no arm of the served class is skipped and counted; the regret never
    /// meets `f64::MIN`.
    #[test]
    fn a_snapshot_without_a_candidate_of_the_class_is_skipped_not_a_giant_regret() {
        let mut journal = synthetic_journal(30, 10_000);
        for arm in &mut journal[5].arms_snapshot {
            arm.key.class = "another.class".into();
        }
        let report = replay(&journal, &standard_variants(10), &CostWeighting::default());
        assert_eq!(report.skipped.no_class_candidate, 1);
        assert_eq!(report.replayed, 29);
        for v in &report.variants {
            let regret = v.mean_regret.expect("a sample");
            assert!(
                (0.0..=1.0).contains(&regret),
                "{}: regret {regret} is a made-up number",
                v.name
            );
            assert!(v
                .valuations
                .iter()
                .flatten()
                .all(|x| x.regret.is_none_or(|g| g.abs() <= 1.0)));
        }
    }

    /// C2: the headline is over the decisions every variant valued. Variant `amnesiac`
    /// (γ=0) forgets `b`'s failure at once and moves to `a`, whose outcomes exist; the
    /// cumulative one keeps picking `b`, which is never served again: it cannot value those
    /// picks. The shared set excludes them, so `amnesiac`'s zero-reward pick there does not
    /// reach the paired number — unlike its own full-sample mean.
    #[test]
    fn the_paired_numbers_are_over_the_decisions_every_variant_valued() {
        // d0 served b (r=0); d1 served a (r=0); d2.. served a (r=1); b seeded higher.
        let mut journal = hand_journal(
            [(2.0, 2.0), (3.0, 1.0)],
            &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            &[Some(1.0); 6],
        );
        journal[0].chosen = Some(Pick::new("local", "b"));
        journal[0].used = Some(Pick::new("local", "b"));
        let variants = vec![
            standard_variants(30)[0].clone(),
            Variant {
                name: "amnesiac".into(),
                decay: 0.0,
                cost_in_reward: false,
                on_policy_learning: false,
            },
        ];
        let report = replay(&journal, &variants, &CostWeighting::default());
        println!("{}", report.render());
        let (cum, amn) = (&report.variants[0], &report.variants[1]);
        assert!(
            amn.valued > cum.valued,
            "precondition: the full samples differ ({} vs {})",
            amn.valued,
            cum.valued
        );
        let shared: Vec<usize> = (0..6)
            .filter(|&i| report.variants.iter().all(|w| w.valuations[i].is_some()))
            .collect();
        assert_eq!(report.paired_n, shared.len());
        assert!(report.paired_n < amn.valued);
        for v in &report.variants {
            let by_hand = shared
                .iter()
                .map(|&i| v.valuations[i].unwrap().reward)
                .sum::<f64>()
                / shared.len() as f64;
            assert!(
                (v.paired.mean_reward.unwrap() - by_hand).abs() < 1e-12,
                "{}",
                v.name
            );
            assert_eq!(v.paired.valued, report.paired_n);
        }
        // The decision the cumulative variant cannot value does not move amnesiac's paired mean.
        assert_ne!(amn.paired.mean_reward, amn.mean_reward);
    }

    /// IPS weighs by the inverse of the logged propensity: reward 1 at p=0.5 twice, reward 0 at
    /// p=1 three times → (2+2)/(2+2+1+1+1) = 4/7, not the unweighted 2/5.
    #[test]
    fn ips_weights_by_the_inverse_logged_propensity() {
        let p = [Some(0.5), Some(0.5), Some(1.0), Some(1.0), Some(1.0)];
        let journal = hand_journal([(9.0, 1.0), (1.0, 9.0)], &[1.0, 1.0, 0.0, 0.0, 0.0], &p);
        let report = replay(&journal, &cumulative_only(), &CostWeighting::default());
        let ips = &report.variants[0].ips;
        assert_eq!((ips.usable, ips.matched, ips.clipped), (5, 5, 0));
        assert!((ips.value.unwrap() - 4.0 / 7.0).abs() < 1e-12, "{ips:?}");
        assert!((ips.coverage.unwrap() - 7.0 / 5.0).abs() < 1e-12);
    }

    /// The clip: p=0.01 is raised to 0.05 (weight 20, not 100) and counted.
    #[test]
    fn ips_clips_the_propensity_and_counts_it() {
        let p = [Some(0.01), Some(1.0), Some(1.0), Some(1.0), Some(1.0)];
        let journal = hand_journal([(9.0, 1.0), (1.0, 9.0)], &[1.0, 0.0, 0.0, 0.0, 0.0], &p);
        let report = replay(&journal, &cumulative_only(), &CostWeighting::default());
        let ips = &report.variants[0].ips;
        assert_eq!(ips.clipped, 1);
        assert!((ips.value.unwrap() - 20.0 / 24.0).abs() < 1e-12, "{ips:?}");
    }

    /// A decision with no usable propensity is counted, never weighted as 1.
    #[test]
    fn ips_counts_decisions_without_a_propensity() {
        let p = [None, Some(1.0), Some(1.0)];
        let mut journal = hand_journal([(9.0, 1.0), (1.0, 9.0)], &[1.0, 0.0, 0.0], &p);
        // The served arm is not the logged choice: the propensity belongs to another pick.
        journal[2].chosen = Some(Pick::new("local", "b"));
        let report = replay(&journal, &cumulative_only(), &CostWeighting::default());
        let ips = &report.variants[0].ips;
        assert_eq!((ips.usable, ips.unusable), (1, 2));
        assert_eq!(ips.value, Some(0.0));
    }

    /// C1: a variant that learns on-policy ignores the failures/successes of an arm it would not
    /// have picked. Off-policy learning flips to `a` once `a`'s journal successes pile up;
    /// on-policy never learns them (it keeps picking `b`, which is not what served).
    #[test]
    fn on_policy_learning_does_not_learn_from_an_arm_it_would_not_have_picked() {
        let journal = hand_journal([(2.0, 2.0), (3.2, 1.0)], &[1.0; 8], &[Some(1.0); 8]);
        let variants = standard_variants(30);
        let report = replay(&journal, &variants, &CostWeighting::default());
        let off = by_name(&report, "cumulative");
        let on = report
            .variants
            .iter()
            .find(|v| v.name == "cumulative [on-policy]")
            .unwrap();
        assert!(off.picks[..5].iter().all(|k| k.model == "b"));
        assert!(off.picks[5..].iter().all(|k| k.model == "a"));
        assert!(on.picks.iter().all(|k| k.model == "b"), "{:?}", on.picks);
        assert_eq!(on.learned_on, 0);
        assert_eq!(off.learned_on, 8);
    }

    /// The report never states a gain without the bias.
    #[test]
    fn the_rendered_report_always_carries_the_bias_notice() {
        let report = replay(
            &synthetic_journal(20, 10_000),
            &standard_variants(10),
            &CostWeighting::default(),
        );
        let text = report.render();
        assert!(text.starts_with("BIAS: the journal was written by the CUMULATIVE policy"));
        assert!(text.contains("UPPER BOUND"));
        assert!(text.contains("-- PAIRED") && text.contains("-- IPS"));
        let empty = replay(&[], &standard_variants(10), &CostWeighting::default());
        assert!(empty.render().contains("UPPER BOUND"));
    }

    #[test]
    fn an_unknown_cost_is_not_priced_at_zero() {
        let w = CostWeighting::default();
        assert_eq!(w.adjust(0.8, None), 0.8);
        assert!((w.adjust(0.8, Some(1.0)) - 0.6).abs() < 1e-12);
    }
}
