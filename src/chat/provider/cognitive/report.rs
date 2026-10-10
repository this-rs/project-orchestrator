//! The shadow report and the decision list behind `GET /api/chat/routing/report`
//! and `GET /api/chat/routing/decisions`.
//!
//! The report answers one question: what would automatic routing have changed?
//!
//! - `agreement_rate`: among decisions that have both the router's pick
//!   (`chosen`) and the pair actually used (`used`), the share where they are
//!   the same. `None` when no decision has both.
//! - `estimated_cost_delta_usd`: the sum, over decisions whose prices of both
//!   picks AND token usage are known, of `cost(pick) - cost(used)` at the real
//!   usage. A positive figure means the router's pick would have cost more.
//!   Two cost bases are never added: a decision whose two picks are not on the
//!   same dollar basis (`reported` or `priced`) is skipped and counted in
//!   `skipped_cost_decisions`. `None` when no decision qualifies, never `0`.
//! - `by_arm`: what each arm earned, from the closed decisions.
//!
//! The decision carries no price: the caller supplies a [`PriceLookup`] (the
//! nexus model prices, decision A1). [`build_report`] uses none, so its cost
//! figure is `None` until a lookup is wired through [`build_report_with`].

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use nexus_claude::agent::CostBasis;
use serde::{Deserialize, Serialize};

use super::candidates::RejectReason;
use super::decision::{CognitiveDecision, Pick};
use super::store::{DecisionFilter, RoutingArmStore};
use crate::evaluation::{self, ClassifierScore};

/// Decisions read per page, and in total, to build a report.
const PAGE: usize = 500;
const MAX_DECISIONS: usize = 10_000;

/// The price of one model, per million tokens, and how it is accounted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickPrice {
    /// USD per million input tokens.
    pub input_per_mtok: f64,
    /// USD per million output tokens.
    pub output_per_mtok: f64,
    /// How the price is accounted.
    pub basis: CostBasis,
}

/// Where the report finds the price of a pick.
pub trait PriceLookup: Send + Sync {
    /// The price of a pick, `None` when unknown.
    fn price(&self, pick: &Pick) -> Option<PickPrice>;
}

/// No price known: the cost delta stays `None`.
pub struct NoPrices;

impl PriceLookup for NoPrices {
    fn price(&self, _pick: &Pick) -> Option<PickPrice> {
        None
    }
}

/// One class of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportByClass {
    /// Class key.
    pub task_class: String,
    /// Decisions of the class.
    pub decisions: usize,
    /// Of which applied.
    pub applied: usize,
    /// Share of comparable decisions where the pick equals the one used.
    pub agreement_rate: Option<f64>,
    /// Estimated USD delta (pick minus used), `None` when nothing qualifies.
    pub estimated_cost_delta_usd: Option<f64>,
}

/// One arm of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportByArm {
    /// Class key.
    pub task_class: String,
    /// Instance.
    pub provider_id: String,
    /// Model.
    pub model: Option<String>,
    /// Closed decisions of the arm.
    pub n: usize,
    /// Mean reward, `None` without any.
    pub mean_reward: Option<f64>,
    /// Mean known marginal cost, `None` when never known.
    pub mean_cost_usd: Option<f64>,
}

/// The shadow report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingReport {
    /// Decisions in the window.
    pub decisions: usize,
    /// Of which applied.
    pub applied: usize,
    /// Share of comparable decisions where the pick equals the one used.
    pub agreement_rate: Option<f64>,
    /// Estimated USD delta (pick minus used), `None` when nothing qualifies.
    pub estimated_cost_delta_usd: Option<f64>,
    /// Decisions left out of the cost delta (unknown price or tokens, or two
    /// different cost bases) among those that have both a pick and a used pair.
    pub skipped_cost_decisions: usize,
    /// Per class.
    pub by_class: Vec<ReportByClass>,
    /// Per arm.
    pub by_arm: Vec<ReportByArm>,
    /// The offline classifier bench (`crate::evaluation`), one row per classifier.
    /// Additive: a report serialised before this field existed reads back as empty.
    #[serde(default)]
    pub classifiers: Vec<ClassifierScore>,
}

#[derive(Default)]
struct Tally {
    decisions: usize,
    applied: usize,
    comparable: usize,
    agreed: usize,
    delta: Option<f64>,
    skipped: usize,
}

impl Tally {
    fn agreement(&self) -> Option<f64> {
        (self.comparable > 0).then(|| self.agreed as f64 / self.comparable as f64)
    }

    fn add(&mut self, decision: &CognitiveDecision, prices: &dyn PriceLookup) {
        self.decisions += 1;
        if decision.applied {
            self.applied += 1;
        }
        let (Some(chosen), Some(used)) = (&decision.chosen, &decision.used) else {
            return;
        };
        self.comparable += 1;
        if chosen == used {
            self.agreed += 1;
        }
        match cost_delta(decision, chosen, used, prices) {
            Some(delta) => self.delta = Some(self.delta.unwrap_or(0.0) + delta),
            None => self.skipped += 1,
        }
    }
}

fn dollars(price: &PickPrice, input: u64, output: u64) -> f64 {
    (input as f64 * price.input_per_mtok + output as f64 * price.output_per_mtok) / 1_000_000.0
}

fn dollar_basis(basis: CostBasis) -> bool {
    matches!(basis, CostBasis::Priced | CostBasis::Reported)
}

/// `cost(chosen) - cost(used)` at the real usage; `None` when a price or the
/// tokens are unknown, or the two picks are not on the same dollar basis.
fn cost_delta(
    decision: &CognitiveDecision,
    chosen: &Pick,
    used: &Pick,
    prices: &dyn PriceLookup,
) -> Option<f64> {
    let outcome = decision.outcome.as_ref()?;
    let (input, output) = (outcome.input_tokens?, outcome.output_tokens?);
    let (p_chosen, p_used) = (prices.price(chosen)?, prices.price(used)?);
    if p_chosen.basis != p_used.basis || !dollar_basis(p_chosen.basis) {
        return None;
    }
    let delta = dollars(&p_chosen, input, output) - dollars(&p_used, input, output);
    delta.is_finite().then_some(delta)
}

/// The shadow report of a project over a window, without prices.
pub async fn build_report(
    store: &dyn RoutingArmStore,
    project_slug: Option<&str>,
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
) -> anyhow::Result<RoutingReport> {
    build_report_with(store, &NoPrices, project_slug, from, to).await
}

/// The shadow report of a project over a window.
pub async fn build_report_with(
    store: &dyn RoutingArmStore,
    prices: &dyn PriceLookup,
    project_slug: Option<&str>,
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
) -> anyhow::Result<RoutingReport> {
    let mut decisions = Vec::new();
    let mut offset = 0;
    loop {
        let page = store
            .decisions(&DecisionFilter {
                project_slug: project_slug.map(str::to_owned),
                since: from,
                session_id: None,
                limit: Some(PAGE),
                offset,
            })
            .await?;
        let got = page.len();
        decisions.extend(page.into_iter().filter(|d| to.is_none_or(|to| d.at <= to)));
        offset += got;
        if got < PAGE || offset >= MAX_DECISIONS {
            break;
        }
    }
    Ok(summarise(&decisions, prices))
}

#[derive(Default)]
struct ArmTally {
    n: usize,
    reward_sum: f64,
    cost_sum: f64,
    cost_n: usize,
}

/// The report of a set of decisions.
pub fn summarise(decisions: &[CognitiveDecision], prices: &dyn PriceLookup) -> RoutingReport {
    let mut total = Tally::default();
    let mut classes: BTreeMap<String, Tally> = BTreeMap::new();
    let mut arms: BTreeMap<(String, String, String), ArmTally> = BTreeMap::new();
    for decision in decisions {
        let class = decision.signature.arm_key();
        total.add(decision, prices);
        classes
            .entry(class.clone())
            .or_default()
            .add(decision, prices);
        let (Some(chosen), Some(outcome)) = (&decision.chosen, &decision.outcome) else {
            continue;
        };
        let Some(reward) = outcome.reward else {
            continue;
        };
        let arm = arms
            .entry((class, chosen.provider_id.clone(), chosen.model.clone()))
            .or_default();
        arm.n += 1;
        arm.reward_sum += reward;
        if let Some(cost) = outcome.cost_usd {
            arm.cost_sum += cost;
            arm.cost_n += 1;
        }
    }
    RoutingReport {
        decisions: total.decisions,
        applied: total.applied,
        agreement_rate: total.agreement(),
        estimated_cost_delta_usd: total.delta,
        skipped_cost_decisions: total.skipped,
        by_class: classes
            .into_iter()
            .map(|(task_class, t)| ReportByClass {
                task_class,
                decisions: t.decisions,
                applied: t.applied,
                agreement_rate: t.agreement(),
                estimated_cost_delta_usd: t.delta,
            })
            .collect(),
        by_arm: arms
            .into_iter()
            .map(|((task_class, provider_id, model), a)| ReportByArm {
                task_class,
                provider_id,
                model: Some(model),
                n: a.n,
                mean_reward: Some(a.reward_sum / a.n as f64),
                mean_cost_usd: (a.cost_n > 0).then(|| a.cost_sum / a.cost_n as f64),
            })
            .collect(),
        classifiers: classifier_bench(),
    }
}

/// The classifier bench rows. The bench is embedded and offline, so it does not
/// fail in practice; if it ever does, the report is served without the rows and
/// the failure is logged rather than turned into an error for the whole report.
fn classifier_bench() -> Vec<ClassifierScore> {
    evaluation::run_embedded().unwrap_or_else(|error| {
        tracing::warn!(%error, "classifier bench unavailable for the routing report");
        Vec::new()
    })
}

/// An alternative as the frontend types it (`RoutingAlternative`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AlternativeView {
    /// Instance.
    pub provider_id: String,
    /// Model.
    pub model: Option<String>,
    /// Score, `None` when filtered out before scoring.
    pub score: Option<f64>,
    /// Rejection code, `None` when merely outscored.
    pub rejected: Option<String>,
    /// Cause of the rejection, when its code has one: for `window_unknown`,
    /// `catalog_offline` or `not_in_catalog`. Omitted otherwise (additive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// An outcome as the frontend types it (`RoutingOutcome`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutcomeView {
    /// Whether the work succeeded.
    pub success: Option<bool>,
    /// Reward in `[0, 1]`.
    pub reward: Option<f64>,
    /// Marginal USD, `None` when unknown.
    pub cost_usd: Option<f64>,
}

/// A decision as the frontend types it (`RoutingDecision`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutingDecisionView {
    /// Identifier.
    pub id: String,
    /// RFC 3339.
    pub at: String,
    /// Mode in force.
    pub mode: super::mode::ProviderRoutingMode,
    /// Stage in force.
    pub stage: super::mode::LearningStage,
    /// Whether the choice was applied.
    pub applied: bool,
    /// Class key.
    pub task_class: String,
    /// Instance of the pick (empty when nothing was eligible).
    pub provider_id: String,
    /// Model of the pick.
    pub model: Option<String>,
    /// Utility of the pick.
    pub score: Option<f64>,
    /// Whether the draw explored.
    pub explored: bool,
    /// Readable reason.
    pub reason: String,
    /// Options considered.
    pub alternatives: Vec<AlternativeView>,
    /// Chat session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Plan run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Outcome, once closed.
    pub outcome: Option<OutcomeView>,
}

impl From<&CognitiveDecision> for RoutingDecisionView {
    fn from(d: &CognitiveDecision) -> Self {
        Self {
            id: d.id.to_string(),
            at: d.at.to_rfc3339(),
            mode: d.mode,
            stage: d.stage,
            applied: d.applied,
            task_class: d.signature.arm_key(),
            provider_id: d
                .chosen
                .as_ref()
                .map(|p| p.provider_id.clone())
                .unwrap_or_default(),
            model: d.chosen.as_ref().map(|p| p.model.clone()),
            score: d.score,
            explored: d.explored,
            reason: d.reason.clone(),
            alternatives: d
                .alternatives
                .iter()
                .map(|a| AlternativeView {
                    provider_id: a.pick.provider_id.clone(),
                    model: Some(a.pick.model.clone()),
                    score: a.score,
                    rejected: a.rejected.as_ref().map(|r| r.code().to_owned()),
                    why: a
                        .rejected
                        .as_ref()
                        .and_then(RejectReason::why)
                        .map(str::to_owned),
                })
                .collect(),
            session_id: d.session_id.map(|u| u.to_string()),
            task_id: d.task_id.map(|u| u.to_string()),
            run_id: d.run_id.map(|u| u.to_string()),
            outcome: d
                .outcome
                .as_ref()
                .filter(|o| o.reward.is_some())
                .map(|o| OutcomeView {
                    success: o.success,
                    reward: o.reward,
                    cost_usd: o.cost_usd,
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::DecisionOutcome;
    use crate::chat::provider::cognitive::mode::{LearningStage, ProviderRoutingMode};
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn decision(class: TaskClass, chosen: &str, used: Option<&str>) -> CognitiveDecision {
        CognitiveDecision {
            id: Uuid::new_v4(),
            at: Utc::now(),
            signature: TaskSignature {
                class,
                ..TaskSignature::utility(class, 9_000, Some("proj"))
            },
            chosen: Some(Pick::new("p", chosen)),
            score: Some(0.5),
            explored: false,
            reason: "test".into(),
            alternatives: vec![],
            applied: false,
            mode: ProviderRoutingMode::Mixed,
            stage: LearningStage::Shadow,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: used.map(|m| Pick::new("p", m)),
        }
    }

    fn closed(mut d: CognitiveDecision, reward: f64, cost: Option<f64>) -> CognitiveDecision {
        d.outcome = Some(DecisionOutcome {
            success: Some(true),
            reward: Some(reward),
            cost_usd: cost,
            input_tokens: Some(1_000_000),
            output_tokens: Some(1_000_000),
            ..DecisionOutcome::default()
        });
        d
    }

    struct Prices(HashMap<String, PickPrice>);
    impl PriceLookup for Prices {
        fn price(&self, pick: &Pick) -> Option<PickPrice> {
            self.0.get(&pick.model).copied()
        }
    }

    fn price(per_mtok: f64, basis: CostBasis) -> PickPrice {
        PickPrice {
            input_per_mtok: per_mtok,
            output_per_mtok: per_mtok,
            basis,
        }
    }

    #[test]
    fn an_empty_window_reports_none_not_zero() {
        let report = summarise(&[], &NoPrices);
        assert_eq!(report.decisions, 0);
        assert_eq!(report.agreement_rate, None);
        assert_eq!(report.estimated_cost_delta_usd, None);
        assert!(report.by_class.is_empty() && report.by_arm.is_empty());
    }

    #[test]
    fn agreement_counts_only_decisions_with_both_picks() {
        let ds = vec![
            decision(TaskClass::Simple, "a", Some("a")),
            decision(TaskClass::Simple, "a", Some("b")),
            decision(TaskClass::Simple, "a", Some("a")),
            decision(TaskClass::Simple, "a", None),
            decision(TaskClass::Complex, "a", Some("b")),
        ];
        let report = summarise(&ds, &NoPrices);
        assert_eq!(report.decisions, 5);
        assert!((report.agreement_rate.unwrap() - 0.5).abs() < 1e-9);
        let simple = report.by_class.iter().find(|c| c.task_class == "simple");
        let simple = simple.unwrap();
        assert_eq!(simple.decisions, 4);
        assert!((simple.agreement_rate.unwrap() - 2.0 / 3.0).abs() < 1e-9);
        let none = summarise(&[decision(TaskClass::Simple, "a", None)], &NoPrices);
        assert_eq!(none.agreement_rate, None);
    }

    #[test]
    fn the_cost_delta_sums_only_decisions_on_one_basis_and_counts_the_rest() {
        let mut table = HashMap::new();
        table.insert("cheap".to_owned(), price(1.0, CostBasis::Priced));
        table.insert("dear".to_owned(), price(5.0, CostBasis::Priced));
        table.insert("sub".to_owned(), price(0.0, CostBasis::Subscription));
        table.insert("rep".to_owned(), price(2.0, CostBasis::Reported));
        let prices = Prices(table);
        let ds = vec![
            // pick cheap, used dear: (1 - 5) * 2 Mtok = -8
            closed(
                decision(TaskClass::Simple, "cheap", Some("dear")),
                0.7,
                None,
            ),
            // pick dear, used cheap: +8
            closed(
                decision(TaskClass::Simple, "dear", Some("cheap")),
                0.7,
                None,
            ),
            // pick cheap, used cheap: 0
            closed(
                decision(TaskClass::Simple, "cheap", Some("cheap")),
                0.7,
                None,
            ),
            // mixed bases: skipped
            closed(decision(TaskClass::Simple, "cheap", Some("sub")), 0.7, None),
            closed(decision(TaskClass::Simple, "rep", Some("cheap")), 0.7, None),
            // unknown price: skipped
            closed(
                decision(TaskClass::Simple, "cheap", Some("mystery")),
                0.7,
                None,
            ),
            // unknown tokens: skipped
            decision(TaskClass::Simple, "cheap", Some("dear")),
            // not comparable: not counted as skipped
            closed(decision(TaskClass::Simple, "cheap", None), 0.7, None),
        ];
        let report = summarise(&ds, &prices);
        assert!((report.estimated_cost_delta_usd.unwrap() - 0.0).abs() < 1e-9);
        assert_eq!(report.skipped_cost_decisions, 4);

        let only_dear = vec![closed(
            decision(TaskClass::Simple, "dear", Some("cheap")),
            0.7,
            None,
        )];
        let report = summarise(&only_dear, &prices);
        assert!((report.estimated_cost_delta_usd.unwrap() - 8.0).abs() < 1e-9);

        let mixed_only = vec![closed(
            decision(TaskClass::Simple, "cheap", Some("sub")),
            0.7,
            None,
        )];
        let report = summarise(&mixed_only, &prices);
        assert_eq!(report.estimated_cost_delta_usd, None, "never 0");
        assert_eq!(report.skipped_cost_decisions, 1);

        // Free/subscription on both sides is no dollar basis either.
        let subs = Prices(HashMap::from([
            ("s1".to_owned(), price(0.0, CostBasis::Subscription)),
            ("s2".to_owned(), price(0.0, CostBasis::Subscription)),
        ]));
        let both_sub = vec![closed(
            decision(TaskClass::Simple, "s1", Some("s2")),
            0.7,
            None,
        )];
        assert_eq!(summarise(&both_sub, &subs).estimated_cost_delta_usd, None);
    }

    #[test]
    fn by_arm_averages_the_closed_decisions_and_an_unknown_cost_is_not_zero() {
        let ds = vec![
            closed(decision(TaskClass::Simple, "a", None), 0.8, Some(0.2)),
            closed(decision(TaskClass::Simple, "a", None), 0.4, None),
            decision(TaskClass::Simple, "a", None),
            closed(decision(TaskClass::Simple, "b", None), 0.6, None),
        ];
        let report = summarise(&ds, &NoPrices);
        let a = report
            .by_arm
            .iter()
            .find(|x| x.model.as_deref() == Some("a"))
            .unwrap();
        assert_eq!(a.n, 2);
        assert!((a.mean_reward.unwrap() - 0.6).abs() < 1e-9);
        assert!((a.mean_cost_usd.unwrap() - 0.2).abs() < 1e-9);
        let b = report
            .by_arm
            .iter()
            .find(|x| x.model.as_deref() == Some("b"))
            .unwrap();
        assert_eq!(b.mean_cost_usd, None);
    }

    #[tokio::test]
    async fn the_report_reads_the_store_by_project_and_window() {
        let store = InMemoryRoutingStore::new();
        let mut applied = decision(TaskClass::Simple, "a", Some("a"));
        applied.applied = true;
        store.put_decision(&applied).await.unwrap();
        let mut other = decision(TaskClass::Simple, "a", Some("a"));
        other.signature.project_slug = Some("other".into());
        store.put_decision(&other).await.unwrap();
        let mut old = decision(TaskClass::Simple, "a", Some("b"));
        old.at = Utc::now() - chrono::Duration::days(10);
        store.put_decision(&old).await.unwrap();

        let all = build_report(&store, Some("proj"), None, None)
            .await
            .unwrap();
        assert_eq!((all.decisions, all.applied), (2, 1));
        let recent = build_report(
            &store,
            Some("proj"),
            Some(Utc::now() - chrono::Duration::days(1)),
            Some(Utc::now() + chrono::Duration::days(1)),
        )
        .await
        .unwrap();
        assert_eq!(recent.decisions, 1);
        let before = build_report(
            &store,
            Some("proj"),
            None,
            Some(Utc::now() - chrono::Duration::days(5)),
        )
        .await
        .unwrap();
        assert_eq!(before.decisions, 1);
        assert_eq!(before.agreement_rate, Some(0.0));
    }

    #[test]
    fn the_report_serialises_to_the_frontend_shape() {
        let ds = vec![closed(
            decision(TaskClass::Simple, "a", Some("a")),
            0.7,
            None,
        )];
        let json = serde_json::to_value(summarise(&ds, &NoPrices)).unwrap();
        for key in [
            "decisions",
            "applied",
            "agreement_rate",
            "estimated_cost_delta_usd",
            "by_class",
            "by_arm",
        ] {
            assert!(json.get(key).is_some(), "{key}");
        }
        assert!(json["estimated_cost_delta_usd"].is_null());
        assert_eq!(json["by_class"][0]["task_class"], "simple");
        assert_eq!(json["by_arm"][0]["provider_id"], "p");
    }

    #[test]
    fn the_report_carries_the_five_classifier_rows_and_reads_back_without_them() {
        let report = summarise(&[], &NoPrices);
        let names: Vec<&str> = report
            .classifiers
            .iter()
            .map(|c| c.classifier.as_str())
            .collect();
        assert_eq!(
            names,
            ["tool_groups", "intent", "task_class", "skills", "triggers"]
        );

        let mut json = serde_json::to_value(&report).unwrap();
        assert!(json["classifiers"][0]["accuracy"].is_number());
        json.as_object_mut().unwrap().remove("classifiers");
        let older: RoutingReport = serde_json::from_value(json).unwrap();
        assert!(older.classifiers.is_empty());
    }

    #[test]
    fn a_decision_view_matches_the_frontend_type() {
        let d = closed(decision(TaskClass::Simple, "a", None), 0.7, Some(0.1));
        let json = serde_json::to_value(RoutingDecisionView::from(&d)).unwrap();
        assert_eq!(json["task_class"], "simple");
        assert_eq!(json["mode"], "mixed");
        assert_eq!(json["stage"], "shadow");
        assert_eq!(json["provider_id"], "p");
        assert_eq!(json["model"], "a");
        assert_eq!(json["outcome"]["reward"], 0.7);
        assert!(json.get("session_id").is_none());
        let mut open = d.clone();
        open.outcome = Some(DecisionOutcome {
            overridden: true,
            ..DecisionOutcome::default()
        });
        let json = serde_json::to_value(RoutingDecisionView::from(&open)).unwrap();
        assert!(json["outcome"].is_null(), "a flag is not an outcome yet");
    }
}
