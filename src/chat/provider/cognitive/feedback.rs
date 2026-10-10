//! Closing the loop: what happened after a decision, turned into a reward for
//! the arm that was chosen.
//!
//! The seam is [`close_decision`]: every place that finishes work a decision
//! routed (the runner when an attempt closes, a chat session when it closes, a
//! delegation) calls it with an [`Outcome`].
//!
//! # Reward
//!
//! [`reward`] maps an [`Outcome`] to `[0, 1]`. The weights are constants of
//! this module, not settings: a reward that moved with the user's knobs would
//! make the arms of different projects incomparable.
//!
//! | signal                                   | weight                         |
//! |------------------------------------------|--------------------------------|
//! | success / failure / unknown              | `+0.50` / `0.00` / `+0.25`     |
//! | solved on the first attempt              | `+0.20` (success only)         |
//! | each attempt beyond the first            | `-0.10`                        |
//! | verification passed / failed             | `+0.15` / `-0.15`              |
//! | marginal cost <= class median            | `+0.15`                        |
//! | median < cost < 3x median                | degressive from `+0.15` to `0` |
//! | cost >= 3x median                        | `0`                            |
//! | cost or median unknown                   | `0` (never "free")             |
//! | duration (known or not)                  | `0` (carried to the arm only)  |
//! | the user switched the model by hand      | `-0.30`                        |
//! | interrupted                              | `-0.20`                        |
//!
//! The sum is clamped to `[0, 1]`. More attempts never raise the reward.
//!
//! # Trajectory
//!
//! [`close_decision_with`] also hands the closed decision to a
//! [`TrajectorySink`]; [`CollectorSink`] feeds the neural-routing collector
//! with a `routing.select_model` decision record.

use uuid::Uuid;

use super::decision::{CognitiveDecision, DecisionOutcome};
use super::mode::RoutingSettings;
use super::store::{ArmObservation, RoutingArmStore};

/// Reward above which an observation counts as a success for the Beta.
pub const SUCCESS_THRESHOLD: f64 = 0.5;

const BASE_SUCCESS: f64 = 0.5;
const BASE_UNKNOWN: f64 = 0.25;
const FIRST_TRY_BONUS: f64 = 0.2;
const EXTRA_ATTEMPT_PENALTY: f64 = 0.1;
const VERIFICATION_WEIGHT: f64 = 0.15;
const CHEAP_BONUS: f64 = 0.15;
/// Cost, as a multiple of the class median, at which the cost bonus is gone.
const COST_BONUS_ENDS_AT: f64 = 3.0;
const OVERRIDE_PENALTY: f64 = 0.3;
const INTERRUPTION_PENALTY: f64 = 0.2;

/// What is observable when the work a decision routed ends.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    /// Whether the work succeeded; `None` when it cannot be told.
    pub success: Option<bool>,
    /// Attempts used, starting at 1 (0 reads as 1).
    pub attempts: u32,
    /// Marginal USD spent; `None` when unknown (never zero).
    pub cost_usd: Option<f64>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// The work was interrupted before it ended.
    pub interrupted: bool,
    /// The user switched the model by hand after the decision.
    pub user_overrode_model: bool,
    /// The task's own verification passed, when one ran.
    pub verification_passed: Option<bool>,
    /// Input tokens consumed; `None` when unknown.
    pub input_tokens: Option<u64>,
    /// Output tokens produced; `None` when unknown.
    pub output_tokens: Option<u64>,
}

/// What the reward needs to know beyond the outcome itself.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RewardContext {
    /// Median marginal cost of the class, `None` when not known yet.
    pub class_median_cost: Option<f64>,
}

/// Reward in `[0, 1]` for an outcome, without a cost reference.
pub fn reward(outcome: &Outcome, settings: &RoutingSettings) -> f64 {
    reward_with(outcome, settings, &RewardContext::default())
}

/// Reward in `[0, 1]` for an outcome, with the class median cost when known.
pub fn reward_with(outcome: &Outcome, _settings: &RoutingSettings, ctx: &RewardContext) -> f64 {
    let attempts = outcome.attempts.max(1);
    let mut r = match outcome.success {
        Some(true) => BASE_SUCCESS,
        Some(false) => 0.0,
        None => BASE_UNKNOWN,
    };
    if outcome.success == Some(true) && attempts == 1 {
        r += FIRST_TRY_BONUS;
    }
    r -= EXTRA_ATTEMPT_PENALTY * f64::from(attempts - 1);
    r += match outcome.verification_passed {
        Some(true) => VERIFICATION_WEIGHT,
        Some(false) => -VERIFICATION_WEIGHT,
        None => 0.0,
    };
    r += cost_bonus(outcome.cost_usd, ctx.class_median_cost);
    if outcome.user_overrode_model {
        r -= OVERRIDE_PENALTY;
    }
    if outcome.interrupted {
        r -= INTERRUPTION_PENALTY;
    }
    if r.is_finite() {
        r.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn cost_bonus(cost: Option<f64>, median: Option<f64>) -> f64 {
    let (Some(cost), Some(median)) = (cost, median) else {
        return 0.0;
    };
    if !cost.is_finite() || !median.is_finite() || cost < 0.0 || median <= 0.0 {
        return 0.0;
    }
    let ratio = cost / median;
    if ratio <= 1.0 {
        CHEAP_BONUS
    } else if ratio >= COST_BONUS_ENDS_AT {
        0.0
    } else {
        CHEAP_BONUS * (COST_BONUS_ENDS_AT - ratio) / (COST_BONUS_ENDS_AT - 1.0)
    }
}

/// Where a closed decision is also sent (the neural-routing trajectory
/// channel). Fire and forget: it must never block nor fail the close.
pub trait TrajectorySink: Send + Sync {
    /// A decision was closed with this reward.
    fn record(&self, decision: &CognitiveDecision, reward: f64);
}

/// Confidence of a decision: `1 - H(p) / ln(k)` over the normalised scores of
/// its scored alternatives. 1 when one option dominates, 0 when they are
/// indistinguishable, and 0 when fewer than two were scored.
pub fn decision_confidence(decision: &CognitiveDecision) -> f64 {
    let scores: Vec<f64> = decision
        .alternatives
        .iter()
        .filter_map(|a| a.score)
        .filter(|s| s.is_finite())
        .map(|s| s.max(0.0))
        .collect();
    let k = scores.len();
    if k < 2 {
        return 0.0;
    }
    let sum: f64 = scores.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy: f64 = scores
        .iter()
        .map(|s| s / sum)
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - entropy / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Sends closed decisions to the neural-routing `TrajectoryCollector`.
///
/// The collector keys trajectories by a session id string: the decision's chat
/// session, else its plan run, else its task, else its own id. The collector
/// drops the event when its channel is full or it is disabled, so this never
/// blocks.
pub struct CollectorSink {
    collector: std::sync::Arc<neural_routing_runtime::TrajectoryCollector>,
}

impl CollectorSink {
    /// A sink over a collector.
    pub fn new(collector: std::sync::Arc<neural_routing_runtime::TrajectoryCollector>) -> Self {
        Self { collector }
    }
}

/// The decision record of a closed routing decision.
pub fn trajectory_record(
    decision: &CognitiveDecision,
    reward: f64,
) -> neural_routing_runtime::DecisionRecord {
    let session = decision
        .session_id
        .or(decision.run_id)
        .or(decision.task_id)
        .unwrap_or(decision.id);
    let (provider, model) = decision
        .chosen
        .as_ref()
        .map(|p| (p.provider_id.as_str(), p.model.as_str()))
        .unwrap_or(("", ""));
    neural_routing_runtime::DecisionRecord {
        session_id: session.to_string(),
        context_embedding: vec![],
        action_type: "routing.select_model".to_string(),
        action_params: serde_json::json!({
            "provider": provider,
            "model": model,
            "arm": decision.arm().map(|a| a.scheduler_key()),
            "score": decision.score,
            "explored": decision.explored,
            "applied": decision.applied,
            "reward": reward,
        }),
        alternatives_count: decision.alternatives.len(),
        chosen_index: 0,
        confidence: decision_confidence(decision),
        tool_usages: vec![],
        touched_entities: vec![],
        timestamp_ms: 0,
        query_embedding: vec![],
        node_features: vec![],
        protocol_run_id: decision.run_id,
        protocol_state: None,
        outcome: None,
    }
}

impl TrajectorySink for CollectorSink {
    fn record(&self, decision: &CognitiveDecision, reward: f64) {
        self.collector
            .record_decision(trajectory_record(decision, reward));
    }
}

/// Records the outcome of a decision: fills the decision's outcome and updates
/// the arm it chose. See [`close_decision_with`].
pub async fn close_decision(
    store: &dyn RoutingArmStore,
    settings: &RoutingSettings,
    decision_id: Uuid,
    outcome: &Outcome,
) -> anyhow::Result<()> {
    close_decision_with(store, settings, None, decision_id, outcome).await
}

/// Records the outcome of a decision: fills the decision's outcome, updates the
/// arm it chose and, when a sink is given, sends the trajectory. A decision
/// that chose nothing, or is unknown, only gets its outcome written (when it
/// exists). Idempotent per decision: an outcome already recorded is not
/// counted twice. An override flagged earlier by [`mark_override`] counts even
/// when `outcome.user_overrode_model` is not set.
pub async fn close_decision_with(
    store: &dyn RoutingArmStore,
    settings: &RoutingSettings,
    sink: Option<&dyn TrajectorySink>,
    decision_id: Uuid,
    outcome: &Outcome,
) -> anyhow::Result<()> {
    let Some(decision) = store.decision(decision_id).await? else {
        return Ok(());
    };
    if is_closed(&decision) {
        return Ok(());
    }
    let flagged = decision.outcome.as_ref().is_some_and(|o| o.overridden);
    let overridden = outcome.user_overrode_model || flagged;
    let effective = Outcome {
        user_overrode_model: overridden,
        ..outcome.clone()
    };
    let reward = reward(&effective, settings);
    store
        .set_outcome(
            decision_id,
            DecisionOutcome {
                success: outcome.success,
                reward: Some(reward),
                cost_usd: outcome.cost_usd,
                duration_ms: outcome.duration_ms,
                input_tokens: outcome.input_tokens,
                output_tokens: outcome.output_tokens,
                overridden,
            },
        )
        .await?;
    if let Some(arm) = decision.arm() {
        store
            .observe(
                &arm,
                &ArmObservation {
                    reward,
                    success_threshold: SUCCESS_THRESHOLD,
                    cost_usd: outcome.cost_usd,
                    latency_ms: outcome.duration_ms,
                },
            )
            .await?;
    }
    if let Some(sink) = sink {
        sink.record(&decision, reward);
    }
    Ok(())
}

fn is_closed(decision: &CognitiveDecision) -> bool {
    decision
        .outcome
        .as_ref()
        .is_some_and(|o| o.reward.is_some())
}

/// Flags that the user switched the model by hand after this decision, so the
/// later [`close_decision`] counts the override. Returns `false` when the
/// decision is unknown or already closed (too late to count).
///
/// The call site is `ChatManager::set_session_model`: when a session that has
/// an open decision gets its model changed by hand, call this with that
/// decision's id.
pub async fn mark_override(store: &dyn RoutingArmStore, decision_id: Uuid) -> anyhow::Result<bool> {
    let Some(decision) = store.decision(decision_id).await? else {
        return Ok(false);
    };
    if is_closed(&decision) {
        return Ok(false);
    }
    let mut outcome = decision.outcome.unwrap_or_default();
    outcome.overridden = true;
    store.set_outcome(decision_id, outcome).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::{
        CognitiveDecision, DecisionAlternative, Pick,
    };
    use crate::chat::provider::cognitive::mode::{LearningStage, ProviderRoutingMode};
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use crate::chat::provider::cognitive::store::{ArmKey, InMemoryRoutingStore};

    fn decision(chosen: Option<Pick>) -> CognitiveDecision {
        CognitiveDecision {
            id: Uuid::new_v4(),
            at: chrono::Utc::now(),
            signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, None),
            chosen,
            score: None,
            explored: false,
            reason: "test".into(),
            alternatives: vec![],
            applied: true,
            mode: ProviderRoutingMode::Mixed,
            stage: LearningStage::Auto,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: None,
        }
    }

    #[tokio::test]
    async fn closing_a_decision_updates_its_arm_once() {
        let store = InMemoryRoutingStore::new();
        let d = decision(Some(Pick::new("deepseek", "v4")));
        store.put_decision(&d).await.unwrap();
        let settings = RoutingSettings::default();
        let outcome = Outcome {
            success: Some(true),
            attempts: 1,
            ..Outcome::default()
        };
        close_decision(&store, &settings, d.id, &outcome)
            .await
            .unwrap();
        close_decision(&store, &settings, d.id, &outcome)
            .await
            .unwrap();
        let arm = store
            .arm(&ArmKey::new("utility.compaction", "deepseek", "v4"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(arm.n, 1, "the second close must not count twice");
        assert_eq!(arm.alpha, 2.0);
        let stored = store.decision(d.id).await.unwrap().unwrap();
        assert_eq!(stored.outcome.unwrap().success, Some(true));
    }

    #[tokio::test]
    async fn an_unknown_decision_or_one_without_a_choice_creates_no_arm() {
        let store = InMemoryRoutingStore::new();
        let settings = RoutingSettings::default();
        close_decision(&store, &settings, Uuid::new_v4(), &Outcome::default())
            .await
            .unwrap();
        let none = decision(None);
        store.put_decision(&none).await.unwrap();
        close_decision(&store, &settings, none.id, &Outcome::default())
            .await
            .unwrap();
        assert!(store
            .arms_of_class("utility.compaction")
            .await
            .unwrap()
            .is_empty());
        assert!(store
            .decision(none.id)
            .await
            .unwrap()
            .unwrap()
            .outcome
            .is_some());
    }

    fn ok() -> Outcome {
        Outcome {
            success: Some(true),
            attempts: 1,
            ..Outcome::default()
        }
    }

    fn r(outcome: &Outcome, median: Option<f64>) -> f64 {
        reward_with(
            outcome,
            &RoutingSettings::default(),
            &RewardContext {
                class_median_cost: median,
            },
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn the_reward_of_each_case() {
        let cases: Vec<(&str, Outcome, Option<f64>, f64)> = vec![
            ("first-try success", ok(), None, 0.7),
            (
                "failure",
                Outcome {
                    success: Some(false),
                    attempts: 1,
                    ..Outcome::default()
                },
                None,
                0.0,
            ),
            ("unknown", Outcome::default(), None, 0.25),
            (
                "success on 2nd attempt",
                Outcome {
                    attempts: 2,
                    ..ok()
                },
                None,
                0.4,
            ),
            (
                "success on 4th attempt",
                Outcome {
                    attempts: 4,
                    ..ok()
                },
                None,
                0.2,
            ),
            (
                "verification passed",
                Outcome {
                    verification_passed: Some(true),
                    ..ok()
                },
                None,
                0.85,
            ),
            (
                "verification failed",
                Outcome {
                    verification_passed: Some(false),
                    ..ok()
                },
                None,
                0.55,
            ),
            (
                "cheap",
                Outcome {
                    cost_usd: Some(0.5),
                    ..ok()
                },
                Some(1.0),
                0.85,
            ),
            (
                "twice the median",
                Outcome {
                    cost_usd: Some(2.0),
                    ..ok()
                },
                Some(1.0),
                0.775,
            ),
            (
                "three times the median",
                Outcome {
                    cost_usd: Some(3.0),
                    ..ok()
                },
                Some(1.0),
                0.7,
            ),
            (
                "unknown cost adds nothing (not free)",
                Outcome {
                    cost_usd: None,
                    ..ok()
                },
                Some(1.0),
                0.7,
            ),
            (
                "no median adds nothing",
                Outcome {
                    cost_usd: Some(0.0),
                    ..ok()
                },
                None,
                0.7,
            ),
            (
                "unknown duration adds nothing",
                Outcome {
                    duration_ms: None,
                    ..ok()
                },
                None,
                0.7,
            ),
            (
                "user override",
                Outcome {
                    user_overrode_model: true,
                    ..ok()
                },
                None,
                0.4,
            ),
            (
                "interrupted",
                Outcome {
                    interrupted: true,
                    ..ok()
                },
                None,
                0.5,
            ),
            (
                "everything bad clamps to 0",
                Outcome {
                    success: Some(false),
                    attempts: 3,
                    user_overrode_model: true,
                    interrupted: true,
                    verification_passed: Some(false),
                    ..Outcome::default()
                },
                None,
                0.0,
            ),
            (
                "everything good clamps to 1",
                Outcome {
                    verification_passed: Some(true),
                    cost_usd: Some(0.1),
                    ..ok()
                },
                Some(1.0),
                1.0,
            ),
        ];
        for (name, outcome, median, expected) in cases {
            let got = r(&outcome, median);
            assert!(close(got, expected), "{name}: got {got}, want {expected}");
            assert!((0.0..=1.0).contains(&got), "{name} out of range");
        }
    }

    #[test]
    fn more_attempts_never_raise_the_reward() {
        for success in [Some(true), Some(false), None] {
            for verification in [None, Some(true), Some(false)] {
                let mut previous = f64::INFINITY;
                for attempts in 0..8 {
                    let outcome = Outcome {
                        success,
                        attempts,
                        verification_passed: verification,
                        ..Outcome::default()
                    };
                    let got = r(&outcome, None);
                    // 0 reads as 1: same reward.
                    assert!(got <= previous + 1e-12, "{success:?} {attempts}");
                    previous = got;
                }
            }
        }
    }

    #[test]
    fn a_costlier_run_never_earns_more_and_bad_medians_are_ignored() {
        let mut previous = f64::INFINITY;
        for tenth in 1..50 {
            let outcome = Outcome {
                cost_usd: Some(f64::from(tenth) / 10.0),
                ..ok()
            };
            let got = r(&outcome, Some(1.0));
            assert!(got <= previous + 1e-12);
            previous = got;
        }
        for median in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let outcome = Outcome {
                cost_usd: Some(1.0),
                ..ok()
            };
            assert!(close(r(&outcome, Some(median)), 0.7));
        }
        assert!(close(reward(&ok(), &RoutingSettings::default()), 0.7));
    }

    #[tokio::test]
    async fn a_manual_switch_flagged_before_the_close_lowers_the_reward() {
        let settings = RoutingSettings::default();
        let store = InMemoryRoutingStore::new();
        let plain = decision(Some(Pick::new("deepseek", "v4")));
        let flagged = decision(Some(Pick::new("deepseek", "v4")));
        store.put_decision(&plain).await.unwrap();
        store.put_decision(&flagged).await.unwrap();

        assert!(mark_override(&store, flagged.id).await.unwrap());
        assert!(!mark_override(&store, Uuid::new_v4()).await.unwrap());
        // Flagged but not closed: not a closed decision yet.
        let pending = store.decision(flagged.id).await.unwrap().unwrap();
        assert!(pending.outcome.as_ref().unwrap().overridden);
        assert_eq!(pending.outcome.unwrap().reward, None);

        close_decision(&store, &settings, plain.id, &ok())
            .await
            .unwrap();
        close_decision(&store, &settings, flagged.id, &ok())
            .await
            .unwrap();
        let reward_of = |d: CognitiveDecision| d.outcome.unwrap().reward.unwrap();
        let a = reward_of(store.decision(plain.id).await.unwrap().unwrap());
        let b = reward_of(store.decision(flagged.id).await.unwrap().unwrap());
        assert!(close(a, 0.7) && close(b, 0.4), "{a} vs {b}");
        // Too late once closed.
        assert!(!mark_override(&store, flagged.id).await.unwrap());
        // The arm saw both: one success (0.7), one below the threshold (0.4).
        let arm = store
            .arm(&ArmKey::new("utility.compaction", "deepseek", "v4"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!((arm.n, arm.alpha, arm.beta), (2, 2.0, 2.0));
    }

    #[derive(Default)]
    struct Spy(std::sync::Mutex<Vec<(Uuid, f64)>>);
    impl TrajectorySink for Spy {
        fn record(&self, decision: &CognitiveDecision, reward: f64) {
            self.0.lock().unwrap().push((decision.id, reward));
        }
    }

    #[tokio::test]
    async fn closing_with_a_sink_sends_one_trajectory_and_only_once() {
        let store = InMemoryRoutingStore::new();
        let d = decision(Some(Pick::new("deepseek", "v4")));
        store.put_decision(&d).await.unwrap();
        let spy = Spy::default();
        let settings = RoutingSettings::default();
        for _ in 0..2 {
            close_decision_with(&store, &settings, Some(&spy), d.id, &ok())
                .await
                .unwrap();
        }
        let seen = spy.0.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, d.id);
        assert!(close(seen[0].1, 0.7));
    }

    fn alt(model: &str, score: Option<f64>) -> DecisionAlternative {
        DecisionAlternative {
            pick: Pick::new("p", model),
            score,
            rejected: None,
        }
    }

    #[test]
    fn confidence_is_one_minus_normalised_entropy() {
        let mut d = decision(Some(Pick::new("p", "a")));
        assert_eq!(decision_confidence(&d), 0.0, "no alternatives");
        d.alternatives = vec![alt("a", Some(0.9))];
        assert_eq!(decision_confidence(&d), 0.0, "k < 2");
        d.alternatives = vec![alt("a", Some(0.5)), alt("b", Some(0.5))];
        assert!(close(decision_confidence(&d), 0.0), "uniform");
        d.alternatives = vec![alt("a", Some(1.0)), alt("b", Some(0.0)), alt("c", None)];
        assert!(close(decision_confidence(&d), 1.0), "one dominates");
        d.alternatives = vec![alt("a", Some(0.8)), alt("b", Some(0.2))];
        let c = decision_confidence(&d);
        assert!(c > 0.0 && c < 1.0, "{c}");
        d.alternatives = vec![alt("a", Some(0.0)), alt("b", Some(0.0))];
        assert_eq!(decision_confidence(&d), 0.0, "all zero");
    }

    #[test]
    fn the_trajectory_record_describes_the_routing_choice() {
        let mut d = decision(Some(Pick::new("deepseek", "v4")));
        d.score = Some(0.8);
        d.explored = true;
        d.alternatives = vec![alt("v4", Some(0.8)), alt("v3", Some(0.2))];
        let record = trajectory_record(&d, 0.7);
        assert_eq!(record.action_type, "routing.select_model");
        assert_eq!(record.session_id, d.id.to_string());
        assert_eq!(record.alternatives_count, 2);
        assert_eq!(record.chosen_index, 0);
        assert!(record.confidence > 0.0);
        assert_eq!(record.action_params["provider"], "deepseek");
        assert_eq!(record.action_params["model"], "v4");
        assert_eq!(
            record.action_params["arm"],
            "utility.compaction|deepseek|v4"
        );
        assert_eq!(record.action_params["explored"], true);
        assert_eq!(record.action_params["applied"], true);
        let session = Uuid::new_v4();
        d.session_id = Some(session);
        assert_eq!(trajectory_record(&d, 0.7).session_id, session.to_string());
    }
}
