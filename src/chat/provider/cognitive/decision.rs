//! The decision record and the seam between deciding and using a decision.
//!
//! [`CognitiveDecision`] is what the router answers and what is persisted, applied
//! or not. [`Decider`] is the one trait the rest of the backend (chat, runner,
//! delegation, per-turn hook) calls: they depend on it, never on the scorer, so
//! each can be built and tested against a fake while the scorer is written.
use crate::neo4j::traits::*;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::candidates::{ModelFacts, RejectReason, Slot};
use super::mode::{LearningStage, ProviderRoutingMode, RoutingSettings};
use super::signature::TaskSignature;

/// A provider instance and one of its models.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pick {
    /// Instance identifier.
    pub provider_id: String,
    /// Model identifier.
    pub model: String,
}

impl Pick {
    /// A pick.
    pub fn new(provider_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model: model.into(),
        }
    }
}

/// One option the router looked at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionAlternative {
    /// The option.
    pub pick: Pick,
    /// Its utility, `None` when it was not scored (rejected before).
    pub score: Option<f64>,
    /// Why it was not eligible, when it was not.
    pub rejected: Option<RejectReason>,
    /// Marginal probability this arm's Thompson draw is the max (`p_top1`),
    /// `None` when it was not scored. What an off-policy estimator needs to
    /// replay the pure-Thompson policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_top1: Option<f64>,
    /// Logged selection propensity under the ε-greedy Thompson policy:
    /// `(1−ε)·p_top1 + ε/k`, before the hysteresis. `None` when not scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propensity: Option<f64>,
}

impl DecisionOutcome {
    /// Recompute `reward()` from the journal alone.
    ///
    /// `None` when the node predates the fields this needs (`attempts`): saying
    /// "unknown" is honest, inventing a default would fabricate a reward the
    /// router never saw.
    pub fn reward_recompute(&self, settings: &super::mode::RoutingSettings) -> Option<f64> {
        let attempts = self.attempts?;
        let outcome = super::feedback::Outcome {
            success: self.success,
            attempts,
            cost_usd: self.cost_usd,
            duration_ms: self.duration_ms,
            verification_passed: self.verification_passed,
            interrupted: false,
            user_overrode_model: self.overridden,
            ..Default::default()
        };
        Some(super::feedback::reward(&outcome, settings))
    }
}

/// What happened afterwards, filled when the work closes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Whether the work succeeded; `None` when it could not be told.
    pub success: Option<bool>,
    /// Reward in `[0, 1]` given to the arm.
    pub reward: Option<f64>,
    /// Marginal USD spent; `None` when unknown (never zero).
    pub cost_usd: Option<f64>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Input tokens of the work; `None` when unknown (never zero).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output tokens of the work; `None` when unknown (never zero).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Attempts the turn took. `None` on nodes written before this existed.
    ///
    /// `reward()` needs it: without it the reward cannot be recomputed from the
    /// journal alone, which is what makes an offline replay possible (PYSR-1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts: Option<u32>,
    /// Whether the verification passed, when there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_passed: Option<bool>,
    /// The user switched the model by hand after the decision. May be set
    /// before the outcome is closed (`reward` is still `None` then).
    #[serde(default)]
    pub overridden: bool,
}

/// A routing decision, applied or not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveDecision {
    /// Identifier; also the seed of the exploration draw, so a decision replays.
    pub id: Uuid,
    /// When it was taken.
    pub at: DateTime<Utc>,
    /// What was asked for.
    pub signature: TaskSignature,
    /// The pair chosen; `None` when nothing was eligible.
    pub chosen: Option<Pick>,
    /// Utility of the chosen pair.
    pub score: Option<f64>,
    /// Whether the draw was an exploration.
    pub explored: bool,
    /// The exploration rate in force (clamped), so a replay reconstructs the
    /// ε-greedy propensity from the logged `p_top1` and `propensity`.
    #[serde(default)]
    pub exploration_epsilon: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<DrawSnapshot>,
    /// Confidence of the decision: `1 − H(p)/ln(k)` over the scored
    /// alternatives' scores. `None` when it was not scored. Persisted so the
    /// dashboard and the off-policy replay see the same number the scorer saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// One readable sentence (no URL, no secret).
    pub reason: String,
    /// The options considered, best first, rejected ones included.
    pub alternatives: Vec<DecisionAlternative>,
    /// Whether the choice was actually used.
    pub applied: bool,
    /// Mode in force.
    pub mode: ProviderRoutingMode,
    /// Stage in force.
    pub stage: LearningStage,
    /// Chat session it belongs to.
    pub session_id: Option<Uuid>,
    /// Task it belongs to.
    pub task_id: Option<Uuid>,
    /// Plan run it belongs to.
    pub run_id: Option<Uuid>,
    /// Turn index, for a per-turn decision.
    pub turn_index: Option<u32>,
    /// Filled when the work closes.
    pub outcome: Option<DecisionOutcome>,
    /// The pair the work actually ran on, when known. Equals `chosen` when the
    /// decision was applied; differs when the declarative choice was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used: Option<Pick>,
}

impl CognitiveDecision {
    /// The arm key this decision's choice belongs to, `None` without a choice.
    pub fn arm(&self) -> Option<super::store::ArmKey> {
        self.chosen.as_ref().map(|pick| {
            super::store::ArmKey::new(self.signature.arm_key(), &pick.provider_id, &pick.model)
        })
    }

    /// The arm key of the pair that actually served the turn, `None` when
    /// neither the served nor the chosen pair is known. Differs from
    /// [`Self::arm`] only when the declarative choice was kept but the work ran
    /// elsewhere (`used`); equals it when the decision was applied. A decision
    /// that was NOT applied ran on the session's current provider, not on
    /// `chosen`, so it never attributes a reward to `chosen` (only `used` may
    /// name the pair that really served it).
    pub fn served_arm(&self) -> Option<super::store::ArmKey> {
        let pick = match &self.used {
            Some(used) => used,
            None if self.applied => self.chosen.as_ref()?,
            None => return None,
        };
        Some(super::store::ArmKey::new(
            self.signature.arm_key(),
            &pick.provider_id,
            &pick.model,
        ))
    }
}

/// The reward a routed turn earns for the arm that served it, from how the
/// turn ended; `None` when the ending is ambiguous and must not move the
/// posterior.
///
/// `stop_reason` is the neutral stop reason (`completed`, `max_turns`,
/// `max_tokens`, `interrupted`, `refusal`, `budget_exceeded`, `error`),
/// `subtype` the provider's result subtype (`success`,
/// `error_during_execution`, …), `is_error` whether the turn reported an
/// error, and `provider_error` whether a typed provider error was the cause
/// (the turn's `error_code`, surfaced by ERR-1).
///
/// * a concluded success (`stop_reason == "completed"`; `subtype == "success"`
///   only when no stop reason is known) earns `1.0`;
/// * a provider error (`is_error` with a provider cause) earns `0.0`;
/// * every other ending — interrupted, max_turns, max_tokens, budget,
///   user-stopped, refusal, an error without a provider cause — is `None`, so
///   the bandit learns which arm is right for the job, not which arm merely
///   did not error.
pub fn turn_reward(
    stop_reason: Option<&str>,
    subtype: &str,
    is_error: bool,
    provider_error: bool,
) -> Option<f64> {
    if is_error {
        return provider_error.then_some(0.0);
    }
    let success = match stop_reason {
        Some("completed") => true,
        Some(_) => false,
        None => subtype == "success",
    };
    success.then_some(1.0)
}

/// Whether a turn that ended in error blames the PROVIDER'S MODEL, for the reward (QUOTA): it does
/// when the error has a typed provider cause, unless that cause is a refusal of the ACCOUNT (a
/// rate limit, an exhausted quota, no credit). Such a turn says the instance is out for a while,
/// not that the arm answers badly: the router excludes the instance until its reset (see
/// [`super::candidates::RejectReason::QuotaExhausted`]), it does not unlearn the model.
pub fn provider_fault(error_code: Option<&str>, quota_refusal: bool) -> bool {
    error_code.is_some() && !quota_refusal
}

/// What to feed the bandit after a turn, decided in ONE place.
///
/// `None` means "feed nothing", and every reason is deliberate: an overridden
/// decision (already fed by `close_routing_decision`), a decision without a served
/// arm, an ending that says nothing about the arm (see [`turn_reward`]), and a
/// failure that is the ACCOUNT's rather than the model's — a rate limit, an
/// exhausted quota, no credit — which says the instance is out, not that the arm
/// answers badly (see [`provider_fault`], QUOTA).
///
/// Extracted from the provider's `after_turn`: that wiring had no test, so a
/// swapped argument or a forgotten guard would have stayed invisible until the
/// posterior drifted.
pub fn turn_settlement(
    decision: &CognitiveDecision,
    stop_reason: Option<&str>,
    subtype: &str,
    is_error: bool,
    error_code: Option<&str>,
    quota_refusal: bool,
) -> Option<(super::store::ArmKey, super::store::ArmObservation)> {
    if decision.outcome.as_ref().is_some_and(|o| o.overridden) {
        return None;
    }
    let reward = turn_reward(
        stop_reason,
        subtype,
        is_error,
        provider_fault(error_code, quota_refusal),
    )?;
    let arm = decision.served_arm()?;
    Some((
        arm,
        super::store::ArmObservation {
            reward,
            success_threshold: super::decider::TURN_REWARD_THRESHOLD,
            decay: 1.0,

            cost_usd: None,
            latency_ms: None,
        },
    ))
}

/// One scored arm, exactly as the draw saw it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotRow {
    pub pick: Pick,
    pub alpha: f64,
    pub beta: f64,
    pub cost_norm: f64,
    pub latency_norm: f64,
    pub quota_norm: f64,
    pub n: u64,
    pub utility: f64,
    pub explored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DrawSnapshot {
    pub settings: super::mode::RoutingSettings,
    pub steady: bool,
    pub current: Option<Pick>,
    pub rows: Vec<SnapshotRow>,
}

/// A request to decide.
#[derive(Debug, Clone)]
pub struct DecideRequest {
    /// What is asked for.
    pub signature: TaskSignature,
    /// Settings in force for the project.
    pub settings: RoutingSettings,
    /// Every (instance, model) the project could use, with its facts.
    pub pool: Vec<ModelFacts>,
    /// Whether the slot is automatic. An explicit slot is never decided.
    pub slot: Slot,
    /// Whether the session would run in trust mode. Informative: it filters no candidate
    /// (decision ebd2b7e7 refuses `Trust` only on a remote machine without `allow_trust`,
    /// and a remote machine is never a candidate).
    pub trust: bool,
    /// Restrict the candidates to one instance (a live session keeps its provider).
    pub restrict_provider: Option<String>,
    /// The pair in force now, for hysteresis on a per-turn decision.
    pub current: Option<Pick>,
    /// Chat session.
    pub session_id: Option<Uuid>,
    /// Task.
    pub task_id: Option<Uuid>,
    /// Plan run.
    pub run_id: Option<Uuid>,
    /// Turn index.
    pub turn_index: Option<u32>,
    /// The work cannot run on another pair than `current`, whatever is chosen (the
    /// compaction summary: neither engine lets the caller name its model). The decision
    /// is still taken and stored, never applied, its reason prefixed with this code and
    /// `used` set to `current`.
    pub not_selectable: Option<&'static str>,
    /// R-S3: the turn follows a change of model made for the SAME class on the previous turn.
    /// No exploration draw, and the hysteresis margin is applied to the posterior MEANS of
    /// the arms (not to the Thompson draws): the pair in force is replaced only by a
    /// challenger better ON AVERAGE by [`super::decider::HYSTERESIS_MARGIN`], so two
    /// same-class turns can never go A → B → A on noisy draws (decision f9363403).
    pub steady: bool,
    /// The caller's guards that narrowed this decision (`same_class_no_explore`,
    /// `after_provider_move`): each is appended to the stored reason as
    /// `; narrowed: <code>`, like `not_moved:` / `not_changed:`.
    pub narrowed: Vec<&'static str>,
    /// A confirmed conversation-segment boundary on this turn, pre-formatted by
    /// the caller (e.g. `boundary(topic)@12`), appended to the reason as
    /// `; segment: <note>` so a stored decision names why a direction changed.
    /// `None` when no boundary confirmed on this turn.
    pub segment_note: Option<String>,
}

impl DecideRequest {
    /// A request with nothing but a signature, settings and pool.
    pub fn new(signature: TaskSignature, settings: RoutingSettings, pool: Vec<ModelFacts>) -> Self {
        Self {
            signature,
            settings,
            pool,
            slot: Slot::Automatic,
            trust: false,
            restrict_provider: None,
            current: None,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            not_selectable: None,
            steady: false,
            narrowed: Vec::new(),
            segment_note: None,
        }
    }
}

/// Takes a routing decision. Implemented by the scorer (B-R4); faked in the
/// tests of the callers.
///
/// Contract: the decision is persisted before it is returned, whether it is
/// applied or not. `applied` follows the mode and the stage: a `shadow` stage
/// never applies, `primary` applies nothing, `executors_only` applies to executors and
/// utilities only, `full` applies to everyone. An explicit slot is returned
/// unapplied with `chosen: None`.
#[async_trait]
pub trait Decider: Send + Sync {
    /// Decides, persists, returns.
    async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision>;

    /// The decision `decide` would take, NOT persisted: what an opening that is refused
    /// before it routes (Auto on a locked vault) offers instead, without leaving a decision
    /// no session will ever close. The default decides (a test fake has no store).
    async fn preview(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
        self.decide(request).await
    }

    /// Feeds the outcome of the turn a decision routed back to the arm that
    /// served it: `1.0` for a concluded success, `0.0` for a provider error,
    /// nothing for an ambiguous ending (interrupted, max_turns, budget, …).
    /// See [`turn_reward`]. The default is a no-op — a fake decider has no arm
    /// store to learn in; only a decider with a store overrides it.
    async fn observe_turn(
        &self,
        _decision: &CognitiveDecision,
        _stop_reason: Option<&str>,
        _subtype: &str,
        _is_error: bool,
        _provider_error: bool,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// PYSR-1, hole 2: `reward()` needs the attempts and the verified flag, which
/// were not in the journal — so an offline replay could not recompute the reward
/// the router actually learned from.
#[cfg(test)]
mod recompute_tests {
    use super::*;
    use crate::neo4j::traits::*;

    fn settings() -> super::super::mode::RoutingSettings {
        super::super::mode::RoutingSettings::default()
    }

    #[test]
    fn the_reward_is_recomputable_from_the_journal_alone() {
        let live = super::super::feedback::Outcome {
            success: Some(true),
            attempts: 1,
            cost_usd: Some(0.02),
            duration_ms: Some(1_200),
            verification_passed: Some(true),
            ..Default::default()
        };
        let expected = super::super::feedback::reward(&live, &settings());

        let journal = DecisionOutcome {
            success: live.success,
            reward: Some(expected),
            cost_usd: live.cost_usd,
            duration_ms: live.duration_ms,
            attempts: Some(live.attempts),
            verification_passed: live.verification_passed,
            ..Default::default()
        };

        assert_eq!(
            journal.reward_recompute(&settings()),
            Some(expected),
            "recomputing from the journal must give the value the router learned from"
        );
    }

    #[test]
    fn a_node_written_before_the_fields_is_not_recomputed() {
        // Better "unknown" than a fabricated zero: a replay must not invent a
        // reward the router never saw.
        let legacy = DecisionOutcome {
            success: Some(true),
            ..Default::default()
        };
        assert_eq!(legacy.reward_recompute(&settings()), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::signature::TaskClass;
    use crate::chat::provider::cognitive::store::ArmKey;
    use crate::neo4j::traits::*;

    fn decision() -> CognitiveDecision {
        CognitiveDecision {
            snapshot: None,
            id: uuid::Uuid::new_v4(),
            at: chrono::Utc::now(),
            signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, None),
            chosen: Some(Pick::new("deepseek", "v4")),
            score: None,
            explored: false,
            exploration_epsilon: 0.0,
            confidence: None,
            reason: "test".into(),
            alternatives: vec![],
            applied: true,
            mode: ProviderRoutingMode::Full,
            stage: LearningStage::Auto,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: None,
        }
    }

    /// The settlement decision, which had no test: a swapped argument or a
    /// forgotten guard at the manager's call site would have stayed invisible
    /// until the posterior drifted.
    #[test]
    fn a_concluded_success_feeds_one() {
        let (_, observation) = turn_settlement(
            &decision(),
            Some("completed"),
            "success",
            false,
            None,
            false,
        )
        .expect("a concluded success must feed the arm");
        assert_eq!(observation.reward, 1.0);
    }

    #[test]
    fn a_provider_error_feeds_zero() {
        let (_, observation) = turn_settlement(
            &decision(),
            Some("error"),
            "error_during_execution",
            true,
            Some("server_error"),
            false,
        )
        .expect("a provider fault is the arm's failure");
        assert_eq!(observation.reward, 0.0);
    }

    #[test]
    fn an_account_refusal_feeds_nothing() {
        // 429, exhausted quota, no credit: the ACCOUNT is out, the arm is not to
        // blame. Feeding a zero here would make the model pay for the bill.
        assert!(turn_settlement(
            &decision(),
            Some("error"),
            "error_during_execution",
            true,
            Some("rate_limit_error"),
            true
        )
        .is_none());
    }

    #[test]
    fn an_overridden_decision_is_never_fed_again() {
        let mut decided = decision();
        decided.outcome = Some(DecisionOutcome {
            overridden: true,
            ..Default::default()
        });
        assert!(
            turn_settlement(&decided, Some("completed"), "success", false, None, false).is_none(),
            "an override is already fed by close_routing_decision"
        );
    }

    #[test]
    fn a_decision_without_a_served_arm_is_not_fed() {
        let mut decided = decision();
        decided.chosen = None;
        assert!(
            turn_settlement(&decided, Some("completed"), "success", false, None, false).is_none()
        );
    }

    #[test]
    fn a_completed_turn_is_a_full_reward() {
        assert_eq!(
            turn_reward(Some("completed"), "success", false, false),
            Some(1.0)
        );
    }

    #[test]
    fn a_success_subtype_without_a_stop_reason_is_a_full_reward() {
        assert_eq!(turn_reward(None, "success", false, false), Some(1.0));
    }

    #[test]
    fn a_provider_error_is_a_full_penalty() {
        assert_eq!(
            turn_reward(Some("error"), "error_during_execution", true, true),
            Some(0.0)
        );
    }

    #[test]
    fn a_turn_the_account_refused_is_not_the_models_failure() {
        // R0: a rate limit or an exhausted quota used to read as `provider_error` and cost the
        // arm a full penalty, so the bandit unlearned a model that was merely out of quota.
        assert!(!provider_fault(Some("rate_limited"), true));
        assert_eq!(
            turn_reward(
                Some("error"),
                "error_during_execution",
                true,
                provider_fault(Some("rate_limited"), true)
            ),
            None
        );
        // A genuine provider failure still costs the arm.
        assert!(provider_fault(Some("protocol"), false));
        assert_eq!(
            turn_reward(
                Some("error"),
                "error_during_execution",
                true,
                provider_fault(Some("protocol"), false)
            ),
            Some(0.0)
        );
        assert!(!provider_fault(None, false));
    }

    #[test]
    fn an_error_without_a_provider_cause_is_ambiguous() {
        assert_eq!(
            turn_reward(Some("error"), "error_during_execution", true, false),
            None
        );
    }

    #[test]
    fn ambiguous_endings_never_feed_the_posterior() {
        for reason in [
            "interrupted",
            "max_turns",
            "budget_exceeded",
            "refusal",
            "max_tokens",
        ] {
            assert_eq!(
                turn_reward(Some(reason), "success", false, false),
                None,
                "{reason} must not read as a success"
            );
        }
    }

    #[test]
    fn a_max_tokens_turn_is_not_a_success_even_when_the_subtype_defaults_to_success() {
        // `default_subtype` maps an unknown non-error stop reason to "success";
        // the neutral stop reason still says the turn did not complete.
        assert_eq!(
            turn_reward(Some("max_tokens"), "success", false, false),
            None
        );
    }

    #[test]
    fn served_arm_prefers_used_and_falls_back_to_chosen() {
        let mut d = decision();
        assert_eq!(d.served_arm(), d.arm(), "applied: served == chosen");
        d.used = Some(Pick::new("glm", "big"));
        assert_eq!(
            d.served_arm(),
            Some(ArmKey::new("utility.compaction", "glm", "big"))
        );
        // A decision that was NOT applied ran on the session's current provider,
        // not on `chosen`: without a `used`, it names no arm.
        d.used = None;
        d.applied = false;
        assert_eq!(d.served_arm(), None, "not applied: never `chosen`");
        // Without a served nor a chosen pair there is no arm to feed.
        d.chosen = None;
        assert_eq!(d.served_arm(), None);
    }
}

impl CognitiveDecision {
    /// Redo the choice from the journal alone.
    pub fn replay(&self) -> Option<Pick> {
        let snap = self.snapshot.as_ref()?;
        if snap.rows.is_empty() {
            return None;
        }
        let key = |p: &Pick| format!("{}/{}", p.provider_id, p.model);
        let utilities: Vec<f64> = snap
            .rows
            .iter()
            .map(|r| {
                if snap.steady {
                    super::decider::utility_of(
                        r.alpha,
                        r.beta,
                        r.cost_norm,
                        r.latency_norm,
                        r.quota_norm,
                        &snap.settings,
                    )
                } else {
                    r.utility
                }
            })
            .collect();
        let mut order: Vec<usize> = (0..snap.rows.len()).collect();
        order.sort_by(|&a, &b| {
            utilities[b]
                .partial_cmp(&utilities[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| key(&snap.rows[a].pick).cmp(&key(&snap.rows[b].pick)))
        });
        let ns: Vec<u64> = order.iter().map(|&i| snap.rows[i].n).collect();
        let us: Vec<f64> = order.iter().map(|&i| utilities[i]).collect();
        let head = snap.rows[order[0]].pick.clone();
        if snap.rows[order[0]].explored {
            return Some(head);
        }
        let current = match snap.current.clone() {
            Some(c) => c,
            None => return Some(head),
        };
        let index = order.iter().position(|&i| snap.rows[i].pick == current)?;
        if index == 0 {
            return Some(head);
        }
        let margin = super::decider::hysteresis_margin_of(&ns, &us, index);
        if us[0] - us[index] < margin {
            Some(current)
        } else {
            Some(head)
        }
    }
}
