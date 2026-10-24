//! The decision record and the seam between deciding and using a decision.
//!
//! [`CognitiveDecision`] is what the router answers and what is persisted, applied
//! or not. [`Decider`] is the one trait the rest of the backend (chat, runner,
//! delegation, per-turn hook) calls: they depend on it, never on the scorer, so
//! each can be built and tested against a fake while the scorer is written.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::signature::TaskClass;
    use crate::chat::provider::cognitive::store::ArmKey;

    fn decision() -> CognitiveDecision {
        CognitiveDecision {
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
