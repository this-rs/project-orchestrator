//! Closing the loop: what happened after a decision, turned into a reward for
//! the arm that was chosen.
//!
//! The seam is [`close_decision`]: every place that finishes work a decision
//! routed (the runner when an attempt closes, a chat session when it closes, a
//! delegation) calls it with an [`Outcome`]. The reward function is the
//! placeholder [`reward`]; B-R5 replaces its body with the full signal set.

use uuid::Uuid;

use super::decision::DecisionOutcome;
use super::mode::RoutingSettings;
use super::store::{ArmObservation, RoutingArmStore};

/// Reward above which an observation counts as a success for the Beta.
pub const SUCCESS_THRESHOLD: f64 = 0.5;

/// What is observable when the work a decision routed ends.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    /// Whether the work succeeded; `None` when it cannot be told.
    pub success: Option<bool>,
    /// Attempts used, starting at 1.
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
}

/// Reward in `[0, 1]` for an outcome.
///
/// PLACEHOLDER (B-R5 replaces the body): success is 1, failure 0, unknown 0.5.
pub fn reward(outcome: &Outcome, _settings: &RoutingSettings) -> f64 {
    match outcome.success {
        Some(true) => 1.0,
        Some(false) => 0.0,
        None => 0.5,
    }
}

/// Records the outcome of a decision: fills the decision's outcome and updates
/// the arm it chose. A decision that chose nothing, or is unknown, only gets
/// its outcome written (when it exists). Idempotent per decision: an outcome
/// already recorded is not counted twice.
pub async fn close_decision(
    store: &dyn RoutingArmStore,
    settings: &RoutingSettings,
    decision_id: Uuid,
    outcome: &Outcome,
) -> anyhow::Result<()> {
    let Some(decision) = store.decision(decision_id).await? else {
        return Ok(());
    };
    if decision.outcome.is_some() {
        return Ok(());
    }
    let reward = reward(outcome, settings);
    store
        .set_outcome(
            decision_id,
            DecisionOutcome {
                success: outcome.success,
                reward: Some(reward),
                cost_usd: outcome.cost_usd,
                duration_ms: outcome.duration_ms,
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::{CognitiveDecision, Pick};
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
}
