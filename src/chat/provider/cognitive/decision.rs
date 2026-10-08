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
}

impl CognitiveDecision {
    /// The arm key this decision's choice belongs to, `None` without a choice.
    pub fn arm(&self) -> Option<super::store::ArmKey> {
        self.chosen.as_ref().map(|pick| {
            super::store::ArmKey::new(self.signature.arm_key(), &pick.provider_id, &pick.model)
        })
    }
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
    /// Whether the session would run in trust mode.
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
        }
    }
}

/// Takes a routing decision. Implemented by the scorer (B-R4); faked in the
/// tests of the callers.
///
/// Contract: the decision is persisted before it is returned, whether it is
/// applied or not. `applied` follows the mode and the stage: a `shadow` stage
/// never applies, `primary` applies nothing, `mixed` applies to executors and
/// utilities only, `full` applies to everyone. An explicit slot is returned
/// unapplied with `chosen: None`.
#[async_trait]
pub trait Decider: Send + Sync {
    /// Decides, persists, returns.
    async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision>;
}
