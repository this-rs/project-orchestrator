//! The cognitive decider: candidates, scoring, hysteresis, a readable reason,
//! and the mode/stage rule that says whether the choice is used.
//!
//! [`decide_with`] is pure: the same `(id, request, arms, hints)` always gives
//! the same decision, which is what makes a decision replayable.
//! [`CognitiveDecider`] only loads the arms, calls it, and PERSISTS the result
//! before returning it, applied or not.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::candidates::{self, ModelFacts, Slot};
use super::decision::{CognitiveDecision, DecideRequest, Decider, DecisionAlternative, Pick};
use super::mode::{LearningStage, ProviderRoutingMode};
use super::scorer::{self, PriorHints, Scored};
use super::signature::TaskSignature;
use super::store::{ArmStats, RoutingArmStore};
use crate::chat::provider::resolver::Role;

/// A challenger must beat the pair in force by this much to replace it.
pub const HYSTERESIS_MARGIN: f64 = 0.1;

/// Reason of a decision that found nothing eligible.
pub const NO_CANDIDATE: &str = "no_candidate";

/// Whether a choice is used, from the mode, the stage and the role.
///
/// `shadow` never applies; `primary` applies nothing; `mixed` applies to
/// executors only; `full` applies to everyone. At the `advisory` stage the
/// pilot is advised, never applied, in every mode.
pub fn is_applied(mode: ProviderRoutingMode, stage: LearningStage, role: Role) -> bool {
    if stage == LearningStage::Shadow {
        return false;
    }
    let by_mode = match mode {
        ProviderRoutingMode::Primary => false,
        ProviderRoutingMode::Mixed => role == Role::Executor,
        ProviderRoutingMode::Full => true,
    };
    match stage {
        LearningStage::Shadow => false,
        LearningStage::Advisory => by_mode && role == Role::Executor,
        LearningStage::Auto => by_mode,
    }
}

fn pick_of(facts: &ModelFacts) -> Pick {
    Pick::new(&facts.provider_id, &facts.model)
}

fn class_words(signature: &TaskSignature) -> String {
    let key = signature.arm_key().replace(['.', '_'], " ");
    if key.starts_with("chat") {
        format!("{key} turn")
    } else {
        format!("{key} task")
    }
}

fn tokens(value: u64) -> String {
    if value >= 1_000 {
        format!("{}k", value / 1_000)
    } else {
        value.to_string()
    }
}

/// One readable sentence, without URL or secret: class, window against need,
/// estimated cost, observed success rate with its count.
fn reason_of(signature: &TaskSignature, best: &Scored, explored: bool, kept: bool) -> String {
    let mut parts = vec![class_words(signature)];
    match best.candidate.context_window {
        Some(window) => parts.push(format!(
            "window {} for {} needed",
            tokens(window),
            tokens(signature.context_need_tokens)
        )),
        None => parts.push(format!(
            "window unknown, {} needed",
            tokens(signature.context_need_tokens)
        )),
    }
    parts.push(match best.arm_mean_cost_usd.or(best.est_turn_cost_usd) {
        Some(cost) => format!("about {cost:.2} USD per turn estimated"),
        None => "cost unknown".to_owned(),
    });
    parts.push(if best.n > 0 {
        let rate = (best.alpha / (best.alpha + best.beta) * 100.0).round();
        format!("{rate:.0}% success over {} runs", best.n)
    } else {
        "no run observed yet".to_owned()
    });
    if explored {
        parts.push("picked by the exploration draw".to_owned());
    }
    if kept {
        parts.push("kept because no challenger is clearly better".to_owned());
    }
    format!(
        "{}/{}: {}",
        best.candidate.provider_id,
        best.candidate.model,
        parts.join(", ")
    )
}

/// The decision for `request`, without touching any store.
pub fn decide_with(
    id: Uuid,
    at: DateTime<Utc>,
    request: &DecideRequest,
    arms: &[ArmStats],
    hints: &PriorHints,
) -> CognitiveDecision {
    let signature = &request.signature;
    let settings = &request.settings;
    let mut decision = CognitiveDecision {
        id,
        at,
        signature: signature.clone(),
        chosen: None,
        score: None,
        explored: false,
        reason: String::new(),
        alternatives: Vec::new(),
        applied: false,
        mode: settings.mode,
        stage: settings.stage,
        session_id: request.session_id,
        task_id: request.task_id,
        run_id: request.run_id,
        turn_index: request.turn_index,
        outcome: None,
        used: None,
    };

    let pool: Vec<ModelFacts> = request
        .pool
        .iter()
        .filter(|facts| {
            request
                .restrict_provider
                .as_deref()
                .is_none_or(|only| facts.provider_id == only)
        })
        .cloned()
        .collect();
    let Some(filtered) = candidates::apply(request.slot, signature, &pool) else {
        debug_assert_eq!(request.slot, Slot::Explicit);
        decision.reason =
            "explicit choice: a provider named by the caller is never replaced".into();
        return decision;
    };
    let rejected = |decision: &mut CognitiveDecision| {
        decision
            .alternatives
            .extend(filtered.rejected.iter().map(|r| DecisionAlternative {
                pick: pick_of(&r.candidate),
                score: None,
                rejected: Some(r.reason.clone()),
            }));
    };
    if filtered.eligible.is_empty() {
        decision.reason = NO_CANDIDATE.into();
        rejected(&mut decision);
        return decision;
    }

    let mut scored = scorer::score(
        signature,
        &filtered.eligible,
        arms,
        hints,
        settings,
        scorer::seed_of(id),
    );
    // Hysteresis: the pair in force stays unless a challenger is clearly better.
    let mut kept = false;
    if !scored[0].explored {
        if let Some(current) = &request.current {
            let at_index = scored
                .iter()
                .position(|s| pick_of(&s.candidate) == *current);
            if let Some(index) = at_index {
                if index > 0 && scored[0].utility - scored[index].utility < HYSTERESIS_MARGIN {
                    let stay = scored.remove(index);
                    scored.insert(0, stay);
                    kept = true;
                }
            }
        }
    }
    let best = &scored[0];
    decision.chosen = Some(pick_of(&best.candidate));
    decision.score = Some(best.utility);
    decision.explored = best.explored;
    decision.reason = reason_of(signature, best, best.explored, kept);
    decision.applied = is_applied(settings.mode, settings.stage, signature.role);
    decision.alternatives = scored
        .iter()
        .map(|s| DecisionAlternative {
            pick: pick_of(&s.candidate),
            score: Some(s.utility),
            rejected: None,
        })
        .collect();
    rejected(&mut decision);
    decision
}

/// The decider backed by the arm store.
pub struct CognitiveDecider {
    store: Arc<dyn RoutingArmStore>,
    hints: Arc<RwLock<PriorHints>>,
}

impl CognitiveDecider {
    /// A decider over `store`, with no alias hint.
    pub fn new(store: Arc<dyn RoutingArmStore>) -> Self {
        Self {
            store,
            hints: Arc::new(RwLock::new(PriorHints::new())),
        }
    }

    /// A decider that shares a hints handle with its owner, who refreshes it
    /// when the alias table changes.
    pub fn with_hints(store: Arc<dyn RoutingArmStore>, hints: Arc<RwLock<PriorHints>>) -> Self {
        Self { store, hints }
    }

    /// Replaces the alias hints.
    pub fn set_hints(&self, hints: PriorHints) {
        *self.hints.write().unwrap_or_else(|e| e.into_inner()) = hints;
    }
}

#[async_trait]
impl Decider for CognitiveDecider {
    async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
        let arms = self
            .store
            .arms_of_class(&request.signature.arm_key())
            .await?;
        let hints = self.hints.read().unwrap_or_else(|e| e.into_inner()).clone();
        let decision = decide_with(Uuid::new_v4(), Utc::now(), request, &arms, &hints);
        // Persisted before it is returned, applied or not.
        self.store.put_decision(&decision).await?;
        Ok(decision)
    }
}

/// Capability probes attempted: (instance, model) to the time of the attempt.
pub type ProbeLog = std::collections::HashMap<(String, String), std::time::Instant>;

/// The routing pieces a chat manager owns: the decider, the handle through
/// which the alias hints are refreshed, and the short-lived health memory.
#[derive(Clone)]
pub struct CognitiveRouting {
    /// The decider the manager calls.
    pub decider: Arc<dyn Decider>,
    /// Refreshed from the stored aliases before each decision.
    pub hints: Arc<RwLock<PriorHints>>,
    /// One health probe per instance per window.
    pub health: Arc<candidates::HealthCache>,
    /// Models whose capability probe was attempted, and when: a probe that fails
    /// is not replayed on every decision.
    pub probed: Arc<std::sync::Mutex<ProbeLog>>,
    /// Where decisions and arms live, when the standard wiring built this; it is
    /// how a session's decision is linked and closed. `None` for a test decider.
    pub store: Option<Arc<dyn RoutingArmStore>>,
    /// The live Anthropic model catalog: where the Claude Code candidates and
    /// their windows come from (`max_input_tokens`). `None` (tests, or no
    /// catalog wired): Claude Code adds only what its provider reports.
    pub claude_code_catalog: Option<Arc<crate::chat::model_catalog::ModelCatalogCache>>,
}

impl CognitiveRouting {
    /// The standard wiring over an arm store.
    pub fn new(store: Arc<dyn RoutingArmStore>) -> Self {
        let hints = Arc::new(RwLock::new(PriorHints::new()));
        Self {
            decider: Arc::new(CognitiveDecider::with_hints(
                Arc::clone(&store),
                Arc::clone(&hints),
            )),
            hints,
            health: Arc::new(candidates::HealthCache::standard()),
            probed: Default::default(),
            store: Some(store),
            claude_code_catalog: None,
        }
    }

    /// Wiring around any decider (tests).
    pub fn with_decider(decider: Arc<dyn Decider>) -> Self {
        Self {
            decider,
            hints: Arc::new(RwLock::new(PriorHints::new())),
            health: Arc::new(candidates::HealthCache::standard()),
            probed: Default::default(),
            store: None,
            claude_code_catalog: None,
        }
    }

    /// Feeds the Claude Code candidates from the live model catalog.
    pub fn with_claude_code_catalog(
        mut self,
        catalog: Arc<crate::chat::model_catalog::ModelCatalogCache>,
    ) -> Self {
        self.claude_code_catalog = Some(catalog);
        self
    }

    /// Replaces the alias hints.
    pub fn set_hints(&self, hints: PriorHints) {
        *self.hints.write().unwrap_or_else(|e| e.into_inner()) = hints;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::candidates::RejectReason;
    use crate::chat::provider::cognitive::mode::RoutingSettings;
    use crate::chat::provider::cognitive::signature::{ContextHints, TaskClass};
    use crate::chat::provider::cognitive::store::{ArmKey, InMemoryRoutingStore};
    use nexus_claude::agent::{CostBasis, ModelPrice};

    fn facts(provider: &str, model: &str, price: f64) -> ModelFacts {
        ModelFacts {
            provider_id: provider.into(),
            model: model.into(),
            supports_tools: true,
            supports_images: false,
            context_window: Some(128_000),
            window_unknown: None,
            price: Some(ModelPrice {
                input_per_mtok: price,
                output_per_mtok: price,
                cache_read_per_mtok: None,
                cache_write_per_mtok: None,
            }),
            cost_basis: CostBasis::Priced,
            healthy: Some(true),
            allowed_for_project: true,
            sandboxed: false,
        }
    }

    fn pilot() -> TaskSignature {
        TaskSignature::from_chat_request("hello", false, Some("po"), ContextHints::default())
    }

    fn executor() -> TaskSignature {
        let mut s = TaskSignature::utility(TaskClass::UtilityFeatureGraph, 9_000, Some("po"));
        s.needs_tools = false;
        s
    }

    fn request(
        signature: TaskSignature,
        mode: ProviderRoutingMode,
        stage: LearningStage,
    ) -> DecideRequest {
        DecideRequest::new(
            signature,
            RoutingSettings {
                mode,
                stage,
                exploration_epsilon: 0.0,
                ..RoutingSettings::default()
            },
            vec![facts("deepseek", "v4", 1.0), facts("glm", "big", 2.0)],
        )
    }

    async fn decide(request: &DecideRequest) -> (CognitiveDecision, Arc<InMemoryRoutingStore>) {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        (decider.decide(request).await.unwrap(), store)
    }

    #[tokio::test]
    async fn applied_follows_the_mode_stage_and_role_table() {
        use LearningStage::*;
        use ProviderRoutingMode::*;
        // (mode, stage, pilot applied, executor applied)
        let table = [
            (Primary, Shadow, false, false),
            (Primary, Advisory, false, false),
            (Primary, Auto, false, false),
            (Mixed, Shadow, false, false),
            (Mixed, Advisory, false, true),
            (Mixed, Auto, false, true),
            (Full, Shadow, false, false),
            (Full, Advisory, false, true),
            (Full, Auto, true, true),
        ];
        for (mode, stage, pilot_applied, executor_applied) in table {
            let (p, _) = decide(&request(pilot(), mode, stage)).await;
            let (e, _) = decide(&request(executor(), mode, stage)).await;
            assert!(p.chosen.is_some() && e.chosen.is_some());
            assert_eq!(p.applied, pilot_applied, "pilot {mode:?} {stage:?}");
            assert_eq!(e.applied, executor_applied, "executor {mode:?} {stage:?}");
            assert_eq!((p.mode, p.stage), (mode, stage));
        }
    }

    #[tokio::test]
    async fn every_decision_is_persisted_before_it_is_returned() {
        let (decision, store) = decide(&request(
            pilot(),
            ProviderRoutingMode::Primary,
            LearningStage::Shadow,
        ))
        .await;
        assert!(!decision.applied);
        assert_eq!(store.decision(decision.id).await.unwrap(), Some(decision));
    }

    #[tokio::test]
    async fn nothing_eligible_is_no_candidate_with_the_rejections_listed() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        for f in &mut req.pool {
            f.allowed_for_project = false;
        }
        let (decision, store) = decide(&req).await;
        assert_eq!(decision.chosen, None);
        assert!(!decision.applied);
        assert_eq!(decision.reason, NO_CANDIDATE);
        assert_eq!(decision.alternatives.len(), 2);
        assert!(decision
            .alternatives
            .iter()
            .all(|a| a.rejected == Some(RejectReason::NotAllowed) && a.score.is_none()));
        assert!(store.decision(decision.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn an_explicit_slot_is_returned_unapplied_without_a_choice() {
        let mut req = request(executor(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.slot = Slot::Explicit;
        let (decision, store) = decide(&req).await;
        assert_eq!(decision.chosen, None);
        assert!(!decision.applied);
        assert!(decision.reason.contains("explicit"));
        assert!(store.decision(decision.id).await.unwrap().is_some());
    }

    /// A session in `Trust` (decision ebd2b7e7): every third party the open path accepts
    /// stays a candidate, sandboxed or not; nothing is rejected for the policy mode.
    #[tokio::test]
    async fn a_trust_session_keeps_every_unsandboxed_third_party_as_a_candidate() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.trust = true;
        assert!(req.pool.iter().all(|f| !f.sandboxed));
        let (decision, _) = decide(&req).await;
        assert!(decision.chosen.is_some(), "{}", decision.reason);
        assert!(decision.applied);
        assert!(
            decision.alternatives.iter().all(|a| a.rejected.is_none()),
            "{:?}",
            decision.alternatives
        );
    }

    #[tokio::test]
    async fn the_candidates_can_be_restricted_to_one_instance() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.restrict_provider = Some("glm".into());
        let (decision, _) = decide(&req).await;
        assert_eq!(decision.chosen, Some(Pick::new("glm", "big")));
        assert_eq!(decision.alternatives.len(), 1);
    }

    #[tokio::test]
    async fn a_learnt_arm_beats_a_dear_one_and_the_alternatives_are_best_first() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let sig = executor();
        for _ in 0..20 {
            store
                .observe(
                    &ArmKey::new(sig.arm_key(), "glm", "big"),
                    &crate::chat::provider::cognitive::store::ArmObservation {
                        reward: 1.0,
                        success_threshold: 0.5,
                        cost_usd: Some(0.05),
                        latency_ms: Some(1_000),
                    },
                )
                .await
                .unwrap();
            store
                .observe(
                    &ArmKey::new(sig.arm_key(), "deepseek", "v4"),
                    &crate::chat::provider::cognitive::store::ArmObservation {
                        reward: 0.0,
                        success_threshold: 0.5,
                        cost_usd: Some(0.05),
                        latency_ms: Some(1_000),
                    },
                )
                .await
                .unwrap();
        }
        let decider = CognitiveDecider::new(store);
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool.push({
            let mut gone = facts("sick", "m", 1.0);
            gone.healthy = Some(false);
            gone
        });
        let decision = decider.decide(&req).await.unwrap();
        assert_eq!(decision.chosen, Some(Pick::new("glm", "big")));
        let scores: Vec<_> = decision
            .alternatives
            .iter()
            .filter_map(|a| a.score)
            .collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(
            decision.alternatives.last().unwrap().rejected,
            Some(RejectReason::Unhealthy)
        );
        assert!(
            decision.reason.contains("95% success over 20 runs"),
            "{}",
            decision.reason
        );
    }

    #[tokio::test]
    async fn the_alias_hint_is_a_prior_the_decider_reads() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store);
        decider.set_hints(PriorHints::new().with("glm", "big", "deep"));
        let mut sig = pilot();
        sig.class = TaskClass::Complex;
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.settings.cost_weight = 0.0;
        // Over many ids the hinted pair wins more often than the other.
        let mut wins = 0;
        for i in 0..200u128 {
            let d = decide_with(
                Uuid::from_u128(i * 7919 + 1),
                Utc::now(),
                &req,
                &[],
                &decider.hints.read().unwrap(),
            );
            if d.chosen == Some(Pick::new("glm", "big")) {
                wins += 1;
            }
        }
        assert!(wins > 110, "{wins}");
    }

    #[test]
    fn the_same_id_arms_and_pool_give_the_same_decision() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.settings.exploration_epsilon = 0.25;
        let sig_arms = vec![ArmStats::fresh(
            ArmKey::new(req.signature.arm_key(), "glm", "big"),
            3.0,
            2.0,
        )];
        let hints = PriorHints::new();
        for i in 0..50u128 {
            let (id, at) = (Uuid::from_u128(i + 1), Utc::now());
            let first = decide_with(id, at, &req, &sig_arms, &hints);
            assert_eq!(first, decide_with(id, at, &req, &sig_arms, &hints));
        }
    }

    #[test]
    fn hysteresis_keeps_the_current_pair_unless_a_challenger_clears_the_margin() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        let scored_order = |req: &DecideRequest, id: Uuid| {
            decide_with(id, Utc::now(), req, &[], &PriorHints::new())
        };
        // Find an id whose best pair is deepseek, then declare glm current.
        let id = (1..200u128)
            .map(Uuid::from_u128)
            .find(|id| {
                let d = scored_order(&req, *id);
                d.chosen == Some(Pick::new("deepseek", "v4"))
                    && d.alternatives[0].score.unwrap() - d.alternatives[1].score.unwrap() < 0.1
            })
            .expect("a close call exists");
        req.current = Some(Pick::new("glm", "big"));
        let kept = scored_order(&req, id);
        assert_eq!(kept.chosen, Some(Pick::new("glm", "big")));
        assert!(kept.reason.contains("kept"));
        // A wide margin switches.
        let wide = (1..2_000u128)
            .map(Uuid::from_u128)
            .find(|id| {
                let mut no_current = req.clone();
                no_current.current = None;
                let d = scored_order(&no_current, *id);
                d.chosen == Some(Pick::new("deepseek", "v4"))
                    && d.alternatives[0].score.unwrap() - d.alternatives[1].score.unwrap() >= 0.1
            })
            .expect("a clear win exists");
        assert_eq!(
            scored_order(&req, wide).chosen,
            Some(Pick::new("deepseek", "v4"))
        );
        // A current pair that is not eligible is not kept.
        req.current = Some(Pick::new("ghost", "x"));
        assert_ne!(scored_order(&req, id).chosen, Some(Pick::new("ghost", "x")));
    }

    #[test]
    fn the_reason_is_one_readable_sentence_without_url_or_secret() {
        let req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        let d = decide_with(
            Uuid::from_u128(5),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        );
        assert!(!d.reason.contains("://") && !d.reason.contains("sk-"));
        assert!(d.reason.contains("window 128k"), "{}", d.reason);
        assert!(d.reason.contains("USD per turn"), "{}", d.reason);
        assert!(d.reason.contains("no run observed yet"), "{}", d.reason);
        assert!(d.reason.contains("chat general turn"), "{}", d.reason);
    }
}
