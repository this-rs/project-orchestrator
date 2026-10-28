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
use super::decision::{
    turn_reward, CognitiveDecision, DecideRequest, Decider, DecisionAlternative, Pick,
};
use super::feedback::{confidence_from_scores, decision_confidence};
use super::mode::{LearningStage, ProviderRoutingMode, RoutingSettings, MAX_EXPLORATION_EPSILON};
use super::scorer::{self, PriorHints, Scored};
use super::segment_store::SegmentStore;
use super::signature::TaskSignature;
use super::store::{ArmObservation, ArmStats, RoutingArmStore};
use crate::chat::provider::resolver::Role;

/// A challenger must beat the pair in force by this much to replace it.
pub const HYSTERESIS_MARGIN: f64 = 0.1;

/// Minimum effective observations (`n_eff`, the observation count: each
/// fractional reward adds exactly 1 to `alpha + beta`) before the
/// confidence-derived hysteresis margin applies. Below it a barely-observed
/// winner must not lock in on thin evidence — the base [`HYSTERESIS_MARGIN`]
/// still decides. Calibrate against the replay journal (PYSR-1) before lowering.
pub const HYSTERESIS_WARMUP: u64 = 10;

/// Reason of a decision that found nothing eligible.
pub const NO_CANDIDATE: &str = "no_candidate";

/// The success threshold of the turn reward. The reward is exactly `0.0` or
/// `1.0`, so any threshold strictly inside `(0, 1)` splits them the same; this
/// one matches the arm store's binary split (a success counts `alpha += 1`, a
/// failure `beta += 1`, i.e. `alpha += reward`, `beta += 1 - reward`).
pub(crate) const TURN_REWARD_THRESHOLD: f64 = 0.5;

/// Whether a choice is used, from the mode, the stage and the role.
///
/// `shadow` never applies; `primary` applies nothing; `executors_only` applies to
/// executors only; `full` applies to everyone. At the `advisory` stage the
/// pilot is advised, never applied, in every mode.
pub fn is_applied(mode: ProviderRoutingMode, stage: LearningStage, role: Role) -> bool {
    if stage == LearningStage::Shadow {
        return false;
    }
    let by_mode = match mode {
        ProviderRoutingMode::Primary => false,
        ProviderRoutingMode::ExecutorsOnly => role == Role::Executor,
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
    if signature.needs_images {
        // F-R4: the pool was constrained, nothing was substituted.
        parts.push("reads the attached images (models without vision excluded)".to_owned());
    }
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

/// Reason code of a compaction decision that is recorded, not applied: the summary model
/// cannot be chosen (the Claude Code CLI summarises on the session's model, measured on
/// 2.1.296), or the conversation is not in mode `full` / `executors_only` + stage `auto`. A
/// native session (nexus contract v8, `Capabilities::compaction_model`) in those modes gets
/// the decision APPLIED through `SessionHooks::compaction_directive` instead
/// (`agent_hooks::directive_for_compaction`).
pub const COMPACTION_MODEL_NOT_SELECTABLE: &str = "compaction_model_not_selectable";

/// The decision for `request`, without touching any store.
pub fn decide_with(
    id: Uuid,
    at: DateTime<Utc>,
    request: &DecideRequest,
    arms: &[ArmStats],
    hints: &PriorHints,
) -> CognitiveDecision {
    let mut decision = decide_free(id, at, request, arms, hints);
    if let Some(code) = request.not_selectable {
        // Taken and stored for learning, but the work runs where it runs: never applied.
        decision.applied = false;
        decision.reason = format!("{code}: {}", decision.reason);
        decision.used = request.current.clone();
    }
    for code in &request.narrowed {
        decision.reason = format!("{}; narrowed: {code}", decision.reason);
    }
    if let Some(note) = &request.segment_note {
        decision.reason = format!("{}; segment: {note}", decision.reason);
    }
    decision
}

/// The utility of a scored pair with its posterior MEAN in place of its Thompson draw.
fn mean_utility(scored: &Scored, settings: &RoutingSettings) -> f64 {
    utility_of(
        scored.alpha,
        scored.beta,
        scored.cost_norm,
        scored.latency_norm,
        scored.quota_norm,
        settings,
    )
}

/// The posterior mean utility, **from the numbers alone**.
///
/// One implementation, shared by the live path and any replay: a replay that
/// recomputed utility its own way would drift from the router silently — the
/// exact failure this journal exists to prevent (PYSR-1). Any future change to
/// the formula therefore reaches both, or neither.
pub fn utility_of(
    alpha: f64,
    beta: f64,
    cost_norm: f64,
    latency_norm: f64,
    quota_norm: f64,
    settings: &RoutingSettings,
) -> f64 {
    alpha / (alpha + beta)
        - settings.cost_weight * cost_norm
        - settings.latency_weight * latency_norm
        - settings.quota_weight * quota_norm
}

/// The hysteresis margin: how much better than the pair in force a challenger
/// must be to replace it. Confidence-derived once the arms in play are warm
/// (L1): a tight field — low confidence, the challenger's advantage is likely
/// noise — widens the margin, while a clear winner keeps the base
/// [`HYSTERESIS_MARGIN`]. Below [`HYSTERESIS_WARMUP`] effective observations on
/// either side the confidence is not trusted and the base margin decides, so a
/// barely-observed arm never locks in on a lucky dominance.
fn hysteresis_margin(
    scored: &[Scored],
    current_index: usize,
    settings: &RoutingSettings,
    steady: bool,
) -> f64 {
    let challenger = &scored[0];
    let incumbent = &scored[current_index];
    if challenger.n < HYSTERESIS_WARMUP || incumbent.n < HYSTERESIS_WARMUP {
        return HYSTERESIS_MARGIN;
    }
    // Confidence over the SAME signal the switch is decided on: the posterior
    // means on a steady turn, the Thompson draws otherwise. A single shared
    // implementation (`confidence_from_scores`), never a local copy.
    let utilities: Vec<f64> = scored
        .iter()
        .map(|s| {
            if steady {
                mean_utility(s, settings)
            } else {
                s.utility
            }
        })
        .collect();
    let confidence = confidence_from_scores(&utilities);
    // Never narrower than the base margin; only wider when uncertain. Confidence
    // 1 (clear winner) keeps the base, confidence 0 (indistinguishable) doubles it.
    HYSTERESIS_MARGIN * (2.0 - confidence)
}

/// [`decide_with`] as if the chosen pair could always be used.
fn decide_free(
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
        exploration_epsilon: 0.0,
        confidence: None,
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
                p_top1: None,
                propensity: None,
            }));
    };
    if filtered.eligible.is_empty() {
        decision.reason = NO_CANDIDATE.into();
        rejected(&mut decision);
        return decision;
    }

    // A steady turn (R-S3) never explores.
    let steady_settings;
    let score_settings = if request.steady {
        let mut s = settings.clone();
        s.exploration_epsilon = 0.0;
        steady_settings = s;
        &steady_settings
    } else {
        settings
    };
    let mut scored = scorer::score(
        signature,
        &filtered.eligible,
        arms,
        hints,
        score_settings,
        scorer::seed_of(id),
    );
    // Hysteresis: the pair in force stays unless a challenger is clearly better — by its
    // draw, or, on a steady turn, by its posterior mean (a bound that does not depend on
    // the noise of the draws).
    let gap = |best: &Scored, now: &Scored| {
        if request.steady {
            mean_utility(best, settings) - mean_utility(now, settings)
        } else {
            best.utility - now.utility
        }
    };
    let mut kept = false;
    if !scored[0].explored {
        if let Some(current) = &request.current {
            let at_index = scored
                .iter()
                .position(|s| pick_of(&s.candidate) == *current);
            if let Some(index) = at_index {
                if index > 0 {
                    let margin = hysteresis_margin(&scored, index, settings, request.steady);
                    if gap(&scored[0], &scored[index]) < margin {
                        let stay = scored.remove(index);
                        scored.insert(0, stay);
                        kept = true;
                    }
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
    // Log the off-policy propensities: the exploration rate in force, the
    // number of scored arms, and per arm both the pure-Thompson top-1
    // probability and the ε-greedy propensity (before the hysteresis).
    let epsilon = score_settings
        .exploration_epsilon
        .clamp(0.0, MAX_EXPLORATION_EPSILON);
    let k = scored.len();
    decision.exploration_epsilon = epsilon;
    decision.alternatives = scored
        .iter()
        .map(|s| DecisionAlternative {
            pick: pick_of(&s.candidate),
            score: Some(s.utility),
            rejected: None,
            p_top1: Some(s.p_top1),
            propensity: Some((1.0 - epsilon) * s.p_top1 + epsilon / k as f64),
        })
        .collect();
    rejected(&mut decision);
    decision.confidence = Some(decision_confidence(&decision));
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
        let decision = self.preview(request).await?;
        // Persisted before it is returned, applied or not.
        self.store.put_decision(&decision).await?;
        Ok(decision)
    }

    async fn preview(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
        let arms = self
            .store
            .arms_of_class(&request.signature.arm_key())
            .await?;
        let hints = self.hints.read().unwrap_or_else(|e| e.into_inner()).clone();
        Ok(decide_with(
            Uuid::new_v4(),
            Utc::now(),
            request,
            &arms,
            &hints,
        ))
    }

    /// Feeds the turn's ending back to the arm that served it: a concluded
    /// success raises its mean, a provider error lowers it, and an ambiguous
    /// ending (interrupted, max_turns, budget, user-stopped, …) leaves the
    /// posterior untouched. The arm is keyed exactly as at decision time
    /// (class × provider × model); a decision without a served pair is never
    /// fed, so the Beta(1, 1) prior of an unknown arm is not corrupted.
    async fn observe_turn(
        &self,
        decision: &CognitiveDecision,
        stop_reason: Option<&str>,
        subtype: &str,
        is_error: bool,
        provider_error: bool,
    ) -> anyhow::Result<()> {
        let Some(reward) = turn_reward(stop_reason, subtype, is_error, provider_error) else {
            return Ok(());
        };
        let Some(arm) = decision.served_arm() else {
            return Ok(());
        };
        self.store
            .observe(
                &arm,
                &ArmObservation {
                    reward,
                    success_threshold: TURN_REWARD_THRESHOLD,
                    decay: 1.0,

                    cost_usd: None,
                    latency_ms: None,
                },
            )
            .await?;
        Ok(())
    }
}

/// What the router knows of one model's capability probe (a tool call sent to the
/// endpoint: a real, possibly paid, request).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeMark {
    /// A probe is running: no other pool build sends a second one, and it counts against
    /// its instance's budget. `ticket` identifies the reservation (only its own probe
    /// records over it); `attempts` are the failed tries before it. Older than
    /// [`ProbePolicy::in_flight_stale`] (its task died), it no longer blocks.
    InFlight {
        since: std::time::Instant,
        ticket: u64,
        attempts: u32,
    },
    /// The probe concluded, tools or not (`ModelNoTools` is a conclusion, recorded by the
    /// provider): never sent again while the instance's provider is the same.
    Concluded,
    /// The endpoint refused the model itself (a 4xx other than 408/429: an embeddings,
    /// speech or image model behind `/chat/completions`): a final answer, never sent
    /// again while the provider is the same. The model stays without tools.
    Unusable,
    /// The probe failed without an answer about the model (network, 401, 408, 429, 5xx,
    /// timeout): retried after a backoff that doubles with each attempt.
    Failed {
        at: std::time::Instant,
        attempts: u32,
    },
}

/// How a probe ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Capabilities came back (tools or `ModelNoTools`).
    Concluded,
    /// The model was refused for good (see [`ProbeMark::Unusable`]).
    Unusable,
    /// No answer about the model.
    Failed,
}

impl ProbeOutcome {
    /// The outcome of a probe that returned `error`. Only a refusal about the MODEL is
    /// final: 400, 404, 422 (nexus spells them `HTTP <status>: …` in `InvalidRequest`).
    /// Every other status nexus also files under `InvalidRequest` (402 out of credit, 408,
    /// 409, 410, …) says nothing final about the model and is retried with backoff, like
    /// 401, 429, 5xx and network errors (review of #709, N1).
    pub fn of_error(error: &nexus_claude::agent::ProviderError) -> Self {
        use nexus_claude::agent::ProviderError;
        const FINAL: [&str; 3] = ["HTTP 400", "HTTP 404", "HTTP 422"];
        match error {
            ProviderError::InvalidRequest { detail }
                if FINAL.iter().any(|status| detail.starts_with(status)) =>
            {
                Self::Unusable
            }
            _ => Self::Failed,
        }
    }
}

/// Whether an error means the ACCOUNT is out of credit (HTTP 402). Unlike a model
/// refusal, this is about the account: auto-routing must stop sending to that
/// instance until credit returns, instead of retrying the same send and erroring
/// the chat.
pub fn is_out_of_credit(error: &nexus_claude::agent::ProviderError) -> bool {
    use nexus_claude::agent::ProviderError;
    matches!(
        error,
        ProviderError::InvalidRequest { detail } if detail.starts_with("HTTP 402")
    )
}

/// Capability probes of the routing: (instance, model) to what is known.
pub type ProbeLog = std::collections::HashMap<(String, String), ProbeMark>;

/// The bounds of the routing's capability probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbePolicy {
    /// Probes that may run at once on one instance, whatever the number of pool builds.
    pub per_instance: usize,
    /// How long one probe may take.
    pub timeout: std::time::Duration,
    /// How long a pool build waits for the probes it started, all instances together;
    /// what is still running then is used by the next builds.
    pub wait: std::time::Duration,
    /// Delay before the first retry of a failed probe; doubled at each further failure.
    pub retry_after: std::time::Duration,
    /// The longest delay between two retries.
    pub max_backoff: std::time::Duration,
    /// After how long a probe still marked running is considered abandoned.
    pub in_flight_stale: std::time::Duration,
}

impl ProbePolicy {
    /// The production bounds: three probes at once per instance, ten seconds each, a
    /// build waiting 1.5 s at most, a failure retried after 10 min, 20, 40… up to a day.
    pub const fn standard() -> Self {
        Self {
            per_instance: 3,
            timeout: std::time::Duration::from_secs(10),
            wait: std::time::Duration::from_millis(1_500),
            retry_after: std::time::Duration::from_secs(600),
            max_backoff: std::time::Duration::from_secs(86_400),
            in_flight_stale: std::time::Duration::from_secs(60),
        }
    }

    /// The delay before retrying a probe that failed `attempts` times (≥ 1).
    pub fn backoff(&self, attempts: u32) -> std::time::Duration {
        let factor = 1u32 << attempts.saturating_sub(1).min(16);
        self.retry_after
            .checked_mul(factor)
            .map_or(self.max_backoff, |d| d.min(self.max_backoff))
    }
}

impl Default for ProbePolicy {
    fn default() -> Self {
        Self::standard()
    }
}

/// Which models a pool build may probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeScope {
    /// None: the pool reports what is already known (primary + shadow, a report).
    None,
    /// Every model of every native instance, within the policy's bounds.
    All,
    /// Only these (instance, model) pairs: the models ticked for the conversation.
    Only(Vec<(String, String)>),
}

impl ProbeScope {
    /// Whether `(instance, model)` may be probed.
    pub fn covers(&self, instance: &str, model: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Only(pairs) => pairs.iter().any(|(i, m)| i == instance && m == model),
        }
    }

    /// The scope the routing settings allow: nothing in `primary` + `shadow` (routing off,
    /// nothing learnt or applied: no request is spent on it), every model otherwise.
    pub fn for_settings(mode: ProviderRoutingMode, stage: LearningStage) -> Self {
        if mode == ProviderRoutingMode::Primary && stage == LearningStage::Shadow {
            Self::None
        } else {
            Self::All
        }
    }
}

/// A model reserved for a probe, and the ticket its outcome is recorded with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    /// The model to probe.
    pub model: String,
    /// The reservation's identity (see [`record_probe`]).
    pub ticket: u64,
}

static NEXT_TICKET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn running(mark: &ProbeMark, now: std::time::Instant, policy: &ProbePolicy) -> bool {
    matches!(mark, ProbeMark::InFlight { since, .. } if now.duration_since(*since) < policy.in_flight_stale)
}

/// Reserves, in `candidates` order, the models of `instance` to probe now: as many as
/// the instance has room for (`policy.per_instance` minus the probes already running on
/// it, from any build), skipping a model whose probe concluded, was refused for good, is
/// running, or failed less than its backoff ago. The reserved ones are marked running.
pub fn reserve_probes(
    log: &mut ProbeLog,
    instance: &str,
    candidates: impl IntoIterator<Item = String>,
    now: std::time::Instant,
    policy: &ProbePolicy,
) -> Vec<Reservation> {
    let in_flight = log
        .iter()
        .filter(|((id, _), mark)| id == instance && running(mark, now, policy))
        .count();
    let room = policy.per_instance.saturating_sub(in_flight);
    let mut reserved: Vec<Reservation> = Vec::new();
    for model in candidates {
        if reserved.len() >= room {
            break;
        }
        let key = (instance.to_owned(), model);
        let attempts = match log.get(&key) {
            None => Some(0),
            Some(ProbeMark::Concluded | ProbeMark::Unusable) => None,
            Some(mark @ ProbeMark::InFlight { attempts, .. }) => {
                (!running(mark, now, policy)).then_some(*attempts)
            }
            Some(ProbeMark::Failed { at, attempts }) => {
                (now.duration_since(*at) >= policy.backoff(*attempts)).then_some(*attempts)
            }
        };
        let Some(attempts) = attempts else { continue };
        if reserved.iter().any(|r| r.model == key.1) {
            continue;
        }
        let ticket = NEXT_TICKET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log.insert(
            key.clone(),
            ProbeMark::InFlight {
                since: now,
                ticket,
                attempts,
            },
        );
        reserved.push(Reservation {
            model: key.1,
            ticket,
        });
    }
    reserved
}

/// Records how the probe reserved with `ticket` ended. Only that reservation's entry is
/// written: a probe whose instance was forgotten meanwhile (its provider rebuilt, the
/// model maybe reserved again since) taught the new provider nothing.
pub fn record_probe(
    log: &mut ProbeLog,
    instance: &str,
    model: &str,
    ticket: u64,
    outcome: ProbeOutcome,
    now: std::time::Instant,
) {
    let Some(entry) = log.get_mut(&(instance.to_owned(), model.to_owned())) else {
        return;
    };
    let ProbeMark::InFlight {
        ticket: mine,
        attempts,
        ..
    } = *entry
    else {
        return;
    };
    if mine != ticket {
        return;
    }
    *entry = match outcome {
        ProbeOutcome::Concluded => ProbeMark::Concluded,
        ProbeOutcome::Unusable => ProbeMark::Unusable,
        ProbeOutcome::Failed => ProbeMark::Failed {
            at: now,
            attempts: attempts + 1,
        },
    };
}

/// Whether a probe of `instance` is running (from any build).
pub fn probing(
    log: &ProbeLog,
    instance: &str,
    now: std::time::Instant,
    policy: &ProbePolicy,
) -> bool {
    log.iter()
        .any(|((id, _), mark)| id == instance && running(mark, now, policy))
}

/// Records the models of `instance` the catalogue itself lists without tools as answered
/// (`Concluded`), so that no path reserves them (the pool's and the background probes of
/// a routed conversation alike; review of #709, N2). A running probe is left alone.
pub fn mark_listed_without_tools<'a>(
    log: &mut ProbeLog,
    instance: &str,
    models: impl IntoIterator<Item = &'a String>,
) {
    for model in models {
        let entry = log
            .entry((instance.to_owned(), model.clone()))
            .or_insert(ProbeMark::Concluded);
        if !matches!(entry, ProbeMark::InFlight { .. }) {
            *entry = ProbeMark::Concluded;
        }
    }
}

/// Forgets every probe of `instance`: its provider was rebuilt (the stored record
/// changed), so what the old one learnt is gone with it.
pub fn forget_instance(log: &mut ProbeLog, instance: &str) {
    log.retain(|(id, _), _| id != instance);
}

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
    /// What is known of each native model's capability probe: a probe is sent once
    /// per model (a failure is retried later, a conclusion never), never twice at once.
    pub probed: Arc<std::sync::Mutex<ProbeLog>>,
    /// Quota of every instance (QUOTA): until when an account refuses (its own report, a dated
    /// refusal, a person's entry), and the fixed cooldown of a refusal that names no date. One
    /// registry for the pool filter, the fallback after a failure and the send path.
    pub quota: Arc<super::super::quota::QuotaRegistry>,
    /// The bounds of those probes.
    pub probe_policy: ProbePolicy,
    /// Where decisions and arms live, when the standard wiring built this; it is
    /// how a session's decision is linked and closed. `None` for a test decider.
    pub store: Option<Arc<dyn RoutingArmStore>>,
    /// Where the conversation segments and boundaries live (the episodic memory),
    /// when a graph-backed store is installed. `None` keeps them in the session
    /// only (the in-memory default).
    pub segment_store: Option<Arc<dyn SegmentStore>>,
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
            quota: Default::default(),
            probe_policy: ProbePolicy::standard(),
            store: Some(store),
            segment_store: None,
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
            quota: Default::default(),
            probe_policy: ProbePolicy::standard(),
            store: None,
            segment_store: None,
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

    /// This router reading its quota from `registry` (the standard wiring shares one with the
    /// chat runtime and the API; the default is private and in memory).
    pub fn with_quota(mut self, registry: Arc<super::super::quota::QuotaRegistry>) -> Self {
        self.quota = registry;
        self
    }

    /// Attaches a graph-backed segment store, so a conversation's segments and
    /// boundaries outlive the process. Without it the default is in-memory only.
    pub fn with_segment_store(mut self, store: Arc<dyn SegmentStore>) -> Self {
        self.segment_store = Some(store);
        self
    }

    /// Marks `instance` out of credit: held out for [`super::super::quota::FALLBACK_COOLDOWN`]
    /// (no date exists for a top-up), unless a reset is already known.
    pub fn mark_out_of_credit(&self, instance: &str) {
        self.quota.mark_out_of_credit(instance);
    }

    /// Whether `instance` is currently out of credit and must be excluded from
    /// the routing pool.
    pub fn out_of_credit(&self, instance: &str) -> bool {
        self.quota.exhausted(instance).is_some()
    }

    /// Replaces the alias hints.
    pub fn set_hints(&self, hints: PriorHints) {
        *self.hints.write().unwrap_or_else(|e| e.into_inner()) = hints;
    }
}

/// PYSR-1: the numeric core of the utility, pinned on its own so a replay can use
/// it without rebuilding `ModelFacts` — and so any change to the formula is a
/// deliberate act, visible here, rather than a silent drift between two copies.
#[cfg(test)]
mod utility_core_tests {
    use super::*;

    #[test]
    fn the_utility_is_the_posterior_mean_minus_the_weighted_penalties() {
        let settings = RoutingSettings {
            cost_weight: 0.3,
            latency_weight: 0.2,
            quota_weight: 0.1,
            ..Default::default()
        };
        // 3/(3+1) - 0.3·0.5 - 0.2·0.25 - 0.1·0.75 = 0.75 - 0.15 - 0.05 - 0.075
        let utility = utility_of(3.0, 1.0, 0.5, 0.25, 0.75, &settings);
        assert!((utility - 0.475).abs() < 1e-12, "got {utility}");
    }

    #[test]
    fn a_zero_weight_setting_leaves_only_the_posterior_mean() {
        let settings = RoutingSettings {
            cost_weight: 0.0,
            latency_weight: 0.0,
            quota_weight: 0.0,
            ..Default::default()
        };
        assert!((utility_of(1.0, 3.0, 9.0, 9.0, 9.0, &settings) - 0.25).abs() < 1e-12);
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
            quota_exhausted: None,
            quota_utilization: None,
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

    /// QUOTA — the test that fails without the change: a pool {A, B} where A has a quota used up
    /// until T. Auto never picks A, and the alternative A says `quota_exhausted` with T (before the
    /// change it said `unhealthy`, or A stayed a candidate when only the 429 text was known). At
    /// T + 1 s, on the same registry and the same clock, A is eligible again: nobody did anything.
    #[tokio::test]
    async fn a_provider_whose_quota_is_used_up_is_excluded_until_its_reset_then_back() {
        use crate::chat::provider::limits::ManualClock;
        use crate::chat::provider::quota::QuotaRegistry;
        use nexus_claude::agent::{QuotaSnapshot, QuotaStatus, QuotaWindow};

        let clock = Arc::new(ManualClock::at("2026-10-21T10:00:00Z"));
        let quota = QuotaRegistry::new().with_clock(clock.clone());
        let reset_ms = 1_792_836_000_000_u64; // 2026-10-24T10:00:00Z
        quota.observe(
            "deepseek",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "seven_day".into(),
                utilization: Some(1.0),
                resets_at_ms: Some(reset_ms),
                status: QuotaStatus::Exhausted,
            }]),
        );
        let pool_at = || {
            let mut pool = vec![facts("deepseek", "v4", 1.0), facts("glm", "big", 2.0)];
            candidates::apply_quota(&mut pool, &quota);
            pool
        };
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = pool_at();
        let (decision, _) = decide(&req).await;
        assert_eq!(
            decision.chosen.as_ref().map(|p| p.provider_id.as_str()),
            Some("glm")
        );
        let rejected = decision
            .alternatives
            .iter()
            .find(|a| a.pick.provider_id == "deepseek")
            .and_then(|a| a.rejected.clone())
            .expect("the exhausted provider is listed as rejected");
        assert_eq!(rejected.code(), "quota_exhausted");
        assert_eq!(
            rejected,
            RejectReason::QuotaExhausted {
                resets_at_ms: reset_ms,
                window: Some("seven_day".into())
            }
        );

        // A second before the reset it is still out; one second after, it is back.
        clock.advance(std::time::Duration::from_secs(3 * 86_400 - 1));
        req.pool = pool_at();
        assert!(req.pool[0].quota_exhausted.is_some());
        clock.advance(std::time::Duration::from_secs(2));
        req.pool = pool_at();
        assert!(req.pool[0].quota_exhausted.is_none());
        let (decision, _) = decide(&req).await;
        let deepseek = decision
            .alternatives
            .iter()
            .find(|a| a.pick.provider_id == "deepseek")
            .expect("listed");
        assert_eq!(deepseek.rejected, None, "eligible again at the reset");
    }

    /// An account that is out is not an unhealthy one: `healthy` is not what says it, and the
    /// quota reason is tested before health (the reason must carry the instant it is back). When
    /// every candidate is out the decision picks nobody (the caller keeps the session's engine)
    /// instead of failing.
    #[test]
    fn the_quota_reason_comes_before_health_and_all_exhausted_picks_nobody() {
        let mut a = facts("deepseek", "v4", 1.0);
        a.healthy = Some(false);
        a.quota_exhausted = Some(candidates::QuotaExhaustion {
            resets_at_ms: 42,
            window: None,
        });
        let mut b = facts("glm", "big", 2.0);
        b.quota_exhausted = Some(candidates::QuotaExhaustion {
            resets_at_ms: 43,
            window: None,
        });
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![a, b];
        let decision = decide_with(
            Uuid::from_u128(1),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        );
        assert_eq!(decision.chosen, None);
        let codes: Vec<_> = decision
            .alternatives
            .iter()
            .filter_map(|a| a.rejected.as_ref().map(RejectReason::code))
            .collect();
        assert_eq!(codes, ["quota_exhausted", "quota_exhausted"]);
    }

    /// A decision with a known chosen pair, made from an empty store: its arm is
    /// not observed yet, which is what the reward half feeds.
    fn routed_decision() -> CognitiveDecision {
        let req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        decide_with(
            Uuid::from_u128(7),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        )
    }

    #[test]
    fn a_decision_logs_p_top1_and_propensity_for_every_scored_arm() {
        let req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        let decision = decide_with(
            Uuid::from_u128(42),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        );
        assert_eq!(decision.exploration_epsilon, 0.0);
        let scored: Vec<_> = decision
            .alternatives
            .iter()
            .filter(|a| a.score.is_some())
            .collect();
        assert_eq!(scored.len(), 2);
        for a in &scored {
            let p = a.p_top1.unwrap();
            assert!((0.0..=1.0).contains(&p));
            // ε = 0 → the propensity is exactly the top-1 probability.
            assert!((a.propensity.unwrap() - p).abs() < 1e-12);
        }
        let sum: f64 = scored.iter().map(|a| a.p_top1.unwrap()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "top-1 probabilities sum to 1, got {sum}"
        );
    }

    #[test]
    fn the_logged_propensity_follows_the_epsilon_greedy_formula() {
        let mut req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        req.settings.exploration_epsilon = 0.2;
        let decision = decide_with(
            Uuid::from_u128(42),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        );
        assert_eq!(decision.exploration_epsilon, 0.2);
        let scored: Vec<_> = decision
            .alternatives
            .iter()
            .filter(|a| a.score.is_some())
            .collect();
        let k = scored.len();
        assert_eq!(k, 2);
        let mut sum = 0.0;
        for a in &scored {
            let p = a.p_top1.unwrap();
            let expected = (1.0 - 0.2) * p + 0.2 / k as f64;
            assert!(
                (a.propensity.unwrap() - expected).abs() < 1e-12,
                "π = (1−ε)·p_top1 + ε/k"
            );
            sum += a.propensity.unwrap();
        }
        assert!((sum - 1.0).abs() < 1e-9, "propensities sum to 1, got {sum}");
    }

    #[tokio::test]
    async fn a_successful_turn_raises_the_served_arms_mean() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        let decision = routed_decision();
        let arm = decision.served_arm().unwrap();
        decider
            .observe_turn(&decision, Some("completed"), "success", false, false)
            .await
            .unwrap();
        let stats = store.arm(&arm).await.unwrap().unwrap();
        assert_eq!((stats.alpha, stats.beta, stats.n), (2.0, 1.0, 1));
        assert!(stats.mean() > 0.5);
    }

    #[tokio::test]
    async fn a_provider_error_lowers_the_served_arms_mean() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        let decision = routed_decision();
        let arm = decision.served_arm().unwrap();
        decider
            .observe_turn(
                &decision,
                Some("error"),
                "error_during_execution",
                true,
                true,
            )
            .await
            .unwrap();
        let stats = store.arm(&arm).await.unwrap().unwrap();
        assert_eq!((stats.alpha, stats.beta, stats.n), (1.0, 2.0, 1));
        assert!(stats.mean() < 0.5);
    }

    #[tokio::test]
    async fn an_interrupted_or_max_turns_turn_leaves_the_posterior_untouched() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        let decision = routed_decision();
        let arm = decision.served_arm().unwrap();
        for (reason, subtype) in [
            ("interrupted", "interrupted"),
            ("max_turns", "error_max_turns"),
        ] {
            decider
                .observe_turn(&decision, Some(reason), subtype, false, false)
                .await
                .unwrap();
        }
        // An error without a provider cause is ambiguous too.
        decider
            .observe_turn(
                &decision,
                Some("error"),
                "error_during_execution",
                true,
                false,
            )
            .await
            .unwrap();
        assert!(
            store.arm(&arm).await.unwrap().is_none(),
            "an ambiguous ending must not create nor move the arm"
        );
    }

    #[tokio::test]
    async fn a_decision_without_a_served_pair_is_never_fed() {
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        let mut decision = routed_decision();
        decision.chosen = None;
        decision.used = None;
        decider
            .observe_turn(&decision, Some("completed"), "success", false, false)
            .await
            .unwrap();
        let class = decision.signature.arm_key();
        assert!(store.arms_of_class(&class).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_preview_decides_like_decide_and_stores_nothing() {
        use crate::chat::provider::cognitive::store::DecisionFilter;
        let req = request(pilot(), ProviderRoutingMode::Full, LearningStage::Auto);
        let store = Arc::new(InMemoryRoutingStore::new());
        let decider = CognitiveDecider::new(store.clone());
        let preview = decider.preview(&req).await.unwrap();
        assert!(store
            .decisions(&DecisionFilter::default())
            .await
            .unwrap()
            .is_empty());
        let decided = decider.decide(&req).await.unwrap();
        // The draw is seeded by the decision id: the pick may differ, never what is
        // eligible nor whether it applies.
        let eligible = |d: &CognitiveDecision| {
            let mut v: Vec<_> = d
                .alternatives
                .iter()
                .map(|a| (a.pick.clone(), a.rejected.is_none()))
                .collect();
            v.sort_by(|a, b| (&a.0.provider_id, &a.0.model).cmp(&(&b.0.provider_id, &b.0.model)));
            v
        };
        assert_eq!(eligible(&preview), eligible(&decided));
        assert_eq!(preview.applied, decided.applied);
        assert!(preview.chosen.is_some());
        assert_eq!(
            store
                .decisions(&DecisionFilter::default())
                .await
                .unwrap()
                .len(),
            1
        );
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
            (ExecutorsOnly, Shadow, false, false),
            (ExecutorsOnly, Advisory, false, true),
            (ExecutorsOnly, Auto, false, true),
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
    async fn a_work_whose_model_cannot_be_chosen_is_decided_and_stored_never_applied() {
        let compaction = TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, Some("po"));
        let mut req = request(compaction, ProviderRoutingMode::Full, LearningStage::Auto);
        req.current = Some(Pick::new("glm", "big"));
        req.not_selectable = Some(COMPACTION_MODEL_NOT_SELECTABLE);
        let (decision, store) = decide(&req).await;
        assert!(decision.chosen.is_some(), "the choice is still made");
        assert!(!decision.applied, "full + auto would apply it otherwise");
        assert!(decision
            .reason
            .starts_with("compaction_model_not_selectable: "));
        assert_eq!(decision.used, Some(Pick::new("glm", "big")));
        assert_eq!(store.decision(decision.id).await.unwrap(), Some(decision));
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
                        decay: 1.0,
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
                        decay: 1.0,
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

    /// Two barely-observed arms of one class whose means are close (2/3 after one run,
    /// 3/5 after two): their Thompson draws are noisy, so the margin on the DRAWS lets the
    /// pair in force change often.
    fn barely_observed(sig: &TaskSignature) -> Vec<ArmStats> {
        let arm = |model: &str, alpha: f64, beta: f64, n: u64| ArmStats {
            n,
            ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", model), alpha, beta)
        };
        vec![arm("a", 2.0, 1.0, 1), arm("b", 3.0, 2.0, 2)]
    }

    /// `turns` consecutive same-class turns, each starting from the pair the previous one
    /// chose; returns how many changed it.
    fn changes_over(turns: u128, steady: bool) -> usize {
        let sig = pilot();
        let arms = barely_observed(&sig);
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![facts("local", "a", 1.0), facts("local", "b", 1.0)];
        req.settings.exploration_epsilon = 0.2;
        req.steady = steady;
        let mut current = Pick::new("local", "a");
        let mut changes = 0;
        for seed in 0..turns {
            req.current = Some(current.clone());
            let d = decide_with(
                Uuid::from_u128(seed),
                Utc::now(),
                &req,
                &arms,
                &PriorHints::new(),
            );
            let chosen = d.chosen.unwrap();
            if chosen != current {
                changes += 1;
                current = chosen;
            }
        }
        changes
    }

    /// R-S3, review of #710 (2): a steady turn (same class right after a change) applies
    /// the hysteresis margin to the posterior MEANS and never explores: on two barely-observed
    /// close arms, a thousand same-class turns never change the pair in force. The same turns
    /// decided on the draws (and with exploration) change it hundreds of times.
    #[test]
    fn steady_turns_never_oscillate_between_two_close_barely_observed_arms() {
        assert_eq!(changes_over(1_000, true), 0);
        let noisy = changes_over(1_000, false);
        assert!(noisy > 100, "the draws alone oscillate: {noisy}");
    }

    /// A steady turn still changes the pair when a challenger is better ON AVERAGE by the
    /// margin; the guards that narrowed a decision are told in its reason.
    #[test]
    fn a_steady_turn_follows_a_clearly_better_mean_and_its_reason_says_it_was_narrowed() {
        let sig = pilot();
        let arms = vec![
            ArmStats {
                n: 20,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "a"), 6.0, 16.0)
            },
            ArmStats {
                n: 20,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "b"), 16.0, 6.0)
            },
        ];
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![facts("local", "a", 1.0), facts("local", "b", 1.0)];
        req.current = Some(Pick::new("local", "a"));
        req.steady = true;
        req.narrowed = vec!["same_class_no_explore", "after_provider_move"];
        let d = decide_with(
            Uuid::from_u128(7),
            Utc::now(),
            &req,
            &arms,
            &PriorHints::new(),
        );
        assert_eq!(d.chosen, Some(Pick::new("local", "b")));
        assert!(
            d.reason
                .ends_with("; narrowed: same_class_no_explore; narrowed: after_provider_move"),
            "{}",
            d.reason
        );
    }

    /// A confirmed conversation-segment boundary is named in the reason, so the
    /// stored decision says why the direction changed (`; segment: boundary(...)@...`).
    #[test]
    fn a_segment_note_is_appended_to_the_reason() {
        let sig = pilot();
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.segment_note = Some("boundary(topic)@12".into());
        let d = decide_with(
            Uuid::from_u128(7),
            Utc::now(),
            &req,
            &[],
            &PriorHints::new(),
        );
        assert!(
            d.reason.ends_with("; segment: boundary(topic)@12"),
            "{}",
            d.reason
        );
    }

    /// Follow-up review of #710 (A): the margin on the means is pinned at 0.1. The pair in
    /// force is `b` (3/5); the challenger `a` (2/3) is better ON AVERAGE by ~0.067, under the
    /// margin: a thousand independent steady turns keep `b`. Red with an argmax on the
    /// means (`gap < 0.0`): `a` is taken whenever it leads the draw.
    #[test]
    fn a_steady_turn_keeps_the_pair_in_force_against_a_mean_gap_under_the_margin() {
        let sig = pilot();
        let arms = barely_observed(&sig);
        let mean = |m: &str| {
            arms.iter()
                .find(|a| a.key.model == m)
                .map(ArmStats::mean)
                .unwrap()
        };
        let gap = mean("a") - mean("b");
        assert!(gap > 0.0 && gap < HYSTERESIS_MARGIN, "{gap}");
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![facts("local", "a", 1.0), facts("local", "b", 1.0)];
        req.settings.exploration_epsilon = 0.2;
        req.steady = true;
        req.current = Some(Pick::new("local", "b"));
        let changes = (0..1_000u128)
            .filter(|seed| {
                let d = decide_with(
                    Uuid::from_u128(*seed),
                    Utc::now(),
                    &req,
                    &arms,
                    &PriorHints::new(),
                );
                d.chosen != Some(Pick::new("local", "b"))
            })
            .count();
        assert_eq!(changes, 0);
    }

    /// L1: the hysteresis margin is confidence-derived once warm. A WARM tight field —
    /// two arms whose means are close, so the confidence is low — widens the margin
    /// past the base 0.1: the pair in force stays against a challenger that is better
    /// ON AVERAGE by ~0.15 (over the base margin, under the doubled one). Red with the
    /// fixed margin: the gap exceeds 0.1 and the challenger takes over.
    #[test]
    fn a_warm_tight_field_widens_the_margin_and_keeps_the_pair_in_force() {
        let sig = pilot();
        let arms = vec![
            // Challenger: mean 0.60.
            ArmStats {
                n: 20,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "a"), 12.0, 8.0)
            },
            // Pair in force: mean 0.45.
            ArmStats {
                n: 20,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "b"), 9.0, 11.0)
            },
        ];
        let mean = |m: &str| {
            arms.iter()
                .find(|a| a.key.model == m)
                .map(ArmStats::mean)
                .unwrap()
        };
        let gap = mean("a") - mean("b");
        assert!(
            gap > HYSTERESIS_MARGIN && gap < 2.0 * HYSTERESIS_MARGIN,
            "{gap}"
        );
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![facts("local", "a", 1.0), facts("local", "b", 1.0)];
        req.settings.cost_weight = 0.0;
        req.settings.latency_weight = 0.0;
        req.steady = true;
        req.current = Some(Pick::new("local", "b"));
        // A seed where the challenger leads on its draw, so the hysteresis is what
        // keeps the incumbent (not a draw ordering that already put it first).
        let id = (0..2_000u128)
            .map(Uuid::from_u128)
            .find(|id| {
                let mut no_current = req.clone();
                no_current.current = None;
                decide_with(*id, Utc::now(), &no_current, &arms, &PriorHints::new()).chosen
                    == Some(Pick::new("local", "a"))
            })
            .expect("a seed where the challenger leads exists");
        let d = decide_with(id, Utc::now(), &req, &arms, &PriorHints::new());
        assert_eq!(d.chosen, Some(Pick::new("local", "b")), "{}", d.reason);
        assert!(d.reason.contains("kept"), "{}", d.reason);
    }

    /// L1: below the warmup the confidence is never trusted — the base margin decides,
    /// so a barely-observed challenger that beats the pair in force by MORE than the
    /// base margin still takes over (a cold pair is never frozen by a widened margin).
    #[test]
    fn a_barely_observed_pair_stays_on_the_base_margin() {
        let sig = pilot();
        let arms = vec![
            ArmStats {
                n: 1,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "a"), 12.0, 8.0)
            },
            ArmStats {
                n: 2,
                ..ArmStats::fresh(ArmKey::new(sig.arm_key(), "local", "b"), 9.0, 11.0)
            },
        ];
        let mut req = request(sig, ProviderRoutingMode::Full, LearningStage::Auto);
        req.pool = vec![facts("local", "a", 1.0), facts("local", "b", 1.0)];
        req.settings.cost_weight = 0.0;
        req.settings.latency_weight = 0.0;
        req.steady = true;
        req.current = Some(Pick::new("local", "b"));
        let switched = (0..500u128).any(|s| {
            let d = decide_with(
                Uuid::from_u128(s),
                Utc::now(),
                &req,
                &arms,
                &PriorHints::new(),
            );
            d.chosen == Some(Pick::new("local", "a"))
        });
        assert!(
            switched,
            "a cold pair must still switch past the base margin"
        );
    }

    fn models(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    fn names(reserved: &[Reservation]) -> Vec<String> {
        reserved.iter().map(|r| r.model.clone()).collect()
    }

    fn end(log: &mut ProbeLog, r: &Reservation, outcome: ProbeOutcome, at: std::time::Instant) {
        record_probe(log, "i", &r.model, r.ticket, outcome, at);
    }

    /// Review of #709 (C2): the budget is the probes RUNNING on the instance, whatever
    /// the number of builds asking.
    #[test]
    fn an_instance_never_has_more_probes_running_than_its_budget() {
        let policy = ProbePolicy::standard();
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        let all = models(&["d", "a", "b", "c", "e"]);
        let first = reserve_probes(&mut log, "i", all.clone(), t0, &policy);
        assert_eq!(
            names(&first),
            ["d", "a", "b"],
            "the default first, then in order"
        );
        // While they run, another build reserves nothing: the instance is full.
        assert!(reserve_probes(&mut log, "i", all.clone(), t0, &policy).is_empty());
        // Another instance has its own budget.
        assert_eq!(
            reserve_probes(&mut log, "j", all.clone(), t0, &policy).len(),
            3
        );
        // One ends: one seat frees up.
        end(&mut log, &first[0], ProbeOutcome::Concluded, t0);
        assert_eq!(
            names(&reserve_probes(&mut log, "i", all, t0, &policy)),
            ["c"]
        );
        assert!(probing(&log, "i", t0, &policy));
    }

    #[test]
    fn a_conclusion_or_a_refusal_is_never_probed_again_and_a_failure_waits_its_retry() {
        let policy = ProbePolicy::standard();
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        let three = models(&["no-tools", "embedding", "flaky"]);
        let r = reserve_probes(&mut log, "i", three.clone(), t0, &policy);
        assert_eq!(names(&r), three);
        end(&mut log, &r[0], ProbeOutcome::Concluded, t0);
        end(&mut log, &r[1], ProbeOutcome::Unusable, t0);
        end(&mut log, &r[2], ProbeOutcome::Failed, t0);
        let soon = t0 + std::time::Duration::from_secs(60);
        assert!(reserve_probes(&mut log, "i", three.clone(), soon, &policy).is_empty());
        let later = t0 + policy.retry_after;
        assert_eq!(
            names(&reserve_probes(
                &mut log,
                "i",
                three.clone(),
                later,
                &policy
            )),
            ["flaky"],
            "ModelNoTools and a refused model are answers: only the failure is retried"
        );
        // A rebuilt instance starts over.
        forget_instance(&mut log, "i");
        assert_eq!(
            reserve_probes(&mut log, "i", three, later, &policy).len(),
            3
        );
    }

    /// Review of #709 (C3): each failure doubles the delay, up to the cap.
    #[test]
    fn each_failure_doubles_the_delay_before_the_next_try() {
        let policy = ProbePolicy::standard();
        assert_eq!(policy.backoff(1), policy.retry_after);
        assert_eq!(policy.backoff(2), policy.retry_after * 2);
        assert_eq!(policy.backoff(3), policy.retry_after * 4);
        assert_eq!(policy.backoff(40), policy.max_backoff);
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        let r = reserve_probes(&mut log, "i", models(&["m"]), t0, &policy);
        end(&mut log, &r[0], ProbeOutcome::Failed, t0);
        let t1 = t0 + policy.retry_after;
        let r = reserve_probes(&mut log, "i", models(&["m"]), t1, &policy);
        assert_eq!(r.len(), 1, "first retry after retry_after");
        end(&mut log, &r[0], ProbeOutcome::Failed, t1);
        assert!(
            reserve_probes(
                &mut log,
                "i",
                models(&["m"]),
                t1 + policy.retry_after,
                &policy
            )
            .is_empty(),
            "the second failure waits twice as long"
        );
        assert_eq!(
            reserve_probes(
                &mut log,
                "i",
                models(&["m"]),
                t1 + policy.retry_after * 2,
                &policy
            )
            .len(),
            1
        );
    }

    /// Review of #709 (C3): a 4xx about the model is final; 408, 429, 5xx, network are not.
    #[test]
    fn a_refusal_of_the_model_is_final_and_a_transient_error_is_not() {
        use nexus_claude::agent::ProviderError;
        let refused = |detail: &str| ProviderError::InvalidRequest {
            detail: detail.into(),
        };
        assert_eq!(
            ProbeOutcome::of_error(&refused("HTTP 404: model or endpoint not found (x)")),
            ProbeOutcome::Unusable
        );
        assert_eq!(
            ProbeOutcome::of_error(&refused("HTTP 400: not a chat model")),
            ProbeOutcome::Unusable
        );
        assert_eq!(
            ProbeOutcome::of_error(&refused("HTTP 408: request timeout")),
            ProbeOutcome::Failed
        );
        // N1: an account out of credit is not a verdict on the model.
        for status in [
            "HTTP 402: insufficient credits",
            "HTTP 409: conflict",
            "HTTP 410: gone",
        ] {
            assert_eq!(
                ProbeOutcome::of_error(&refused(status)),
                ProbeOutcome::Failed,
                "{status}"
            );
        }
        assert_eq!(
            ProbeOutcome::of_error(&refused("HTTP 422: unprocessable model")),
            ProbeOutcome::Unusable
        );
        for transient in [
            ProviderError::RateLimited {
                retry_after_ms: None,
                resets_at_ms: None,
            },
            ProviderError::Overloaded,
            ProviderError::Unauthorized,
            ProviderError::unreachable("connection refused"),
        ] {
            assert_eq!(ProbeOutcome::of_error(&transient), ProbeOutcome::Failed);
        }
    }

    #[test]
    fn an_out_of_credit_error_is_classified_and_cooldown_excludes_then_resumes() {
        use nexus_claude::agent::ProviderError;
        let refused = |detail: &str| ProviderError::InvalidRequest {
            detail: detail.to_string(),
        };
        // 402 is about the ACCOUNT, not the model.
        assert!(is_out_of_credit(&refused("HTTP 402: insufficient credits")));
        assert!(!is_out_of_credit(&refused("HTTP 400: not a chat model")));
        assert!(!is_out_of_credit(&ProviderError::RateLimited {
            retry_after_ms: None,
            resets_at_ms: None,
        }));

        let clock = std::sync::Arc::new(crate::chat::provider::limits::ManualClock::at(
            "2026-10-21T10:00:00Z",
        ));
        let routing = CognitiveRouting::new(std::sync::Arc::new(InMemoryRoutingStore::new()))
            .with_quota(std::sync::Arc::new(
                crate::chat::provider::quota::QuotaRegistry::new().with_clock(clock.clone()),
            ));
        let cooldown = crate::chat::provider::quota::FALLBACK_COOLDOWN;
        assert!(!routing.out_of_credit("deepseek"));
        routing.mark_out_of_credit("deepseek");
        assert!(routing.out_of_credit("deepseek"));
        clock.advance(cooldown - std::time::Duration::from_secs(1));
        assert!(routing.out_of_credit("deepseek"));
        clock.advance(std::time::Duration::from_secs(2));
        assert!(
            !routing.out_of_credit("deepseek"),
            "back after the cooldown"
        );
        // Another instance is unaffected.
        assert!(!routing.out_of_credit("claude-code"));
    }

    /// Review of #709 (N2): a model the catalogue lists without tools is never reserved,
    /// by any path.
    #[test]
    fn a_model_the_catalogue_lists_without_tools_is_never_reserved() {
        let policy = ProbePolicy::standard();
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        let listed = models(&["plain"]);
        mark_listed_without_tools(&mut log, "i", &listed);
        assert_eq!(
            names(&reserve_probes(
                &mut log,
                "i",
                models(&["plain", "m"]),
                t0,
                &policy
            )),
            ["m"]
        );
        // Even after a failure would have been retried.
        let later = t0 + policy.max_backoff;
        assert!(reserve_probes(&mut log, "i", listed.clone(), later, &policy).is_empty());
        // A probe already running keeps its reservation (its own outcome records).
        let mut running = ProbeLog::new();
        let r = reserve_probes(&mut running, "i", listed.clone(), t0, &policy);
        mark_listed_without_tools(&mut running, "i", &listed);
        assert!(matches!(
            running[&("i".to_owned(), "plain".to_owned())],
            ProbeMark::InFlight { ticket, .. } if ticket == r[0].ticket
        ));
    }

    #[test]
    fn an_abandoned_probe_stops_blocking_after_its_staleness() {
        let policy = ProbePolicy::standard();
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        assert_eq!(
            reserve_probes(&mut log, "i", models(&["m"]), t0, &policy).len(),
            1
        );
        let stale = t0 + policy.in_flight_stale;
        assert!(!probing(&log, "i", stale, &policy));
        assert_eq!(
            reserve_probes(&mut log, "i", models(&["m"]), stale, &policy).len(),
            1
        );
    }

    /// Review of #709 (C5): the old provider's probe ends after a rebuild and a new
    /// reservation of the same model: it does not write over the new one.
    #[test]
    fn a_probe_of_a_forgotten_generation_never_records_over_the_new_one() {
        let policy = ProbePolicy::standard();
        let mut log = ProbeLog::new();
        let t0 = std::time::Instant::now();
        let old = reserve_probes(&mut log, "i", models(&["m"]), t0, &policy).remove(0);
        forget_instance(&mut log, "i");
        record_probe(&mut log, "i", "m", old.ticket, ProbeOutcome::Concluded, t0);
        assert!(log.is_empty(), "nothing recorded for a forgotten instance");
        let t1 = t0 + std::time::Duration::from_millis(1);
        let new = reserve_probes(&mut log, "i", models(&["m"]), t1, &policy).remove(0);
        record_probe(&mut log, "i", "m", old.ticket, ProbeOutcome::Concluded, t1);
        assert!(
            matches!(log[&("i".to_owned(), "m".to_owned())], ProbeMark::InFlight { ticket, .. } if ticket == new.ticket),
            "{log:?}"
        );
        // Its own probe records it.
        end(&mut log, &new, ProbeOutcome::Failed, t1);
        assert!(matches!(
            log[&("i".to_owned(), "m".to_owned())],
            ProbeMark::Failed { attempts: 1, .. }
        ));
    }

    #[test]
    fn primary_and_shadow_probe_nothing_and_a_tick_list_only_itself() {
        use LearningStage::*;
        use ProviderRoutingMode::*;
        assert_eq!(ProbeScope::for_settings(Primary, Shadow), ProbeScope::None);
        for (mode, stage) in [(Primary, Auto), (ExecutorsOnly, Shadow), (Full, Advisory)] {
            assert_eq!(ProbeScope::for_settings(mode, stage), ProbeScope::All);
        }
        let only = ProbeScope::Only(vec![("i".into(), "a".into())]);
        assert!(only.covers("i", "a"));
        assert!(!only.covers("i", "b") && !only.covers("j", "a"));
        assert!(!ProbeScope::None.covers("i", "a"));
    }
}
