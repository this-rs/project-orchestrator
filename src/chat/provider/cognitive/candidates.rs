//! Candidates: the (instance, model) pairs a request may be routed to.
//!
//! Hard constraints only; scoring is not here. A pair that fails a constraint
//! is rejected with a typed, displayable [`RejectReason`], so a decision can
//! say why an alternative was not eligible. Prices come from nexus
//! (`ModelInfo.pricing`, decision A1): this module owns no price table.
//!
//! The filter applies to automatic slots only. A choice the caller named
//! (request, task, persona) is never filtered nor substituted (A16, A20).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use nexus_claude::agent::{Capabilities, CostBasis, ModelPrice, SandboxLevel};
use serde::{Deserialize, Serialize};

use super::signature::TaskSignature;
use crate::chat::provider::resolver::{is_remote_instance, CLAUDE_CODE};

/// Output tokens assumed for the cost estimate of one median turn.
const TURN_OUTPUT_TOKENS: f64 = 2_000.0;

/// What is known about one model of one instance, at decision time.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFacts {
    /// Instance identifier.
    pub provider_id: String,
    /// Model identifier.
    pub model: String,
    /// The model calls tools.
    pub supports_tools: bool,
    /// The model accepts images.
    pub supports_images: bool,
    /// Context window in tokens; `None` when unknown (never assumed).
    pub context_window: Option<u64>,
    /// Price per million tokens, from nexus; `None` when unknown.
    pub price: Option<ModelPrice>,
    /// How the cost is accounted.
    pub cost_basis: CostBasis,
    /// Live health: `Some(false)` is unhealthy, `None` was not measured.
    pub healthy: Option<bool>,
    /// The project consented to this instance (A28).
    pub allowed_for_project: bool,
    /// The tools run inside a sandbox.
    pub sandboxed: bool,
}

impl ModelFacts {
    /// Facts from the capabilities nexus reports for the model.
    pub fn from_capabilities(
        provider_id: impl Into<String>,
        model: impl Into<String>,
        caps: &Capabilities,
        price: Option<ModelPrice>,
        healthy: Option<bool>,
        allowed_for_project: bool,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            model: model.into(),
            supports_tools: caps.tools,
            supports_images: caps.images,
            context_window: caps.context_window.map(|w| w.value),
            price,
            cost_basis: caps.cost,
            healthy,
            allowed_for_project,
            sandboxed: caps.sandbox != SandboxLevel::None,
        }
    }

    /// Estimated USD of one median turn for `signature`; `None` when the price
    /// is unknown (never zero) or the cost is not marginal money.
    pub fn estimated_turn_cost_usd(&self, signature: &TaskSignature) -> Option<f64> {
        if matches!(self.cost_basis, CostBasis::Free | CostBasis::Subscription) {
            return None;
        }
        let price = self.price?;
        Some(
            (signature.context_need_tokens as f64 * price.input_per_mtok
                + TURN_OUTPUT_TOKENS * price.output_per_mtok)
                / 1_000_000.0,
        )
    }
}

/// Why a pair is not eligible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum RejectReason {
    /// A machine is chosen by a person, never by the router.
    Remote,
    /// The project did not consent to this instance's origin.
    NotAllowed,
    /// The instance is unhealthy right now.
    Unhealthy,
    /// The request needs tools and the model cannot call them.
    NoTools,
    /// The window is smaller than the request needs, or unknown.
    ContextTooSmall {
        /// Tokens the request needs.
        need: u64,
        /// Tokens the model holds; `None` when unknown.
        have: Option<u64>,
    },
    /// The input carries images and the model cannot read them.
    NoImages,
    /// A median turn costs more than the marginal budget left.
    OverBudget,
    /// Trust mode on a third party without a sandbox (A35).
    TrustWithoutSandbox,
}

impl RejectReason {
    /// Stable code, for logs and the UI.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Remote => "remote",
            Self::NotAllowed => "not_allowed",
            Self::Unhealthy => "unhealthy",
            Self::NoTools => "no_tools",
            Self::ContextTooSmall { .. } => "context_too_small",
            Self::NoImages => "no_images",
            Self::OverBudget => "over_budget",
            Self::TrustWithoutSandbox => "trust_without_sandbox",
        }
    }
}

/// A pair that was not eligible, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejected {
    /// The pair.
    pub candidate: ModelFacts,
    /// First constraint it failed.
    pub reason: RejectReason,
}

/// Who named the pair a slot would hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// Request, task or persona named it: never filtered, never substituted.
    Explicit,
    /// A rule, the default or the router chose it: subject to the constraints.
    Automatic,
}

/// Outcome of the filter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Filtered {
    /// Eligible pairs, in input order.
    pub eligible: Vec<ModelFacts>,
    /// Pairs rejected, with their first failing constraint.
    pub rejected: Vec<Rejected>,
}

/// Why the first failing constraint, if any. Order: ownership of the choice
/// (remote, consent, health) before capability before money.
fn check(signature: &TaskSignature, facts: &ModelFacts, trust: bool) -> Option<RejectReason> {
    if is_remote_instance(&facts.provider_id) {
        return Some(RejectReason::Remote);
    }
    if !facts.allowed_for_project {
        return Some(RejectReason::NotAllowed);
    }
    if facts.healthy == Some(false) {
        return Some(RejectReason::Unhealthy);
    }
    if signature.needs_tools && !facts.supports_tools {
        return Some(RejectReason::NoTools);
    }
    match facts.context_window {
        Some(window) if window >= signature.context_need_tokens => {}
        have => {
            return Some(RejectReason::ContextTooSmall {
                need: signature.context_need_tokens,
                have,
            })
        }
    }
    if signature.needs_images && !facts.supports_images {
        return Some(RejectReason::NoImages);
    }
    let cost = facts.estimated_turn_cost_usd(signature);
    if matches!((signature.budget_remaining_usd, cost), (Some(budget), Some(cost)) if cost > budget)
    {
        return Some(RejectReason::OverBudget);
    }
    if trust && !facts.sandboxed && facts.provider_id != CLAUDE_CODE {
        return Some(RejectReason::TrustWithoutSandbox);
    }
    None
}

/// Filters `pool` for `signature`. `trust` is whether the session would run in
/// trust mode. An explicit slot is outside the filter: `None`.
pub fn apply(
    slot: Slot,
    signature: &TaskSignature,
    pool: &[ModelFacts],
    trust: bool,
) -> Option<Filtered> {
    if slot == Slot::Explicit {
        return None;
    }
    let mut filtered = Filtered::default();
    for facts in pool {
        match check(signature, facts, trust) {
            None => filtered.eligible.push(facts.clone()),
            Some(reason) => filtered.rejected.push(Rejected {
                candidate: facts.clone(),
                reason,
            }),
        }
    }
    Some(filtered)
}

/// Health of an instance, remembered for a short while so a decision never
/// probes N instances. One probe per instance per window.
pub struct HealthCache {
    ttl: Duration,
    entries: Mutex<HashMap<String, (Instant, bool)>>,
}

impl HealthCache {
    /// A cache whose entries live `ttl`.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The default window: 60 s.
    pub fn standard() -> Self {
        Self::new(Duration::from_secs(60))
    }

    /// The remembered health of `provider_id`, `None` when absent or expired.
    pub fn get(&self, provider_id: &str, now: Instant) -> Option<bool> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries
            .get(provider_id)
            .filter(|(at, _)| now.saturating_duration_since(*at) < self.ttl)
            .map(|(_, healthy)| *healthy)
    }

    /// Remembers a measure.
    pub fn put(&self, provider_id: &str, healthy: bool, now: Instant) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(provider_id.to_owned(), (now, healthy));
    }

    /// The remembered health, else `probe()`'s answer, remembered. `probe` runs
    /// at most once per instance per window.
    pub fn get_or_probe(
        &self,
        provider_id: &str,
        now: Instant,
        probe: impl FnOnce() -> bool,
    ) -> bool {
        if let Some(healthy) = self.get(provider_id, now) {
            return healthy;
        }
        let healthy = probe();
        self.put(provider_id, healthy, now);
        healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::signature::{ContextHints, TaskClass};

    fn good() -> ModelFacts {
        ModelFacts {
            provider_id: "deepseek".into(),
            model: "v4".into(),
            supports_tools: true,
            supports_images: true,
            context_window: Some(128_000),
            price: Some(ModelPrice {
                input_per_mtok: 1.0,
                output_per_mtok: 2.0,
                cache_read_per_mtok: None,
                cache_write_per_mtok: None,
            }),
            cost_basis: CostBasis::Priced,
            healthy: Some(true),
            allowed_for_project: true,
            sandboxed: false,
        }
    }

    fn signature() -> TaskSignature {
        TaskSignature::utility(TaskClass::UtilityFeatureGraph, 20_000, None)
    }

    fn tooled() -> TaskSignature {
        TaskSignature::from_chat_request("hello", false, None, ContextHints::default())
    }

    fn reason(signature: &TaskSignature, facts: ModelFacts, trust: bool) -> Option<RejectReason> {
        let out = apply(Slot::Automatic, signature, &[facts], trust).unwrap();
        out.rejected.into_iter().next().map(|r| r.reason)
    }

    #[test]
    fn a_pair_that_meets_every_constraint_is_eligible() {
        let out = apply(Slot::Automatic, &tooled(), &[good()], false).unwrap();
        assert_eq!(out.eligible, vec![good()]);
        assert!(out.rejected.is_empty());
    }

    #[test]
    fn a_remote_machine_is_never_a_candidate() {
        let mut facts = good();
        facts.provider_id = "claude-code@box".into();
        assert_eq!(reason(&tooled(), facts, false), Some(RejectReason::Remote));
    }

    #[test]
    fn a_pair_without_the_projects_consent_is_not_allowed() {
        let mut facts = good();
        facts.allowed_for_project = false;
        assert_eq!(
            reason(&tooled(), facts, false),
            Some(RejectReason::NotAllowed)
        );
    }

    #[test]
    fn an_unhealthy_instance_is_out_and_an_unmeasured_one_is_in() {
        let mut facts = good();
        facts.healthy = Some(false);
        assert_eq!(
            reason(&tooled(), facts.clone(), false),
            Some(RejectReason::Unhealthy)
        );
        facts.healthy = None;
        assert_eq!(reason(&tooled(), facts, false), None);
    }

    #[test]
    fn a_request_that_needs_tools_refuses_a_model_that_cannot_call_them() {
        let mut facts = good();
        facts.supports_tools = false;
        assert_eq!(
            reason(&tooled(), facts.clone(), false),
            Some(RejectReason::NoTools)
        );
        // A utility call needs none.
        assert_eq!(reason(&signature(), facts, false), None);
    }

    #[test]
    fn a_window_smaller_than_the_need_or_unknown_is_rejected() {
        let mut facts = good();
        facts.context_window = Some(4_000);
        let need = signature().context_need_tokens;
        assert_eq!(
            reason(&signature(), facts.clone(), false),
            Some(RejectReason::ContextTooSmall {
                need,
                have: Some(4_000)
            })
        );
        facts.context_window = None;
        assert_eq!(
            reason(&signature(), facts, false),
            Some(RejectReason::ContextTooSmall { need, have: None })
        );
    }

    #[test]
    fn images_need_a_model_that_reads_them() {
        let mut with_images = tooled();
        with_images.needs_images = true;
        let mut facts = good();
        facts.supports_images = false;
        assert_eq!(
            reason(&with_images, facts.clone(), false),
            Some(RejectReason::NoImages)
        );
        assert_eq!(reason(&tooled(), facts, false), None);
    }

    #[test]
    fn a_turn_dearer_than_the_budget_left_is_rejected_and_an_unknown_price_passes() {
        let mut signature = signature();
        signature.budget_remaining_usd = Some(0.000_001);
        assert_eq!(
            reason(&signature, good(), false),
            Some(RejectReason::OverBudget)
        );
        // Unknown price: never zero, never a reason to refuse.
        let mut unpriced = good();
        unpriced.price = None;
        assert_eq!(reason(&signature, unpriced, false), None);
        // A free or subscription model costs no marginal money.
        let mut free = good();
        free.cost_basis = CostBasis::Free;
        assert_eq!(reason(&signature, free, false), None);
    }

    #[test]
    fn trust_mode_needs_a_sandbox_unless_it_is_claude_code() {
        assert_eq!(
            reason(&tooled(), good(), true),
            Some(RejectReason::TrustWithoutSandbox)
        );
        let mut sandboxed = good();
        sandboxed.sandboxed = true;
        assert_eq!(reason(&tooled(), sandboxed, true), None);
        let mut claude = good();
        claude.provider_id = CLAUDE_CODE.into();
        assert_eq!(reason(&tooled(), claude, true), None);
    }

    #[test]
    fn an_explicit_choice_never_enters_the_filter() {
        let mut facts = good();
        facts.allowed_for_project = false;
        facts.supports_tools = false;
        assert_eq!(apply(Slot::Explicit, &tooled(), &[facts], false), None);
    }

    #[test]
    fn the_first_failing_constraint_is_the_one_reported() {
        let mut facts = good();
        facts.allowed_for_project = false;
        facts.healthy = Some(false);
        facts.supports_tools = false;
        assert_eq!(
            reason(&tooled(), facts, false),
            Some(RejectReason::NotAllowed)
        );
    }

    #[test]
    fn health_is_probed_once_per_instance_per_window() {
        let cache = HealthCache::new(Duration::from_secs(60));
        let t0 = Instant::now();
        let mut probes = 0;
        for _ in 0..5 {
            assert!(cache.get_or_probe("a", t0, || {
                probes += 1;
                true
            }));
        }
        assert_eq!(probes, 1);
        // Another instance is its own entry; an expired entry is probed again.
        assert!(!cache.get_or_probe("b", t0, || false));
        let later = t0 + Duration::from_secs(61);
        assert_eq!(cache.get("a", later), None);
        assert!(!cache.get_or_probe("a", later, || false));
    }

    #[test]
    fn the_estimate_is_none_without_a_price_and_grows_with_the_context() {
        let small = TaskSignature::utility(TaskClass::UtilityCompaction, 10_000, None);
        let big = TaskSignature::utility(TaskClass::UtilityCompaction, 100_000, None);
        let facts = good();
        assert!(
            facts.estimated_turn_cost_usd(&small).unwrap()
                < facts.estimated_turn_cost_usd(&big).unwrap()
        );
        let mut unpriced = good();
        unpriced.price = None;
        assert_eq!(unpriced.estimated_turn_cost_usd(&small), None);
    }
}
