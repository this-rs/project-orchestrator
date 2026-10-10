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
use crate::chat::provider::resolver::is_remote_instance;

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
    /// Why the window is unknown, when its source can say (Claude Code: the live
    /// catalog). `None` with a known window, or when nothing names the cause.
    pub window_unknown: Option<UnknownWindow>,
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
            window_unknown: None,
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
    /// The window is unknown and its source says why (Claude Code: the live
    /// catalog is offline, or lists no window for the model). Never assumed.
    WindowUnknown {
        /// Tokens the request needs.
        need: u64,
        /// Why the window is unknown.
        why: UnknownWindow,
    },
    /// The input carries images and the model cannot read them.
    NoImages,
    /// A median turn costs more than the marginal budget left.
    OverBudget,
    /// Trust mode on a third party without a sandbox (A35). No longer produced: decision
    /// ebd2b7e7 (2026-10-07) opens `Trust` on every provider except a remote machine without
    /// `allow_trust`, and a remote machine is never a candidate ([`Self::Remote`]). Kept so
    /// the decisions stored before still read.
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
            Self::WindowUnknown { .. } => "window_unknown",
            Self::NoImages => "no_images",
            Self::OverBudget => "over_budget",
            Self::TrustWithoutSandbox => "trust_without_sandbox",
        }
    }
}

/// Why a model's window is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownWindow {
    /// The live Models API was not reached (no credential, unreachable, nothing
    /// fetched yet): the static fallback carries no window.
    CatalogOffline,
    /// The live catalog does not list this model, or lists it without
    /// `max_input_tokens`.
    NotInCatalog,
}

impl std::fmt::Display for UnknownWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CatalogOffline => "window unknown: catalog offline",
            Self::NotInCatalog => "window unknown: not reported by the live catalog",
        })
    }
}

/// The Claude Code models the router may choose from: one per model of the
/// live Anthropic catalog, with the window the Models API reports for it
/// (`max_input_tokens`) and the other capabilities the Claude Code provider
/// declares (`caps_of`: tools, images, cost basis, sandbox).
///
/// A window is never invented: offline, or for a model the listing gives none,
/// the window is `None` and [`UnknownWindow`] says why, so the filter rejects the
/// model with that reason. The price is nexus's when it has one (`price_of`),
/// else unknown (A1: no price table here).
pub fn claude_code_facts(
    provider_id: &str,
    catalog: &crate::chat::model_catalog::CatalogSnapshot,
    caps_of: impl Fn(&str) -> Capabilities,
    price_of: impl Fn(&str) -> Option<ModelPrice>,
    healthy: Option<bool>,
    allowed_for_project: bool,
) -> Vec<ModelFacts> {
    catalog
        .models
        .iter()
        .map(|model| {
            let caps = caps_of(&model.id);
            let mut facts = ModelFacts::from_capabilities(
                provider_id,
                model.id.clone(),
                &caps,
                price_of(&model.id),
                healthy,
                allowed_for_project,
            );
            // The catalog's window first; the provider's own (configured) one
            // only when the catalog has none.
            facts.context_window = model.max_input_tokens.or(facts.context_window);
            if facts.context_window.is_none() {
                facts.window_unknown = Some(if catalog.live {
                    UnknownWindow::NotInCatalog
                } else {
                    UnknownWindow::CatalogOffline
                });
            }
            facts
        })
        .collect()
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
fn check(signature: &TaskSignature, facts: &ModelFacts) -> Option<RejectReason> {
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
    match (facts.context_window, facts.window_unknown) {
        (Some(window), _) if window >= signature.context_need_tokens => {}
        (None, Some(why)) => {
            return Some(RejectReason::WindowUnknown {
                need: signature.context_need_tokens,
                why,
            })
        }
        (have, _) => {
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
    // `Trust` gates nothing here: the rule (decision ebd2b7e7, which replaces A35) refuses it
    // only on a remote machine without `allow_trust` (`ChatManager::authorize_provider_use`),
    // and a remote machine never gets this far. The sandbox level informs, it never filters.
    None
}

/// Filters `pool` for `signature`. An explicit slot is outside the filter: `None`.
/// Whether the session runs in `Trust` filters nothing (see
/// [`RejectReason::TrustWithoutSandbox`]).
pub fn apply(slot: Slot, signature: &TaskSignature, pool: &[ModelFacts]) -> Option<Filtered> {
    if slot == Slot::Explicit {
        return None;
    }
    let mut filtered = Filtered::default();
    for facts in pool {
        match check(signature, facts) {
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
    use crate::chat::provider::resolver::CLAUDE_CODE;

    fn good() -> ModelFacts {
        ModelFacts {
            provider_id: "deepseek".into(),
            model: "v4".into(),
            supports_tools: true,
            supports_images: true,
            context_window: Some(128_000),
            window_unknown: None,
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

    fn reason(signature: &TaskSignature, facts: ModelFacts) -> Option<RejectReason> {
        let out = apply(Slot::Automatic, signature, &[facts]).unwrap();
        out.rejected.into_iter().next().map(|r| r.reason)
    }

    #[test]
    fn a_pair_that_meets_every_constraint_is_eligible() {
        let out = apply(Slot::Automatic, &tooled(), &[good()]).unwrap();
        assert_eq!(out.eligible, vec![good()]);
        assert!(out.rejected.is_empty());
    }

    #[test]
    fn a_remote_machine_is_never_a_candidate() {
        let mut facts = good();
        facts.provider_id = "claude-code@box".into();
        assert_eq!(reason(&tooled(), facts), Some(RejectReason::Remote));
    }

    #[test]
    fn a_pair_without_the_projects_consent_is_not_allowed() {
        let mut facts = good();
        facts.allowed_for_project = false;
        assert_eq!(reason(&tooled(), facts), Some(RejectReason::NotAllowed));
    }

    #[test]
    fn an_unhealthy_instance_is_out_and_an_unmeasured_one_is_in() {
        let mut facts = good();
        facts.healthy = Some(false);
        assert_eq!(
            reason(&tooled(), facts.clone()),
            Some(RejectReason::Unhealthy)
        );
        facts.healthy = None;
        assert_eq!(reason(&tooled(), facts), None);
    }

    #[test]
    fn a_request_that_needs_tools_refuses_a_model_that_cannot_call_them() {
        let mut facts = good();
        facts.supports_tools = false;
        assert_eq!(
            reason(&tooled(), facts.clone()),
            Some(RejectReason::NoTools)
        );
        // A utility call needs none.
        assert_eq!(reason(&signature(), facts), None);
    }

    #[test]
    fn a_window_smaller_than_the_need_or_unknown_is_rejected() {
        let mut facts = good();
        facts.context_window = Some(4_000);
        let need = signature().context_need_tokens;
        assert_eq!(
            reason(&signature(), facts.clone()),
            Some(RejectReason::ContextTooSmall {
                need,
                have: Some(4_000)
            })
        );
        facts.context_window = None;
        assert_eq!(
            reason(&signature(), facts),
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
            reason(&with_images, facts.clone()),
            Some(RejectReason::NoImages)
        );
        assert_eq!(reason(&tooled(), facts), None);
    }

    #[test]
    fn a_turn_dearer_than_the_budget_left_is_rejected_and_an_unknown_price_passes() {
        let mut signature = signature();
        signature.budget_remaining_usd = Some(0.000_001);
        assert_eq!(reason(&signature, good()), Some(RejectReason::OverBudget));
        // Unknown price: never zero, never a reason to refuse.
        let mut unpriced = good();
        unpriced.price = None;
        assert_eq!(reason(&signature, unpriced), None);
        // A free or subscription model costs no marginal money.
        let mut free = good();
        free.cost_basis = CostBasis::Free;
        assert_eq!(reason(&signature, free), None);
    }

    /// Decision ebd2b7e7: `Trust` opens on every provider except a remote machine without
    /// `allow_trust`, the rule `authorize_provider_use` applies. The filter takes no trust
    /// flag at all: an unsandboxed third party stays a candidate, and a remote machine is
    /// out whatever its record says (a machine is chosen by a person).
    #[test]
    fn trust_rejects_no_provider_the_open_path_accepts() {
        assert_eq!(reason(&tooled(), good()), None);
        let mut claude = good();
        claude.provider_id = CLAUDE_CODE.into();
        assert_eq!(reason(&tooled(), claude), None);
        let mut remote = good();
        remote.provider_id = "claude-code@box".into();
        assert_eq!(reason(&tooled(), remote), Some(RejectReason::Remote));
    }

    #[test]
    fn an_explicit_choice_never_enters_the_filter() {
        let mut facts = good();
        facts.allowed_for_project = false;
        facts.supports_tools = false;
        assert_eq!(apply(Slot::Explicit, &tooled(), &[facts]), None);
    }

    #[test]
    fn the_first_failing_constraint_is_the_one_reported() {
        let mut facts = good();
        facts.allowed_for_project = false;
        facts.healthy = Some(false);
        facts.supports_tools = false;
        assert_eq!(reason(&tooled(), facts), Some(RejectReason::NotAllowed));
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

    /// Claude Code as the router sees it: windows from the live Anthropic catalog
    /// (B-R3). The capabilities are the real Claude Code provider's (nexus).
    mod claude_code_window {
        use super::*;
        use crate::chat::model_catalog::{live_snapshot_from_listing, ModelCatalogCache};
        use nexus_claude::providers::claude_code::ClaudeCodeConfig;

        fn caps(model: &str) -> Capabilities {
            ClaudeCodeConfig::default().capabilities(Some(model))
        }

        /// A Models API page: Opus 5.5 with a 1M window, Haiku 4.5 (dated
        /// snapshot) with 200k, one model listed without a window.
        fn live() -> crate::chat::model_catalog::CatalogSnapshot {
            live_snapshot_from_listing(serde_json::json!({
                "data": [
                    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5",
                     "max_input_tokens": 1_000_000, "max_tokens": 128_000},
                    {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5",
                     "max_input_tokens": 200_000, "max_tokens": 64_000},
                    {"id": "claude-zeta-9-9", "display_name": "Claude Zeta 9.9"},
                ],
                "has_more": false,
                "last_id": "claude-zeta-9-9",
            }))
        }

        fn facts(
            provider: &str,
            catalog: &crate::chat::model_catalog::CatalogSnapshot,
        ) -> Vec<ModelFacts> {
            claude_code_facts(provider, catalog, caps, |_| None, Some(true), true)
        }

        /// A chat turn with tools and images that needs `need` tokens.
        fn turn(need: u64) -> TaskSignature {
            let mut signature = TaskSignature::from_chat_request(
                "refactor this",
                true,
                None,
                ContextHints::default(),
            );
            signature.context_need_tokens = need;
            signature
        }

        fn rejection(out: &Filtered, model: &str) -> Option<RejectReason> {
            out.rejected
                .iter()
                .find(|r| r.candidate.model == model)
                .map(|r| r.reason.clone())
        }

        #[test]
        fn a_live_window_makes_claude_code_eligible_for_a_task_that_fits() {
            let pool = facts(CLAUDE_CODE, &live());
            let opus = pool.iter().find(|f| f.model == "claude-opus-5-5").unwrap();
            assert_eq!(opus.context_window, Some(1_000_000));
            assert_eq!(opus.window_unknown, None);
            assert!(opus.supports_tools && opus.supports_images);
            assert_eq!(opus.price, None, "no nexus price: unknown, never zero");

            let out = apply(Slot::Automatic, &turn(300_000), &pool).unwrap();
            let eligible: Vec<_> = out.eligible.iter().map(|f| f.model.as_str()).collect();
            assert_eq!(eligible, vec!["claude-opus-5-5"], "{:?}", out.rejected);
            assert!(out.eligible.iter().all(|f| f.provider_id == CLAUDE_CODE));
            // Haiku 4.5 holds 200k: too small for this task, and says so with numbers.
            assert_eq!(
                rejection(&out, "claude-haiku-4-5"),
                Some(RejectReason::ContextTooSmall {
                    need: 300_000,
                    have: Some(200_000)
                })
            );
            // A smaller task fits both.
            let out = apply(Slot::Automatic, &turn(150_000), &pool).unwrap();
            let eligible: Vec<_> = out.eligible.iter().map(|f| f.model.as_str()).collect();
            assert!(
                eligible.contains(&"claude-opus-5-5") && eligible.contains(&"claude-haiku-4-5")
            );
        }

        #[test]
        fn a_model_the_live_catalog_gives_no_window_is_rejected_by_name() {
            let out = apply(Slot::Automatic, &turn(10_000), &facts(CLAUDE_CODE, &live())).unwrap();
            for model in ["claude-zeta-9-9", "claude-sonnet-4-6"] {
                assert_eq!(
                    rejection(&out, model),
                    Some(RejectReason::WindowUnknown {
                        need: 10_000,
                        why: UnknownWindow::NotInCatalog
                    }),
                    "{model}"
                );
            }
        }

        #[tokio::test]
        async fn an_offline_catalog_leaves_every_claude_code_model_ineligible_by_name() {
            let offline = ModelCatalogCache::new(None).snapshot().await;
            assert!(!offline.live);
            let pool = facts(CLAUDE_CODE, &offline);
            assert!(
                !pool.is_empty(),
                "the models stay listed, with their reason"
            );
            let out = apply(Slot::Automatic, &turn(10_000), &pool).unwrap();
            assert!(out.eligible.is_empty());
            let reason = RejectReason::WindowUnknown {
                need: 10_000,
                why: UnknownWindow::CatalogOffline,
            };
            assert!(out.rejected.iter().all(|r| r.reason == reason));
            assert_eq!(reason.code(), "window_unknown");
            assert_eq!(
                UnknownWindow::CatalogOffline.to_string(),
                "window unknown: catalog offline"
            );
            assert_eq!(
                serde_json::to_value(&reason).unwrap(),
                serde_json::json!({"reason": "window_unknown", "need": 10_000, "why": "catalog_offline"})
            );
        }

        #[test]
        fn a_remote_claude_code_stays_out_and_consent_still_rules() {
            let out = apply(
                Slot::Automatic,
                &turn(10_000),
                &facts("claude-code@box", &live()),
            )
            .unwrap();
            assert!(out.eligible.is_empty());
            assert!(out
                .rejected
                .iter()
                .all(|r| r.reason == RejectReason::Remote));

            let refused =
                claude_code_facts(CLAUDE_CODE, &live(), caps, |_| None, Some(true), false);
            let out = apply(Slot::Automatic, &turn(10_000), &refused).unwrap();
            assert!(out.eligible.is_empty());
            assert!(out
                .rejected
                .iter()
                .all(|r| r.reason == RejectReason::NotAllowed));

            let down = claude_code_facts(CLAUDE_CODE, &live(), caps, |_| None, Some(false), true);
            let out = apply(Slot::Automatic, &turn(10_000), &down).unwrap();
            assert!(out
                .rejected
                .iter()
                .all(|r| r.reason == RejectReason::Unhealthy));
        }

        #[test]
        fn an_explicit_claude_code_choice_is_untouched_and_a_nexus_price_is_kept() {
            let offline_like = facts(CLAUDE_CODE, &live());
            assert_eq!(apply(Slot::Explicit, &turn(10_000), &offline_like), None);
            let price = ModelPrice {
                input_per_mtok: 4.0,
                output_per_mtok: 20.0,
                cache_read_per_mtok: None,
                cache_write_per_mtok: None,
            };
            let priced = claude_code_facts(
                CLAUDE_CODE,
                &live(),
                caps,
                |m| (m == "claude-opus-5-5").then_some(price),
                Some(true),
                true,
            );
            let opus = priced
                .iter()
                .find(|f| f.model == "claude-opus-5-5")
                .unwrap();
            assert_eq!(opus.price, Some(price));
        }
    }
}
