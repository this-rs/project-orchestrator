//! Routing — trait-based interface for prompt section selection.
//!
//! The `RoutingProvider` trait abstracts how prompt sections and tool groups
//! are selected. This enables swapping the selection strategy without changing
//! the `FsmPromptComposer`.
//!
//! Current implementations:
//! - [`HeuristicRouter`]: reproduces the scaffolding + intent + FSM logic
//!   from `prompt_sections::{select_sections, select_tool_groups}`.
//!
//! Future integration:
//! - `DualTrackRouter` (Neural Route Learning) will implement this trait,
//!   using learned weights from past conversations to optimize section selection.
//!
//! ```text
//! ┌───────────────────────────┐
//! │      RoutingProvider      │
//! │  fn route(&RoutingContext)│
//! │       → RoutingDecision   │
//! └───────────┬───────────────┘
//!             │
//!     ┌───────┴───────────┐
//!     │                   │
//! HeuristicRouter    DualTrackRouter (future)
//! (scaffolding +     (neural weights +
//!  intent rules)      learned preferences)
//! ```

use super::prompt_sections::{
    select_sections, select_tool_groups, BasePromptSection, ComposerContext, PromptSectionId,
    ToolGroupSelectionContext, ToolRefGroupId,
};

// ============================================================================
// RoutingContext — all signals available for routing decisions
// ============================================================================

/// All signals available to the routing system for making section selection
/// and tool group decisions.
///
/// This is a superset of `ComposerContext` and `ToolGroupSelectionContext`,
/// combining structural, behavioral, and intent signals.
#[derive(Debug, Clone, Default)]
pub struct RoutingContext {
    // ── Scaffolding & project state ────────────────────────────────
    /// Current scaffolding level (0–4). 0 = full guidance, 4 = expert.
    pub scaffolding_level: u8,
    /// Whether there is at least one active plan.
    pub has_active_plan: bool,
    /// Whether there are active protocol runs.
    pub has_active_protocol: bool,
    /// Number of tasks across active plans.
    pub task_count: usize,
    /// Whether the project has sibling projects (multi-project workspace).
    pub is_multi_project: bool,

    // ── FSM state ──────────────────────────────────────────────────
    /// Tool names whitelisted by the current FSM state's `available_tools`.
    /// Empty = no FSM restriction.
    pub fsm_available_tools: Vec<String>,

    // ── Intent signals ─────────────────────────────────────────────
    /// The user's current message (for keyword-based intent detection).
    pub user_message: String,
    /// Detected intent from the enrichment pipeline (e.g., "planning", "code", "debug").
    /// `None` if no intent was detected.
    pub detected_intent: Option<String>,
    // ── Future: neural signals ─────────────────────────────────────
    // pub conversation_embedding: Option<Vec<f32>>,
    // pub session_history_summary: Option<String>,
    // pub user_preference_vector: Option<Vec<f32>>,
}

// Default is derived — all fields have natural defaults (0, false, empty, None).

// ============================================================================
// RoutingDecision — the output of a RoutingProvider
// ============================================================================

/// The routing decision: which base sections and tool groups to include,
/// with optional per-section weight and token budget hints.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected base prompt sections, filtered and ordered.
    pub sections: Vec<BasePromptSection>,
    /// Selected tool reference groups.
    pub tool_groups: Vec<ToolRefGroupId>,
    /// Per-section hints (weights and optional token budgets).
    /// Sections not in this map use default weight 1.0.
    pub section_hints: Vec<SectionHint>,
}

/// A hint about a specific prompt section's importance and budget.
///
/// Used by the `FsmPromptComposer` to:
/// - Filter sections (weight = 0.0 → exclude)
/// - Prioritize truncation (lower weight → truncated first)
/// - Allocate token budgets (optional per-section cap)
///
/// ## Future DualTrackRouter integration
///
/// The DualTrackRouter will produce `SectionHint`s with learned weights
/// based on conversation patterns. For example, if a user rarely needs
/// the "Deep Maintenance" section, its weight will be low (0.1–0.2),
/// causing it to be excluded at higher scaffolding levels even though
/// the heuristic would include it.
///
/// ```rust,ignore
/// // Example: DualTrackRouter output for a code-focused conversation
/// vec![
///     SectionHint { section_id: IdentityRole,           weight: 1.0, token_budget: None },
///     SectionHint { section_id: BestPracticesSearch,     weight: 0.9, token_budget: None },
///     SectionHint { section_id: BestPracticesAdvanced,   weight: 0.2, token_budget: Some(500) },
///     SectionHint { section_id: DeepMaintenance,         weight: 0.0, token_budget: None }, // excluded
/// ]
/// ```
#[derive(Debug, Clone)]
pub struct SectionHint {
    /// Which section this hint applies to.
    pub section_id: PromptSectionId,
    /// Weight: 0.0 = exclude, 1.0 = must include.
    /// Values between 0 and 1 indicate relative importance for truncation ordering.
    pub weight: f32,
    /// Optional per-section token budget (in characters, ~4 chars/token).
    /// `None` = no specific budget (uses the global budget).
    pub token_budget: Option<usize>,
}

// ============================================================================
// RoutingProvider — the trait
// ============================================================================

/// Trait for prompt section routing decisions.
///
/// The `FsmPromptComposer` calls `route()` to determine which base sections
/// and tool groups to include in the system prompt.
///
/// ## Implementing a custom router
///
/// ```rust,ignore
/// use crate::chat::routing::{RoutingProvider, RoutingContext, RoutingDecision};
///
/// struct DualTrackRouter {
///     model: NeuralRouteModel,
/// }
///
/// impl RoutingProvider for DualTrackRouter {
///     fn route(&self, ctx: &RoutingContext) -> RoutingDecision {
///         // 1. Encode context into feature vector
///         let features = self.model.encode_context(ctx);
///         // 2. Predict section weights
///         let weights = self.model.predict_weights(&features);
///         // 3. Build decision with learned weights
///         RoutingDecision {
///             sections: filter_sections_by_weights(&weights),
///             tool_groups: predict_tool_groups(&features),
///             section_hints: weights_to_hints(&weights),
///         }
///     }
/// }
/// ```
pub trait RoutingProvider: Send + Sync {
    /// Produce a routing decision given the current context.
    ///
    /// The decision determines which sections and tool groups are included
    /// in the composed prompt.
    fn route(&self, ctx: &RoutingContext) -> RoutingDecision;
}

// ============================================================================
// HeuristicRouter — default implementation reproducing current behavior
// ============================================================================

/// Heuristic-based router that reproduces the existing scaffolding + intent
/// + FSM selection logic.
///
/// This is the permanent fallback: even when the DualTrackRouter is available,
/// the HeuristicRouter provides the baseline that the neural router modulates.
///
/// Selection rules:
/// - Base sections: filtered by `ActivationCondition` (scaffolding level,
///   active plan/protocol, task count)
/// - Tool groups: `Core` + `Knowledge` always, then intent keywords,
///   FSM whitelist, scaffolding level, multi-project context
pub struct HeuristicRouter;

impl RoutingProvider for HeuristicRouter {
    fn route(&self, ctx: &RoutingContext) -> RoutingDecision {
        // ── Select base sections via existing logic ────────────────────
        let composer_ctx = ComposerContext {
            scaffolding_level: ctx.scaffolding_level,
            has_active_plan: ctx.has_active_plan,
            has_active_protocol: ctx.has_active_protocol,
            task_count: ctx.task_count,
        };
        let sections = select_sections(&composer_ctx);

        // ── Select tool groups via existing logic ──────────────────────
        let tool_group_ctx = ToolGroupSelectionContext {
            scaffolding_level: ctx.scaffolding_level,
            has_active_protocol: ctx.has_active_protocol,
            is_multi_project: ctx.is_multi_project,
            fsm_available_tools: ctx.fsm_available_tools.clone(),
            user_intent_keywords: vec![ctx.user_message.clone()],
        };
        let tool_groups = select_tool_groups(&tool_group_ctx);

        // HeuristicRouter produces no per-section hints — all included
        // sections have equal weight (1.0).
        RoutingDecision {
            sections,
            tool_groups,
            section_hints: vec![],
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_router_l0_full_includes_all_sections() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
            is_multi_project: true,
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // At L0 with everything active, all 16 sections should be included
        assert_eq!(decision.sections.len(), PromptSectionId::ALL.len());
        // All 7 tool groups should be included
        assert_eq!(decision.tool_groups.len(), ToolRefGroupId::ALL.len());
        // No hints from heuristic router
        assert!(decision.section_hints.is_empty());
    }

    #[test]
    fn test_heuristic_router_l4_reduces_sections() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 4,
            has_active_plan: false,
            has_active_protocol: false,
            task_count: 0,
            is_multi_project: false,
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // At L4 with nothing active, far fewer sections
        assert!(decision.sections.len() < PromptSectionId::ALL.len());
        // Should still include IdentityRole (always) and MegatoolsSyntax (always)
        let ids: Vec<_> = decision.sections.iter().map(|s| s.id).collect();
        assert!(ids.contains(&PromptSectionId::IdentityRole));
        assert!(ids.contains(&PromptSectionId::MegatoolsSyntax));

        // Tool groups: only Core + Knowledge (no intent, no protocol, no multi-project)
        assert_eq!(decision.tool_groups.len(), 2);
        assert!(decision.tool_groups.contains(&ToolRefGroupId::Core));
        assert!(decision.tool_groups.contains(&ToolRefGroupId::Knowledge));
    }

    #[test]
    fn test_heuristic_router_code_intent_includes_code_group() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 3,
            user_message: "find the function that handles file imports".to_string(),
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // "function" and "file" and "import" are code keywords
        assert!(decision
            .tool_groups
            .contains(&ToolRefGroupId::CodeExploration));
    }

    #[test]
    fn test_heuristic_router_fsm_whitelist_filters_tool_groups() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 2,
            has_active_protocol: true,
            fsm_available_tools: vec!["project".into(), "plan".into(), "note".into()],
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // FSM whitelist only includes Core tools (project, plan) + Knowledge (note)
        assert!(decision.tool_groups.contains(&ToolRefGroupId::Core));
        assert!(decision.tool_groups.contains(&ToolRefGroupId::Knowledge));
        // Should NOT include CodeExploration (no "code" in whitelist)
        assert!(!decision
            .tool_groups
            .contains(&ToolRefGroupId::CodeExploration));
    }

    #[test]
    fn test_heuristic_matches_select_sections_directly() {
        // Verify HeuristicRouter produces the same sections as calling select_sections directly
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 2,
            has_active_plan: true,
            has_active_protocol: false,
            task_count: 5,
            is_multi_project: false,
            user_message: "how do I create a plan?".to_string(),
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // Same context via ComposerContext
        let composer_ctx = ComposerContext {
            scaffolding_level: 2,
            has_active_plan: true,
            has_active_protocol: false,
            task_count: 5,
        };
        let direct_sections = select_sections(&composer_ctx);

        // Should produce identical section lists
        let router_ids: Vec<_> = decision.sections.iter().map(|s| s.id).collect();
        let direct_ids: Vec<_> = direct_sections.iter().map(|s| s.id).collect();
        assert_eq!(router_ids, direct_ids);
    }

    #[test]
    fn test_routing_context_default() {
        let ctx = RoutingContext::default();
        assert_eq!(ctx.scaffolding_level, 0);
        assert!(!ctx.has_active_plan);
        assert!(!ctx.has_active_protocol);
        assert_eq!(ctx.task_count, 0);
        assert!(!ctx.is_multi_project);
        assert!(ctx.fsm_available_tools.is_empty());
        assert!(ctx.user_message.is_empty());
        assert!(ctx.detected_intent.is_none());
    }

    #[test]
    fn test_section_hint_properties() {
        let hint = SectionHint {
            section_id: PromptSectionId::DeepMaintenance,
            weight: 0.0,
            token_budget: None,
        };
        assert_eq!(hint.weight, 0.0);
        assert!(hint.token_budget.is_none());

        let hint_with_budget = SectionHint {
            section_id: PromptSectionId::BestPracticesAdvanced,
            weight: 0.5,
            token_budget: Some(2000),
        };
        assert_eq!(hint_with_budget.token_budget, Some(2000));
    }
}
