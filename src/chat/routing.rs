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

use serde::Serialize;

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

    // ── External MCP servers ────────────────────────────────────────
    /// Whether external MCP servers are connected (triggers External tool group).
    pub external_tools_available: bool,

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
    // ── Neural signals (for DualTrackRouter) ───────────────────────
    /// Pre-computed embedding of the user's message (from EmbeddingProvider).
    /// Computed async by the caller, passed here for sync routing.
    /// `None` if embeddings are unavailable → DualTrackRouter falls back to heuristics.
    pub message_embedding: Option<Vec<f32>>,
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
            external_tools_available: ctx.external_tools_available,
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
// RoutingDecisionRecord — trajectory tracking for neural feedback
// ============================================================================

/// A serializable record of a routing decision, suitable for the
/// `TrajectoryCollector` neural feedback loop.
///
/// Captures which sections and tool groups were selected, with what confidence,
/// so that future sessions can learn from past routing decisions.
#[derive(Debug, Clone, Serialize)]
pub struct RoutingDecisionRecord {
    /// Which sections were included (by ID).
    pub selected_sections: Vec<String>,
    /// Which tool groups were included (by ID).
    pub selected_tool_groups: Vec<String>,
    /// Per-section weights from the router (empty for HeuristicRouter).
    pub section_weights: Vec<(String, f32)>,
    /// Scaffolding level at decision time.
    pub scaffolding_level: u8,
    /// Whether embeddings were used for this decision.
    pub used_embeddings: bool,
    /// Detected intent (if any).
    pub detected_intent: Option<String>,
}

impl RoutingDecision {
    /// Convert this routing decision into a trajectory-compatible record.
    ///
    /// The `ctx` provides the scaffolding level and intent metadata that
    /// contextualizes the decision for the neural feedback loop.
    pub fn to_trajectory_record(&self, ctx: &RoutingContext) -> RoutingDecisionRecord {
        RoutingDecisionRecord {
            selected_sections: self.sections.iter().map(|s| s.id.to_string()).collect(),
            selected_tool_groups: self
                .tool_groups
                .iter()
                .map(|g| format!("{:?}", g))
                .collect(),
            section_weights: self
                .section_hints
                .iter()
                .map(|h| (h.section_id.to_string(), h.weight))
                .collect(),
            scaffolding_level: ctx.scaffolding_level,
            used_embeddings: ctx.message_embedding.is_some(),
            detected_intent: ctx.detected_intent.clone(),
        }
    }

    /// Emit this routing decision to the TrajectoryCollector as a `DecisionRecord`.
    ///
    /// Fire-and-forget: never blocks, drops silently if channel is full.
    pub fn emit_to_trajectory(
        &self,
        ctx: &RoutingContext,
        session_id: &str,
        collector: &neural_routing_runtime::TrajectoryCollector,
    ) {
        let record = self.to_trajectory_record(ctx);
        let params = serde_json::to_value(&record).unwrap_or_default();

        collector.record_decision(neural_routing_runtime::DecisionRecord {
            session_id: session_id.to_string(),
            context_embedding: vec![], // Will be computed by VectorBuilder
            action_type: "routing.select_sections".to_string(),
            action_params: params,
            alternatives_count: PromptSectionId::ALL.len(),
            chosen_index: 0,
            confidence: if self.section_hints.is_empty() {
                0.5 // Heuristic-only → moderate confidence
            } else {
                // Average of section weights as confidence proxy
                let sum: f32 = self.section_hints.iter().map(|h| h.weight).sum();
                (sum / self.section_hints.len() as f32) as f64
            },
            tool_usages: vec![],
            touched_entities: vec![],
            timestamp_ms: 0, // Collector fills this
            query_embedding: ctx.message_embedding.clone().unwrap_or_default(),
            node_features: vec![],
            protocol_run_id: None,
            protocol_state: None,
        });
    }
}

// ============================================================================
// DualTrackRouter — embedding-enhanced routing
// ============================================================================

/// Section centroid: a semantic description embedded as a vector, associated
/// with a `PromptSectionId`.
#[derive(Debug, Clone)]
struct SectionCentroid {
    section_id: PromptSectionId,
    embedding: Vec<f32>,
}

/// Semantic descriptions for each prompt section, used to compute centroids.
/// These are short, keyword-rich descriptions optimized for embedding similarity.
fn section_semantic_descriptions() -> Vec<(PromptSectionId, &'static str)> {
    vec![
        (PromptSectionId::IdentityRole, "agent identity role MCP tools project orchestrator"),
        (PromptSectionId::MegatoolsSyntax, "mega-tools API syntax action parameters calling convention"),
        (PromptSectionId::DataModel, "data model entities hierarchy project plan task step decision constraint note"),
        (PromptSectionId::TreeSitterSync, "tree-sitter code sync parse source files functions structs imports"),
        (PromptSectionId::GitWorkflow, "git workflow branch commit push merge atomic changes version control"),
        (PromptSectionId::TaskExecutionProtocol, "task execution protocol warm-up preparation implementation closure verification"),
        (PromptSectionId::PlanningProtocol, "planning protocol analyze create plan decompose tasks steps constraints milestones"),
        (PromptSectionId::StatusManagement, "status management transitions pending in_progress completed blocked failed"),
        (PromptSectionId::BestPracticesImpact, "impact analysis dependencies file references call graph architecture GDS communities"),
        (PromptSectionId::BestPracticesSearch, "search strategy MCP-first code exploration find references semantic search"),
        (PromptSectionId::BestPracticesKnowledge, "knowledge capture notes gotcha pattern guideline RFC decisions documentation"),
        (PromptSectionId::BestPracticesAdvanced, "advanced protocols knowledge fabric workspace inheritance process detection wave dispatch"),
        (PromptSectionId::BestPracticesPersonas, "personas episodes structural analysis memory horizons intent search scaffolding"),
        (PromptSectionId::DeepMaintenance, "deep maintenance synapses energy staleness skills anchoring consolidation stagnation"),
        (PromptSectionId::PlanExecutionAutomation, "plan execution automation autonomous run triggers delegation waves"),
        (PromptSectionId::DelegationAwareness, "delegation delegate sub-agent task parallel actionable autonomous spawn"),
        (PromptSectionId::ToolReference, "tool reference mega-tools full API documentation parameters actions"),
    ]
}

/// DualTrackRouter combines heuristic section selection with embedding-based
/// cosine similarity scoring.
///
/// Architecture:
/// 1. Heuristic pass: `HeuristicRouter` selects candidate sections (same as before)
/// 2. Embedding pass: cosine similarity between user message embedding and
///    pre-computed section centroids produces a relevance score per section
/// 3. Fusion: scores are combined with configurable weights to produce
///    `SectionHint`s that modulate the heuristic baseline
///
/// Falls back to pure heuristic routing when `message_embedding` is `None`.
pub struct DualTrackRouter {
    /// Pre-computed embeddings for each section's semantic description.
    centroids: Vec<SectionCentroid>,
    /// Weight for the heuristic score (0.0–1.0). Default: 0.6.
    heuristic_weight: f32,
    /// Weight for the embedding score (0.0–1.0). Default: 0.4.
    embedding_weight: f32,
    /// Minimum fused score to keep a section (below → weight 0.0). Default: 0.15.
    min_score_threshold: f32,
}

impl DualTrackRouter {
    /// Create a new DualTrackRouter with pre-computed centroids.
    ///
    /// Call [`DualTrackRouter::build()`] for async construction with an
    /// `EmbeddingProvider`.
    pub fn new(
        centroids: Vec<(PromptSectionId, Vec<f32>)>,
        heuristic_weight: f32,
        embedding_weight: f32,
    ) -> Self {
        Self {
            centroids: centroids
                .into_iter()
                .map(|(id, emb)| SectionCentroid {
                    section_id: id,
                    embedding: emb,
                })
                .collect(),
            heuristic_weight,
            embedding_weight,
            min_score_threshold: 0.15,
        }
    }

    /// Build a DualTrackRouter by embedding section descriptions via the
    /// provided `EmbeddingProvider`.
    ///
    /// This is async because it calls `embed_batch()`. Should be called once
    /// at startup or when the embedding model changes.
    pub async fn build(
        provider: &dyn crate::embeddings::EmbeddingProvider,
        heuristic_weight: f32,
        embedding_weight: f32,
    ) -> Result<Self, anyhow::Error> {
        let descriptions = section_semantic_descriptions();
        let texts: Vec<String> = descriptions
            .iter()
            .map(|(_, desc)| desc.to_string())
            .collect();

        let embeddings = provider.embed_batch(&texts).await?;

        let centroids: Vec<SectionCentroid> = descriptions
            .iter()
            .zip(embeddings)
            .map(|((id, _), emb)| SectionCentroid {
                section_id: *id,
                embedding: emb,
            })
            .collect();

        Ok(Self {
            centroids,
            heuristic_weight,
            embedding_weight,
            min_score_threshold: 0.15,
        })
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Get the embedding score for a given section ID.
    fn embedding_score(&self, section_id: PromptSectionId, message_embedding: &[f32]) -> f32 {
        self.centroids
            .iter()
            .find(|c| c.section_id == section_id)
            .map(|c| Self::cosine_similarity(&c.embedding, message_embedding))
            .unwrap_or(0.0)
            // Normalize from [-1, 1] to [0, 1]
            .mul_add(0.5, 0.5)
    }
}

impl RoutingProvider for DualTrackRouter {
    fn route(&self, ctx: &RoutingContext) -> RoutingDecision {
        // ── Step 1: Get heuristic baseline ────────────────────────────
        let heuristic = HeuristicRouter;
        let baseline = heuristic.route(ctx);

        // No embedding available → fall back to pure heuristic
        let message_emb = match &ctx.message_embedding {
            Some(emb) if !emb.is_empty() => emb,
            _ => return baseline,
        };

        // ── Step 2: Score each heuristic-selected section ─────────────
        let mut section_hints = Vec::new();
        let mut sections = Vec::new();

        for section in &baseline.sections {
            // Heuristic says "include" → base score 1.0
            let h_score: f32 = 1.0;
            let e_score = self.embedding_score(section.id, message_emb);

            // Fused score: weighted combination
            let fused = self.heuristic_weight * h_score + self.embedding_weight * e_score;

            if fused >= self.min_score_threshold {
                sections.push(section.clone());
                section_hints.push(SectionHint {
                    section_id: section.id,
                    weight: fused.min(1.0),
                    token_budget: None,
                });
            }
            // If fused < threshold → section dropped (weight effectively 0)
        }

        RoutingDecision {
            sections,
            tool_groups: baseline.tool_groups,
            section_hints,
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
            external_tools_available: true,
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // At L0 with everything active, all 17 sections should be included
        assert_eq!(decision.sections.len(), PromptSectionId::ALL.len());
        // All 8 tool groups should be included
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

    // ── DualTrackRouter tests ─────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = DualTrackRouter::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = DualTrackRouter::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = DualTrackRouter::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = DualTrackRouter::cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = DualTrackRouter::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_dual_track_no_embedding_falls_back_to_heuristic() {
        let centroids = vec![
            (PromptSectionId::IdentityRole, vec![1.0, 0.0, 0.0]),
            (PromptSectionId::DataModel, vec![0.0, 1.0, 0.0]),
        ];
        let router = DualTrackRouter::new(centroids, 0.6, 0.4);

        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 5,
            message_embedding: None, // No embedding → fallback
            ..Default::default()
        };

        let decision = router.route(&ctx);
        let heuristic_decision = HeuristicRouter.route(&ctx);

        // Should produce identical results to heuristic
        assert_eq!(decision.sections.len(), heuristic_decision.sections.len());
        assert!(decision.section_hints.is_empty());
    }

    #[test]
    fn test_dual_track_with_embedding_produces_hints() {
        let centroids = vec![
            (PromptSectionId::IdentityRole, vec![1.0, 0.0, 0.0]),
            (PromptSectionId::DataModel, vec![0.0, 1.0, 0.0]),
            (PromptSectionId::GitWorkflow, vec![0.0, 0.0, 1.0]),
        ];
        let router = DualTrackRouter::new(centroids, 0.6, 0.4);

        // Message embedding close to DataModel centroid
        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 5,
            message_embedding: Some(vec![0.1, 0.95, 0.05]),
            ..Default::default()
        };

        let decision = router.route(&ctx);

        // Should produce section_hints (non-empty)
        assert!(!decision.section_hints.is_empty());

        // DataModel should have the highest weight among hints that have centroids
        let data_model_hint = decision
            .section_hints
            .iter()
            .find(|h| h.section_id == PromptSectionId::DataModel);
        let git_hint = decision
            .section_hints
            .iter()
            .find(|h| h.section_id == PromptSectionId::GitWorkflow);

        if let (Some(dm), Some(gw)) = (data_model_hint, git_hint) {
            assert!(
                dm.weight > gw.weight,
                "DataModel ({:.3}) should score higher than GitWorkflow ({:.3})",
                dm.weight,
                gw.weight
            );
        }
    }

    #[test]
    fn test_dual_track_empty_embedding_falls_back() {
        let centroids = vec![(PromptSectionId::IdentityRole, vec![1.0, 0.0])];
        let router = DualTrackRouter::new(centroids, 0.6, 0.4);

        let ctx = RoutingContext {
            message_embedding: Some(vec![]), // Empty vec → fallback
            ..Default::default()
        };

        let decision = router.route(&ctx);
        // Should fall back, no hints
        assert!(decision.section_hints.is_empty());
    }

    #[test]
    fn test_dual_track_all_sections_above_threshold() {
        // With heuristic_weight=0.6, even a zero embedding score gives 0.6*1.0 + 0.4*0.5 = 0.8
        // (0.5 because cosine=0 → normalized to 0.5)
        // So with default threshold 0.15, no sections should be dropped
        let centroids = vec![(PromptSectionId::IdentityRole, vec![1.0, 0.0])];
        let router = DualTrackRouter::new(centroids, 0.6, 0.4);

        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 5,
            message_embedding: Some(vec![0.5, 0.5]),
            ..Default::default()
        };

        let decision = router.route(&ctx);
        let heuristic_count = HeuristicRouter.route(&ctx).sections.len();
        // Should not drop any sections (all above threshold)
        assert_eq!(decision.sections.len(), heuristic_count);
    }

    #[test]
    fn test_section_semantic_descriptions_covers_all() {
        let descriptions = section_semantic_descriptions();
        let described_ids: Vec<_> = descriptions.iter().map(|(id, _)| *id).collect();
        for id in PromptSectionId::ALL {
            assert!(
                described_ids.contains(id),
                "Missing semantic description for {:?}",
                id
            );
        }
    }

    // ── RoutingDecisionRecord tests ────────────────────────────────

    #[test]
    fn test_to_trajectory_record_heuristic() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 2,
            has_active_plan: true,
            task_count: 5,
            message_embedding: None,
            ..Default::default()
        };
        let decision = router.route(&ctx);
        let record = decision.to_trajectory_record(&ctx);

        assert_eq!(record.scaffolding_level, 2);
        assert!(!record.used_embeddings);
        assert!(!record.selected_sections.is_empty());
        assert!(!record.selected_tool_groups.is_empty());
        assert!(record.section_weights.is_empty()); // HeuristicRouter produces no hints
        assert!(record.detected_intent.is_none());
    }

    #[test]
    fn test_to_trajectory_record_with_embeddings() {
        let centroids = vec![
            (PromptSectionId::IdentityRole, vec![1.0, 0.0]),
            (PromptSectionId::DataModel, vec![0.0, 1.0]),
        ];
        let router = DualTrackRouter::new(centroids, 0.6, 0.4);
        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 5,
            message_embedding: Some(vec![0.5, 0.5]),
            detected_intent: Some("planning".to_string()),
            ..Default::default()
        };
        let decision = router.route(&ctx);
        let record = decision.to_trajectory_record(&ctx);

        assert!(record.used_embeddings);
        assert!(!record.section_weights.is_empty());
        assert_eq!(record.detected_intent, Some("planning".to_string()));
    }

    #[test]
    fn test_trajectory_record_serializes() {
        let record = RoutingDecisionRecord {
            selected_sections: vec!["identity_role".into(), "data_model".into()],
            selected_tool_groups: vec!["Core".into()],
            section_weights: vec![("identity_role".into(), 0.9)],
            scaffolding_level: 3,
            used_embeddings: true,
            detected_intent: Some("code".into()),
        };
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["scaffolding_level"], 3);
        assert_eq!(json["used_embeddings"], true);
        assert_eq!(json["selected_sections"].as_array().unwrap().len(), 2);
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

    // ── Compact variant routing tests ───────────────────────────────

    #[test]
    fn test_heuristic_router_l0_uses_full_content() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 0,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
            ..Default::default()
        };

        let decision = router.route(&ctx);
        let advanced = decision
            .sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesAdvanced)
            .expect("BestPracticesAdvanced should be present at L0");

        // L0 should use full content (contains code examples)
        assert!(
            advanced.content.contains("```"),
            "L0 BestPracticesAdvanced should contain code examples"
        );
    }

    #[test]
    fn test_heuristic_router_l1_uses_compact_content() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 1,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
            ..Default::default()
        };

        let decision = router.route(&ctx);
        let advanced = decision
            .sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesAdvanced)
            .expect("BestPracticesAdvanced should be present at L1");

        // L1 should use compact content (no code examples)
        assert!(
            !advanced.content.contains("```"),
            "L1 BestPracticesAdvanced should NOT contain code examples"
        );
        assert!(
            advanced.content.contains("(compact)"),
            "L1 BestPracticesAdvanced should be the compact variant"
        );
    }

    #[test]
    fn test_heuristic_router_l2_excludes_advanced() {
        let router = HeuristicRouter;
        let ctx = RoutingContext {
            scaffolding_level: 2,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
            ..Default::default()
        };

        let decision = router.route(&ctx);
        let ids: Vec<_> = decision.sections.iter().map(|s| s.id).collect();
        assert!(
            !ids.contains(&PromptSectionId::BestPracticesAdvanced),
            "BestPracticesAdvanced should be absent at L2"
        );
    }
}
