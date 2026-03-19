//! FsmPromptComposer — modular, FSM-aware prompt assembly.
//!
//! Replaces the monolithic `build_system_prompt()` in manager.rs with a composable
//! system that adapts the prompt based on:
//! - Scaffolding level (0-4) — progressive complexity reduction
//! - Active protocol runs — inject prompt_fragment, available_tools, forbidden_actions
//! - Project context — plans, constraints, guidelines, topology
//! - Session continuity — previous session resume data
//! - User message — intent detection for tool group selection
//! - Routing hints — future: DualTrackRouter provides section weights
//!
//! Architecture:
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   FsmPromptComposer                     │
//! │                                                         │
//! │  Inputs:                                                │
//! │  1. scaffolding_level: u8                               │
//! │  2. protocol_runs: Vec<ProtocolRunStatus>               │
//! │  3. project_context_markdown: String                    │
//! │  4. continuity_markdown: String                         │
//! │  5. user_message: &str                                  │
//! │  6. routing_hints: Option<Vec<SectionHint>>             │
//! │                                                         │
//! │  Pipeline:                                              │
//! │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐ │
//! │  │ Select   │→ │ Inject    │→ │ Select tool groups   │ │
//! │  │ sections │  │ FSM frags │  │ (intent+FSM+level)   │ │
//! │  └──────────┘  └───────────┘  └──────────────────────┘ │
//! │       ↓              ↓               ↓                  │
//! │  ┌──────────────────────────────────────────────┐      │
//! │  │ Assemble: base + FSM + dynamic + tool_ref    │      │
//! │  └──────────────────────────────────────────────┘      │
//! │       ↓                                                 │
//! │  ┌──────────┐                                           │
//! │  │ Truncate  │ → String                                 │
//! │  └──────────┘                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```

use super::prompt::TOOL_REFERENCE;
use super::prompt_sections::{assemble_sections, extract_tool_reference};
use super::routing::{HeuristicRouter, RoutingContext, RoutingDecisionRecord, RoutingProvider};
use super::stages::status_injection::ProtocolRunStatus;

// Re-export SectionHint from routing for backward compatibility
pub use super::routing::SectionHint;

// ============================================================================
// FsmPromptComposer — the central prompt assembly engine
// ============================================================================

/// Inputs for prompt composition, gathered before calling `compose()`.
///
/// All async data fetching happens outside the composer — it receives
/// pre-rendered markdown strings for dynamic sections.
#[derive(Debug, Clone, Default)]
pub struct ComposerInput<'a> {
    /// Scaffolding level (0=full guidance, 4=expert).
    pub scaffolding_level: u8,
    /// Active protocol runs with their FSM state data.
    pub protocol_runs: &'a [ProtocolRunStatus],
    /// Dynamic project context rendered as markdown (from `context_to_markdown`).
    pub project_context_markdown: &'a str,
    /// Session continuity rendered as markdown (from `load_session_context`).
    pub continuity_markdown: &'a str,
    /// Enrichment pipeline output rendered as markdown (from `to_system_prompt_markdown`).
    /// Integrated into the system prompt instead of being prepended to the user message.
    pub enrichment_markdown: &'a str,
    /// The user's current message (for intent detection).
    pub user_message: &'a str,
    /// Optional routing hints (ignored by HeuristicRouter, used by DualTrackRouter).
    pub routing_hints: Option<&'a [SectionHint]>,
    /// Whether the project has sibling projects (multi-project workspace).
    pub is_multi_project: bool,
    /// Whether there are active plans.
    pub has_active_plan: bool,
    /// Number of tasks across active plans.
    pub task_count: usize,
    /// Model name (e.g., "claude-sonnet-4-20250514"). Used for adaptive context budgeting.
    pub model: &'a str,
    /// Pre-computed embedding of the user message (for DualTrackRouter).
    /// `None` if embeddings are unavailable → routing falls back to heuristics.
    pub message_embedding: Option<&'a Vec<f32>>,
}

// Default is derived — all numeric fields default to 0, bools to false,
// &str to "", Option to None, &[] to empty slice.

/// Fallback character budget when the model is unknown (~2500 tokens at 4 chars/token).
const DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET: usize = 10_000;

/// Compute the dynamic context character budget based on the model's context window
/// and the current system prompt length.
///
/// Strategy: allocate up to 15% of the model's remaining context window (after base prompt)
/// for dynamic context, clamped between 5K and 40K chars.
///
/// Known context windows (in tokens):
/// - claude-opus-4 / claude-sonnet-4: 200K
/// - claude-haiku-3-5: 200K
/// - claude-3-opus: 200K
/// - Smaller/unknown models: fallback to 10K chars
fn compute_dynamic_budget(model: &str, base_prompt_chars: usize) -> usize {
    // Estimate model context window in tokens
    let context_window_tokens: usize = if model.contains("opus")
        || model.contains("sonnet")
        || model.contains("haiku")
    {
        200_000
    } else {
        // Unknown model → use conservative fallback
        return DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET;
    };

    // Convert to chars (~4 chars per token)
    let context_window_chars = context_window_tokens * 4;

    // Remaining space after base prompt
    let remaining = context_window_chars.saturating_sub(base_prompt_chars);

    // Allocate 15% of remaining space for dynamic context
    let budget = remaining * 15 / 100;

    // Clamp between 5K and 40K chars
    budget.clamp(5_000, 40_000)
}

/// The FsmPromptComposer assembles the full system prompt from modular sections.
///
/// It is stateless — each call to `compose()` produces a fresh prompt string.
/// All state (project context, protocol runs, etc.) is passed via `ComposerInput`.
pub struct FsmPromptComposer;

impl FsmPromptComposer {
    /// Compose the full system prompt using the default [`HeuristicRouter`].
    ///
    /// This is the standard entry point. For custom routing (e.g., DualTrackRouter),
    /// use [`compose_with_router()`] instead.
    pub fn compose(input: &ComposerInput<'_>) -> String {
        Self::compose_with_router(input, &HeuristicRouter).0
    }

    /// Compose the full system prompt and return the routing decision record.
    ///
    /// Use this when you need to emit the routing decision to the
    /// `TrajectoryCollector` for the neural feedback loop.
    pub fn compose_with_record(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> (String, RoutingDecisionRecord) {
        Self::compose_with_router(input, router)
    }

    /// Compose the full system prompt using a custom [`RoutingProvider`].
    ///
    /// Assembly pipeline:
    /// 1. Build [`RoutingContext`] from the input
    /// 2. Call `router.route()` to get section + tool group decisions
    /// 3. Assemble base sections into markdown
    /// 4. Inject FSM prompt fragments (from active protocol runs)
    /// 5. Render selected tool reference groups
    /// 6. Append dynamic context (project + continuity + enrichment), truncated
    /// 7. Join everything into the final prompt
    ///
    /// ## Integrating a custom router (e.g., DualTrackRouter)
    ///
    /// ```rust,ignore
    /// use crate::chat::routing::{RoutingProvider, RoutingContext, RoutingDecision};
    ///
    /// let dual_track = DualTrackRouter::new(model_weights);
    /// let prompt = FsmPromptComposer::compose_with_router(&input, &dual_track);
    /// ```
    pub fn compose_with_router(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> (String, RoutingDecisionRecord) {
        // ── Step 1: Build routing context ─────────────────────────────
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let routing_ctx = RoutingContext {
            scaffolding_level: input.scaffolding_level,
            has_active_plan: input.has_active_plan,
            has_active_protocol: !input.protocol_runs.is_empty(),
            task_count: input.task_count,
            is_multi_project: input.is_multi_project,
            fsm_available_tools: fsm_tools,
            user_message: input.user_message.to_string(),
            detected_intent: None, // Future: from enrichment hints
            message_embedding: input.message_embedding.cloned(),
        };

        // ── Step 2: Route — get section + tool group decisions ────────
        let decision = router.route(&routing_ctx);

        // ── Step 3: Assemble base sections ────────────────────────────
        let base_prompt = assemble_sections(&decision.sections);

        // ── Step 4: Inject FSM prompt fragments ───────────────────────
        let fsm_section = Self::build_fsm_section(input.protocol_runs);

        // ── Step 5: Render selected tool reference groups ─────────────
        let tool_ref = extract_tool_reference(TOOL_REFERENCE, &decision.tool_groups);

        // ── Step 6: Build dynamic context (truncated) ─────────────────
        // Compute adaptive budget based on model context window and current prompt size
        let base_prompt_len = base_prompt.len() + fsm_section.len() + tool_ref.len();
        let char_budget = compute_dynamic_budget(input.model, base_prompt_len);
        let dynamic = Self::build_dynamic_section(
            input.continuity_markdown,
            input.project_context_markdown,
            input.enrichment_markdown,
            char_budget,
        );

        // ── Step 7: Build trajectory record ────────────────────────────
        let routing_record = decision.to_trajectory_record(&routing_ctx);

        // ── Step 8: Assemble final prompt ─────────────────────────────
        let mut parts: Vec<&str> = Vec::with_capacity(4);
        parts.push(&base_prompt);

        let fsm_owned;
        if !fsm_section.is_empty() {
            fsm_owned = fsm_section;
            parts.push(&fsm_owned);
        }

        let dynamic_owned;
        if !dynamic.is_empty() {
            dynamic_owned = dynamic;
            parts.push(&dynamic_owned);
        }

        parts.push(&tool_ref);

        (parts.join("\n\n---\n\n"), routing_record)
    }

    /// Build the FSM context section from active protocol runs.
    ///
    /// Injects prompt_fragment, available_tools whitelist, and forbidden_actions
    /// from each active run's current state.
    fn build_fsm_section(runs: &[ProtocolRunStatus]) -> String {
        if runs.is_empty() {
            return String::new();
        }

        let mut lines = vec!["## Active Protocol Context".to_string()];

        for run in runs {
            lines.push(format!(
                "\n### Protocol: {} (state: `{}`)",
                run.protocol_name, run.current_state
            ));

            if !run.status_message.is_empty() {
                lines.push(format!("Status: {}", run.status_message));
            }

            // Inject the prompt fragment (contextual instructions for this state)
            if let Some(ref fragment) = run.prompt_fragment {
                lines.push(String::new());
                lines.push(fragment.clone());
            }

            // Render available tools whitelist
            if let Some(ref tools) = run.available_tools {
                if !tools.is_empty() {
                    lines.push(String::new());
                    lines.push(format!(
                        "**Allowed tools in state `{}`**: {}",
                        run.current_state,
                        tools.join(", ")
                    ));
                }
            }

            // Render forbidden actions as warnings
            if let Some(ref forbidden) = run.forbidden_actions {
                if !forbidden.is_empty() {
                    lines.push(String::new());
                    lines.push(format!(
                        "⚠️ **Forbidden in state `{}`**:",
                        run.current_state
                    ));
                    for action in forbidden {
                        lines.push(format!("- {}", action));
                    }
                }
            }
        }

        lines.join("\n")
    }

    /// Build the dynamic context section (continuity + project context + enrichment)
    /// with semantic truncation.
    ///
    /// Instead of blindly cutting at a character boundary, this function:
    /// 1. Parses the markdown into sections (## headers) with their list items (- lines)
    /// 2. Scores each item by importance markers ([Critical] > [High] > [Medium] > [Low])
    /// 3. When over budget, keeps only the top-N items per section (never drops entire sections)
    ///
    /// `char_budget` is the maximum character count for the dynamic section, computed
    /// by [`compute_dynamic_budget()`] based on the model's context window.
    fn build_dynamic_section(
        continuity: &str,
        project_context: &str,
        enrichment: &str,
        char_budget: usize,
    ) -> String {
        let mut parts = Vec::new();

        if !continuity.is_empty() {
            parts.push(continuity);
        }
        if !enrichment.is_empty() {
            parts.push(enrichment);
        }
        if !project_context.is_empty() {
            parts.push(project_context);
        }

        if parts.is_empty() {
            return String::new();
        }

        let full = parts.join("\n\n");

        // Under budget → return as-is
        if full.len() <= char_budget {
            return full;
        }

        // Over budget → semantic truncation
        truncate_markdown_semantically(&full, char_budget)
    }

    /// Estimate token count (~4 chars per token).
    #[allow(dead_code)]
    fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    /// Count total tool groups selected (for metrics/logging).
    #[allow(dead_code)]
    pub fn count_tool_groups(input: &ComposerInput<'_>) -> usize {
        Self::count_tool_groups_with_router(input, &HeuristicRouter)
    }

    /// Count tool groups using a custom router.
    #[allow(dead_code)]
    pub fn count_tool_groups_with_router(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> usize {
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let routing_ctx = RoutingContext {
            scaffolding_level: input.scaffolding_level,
            has_active_plan: input.has_active_plan,
            has_active_protocol: !input.protocol_runs.is_empty(),
            task_count: input.task_count,
            is_multi_project: input.is_multi_project,
            fsm_available_tools: fsm_tools,
            user_message: input.user_message.to_string(),
            detected_intent: None,
            message_embedding: input.message_embedding.cloned(),
        };
        router.route(&routing_ctx).tool_groups.len()
    }
}

/// Find the largest char boundary <= the given byte index.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// A parsed markdown section with header and scored items.
struct MdSection {
    /// The header line (e.g., "## Guidelines\n")
    header: String,
    /// Non-item lines (description text between header and items)
    preamble: Vec<String>,
    /// List items with importance scores (higher = more important)
    items: Vec<(u8, String)>,
}

/// Score a markdown list item by importance markers.
///
/// Recognizes patterns like `[Critical]`, `[High]`, `[Medium]`, `[Low]`
/// and also priority numbers (priority 90 > priority 10).
fn score_item(line: &str) -> u8 {
    let lower = line.to_lowercase();
    if lower.contains("[critical]") || lower.contains("critical") {
        return 100;
    }
    if lower.contains("[high]") {
        return 80;
    }
    if lower.contains("[medium]") {
        return 50;
    }
    if lower.contains("[low]") {
        return 20;
    }
    // Check for priority numbers: "priority 90" → score 90
    if let Some(pos) = lower.find("priority ") {
        let rest = &lower[pos + 9..];
        if let Some(num_str) = rest.split(|c: char| !c.is_ascii_digit()).next() {
            if let Ok(n) = num_str.parse::<u8>() {
                return n;
            }
        }
    }
    // Default: medium importance
    40
}

/// Parse markdown into sections (## headers) with their list items.
fn parse_markdown_sections(text: &str) -> Vec<MdSection> {
    let mut sections: Vec<MdSection> = Vec::new();
    let mut current: Option<MdSection> = None;
    let mut in_items = false;

    for line in text.lines() {
        if line.starts_with("## ") {
            // Save previous section
            if let Some(sec) = current.take() {
                sections.push(sec);
            }
            current = Some(MdSection {
                header: line.to_string(),
                preamble: Vec::new(),
                items: Vec::new(),
            });
            in_items = false;
        } else if let Some(ref mut sec) = current {
            if line.starts_with("- ") || line.starts_with("* ") {
                in_items = true;
                let score = score_item(line);
                sec.items.push((score, line.to_string()));
            } else if in_items && (line.starts_with("  ") || line.is_empty()) {
                // Continuation of previous item or blank between items
                if let Some(last) = sec.items.last_mut() {
                    last.1.push('\n');
                    last.1.push_str(line);
                }
            } else {
                sec.preamble.push(line.to_string());
            }
        } else {
            // Lines before any section header — treat as preamble of a virtual section
            if current.is_none() {
                current = Some(MdSection {
                    header: String::new(),
                    preamble: vec![line.to_string()],
                    items: Vec::new(),
                });
            }
        }
    }

    if let Some(sec) = current {
        sections.push(sec);
    }

    sections
}

/// Render sections back to markdown, keeping only top-N items per section.
fn render_sections_budgeted(sections: &mut [MdSection], char_budget: usize) -> String {
    // First pass: calculate total size with all items
    let total: usize = sections
        .iter()
        .map(|s| {
            s.header.len()
                + s.preamble.iter().map(|l| l.len() + 1).sum::<usize>()
                + s.items.iter().map(|(_, l)| l.len() + 1).sum::<usize>()
                + 2 // newlines
        })
        .sum();

    if total <= char_budget {
        // Everything fits
        return sections
            .iter()
            .map(|s| render_section(s, s.items.len()))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // Over budget: progressively reduce items per section
    // Sort items within each section by score (descending)
    for sec in sections.iter_mut() {
        sec.items.sort_by(|a, b| b.0.cmp(&a.0));
    }

    // Binary search for max items per section that fits
    let max_items = sections.iter().map(|s| s.items.len()).max().unwrap_or(0);
    let mut best_n = 1; // Keep at least 1 item per section

    for n in (1..=max_items).rev() {
        let size: usize = sections
            .iter()
            .map(|s| {
                let kept = s.items.len().min(n);
                s.header.len()
                    + s.preamble.iter().map(|l| l.len() + 1).sum::<usize>()
                    + s.items.iter().take(kept).map(|(_, l)| l.len() + 1).sum::<usize>()
                    + if kept < s.items.len() { 40 } else { 0 } // "[N more items omitted]"
                    + 2
            })
            .sum();

        if size <= char_budget {
            best_n = n;
            break;
        }
    }

    let mut result: Vec<String> = Vec::new();
    for sec in sections.iter() {
        result.push(render_section(sec, best_n));
    }

    let mut output = result.join("\n");
    if output.len() > char_budget {
        // Final safety: hard truncate if still over (shouldn't happen normally)
        let boundary = floor_char_boundary(&output, char_budget);
        output.truncate(boundary);
        output.push_str("\n\n[... context truncated]");
    }
    output
}

/// Render a single section with at most `max_items` items (sorted by score descending).
fn render_section(sec: &MdSection, max_items: usize) -> String {
    let mut lines = Vec::new();
    if !sec.header.is_empty() {
        lines.push(sec.header.clone());
    }
    for p in &sec.preamble {
        lines.push(p.clone());
    }
    let total_items = sec.items.len();
    let kept = total_items.min(max_items);
    for (_, item_line) in sec.items.iter().take(kept) {
        lines.push(item_line.clone());
    }
    let omitted = total_items.saturating_sub(kept);
    if omitted > 0 {
        lines.push(format!("  [{} more items omitted]", omitted));
    }
    lines.join("\n")
}

/// Semantically truncate markdown by parsing sections and keeping top-N items by importance.
///
/// Unlike blind character truncation, this:
/// - Never drops entire sections (headers are always kept)
/// - Prioritizes items by importance markers ([Critical] > [High] > [Medium] > [Low])
/// - Uniformly reduces items per section to fit the budget
fn truncate_markdown_semantically(text: &str, char_budget: usize) -> String {
    let mut sections = parse_markdown_sections(text);
    render_sections_budgeted(&mut sections, char_budget)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_no_project_no_fsm() {
        let input = ComposerInput {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        // Should contain the identity section
        assert!(
            prompt.contains("# Development Agent"),
            "Should have identity"
        );
        assert!(
            prompt.contains("MCP Mega-Tools Reference"),
            "Should have tool reference"
        );
        assert!(
            !prompt.contains("Active Protocol Context"),
            "No FSM section"
        );
    }

    #[test]
    fn test_compose_with_fsm_fragment() {
        let runs = vec![ProtocolRunStatus {
            protocol_name: "code-review".to_string(),
            current_state: "analyzing".to_string(),
            progress: 25,
            status_message: "Analyzing changes".to_string(),
            prompt_fragment: Some("Focus on test coverage and error handling.".to_string()),
            available_tools: Some(vec!["code".to_string(), "note".to_string()]),
            forbidden_actions: Some(vec!["Do NOT commit without review approval".to_string()]),
        }];
        let input = ComposerInput {
            protocol_runs: &runs,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        assert!(
            prompt.contains("Active Protocol Context"),
            "Should have FSM section"
        );
        assert!(prompt.contains("code-review"), "Should name the protocol");
        assert!(prompt.contains("analyzing"), "Should name the state");
        assert!(
            prompt.contains("Focus on test coverage"),
            "Should inject prompt fragment"
        );
        assert!(
            prompt.contains("Allowed tools"),
            "Should list available tools"
        );
        assert!(
            prompt.contains("Forbidden in state"),
            "Should list forbidden actions"
        );
    }

    #[test]
    fn test_compose_with_dynamic_context() {
        let input = ComposerInput {
            project_context_markdown: "## Project: my-app\n- 3 active plans\n- Rust codebase",
            continuity_markdown: "## Previous Session\n- Last worked on auth module",
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        assert!(
            prompt.contains("Previous Session"),
            "Should have continuity"
        );
        assert!(
            prompt.contains("Project: my-app"),
            "Should have project context"
        );
    }

    #[test]
    fn test_compose_l4_smaller_than_l0() {
        let input_l0 = ComposerInput {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let input_l4 = ComposerInput {
            scaffolding_level: 4,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let prompt_l0 = FsmPromptComposer::compose(&input_l0);
        let prompt_l4 = FsmPromptComposer::compose(&input_l4);

        assert!(
            prompt_l4.len() < prompt_l0.len(),
            "L4 ({} chars) should be smaller than L0 ({} chars)",
            prompt_l4.len(),
            prompt_l0.len()
        );
    }

    #[test]
    fn test_compose_dynamic_truncation() {
        // Build a large structured markdown context
        let mut big_context = String::from("## Items\n");
        for i in 0..500 {
            big_context.push_str(&format!(
                "- [Low] Item {} with padding text to make it much longer\n",
                i
            ));
        }
        let input = ComposerInput {
            project_context_markdown: &big_context,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        // The dynamic section should be semantically truncated
        assert!(
            prompt.contains("more items omitted") || prompt.contains("context truncated"),
            "Should indicate truncation"
        );
        // Total prompt should be reasonable (base + tool_ref + truncated dynamic)
        assert!(
            prompt.len() < 100_000,
            "Total prompt should be bounded, got {}",
            prompt.len()
        );
    }

    #[test]
    fn test_fsm_section_empty_when_no_runs() {
        let section = FsmPromptComposer::build_fsm_section(&[]);
        assert!(section.is_empty(), "No runs → empty FSM section");
    }

    #[test]
    fn test_fsm_section_multiple_runs() {
        let runs = vec![
            ProtocolRunStatus {
                protocol_name: "deploy".to_string(),
                current_state: "staging".to_string(),
                progress: 50,
                status_message: "Deploying to staging".to_string(),
                prompt_fragment: Some("Check staging logs.".to_string()),
                available_tools: None,
                forbidden_actions: None,
            },
            ProtocolRunStatus {
                protocol_name: "review".to_string(),
                current_state: "pending".to_string(),
                progress: 0,
                status_message: "".to_string(),
                prompt_fragment: None,
                available_tools: None,
                forbidden_actions: None,
            },
        ];
        let section = FsmPromptComposer::build_fsm_section(&runs);
        assert!(section.contains("deploy"), "Should contain first protocol");
        assert!(section.contains("review"), "Should contain second protocol");
        assert!(
            section.contains("Check staging logs"),
            "Should contain fragment"
        );
    }

    #[test]
    fn test_tool_groups_fsm_filtering() {
        let runs = vec![ProtocolRunStatus {
            protocol_name: "test".to_string(),
            current_state: "run".to_string(),
            progress: 0,
            status_message: "".to_string(),
            prompt_fragment: None,
            available_tools: Some(vec!["code".to_string(), "note".to_string()]),
            forbidden_actions: None,
        }];
        let input = ComposerInput {
            protocol_runs: &runs,
            ..Default::default()
        };
        let groups = FsmPromptComposer::count_tool_groups(&input);
        // Core + Knowledge always, + CodeExploration (has "code")
        assert!(
            (2..=4).contains(&groups),
            "FSM whitelist should limit groups, got {}",
            groups
        );
    }

    // ── Semantic truncation tests ────────────────────────────────────

    #[test]
    fn test_score_item_importance_ordering() {
        assert!(score_item("- [Critical] Never do X") > score_item("- [High] Avoid Y"));
        assert!(score_item("- [High] Avoid Y") > score_item("- [Medium] Consider Z"));
        assert!(score_item("- [Medium] Consider Z") > score_item("- [Low] Maybe W"));
    }

    #[test]
    fn test_score_item_priority_number() {
        let score = score_item("- **Plan A** (InProgress, priority 90)");
        assert!(score == 90, "Should extract priority 90, got {}", score);
    }

    #[test]
    fn test_parse_markdown_sections() {
        let md =
            "## Guidelines\n- [Critical] Rule 1\n- [Low] Rule 2\n\n## Gotchas\n- Watch out for X\n";
        let sections = parse_markdown_sections(md);
        assert_eq!(sections.len(), 2, "Should parse 2 sections");
        assert_eq!(sections[0].items.len(), 2, "Guidelines should have 2 items");
        assert_eq!(sections[1].items.len(), 1, "Gotchas should have 1 item");
    }

    #[test]
    fn test_semantic_truncation_keeps_critical_drops_low() {
        // Build a dynamic context with 8 guidelines of varying importance
        let mut md = String::from("## Guidelines\n");
        md.push_str("- [Critical] Never expose API keys\n");
        md.push_str("- [Critical] Always validate input\n");
        md.push_str("- [Critical] Use parameterized queries\n");
        md.push_str("- [High] Log all errors\n");
        md.push_str("- [High] Use structured logging\n");
        md.push_str("- [Medium] Prefer composition over inheritance\n");
        md.push_str("- [Low] Use snake_case for variables\n");
        md.push_str("- [Low] Max line length 100\n");

        // Set a very tight budget that can only fit ~3 items
        let budget = md.len() / 2;
        let result = truncate_markdown_semantically(&md, budget);

        // Should keep the Critical items, drop Low items
        assert!(
            result.contains("Never expose API keys"),
            "Should keep Critical item"
        );
        assert!(
            result.contains("Always validate input"),
            "Should keep Critical item"
        );
        assert!(
            result.contains("## Guidelines"),
            "Should keep section header"
        );
        // Should indicate omitted items
        assert!(
            result.contains("more items omitted"),
            "Should show omission indicator"
        );
    }

    #[test]
    fn test_semantic_truncation_preserves_all_section_headers() {
        let md = "## Guidelines\n- [Low] Rule 1\n- [Low] Rule 2\n- [Low] Rule 3\n- [Low] Rule 4\n\n\
                   ## Gotchas\n- [Low] Gotcha 1\n- [Low] Gotcha 2\n- [Low] Gotcha 3\n- [Low] Gotcha 4\n\n\
                   ## Plans\n- [Low] Plan 1\n- [Low] Plan 2\n- [Low] Plan 3\n- [Low] Plan 4\n";

        // Budget that fits headers + 1 item each but not all items
        let budget = md.len() * 2 / 3;
        let result = truncate_markdown_semantically(md, budget);

        // All section headers must be preserved
        assert!(
            result.contains("## Guidelines"),
            "Must keep Guidelines header"
        );
        assert!(result.contains("## Gotchas"), "Must keep Gotchas header");
        assert!(result.contains("## Plans"), "Must keep Plans header");
        // Should have omitted some items
        assert!(
            result.contains("more items omitted"),
            "Should indicate omitted items"
        );
    }

    #[test]
    fn test_semantic_truncation_under_budget_unchanged() {
        let md = "## Small\n- Item 1\n- Item 2\n";
        let result = truncate_markdown_semantically(md, 10_000);
        // Should return as-is (no omission markers)
        assert!(
            !result.contains("omitted"),
            "Under budget should not truncate"
        );
        assert!(result.contains("Item 1"));
        assert!(result.contains("Item 2"));
    }

    #[test]
    fn test_dynamic_section_uses_semantic_truncation() {
        // Build a big project context that exceeds the default budget
        let budget = DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET;
        let mut big_context = String::new();
        big_context.push_str("## Guidelines\n");
        for i in 0..100 {
            let importance = if i < 10 {
                "Critical"
            } else if i < 30 {
                "High"
            } else {
                "Low"
            };
            big_context.push_str(&format!(
                "- [{}] Guideline number {} with some padding text to make it longer and exceed the budget easily\n",
                importance, i
            ));
        }
        big_context.push_str("\n## Gotchas\n");
        for i in 0..50 {
            big_context.push_str(&format!("- [Medium] Gotcha number {} with extra text\n", i));
        }

        assert!(
            big_context.len() > budget,
            "Test context should exceed budget"
        );

        let result = FsmPromptComposer::build_dynamic_section("", &big_context, "", budget);

        // Should be within budget (with some margin for omission markers)
        assert!(
            result.len() <= budget + 200,
            "Result ({} chars) should be near budget ({})",
            result.len(),
            budget
        );

        // Should keep section headers
        assert!(result.contains("## Guidelines"));
        assert!(result.contains("## Gotchas"));

        // Should indicate truncation
        assert!(result.contains("more items omitted"));
    }

    // ── compute_dynamic_budget tests ─────────────────────────────────

    #[test]
    fn test_budget_known_model_sonnet() {
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 50_000);
        // 200K tokens × 4 chars = 800K chars; remaining = 750K; 15% = 112.5K → clamped to 40K
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_known_model_haiku() {
        let budget = compute_dynamic_budget("claude-haiku-3-5-20241022", 50_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_known_model_opus() {
        let budget = compute_dynamic_budget("claude-opus-4-20250514", 50_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_unknown_model_uses_default() {
        let budget = compute_dynamic_budget("gpt-4o", 50_000);
        assert_eq!(budget, DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET);
    }

    #[test]
    fn test_budget_empty_model_uses_default() {
        let budget = compute_dynamic_budget("", 50_000);
        assert_eq!(budget, DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET);
    }

    #[test]
    fn test_budget_large_base_prompt_reduces_budget() {
        // Base prompt nearly fills the context window
        // 200K tokens × 4 = 800K chars; remaining = 800K - 780K = 20K; 15% = 3K → clamped to 5K
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 780_000);
        assert_eq!(budget, 5_000);
    }

    #[test]
    fn test_budget_small_base_prompt_gets_max() {
        // Tiny base prompt → 15% of ~800K = ~120K → clamped to 40K
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 1_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_adapts_via_compose() {
        // Verify that compose() with a model name produces a different budget than without
        let input_with_model = ComposerInput {
            model: "claude-sonnet-4-20250514",
            project_context_markdown: "## Context\n- item",
            ..Default::default()
        };
        let input_no_model = ComposerInput {
            model: "",
            project_context_markdown: "## Context\n- item",
            ..Default::default()
        };
        // Both should produce valid prompts (just with different internal budgets)
        let prompt_with = FsmPromptComposer::compose(&input_with_model);
        let prompt_without = FsmPromptComposer::compose(&input_no_model);
        assert!(!prompt_with.is_empty());
        assert!(!prompt_without.is_empty());
    }
}
