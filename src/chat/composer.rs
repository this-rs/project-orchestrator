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
use super::prompt_sections::{
    assemble_sections, extract_tool_reference, select_sections, select_tool_groups,
    ComposerContext, ToolGroupSelectionContext,
};
use super::stages::status_injection::ProtocolRunStatus;

// ============================================================================
// SectionHint — future routing interface for DualTrackRouter
// ============================================================================

/// A hint from the routing system about a specific prompt section.
///
/// Future: the DualTrackRouter will produce these to influence which sections
/// are included and with what token budget. For now, this is a placeholder.
#[derive(Debug, Clone)]
pub struct SectionHint {
    pub section_id: super::prompt_sections::PromptSectionId,
    /// Weight: 0.0 = exclude, 1.0 = must include.
    pub weight: f32,
    /// Optional per-section token budget.
    pub token_budget: Option<usize>,
}

// ============================================================================
// FsmPromptComposer — the central prompt assembly engine
// ============================================================================

/// Inputs for prompt composition, gathered before calling `compose()`.
///
/// All async data fetching happens outside the composer — it receives
/// pre-rendered markdown strings for dynamic sections.
pub struct ComposerInput<'a> {
    /// Scaffolding level (0=full guidance, 4=expert).
    pub scaffolding_level: u8,
    /// Active protocol runs with their FSM state data.
    pub protocol_runs: &'a [ProtocolRunStatus],
    /// Dynamic project context rendered as markdown (from `context_to_markdown`).
    pub project_context_markdown: &'a str,
    /// Session continuity rendered as markdown (from `load_session_context`).
    pub continuity_markdown: &'a str,
    /// The user's current message (for intent detection).
    pub user_message: &'a str,
    /// Optional routing hints from DualTrackRouter (future).
    pub routing_hints: Option<&'a [SectionHint]>,
    /// Whether the project has sibling projects (multi-project workspace).
    pub is_multi_project: bool,
    /// Whether there are active plans.
    pub has_active_plan: bool,
    /// Number of tasks across active plans.
    pub task_count: usize,
}

impl<'a> Default for ComposerInput<'a> {
    fn default() -> Self {
        Self {
            scaffolding_level: 0,
            protocol_runs: &[],
            project_context_markdown: "",
            continuity_markdown: "",
            user_message: "",
            routing_hints: None,
            is_multi_project: false,
            has_active_plan: false,
            task_count: 0,
        }
    }
}

/// Maximum character budget for the dynamic context section (~2500 tokens at 4 chars/token).
const DYNAMIC_CONTEXT_CHAR_BUDGET: usize = 10_000;

/// The FsmPromptComposer assembles the full system prompt from modular sections.
///
/// It is stateless — each call to `compose()` produces a fresh prompt string.
/// All state (project context, protocol runs, etc.) is passed via `ComposerInput`.
pub struct FsmPromptComposer;

impl FsmPromptComposer {
    /// Compose the full system prompt from the given inputs.
    ///
    /// Assembly pipeline:
    /// 1. Select base sections (scaffolding + context filtering)
    /// 2. Assemble base sections into markdown
    /// 3. Inject FSM prompt fragments (from active protocol runs)
    /// 4. Select and render tool reference groups
    /// 5. Append dynamic context (project + continuity), truncated
    /// 6. Join everything into the final prompt
    pub fn compose(input: &ComposerInput<'_>) -> String {
        // ── Step 1: Select base sections ──────────────────────────────
        let composer_ctx = ComposerContext {
            scaffolding_level: input.scaffolding_level,
            has_active_plan: input.has_active_plan,
            has_active_protocol: !input.protocol_runs.is_empty(),
            task_count: input.task_count,
        };
        let sections = select_sections(&composer_ctx);

        // ── Step 2: Assemble base sections ────────────────────────────
        let base_prompt = assemble_sections(&sections);

        // ── Step 3: Inject FSM prompt fragments ───────────────────────
        let fsm_section = Self::build_fsm_section(input.protocol_runs);

        // ── Step 4: Select and render tool reference ──────────────────
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let tool_group_ctx = ToolGroupSelectionContext {
            scaffolding_level: input.scaffolding_level,
            has_active_protocol: !input.protocol_runs.is_empty(),
            is_multi_project: input.is_multi_project,
            fsm_available_tools: fsm_tools,
            user_intent_keywords: vec![input.user_message.to_string()],
        };
        let tool_groups = select_tool_groups(&tool_group_ctx);
        let tool_ref = extract_tool_reference(TOOL_REFERENCE, &tool_groups);

        // ── Step 5: Build dynamic context (truncated) ─────────────────
        let dynamic = Self::build_dynamic_section(
            input.continuity_markdown,
            input.project_context_markdown,
        );

        // ── Step 6: Assemble final prompt ─────────────────────────────
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

        parts.join("\n\n---\n\n")
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

    /// Build the dynamic context section (continuity + project context), truncated.
    fn build_dynamic_section(continuity: &str, project_context: &str) -> String {
        let mut parts = Vec::new();

        if !continuity.is_empty() {
            parts.push(continuity);
        }
        if !project_context.is_empty() {
            parts.push(project_context);
        }

        if parts.is_empty() {
            return String::new();
        }

        let full = parts.join("\n\n");

        // Truncate to budget
        if full.len() <= DYNAMIC_CONTEXT_CHAR_BUDGET {
            full
        } else {
            // Simple truncation: cut at char boundary, add ellipsis
            let truncated = &full[..floor_char_boundary(&full, DYNAMIC_CONTEXT_CHAR_BUDGET)];
            format!("{}\n\n[... dynamic context truncated to fit token budget]", truncated)
        }
    }

    /// Estimate token count (~4 chars per token).
    #[allow(dead_code)]
    fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    /// Count total tool groups selected (for metrics/logging).
    #[allow(dead_code)]
    pub fn count_tool_groups(input: &ComposerInput<'_>) -> usize {
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let ctx = ToolGroupSelectionContext {
            scaffolding_level: input.scaffolding_level,
            has_active_protocol: !input.protocol_runs.is_empty(),
            is_multi_project: input.is_multi_project,
            fsm_available_tools: fsm_tools,
            user_intent_keywords: vec![input.user_message.to_string()],
        };
        select_tool_groups(&ctx).len()
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
        assert!(prompt.contains("# Development Agent"), "Should have identity");
        assert!(prompt.contains("MCP Mega-Tools Reference"), "Should have tool reference");
        assert!(!prompt.contains("Active Protocol Context"), "No FSM section");
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
            forbidden_actions: Some(vec![
                "Do NOT commit without review approval".to_string(),
            ]),
        }];
        let input = ComposerInput {
            protocol_runs: &runs,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        assert!(prompt.contains("Active Protocol Context"), "Should have FSM section");
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

        assert!(prompt.contains("Previous Session"), "Should have continuity");
        assert!(prompt.contains("Project: my-app"), "Should have project context");
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
        let big_context = "x".repeat(20_000);
        let input = ComposerInput {
            project_context_markdown: &big_context,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        // The dynamic section should be truncated
        assert!(
            prompt.contains("truncated to fit token budget"),
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
        assert!(section.contains("Check staging logs"), "Should contain fragment");
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
            groups >= 2 && groups <= 4,
            "FSM whitelist should limit groups, got {}",
            groups
        );
    }
}
