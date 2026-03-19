//! Dynamic prompt builder for Runner constraints.
//!
//! Replaces the old static `RUNNER_CONSTRAINTS` const with a template-based
//! system that injects contextual variables (branch, forbidden files, skills, etc.).
//!
//! Also provides [`PromptBuilder`] — a composable builder for assembling
//! multi-section agent prompts with deterministic section ordering.

use serde::{Deserialize, Serialize};

// ============================================================================
// PromptBuilder — composable prompt assembly
// ============================================================================

/// A section of a composable prompt, rendered in priority order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "content")]
pub enum PromptSection {
    /// Task description (highest priority — always first)
    TaskDescription(String),
    /// Steps to complete
    Steps(String),
    /// Plan constraints
    Constraints(String),
    /// Knowledge notes (guidelines, gotchas, patterns)
    KnowledgeNotes(String),
    /// Persona context (knowledge subgraph from PersonaStack)
    PersonaContext(String),
    /// Activated skill context
    SkillContext(String),
    /// File context (symbols, dependencies)
    FileContext(String),
    /// Runner execution constraints
    RunnerConstraints(String),
    /// Enrichment pipeline output (injected by EnrichmentPipeline)
    Enrichment(String),
    /// Propagated notes from Knowledge Fabric
    PropagatedNotes(String),
    /// Custom section (lowest priority)
    Custom(String),
}

impl PromptSection {
    /// Priority order for sorting (lower = rendered first).
    fn priority(&self) -> u8 {
        match self {
            Self::TaskDescription(_) => 0,
            Self::Steps(_) => 1,
            Self::Constraints(_) => 2,
            Self::KnowledgeNotes(_) => 3,
            Self::PropagatedNotes(_) => 4,
            Self::PersonaContext(_) => 5,
            Self::SkillContext(_) => 6,
            Self::FileContext(_) => 7,
            Self::Enrichment(_) => 8,
            Self::RunnerConstraints(_) => 9,
            Self::Custom(_) => 10,
        }
    }

    /// Section title for markdown rendering.
    fn title(&self) -> &str {
        match self {
            Self::TaskDescription(_) => "Task",
            Self::Steps(_) => "Steps",
            Self::Constraints(_) => "Constraints",
            Self::KnowledgeNotes(_) => "Knowledge Notes",
            Self::PropagatedNotes(_) => "Propagated Knowledge",
            Self::PersonaContext(_) => "Persona Context",
            Self::SkillContext(_) => "Skill Context",
            Self::FileContext(_) => "Files to Modify",
            Self::Enrichment(_) => "Enrichment Context",
            Self::RunnerConstraints(_) => "Runner Constraints",
            Self::Custom(_) => "Additional Context",
        }
    }

    /// Get the inner content string.
    fn content(&self) -> &str {
        match self {
            Self::TaskDescription(s)
            | Self::Steps(s)
            | Self::Constraints(s)
            | Self::KnowledgeNotes(s)
            | Self::PropagatedNotes(s)
            | Self::PersonaContext(s)
            | Self::SkillContext(s)
            | Self::FileContext(s)
            | Self::Enrichment(s)
            | Self::RunnerConstraints(s)
            | Self::Custom(s) => s,
        }
    }
}

/// Composable prompt builder that assembles multi-section agent prompts.
///
/// Sections are rendered in deterministic priority order regardless of insertion order.
/// Use method chaining for ergonomic construction:
/// ```ignore
/// let prompt = PromptBuilder::new()
///     .with_task("Implement feature X")
///     .with_steps("1. Create module\n2. Add tests")
///     .with_constraints("- Must be async")
///     .build();
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptBuilder {
    sections: Vec<PromptSection>,
}

impl PromptBuilder {
    /// Create a new empty PromptBuilder.
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    /// Add an arbitrary section.
    pub fn add_section(mut self, section: PromptSection) -> Self {
        self.sections.push(section);
        self
    }

    /// Add a task description section.
    pub fn with_task(self, description: impl Into<String>) -> Self {
        self.add_section(PromptSection::TaskDescription(description.into()))
    }

    /// Add a steps section.
    pub fn with_steps(self, steps: impl Into<String>) -> Self {
        self.add_section(PromptSection::Steps(steps.into()))
    }

    /// Add a constraints section.
    pub fn with_constraints(self, constraints: impl Into<String>) -> Self {
        self.add_section(PromptSection::Constraints(constraints.into()))
    }

    /// Add a knowledge notes section.
    pub fn with_knowledge_notes(self, notes: impl Into<String>) -> Self {
        self.add_section(PromptSection::KnowledgeNotes(notes.into()))
    }

    /// Add a persona context section (from PersonaStack rendering).
    pub fn with_persona_context(self, context: impl Into<String>) -> Self {
        self.add_section(PromptSection::PersonaContext(context.into()))
    }

    /// Add a skill context section.
    pub fn with_skill_context(self, context: impl Into<String>) -> Self {
        self.add_section(PromptSection::SkillContext(context.into()))
    }

    /// Add a file context section.
    pub fn with_file_context(self, context: impl Into<String>) -> Self {
        self.add_section(PromptSection::FileContext(context.into()))
    }

    /// Add runner constraints section.
    pub fn with_runner_constraints(self, constraints: impl Into<String>) -> Self {
        self.add_section(PromptSection::RunnerConstraints(constraints.into()))
    }

    /// Add enrichment pipeline output section.
    pub fn with_enrichment(self, enrichment: impl Into<String>) -> Self {
        self.add_section(PromptSection::Enrichment(enrichment.into()))
    }

    /// Add propagated notes section.
    pub fn with_propagated_notes(self, notes: impl Into<String>) -> Self {
        self.add_section(PromptSection::PropagatedNotes(notes.into()))
    }

    /// Add a custom section.
    pub fn with_custom(self, content: impl Into<String>) -> Self {
        self.add_section(PromptSection::Custom(content.into()))
    }

    /// Get the sections (for JSON inspection).
    pub fn sections(&self) -> &[PromptSection] {
        &self.sections
    }

    /// Build the final prompt string, sections sorted by priority.
    pub fn build(mut self) -> String {
        self.sections.sort_by_key(|s| s.priority());

        let mut prompt = String::new();
        for section in &self.sections {
            let content = section.content();
            if content.is_empty() {
                continue;
            }
            // TaskDescription gets # heading, others get ##
            match section {
                PromptSection::TaskDescription(_) => {
                    prompt.push_str(&format!("# {}\n\n{}\n\n", section.title(), content));
                }
                _ => {
                    prompt.push_str(&format!("## {}\n{}\n\n", section.title(), content));
                }
            }
        }
        prompt
    }

    /// Build and return structured sections as JSON-serializable data.
    pub fn build_structured(mut self) -> StructuredPrompt {
        self.sections.sort_by_key(|s| s.priority());
        let rendered = {
            let clone = self.clone();
            clone.build()
        };
        StructuredPrompt {
            sections: self.sections,
            rendered,
        }
    }
}

/// A structured prompt with both sections and rendered output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredPrompt {
    /// Individual sections (for review/inspection)
    pub sections: Vec<PromptSection>,
    /// Full rendered prompt string
    pub rendered: String,
}

/// Context used to build the dynamic runner constraints prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerPromptContext {
    pub git_branch: String,
    pub task_tags: Vec<String>,
    pub affected_files: Vec<String>,
    pub forbidden_files: Vec<String>,
    pub skill_context: Option<String>,
    pub frustration_level: f64,
    pub wave_number: usize,
    pub parallel_agents: usize,
    /// Scaffolding level (0-4) for adaptive constraint verbosity.
    pub scaffolding_level: u8,
}

impl RunnerPromptContext {
    /// Default context for a single-agent run (no parallelism, no special constraints).
    pub fn single_agent(git_branch: String) -> Self {
        Self {
            git_branch,
            task_tags: vec![],
            affected_files: vec![],
            forbidden_files: vec![],
            skill_context: None,
            frustration_level: 0.0,
            wave_number: 1,
            parallel_agents: 1,
            scaffolding_level: 0,
        }
    }
}

/// Base constraints text — full version for L0-L1 (verbose guidance).
const BASE_CONSTRAINTS: &str = r#"
## Runner Execution Mode

You are an **autonomous code execution agent** spawned by the PlanRunner.
Your ONLY job is to **write code, test it, and commit it**. You are NOT in a conversation.

### Behavior Rules
1. **Execute immediately** — read the task, analyze the code, write the fix, test, commit. No discussion.
2. **DO NOT** call `task(action: "update", status: ...)` or `step(action: "update", status: ...)` via MCP — the Runner manages all status transitions.
3. **DO NOT** ask questions, request confirmation, or explain your reasoning at length. Just do the work.
4. **DO NOT** use MCP project orchestrator tools for searching code — use Read, Grep, Glob directly for speed.
5. Make atomic commits with conventional format: `<type>(<scope>): <short description>`.
6. **NEVER** commit sensitive files (.env, credentials, *.key, *.pem, *.secret).
7. After writing code, ALWAYS run `cargo check` (Rust) or the relevant build command to verify compilation.
8. If tests are mentioned in steps, run them.
9. If `cargo check` or tests fail, fix the errors and retry — do not give up.
10. When done with ALL steps, make a final commit summarizing the work.

### Execution Flow
1. Read the affected files listed below
2. For each step: implement → verify → move to next
3. Run `cargo check` / `cargo test` as verification
4. Commit with a clear message
5. You are DONE when all steps are implemented and the code compiles
"#;

/// Concise constraints for L2-L3 (reduced guidance — conventions assumed known).
const CONCISE_CONSTRAINTS: &str = r#"
## Runner Execution Mode

Autonomous agent. Write code, test, commit. No discussion, no MCP status updates.
Commits: `<type>(<scope>): <desc>`. Never commit secrets. Verify with `cargo check`.
"#;

/// Minimal constraints for L4 (expert mode — only critical warnings).
const MINIMAL_CONSTRAINTS: &str = r#"
## Runner Mode
Autonomous. Code → test → commit. No MCP status calls. No secrets in commits.
"#;

/// Build the full runner constraints prompt from the given context.
///
/// Adapts verbosity based on `scaffolding_level`:
/// - L0-L1: Full verbose guidance (BASE_CONSTRAINTS)
/// - L2-L3: Concise (CONCISE_CONSTRAINTS)
/// - L4: Minimal (MINIMAL_CONSTRAINTS)
pub fn build_runner_constraints(ctx: &RunnerPromptContext) -> String {
    let base = match ctx.scaffolding_level {
        0..=1 => BASE_CONSTRAINTS,
        2..=3 => CONCISE_CONSTRAINTS,
        _ => MINIMAL_CONSTRAINTS,
    };
    let mut out = String::from(base);

    // Branch constraint
    if !ctx.git_branch.is_empty() {
        out.push_str(&format!(
            "\nAll commits must be on branch `{}`.\n",
            ctx.git_branch
        ));
    }

    // Forbidden files (parallel agent safety)
    if !ctx.forbidden_files.is_empty() {
        out.push_str(&format!(
            "\nDO NOT modify these files (managed by other agents): {}\n",
            ctx.forbidden_files.join(", ")
        ));
    }

    // Activated skill knowledge
    if let Some(ref skill_ctx) = ctx.skill_context {
        out.push_str(&format!("\n## Activated Knowledge\n{}\n", skill_ctx));
    }

    // High frustration warning
    if ctx.frustration_level > 0.7 {
        out.push_str("\n⚠️ High frustration — use deep reasoning before acting.\n");
    }

    // Parallel agent awareness
    if ctx.parallel_agents > 1 {
        out.push_str(&format!(
            "\nYou are agent {} of {} parallel agents in this wave.\n",
            ctx.wave_number, ctx.parallel_agents
        ));
    }

    // Tag-conditional constraints
    build_tag_constraints(&ctx.task_tags, &mut out);

    out
}

/// Build a complete **system prompt** for runner-spawned agents.
///
/// Unlike the generic PO system prompt (conversational assistant), this prompt
/// configures Claude as an **autonomous code execution agent** that writes code,
/// tests, and commits without conversation.
///
/// This is injected as the `system_prompt` in `ClaudeCodeOptions`, giving it
/// stronger behavioral anchoring than user-message constraints.
pub fn build_runner_system_prompt(ctx: &RunnerPromptContext) -> String {
    let base = match ctx.scaffolding_level {
        0..=1 => RUNNER_SYSTEM_PROMPT_FULL,
        2..=3 => RUNNER_SYSTEM_PROMPT_CONCISE,
        _ => RUNNER_SYSTEM_PROMPT_MINIMAL,
    };

    let mut out = String::from(base);

    // Branch constraint
    if !ctx.git_branch.is_empty() {
        out.push_str(&format!(
            "\nAll commits must be on branch `{}`.\n",
            ctx.git_branch
        ));
    }

    // Forbidden files (parallel agent safety)
    if !ctx.forbidden_files.is_empty() {
        out.push_str(&format!(
            "\n**DO NOT** modify these files (managed by other agents in this wave): {}\n",
            ctx.forbidden_files.join(", ")
        ));
    }

    // Activated skill knowledge
    if let Some(ref skill_ctx) = ctx.skill_context {
        out.push_str(&format!("\n## Activated Knowledge\n{}\n", skill_ctx));
    }

    // High frustration warning
    if ctx.frustration_level > 0.7 {
        out.push_str(
            "\n⚠️ High frustration detected on this task — previous attempts failed. \
             Use deep reasoning and try a DIFFERENT approach than previous attempts.\n",
        );
    }

    // Parallel agent awareness
    if ctx.parallel_agents > 1 {
        out.push_str(&format!(
            "\nYou are agent {} of {} parallel agents in this wave. \
             Stay within your assigned files.\n",
            ctx.wave_number, ctx.parallel_agents
        ));
    }

    // Tag-conditional constraints
    build_tag_constraints(&ctx.task_tags, &mut out);

    out
}

/// Full runner system prompt — L0-L1 (verbose guidance for new projects).
const RUNNER_SYSTEM_PROMPT_FULL: &str = r#"You are an **autonomous code execution agent** spawned by the PlanRunner.

Your ONLY job is to **write code, test it, and commit it**. You are NOT in a conversation with a human. Do NOT greet, do NOT explain your reasoning at length, do NOT ask questions or request confirmation. Execute immediately.

## Behavior Rules

1. **Execute immediately** — read the task, analyze the code, implement the solution, test, commit.
2. **DO NOT** call `task(action: "update")` or `step(action: "update")` via MCP — the Runner manages all status transitions.
3. **DO NOT** use MCP project orchestrator tools for searching code — use Read, Grep, Glob directly for speed.
4. Make atomic commits with conventional format: `<type>(<scope>): <short description>`.
5. **NEVER** commit sensitive files (.env, credentials, *.key, *.pem, *.secret).
6. After writing code, ALWAYS run `cargo check` (Rust) or the relevant build command to verify compilation.
7. If tests are mentioned in steps, run them. If `cargo check` or tests fail, fix the errors and retry — do not give up.
8. When done with ALL steps, make a final commit summarizing the work.

## Execution Flow

1. Read the task description and steps provided in the user message
2. Read the affected files listed in the task
3. For each step: implement → verify (cargo check / cargo test) → move to next
4. Commit with a clear conventional message
5. You are DONE when all steps are implemented and the code compiles

## Important

- The user message contains the FULL task context: description, steps, constraints, affected files, knowledge notes
- Start coding IMMEDIATELY after reading the task — no preamble, no discussion
- If you encounter an error, debug it yourself — read error messages, check types, fix imports
- You have full filesystem access and bypassPermissions mode — use it
"#;

/// Concise runner system prompt — L2-L3 (reduced guidance).
const RUNNER_SYSTEM_PROMPT_CONCISE: &str = r#"You are an autonomous code execution agent. Write code, test, commit. No conversation, no MCP status updates.

Rules: Execute immediately. Read task from user message. Implement each step. Run cargo check. Commit with `<type>(<scope>): <desc>`. Never commit secrets. Fix errors yourself. No greetings, no questions.
"#;

/// Minimal runner system prompt — L4 (expert mode).
const RUNNER_SYSTEM_PROMPT_MINIMAL: &str = r#"Autonomous code agent. Code → test → commit. No conversation. No MCP status calls. No secrets in commits.
"#;

/// Append tag-specific constraints based on task tags.
fn build_tag_constraints(tags: &[String], out: &mut String) {
    let tag_set: std::collections::HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();

    if tag_set.contains("test") || tag_set.contains("testing") {
        out.push_str(
            "\n### Test Constraints\n\
             - Run the FULL test suite after changes (`cargo test` or equivalent)\n\
             - Ensure all existing tests still pass — do not delete or skip tests\n\
             - Add tests for new functionality\n",
        );
    }

    if tag_set.contains("refactor") || tag_set.contains("refactoring") {
        out.push_str(
            "\n### Refactor Constraints\n\
             - DO NOT change any public API signatures (function names, parameters, return types)\n\
             - Preserve backward compatibility — all callers must work unchanged\n\
             - Run tests before AND after refactoring to confirm no regressions\n",
        );
    }

    if tag_set.contains("docs") || tag_set.contains("documentation") {
        out.push_str(
            "\n### Documentation Constraints\n\
             - Update doc comments on modified functions/structs\n\
             - Ensure examples in docs compile (if any)\n\
             - Keep README/CHANGELOG up to date if relevant\n",
        );
    }

    if tag_set.contains("security") {
        out.push_str(
            "\n### Security Constraints\n\
             - Review for injection, XSS, CSRF vulnerabilities\n\
             - Validate and sanitize all user inputs\n\
             - Never log sensitive data (passwords, tokens, PII)\n",
        );
    }

    if tag_set.contains("fix") || tag_set.contains("bugfix") {
        out.push_str(
            "\n### Bug Fix Constraints\n\
             - Write a regression test that reproduces the bug BEFORE fixing it\n\
             - Verify the test fails without the fix and passes with it\n",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PromptBuilder tests
    // ========================================================================

    #[test]
    fn test_prompt_builder_empty() {
        let prompt = PromptBuilder::new().build();
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_prompt_builder_single_section() {
        let prompt = PromptBuilder::new()
            .with_task("Implement feature X")
            .build();
        assert!(prompt.contains("# Task"));
        assert!(prompt.contains("Implement feature X"));
    }

    #[test]
    fn test_prompt_builder_priority_ordering() {
        // Add sections in reverse priority order
        let prompt = PromptBuilder::new()
            .with_custom("custom content")
            .with_runner_constraints("runner rules")
            .with_task("task description")
            .with_steps("step 1")
            .with_constraints("constraint A")
            .build();

        // Task should come before Steps, Steps before Constraints, etc.
        let task_pos = prompt.find("# Task").unwrap();
        let steps_pos = prompt.find("## Steps").unwrap();
        let constraints_pos = prompt.find("## Constraints").unwrap();
        let runner_pos = prompt.find("## Runner Constraints").unwrap();
        let custom_pos = prompt.find("## Additional Context").unwrap();

        assert!(task_pos < steps_pos);
        assert!(steps_pos < constraints_pos);
        assert!(constraints_pos < runner_pos);
        assert!(runner_pos < custom_pos);
    }

    #[test]
    fn test_prompt_builder_skips_empty_sections() {
        let prompt = PromptBuilder::new()
            .with_task("task")
            .with_steps("")
            .with_constraints("constraint")
            .build();
        assert!(prompt.contains("# Task"));
        assert!(prompt.contains("## Constraints"));
        assert!(!prompt.contains("## Steps"));
    }

    #[test]
    fn test_prompt_builder_chaining() {
        let prompt = PromptBuilder::new()
            .with_task("task")
            .with_knowledge_notes("- Note 1\n- Note 2")
            .with_persona_context("### Primary Persona — neo4j-expert")
            .with_skill_context("Use async/await")
            .with_file_context("src/main.rs: rust, symbols: main")
            .with_enrichment("<enrichment>content</enrichment>")
            .with_propagated_notes("- [gotcha] Watch out for nulls")
            .build();

        assert!(prompt.contains("# Task"));
        assert!(prompt.contains("## Knowledge Notes"));
        assert!(prompt.contains("## Persona Context"));
        assert!(prompt.contains("## Skill Context"));
        assert!(prompt.contains("## Files to Modify"));
        assert!(prompt.contains("## Enrichment Context"));
        assert!(prompt.contains("## Propagated Knowledge"));
    }

    #[test]
    fn test_prompt_builder_structured() {
        let structured = PromptBuilder::new()
            .with_task("task")
            .with_constraints("constraint")
            .build_structured();

        assert_eq!(structured.sections.len(), 2);
        assert!(structured.rendered.contains("# Task"));
        assert!(structured.rendered.contains("## Constraints"));
    }

    #[test]
    fn test_prompt_section_serialization() {
        let section = PromptSection::TaskDescription("test".to_string());
        let json = serde_json::to_string(&section).unwrap();
        assert!(json.contains("TaskDescription"));
        let deserialized: PromptSection = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content(), "test");
    }

    // ========================================================================
    // RunnerPromptContext / build_runner_constraints tests (existing)
    // ========================================================================

    #[test]
    fn test_single_agent_matches_base() {
        let ctx = RunnerPromptContext::single_agent(String::new());
        let result = build_runner_constraints(&ctx);
        // L0 default → full BASE_CONSTRAINTS
        assert_eq!(result, BASE_CONSTRAINTS);
    }

    #[test]
    fn test_base_constraints_content() {
        assert!(BASE_CONSTRAINTS.contains("## Runner Execution Mode"));
        assert!(BASE_CONSTRAINTS.contains("autonomous code execution agent"));
        assert!(BASE_CONSTRAINTS.contains("DO NOT"));
        assert!(BASE_CONSTRAINTS.contains("task(action: \"update\", status"));
        assert!(BASE_CONSTRAINTS.contains(".env"));
        assert!(BASE_CONSTRAINTS.contains("cargo check"));
    }

    #[test]
    fn test_branch_injected() {
        let ctx = RunnerPromptContext::single_agent("feat/my-branch".to_string());
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("branch `feat/my-branch`"));
    }

    #[test]
    fn test_forbidden_files() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.forbidden_files = vec!["src/main.rs".to_string(), "Cargo.toml".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("DO NOT modify these files"));
        assert!(result.contains("src/main.rs, Cargo.toml"));
    }

    #[test]
    fn test_skill_context() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.skill_context = Some("Use the flux capacitor API.".to_string());
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("## Activated Knowledge"));
        assert!(result.contains("flux capacitor API"));
    }

    #[test]
    fn test_frustration_warning() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.frustration_level = 0.8;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("High frustration"));
    }

    #[test]
    fn test_no_frustration_warning_below_threshold() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.frustration_level = 0.5;
        let result = build_runner_constraints(&ctx);
        assert!(!result.contains("frustration"));
    }

    #[test]
    fn test_parallel_agents() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.parallel_agents = 3;
        ctx.wave_number = 2;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("agent 2 of 3 parallel agents"));
    }

    // ========================================================================
    // Scaffolding level tests
    // ========================================================================

    #[test]
    fn test_scaffolding_l0_verbose() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.scaffolding_level = 0;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("autonomous code execution agent"));
        assert!(result.contains("### Behavior Rules"));
        assert!(result.contains("### Execution Flow"));
    }

    #[test]
    fn test_scaffolding_l2_concise() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.scaffolding_level = 2;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("## Runner Execution Mode"));
        assert!(result.contains("Autonomous agent"));
        assert!(!result.contains("### Behavior Rules"));
    }

    #[test]
    fn test_scaffolding_l4_minimal() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.scaffolding_level = 4;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("## Runner Mode"));
        assert!(!result.contains("### Behavior Rules"));
        assert!(!result.contains("Autonomous agent. Write"));
    }

    #[test]
    fn test_scaffolding_levels_decrease_size() {
        let l0 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 0;
            build_runner_constraints(&ctx)
        };
        let l2 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 2;
            build_runner_constraints(&ctx)
        };
        let l4 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 4;
            build_runner_constraints(&ctx)
        };
        assert!(
            l0.len() > l2.len(),
            "L0 ({}) should be longer than L2 ({})",
            l0.len(),
            l2.len()
        );
        assert!(
            l2.len() > l4.len(),
            "L2 ({}) should be longer than L4 ({})",
            l2.len(),
            l4.len()
        );
    }

    // ========================================================================
    // Tag-conditional constraint tests
    // ========================================================================

    #[test]
    fn test_tag_test_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["test".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Test Constraints"));
        assert!(result.contains("FULL test suite"));
    }

    #[test]
    fn test_tag_refactor_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["refactor".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Refactor Constraints"));
        assert!(result.contains("public API signatures"));
    }

    #[test]
    fn test_tag_docs_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["docs".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Documentation Constraints"));
    }

    #[test]
    fn test_tag_security_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["security".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Security Constraints"));
    }

    #[test]
    fn test_tag_bugfix_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["fix".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Bug Fix Constraints"));
        assert!(result.contains("regression test"));
    }

    #[test]
    fn test_no_tag_no_extra_constraints() {
        let ctx = RunnerPromptContext::single_agent(String::new());
        let result = build_runner_constraints(&ctx);
        assert!(!result.contains("### Test Constraints"));
        assert!(!result.contains("### Refactor Constraints"));
        assert!(!result.contains("### Documentation Constraints"));
    }

    #[test]
    fn test_multiple_tags_combined() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["test".to_string(), "refactor".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Test Constraints"));
        assert!(result.contains("### Refactor Constraints"));
    }

    #[test]
    fn test_tag_synonym_testing() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["testing".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Test Constraints"));
    }

    #[test]
    fn test_tag_synonym_refactoring() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["refactoring".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Refactor Constraints"));
    }

    #[test]
    fn test_tag_synonym_documentation() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["documentation".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Documentation Constraints"));
    }

    #[test]
    fn test_tag_synonym_bugfix() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["bugfix".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("### Bug Fix Constraints"));
    }

    // ========================================================================
    // PersonaContext section tests
    // ========================================================================

    #[test]
    fn test_persona_context_section_priority() {
        let section = PromptSection::PersonaContext("test".to_string());
        // PersonaContext priority = 5 (after PropagatedNotes=4, before SkillContext=6)
        assert_eq!(section.priority(), 5);
    }

    #[test]
    fn test_persona_context_section_title() {
        let section = PromptSection::PersonaContext("test".to_string());
        assert_eq!(section.title(), "Persona Context");
    }

    #[test]
    fn test_persona_context_section_content() {
        let content = "### Primary Persona — neo4j-expert\nEnergy: 0.85";
        let section = PromptSection::PersonaContext(content.to_string());
        assert_eq!(section.content(), content);
    }

    #[test]
    fn test_with_persona_context_builder() {
        let prompt = PromptBuilder::new()
            .with_persona_context("### Primary Persona — rust-expert\nEnergy: 0.9")
            .build();
        assert!(prompt.contains("## Persona Context"));
        assert!(prompt.contains("rust-expert"));
        assert!(prompt.contains("Energy: 0.9"));
    }

    #[test]
    fn test_persona_context_ordering_after_knowledge_notes() {
        let prompt = PromptBuilder::new()
            .with_persona_context("persona content")
            .with_knowledge_notes("knowledge content")
            .build();

        let knowledge_pos = prompt.find("## Knowledge Notes").unwrap();
        let persona_pos = prompt.find("## Persona Context").unwrap();
        assert!(
            knowledge_pos < persona_pos,
            "KnowledgeNotes (priority 3) should render before PersonaContext (priority 5)"
        );
    }

    #[test]
    fn test_persona_context_ordering_before_skill_context() {
        let prompt = PromptBuilder::new()
            .with_skill_context("skill content")
            .with_persona_context("persona content")
            .build();

        let persona_pos = prompt.find("## Persona Context").unwrap();
        let skill_pos = prompt.find("## Skill Context").unwrap();
        assert!(
            persona_pos < skill_pos,
            "PersonaContext (priority 5) should render before SkillContext (priority 6)"
        );
    }

    #[test]
    fn test_persona_context_ordering_after_propagated_notes() {
        let prompt = PromptBuilder::new()
            .with_persona_context("persona content")
            .with_propagated_notes("propagated content")
            .build();

        let propagated_pos = prompt.find("## Propagated Knowledge").unwrap();
        let persona_pos = prompt.find("## Persona Context").unwrap();
        assert!(
            propagated_pos < persona_pos,
            "PropagatedNotes (priority 4) should render before PersonaContext (priority 5)"
        );
    }

    #[test]
    fn test_persona_context_empty_skipped() {
        let prompt = PromptBuilder::new()
            .with_task("task")
            .with_persona_context("")
            .build();
        assert!(prompt.contains("# Task"));
        assert!(!prompt.contains("Persona Context"));
    }

    #[test]
    fn test_persona_context_in_full_prompt() {
        // Simulate a full runner prompt with persona context included
        let prompt = PromptBuilder::new()
            .with_task("Implement cache layer")
            .with_steps("1. Add Redis\n2. Wire up")
            .with_constraints("- Must be async")
            .with_knowledge_notes("- Use tokio::sync")
            .with_propagated_notes("- [gotcha] Watch TTL")
            .with_persona_context(
                "### Primary Persona — cache-expert\nEnergy: 0.92\nFiles: src/cache.rs",
            )
            .with_skill_context("Skill: redis-patterns")
            .with_file_context("src/cache.rs")
            .with_runner_constraints("Branch: feat/cache")
            .build();

        // Verify all sections present and in order
        let positions: Vec<usize> = vec![
            prompt.find("# Task").unwrap(),
            prompt.find("## Steps").unwrap(),
            prompt.find("## Constraints").unwrap(),
            prompt.find("## Knowledge Notes").unwrap(),
            prompt.find("## Propagated Knowledge").unwrap(),
            prompt.find("## Persona Context").unwrap(),
            prompt.find("## Skill Context").unwrap(),
            prompt.find("## Files to Modify").unwrap(),
            prompt.find("## Runner Constraints").unwrap(),
        ];
        // Verify strictly increasing order
        for i in 1..positions.len() {
            assert!(
                positions[i] > positions[i - 1],
                "Section at index {} should come after section at index {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_persona_context_serialization_roundtrip() {
        let section =
            PromptSection::PersonaContext("### Persona — expert\nEnergy: 0.8".to_string());
        let json = serde_json::to_string(&section).unwrap();
        assert!(json.contains("PersonaContext"));
        let deserialized: PromptSection = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content(), "### Persona — expert\nEnergy: 0.8");
    }

    #[test]
    fn test_persona_context_structured_output() {
        let structured = PromptBuilder::new()
            .with_task("task")
            .with_persona_context("persona data")
            .build_structured();

        assert_eq!(structured.sections.len(), 2);
        assert!(structured.rendered.contains("## Persona Context"));
        assert!(structured.rendered.contains("persona data"));

        // Verify PersonaContext is the second section (after TaskDescription)
        match &structured.sections[1] {
            PromptSection::PersonaContext(s) => assert_eq!(s, "persona data"),
            other => panic!("Expected PersonaContext, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_persona_contexts_both_rendered() {
        // Edge case: what happens if two PersonaContext sections are added?
        let prompt = PromptBuilder::new()
            .with_persona_context("Persona A context")
            .with_persona_context("Persona B context")
            .build();

        // Both should appear (PromptBuilder doesn't deduplicate)
        assert!(prompt.contains("Persona A context"));
        assert!(prompt.contains("Persona B context"));
    }

    // ========================================================================
    // build_runner_system_prompt tests
    // ========================================================================

    #[test]
    fn test_runner_system_prompt_contains_execution_mode() {
        let ctx = RunnerPromptContext::single_agent("main".to_string());
        let result = build_runner_system_prompt(&ctx);
        assert!(
            result.contains("autonomous code execution agent"),
            "Runner system prompt must declare autonomous mode"
        );
        assert!(
            result.contains("NOT in a conversation"),
            "Runner system prompt must explicitly state no conversation"
        );
        assert!(
            result.contains("Do NOT greet"),
            "Runner system prompt must forbid greetings"
        );
        assert!(
            result.contains("Execute immediately"),
            "Runner system prompt must instruct immediate execution"
        );
    }

    #[test]
    fn test_runner_system_prompt_does_not_contain_generic_po() {
        let ctx = RunnerPromptContext::single_agent("main".to_string());
        let result = build_runner_system_prompt(&ctx);
        // Should NOT contain generic PO assistant phrases
        assert!(
            !result.contains("## Runner Execution Mode"),
            "Runner system prompt should not use old constraint heading"
        );
    }

    #[test]
    fn test_runner_system_prompt_branch_injection() {
        let ctx = RunnerPromptContext::single_agent("feat/my-feature".to_string());
        let result = build_runner_system_prompt(&ctx);
        assert!(result.contains("branch `feat/my-feature`"));
    }

    #[test]
    fn test_runner_system_prompt_forbidden_files() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.forbidden_files = vec!["src/main.rs".to_string()];
        let result = build_runner_system_prompt(&ctx);
        assert!(result.contains("DO NOT"));
        assert!(result.contains("src/main.rs"));
    }

    #[test]
    fn test_runner_system_prompt_scaffolding_levels() {
        let l0 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 0;
            build_runner_system_prompt(&ctx)
        };
        let l2 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 2;
            build_runner_system_prompt(&ctx)
        };
        let l4 = {
            let mut ctx = RunnerPromptContext::single_agent(String::new());
            ctx.scaffolding_level = 4;
            build_runner_system_prompt(&ctx)
        };

        // All levels should mention autonomous/code
        assert!(l0.contains("autonomous code execution agent"));
        assert!(l2.contains("autonomous"));
        assert!(l4.contains("Autonomous"));

        // Sizes should decrease with scaffolding level
        assert!(
            l0.len() > l2.len(),
            "L0 ({}) should be longer than L2 ({})",
            l0.len(),
            l2.len()
        );
        assert!(
            l2.len() > l4.len(),
            "L2 ({}) should be longer than L4 ({})",
            l2.len(),
            l4.len()
        );
    }

    #[test]
    fn test_runner_system_prompt_frustration() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.frustration_level = 0.9;
        let result = build_runner_system_prompt(&ctx);
        assert!(result.contains("frustration"));
        assert!(result.contains("DIFFERENT approach"));
    }

    #[test]
    fn test_runner_system_prompt_tag_constraints() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.task_tags = vec!["test".to_string(), "security".to_string()];
        let result = build_runner_system_prompt(&ctx);
        assert!(result.contains("### Test Constraints"));
        assert!(result.contains("### Security Constraints"));
    }
}
