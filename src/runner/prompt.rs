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
            Self::SkillContext(_) => 5,
            Self::FileContext(_) => 6,
            Self::Enrichment(_) => 7,
            Self::RunnerConstraints(_) => 8,
            Self::Custom(_) => 9,
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
        }
    }
}

/// Base constraints text — identical to the old `RUNNER_CONSTRAINTS` const.
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

/// Build the full runner constraints prompt from the given context.
///
/// When called with `RunnerPromptContext::single_agent(...)`, the output is
/// identical to the old static `RUNNER_CONSTRAINTS` const.
pub fn build_runner_constraints(ctx: &RunnerPromptContext) -> String {
    let mut out = String::from(BASE_CONSTRAINTS);

    // Branch constraint
    if !ctx.git_branch.is_empty() {
        out.push_str(&format!(
            "\nTous tes commits doivent être sur la branche `{}`.\n",
            ctx.git_branch
        ));
    }

    // Forbidden files (parallel agent safety)
    if !ctx.forbidden_files.is_empty() {
        out.push_str(&format!(
            "\nNE MODIFIE PAS ces fichiers, ils sont gérés par d'autres agents : {}\n",
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
            "\n⚠️ Frustration élevée — utilise le deep reasoning avant d'agir.\n",
        );
    }

    // Parallel agent awareness
    if ctx.parallel_agents > 1 {
        out.push_str(&format!(
            "\nTu es l'agent {} parmi {} agents parallèles dans cette wave.\n",
            ctx.wave_number, ctx.parallel_agents
        ));
    }

    out
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
            .with_skill_context("Use async/await")
            .with_file_context("src/main.rs: rust, symbols: main")
            .with_enrichment("<enrichment>content</enrichment>")
            .with_propagated_notes("- [gotcha] Watch out for nulls")
            .build();

        assert!(prompt.contains("# Task"));
        assert!(prompt.contains("## Knowledge Notes"));
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
        assert!(result.contains("branche `feat/my-branch`"));
    }

    #[test]
    fn test_forbidden_files() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.forbidden_files = vec!["src/main.rs".to_string(), "Cargo.toml".to_string()];
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("NE MODIFIE PAS"));
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
        assert!(result.contains("Frustration élevée"));
    }

    #[test]
    fn test_no_frustration_warning_below_threshold() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.frustration_level = 0.5;
        let result = build_runner_constraints(&ctx);
        assert!(!result.contains("Frustration"));
    }

    #[test]
    fn test_parallel_agents() {
        let mut ctx = RunnerPromptContext::single_agent(String::new());
        ctx.parallel_agents = 3;
        ctx.wave_number = 2;
        let result = build_runner_constraints(&ctx);
        assert!(result.contains("l'agent 2 parmi 3 agents parallèles"));
    }
}
