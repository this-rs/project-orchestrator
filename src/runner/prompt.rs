//! Dynamic prompt builder for Runner constraints.
//!
//! Replaces the old static `RUNNER_CONSTRAINTS` const with a template-based
//! system that injects contextual variables (branch, forbidden files, skills, etc.).

use serde::{Deserialize, Serialize};

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
