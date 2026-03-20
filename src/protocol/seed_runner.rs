//! Seed module for the 3 plan-runner lifecycle protocols.
//!
//! Creates the FSMs via the compose pattern (Protocol + States + Transitions)
//! using `GraphStore` trait methods directly. Each protocol models a different
//! execution lifecycle:
//!
//! - **plan-runner-full**: Standard implementation flow with PR decision
//! - **plan-runner-light**: Lightweight exploration flow (no PR)
//! - **plan-runner-reviewed**: Critical flow with mandatory human review gate
//!
//! Run via `admin(action: "seed_runner_protocols")` or called from the main seed.
//!
//! Idempotent — skips protocols that already exist by name.

use std::collections::HashMap;

use anyhow::{Context, Result};
use tracing::info;
use uuid::Uuid;

use crate::neo4j::GraphStore;
use crate::protocol::models::{Protocol, ProtocolState, ProtocolTransition};
use crate::protocol::routing::RelevanceVector;
use crate::protocol::StateType;

// ============================================================================
// Types
// ============================================================================

/// A composed protocol definition for seeding (states + transitions + metadata).
struct RunnerProtocolDef {
    name: &'static str,
    description: &'static str,
    relevance_vector: RelevanceVector,
    states: Vec<RunnerStateDef>,
    transitions: Vec<RunnerTransitionDef>,
}

struct RunnerStateDef {
    name: &'static str,
    state_type: StateType,
    prompt_fragment: &'static str,
    available_tools: Option<Vec<&'static str>>,
    forbidden_actions: Option<Vec<&'static str>>,
    action: Option<&'static str>,
}

struct RunnerTransitionDef {
    from: &'static str,
    to: &'static str,
    trigger: &'static str,
    guard: Option<&'static str>,
}

/// Result of the runner protocol seeding operation.
#[derive(Debug, serde::Serialize)]
pub struct RunnerSeedResult {
    pub created: usize,
    pub skipped: usize,
    pub details: Vec<String>,
}

// ============================================================================
// Public API
// ============================================================================

/// Seed the 3 plan-runner lifecycle protocols via the compose pattern.
///
/// Creates: Protocol + States + Transitions for each runner protocol.
/// Idempotent — skips protocols that already exist by name within the project.
pub async fn seed_runner_protocols(
    graph: &dyn GraphStore,
    project_id: Uuid,
) -> Result<RunnerSeedResult> {
    let defs = build_runner_protocol_defs();

    let mut created = 0;
    let mut skipped = 0;
    let mut details = Vec::new();

    for def in &defs {
        // Idempotence: skip if protocol already exists
        if let Some(existing_id) = graph
            .get_protocol_by_name_and_project(def.name, project_id)
            .await
            .with_context(|| format!("Failed to check existence of protocol '{}'", def.name))?
        {
            info!(
                protocol = def.name,
                id = %existing_id,
                "Runner protocol already exists — skipping"
            );
            details.push(format!("{}: already exists ({})", def.name, existing_id));
            skipped += 1;
            continue;
        }

        // Build name→UUID map for states
        let mut name_to_id: HashMap<&str, Uuid> = HashMap::new();
        let mut state_objects: Vec<ProtocolState> = Vec::new();

        let placeholder_entry = Uuid::new_v4();
        let mut proto = Protocol::new(project_id, def.name, placeholder_entry);
        proto.description = def.description.to_string();
        proto.protocol_category = crate::protocol::ProtocolCategory::Business;
        proto.relevance_vector = Some(def.relevance_vector.clone());

        // Create states
        for s in &def.states {
            let mut ps = ProtocolState::new(proto.id, s.name);
            ps.state_type = s.state_type;
            ps.prompt_fragment = Some(s.prompt_fragment.to_string());
            ps.available_tools = s
                .available_tools
                .as_ref()
                .map(|v| v.iter().map(|t| t.to_string()).collect());
            ps.forbidden_actions = s
                .forbidden_actions
                .as_ref()
                .map(|v| v.iter().map(|a| a.to_string()).collect());
            ps.action = s.action.map(|a| a.to_string());
            name_to_id.insert(s.name, ps.id);
            state_objects.push(ps);
        }

        // Set entry_state to the Start state
        if let Some(start) = state_objects
            .iter()
            .find(|s| s.state_type == StateType::Start)
        {
            proto.entry_state = start.id;
        }

        // Set terminal_states
        proto.terminal_states = state_objects
            .iter()
            .filter(|s| s.state_type == StateType::Terminal)
            .map(|s| s.id)
            .collect();

        // Upsert protocol
        graph
            .upsert_protocol(&proto)
            .await
            .with_context(|| format!("Failed to upsert protocol '{}'", def.name))?;

        // Upsert states
        for ps in &state_objects {
            graph
                .upsert_protocol_state(ps)
                .await
                .with_context(|| format!("Failed to upsert state '{}/{}'", def.name, ps.name))?;
        }

        // Create transitions
        for t in &def.transitions {
            let from_id = name_to_id.get(t.from).ok_or_else(|| {
                anyhow::anyhow!(
                    "Transition from_state '{}' not found in protocol '{}'",
                    t.from,
                    def.name
                )
            })?;
            let to_id = name_to_id.get(t.to).ok_or_else(|| {
                anyhow::anyhow!(
                    "Transition to_state '{}' not found in protocol '{}'",
                    t.to,
                    def.name
                )
            })?;

            let mut pt = ProtocolTransition::new(proto.id, *from_id, *to_id, t.trigger);
            pt.guard = t.guard.map(|g| g.to_string());

            graph
                .upsert_protocol_transition(&pt)
                .await
                .with_context(|| {
                    format!(
                        "Failed to upsert transition '{} -> {}' in '{}'",
                        t.from, t.to, def.name
                    )
                })?;
        }

        info!(
            protocol = def.name,
            states = state_objects.len(),
            transitions = def.transitions.len(),
            "Seeded runner protocol"
        );
        details.push(format!(
            "{}: created ({} states, {} transitions)",
            def.name,
            state_objects.len(),
            def.transitions.len()
        ));
        created += 1;
    }

    info!(created, skipped, "Runner protocol seeding complete");

    Ok(RunnerSeedResult {
        created,
        skipped,
        details,
    })
}

// ============================================================================
// Protocol definitions
// ============================================================================

fn build_runner_protocol_defs() -> Vec<RunnerProtocolDef> {
    vec![
        build_plan_runner_full(),
        build_plan_runner_light(),
        build_plan_runner_reviewed(),
    ]
}

// ============================================================================
// 1. plan-runner-full (implementation lifecycle)
// ============================================================================

fn build_plan_runner_full() -> RunnerProtocolDef {
    RunnerProtocolDef {
        name: "plan-runner-full",
        description: "Standard plan runner lifecycle for implementation tasks. \
            Covers execution, post-run analysis, and optional PR creation.",
        relevance_vector: RelevanceVector {
            phase: 0.5,
            structure: 0.7,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        },
        states: vec![
            RunnerStateDef {
                name: "approved",
                state_type: StateType::Start,
                action: Some("plan(action: \"get\")"),
                prompt_fragment: "\
Plan is APPROVED and ready for execution. Load the full plan context: \
tasks, steps, constraints, and affected files. Verify all tasks have \
affected_files populated and dependencies are satisfiable. \
Call task(get_next) to identify the first executable task. \
Do NOT start executing until the plan structure is validated.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT start executing tasks before validating the plan structure",
                    "Do NOT modify the plan during the approved phase",
                ]),
            },
            RunnerStateDef {
                name: "executing",
                state_type: StateType::Intermediate,
                action: Some("task(action: \"get_next\")"),
                prompt_fragment: "\
EXECUTING tasks. Follow the Task Execution Protocol for each task: \
(1) task(get_next) to pick the next ready task, \
(2) task(get_context) to load knowledge and affected files, \
(3) update step statuses in real-time as you work, \
(4) run cargo check / tests after each significant change, \
(5) commit atomically with conventional format. \
Track all files modified for the post_run phase.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip step status updates — they enable real-time tracking",
                    "Do NOT commit without running cargo check first",
                    "Do NOT batch multiple tasks into a single commit",
                ]),
            },
            RunnerStateDef {
                name: "post_run",
                state_type: StateType::Intermediate,
                action: Some("commit(action: \"link_to_plan\")"),
                prompt_fragment: "\
POST-RUN analysis. All tasks are complete. Perform final validation: \
(1) Run the full test suite (cargo test), \
(2) Link all commits to the plan via commit(link_to_plan), \
(3) Create summary notes for patterns, gotchas, and decisions discovered, \
(4) Verify all task/step statuses are current (no stale in_progress). \
Prepare commit list for the PR decision phase.",
                available_tools: Some(vec!["commit", "note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the test suite — broken builds must not reach PR",
                    "Do NOT forget to link commits to the plan",
                ]),
            },
            RunnerStateDef {
                name: "pr_decision",
                state_type: StateType::Intermediate,
                action: Some("plan(action: \"get\")"),
                prompt_fragment: "\
PR DECISION. Evaluate whether a pull request should be created: \
(1) Check the number of files changed and commits made, \
(2) If changes are non-trivial (>1 file or >1 commit), create a PR, \
(3) Use gh pr create with a summary of changes, test results, and plan link, \
(4) If changes are trivial or the plan was exploratory, skip PR creation. \
Record the decision via decision(add) for traceability.",
                available_tools: Some(vec!["commit", "plan", "decision", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT create PRs for exploratory or trivial changes",
                    "Do NOT skip the decision record — PR choices must be traceable",
                ]),
            },
            RunnerStateDef {
                name: "completed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Plan execution COMPLETED successfully. All tasks are done, the test suite passes, \
commits are linked to the plan, and the PR decision has been recorded. \
The knowledge graph has been enriched with notes from this execution. No further action required.",
                available_tools: Some(vec!["note", "plan"]),
                forbidden_actions: None,
            },
            RunnerStateDef {
                name: "failed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Plan execution FAILED. Document the failure: (1) which task/step failed, \
(2) the error message or root cause, (3) what was completed before failure, \
(4) recommended next steps for retry or manual intervention. \
Create a gotcha note if the failure reveals a systemic issue.",
                available_tools: Some(vec!["note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT retry automatically without diagnosing the root cause",
                ]),
            },
        ],
        transitions: vec![
            RunnerTransitionDef {
                from: "approved",
                to: "executing",
                trigger: "run_started",
                guard: None,
            },
            RunnerTransitionDef {
                from: "approved",
                to: "failed",
                trigger: "execution_failed",
                guard: Some("plan validation failed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "post_run",
                trigger: "child_completed",
                guard: Some("all tasks completed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "failed",
                trigger: "execution_failed",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "pr_decision",
                trigger: "commits_linked",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "failed",
                trigger: "execution_failed",
                guard: Some("test suite failed"),
            },
            RunnerTransitionDef {
                from: "pr_decision",
                to: "completed",
                trigger: "pr_created",
                guard: None,
            },
            RunnerTransitionDef {
                from: "pr_decision",
                to: "completed",
                trigger: "skip_pr",
                guard: None,
            },
        ],
    }
}

// ============================================================================
// 2. plan-runner-light (exploration lifecycle)
// ============================================================================

fn build_plan_runner_light() -> RunnerProtocolDef {
    RunnerProtocolDef {
        name: "plan-runner-light",
        description: "Lightweight plan runner for exploration and research tasks. \
            No PR decision phase — execution flows directly to completion.",
        relevance_vector: RelevanceVector {
            phase: 0.25,
            structure: 0.3,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.3,
        },
        states: vec![
            RunnerStateDef {
                name: "approved",
                state_type: StateType::Start,
                action: Some("plan(action: \"get\")"),
                prompt_fragment: "\
Plan is APPROVED for lightweight execution. This is an exploration or research \
task — focus on learning and knowledge capture rather than production code. \
Load the plan context and verify task structure. \
Exploration plans may have fewer constraints and looser affected_files.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT enforce strict affected_files validation for exploration plans",
                    "Do NOT block execution for missing constraints",
                ]),
            },
            RunnerStateDef {
                name: "executing",
                state_type: StateType::Intermediate,
                action: Some("task(action: \"get_next\")"),
                prompt_fragment: "\
EXECUTING exploration tasks. For each task: \
(1) Focus on understanding and documenting findings, \
(2) Create notes liberally — exploration generates more knowledge than code, \
(3) Update step statuses as you progress, \
(4) Code changes are optional — the primary output is knowledge. \
Commit any code changes with conventional format.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip note creation — exploration without documentation is wasted effort",
                    "Do NOT over-engineer exploratory code",
                ]),
            },
            RunnerStateDef {
                name: "post_run",
                state_type: StateType::Intermediate,
                action: Some("note(action: \"create\")"),
                prompt_fragment: "\
POST-RUN wrap-up. Exploration complete. Synthesize findings: \
(1) Create a summary observation note with key discoveries, \
(2) Link any code changes to the plan via commit(link_to_plan), \
(3) If exploration revealed actionable work, document it as a follow-up task, \
(4) Verify all step statuses are finalized.",
                available_tools: Some(vec!["note", "commit", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the summary note — it captures the exploration's value",
                ]),
            },
            RunnerStateDef {
                name: "completed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Exploration COMPLETED. All findings have been documented as observation notes, \
key knowledge has been captured in the knowledge graph, and follow-up work \
has been identified if applicable. No PR is needed for exploration tasks.",
                available_tools: Some(vec!["note", "plan"]),
                forbidden_actions: None,
            },
            RunnerStateDef {
                name: "failed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Exploration FAILED. Document what was attempted, what blocked progress, \
and any partial findings that may be useful for future attempts. \
Create a gotcha note if the failure reveals environmental or tooling issues.",
                available_tools: Some(vec!["note", "task", "step", "plan"]),
                forbidden_actions: Some(vec![
                    "Do NOT discard partial findings — even failed explorations generate knowledge",
                ]),
            },
        ],
        transitions: vec![
            RunnerTransitionDef {
                from: "approved",
                to: "executing",
                trigger: "run_started",
                guard: None,
            },
            RunnerTransitionDef {
                from: "approved",
                to: "failed",
                trigger: "execution_failed",
                guard: Some("plan validation failed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "post_run",
                trigger: "child_completed",
                guard: Some("all tasks completed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "failed",
                trigger: "execution_failed",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "completed",
                trigger: "commits_linked",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "failed",
                trigger: "execution_failed",
                guard: None,
            },
        ],
    }
}

// ============================================================================
// 3. plan-runner-reviewed (critical lifecycle with review gate)
// ============================================================================

fn build_plan_runner_reviewed() -> RunnerProtocolDef {
    RunnerProtocolDef {
        name: "plan-runner-reviewed",
        description: "Critical plan runner lifecycle with mandatory review gate. \
            Like plan-runner-full but adds an awaiting_review state between \
            PR creation and completion for human approval.",
        relevance_vector: RelevanceVector {
            phase: 0.5,
            structure: 0.9,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.7,
        },
        states: vec![
            RunnerStateDef {
                name: "approved",
                state_type: StateType::Start,
                action: Some("plan(action: \"get\")"),
                prompt_fragment: "\
Plan is APPROVED for critical execution with mandatory review. \
This plan requires human review before completion. Load full context: \
tasks, steps, constraints, affected files. Validate thoroughly — \
critical plans have stricter quality gates. \
Verify all constraints are satisfiable before starting.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT start executing without validating ALL constraints",
                    "Do NOT relax quality gates for critical plans",
                ]),
            },
            RunnerStateDef {
                name: "executing",
                state_type: StateType::Intermediate,
                action: Some("task(action: \"get_next\")"),
                prompt_fragment: "\
EXECUTING critical tasks. Apply heightened rigor: \
(1) Run tests after EVERY change, not just significant ones, \
(2) Document every decision with decision(add) — reviewers need context, \
(3) Create detailed commit messages explaining the 'why', \
(4) Track all files modified for comprehensive review preparation. \
Quality over speed — this code will be reviewed.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip tests between changes — critical code needs continuous validation",
                    "Do NOT make undocumented decisions — reviewers need full context",
                    "Do NOT combine unrelated changes in a single commit",
                ]),
            },
            RunnerStateDef {
                name: "post_run",
                state_type: StateType::Intermediate,
                action: Some("commit(action: \"link_to_plan\")"),
                prompt_fragment: "\
POST-RUN validation for critical changes. Perform thorough checks: \
(1) Run the full test suite AND any integration tests, \
(2) Link ALL commits to the plan, \
(3) Create comprehensive notes documenting every change rationale, \
(4) Prepare a detailed PR description with test evidence, \
(5) List any risks or caveats for the reviewer.",
                available_tools: Some(vec!["commit", "note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip integration tests for critical changes",
                    "Do NOT create the PR without comprehensive documentation",
                ]),
            },
            RunnerStateDef {
                name: "pr_decision",
                state_type: StateType::Intermediate,
                action: Some("plan(action: \"get\")"),
                prompt_fragment: "\
PR CREATION for reviewed changes. A PR is MANDATORY for critical plans. \
Create the PR with: (1) detailed summary of all changes, \
(2) test results and evidence, (3) risk assessment, \
(4) link to the plan and related decisions. \
The PR must be ready for review — no draft PRs for critical changes.",
                available_tools: Some(vec!["commit", "plan", "decision", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip PR creation — it is mandatory for critical plans",
                    "Do NOT create draft PRs — critical changes need immediate review readiness",
                ]),
            },
            RunnerStateDef {
                name: "awaiting_review",
                state_type: StateType::Intermediate,
                action: None,
                prompt_fragment: "\
AWAITING REVIEW. The PR has been created and is pending human approval. \
While waiting: (1) Monitor for review comments via note(search), \
(2) Respond to reviewer questions with additional context, \
(3) If changes are requested, address them and update the PR. \
Do NOT merge or mark as complete until review_approved is received.",
                available_tools: Some(vec!["note", "chat", "decision", "commit"]),
                forbidden_actions: Some(vec![
                    "Do NOT mark the plan as completed before review approval",
                    "Do NOT ignore reviewer feedback — address every comment",
                    "Do NOT force-merge without explicit approval",
                ]),
            },
            RunnerStateDef {
                name: "completed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Plan execution COMPLETED with review approval. All tasks done, tests pass, \
PR approved and merged. Record the approval as a decision for traceability.",
                available_tools: Some(vec!["note", "plan", "decision"]),
                forbidden_actions: None,
            },
            RunnerStateDef {
                name: "failed",
                state_type: StateType::Terminal,
                action: None,
                prompt_fragment: "\
Plan execution FAILED. Document comprehensively: (1) failure point and root cause, \
(2) review feedback if rejection-related, (3) all completed work, \
(4) specific remediation steps. Critical failures may need escalation — \
flag for human attention via a high-importance gotcha note.",
                available_tools: Some(vec!["note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT retry critical failures without human acknowledgment",
                ]),
            },
        ],
        transitions: vec![
            RunnerTransitionDef {
                from: "approved",
                to: "executing",
                trigger: "run_started",
                guard: None,
            },
            RunnerTransitionDef {
                from: "approved",
                to: "failed",
                trigger: "execution_failed",
                guard: Some("plan validation failed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "post_run",
                trigger: "child_completed",
                guard: Some("all tasks completed"),
            },
            RunnerTransitionDef {
                from: "executing",
                to: "failed",
                trigger: "execution_failed",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "pr_decision",
                trigger: "commits_linked",
                guard: None,
            },
            RunnerTransitionDef {
                from: "post_run",
                to: "failed",
                trigger: "execution_failed",
                guard: Some("test suite failed"),
            },
            RunnerTransitionDef {
                from: "pr_decision",
                to: "awaiting_review",
                trigger: "pr_created",
                guard: None,
            },
            RunnerTransitionDef {
                from: "awaiting_review",
                to: "completed",
                trigger: "review_approved",
                guard: None,
            },
            RunnerTransitionDef {
                from: "awaiting_review",
                to: "failed",
                trigger: "review_rejected",
                guard: None,
            },
        ],
    }
}

// ============================================================================
// ProtocolSeed fragments (for seed_prompt_fragments compatibility)
// ============================================================================

use super::seed::{ProtocolSeed, StateFragment};

pub(crate) fn seed_plan_runner_full() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "plan-runner-full",
        states: vec![
            StateFragment {
                state_name: "approved",
                prompt_fragment: "\
Plan is APPROVED and ready for execution. Load the full plan context: \
tasks, steps, constraints, and affected files. Verify all tasks have \
affected_files populated and dependencies are satisfiable. \
Call task(get_next) to identify the first executable task. \
Do NOT start executing until the plan structure is validated.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT start executing tasks before validating the plan structure",
                    "Do NOT modify the plan during the approved phase",
                ]),
            },
            StateFragment {
                state_name: "executing",
                prompt_fragment: "\
EXECUTING tasks. Follow the Task Execution Protocol for each task: \
(1) task(get_next) to pick the next ready task, \
(2) task(get_context) to load knowledge and affected files, \
(3) update step statuses in real-time as you work, \
(4) run cargo check / tests after each significant change, \
(5) commit atomically with conventional format. \
Track all files modified for the post_run phase.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip step status updates — they enable real-time tracking",
                    "Do NOT commit without running cargo check first",
                    "Do NOT batch multiple tasks into a single commit",
                ]),
            },
            StateFragment {
                state_name: "post_run",
                prompt_fragment: "\
POST-RUN analysis. All tasks are complete. Perform final validation: \
(1) Run the full test suite (cargo test), \
(2) Link all commits to the plan via commit(link_to_plan), \
(3) Create summary notes for patterns, gotchas, and decisions discovered, \
(4) Verify all task/step statuses are current (no stale in_progress). \
Prepare commit list for the PR decision phase.",
                available_tools: Some(vec!["commit", "note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the test suite — broken builds must not reach PR",
                    "Do NOT forget to link commits to the plan",
                ]),
            },
            StateFragment {
                state_name: "pr_decision",
                prompt_fragment: "\
PR DECISION. Evaluate whether a pull request should be created: \
(1) Check the number of files changed and commits made, \
(2) If changes are non-trivial (>1 file or >1 commit), create a PR, \
(3) Use gh pr create with a summary of changes, test results, and plan link, \
(4) If changes are trivial or the plan was exploratory, skip PR creation. \
Record the decision via decision(add) for traceability.",
                available_tools: Some(vec!["commit", "plan", "decision", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT create PRs for exploratory or trivial changes",
                    "Do NOT skip the decision record — PR choices must be traceable",
                ]),
            },
            StateFragment {
                state_name: "completed",
                prompt_fragment: "\
Plan execution COMPLETED successfully. All tasks are done, the test suite passes, \
commits are linked to the plan, and the PR decision has been recorded. \
The knowledge graph has been enriched with notes from this execution. No further action required.",
                available_tools: Some(vec!["note", "plan"]),
                forbidden_actions: None,
            },
            StateFragment {
                state_name: "failed",
                prompt_fragment: "\
Plan execution FAILED. Document the failure: (1) which task/step failed, \
(2) the error message or root cause, (3) what was completed before failure, \
(4) recommended next steps for retry or manual intervention. \
Create a gotcha note if the failure reveals a systemic issue.",
                available_tools: Some(vec!["note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT retry automatically without diagnosing the root cause",
                ]),
            },
        ],
    }
}

pub(crate) fn seed_plan_runner_light() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "plan-runner-light",
        states: vec![
            StateFragment {
                state_name: "approved",
                prompt_fragment: "\
Plan is APPROVED for lightweight execution. This is an exploration or research \
task — focus on learning and knowledge capture rather than production code. \
Load the plan context and verify task structure. \
Exploration plans may have fewer constraints and looser affected_files.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT enforce strict affected_files validation for exploration plans",
                    "Do NOT block execution for missing constraints",
                ]),
            },
            StateFragment {
                state_name: "executing",
                prompt_fragment: "\
EXECUTING exploration tasks. For each task: \
(1) Focus on understanding and documenting findings, \
(2) Create notes liberally — exploration generates more knowledge than code, \
(3) Update step statuses as you progress, \
(4) Code changes are optional — the primary output is knowledge. \
Commit any code changes with conventional format.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip note creation — exploration without documentation is wasted effort",
                    "Do NOT over-engineer exploratory code",
                ]),
            },
            StateFragment {
                state_name: "post_run",
                prompt_fragment: "\
POST-RUN wrap-up. Exploration complete. Synthesize findings: \
(1) Create a summary observation note with key discoveries, \
(2) Link any code changes to the plan via commit(link_to_plan), \
(3) If exploration revealed actionable work, document it as a follow-up task, \
(4) Verify all step statuses are finalized.",
                available_tools: Some(vec!["note", "commit", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the summary note — it captures the exploration's value",
                ]),
            },
            StateFragment {
                state_name: "completed",
                prompt_fragment: "\
Exploration COMPLETED. All findings have been documented as observation notes, \
key knowledge has been captured in the knowledge graph, and follow-up work \
has been identified if applicable. No PR is needed for exploration tasks.",
                available_tools: Some(vec!["note", "plan"]),
                forbidden_actions: None,
            },
            StateFragment {
                state_name: "failed",
                prompt_fragment: "\
Exploration FAILED. Document what was attempted, what blocked progress, \
and any partial findings that may be useful for future attempts. \
Create a gotcha note if the failure reveals environmental or tooling issues.",
                available_tools: Some(vec!["note", "task", "step", "plan"]),
                forbidden_actions: Some(vec![
                    "Do NOT discard partial findings — even failed explorations generate knowledge",
                ]),
            },
        ],
    }
}

pub(crate) fn seed_plan_runner_reviewed() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "plan-runner-reviewed",
        states: vec![
            StateFragment {
                state_name: "approved",
                prompt_fragment: "\
Plan is APPROVED for critical execution with mandatory review. \
This plan requires human review before completion. Load full context: \
tasks, steps, constraints, affected files. Validate thoroughly — \
critical plans have stricter quality gates. \
Verify all constraints are satisfiable before starting.",
                available_tools: Some(vec!["plan", "task", "step", "constraint", "code", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT start executing without validating ALL constraints",
                    "Do NOT relax quality gates for critical plans",
                ]),
            },
            StateFragment {
                state_name: "executing",
                prompt_fragment: "\
EXECUTING critical tasks. Apply heightened rigor: \
(1) Run tests after EVERY change, not just significant ones, \
(2) Document every decision with decision(add) — reviewers need context, \
(3) Create detailed commit messages explaining the 'why', \
(4) Track all files modified for comprehensive review preparation. \
Quality over speed — this code will be reviewed.",
                available_tools: None,
                forbidden_actions: Some(vec![
                    "Do NOT skip tests between changes — critical code needs continuous validation",
                    "Do NOT make undocumented decisions — reviewers need full context",
                    "Do NOT combine unrelated changes in a single commit",
                ]),
            },
            StateFragment {
                state_name: "post_run",
                prompt_fragment: "\
POST-RUN validation for critical changes. Perform thorough checks: \
(1) Run the full test suite AND any integration tests, \
(2) Link ALL commits to the plan, \
(3) Create comprehensive notes documenting every change rationale, \
(4) Prepare a detailed PR description with test evidence, \
(5) List any risks or caveats for the reviewer.",
                available_tools: Some(vec!["commit", "note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip integration tests for critical changes",
                    "Do NOT create the PR without comprehensive documentation",
                ]),
            },
            StateFragment {
                state_name: "pr_decision",
                prompt_fragment: "\
PR CREATION for reviewed changes. A PR is MANDATORY for critical plans. \
Create the PR with: (1) detailed summary of all changes, \
(2) test results and evidence, (3) risk assessment, \
(4) link to the plan and related decisions. \
The PR must be ready for review — no draft PRs for critical changes.",
                available_tools: Some(vec!["commit", "plan", "decision", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip PR creation — it is mandatory for critical plans",
                    "Do NOT create draft PRs — critical changes need immediate review readiness",
                ]),
            },
            StateFragment {
                state_name: "awaiting_review",
                prompt_fragment: "\
AWAITING REVIEW. The PR has been created and is pending human approval. \
While waiting: (1) Monitor for review comments via note(search), \
(2) Respond to reviewer questions with additional context, \
(3) If changes are requested, address them and update the PR. \
Do NOT merge or mark as complete until review_approved is received.",
                available_tools: Some(vec!["note", "chat", "decision", "commit"]),
                forbidden_actions: Some(vec![
                    "Do NOT mark the plan as completed before review approval",
                    "Do NOT ignore reviewer feedback — address every comment",
                    "Do NOT force-merge without explicit approval",
                ]),
            },
            StateFragment {
                state_name: "completed",
                prompt_fragment: "\
Plan execution COMPLETED with review approval. All tasks done, tests pass, \
PR approved and merged. Record the approval as a decision for traceability.",
                available_tools: Some(vec!["note", "plan", "decision"]),
                forbidden_actions: None,
            },
            StateFragment {
                state_name: "failed",
                prompt_fragment: "\
Plan execution FAILED. Document comprehensively: (1) failure point and root cause, \
(2) review feedback if rejection-related, (3) all completed work, \
(4) specific remediation steps. Critical failures may need escalation — \
flag for human attention via a high-importance gotcha note.",
                available_tools: Some(vec!["note", "task", "step", "plan", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT retry critical failures without human acknowledgment",
                ]),
            },
        ],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_protocol_defs_count() {
        let defs = build_runner_protocol_defs();
        assert_eq!(defs.len(), 3);
    }

    #[test]
    fn test_plan_runner_full_states() {
        let def = build_plan_runner_full();
        assert_eq!(def.name, "plan-runner-full");
        let names: Vec<&str> = def.states.iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            vec![
                "approved",
                "executing",
                "post_run",
                "pr_decision",
                "completed",
                "failed"
            ]
        );
        // Check state types
        assert_eq!(def.states[0].state_type, StateType::Start);
        assert_eq!(def.states[1].state_type, StateType::Intermediate);
        assert_eq!(def.states[4].state_type, StateType::Terminal);
        assert_eq!(def.states[5].state_type, StateType::Terminal);
        // Check transitions
        assert_eq!(def.transitions.len(), 8);
    }

    #[test]
    fn test_plan_runner_light_no_pr_decision() {
        let def = build_plan_runner_light();
        assert_eq!(def.name, "plan-runner-light");
        let names: Vec<&str> = def.states.iter().map(|s| s.name).collect();
        assert!(
            !names.contains(&"pr_decision"),
            "Light runner should not have pr_decision"
        );
        assert_eq!(names.len(), 5);
        assert_eq!(def.transitions.len(), 6);
    }

    #[test]
    fn test_plan_runner_reviewed_has_awaiting_review() {
        let def = build_plan_runner_reviewed();
        assert_eq!(def.name, "plan-runner-reviewed");
        let names: Vec<&str> = def.states.iter().map(|s| s.name).collect();
        assert!(names.contains(&"awaiting_review"));
        assert_eq!(names.len(), 7);
        // awaiting_review should have 2 outgoing transitions
        let awaiting_transitions: Vec<_> = def
            .transitions
            .iter()
            .filter(|t| t.from == "awaiting_review")
            .collect();
        assert_eq!(
            awaiting_transitions.len(),
            2,
            "awaiting_review should have 2 outgoing transitions (review_approved, review_rejected)"
        );
        assert_eq!(def.transitions.len(), 9);
    }

    #[test]
    fn test_all_runner_fragments_non_empty() {
        let defs = build_runner_protocol_defs();
        for def in &defs {
            for state in &def.states {
                assert!(
                    !state.prompt_fragment.is_empty(),
                    "Empty prompt_fragment for {}/{}",
                    def.name,
                    state.name
                );
                let word_count = state.prompt_fragment.split_whitespace().count();
                assert!(
                    word_count >= 20,
                    "Fragment too short ({} words) for {}/{}",
                    word_count,
                    def.name,
                    state.name
                );
                assert!(
                    word_count <= 300,
                    "Fragment too long ({} words) for {}/{}",
                    word_count,
                    def.name,
                    state.name
                );
            }
        }
    }

    #[test]
    fn test_all_runner_forbidden_actions_format() {
        let defs = build_runner_protocol_defs();
        for def in &defs {
            for state in &def.states {
                if let Some(actions) = &state.forbidden_actions {
                    for action in actions {
                        assert!(
                            action.starts_with("Do NOT"),
                            "Forbidden action should start with 'Do NOT': got '{}' in {}/{}",
                            action,
                            def.name,
                            state.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_runner_relevance_vectors() {
        let full = build_plan_runner_full();
        assert!((full.relevance_vector.phase - 0.5).abs() < f64::EPSILON);
        assert!((full.relevance_vector.structure - 0.7).abs() < f64::EPSILON);

        let light = build_plan_runner_light();
        assert!((light.relevance_vector.phase - 0.25).abs() < f64::EPSILON);
        assert!((light.relevance_vector.structure - 0.3).abs() < f64::EPSILON);

        let reviewed = build_plan_runner_reviewed();
        assert!((reviewed.relevance_vector.structure - 0.9).abs() < f64::EPSILON);
        assert!((reviewed.relevance_vector.lifecycle - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_runner_available_tools_are_valid() {
        let valid_tools = [
            "project",
            "plan",
            "task",
            "step",
            "decision",
            "constraint",
            "release",
            "milestone",
            "commit",
            "note",
            "workspace",
            "workspace_milestone",
            "resource",
            "component",
            "chat",
            "feature_graph",
            "code",
            "reasoning",
            "admin",
            "skill",
            "analysis_profile",
            "protocol",
            "persona",
            "episode",
            "sharing",
        ];

        let defs = build_runner_protocol_defs();
        for def in &defs {
            for state in &def.states {
                if let Some(tools) = &state.available_tools {
                    for tool in tools {
                        assert!(
                            valid_tools.contains(tool),
                            "Invalid tool '{}' in {}/{}",
                            tool,
                            def.name,
                            state.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_runner_unique_state_names() {
        let defs = build_runner_protocol_defs();
        for def in &defs {
            let mut seen = std::collections::HashSet::new();
            for state in &def.states {
                assert!(
                    seen.insert(state.name),
                    "Duplicate state name '{}' in protocol '{}'",
                    state.name,
                    def.name
                );
            }
        }
    }

    #[test]
    fn test_runner_transitions_reference_valid_states() {
        let defs = build_runner_protocol_defs();
        for def in &defs {
            let state_names: std::collections::HashSet<&str> =
                def.states.iter().map(|s| s.name).collect();
            for t in &def.transitions {
                assert!(
                    state_names.contains(t.from),
                    "Transition from '{}' references non-existent state in '{}'",
                    t.from,
                    def.name
                );
                assert!(
                    state_names.contains(t.to),
                    "Transition to '{}' references non-existent state in '{}'",
                    t.to,
                    def.name
                );
            }
        }
    }

    #[test]
    fn test_runner_fragments_end_with_punctuation() {
        let defs = build_runner_protocol_defs();
        for def in &defs {
            for state in &def.states {
                let trimmed = state.prompt_fragment.trim();
                assert!(
                    trimmed.ends_with('.') || trimmed.ends_with(':') || trimmed.ends_with(')'),
                    "Fragment for {}/{} should end with punctuation, got: ...{}",
                    def.name,
                    state.name,
                    &trimmed[trimmed.len().saturating_sub(20)..],
                );
            }
        }
    }

    // Async tests for seed_runner_protocols
    use crate::neo4j::mock::MockGraphStore;

    #[tokio::test]
    async fn test_seed_runner_protocols_creates_all_three() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let result = seed_runner_protocols(&mock, project_id).await.unwrap();

        assert_eq!(result.created, 3);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.details.len(), 3);

        // Verify protocols exist
        let full = mock
            .get_protocol_by_name_and_project("plan-runner-full", project_id)
            .await
            .unwrap();
        assert!(full.is_some(), "plan-runner-full should exist");

        let light = mock
            .get_protocol_by_name_and_project("plan-runner-light", project_id)
            .await
            .unwrap();
        assert!(light.is_some(), "plan-runner-light should exist");

        let reviewed = mock
            .get_protocol_by_name_and_project("plan-runner-reviewed", project_id)
            .await
            .unwrap();
        assert!(reviewed.is_some(), "plan-runner-reviewed should exist");
    }

    #[tokio::test]
    async fn test_seed_runner_protocols_idempotent() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Run twice
        let result1 = seed_runner_protocols(&mock, project_id).await.unwrap();
        let result2 = seed_runner_protocols(&mock, project_id).await.unwrap();

        assert_eq!(result1.created, 3);
        assert_eq!(result1.skipped, 0);
        assert_eq!(result2.created, 0);
        assert_eq!(result2.skipped, 3);

        // Should still only have 3 protocols
        let protocols = mock.protocols.read().await;
        let runner_protocols: Vec<_> = protocols
            .values()
            .filter(|p| p.name.starts_with("plan-runner-"))
            .collect();
        assert_eq!(runner_protocols.len(), 3, "No duplicates after double seed");
    }

    #[tokio::test]
    async fn test_seed_runner_protocols_states_have_fragments() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        seed_runner_protocols(&mock, project_id).await.unwrap();

        // Check all states have prompt_fragment set
        let states = mock.protocol_states.read().await;
        for state in states.values() {
            assert!(
                state.prompt_fragment.is_some(),
                "State '{}' should have prompt_fragment after compose",
                state.name
            );
        }
    }

    #[tokio::test]
    async fn test_seed_runner_full_has_correct_state_count() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        seed_runner_protocols(&mock, project_id).await.unwrap();

        let full_id = mock
            .get_protocol_by_name_and_project("plan-runner-full", project_id)
            .await
            .unwrap()
            .unwrap();

        let states = mock.get_protocol_states(full_id).await.unwrap();
        assert_eq!(states.len(), 6);
    }

    #[tokio::test]
    async fn test_seed_runner_reviewed_has_awaiting_review_state() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        seed_runner_protocols(&mock, project_id).await.unwrap();

        let reviewed_id = mock
            .get_protocol_by_name_and_project("plan-runner-reviewed", project_id)
            .await
            .unwrap()
            .unwrap();

        let states = mock.get_protocol_states(reviewed_id).await.unwrap();
        let state_names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert!(
            state_names.contains(&"awaiting_review"),
            "plan-runner-reviewed should have awaiting_review state"
        );

        let transitions = mock.get_protocol_transitions(reviewed_id).await.unwrap();
        // Find transitions FROM awaiting_review
        let awaiting_state = states.iter().find(|s| s.name == "awaiting_review").unwrap();
        let outgoing: Vec<_> = transitions
            .iter()
            .filter(|t| t.from_state == awaiting_state.id)
            .collect();
        assert_eq!(
            outgoing.len(),
            2,
            "awaiting_review should have 2 outgoing transitions"
        );
    }

    #[tokio::test]
    async fn test_seed_runner_light_no_pr_decision_state() {
        let mock = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        seed_runner_protocols(&mock, project_id).await.unwrap();

        let light_id = mock
            .get_protocol_by_name_and_project("plan-runner-light", project_id)
            .await
            .unwrap()
            .unwrap();

        let states = mock.get_protocol_states(light_id).await.unwrap();
        let state_names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert!(
            !state_names.contains(&"pr_decision"),
            "plan-runner-light should NOT have pr_decision state"
        );
        assert_eq!(states.len(), 5);
    }
}
