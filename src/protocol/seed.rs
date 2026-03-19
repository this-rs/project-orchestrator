//! Seed module for protocol prompt fragments.
//!
//! Populates `prompt_fragment`, `available_tools`, and `forbidden_actions`
//! on the 5 critical protocols' states. Matches by protocol name + state name
//! (not UUID) so the seed is portable across environments.
//!
//! Run via `admin(action: "seed_prompt_fragments")` or the REST endpoint
//! `POST /api/admin/seed-prompt-fragments`.

use anyhow::{Context, Result};
use tracing::info;
use uuid::Uuid;

use crate::neo4j::GraphStore;

/// A single state fragment definition.
struct StateFragment {
    state_name: &'static str,
    prompt_fragment: &'static str,
    available_tools: Option<Vec<&'static str>>,
    forbidden_actions: Option<Vec<&'static str>>,
}

/// A protocol with all its state fragments.
struct ProtocolSeed {
    protocol_name: &'static str,
    states: Vec<StateFragment>,
}

/// Result of the seed operation.
#[derive(Debug, serde::Serialize)]
pub struct SeedResult {
    pub updated: usize,
    pub skipped: usize,
    pub protocols_found: usize,
    pub protocols_missing: Vec<String>,
}

/// Seed prompt fragments for the 5 critical protocols.
///
/// Uses `GraphStore` trait methods to find protocols by name, fetch their states,
/// enrich them with prompt fragments, and upsert back.
///
/// Idempotent — safe to run multiple times.
pub async fn seed_prompt_fragments(graph: &dyn GraphStore, project_id: Uuid) -> Result<SeedResult> {
    let protocol_seeds = build_protocol_seeds();
    let mut updated = 0;
    let mut skipped = 0;
    let mut protocols_found = 0;
    let mut protocols_missing = Vec::new();

    for proto_seed in &protocol_seeds {
        // Find the protocol by name in this project
        let protocol_id = match graph
            .get_protocol_by_name_and_project(proto_seed.protocol_name, project_id)
            .await
            .with_context(|| format!("Failed to look up protocol '{}'", proto_seed.protocol_name))?
        {
            Some(id) => {
                protocols_found += 1;
                id
            }
            None => {
                info!(
                    protocol = proto_seed.protocol_name,
                    "Protocol not found in project — skipping"
                );
                protocols_missing.push(proto_seed.protocol_name.to_string());
                skipped += proto_seed.states.len();
                continue;
            }
        };

        // Fetch existing states
        let existing_states = graph
            .get_protocol_states(protocol_id)
            .await
            .with_context(|| {
                format!(
                    "Failed to get states for protocol '{}'",
                    proto_seed.protocol_name
                )
            })?;

        // For each seed fragment, find the matching state and upsert
        for fragment in &proto_seed.states {
            if let Some(mut state) = existing_states
                .iter()
                .find(|s| s.name == fragment.state_name)
                .cloned()
            {
                // Enrich the state with prompt fragment data
                state.prompt_fragment = Some(fragment.prompt_fragment.to_string());
                state.available_tools = fragment
                    .available_tools
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.to_string()).collect());
                state.forbidden_actions = fragment
                    .forbidden_actions
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.to_string()).collect());

                graph.upsert_protocol_state(&state).await.with_context(|| {
                    format!(
                        "Failed to upsert state '{}/{}",
                        proto_seed.protocol_name, fragment.state_name
                    )
                })?;

                updated += 1;
                info!(
                    protocol = proto_seed.protocol_name,
                    state = fragment.state_name,
                    "Seeded prompt fragment"
                );
            } else {
                skipped += 1;
                info!(
                    protocol = proto_seed.protocol_name,
                    state = fragment.state_name,
                    "State not found — skipped"
                );
            }
        }
    }

    info!(
        updated,
        skipped, protocols_found, "Prompt fragment seeding complete"
    );

    Ok(SeedResult {
        updated,
        skipped,
        protocols_found,
        protocols_missing,
    })
}

fn build_protocol_seeds() -> Vec<ProtocolSeed> {
    vec![
        seed_session_lifecycle(),
        seed_rfc_lifecycle(),
        seed_wave_dispatch(),
        seed_diagnostic_triage(),
        seed_auto_maintenance(),
    ]
}

// ============================================================================
// 1. session-lifecycle
// ============================================================================

fn seed_session_lifecycle() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "session-lifecycle",
        states: vec![
            StateFragment {
                state_name: "warm_up",
                prompt_fragment: "\
You are in the WARM-UP phase. Your sole objective is to load relevant context \
before any work begins. Execute at least 3 of these 5 actions: \
(1) note(search_semantic) for the user's topic, \
(2) note(get_context) on relevant files, \
(3) decision(search_semantic) for past architectural choices, \
(4) note(get_propagated) for files likely to be touched, \
(5) note(search) with exact keywords. \
Summarize what you found before proceeding. Do NOT start modifying code yet.",
                available_tools: Some(vec!["note", "decision", "code", "project", "task", "plan"]),
                forbidden_actions: Some(vec![
                    "Do NOT edit or write any file during warm-up",
                    "Do NOT create commits",
                    "Do NOT skip warm-up even if the task seems simple",
                ]),
            },
            StateFragment {
                state_name: "working",
                prompt_fragment: "\
You are in the WORKING phase. Apply the impact-first protocol before each file \
modification: check node importance, load propagated notes, verify topology rules. \
After each significant change, capture knowledge: create notes for gotchas, patterns, \
and conventions discovered. Update task/step statuses in real time. \
Track every file you read or modify for the closing phase.",
                available_tools: None, // all tools allowed
                forbidden_actions: Some(vec![
                    "Do NOT modify files without running analyze_impact first",
                    "Do NOT batch status updates — update after each step completion",
                ]),
            },
            StateFragment {
                state_name: "checkpoint",
                prompt_fragment: "\
CHECKPOINT — pause and reflect. Count files modified vs notes created. \
If the ratio exceeds 5:1, you are losing knowledge. For each unrecorded \
discovery (bug root cause, pattern, convention, gotcha), create a note NOW. \
Review open task/step statuses — are any stale? \
If you've been working for a while, consider whether the current approach \
is still optimal or if you should pivot.",
                available_tools: Some(vec!["note", "decision", "task", "step", "commit"]),
                forbidden_actions: Some(vec![
                    "Do NOT continue coding without completing the reflection",
                    "Do NOT skip note creation if the ratio is above 5:1",
                ]),
            },
            StateFragment {
                state_name: "closing",
                prompt_fragment: "\
SESSION CLOSING — externalize all remaining knowledge before context is lost. \
(1) Create a session summary observation note listing: files modified, decisions made, \
notes created, open questions. \
(2) Call chat(add_discussed) with ALL files significantly touched. \
(3) Verify every task/step status is current (no stale in_progress). \
(4) If work is incomplete, document the exact resumption point. \
This is your last chance to persist what you learned.",
                available_tools: Some(vec!["note", "chat", "task", "step", "commit", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT start new features during closing",
                    "Do NOT skip chat(add_discussed) — it feeds the Knowledge Fabric",
                ]),
            },
            StateFragment {
                state_name: "closed",
                prompt_fragment: "\
Session is closed. All knowledge has been externalized, all statuses are current, \
and discussed entities are tracked. No further actions needed.",
                available_tools: None,
                forbidden_actions: None,
            },
        ],
    }
}

// ============================================================================
// 2. rfc-lifecycle
// ============================================================================

fn seed_rfc_lifecycle() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "rfc-lifecycle",
        states: vec![
            StateFragment {
                state_name: "draft",
                prompt_fragment: "\
You are DRAFTING an RFC. Write a structured proposal with these mandatory sections: \
**Problem** (what issue or need), **Proposed Solution** (detailed approach with code \
examples if relevant), **Alternatives** (other options considered and why rejected), \
**Impact** (affected files, components, breaking changes, migration path). \
Use code(analyze_impact) and code(get_architecture) to ground the proposal in \
actual codebase structure. Create the RFC as note(type: rfc, importance: high).",
                available_tools: Some(vec!["note", "code", "decision", "project"]),
                forbidden_actions: Some(vec![
                    "Do NOT implement any code changes during draft phase",
                    "Do NOT skip the Alternatives section",
                    "Do NOT propose changes without running analyze_impact first",
                ]),
            },
            StateFragment {
                state_name: "proposed",
                prompt_fragment: "\
RFC is PROPOSED and awaiting review. Ensure the RFC note is linked to the project \
and any relevant files via note(link_to_entity). Notify stakeholders by summarizing \
the key trade-offs. Prepare a list of specific questions for reviewers to focus on. \
Do not proceed to implementation until the RFC advances to accepted.",
                available_tools: Some(vec!["note", "code", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT start implementation — the RFC is not yet accepted",
                    "Do NOT modify the RFC content without re-proposing",
                ]),
            },
            StateFragment {
                state_name: "under_review",
                prompt_fragment: "\
RFC is UNDER REVIEW. Collect and address feedback systematically. For each piece of \
feedback: (1) acknowledge it, (2) if it requires changes, update the RFC content, \
(3) if you disagree, document your reasoning. Track all review comments. \
When all feedback is addressed, the RFC can advance to accepted or be rejected.",
                available_tools: Some(vec!["note", "decision", "code"]),
                forbidden_actions: Some(vec![
                    "Do NOT ignore reviewer feedback",
                    "Do NOT start implementation while review is ongoing",
                ]),
            },
            StateFragment {
                state_name: "accepted",
                prompt_fragment: "\
RFC is ACCEPTED. Record this as a formal architectural decision via \
decision(add) with the rationale, alternatives considered, and chosen option. \
Link the decision to all affected files using decision(add_affects). \
The RFC is now ready for planning — advance to the planning state.",
                available_tools: Some(vec!["note", "decision", "plan", "code"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip recording the formal decision",
                    "Do NOT start coding without creating a plan first",
                ]),
            },
            StateFragment {
                state_name: "planning",
                prompt_fragment: "\
Create an implementation PLAN for the accepted RFC. Use the Planning Protocol: \
(1) plan(create) linked to the project, \
(2) Decompose into tasks with steps (minimum 2-3 steps per task), \
(3) Define task dependencies and affected_files, \
(4) Add constraints from the RFC (performance, compatibility, security), \
(5) Link the RFC note to the plan via note(link_to_entity, plan). \
Use plan(get_waves) to verify parallelizability.",
                available_tools: Some(vec![
                    "plan",
                    "task",
                    "step",
                    "constraint",
                    "note",
                    "milestone",
                    "code",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT create tasks without steps",
                    "Do NOT skip affected_files on tasks — wave dispatch needs them",
                    "Do NOT forget to link the RFC note to the plan",
                ]),
            },
            StateFragment {
                state_name: "in_progress",
                prompt_fragment: "\
RFC implementation is IN PROGRESS. Execute tasks following the Task Execution Protocol: \
get_next → get_context → update status → execute steps → commit → link. \
After each commit, use commit(link_to_plan) to maintain traceability. \
When all tasks are complete, verify the RFC's acceptance criteria are met \
before advancing to implemented.",
                available_tools: None, // all tools needed during implementation
                forbidden_actions: Some(vec![
                    "Do NOT skip commit linking — traceability is mandatory",
                    "Do NOT mark tasks completed without verifying acceptance criteria",
                ]),
            },
            StateFragment {
                state_name: "implemented",
                prompt_fragment: "\
RFC is IMPLEMENTED. All tasks are complete and acceptance criteria are met. \
Create a final summary note documenting: what was built, key decisions made \
during implementation, any deviations from the original proposal, and lessons learned. \
Link the final commit to the plan. The RFC lifecycle is complete.",
                available_tools: Some(vec!["note", "commit", "plan", "decision"]),
                forbidden_actions: None,
            },
            StateFragment {
                state_name: "rejected",
                prompt_fragment: "\
RFC has been REJECTED. Document the rejection rationale as a decision note. \
Record what was learned from the proposal process — even rejected RFCs \
generate valuable knowledge about constraints, trade-offs, and requirements. \
Link the rejection decision to the RFC note.",
                available_tools: Some(vec!["note", "decision"]),
                forbidden_actions: Some(vec!["Do NOT implement any part of a rejected RFC"]),
            },
            StateFragment {
                state_name: "superseded",
                prompt_fragment: "\
RFC has been SUPERSEDED by a newer proposal. Use note(supersede) to link \
the old RFC to its successor. Ensure the successor RFC references the \
original's rationale and explains what changed. Preserve the knowledge chain.",
                available_tools: Some(vec!["note", "decision"]),
                forbidden_actions: Some(vec![
                    "Do NOT delete the superseded RFC — it's part of the decision history",
                ]),
            },
        ],
    }
}

// ============================================================================
// 3. wave-dispatch
// ============================================================================

fn seed_wave_dispatch() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "wave-dispatch",
        states: vec![
            StateFragment {
                state_name: "compute_waves",
                prompt_fragment: "\
COMPUTE execution waves. Call plan(get_waves, plan_id) to get the topological sort. \
Verify that ALL tasks have affected_files populated — empty affected_files means \
the wave splitter cannot detect conflicts. If any task lacks affected_files, \
call task(enrich) or manually add them before proceeding. \
Review the wave structure: tasks in the same wave must not share affected_files.",
                available_tools: Some(vec!["plan", "task", "step", "code", "constraint"]),
                forbidden_actions: Some(vec![
                    "Do NOT dispatch agents before waves are computed and validated",
                    "Do NOT proceed if tasks have empty affected_files",
                ]),
            },
            StateFragment {
                state_name: "prepare_wave",
                prompt_fragment: "\
PREPARE the current wave. For each task in this wave: \
(1) task(get_prompt) to build the sub-agent prompt, \
(2) constraint(list) to include plan constraints, \
(3) note(get_context) for each affected file to inject relevant knowledge. \
Determine the wave type: single-task waves get foreground agents (can compile), \
multi-task waves get background agents (code-only, no build).",
                available_tools: Some(vec!["task", "step", "constraint", "note", "plan", "code"]),
                forbidden_actions: Some(vec![
                    "Do NOT launch agents without loading task context first",
                    "Do NOT allow multi-task wave agents to run builds — file conflicts risk",
                ]),
            },
            StateFragment {
                state_name: "dispatch_parallel",
                prompt_fragment: "\
DISPATCH sub-agents. Launch one Agent per task using: \
Agent(subagent_type: general-purpose, run_in_background: true) for multi-task waves, \
Agent(subagent_type: general-purpose) in foreground for single-task waves. \
Each agent receives: task prompt, affected_files isolation rule, plan constraints, \
and instructions to update steps/notes via MCP autonomously. \
Send ALL agent launches in a SINGLE message for true parallelism.",
                available_tools: None, // needs Agent tool
                forbidden_actions: Some(vec![
                    "Do NOT launch agents sequentially in separate messages — they serialize",
                    "Do NOT forget isolation rules: agents must only modify their affected_files",
                    "Do NOT give build permissions to agents in multi-task waves",
                ]),
            },
            StateFragment {
                state_name: "await_wave",
                prompt_fragment: "\
AWAIT wave completion. Use TaskOutput(block: true) on ALL agent IDs from this wave. \
Monitor progress via step(get_progress) for each task. If an agent fails, \
check its output for error details. Do NOT proceed to validation until \
ALL agents in the wave have completed (success or failure).",
                available_tools: Some(vec!["step", "task", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT proceed to validate_wave before all agents complete",
                    "Do NOT launch next wave agents while current wave is running",
                ]),
            },
            StateFragment {
                state_name: "validate_wave",
                prompt_fragment: "\
VALIDATE wave results. (1) Run a single build check (cargo check / npm build) \
to verify all agents' changes compile together. (2) Read notes created by \
sub-agents — they may contain cross-wave knowledge needed for later waves. \
(3) Update task statuses: completed for success, failed for errors. \
(4) If build fails, identify which agent's changes broke it and fix before continuing.",
                available_tools: Some(vec!["task", "step", "note", "code", "commit"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the build validation step",
                    "Do NOT mark tasks completed if the build is broken",
                ]),
            },
            StateFragment {
                state_name: "next_wave_or_done",
                prompt_fragment: "\
CHECK remaining waves. If more waves exist, commit the current wave's changes, \
then loop back to prepare_wave for the next wave. If all waves are done, \
run a final build + test suite, update plan status to completed, \
and create a summary commit linking to the plan.",
                available_tools: Some(vec!["plan", "task", "commit", "note"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the final build+test when all waves complete",
                ]),
            },
            StateFragment {
                state_name: "plan_complete",
                prompt_fragment: "\
Plan execution is COMPLETE. All waves executed, all tasks done, build passes. \
Create a summary note documenting: total waves, tasks completed, time taken, \
any issues encountered. Link the final commit to the plan.",
                available_tools: Some(vec!["plan", "commit", "note", "milestone", "release"]),
                forbidden_actions: None,
            },
        ],
    }
}

// ============================================================================
// 4. diagnostic-triage
// ============================================================================

fn seed_diagnostic_triage() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "diagnostic-triage",
        states: vec![
            StateFragment {
                state_name: "identify_symptom",
                prompt_fragment: "\
IDENTIFY the symptom. Parse the user's bug report to extract: \
(1) affected area — which file, module, or feature, \
(2) expected vs actual behavior, \
(3) reproduction context — when does it happen, what triggers it. \
Search for entity references (function names, file paths, error messages) \
in the user's message. Do NOT jump to hypotheses yet — focus on observation.",
                available_tools: Some(vec![
                    "code", "note",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT start fixing code before completing the triage",
                    "Do NOT assume the root cause without evidence",
                    "Do NOT skip symptom extraction — jumping to code is the #1 debugging mistake",
                ]),
            },
            StateFragment {
                state_name: "load_known_issues",
                prompt_fragment: "\
SEARCH existing knowledge for this symptom. Execute ALL of: \
(1) note(search_semantic, query=symptom_description) — check for existing gotchas, \
(2) decision(search_semantic) — check for architectural decisions that may explain the behavior, \
(3) code(get_file_co_changers) for suspected files — temporal coupling may reveal the cause. \
If a matching gotcha exists, the bug may already be documented. Check note status — \
if it was marked resolved, this could be a regression.",
                available_tools: Some(vec![
                    "note", "decision", "code", "commit",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the knowledge search — existing notes prevent redundant investigation",
                    "Do NOT ignore gotcha notes with high scar_intensity",
                ]),
            },
            StateFragment {
                state_name: "map_blast_radius",
                prompt_fragment: "\
MAP the blast radius. Use: \
(1) code(analyze_impact) on suspected files to see what they affect, \
(2) code(get_call_graph) on involved functions to understand the call chain, \
(3) code(get_file_dependencies) to see imports and dependents. \
Build a mental model of which functions/files COULD cause the symptom. \
Narrow the investigation to the smallest set of suspects.",
                available_tools: Some(vec![
                    "code", "note",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT read entire files blindly — use the graph to narrow scope first",
                    "Do NOT expand the blast radius without evidence",
                ]),
            },
            StateFragment {
                state_name: "check_recent_changes",
                prompt_fragment: "\
CHECK recent changes. Call commit(get_file_history) for each file in the blast radius. \
Cross-reference commit timestamps with the symptom timeline. \
A recent commit modifying a related file is the prime suspect. \
If the symptom is new, focus on commits in the last 48h. \
Look for patterns: does the same file appear in multiple recent commits? (churn = risk)",
                available_tools: Some(vec![
                    "commit", "code", "note",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT blame a commit without verifying the change actually relates to the symptom",
                ]),
            },
            StateFragment {
                state_name: "investigate",
                prompt_fragment: "\
INVESTIGATE with a focused hypothesis. Based on the previous steps, you should now have \
a specific theory about the root cause. Read the suspected code, run relevant tests, \
check error logs. Use the narrowed hypothesis — do NOT grep the entire codebase. \
If your hypothesis is wrong, return to map_blast_radius with new information.",
                available_tools: None, // all tools needed
                forbidden_actions: Some(vec![
                    "Do NOT investigate without a hypothesis — that's blind debugging",
                    "Do NOT apply a fix without understanding the root cause",
                ]),
            },
            StateFragment {
                state_name: "capture_resolution",
                prompt_fragment: "\
CAPTURE the resolution. Create a note(type: gotcha, importance: high) documenting: \
(1) the symptom, (2) the root cause, (3) the fix applied, (4) how to prevent recurrence. \
Link the note to affected files via note(link_to_entity). \
If this was a regression, tag as 'regression' + 'needs-test'. \
If it revealed an architectural issue, create a decision(add) for the record.",
                available_tools: Some(vec![
                    "note", "decision", "commit", "task", "step",
                ]),
                forbidden_actions: Some(vec![
                    "Do NOT close the bug without documenting the root cause",
                    "Do NOT skip linking the note to affected files",
                ]),
            },
        ],
    }
}

// ============================================================================
// 5. auto-maintenance
// ============================================================================

fn seed_auto_maintenance() -> ProtocolSeed {
    ProtocolSeed {
        protocol_name: "auto-maintenance",
        states: vec![
            StateFragment {
                state_name: "health_check",
                prompt_fragment: "\
Run COMPREHENSIVE health assessment. Call admin(persist_health_report, project_id) \
to capture current state: hotspots, knowledge gaps, risk assessment, neural metrics, \
and audit gaps. This persists a timestamped health note for delta comparison. \
The report is the foundation for all subsequent triage and remediation.",
                available_tools: Some(vec!["admin", "code", "project"]),
                forbidden_actions: Some(vec![
                    "Do NOT skip the health report — all downstream decisions depend on it",
                    "Do NOT start fixing issues before the report is complete",
                ]),
            },
            StateFragment {
                state_name: "analyze_delta",
                prompt_fragment: "\
COMPARE current health with the previous report. Search for the last health-check note \
via note(search, 'health-check auto-generated'). Identify degradation trends: \
new hotspots appearing, growing knowledge gaps, dying neurons (energy approaching 0), \
stale notes accumulating. Highlight any metric that changed by more than 20% since last check. \
If no previous report exists, skip delta and proceed to triage with absolute values.",
                available_tools: Some(vec!["note", "code", "admin"]),
                forbidden_actions: Some(vec![
                    "Do NOT ignore degradation trends — they compound over time",
                ]),
            },
            StateFragment {
                state_name: "triage",
                prompt_fragment: "\
CLASSIFY issues by severity and actionability. Three categories: \
(1) AUTO-FIXABLE — stale scores, missing synapses, orphan cleanup: handle in auto_fix, \
(2) AGENT-FIXABLE — knowledge gaps, undocumented hotspots, orphan decisions: create tasks, \
(3) HUMAN-REQUIRED — architecture changes, major refactoring: flag for human review. \
Build a prioritized remediation list. Critical items first.",
                available_tools: Some(vec!["note", "code", "admin"]),
                forbidden_actions: Some(vec![
                    "Do NOT attempt human-required fixes autonomously",
                    "Do NOT skip prioritization — fix high-impact issues first",
                ]),
            },
            StateFragment {
                state_name: "auto_fix",
                prompt_fragment: "\
Execute AUTOMATIC remediations that require no human judgment: \
(1) admin(update_staleness_scores) — refresh note freshness, \
(2) admin(decay_synapses) — prune weak neural connections, \
(3) admin(backfill_synapses) — create missing synapse links, \
(4) admin(backfill_touches) — rebuild TOUCHES relations from git, \
(5) admin(update_fabric_scores) — recalculate multi-layer GDS scores, \
(6) admin(maintain_skills, 'daily') — skill lifecycle maintenance. \
These are safe, idempotent operations.",
                available_tools: Some(vec!["admin"]),
                forbidden_actions: Some(vec![
                    "Do NOT run deep_maintenance here — it's too heavy for daily runs",
                    "Do NOT modify code or notes during auto_fix — only graph maintenance",
                ]),
            },
            StateFragment {
                state_name: "plan_remediation",
                prompt_fragment: "\
Create a REMEDIATION PLAN for agent-fixable issues. Use plan(create) with: \
(1) A task for each high-risk file needing documentation, \
(2) A task for each knowledge gap (file with 0 linked notes + high PageRank), \
(3) A task for each orphan decision needing AFFECTS links. \
Set priority based on the risk_score from the health report. \
Only create the plan if there are actionable issues — do not create empty plans.",
                available_tools: Some(vec!["plan", "task", "step", "note", "decision", "code"]),
                forbidden_actions: Some(vec![
                    "Do NOT create empty remediation plans",
                    "Do NOT create tasks for human-required issues",
                    "Do NOT execute the remediation plan now — just create it for later",
                ]),
            },
            StateFragment {
                state_name: "maintained",
                prompt_fragment: "\
Maintenance cycle COMPLETE. Health check done, auto-fixes applied, \
remediation plan created if needed. The knowledge graph is healthier than before. \
No further action required until the next maintenance trigger.",
                available_tools: None,
                forbidden_actions: None,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_fragments_non_empty() {
        let protocols = build_protocol_seeds();
        assert_eq!(protocols.len(), 5, "Expected 5 protocol seeds");

        for proto in &protocols {
            assert!(
                !proto.states.is_empty(),
                "Protocol {} has no states",
                proto.protocol_name
            );
            for state in &proto.states {
                assert!(
                    !state.prompt_fragment.is_empty(),
                    "Empty prompt_fragment for {}/{}",
                    proto.protocol_name,
                    state.state_name
                );
                // Fragments should be between 50-300 words
                let word_count = state.prompt_fragment.split_whitespace().count();
                assert!(
                    word_count >= 20,
                    "Fragment too short ({} words) for {}/{}",
                    word_count,
                    proto.protocol_name,
                    state.state_name
                );
                assert!(
                    word_count <= 300,
                    "Fragment too long ({} words) for {}/{}",
                    word_count,
                    proto.protocol_name,
                    state.state_name
                );
            }
        }
    }

    #[test]
    fn test_state_count_matches_protocols() {
        let protocols = build_protocol_seeds();
        let counts: Vec<(&str, usize)> = protocols
            .iter()
            .map(|p| (p.protocol_name, p.states.len()))
            .collect();

        // session-lifecycle: 5 states
        assert_eq!(counts[0], ("session-lifecycle", 5));
        // rfc-lifecycle: 9 states
        assert_eq!(counts[1], ("rfc-lifecycle", 9));
        // wave-dispatch: 7 states
        assert_eq!(counts[2], ("wave-dispatch", 7));
        // diagnostic-triage: 6 states
        assert_eq!(counts[3], ("diagnostic-triage", 6));
        // auto-maintenance: 6 states
        assert_eq!(counts[4], ("auto-maintenance", 6));
    }

    #[test]
    fn test_forbidden_actions_format() {
        let protocols = build_protocol_seeds();
        for proto in &protocols {
            for state in &proto.states {
                if let Some(actions) = &state.forbidden_actions {
                    for action in actions {
                        assert!(
                            action.starts_with("Do NOT"),
                            "Forbidden action should start with 'Do NOT': got '{}' in {}/{}",
                            action,
                            proto.protocol_name,
                            state.state_name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_available_tools_are_valid() {
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

        let protocols = build_protocol_seeds();
        for proto in &protocols {
            for state in &proto.states {
                if let Some(tools) = &state.available_tools {
                    for tool in tools {
                        assert!(
                            valid_tools.contains(tool),
                            "Invalid tool '{}' in {}/{}",
                            tool,
                            proto.protocol_name,
                            state.state_name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_seed_result_serialization() {
        let result = SeedResult {
            updated: 10,
            skipped: 2,
            protocols_found: 4,
            protocols_missing: vec!["auto-maintenance".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"updated\":10"));
        assert!(json.contains("\"skipped\":2"));
        assert!(json.contains("\"protocols_found\":4"));
        assert!(json.contains("auto-maintenance"));
    }

    #[test]
    fn test_seed_result_debug() {
        let result = SeedResult {
            updated: 0,
            skipped: 0,
            protocols_found: 0,
            protocols_missing: vec![],
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("SeedResult"));
        assert!(debug.contains("updated: 0"));
    }

    #[test]
    fn test_unique_state_names_per_protocol() {
        let protocols = build_protocol_seeds();
        for proto in &protocols {
            let mut seen = std::collections::HashSet::new();
            for state in &proto.states {
                assert!(
                    seen.insert(state.state_name),
                    "Duplicate state name '{}' in protocol '{}'",
                    state.state_name,
                    proto.protocol_name
                );
            }
        }
    }

    #[test]
    fn test_unique_protocol_names() {
        let protocols = build_protocol_seeds();
        let mut seen = std::collections::HashSet::new();
        for proto in &protocols {
            assert!(
                seen.insert(proto.protocol_name),
                "Duplicate protocol name '{}'",
                proto.protocol_name
            );
        }
    }

    #[test]
    fn test_all_fragments_end_with_period_or_instruction() {
        let protocols = build_protocol_seeds();
        for proto in &protocols {
            for state in &proto.states {
                let trimmed = state.prompt_fragment.trim();
                assert!(
                    trimmed.ends_with('.') || trimmed.ends_with(':') || trimmed.ends_with(')'),
                    "Fragment for {}/{} should end with punctuation, got: ...{}",
                    proto.protocol_name,
                    state.state_name,
                    &trimmed[trimmed.len().saturating_sub(20)..],
                );
            }
        }
    }

    #[test]
    fn test_session_lifecycle_has_warm_up_and_closing() {
        let protocols = build_protocol_seeds();
        let session = &protocols[0];
        assert_eq!(session.protocol_name, "session-lifecycle");
        let state_names: Vec<&str> = session.states.iter().map(|s| s.state_name).collect();
        assert!(state_names.contains(&"warm_up"));
        assert!(state_names.contains(&"closing"));
        assert!(state_names.contains(&"closed"));
    }

    #[test]
    fn test_wave_dispatch_has_compute_and_dispatch() {
        let protocols = build_protocol_seeds();
        let wave = protocols
            .iter()
            .find(|p| p.protocol_name == "wave-dispatch")
            .unwrap();
        let state_names: Vec<&str> = wave.states.iter().map(|s| s.state_name).collect();
        assert!(state_names.contains(&"compute_waves"));
        assert!(state_names.contains(&"dispatch_parallel"));
    }

    #[test]
    fn test_rfc_lifecycle_has_full_flow() {
        let protocols = build_protocol_seeds();
        let rfc = protocols
            .iter()
            .find(|p| p.protocol_name == "rfc-lifecycle")
            .unwrap();
        let state_names: Vec<&str> = rfc.states.iter().map(|s| s.state_name).collect();
        assert!(state_names.contains(&"draft"));
        assert!(state_names.contains(&"accepted"));
        assert!(state_names.contains(&"rejected"));
        assert!(state_names.contains(&"implemented"));
    }
}
