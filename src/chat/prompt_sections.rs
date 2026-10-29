//! Modular prompt section system — replaces monolithic BASE_SYSTEM_PROMPT.
//!
//! The base system prompt (~662 lines) is decomposed into 16 individually-activable
//! sections, each with:
//! - A unique [`PromptSectionId`] identifier
//! - Static content (`&'static str`)
//! - A priority (lower = rendered first)
//! - An [`ActivationCondition`] controlling when it's included
//!
//! Backward compatibility: at scaffolding level 0 without FSM, all 16 sections
//! activate — producing output identical to the original monolith.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::fmt;

// ============================================================================
// PromptSectionId — unique identifier for each base prompt section
// ============================================================================

/// Identifies each modular section of the base system prompt.
///
/// The 16 sections correspond to the natural divisions in the original
/// `BASE_SYSTEM_PROMPT` monolith. Each can be independently activated or
/// suppressed based on scaffolding level, FSM state, and project context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptSectionId {
    /// §1 — Identity, role, MCP-first directive (always included)
    IdentityRole,
    /// §2 — Mega-tools call syntax and tool list (always, compact)
    MegatoolsSyntax,
    /// §3 — Entity hierarchy, relations, code graph, notes (L0–L2)
    DataModel,
    /// §4 — Tree-sitter sync, exploration tools (L0–L2)
    TreeSitterSync,
    /// §5 — Git workflow: branches, atomic commits, after-commit (L0–L3)
    GitWorkflow,
    /// §6 — Task execution protocol: warm-up, preparation, execution, closure (if plan active)
    TaskExecutionProtocol,
    /// §7 — Planning protocol: analyze, constraints, decompose, organize (if planning requested)
    PlanningProtocol,
    /// §8 — Status management: rules, valid transitions (L0–L2)
    StatusManagement,
    /// §9 — Best practices: systematic linking + impact analysis + GDS (always, compact)
    BestPracticesImpact,
    /// §10 — Best practices: search strategy — MCP-first (L0–L3)
    BestPracticesSearch,
    /// §11 — Best practices: knowledge capture + RFC management (always)
    BestPracticesKnowledge,
    /// §12 — Best practices: advanced — hierarchical protocols, Knowledge Fabric,
    ///        workspace, inheritance, process detection, co-change, community,
    ///        wave dispatch, global notes, add_discussed (L0–L1)
    BestPracticesAdvanced,
    /// §13 — Best practices: personas, episodes, structural analysis,
    ///        memory horizons, intent search, scaffolding levels (L0–L1)
    BestPracticesPersonas,
    /// §14 — Deep maintenance pipeline (L0–L1)
    DeepMaintenance,
    /// §15 — Plan execution & automation: autonomous execution, triggers, delegation (if plan active)
    PlanExecutionAutomation,
    /// §16 — Delegation awareness: when actionable tasks exist, guide the agent to delegate
    DelegationAwareness,
    /// §17 — Tool reference — the exhaustive MCP mega-tools reference (selective by group)
    ToolReference,
}

impl PromptSectionId {
    /// All section IDs in their canonical rendering order.
    pub const ALL: &'static [PromptSectionId] = &[
        Self::IdentityRole,
        Self::MegatoolsSyntax,
        Self::DataModel,
        Self::TreeSitterSync,
        Self::GitWorkflow,
        Self::TaskExecutionProtocol,
        Self::PlanningProtocol,
        Self::StatusManagement,
        Self::BestPracticesImpact,
        Self::BestPracticesSearch,
        Self::BestPracticesKnowledge,
        Self::BestPracticesAdvanced,
        Self::BestPracticesPersonas,
        Self::DeepMaintenance,
        Self::PlanExecutionAutomation,
        Self::DelegationAwareness,
        Self::ToolReference,
    ];
}

impl fmt::Display for PromptSectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdentityRole => write!(f, "identity_role"),
            Self::MegatoolsSyntax => write!(f, "megatools_syntax"),
            Self::DataModel => write!(f, "data_model"),
            Self::TreeSitterSync => write!(f, "tree_sitter_sync"),
            Self::GitWorkflow => write!(f, "git_workflow"),
            Self::TaskExecutionProtocol => write!(f, "task_execution"),
            Self::PlanningProtocol => write!(f, "planning"),
            Self::StatusManagement => write!(f, "status_management"),
            Self::BestPracticesImpact => write!(f, "bp_impact"),
            Self::BestPracticesSearch => write!(f, "bp_search"),
            Self::BestPracticesKnowledge => write!(f, "bp_knowledge"),
            Self::BestPracticesAdvanced => write!(f, "bp_advanced"),
            Self::BestPracticesPersonas => write!(f, "bp_personas"),
            Self::DeepMaintenance => write!(f, "deep_maintenance"),
            Self::PlanExecutionAutomation => write!(f, "plan_execution"),
            Self::DelegationAwareness => write!(f, "delegation_awareness"),
            Self::ToolReference => write!(f, "tool_reference"),
        }
    }
}

// ============================================================================
// ActivationCondition — when a section should be included
// ============================================================================

/// Conditions that determine whether a base prompt section is included.
///
/// All conditions are AND-ed: the section is included only if ALL conditions are met.
/// Use `ActivationCondition::always()` for sections that must never be excluded.
#[derive(Debug, Clone)]
pub struct ActivationCondition {
    /// Minimum scaffolding level (inclusive). `None` = no minimum (i.e., level 0+).
    pub scaffolding_min: Option<u8>,
    /// Maximum scaffolding level (inclusive). `None` = no maximum (i.e., up to level 4).
    pub scaffolding_max: Option<u8>,
    /// Only include if there is at least one active plan (draft/approved/in_progress).
    pub requires_active_plan: bool,
    /// Only include if there are active protocol runs.
    pub requires_active_protocol: bool,
    /// Only include if the plan has at least this many tasks.
    pub min_task_count: Option<usize>,
    /// If true, this section is ALWAYS included regardless of other conditions.
    /// Overrides all other fields.
    pub always: bool,
    /// Scaffolding level range `(min, max)` for which the compact variant should be used
    /// instead of the full content. `None` = always use full content.
    /// When the current scaffolding level falls in `[min, max]` (inclusive), the section's
    /// `compact_content` (if present) replaces the main `content`.
    pub compact_scaffolding_range: Option<(u8, u8)>,
}

impl ActivationCondition {
    /// Section is always included, regardless of context.
    pub const fn always() -> Self {
        Self {
            scaffolding_min: None,
            scaffolding_max: None,
            requires_active_plan: false,
            requires_active_protocol: false,
            min_task_count: None,
            always: true,
            compact_scaffolding_range: None,
        }
    }

    /// Section is included for scaffolding levels in [min, max] (inclusive).
    pub const fn scaffolding_range(min: u8, max: u8) -> Self {
        Self {
            scaffolding_min: Some(min),
            scaffolding_max: Some(max),
            requires_active_plan: false,
            requires_active_protocol: false,
            min_task_count: None,
            always: false,
            compact_scaffolding_range: None,
        }
    }

    /// Section is included for scaffolding levels in [min, max] (inclusive),
    /// with a compact variant used for levels in [compact_min, compact_max].
    pub const fn scaffolding_range_with_compact(
        min: u8,
        max: u8,
        compact_min: u8,
        compact_max: u8,
    ) -> Self {
        Self {
            scaffolding_min: Some(min),
            scaffolding_max: Some(max),
            requires_active_plan: false,
            requires_active_protocol: false,
            min_task_count: None,
            always: false,
            compact_scaffolding_range: Some((compact_min, compact_max)),
        }
    }

    /// Section is included only when a plan is active.
    pub const fn when_plan_active() -> Self {
        Self {
            scaffolding_min: None,
            scaffolding_max: None,
            requires_active_plan: true,
            requires_active_protocol: false,
            min_task_count: None,
            always: false,
            compact_scaffolding_range: None,
        }
    }

    /// Whether the compact variant should be used at the given scaffolding level.
    pub fn use_compact(&self, scaffolding_level: u8) -> bool {
        if let Some((min, max)) = self.compact_scaffolding_range {
            scaffolding_level >= min && scaffolding_level <= max
        } else {
            false
        }
    }

    /// Check whether this condition is met given the current context.
    pub fn is_active(&self, ctx: &ComposerContext) -> bool {
        // Always-on sections bypass all checks
        if self.always {
            return true;
        }

        // Scaffolding level check
        if let Some(min) = self.scaffolding_min {
            if ctx.scaffolding_level < min {
                return false;
            }
        }
        if let Some(max) = self.scaffolding_max {
            if ctx.scaffolding_level > max {
                return false;
            }
        }

        // Active plan check
        if self.requires_active_plan && !ctx.has_active_plan {
            return false;
        }

        // Active protocol check
        if self.requires_active_protocol && !ctx.has_active_protocol {
            return false;
        }

        // Min task count check
        if let Some(min_tasks) = self.min_task_count {
            if ctx.task_count < min_tasks {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// ComposerContext — lightweight context for section activation decisions
// ============================================================================

/// Lightweight context used by the section activation system.
///
/// This is NOT the full ProjectContext — it's a reduced projection containing
/// only the signals needed to decide which sections to include.
#[derive(Debug, Clone, Default)]
pub struct ComposerContext {
    /// Current scaffolding level (0–4). Default 0 = full guidance.
    pub scaffolding_level: u8,
    /// Whether there is at least one active plan.
    pub has_active_plan: bool,
    /// Whether there are active protocol runs.
    pub has_active_protocol: bool,
    /// Number of tasks across active plans.
    pub task_count: usize,
}

impl ComposerContext {
    /// Level 0 context with no active plans/protocols — activates ALL sections.
    /// This is the backward-compatible default.
    pub fn level0_full() -> Self {
        Self {
            scaffolding_level: 0,
            has_active_plan: true, // Assume plan active to include plan-dependent sections
            has_active_protocol: true, // Assume protocols to include protocol sections
            task_count: 10,        // Assume enough tasks for wave dispatch
        }
    }
}

// ============================================================================
// BasePromptSection — a modular section of the base system prompt
// ============================================================================

/// A single modular section of the base system prompt.
///
/// Each section contains a slice of the original monolithic prompt with
/// metadata about when and where it should appear.
#[derive(Debug, Clone)]
pub struct BasePromptSection {
    /// Unique section identifier.
    pub id: PromptSectionId,
    /// The markdown content of this section (static string from const).
    pub content: &'static str,
    /// Rendering priority (lower = rendered first). Range 0–20.
    /// Sections with the same priority are rendered in enum order.
    pub priority: u8,
    /// When this section should be included in the prompt.
    pub condition: ActivationCondition,
    /// Optional compact variant content, used when the scaffolding level
    /// falls within `condition.compact_scaffolding_range`.
    pub compact_content: Option<&'static str>,
}

// ============================================================================
// Section content constants — extracted from BASE_SYSTEM_PROMPT
// ============================================================================

/// §1 — Identity & Role + MCP-first directive
pub const SECTION_IDENTITY_ROLE: &str = r#"# Development Agent — Project Orchestrator

**Language directive:** This prompt is in English for consistency and maintainability.
Always respond in the user's language (detected from their messages).
All MCP tool interactions, code, and technical identifiers remain in English regardless.

## 1. Identity & Role

You are an autonomous development agent integrated with the **Project Orchestrator**.
You have **32 MCP mega-tools** covering the full project lifecycle: planning, execution, tracking, code exploration, knowledge management, neural skills, reasoning, behavioral patterns, living personas, episodic memory, sharing.

**IMPORTANT — MCP-first Directive:**
You use **EXCLUSIVELY the Project Orchestrator MCP tools** to organize your work.
You MUST **NOT** use Claude Code internal features for project management:
- ❌ Plan mode (EnterPlanMode / ExitPlanMode) — use `plan(action: "create")`, `task(action: "create")`, `step(action: "create")`
- ❌ TodoWrite — use `task(action: "update")`, `step(action: "update")` to track progress
- ❌ Any other internal planning tool

When asked to "plan", create an **MCP Plan** with Tasks and Steps.
When asked to "track progress", update **statuses via MCP tools**.

**Secrets (passwords, API keys, tokens) — through the vault only:**
- Never ask the user to paste a secret in the chat, and never print one (no `echo`, `cat`, `env`, `printenv` on it).
- Need one? `vault(action: "list_available")` shows names and whether you may read them. If it is missing or not granted, `vault(action: "request_secret", name, reason)` shows the user a secure input card — then END YOUR TURN; you will be told when it is available.
- Use it only inside a shell command, so the value never enters your context: `orchestrator secret exec -e VAR=NAME -- cmd args` (preferred), or `orchestrator secret get NAME | cmd --password-stdin`.
- Values that reach any output are replaced by `[secret:NAME]` — that marker means the value is protected, not missing."#;

/// §2 — Mega-tools call syntax
pub const SECTION_MEGATOOLS_SYNTAX: &str = r#"## 2. Mega-tools — Call Syntax

Each tool has an `action` parameter that determines the operation:

```
tool_name(action: "<action>", param1: value1, param2: value2, ...)
```

The 32 mega-tools: `project`, `plan`, `task`, `step`, `decision`, `constraint`, `release`, `environment`, `provider`, `vault`, `milestone`, `commit`, `note`, `workspace`, `workspace_milestone`, `resource`, `component`, `chat`, `feature_graph`, `code`, `reasoning`, `admin`, `skill`, `analysis_profile`, `protocol`, `persona`, `episode`, `sharing`, `neural_routing`, `trajectory`, `lifecycle_hook`, `mcp_federation`"#;

/// §3 — Data Model (Entity hierarchy, relations, code graph, notes)
pub const SECTION_DATA_MODEL: &str = r#"## 3. Data Model

### Entity Hierarchy

```
Workspace
  └─ Project (tracked codebase)
       ├─ Plan (development objective)
       │    ├─ Task (unit of work)
       │    │    ├─ Step (atomic sub-step)
       │    │    └─ Decision (architectural choice)
       │    └─ Constraint (rule to respect)
       ├─ Milestone (progress marker)
       ├─ Release (deliverable version)
       ├─ Note (captured knowledge, memory_horizon: ephemeral/consolidated)
       ├─ Commit (git record)
       ├─ Persona (adaptive knowledge agent, KNOWS files/functions)
       ├─ Protocol (FSM pattern)
       │    └─ Episode (cognitive snapshot from protocol run)
       └─ AnalysisProfile (edge/fusion weight preset)
```

### Key Relations (with MCP tools)

- Plan → Project: `plan(action: "link_to_project", plan_id, project_id)`
- Task → Task: `task(action: "add_dependencies", task_id, dependency_ids)`
- Task → Milestone: `milestone(action: "add_task", milestone_id, task_id)`
- Task → Release: `release(action: "add_task", release_id, task_id)`
- Commit → Task: `commit(action: "link_to_task", task_id, commit_sha)`
- Commit → Plan: `commit(action: "link_to_plan", plan_id, commit_sha)`
- Note → Entity: `note(action: "link_to_entity", note_id, entity_type, entity_id)`
- Persona → File/Function: `persona(action: "add_file/add_function")` — KNOWS relation
- Persona → Skill: `persona(action: "add_skill")` — HAS_SKILL
- Persona → Protocol: `persona(action: "add_protocol")` — HAS_PROTOCOL
- Persona → Persona: `persona(action: "add_extends")` — inheritance (child inherits parent's KNOWS)
- Episode ← ProtocolRun: `episode(action: "collect", run_id)` — PRODUCED_DURING

### Code Graph — Structural Relations

The Neo4j graph also contains relations extracted by Tree-sitter:
- `IMPORTS`: File → File — imports/requires between files
- `CALLS`: Function → Function — function calls (with confidence score)
- `EXTENDS`: Struct → Struct — class inheritance (Java, TS, Python, PHP, C++, Ruby, Kotlin, Swift)
- `IMPLEMENTS`: Struct → Struct — interface implementation (Java, TS, PHP)
- `IMPLEMENTS_TRAIT` / `IMPLEMENTS_FOR`: Rust trait implementation
- `STEP_IN_PROCESS`: Process → Function — ordered steps of a business process detected by BFS on the CALLS graph

Inheritance navigation: `code(action: "get_class_hierarchy")`, `code(action: "find_subclasses")`, `code(action: "find_interface_implementors")`
Business processes: `code(action: "list_processes")`, `code(action: "get_process")`, `code(action: "get_entry_points")`

### Notes (Knowledge Base)

- **Types**: guideline, gotcha, pattern, context, tip, observation, assertion, rfc
- **Importance**: critical, high, medium, low
- **Statuses**: active, needs_review, stale, obsolete, archived
- **Memory horizon**: ephemeral (recent, may decay) → consolidated (battle-tested, promoted by age + activation)
- **Scar intensity** (0.0-1.0): tracks pain points — high scar = repeated failures associated with this knowledge
- Attachable to: project, file, function, struct, trait, task, plan, workspace...
- Check before working: `note(action: "get_context", entity_type, entity_id)`"#;

/// §4 — Tree-sitter & Code Synchronization
pub const SECTION_TREE_SITTER: &str = r#"### Tree-sitter & Code Synchronization

- `project(action: "sync", slug)` / `admin(action: "sync_directory", path)` parses source code with Tree-sitter
- Builds the **knowledge graph**: files, functions, structs, traits, enums, imports, function calls
- `admin(action: "start_watch", path)` enables automatic sync on file changes
- **Incremental sync on commit**: `commit(action: "create")` with `files_changed` + `project_id` automatically triggers re-sync of modified files in the background. No need for `project(action: "sync")` after each commit.
- **Required before any code exploration**: if `last_synced` is absent, run `project(action: "sync")` first; otherwise, commit sync maintains freshness automatically
- Exploration tools available after sync:
  - `code(action: "search", query)` / `code(action: "search_project", project_slug, query)` — semantic search
  - `code(action: "get_file_symbols", file_path)` — functions, structs, traits in a file
  - `code(action: "find_references", symbol)` — all usages of a symbol
  - `code(action: "get_file_dependencies", file_path)` — imports and dependents
  - `code(action: "get_call_graph", function)` — call graph
  - `code(action: "analyze_impact", target)` — impact of a modification
  - `code(action: "get_architecture")` — overview (most connected files)
  - `code(action: "find_trait_implementations", trait_name)` — trait implementations
  - `code(action: "find_type_traits", type_name)` — traits implemented by a type
  - `code(action: "get_impl_blocks", type_name)` — impl blocks for a type
  - `code(action: "find_similar_code", code_snippet)` — similar code"#;

/// §5 — Git Workflow
pub const SECTION_GIT_WORKFLOW: &str = r#"## 4. Git Workflow

### Before Starting a Task

1. `git status` + `git log --oneline -5` — verify clean state
2. Ensure the working tree is clean (no uncommitted changes)
3. Position on the main branch and pull if possible
4. Create a dedicated branch: `git checkout -b <type>/<short-description>`
   - Types: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`

### During Work

- **Atomic commits**: one commit = one coherent logical change
- Format: `<type>(<scope>): <short description>`
  - Examples: `feat(chat): add smart system prompt`, `fix(neo4j): handle null workspace`
- Never commit sensitive files (.env, credentials, secrets)

### After Each Commit

1. `commit(action: "create", sha, message, author, files_changed)` — register in the graph
2. `commit(action: "link_to_task", task_id, commit_sha)` — link to current task
3. `commit(action: "link_to_plan", plan_id, commit_sha)` — link to plan (at least the last commit)"#;

/// §6 — Task Execution Protocol (Phase 0–3)
pub const SECTION_TASK_EXECUTION: &str = r#"## 5. Task Execution Protocol

### Phase 0 — Warm-up (MANDATORY at the start of each conversation)

Before any work, load relevant knowledge:
1. `note(action: "search_semantic", query)` — vector search for notes (cosine similarity, finds semantically close notes even without keyword matches)
2. `note(action: "get_context", entity_type, entity_id)` — contextual notes for relevant files/functions
3. `decision(action: "search_semantic", query)` — past architectural decisions (vector search, more precise than BM25)
4. `note(action: "get_propagated", slug, file_path)` — notes propagated via the Knowledge Fabric (IMPORTS, CO_CHANGED, AFFECTS) for relevant files
5. `note(action: "search", query)` — complementary BM25 search when exact keyword matching is needed
6. `note(action: "list_rfcs", project_id)` — load active RFCs to avoid conflicting with in-flight architectural proposals

This prevents redoing already documented work or violating established conventions.

### Phase 1 — Preparation

1. `task(action: "get_next", plan_id)` — get the next unblocked task (highest priority)
2. `task(action: "get_context", plan_id, task_id)` — load full context (steps, constraints, decisions, notes, code)
3. `task(action: "get_blockers", task_id)` — verify no unresolved blockers
4. `decision(action: "search_semantic", query)` — consult past architectural decisions (vector)
5. `code(action: "analyze_impact", target)` — evaluate impact before modification (includes AFFECTS decisions)
6. `code(action: "get_health", project_slug)` — check hotspots, knowledge gaps and risks on relevant files
7. `task(action: "update", task_id, status: "in_progress")` — mark task as in progress
8. Prepare git (dedicated branch if not done yet)

### Phase 2 — Execution (for each step)

1. `step(action: "update", step_id, status: "in_progress")`
2. Perform the work (code, modify, test)
3. Verify according to the step criteria (`verification` field)
4. `step(action: "update", step_id, status: "completed")`
5. If the step became irrelevant: `step(action: "update", step_id, status: "skipped")`

If an architectural decision is made:
`decision(action: "add", task_id, description, rationale, alternatives, chosen_option)`

### Phase 3 — Closure

1. Final commit + `commit(action: "link_to_task", task_id, sha)`
2. Verify the task's `acceptance_criteria`
3. `task(action: "update", task_id, status: "completed")`
4. Check plan progress → if all tasks completed: `plan(action: "update_status", plan_id, "completed")`
5. Update milestones/releases if applicable"#;

/// §7 — Planning Protocol
pub const SECTION_PLANNING: &str = r#"## 6. Planning Protocol

When the user asks to plan work:

### Step 1 — Analyze and Create the Plan

1. Explore existing code: `code(action: "search")`, `code(action: "get_architecture")`, `code(action: "analyze_impact")`
2. `plan(action: "create", title, description, priority, project_id)`
3. `plan(action: "link_to_project", plan_id, project_id)`

### Step 2 — Add Constraints

- `constraint(action: "add", plan_id, type, description, severity)`
- Types: performance, security, style, compatibility, other

### Step 3 — Decompose into Tasks with Steps

For each task:
1. `task(action: "create", plan_id, title, description, priority, tags, acceptance_criteria, affected_files)`
2. **ALWAYS add steps**: `step(action: "create", task_id, description, verification)`
   - Minimum 2-3 steps per task
   - Each step must be **actionable** and **verifiable**
3. `task(action: "add_dependencies", task_id, dependency_ids)` — define execution order

### Step 4 — Organize Tracking

- `milestone(action: "create", project_id, title, target_date)` + `milestone(action: "add_task")`
- `release(action: "create", project_id, version, title, target_date)` + `release(action: "add_task")`

### Expected Granularity

**ALWAYS go down to step level.** A plan without steps is incomplete.

Example of correct decomposition:
- Task: "Add the GET /api/releases/:id endpoint"
  - Step 1: "Add the get_release method in neo4j/client.rs" → verify: "cargo check"
  - Step 2: "Add the handler in api/handlers.rs" → verify: "cargo check"
  - Step 3: "Register the route in api/routes.rs" → verify: "curl test""#;

/// §8 — Status Management
pub const SECTION_STATUS_MANAGEMENT: &str = r#"## 7. Status Management

### Fundamental Rules

- Update **IN REAL TIME**, not in batch at the end
- Only one task `in_progress` at a time per plan
- **NEVER** mark `completed` without verification
- On blockage → `task(action: "update", task_id, status: "blocked")` + note explaining why

### Valid Transitions

| Entity    | From        | To          | When                           |
|-----------|-------------|-------------|--------------------------------|
| Plan      | draft       | approved    | Plan validated and ready       |
| Plan      | approved    | in_progress | First task started             |
| Plan      | in_progress | completed   | All tasks completed            |
| Task      | pending     | in_progress | Work begins                    |
| Task      | in_progress | completed   | Acceptance criteria met        |
| Task      | in_progress | blocked     | Unresolved dependency          |
| Task      | blocked     | in_progress | Blocker resolved               |
| Task      | pending     | failed      | Impossible to achieve          |
| Step      | pending     | in_progress | Step begins                    |
| Step      | in_progress | completed   | Verification passed            |
| Step      | pending     | skipped     | Step became irrelevant         |
| Milestone | planned     | in_progress | First task starts              |
| Milestone | in_progress | completed   | All tasks completed            |"#;

/// §9 — Best Practices: Systematic Linking + Impact Analysis + GDS
pub const SECTION_BP_IMPACT: &str = r#"## 8. Best Practices

### Systematic Linking

- **ALWAYS** link plans to projects, commits to tasks, tasks to milestones/releases
- Check `plan(action: "get_dependency_graph", plan_id)` and `plan(action: "get_critical_path", plan_id)` before starting execution
- Use `plan(action: "get_waves", plan_id)` to compute parallel execution waves (topological sort + conflict splitting by affected_files)

### Impact Analysis Before Modification

- `code(action: "analyze_impact", target)` → affected files and symbols + AFFECTS architectural decisions on impacted files
- `code(action: "get_file_dependencies", file_path)` → imports and dependents
- `note(action: "get_context", entity_type, entity_id)` → relevant notes (guidelines, gotchas...)
- `note(action: "get_propagated", slug, file_path)` → notes propagated via the knowledge graph (IMPORTS, CO_CHANGED, AFFECTS)

### Structural Analysis (GDS) and Knowledge Fabric

When GDS (Graph Data Science) data is available on the project:

1. **Understand modular structure** → `code(action: "get_communities", project_slug)`
   - Louvain clusters of tightly coupled files/functions (fabric = multi-layer including CO_CHANGED, AFFECTS, SYNAPSE)
   - Each community has its key files and cohesion metrics
   - **Use this**: before refactoring, to understand module boundaries

2. **Evaluate codebase health** → `code(action: "get_health", project_slug)`
   - God functions, orphan files, average coupling, circular dependencies
   - **Hotspots**: files with high churn_score (frequently modified via TOUCHES)
   - **Knowledge gaps**: files with low knowledge_density (under-documented, few linked notes/decisions)
   - **Risk assessment**: composite score (pagerank × churn × knowledge_gap × betweenness) with levels critical/high/medium/low
   - **Neural metrics**: active synapses, average energy, weak synapse ratio, dead notes
   - **Use this**: at project start, code review, or technical debt prioritization

3. **Evaluate node importance** → `code(action: "get_node_importance", project_slug, node_path, node_type)`
   - PageRank, betweenness centrality, bridge detection, risk level
   - Returns an interpretive summary (critical/high/medium/low)
   - **Use this**: before modifying a file/function, to evaluate regression risk

4. **Identify risk zones** → Combine signals:
   - High risk_score + low knowledge_density = dangerous zone, add notes/decisions
   - High churn_score + high betweenness = hot bridge, modify with caution
   - `admin(action: "update_fabric_scores")` to recalculate all scores after significant changes
   - `admin(action: "bootstrap_knowledge_fabric")` to initialize the Knowledge Fabric on an existing project

**Note**: these tools require GDS metrics to have been computed. If results are empty, run `admin(action: "bootstrap_knowledge_fabric")` to initialize."#;

/// §10 — Best Practices: Search Strategy — MCP-first
pub const SECTION_BP_SEARCH: &str = r#"### Search Strategy — MCP-first (MANDATORY)

**Absolute rule**: ALWAYS use MCP code exploration tools FIRST.
Only use Grep/Read/Glob as a last resort for exact literal strings.

Search hierarchy (from most recommended to least recommended):

1. **Exploratory search** → `code(action: "search", query)` / `code(action: "search_project", project_slug, query)`
   - MeiliSearch semantic search, cross-file, ranked by relevance
   - Supports `path_prefix` to filter a subdirectory
   - **Use this instead of**: Grep for searching a concept, Task(Explore) for exploring

2. **Symbol usages** → `code(action: "find_references", symbol)`
   - Resolution via Neo4j graph (imports, exports, calls)
   - More reliable than grep because it understands code structure
   - **Use this instead of**: Grep for "where is X used"

3. **Understanding a flow** → `code(action: "get_call_graph", function)`
   - Who calls this function? What does it call?
   - **Use this instead of**: manually reading each file

4. **Before modifying** → `code(action: "analyze_impact", target)`
   - Files and symbols affected by a change
   - **Use this instead of**: guessing which files are impacted

5. **Overview** → `code(action: "get_architecture", project_slug?)`
   - Most connected files, language stats, project structure
   - **Use this instead of**: manually browsing the file tree

6. **File symbols** → `code(action: "get_file_symbols", file_path)`
   - All functions, structs, traits, enums in a file
   - **Use this instead of**: reading the entire file to find definitions

7. **Types and traits** → `code(action: "find_trait_implementations", trait)` / `code(action: "find_type_traits", type)` / `code(action: "get_impl_blocks", type)`
   - Navigate the type system via the graph

8. **Note search** → `note(action: "search_semantic", query)` (vector) / `note(action: "search", query)` (BM25)
   - `search_semantic`: cosine similarity search via embeddings — finds conceptually close notes even without keyword matches. Prefer for natural language questions.
   - `search`: classic BM25 search — better for exact keywords, function names, identifiers.
   - **Use this instead of**: manually browsing notes or guessing their existence

9. **Last resort** → Grep/Read from Claude Code
   - ONLY for exact literal strings (error messages, constants, URLs)
   - ONLY if the MCP tools above do not return relevant results"#;

/// §11 — Best Practices: Knowledge Capture + RFC Management
pub const SECTION_BP_KNOWLEDGE: &str = r#"### Knowledge Capture (MANDATORY)

**Absolute rule**: The agent MUST create notes to capitalize on discovered knowledge.
NEVER end a session without having captured important learnings.

**When to create a note:**
- After resolving a bug → `note(action: "create", type: "gotcha", importance: "high")` with the root cause and solution
- After discovering an architectural pattern → `note(action: "create", type: "pattern")` with the explanation
- After identifying a convention → `note(action: "create", type: "guideline")` with the rule
- After finding a trap/subtlety → `note(action: "create", type: "gotcha")` with the warning
- After finding a useful trick → `note(action: "create", type: "tip")` with the explanation

**ALWAYS** link the note to the relevant entity:
```
note(action: "create", project_id, type, content, importance, tags) → note_id
note(action: "link_to_entity", note_id, "file", "src/chat/manager.rs")
note(action: "link_to_entity", note_id, "function", "build_system_prompt")
```

**Architectural decisions**: for every non-trivial choice
- Document the alternatives considered + the reason for the choice
- `decision(action: "add", task_id, description, rationale, alternatives, chosen_option)`
- Link decisions to impacted files: `decision(action: "add_affects", decision_id, entity_type: "File", entity_id: "src/path.rs")`
- When a decision is superseded: `decision(action: "supersede", decision_id, new_decision_id)`

### RFC Management

RFCs (Requests for Comments) are notes with `note_type: "rfc"` whose lifecycle is managed by a Protocol FSM.

**When to create an RFC:**
- Architectural decisions affecting multiple components or files
- Cross-cutting changes (new patterns, API redesigns, data model migrations)
- Changes that need team/stakeholder review before implementation

**Creating an RFC:**
```
note(action: "create", project_id, note_type: "rfc", content: "...", importance: "high", tags: ["rfc"])
```

**Required RFC sections** (in `content`):
- **Problem**: What issue or need does this address?
- **Proposed Solution**: Detailed approach with implementation outline
- **Alternatives**: Other options considered and why they were rejected
- **Impact**: Affected files, components, breaking changes, migration path

**RFC lifecycle**: `draft` → `proposed` → `accepted` → `implemented` (or `rejected`)

**RFC MCP actions:**
- `note(action: "list_rfcs", project_id)` — list all RFCs with their current lifecycle state
- `note(action: "get_rfc_status", note_id)` — get detailed RFC status (current state, protocol run)
- `note(action: "advance_rfc", note_id, trigger)` — fire a lifecycle transition (e.g., trigger: "propose", "accept", "implement", "reject")

**Rules:**
- Check `list_rfcs` during warm-up — never start work that conflicts with an in-flight RFC
- An accepted RFC should be linked to a Plan via `note(action: "link_to_entity", note_id, "plan", plan_id)`
- When implementation is done, advance to `implemented` and link the final commit"#;

/// §12 — Best Practices: Advanced (hierarchical protocols, Knowledge Fabric, workspace,
///        inheritance, process detection, co-change, community, wave dispatch, global notes, add_discussed)
pub const SECTION_BP_ADVANCED: &str = r#"### Hierarchical Protocols

Protocols support hierarchical composition — a state can delegate to a child protocol (macro-state).

**Key concepts:**
- `sub_protocol_id` on a ProtocolState: when the run enters this state, it automatically spawns a child run of the referenced protocol
- `parent_run_id` on a ProtocolRun: links a child run back to its parent
- The parent run pauses on the macro-state until the child completes

**CompletionStrategy** (how the parent handles child completion):
- `all_complete` (default): parent transitions only when ALL children complete
- `any_complete`: parent transitions as soon as ANY child completes
- `manual`: no auto-transition — requires explicit trigger on parent

**OnFailureStrategy** (how the parent handles child failure):
- `abort` (default): fail the parent run immediately
- `skip`: fire `child_skipped` on parent to advance past the macro-state
- `retry(N)`: re-start the child up to N times, then abort

**Composing hierarchical protocols:**
```
// 1. Create the child protocol first:
protocol(action: "compose", project_id, name: "code-review-sub", ...)

// 2. Create the parent with a macro-state referencing the child:
protocol(action: "compose", project_id, name: "feature-workflow",
  states: [
    { name: "plan", state_type: "start" },
    { name: "implement", state_type: "intermediate", sub_protocol_id: "<child-protocol-id>" },
    { name: "done", state_type: "terminal" }
  ],
  transitions: [
    { from_state: "plan", to_state: "implement", trigger: "plan_approved" },
    { from_state: "implement", to_state: "done", trigger: "child_completed" }
  ]
)

// 3. Start a run — entering "implement" auto-spawns the child:
protocol(action: "start_run", protocol_id: "<parent-id>")
protocol(action: "transition", run_id: "<parent-run>", trigger: "plan_approved")
// → child run created automatically, parent waits
```

**Inspecting the run hierarchy:**
- `protocol(action: "get_run", run_id)` — returns the run with its current state, status, and visit history
- `protocol(action: "get_run_tree", run_id)` — returns the run with its full child hierarchy (parent + all nested child runs recursively)
- `protocol(action: "get_run_children", run_id)` — returns only the direct child runs of a given run

**Use cases:** Multi-phase workflows (plan→implement→review), RFC lifecycle with embedded review protocols, wave execution with per-task sub-protocols.

### Knowledge Fabric

The Knowledge Fabric connects all graph entities via semantic relations:

**Automatic relations** (created by the system):
- `TOUCHES`: Commit → File (with additions/deletions) — created automatically via `commit(action: "create", files_changed)`
- `CO_CHANGED`: File ↔ File — files that frequently change together (computed from TOUCHES)
- `SYNAPSE`: Note ↔ Note — weighted neural connections (created by spreading activation, reinforced by co-activation)

**Explicit relations** (created by the agent):
- `AFFECTS`: Decision → File/Function — architectural decisions impacting code
- `DISCUSSED`: ChatSession → File/Function — files discussed in a conversation
- `LINKED_TO`: Note → File/Function/Struct — notes attached to code entities

**Neural feedback cycle**:
1. Notes linked to the same file develop SYNAPSE connections (neural links)
2. Note energy propagates via spreading activation (co-activation reinforces synapses)
3. Weak synapses naturally decay (`admin(action: "decay_synapses")`)
4. Inactive notes lose their energy and become "dead"
5. `admin(action: "update_energy_scores")` recalculates global energy

**Maintenance**:
- `admin(action: "bootstrap_knowledge_fabric")` — initialize the full Knowledge Fabric on an existing project
- `admin(action: "update_fabric_scores")` — recalculate all multi-layer GDS scores
- `admin(action: "update_staleness_scores")` — recalculate note freshness

### Workspace (multi-project)

- `workspace(action: "get_overview", slug)` — workspace overview
- `workspace_milestone(action: "create", slug, title)` — cross-project milestones
- `workspace(action: "get_topology", slug)` — components and service dependencies
- `resource(action: "list", workspace_slug)` — API contracts, shared schemas

### Inheritance strategy (EXTENDS / IMPLEMENTS)

Before modifying a class in an OOP language:
1. `code(action: "get_class_hierarchy", type_name)` — check parents AND children (modifying a class can break its subclasses)
2. `code(action: "find_subclasses", class_name)` — identify all transitive subclasses
3. `code(action: "find_interface_implementors", interface_name)` — all implementations of an interface

**Rule**: always check the inheritance hierarchy BEFORE modifying a protected/public method. A signature change in a parent class can silently break N subclasses.

### Process detection & workflow awareness

Business processes are BFS traces on the CALLS graph from entry points:
- `code(action: "get_entry_points", project_slug)` — main functions, HTTP handlers, CLI commands, event handlers
- `code(action: "list_processes", project_slug)` — all detected processes (name, entry point, steps count)
- `code(action: "get_process", process_id)` — ordered steps of a process

**Use this**: before modifying cross-cutting code (middleware, shared service), explore the processes that traverse it to anticipate side effects.

### Co-change patterns in impact analysis

Complement `analyze_impact` (structural) with co-change signals:
- `code(action: "get_co_change_graph", project_slug)` — global graph of co-modified files
- `code(action: "get_file_co_changers", file_path)` — files often modified alongside a given file

Files that change together are often coupled even without direct imports (temporal coupling). Remember to check them after a modification.

### Community-aware planning

Use `code(action: "get_communities", project_slug)` to segment tasks during planning:
- Each community = cluster of tightly coupled files/functions (`cohesion` score)
- A task should not cross more than 2 communities unless it's a cross-cutting refactoring
- Enriched labels (`enriched_by` field) help name functional modules
- `code(action: "enrich_communities", project_slug)` — enrich labels via LLM (batch)

### Wave Dispatch (parallel plan execution)

When a plan has parallelizable tasks:
1. `plan(action: "get_waves", plan_id)` — computes execution waves (topological sort + conflict splitting by `affected_files`)
2. For each wave, launch tasks with `Task(subagent_type: "general-purpose", run_in_background: true)` — one agent per task
3. Each sub-agent receives the full task context via `task(action: "get_prompt")` and AUTONOMOUSLY updates its steps, creates notes/decisions via MCP
4. `TaskOutput(block: true)` on ALL agent IDs in the wave before proceeding to the next wave
5. **Gotcha**: multiple Bash calls in the same message are SERIALIZED — use `run_in_background: true` for true parallelism
6. Detailed gotchas, sub-agent prompt templates → `note(action: "search", query: "wave-dispatcher")`

### Global vs project-scoped notes

- **Project-scoped note** (with `project_id`): gotcha/pattern specific to a project (e.g., "this API returns 204 not 200")
- **Global note** (without `project_id`): cross-project convention, workspace-wide pattern (e.g., "always use ULID for IDs")
- `note(action: "list_project", slug)` — list notes for a specific project
- `note(action: "list")` — also includes global notes

### add_discussed mandatory

Call `chat(action: "add_discussed", session_id, entities)` for every file/function significantly modified or analyzed during the session. This feeds the DISCUSSED relations in the Knowledge Fabric and improves contextual propagation for future sessions."#;

/// §13 — Best Practices: Personas, Episodes, Structural Analysis, Memory, Intent Search, Scaffolding
pub const SECTION_BP_PERSONAS: &str = r#"### Living Personas

Personas are adaptive knowledge agents scoped to code regions. They accumulate KNOWS relations to files/functions, carry skills and protocols, and can inherit from parent personas.

**When to use:**
- `persona(action: "detect", project_id)` — auto-detect personas from graph clusters (communities + skill distribution)
- `persona(action: "auto_build", project_id)` — auto-build personas from detected patterns
- `persona(action: "find_for_file", project_id, file_path)` — find the best persona for a file (KNOWS + inheritance)
- `persona(action: "activate", persona_id, query)` — activate a persona to get context-aware responses

**Persona composition:**
- `persona(action: "add_file/add_function", persona_id, ...)` — add KNOWS relations
- `persona(action: "add_skill/add_protocol", persona_id, ...)` — attach skills/protocols
- `persona(action: "add_extends", persona_id, parent_id)` — inheritance (child inherits parent's KNOWS)

**Workflow:** detect → auto_build → (optional manual refinement) → find_for_file/activate during work.

### Episodic Memory

Episodes are cognitive snapshots collected from protocol runs: stimulus (what triggered it), process (what happened), outcome (result + learnings).

- `episode(action: "collect", run_id)` — collect episode from a completed protocol run
- `episode(action: "list", project_id)` — list episodes for a project
- `episode(action: "get", episode_id)` — get episode details
- `episode(action: "export", episode_id)` — export for cross-instance portability

**When to collect:** after every significant protocol run (especially system-inference, code-review, wave-execution). Episodes feed pattern recognition for future similar situations.

### Structural Analysis Advanced

Beyond basic GDS metrics, the system offers deeper structural tools:

**Context Cards** — pre-computed contextual profiles per file:
- `code(action: "get_context_card", project_slug, file_path)` — get card (dependencies, co-changers, notes, decisions, hotspot score)
- `code(action: "refresh_context_cards", project_slug)` — refresh all cards after significant changes

**Structural DNA & Twins** — file fingerprinting based on graph topology:
- `code(action: "get_structural_profile", project_slug, file_path)` — get DNA profile
- `code(action: "find_structural_twins", project_slug, file_path, top_n)` — find similar files by DNA cosine
- `code(action: "cluster_dna", project_slug, n_clusters)` — K-means clustering on DNA vectors
- `code(action: "find_cross_project_twins", workspace_slug, file_path, source_project_slug)` — twins across projects

**Stress Testing** — simulate failures to find fragile points:
- `code(action: "stress_test_node", project_slug, target_id)` — simulate node removal
- `code(action: "stress_test_edge", project_slug, from_id, to_id)` — simulate edge removal
- `code(action: "stress_test_cascade", project_slug, target_id, max_iterations)` — cascade failure simulation
- `code(action: "find_bridges", project_slug)` — find single points of failure

**Homeostasis & Drift** — graph stability over time:
- `code(action: "get_homeostasis", project_slug)` — stability/instability metrics
- `code(action: "get_structural_drift", project_slug)` — structural evolution tracking

**Link Prediction** — discover missing relations:
- `code(action: "predict_missing_links", project_slug, min_plausibility)` — predict missing graph edges
- `code(action: "check_link_plausibility", project_slug, source, target)` — check if a specific link should exist

### Memory Horizons & Neural Scars

Notes have a lifecycle from ephemeral to consolidated:
- **Ephemeral notes**: recent observations, tips — may be promoted or decay
- **Consolidated notes**: battle-tested knowledge — promoted from ephemeral based on age + activation count
- `admin(action: "consolidate_memory")` — promote eligible ephemeral notes to consolidated

**Neural scars** track pain points on notes/decisions (scar_intensity field):
- High scar intensity = repeated failures/issues associated with this knowledge
- `admin(action: "heal_scars", node_id)` — reset scar_intensity after the root cause is resolved
- Scars influence note priority in search results and context propagation

### Intent-Aware Search

`note(action: "search_semantic")` and `code(action: "search")` support advanced intent parameters:
- `temperature` (0-1): controls result diversity (0 = precise, 1 = exploratory)
- `intent_mode`: "explore" (broad discovery), "plan" (action-oriented), "debug" (error-focused), "review" (quality-focused)
- `profile`: analysis profile name to customize edge/fusion weights

**Use this:** set intent_mode to match your current phase. During warm-up use "explore", during task execution use "plan" or "debug", during review use "review".

### Scaffolding Levels

The system auto-adapts prompt complexity based on project maturity (0-4):
- **Level 0** (New): Full guidance — all protocols, examples, step-by-step
- **Level 1** (Early): Moderate guidance — protocols + tips, fewer examples
- **Level 2** (Growing): Reduced guidance — key reminders only
- **Level 3** (Mature): Minimal guidance — conventions assumed known
- **Level 4** (Expert): Expert mode — only critical warnings

- `project(action: "get_scaffolding_level", slug)` — get current level (auto-computed from maturity metrics)
- `project(action: "set_scaffolding_override", slug, level)` — override level (null = auto)"#;

/// §12 compact — Best Practices Advanced (condensed for L1, ~1000 tokens)
/// Removes code examples, detailed hierarchical protocol composition,
/// wave dispatch gotchas, and condenses sub-sections into bullet points.
pub const SECTION_BP_ADVANCED_COMPACT: &str = r#"### Advanced Protocols (compact)

**Hierarchical Protocols** — Protocols support parent/child composition via `sub_protocol_id` on states.
- Parent pauses on macro-state until child completes; child linked via `parent_run_id`
- CompletionStrategy: `all_complete` (default) | `any_complete` | `manual`
- OnFailureStrategy: `abort` (default) | `skip` | `retry(N)`
- Create child protocol first, then reference it in parent state's `sub_protocol_id`
- Entering a macro-state auto-spawns the child run; parent waits for completion
- Inspect hierarchy: `get_run` (single run), `get_run_tree` (full hierarchy), `get_run_children` (direct children)
- Use cases: multi-phase workflows (plan→implement→review), RFC lifecycle with embedded review protocols

**Knowledge Fabric** — Connects all graph entities via semantic relations:
- Automatic: TOUCHES (commit→file, with additions/deletions), CO_CHANGED (temporal coupling from TOUCHES), SYNAPSE (note↔note neural links from spreading activation)
- Explicit: AFFECTS (decision→code), DISCUSSED (session→code), LINKED_TO (note→code)
- Neural cycle: linked notes develop synapses → energy propagates → weak synapses decay → inactive notes lose energy
- Maintenance: `bootstrap_knowledge_fabric` (initialize), `update_fabric_scores` (recalculate GDS), `update_staleness_scores` (note freshness)

**Workspace (multi-project):**
- `workspace(action: "get_overview", slug)` — workspace overview
- `workspace_milestone(action: "create", slug, title)` — cross-project milestones
- `workspace(action: "get_topology", slug)` — components and service dependencies
- `resource(action: "list", workspace_slug)` — API contracts, shared schemas

**Inheritance strategy** — Before modifying a class in an OOP language:
- `code(action: "get_class_hierarchy", type_name)` — check parents AND children
- `code(action: "find_subclasses", class_name)` — all transitive subclasses
- `code(action: "find_interface_implementors", interface_name)` — all implementations
- Rule: always check hierarchy BEFORE modifying a protected/public method

**Process detection** — Business processes are BFS traces on CALLS graph:
- `code(action: "get_entry_points/list_processes/get_process")` — discover and inspect processes
- Before modifying cross-cutting code, explore processes that traverse it to anticipate side effects

**Co-change patterns** — Complement structural impact analysis with temporal coupling:
- `code(action: "get_co_change_graph/get_file_co_changers")` — files often modified together
- Files that co-change are often coupled even without direct imports

**Community-aware planning** — `code(action: "get_communities")` to segment tasks:
- Each community = cluster of tightly coupled files (cohesion score)
- A task should not cross more than 2 communities unless cross-cutting refactoring
- `code(action: "enrich_communities")` — enrich labels via LLM

**Wave Dispatch** — For plans with parallelizable tasks:
- `plan(action: "get_waves")` → launch sub-agents per wave (one per task, `run_in_background: true`) → `TaskOutput(block: true)` before next wave
- Each sub-agent gets context via `task(action: "get_prompt")` and autonomously updates steps/notes/decisions

**Global vs project-scoped notes:**
- Project-scoped (with `project_id`): project-specific gotchas/patterns
- Global (without `project_id`): cross-project conventions
- `note(action: "list_project")` for project notes, `note(action: "list")` includes global

**add_discussed mandatory** — Call `chat(action: "add_discussed", session_id, entities)` for every file/function significantly modified or analyzed. Feeds DISCUSSED relations in the Knowledge Fabric."#;

/// §13 compact — Best Practices Personas (condensed for L1, ~800 tokens)
/// Condenses persona workflow, episodes, structural analysis, memory, and intent search
/// into concise bullet points without detailed tool parameter listings.
pub const SECTION_BP_PERSONAS_COMPACT: &str = r#"### Personas & Advanced Features (compact)

**Living Personas** — Adaptive knowledge agents scoped to code regions, accumulating KNOWS relations:
- `persona(action: "detect", project_id)` — auto-detect from graph clusters (communities + skill distribution)
- `persona(action: "auto_build", project_id)` — auto-build from detected patterns
- `persona(action: "find_for_file", project_id, file_path)` — find best persona (KNOWS + inheritance)
- `persona(action: "activate", persona_id, query)` — context-aware activation
- Compose: `add_file`, `add_function`, `add_skill`, `add_protocol`, `add_extends` (inheritance)
- Workflow: detect → auto_build → (optional refinement) → find_for_file/activate during work

**Episodic Memory** — Cognitive snapshots from protocol runs (stimulus → process → outcome):
- `episode(action: "collect", run_id)` — collect after significant protocol runs (system-inference, code-review, wave-execution)
- `episode(action: "list/get/export")` — retrieval and cross-instance portability
- Episodes feed pattern recognition for future similar situations

**Structural Analysis Advanced:**
- Context Cards: `get_context_card` (dependencies, co-changers, notes, hotspot), `refresh_context_cards` (after changes)
- Structural DNA & Twins: `get_structural_profile` (fingerprint), `find_structural_twins` (DNA cosine similarity), `cluster_dna` (K-means), `find_cross_project_twins` (workspace-wide)
- Stress Testing: `stress_test_node/edge/cascade` (simulate failures), `find_bridges` (single points of failure)
- Homeostasis & Drift: `get_homeostasis` (stability metrics), `get_structural_drift` (evolution tracking)
- Link Prediction: `predict_missing_links` (discover missing edges), `check_link_plausibility` (specific link check)

**Memory Horizons & Neural Scars:**
- Notes lifecycle: ephemeral (recent) → consolidated (battle-tested, promoted by age + activation count)
- `admin(action: "consolidate_memory")` — promote eligible notes
- Neural scars: `scar_intensity` tracks pain points, influences search priority and context propagation
- `admin(action: "heal_scars", node_id)` — reset after root cause resolved

**Intent-Aware Search** — `note(action: "search_semantic")` and `code(action: "search")` support:
- `temperature` (0-1): result diversity (0 = precise, 1 = exploratory)
- `intent_mode`: "explore" | "plan" | "debug" | "review" — match to current phase
- `profile`: analysis profile name for custom edge/fusion weights

**Scaffolding Levels** — Auto-adapts prompt complexity based on project maturity (0-4):
- L0 (New) → L1 (Early) → L2 (Growing) → L3 (Mature) → L4 (Expert)
- `project(action: "get_scaffolding_level/set_scaffolding_override")`"#;

/// §14 — Deep Maintenance
pub const SECTION_DEEP_MAINTENANCE: &str = r#"### Deep Maintenance

Full maintenance pipeline for graph health:
- `admin(action: "deep_maintenance")` — runs ALL maintenance in order: decay_synapses → update_energy_scores → update_staleness_scores → maintain_skills → auto_anchor_notes → consolidate_memory
- `admin(action: "detect_stagnation")` — find graph regions with low energy and no recent activity
- `admin(action: "auto_anchor_notes")` — auto-link orphan notes to code entities via content analysis
- `admin(action: "reconstruct_knowledge")` — rebuild missing knowledge links from existing graph signals
- `admin(action: "detect_skill_fission")` — find skills that should split (low cohesion, high internal distance)
- `admin(action: "detect_skill_fusion")` — find skills that should merge (high overlap, strong inter-synapses)

**When to run:** `deep_maintenance` weekly or after major refactoring. `detect_stagnation` to find neglected code areas. `detect_skill_fission/fusion` when skill count grows beyond ~20."#;

/// §15 — Plan Execution & Automation
pub const SECTION_PLAN_EXECUTION: &str = r#"### Plan Execution & Automation

Plans can be executed autonomously and triggered automatically:

**Autonomous execution:**
- `plan(action: "run", plan_id, cwd, project_slug)` — execute plan (spawns agent, processes tasks in wave order)
- `plan(action: "run_status", plan_id)` — check execution status
- `plan(action: "cancel_run", plan_id)` — cancel in-progress execution
- `plan(action: "auto_pr", plan_id)` — auto-generate PR from completed run

**Triggers (automated plan re-execution):**
- `plan(action: "add_trigger", plan_id, trigger_type, config, cooldown_secs)` — schedule/webhook/event triggers
- `plan(action: "list_triggers/remove_trigger/enable_trigger/disable_trigger")` — manage triggers

**Run history & analytics:**
- `plan(action: "list_runs", plan_id)` — execution history
- `plan(action: "compare_runs", plan_id, run_ids)` — diff metrics between runs
- `plan(action: "predict_run", plan_id)` — predict next run outcome from history

**Task delegation:**
- `plan(action: "delegate_task", plan_id, task_id)` — delegate task to sub-agent for autonomous execution
- `plan(action: "enrich", plan_id)` — auto-enrich plan with affected files and dependencies"#;

/// §16 — Delegation awareness: proactive delegation guidance
pub const SECTION_DELEGATION_AWARENESS: &str = r#"## Delegation Awareness

When an active plan has **actionable tasks** (pending with all dependencies satisfied), you should proactively consider delegation:

**When to delegate:**
- Multiple independent tasks are ready (check `plan(action: "get_waves", plan_id)`)
- The current task is completed and the next task has no dependency on user input
- The user is discussing implementation and tasks align with the conversation topic
- At scaffolding level 2+, prefer autonomous delegation over asking permission

**How to delegate:**
1. Use `task(action: "get_next", plan_id)` to find the next actionable task
2. Use `plan(action: "delegate_task", plan_id, task_id)` to spawn a sub-agent
3. The sub-agent works inside this session's envelope: same project, a directory this session can see, a permission mode no wider than this session's. A sub-agent cannot delegate further, and at most 4 run at once
4. Monitor progress via `plan(action: "run_status", plan_id)` or task status updates

**Delegation hints:**
- Tasks with `affected_files` that don't overlap can run in parallel
- Prefer delegation for well-defined tasks (clear acceptance criteria, specific affected files)
- Keep complex/ambiguous tasks for interactive discussion with the user
- If a previous delegation failed (check task `frustration_score`), discuss the approach before re-delegating"#;

// ============================================================================
// Section registry — all 17 base sections with their metadata
// ============================================================================

/// Returns all 16 base prompt sections with their activation conditions.
///
/// At scaffolding level 0 with active plan + protocols, ALL sections activate,
/// producing output equivalent to the original `BASE_SYSTEM_PROMPT`.
pub fn all_base_sections() -> Vec<BasePromptSection> {
    vec![
        BasePromptSection {
            id: PromptSectionId::IdentityRole,
            content: SECTION_IDENTITY_ROLE,
            priority: 0,
            condition: ActivationCondition::always(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::MegatoolsSyntax,
            content: SECTION_MEGATOOLS_SYNTAX,
            priority: 1,
            condition: ActivationCondition::always(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::DataModel,
            content: SECTION_DATA_MODEL,
            priority: 2,
            condition: ActivationCondition::scaffolding_range(0, 2),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::TreeSitterSync,
            content: SECTION_TREE_SITTER,
            priority: 3,
            condition: ActivationCondition::scaffolding_range(0, 2),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::GitWorkflow,
            content: SECTION_GIT_WORKFLOW,
            priority: 4,
            condition: ActivationCondition::scaffolding_range(0, 3),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::TaskExecutionProtocol,
            content: SECTION_TASK_EXECUTION,
            priority: 5,
            condition: ActivationCondition::when_plan_active(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::PlanningProtocol,
            content: SECTION_PLANNING,
            priority: 6,
            // Always included since planning can be requested at any time
            condition: ActivationCondition::scaffolding_range(0, 3),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::StatusManagement,
            content: SECTION_STATUS_MANAGEMENT,
            priority: 7,
            condition: ActivationCondition::scaffolding_range(0, 2),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesImpact,
            content: SECTION_BP_IMPACT,
            priority: 8,
            condition: ActivationCondition::always(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesSearch,
            content: SECTION_BP_SEARCH,
            priority: 9,
            condition: ActivationCondition::scaffolding_range(0, 3),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesKnowledge,
            content: SECTION_BP_KNOWLEDGE,
            priority: 10,
            condition: ActivationCondition::always(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesAdvanced,
            content: SECTION_BP_ADVANCED,
            priority: 11,
            // L0: full content, L1: compact variant
            condition: ActivationCondition::scaffolding_range_with_compact(0, 1, 1, 1),
            compact_content: Some(SECTION_BP_ADVANCED_COMPACT),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesPersonas,
            content: SECTION_BP_PERSONAS,
            priority: 12,
            // L0: full content, L1: compact variant
            condition: ActivationCondition::scaffolding_range_with_compact(0, 1, 1, 1),
            compact_content: Some(SECTION_BP_PERSONAS_COMPACT),
        },
        BasePromptSection {
            id: PromptSectionId::DeepMaintenance,
            content: SECTION_DEEP_MAINTENANCE,
            priority: 13,
            condition: ActivationCondition::scaffolding_range(0, 1),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::PlanExecutionAutomation,
            content: SECTION_PLAN_EXECUTION,
            priority: 14,
            condition: ActivationCondition::when_plan_active(),
            compact_content: None,
        },
        BasePromptSection {
            id: PromptSectionId::DelegationAwareness,
            content: SECTION_DELEGATION_AWARENESS,
            priority: 15,
            condition: ActivationCondition {
                scaffolding_min: None,
                scaffolding_max: None,
                requires_active_plan: true,
                requires_active_protocol: false,
                min_task_count: Some(2), // Only show when there are multiple tasks
                always: false,
                compact_scaffolding_range: None,
            },
            compact_content: None,
        },
        // §17 — Tool Reference is handled separately (selective by group)
        // but included in the registry for completeness
        BasePromptSection {
            id: PromptSectionId::ToolReference,
            content: "", // Content comes from TOOL_REFERENCE or selected groups
            priority: 20,
            condition: ActivationCondition::always(),
            compact_content: None,
        },
    ]
}

/// Select sections that are active given the current context.
///
/// Returns sections filtered by their activation conditions, sorted by priority.
/// When a section has a compact variant and the scaffolding level falls within
/// its `compact_scaffolding_range`, the compact content replaces the full content.
pub fn select_sections(ctx: &ComposerContext) -> Vec<BasePromptSection> {
    all_base_sections()
        .into_iter()
        .filter(|s| s.condition.is_active(ctx))
        .map(|mut s| {
            if s.condition.use_compact(ctx.scaffolding_level) {
                if let Some(compact) = s.compact_content {
                    s.content = compact;
                }
            }
            s
        })
        .collect()
}

/// Assemble selected sections into a single prompt string.
///
/// Joins section contents with double newlines, preserving the original
/// markdown structure.
pub fn assemble_sections(sections: &[BasePromptSection]) -> String {
    sections
        .iter()
        .filter(|s| !s.content.is_empty())
        .map(|s| s.content)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Assemble sections with weight-based truncation.
///
/// When the total base prompt exceeds `max_chars`, sections with the lowest
/// weights (from `SectionHint`) are truncated first. Sections not in the
/// hints map are treated as weight 1.0 (kept at full size).
///
/// Truncation strategy:
/// 1. Sections with weight 0.0 are excluded entirely
/// 2. Remaining sections are sorted by weight ascending
/// 3. Lowest-weight sections are progressively truncated (to 50% then 25% then 0%)
///    until the total fits within `max_chars`
pub fn assemble_sections_weighted(
    sections: &[BasePromptSection],
    hints: &[(PromptSectionId, f32)],
    max_chars: usize,
) -> String {
    use std::collections::HashMap;

    let weight_map: HashMap<PromptSectionId, f32> = hints.iter().cloned().collect();

    // Filter out weight=0.0 sections and empty sections
    let weighted: Vec<(&BasePromptSection, f32)> = sections
        .iter()
        .filter(|s| !s.content.is_empty())
        .filter_map(|s| {
            let w = weight_map.get(&s.id).copied().unwrap_or(1.0);
            if w <= 0.0 {
                None
            } else {
                Some((s, w))
            }
        })
        .collect();

    // Check if we're under budget with all sections
    let total: usize = weighted.iter().map(|(s, _)| s.content.len()).sum();
    if total <= max_chars {
        return weighted
            .iter()
            .map(|(s, _)| s.content)
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    // Over budget: progressively truncate lowest-weight sections
    // Sort by weight ascending (lowest weight = truncated first)
    let mut by_weight: Vec<(usize, f32)> = weighted
        .iter()
        .enumerate()
        .map(|(i, (_, w))| (i, *w))
        .collect();
    by_weight.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build content with truncation ratios
    let mut ratios: Vec<f32> = vec![1.0; weighted.len()];
    let truncation_steps = [0.5, 0.25, 0.0]; // Progressive truncation

    'outer: for &step in &truncation_steps {
        for &(idx, _weight) in &by_weight {
            ratios[idx] = step;
            let current_total: usize = weighted
                .iter()
                .enumerate()
                .map(|(i, (s, _))| {
                    let content_len = s.content.len();
                    (content_len as f32 * ratios[i]) as usize
                })
                .sum();
            if current_total <= max_chars {
                break 'outer;
            }
        }
    }

    // Render with applied ratios
    weighted
        .iter()
        .enumerate()
        .filter_map(|(i, (s, _))| {
            let ratio = ratios[i];
            if ratio <= 0.0 {
                return None;
            }
            if ratio >= 1.0 {
                return Some(s.content.to_string());
            }
            // Truncate to ratio of original length
            let target_len = (s.content.len() as f32 * ratio) as usize;
            let boundary = floor_char_boundary_pub(s.content, target_len);
            let mut truncated = s.content[..boundary].to_string();
            truncated.push_str("\n\n[... section truncated]");
            Some(truncated)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Find the largest char boundary <= the given byte index.
/// Public variant for use from other modules.
pub fn floor_char_boundary_pub(s: &str, index: usize) -> usize {
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
// Audience — the engine's sub-agent vocabulary, and the frame of a child session
// ============================================================================

/// The Wave Dispatch steps as Claude Code understands them (`Task`, `TaskOutput`,
/// `run_in_background`), byte for byte as they stand in the advanced best-practices
/// section: what [`adapt_subagent_vocabulary`] replaces on another engine. A test keeps
/// the two texts from drifting apart.
pub const WAVE_DISPATCH_CLAUDE_CODE: &str = r#"2. For each wave, launch tasks with `Task(subagent_type: "general-purpose", run_in_background: true)` — one agent per task
3. Each sub-agent receives the full task context via `task(action: "get_prompt")` and AUTONOMOUSLY updates its steps, creates notes/decisions via MCP
4. `TaskOutput(block: true)` on ALL agent IDs in the wave before proceeding to the next wave
5. **Gotcha**: multiple Bash calls in the same message are SERIALIZED — use `run_in_background: true` for true parallelism"#;

/// The same steps for an engine whose sub-agent tool is `Agent` (the native harness,
/// Codex, ACP): the call returns with the child's answer, several calls in one message
/// run side by side, there is no background mode and nothing to collect afterwards.
pub const WAVE_DISPATCH_AGENT: &str = r#"2. For each wave, launch its tasks with one `Agent(description, prompt)` call per task, all in the SAME message: they run side by side, and each call returns with its agent's final answer (no background mode, nothing to collect afterwards)
3. Each sub-agent receives the full task context via `task(action: "get_prompt")` and AUTONOMOUSLY updates its steps, creates notes/decisions via MCP
4. Read every answer of the wave before launching the next wave
5. **Gotcha**: a sub-agent starts with no memory of this conversation and cannot start sub-agents of its own — its prompt must say everything it needs"#;

/// The compact line of the same subsection (the compact variant of the section).
pub const WAVE_DISPATCH_CLAUDE_CODE_COMPACT: &str = "- `plan(action: \"get_waves\")` → launch sub-agents per wave (one per task, `run_in_background: true`) → `TaskOutput(block: true)` before next wave";
/// Its `Agent` counterpart.
pub const WAVE_DISPATCH_AGENT_COMPACT: &str = "- `plan(action: \"get_waves\")` → one `Agent(description, prompt)` call per task of the wave, all in the same message (they run side by side, each returns its agent's answer) → read every answer before the next wave";

/// The search section's mention of Claude Code's exploring agent, and its `Agent` form
/// (the native `Agent` takes a `subagent_type` too).
pub const EXPLORE_CLAUDE_CODE: &str = "Task(Explore) for exploring";
/// Its `Agent` counterpart.
pub const EXPLORE_AGENT: &str = "Agent(subagent_type: \"Explore\") for exploring";

/// Which words an engine has for sub-agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubagentVocabulary {
    /// Claude Code: `Task(subagent_type, run_in_background)`, then `TaskOutput(block)`.
    /// The words written in the sections: this vocabulary changes nothing.
    #[default]
    ClaudeCode,
    /// `Agent(description, prompt)`, synchronous: the native harness, Codex, ACP.
    Agent,
}

impl SubagentVocabulary {
    /// The vocabulary of the engine a session runs on. Claude Code (also an empty id,
    /// which the stored sessions use for it) keeps its own words; every other provider
    /// gets `Agent`.
    pub fn of_provider(provider_id: &str) -> Self {
        if provider_id.is_empty() || provider_id == super::provider::resolver::CLAUDE_CODE {
            Self::ClaudeCode
        } else {
            Self::Agent
        }
    }
}

/// Rewrite the sub-agent vocabulary of an assembled base prompt for `vocabulary`.
/// Claude Code's is the one written in the sections: for it the prompt comes back as
/// it went in.
pub fn adapt_subagent_vocabulary(prompt: String, vocabulary: SubagentVocabulary) -> String {
    match vocabulary {
        SubagentVocabulary::ClaudeCode => prompt,
        SubagentVocabulary::Agent => prompt
            .replace(WAVE_DISPATCH_CLAUDE_CODE, WAVE_DISPATCH_AGENT)
            .replace(
                WAVE_DISPATCH_CLAUDE_CODE_COMPACT,
                WAVE_DISPATCH_AGENT_COMPACT,
            )
            .replace(EXPLORE_CLAUDE_CODE, EXPLORE_AGENT),
    }
}

/// How a child of a conversation was opened, for its prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildFrame {
    /// The session that opened it.
    pub parent_session_id: String,
    /// `true`: it answers one sub-agent call of the parent, and its final message is
    /// that call's result; `false`: a sub-conversation opened beside the parent, read
    /// by whoever opened it.
    pub answers_a_call: bool,
}

/// The section a child session gets and a root session never does: who opened it,
/// where its answer goes, what nobody will do for it. Not a routed section: it is
/// appended after the base sections and never weighted away.
pub fn child_section(frame: &ChildFrame) -> String {
    let parent = &frame.parent_session_id;
    if frame.answers_a_call {
        format!(
            "## You are a sub-agent\n\n\
             This session was opened by session `{parent}` through its sub-agent call. Your \
             final message is returned to that session, verbatim, as the result of its call: \
             it sees nothing else of your work.\n\
             - No human reads this conversation as it happens, and nobody will answer a \
             question asked here: decide, or report what you could not decide. A permission \
             the parent did not already grant fails at once.\n\
             - End with a complete, self-contained answer: findings with their file paths, \
             what you changed, what remains.\n\
             - You cannot start sub-agents of your own (one level only).\n\
             - Keep the usual tracking (steps, notes, decisions through the MCP tools): it is \
             how the parent, and the people behind it, see what you did."
        )
    } else {
        format!(
            "## You are a sub-conversation\n\n\
             This session was opened beside session `{parent}`, as a sub-conversation of it. \
             Whoever opened it (that session, or a person watching it) reads your answers: \
             write each one complete and self-contained.\n\
             - You cannot start sub-agents of your own (one level only).\n\
             - Keep the usual tracking (steps, notes, decisions through the MCP tools): it is \
             how the parent, and the people behind it, see what you did."
        )
    }
}

// ============================================================================
// 3D scenes — the ```iso-scene block the chat interface draws
// ============================================================================

/// What the agent of a conversation a person reads in the PO interface is told about
/// the ```iso-scene block (frontend `src/components/chat/viz/iso/`, PR #400): the
/// interface draws it as an animated 3D scene, any other reader sees its JSON. Added by
/// the composer only when [`super::composer::PromptAudience::displays_scenes`] says so.
/// Its vocabulary and example are checked against the frontend's contract by
/// `tests::the_iso_scene_section_matches_the_frontend_contract`.
pub const SECTION_ISO_SCENE: &str = r#"## 3D scenes

The interface draws a fenced `iso-scene` block as an animated 3D scene. Use one (at most one per answer) when a picture explains better than prose: a mechanism, an architecture, a progression, interacting parts; never for plain text or code, and still say it in words.

The body is JSON data only; an unknown field refuses the scene:

```iso-scene
{"title":"Release","stations":[
 {"mech":"dependency-graph","tone":"violet","label":"Plan"},
 {"mech":"parallel-lanes","tone":"sky","label":"Build","after":[0]},
 {"ast":{"kind":"use","ref":"flag"},"tone":"green","label":"Ship","after":[1]}],
"build":{"mode":"sequential"}}
```

- Scene fields: title, caption, layout, stations, build, focal (perspective, 0 = isometric). Station fields: one of mech or ast, tone, label (≤ 60 chars), after (indices, no cycle).
- Keep 2 to 5 stations (more rarely fit). Normally omit layout; name one only when its shape means something: arch (a progression), ring (a whole and its parts), wings (two sides).
- mech: drawers, dependency-graph, parallel-lanes, knowledge-graph, local-vault, skill-cartridge, stair-procedure, consent-gate, model-selector, reach-devices.
- tone: green, indigo, amber, red, violet, sky. build.mode: parallel, sequential, waves.
- One object: {"title":"...","ast":NODE}. NODE kinds: use (ref: agent, screen, chip, flag, arrow, server, document, node, step), group (children), slab (size), sphere or ring (r), puck or cone (r, h), tube (to, r), lathe (profile, seg), capsule (r, h), torus (r, t), repeat (shape, positions 1-128 — the shape counts once); optional at, rot, tone, mat, motion. Vectors [x,y,z]; sizes 0.01-6; at most 8 levels, 160 shapes. Prefer a mech."#;

// ============================================================================
// Tool Reference Groups — selective rendering of TOOL_REFERENCE
// ============================================================================

/// Identifies a group of related MCP tools for selective rendering.
///
/// Instead of always including the full ~600-line TOOL_REFERENCE, the system
/// selects relevant groups based on intent, FSM state, and scaffolding level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolRefGroupId {
    /// project, plan, task, step, constraint, release, environment, provider, vault,
    /// milestone — always included
    Core,
    /// note, decision, commit — always included (compact)
    Knowledge,
    /// code, analysis_profile — included when code questions / file mentions
    CodeExploration,
    /// admin — included at L0-L1 or when GDS/maintenance requested
    Structural,
    /// protocol, skill, persona, episode, lifecycle_hook — included when protocols active or L0-L1
    Behavioral,
    /// workspace, workspace_milestone, resource, component — included in multi-project
    Workspace,
    /// chat, feature_graph, sharing, reasoning, neural_routing, trajectory — included on demand
    Collaboration,
    /// mcp_federation — included when external MCP servers are connected
    External,
}

impl ToolRefGroupId {
    /// All group IDs.
    pub const ALL: &'static [ToolRefGroupId] = &[
        Self::Core,
        Self::Knowledge,
        Self::CodeExploration,
        Self::Structural,
        Self::Behavioral,
        Self::Workspace,
        Self::Collaboration,
        Self::External,
    ];

    /// Tool names belonging to this group.
    pub fn tool_names(&self) -> &'static [&'static str] {
        match self {
            Self::Core => &[
                "project",
                "plan",
                "task",
                "step",
                "constraint",
                "release",
                "environment",
                "provider",
                "vault",
                "milestone",
                "day_plan",
            ],
            Self::Knowledge => &["note", "decision", "commit"],
            Self::CodeExploration => &["code", "analysis_profile"],
            Self::Structural => &["admin"],
            Self::Behavioral => &["protocol", "skill", "persona", "episode", "lifecycle_hook"],
            Self::Workspace => &["workspace", "workspace_milestone", "resource", "component"],
            Self::Collaboration => &[
                "chat",
                "feature_graph",
                "sharing",
                "reasoning",
                "neural_routing",
                "trajectory",
            ],
            Self::External => &["mcp_federation"],
        }
    }

    /// Display name for the group header.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Core => "Core (Project, Planning, Tracking)",
            Self::Knowledge => "Knowledge (Notes, Decisions, Commits)",
            Self::CodeExploration => "Code Exploration & Analytics",
            Self::Structural => "Admin & Sync",
            Self::Behavioral => {
                "Behavioral (Protocols, Skills, Personas, Episodes, Lifecycle Hooks)"
            }
            Self::Workspace => "Workspace (Multi-project)",
            Self::Collaboration => {
                "Collaboration (Chat, Features, Sharing, Reasoning, Neural Routing)"
            }
            Self::External => "External (MCP Federation)",
        }
    }
}

/// Context for tool group selection.
#[derive(Default)]
pub struct ToolGroupSelectionContext {
    pub scaffolding_level: u8,
    pub has_active_protocol: bool,
    pub is_multi_project: bool,
    /// Whether external MCP servers are connected (triggers External group).
    pub external_tools_available: bool,
    /// Tool names whitelisted by the current FSM state's `available_tools`.
    /// Empty = no FSM restriction (include all).
    pub fsm_available_tools: Vec<String>,
    /// Keywords detected in the user message for intent matching.
    pub user_intent_keywords: Vec<String>,
}

// Default is derived — all fields have natural defaults (0, false, empty vecs).

/// Check if `text` contains `keyword` as a whole word (not as a substring).
///
/// Uses word-boundary detection: the character before and after the keyword
/// must be non-alphanumeric (or absent). This prevents "profile" matching "file",
/// "findings" matching "find", etc.
///
/// Multi-word keywords (e.g. "call graph", "state machine") are matched via
/// simple substring contains, since they are already specific enough.
fn contains_word(text: &str, keyword: &str) -> bool {
    // Multi-word keywords: substring match is fine (already specific)
    if keyword.contains(' ') {
        return text.contains(keyword);
    }
    let kw_len = keyword.len();
    if kw_len == 0 || kw_len > text.len() {
        return false;
    }
    let mut start = 0usize;
    while let Some(off) = text[start..].find(keyword) {
        let i = start + off;

        // Boundary check is Unicode-aware: `is_ascii_alphanumeric` treats the
        // leading byte of an accented char as a boundary, so "partage" matched
        // inside "partagée" and "note" inside "noté". `char::is_alphanumeric`
        // does not.
        let before_ok = text[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());

        let mut after = text[i + kw_len..].chars();
        let after_ok = match after.next() {
            None => true,
            Some(c) if !c.is_alphanumeric() => true,
            // Tolerate a single plural marker: "sessions" matches "session",
            // "compétences" matches "compétence". "findings" still does not
            // match "find" (the next char is 'i', not a boundary).
            Some('s') | Some('S') => after.next().is_none_or(|c| !c.is_alphanumeric()),
            Some(_) => false,
        };

        if before_ok && after_ok {
            return true;
        }
        start = i + kw_len;
    }
    false
}

/// Intent keywords that trigger specific tool groups.
const CODE_KEYWORDS: &[&str] = &[
    // EN
    "code",
    "function",
    "file",
    "import",
    "search",
    "find",
    "reference",
    "symbol",
    "architecture",
    "impact",
    "call graph",
    "dependency",
    "dependencies",
    "trait",
    "struct",
    "class",
    "method",
    "refactor",
    "signature",
    "codebase",
    // FR
    "cherche",
    "chercher",
    "fonction",
    "fichier",
    "recherche",
    "référence",
    "symbole",
    "graphe d'appels",
    "dépendance",
    "classe",
    "méthode",
    "refactorisation",
    "arborescence",
    "appelant",
    "appelé",
];
const STRUCTURAL_KEYWORDS: &[&str] = &[
    // EN
    "admin",
    "sync",
    "maintenance",
    "health",
    "fabric",
    "gds",
    "neuron",
    "synapse",
    "energy",
    "staleness",
    "hotspot",
    "technical debt",
    "risk score",
    // FR
    "synchronisation",
    "synchroniser",
    "santé",
    "neurone",
    "énergie",
    "obsolescence",
    "point chaud",
    "points chauds",
    "dette technique",
    "score de risque",
    "maintenance profonde",
];
const BEHAVIORAL_KEYWORDS: &[&str] = &[
    // EN
    "protocol",
    "skill",
    "persona",
    "episode",
    "fsm",
    "state machine",
    "transition",
    "lifecycle",
    "hook",
    // FR
    "protocole",
    "compétence",
    "épisode",
    "machine à états",
    "machine a etats",
    "cycle de vie",
    "état de la machine",
];
const WORKSPACE_KEYWORDS: &[&str] = &[
    // EN
    "workspace",
    "component",
    "resource",
    "multi-project",
    "topology",
    "cross-project",
    "monorepo",
    // FR
    "espace de travail",
    "composant",
    "ressource",
    "multi-projet",
    "topologie",
    "inter-projet",
    "inter-projets",
    "transverse",
];
const COLLAB_KEYWORDS: &[&str] = &[
    // EN
    "chat",
    "conversation",
    "discussion",
    "feature graph",
    "sharing",
    "share",
    "reasoning",
    "trajectory",
    "neural routing",
    // FR
    "partage",
    "partager",
    "raisonnement",
    "arbre de raisonnement",
    "trajectoire",
    "routage neuronal",
    "graphe de fonctionnalités",
    // NOTE: "session", "reason" and "raison" were REMOVED. Measured by
    // tests/routing_metamorphic.rs: the filler sentence "On en reparle à la
    // prochaine session." injected the whole Collaboration group (~1223 tokens)
    // into 20/20 unrelated messages. A bare generic noun that names the medium
    // rather than the intent must not be a sole trigger.
];

/// Select which tool reference groups to include in the prompt.
///
/// Selection logic:
/// 1. Core + Knowledge are ALWAYS included
/// 2. FSM available_tools whitelist overrides everything (if non-empty)
/// 3. Intent keywords trigger relevant groups
/// 4. Scaffolding level gates advanced groups (Structural, Behavioral at L0-L1)
/// 5. Multi-project context triggers Workspace
pub fn select_tool_groups(ctx: &ToolGroupSelectionContext) -> Vec<ToolRefGroupId> {
    // If FSM restricts tools, only include groups that have at least one whitelisted tool
    if !ctx.fsm_available_tools.is_empty() {
        return ToolRefGroupId::ALL
            .iter()
            .copied()
            .filter(|group| {
                // Core is always included
                if matches!(group, ToolRefGroupId::Core | ToolRefGroupId::Knowledge) {
                    return true;
                }
                // Include group if any of its tools is in the whitelist
                group.tool_names().iter().any(|tool| {
                    ctx.fsm_available_tools
                        .iter()
                        .any(|allowed| allowed == tool)
                })
            })
            .collect();
    }

    let mut selected = vec![ToolRefGroupId::Core, ToolRefGroupId::Knowledge];
    let msg_lower: String = ctx.user_intent_keywords.join(" ").to_lowercase();

    // Intent-based inclusion
    let has_code_intent = CODE_KEYWORDS.iter().any(|kw| contains_word(&msg_lower, kw));
    let has_structural_intent = STRUCTURAL_KEYWORDS
        .iter()
        .any(|kw| contains_word(&msg_lower, kw));
    let has_behavioral_intent = BEHAVIORAL_KEYWORDS
        .iter()
        .any(|kw| contains_word(&msg_lower, kw));
    let has_workspace_intent = WORKSPACE_KEYWORDS
        .iter()
        .any(|kw| contains_word(&msg_lower, kw));
    let has_collab_intent = COLLAB_KEYWORDS
        .iter()
        .any(|kw| contains_word(&msg_lower, kw));

    // CodeExploration: included by intent OR at L0-L2 (common need)
    if has_code_intent || ctx.scaffolding_level <= 2 {
        selected.push(ToolRefGroupId::CodeExploration);
    }

    // Structural: included by intent OR at L0-L1
    if has_structural_intent || ctx.scaffolding_level <= 1 {
        selected.push(ToolRefGroupId::Structural);
    }

    // Behavioral: included by intent OR active protocols OR L0-L1
    if has_behavioral_intent || ctx.has_active_protocol || ctx.scaffolding_level <= 1 {
        selected.push(ToolRefGroupId::Behavioral);
    }

    // Workspace: included by intent OR multi-project context OR L0
    if has_workspace_intent || ctx.is_multi_project || ctx.scaffolding_level == 0 {
        selected.push(ToolRefGroupId::Workspace);
    }

    // Collaboration: included by intent OR L0
    if has_collab_intent || ctx.scaffolding_level == 0 {
        selected.push(ToolRefGroupId::Collaboration);
    }

    // External: included when external MCP servers are connected
    if ctx.external_tools_available {
        selected.push(ToolRefGroupId::External);
    }

    selected
}

/// Extract tool sections from the full TOOL_REFERENCE text for a given set of groups.
///
/// Parses TOOL_REFERENCE by finding `## toolname` headers and extracting only
/// the sections whose tool names match the selected groups.
pub fn extract_tool_reference(tool_reference: &str, groups: &[ToolRefGroupId]) -> String {
    // Collect all tool names we want
    let wanted_tools: Vec<&str> = groups
        .iter()
        .flat_map(|g| g.tool_names().iter().copied())
        .collect();

    let mut result = String::from("# MCP Mega-Tools Reference\n\nAll tools require `action` (string). UUIDs are strings. Dates are ISO 8601.\n");
    let mut current_tool: Option<&str> = None;
    let mut current_section = String::new();
    let mut include_current = false;

    for line in tool_reference.lines() {
        // Detect `## toolname` headers
        if line.starts_with("## ") && !line.starts_with("### ") {
            // Flush previous section if included
            if include_current && !current_section.is_empty() {
                result.push('\n');
                result.push_str(&current_section);
            }

            // Parse new tool name
            let tool_name = line[3..].split_whitespace().next().unwrap_or("");
            // Tool name is the word after "## " — e.g., "## project" → "project"
            // But some lines are "## project\n" and some are like "## workspace_milestone"
            current_tool = Some(tool_name);
            include_current = wanted_tools.contains(&tool_name);
            current_section.clear();
            if include_current {
                current_section.push_str(line);
                current_section.push('\n');
            }
        } else if line.starts_with("# ") && !line.starts_with("## ") {
            // Skip the main header (already added above)
            // Also flush any pending section
            if include_current && !current_section.is_empty() {
                result.push('\n');
                result.push_str(&current_section);
            }
            current_section.clear();
            include_current = false;
            current_tool = None;
        } else if current_tool.is_some() && include_current {
            current_section.push_str(line);
            current_section.push('\n');
        }
    }

    // Flush last section
    if include_current && !current_section.is_empty() {
        result.push('\n');
        result.push_str(&current_section);
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    /// The Claude Code needles are the sections' own text, and the `Agent` vocabulary
    /// leaves no Claude Code sub-agent word behind, in the full and compact variants.
    #[test]
    fn the_wave_dispatch_needles_match_the_sections_and_agent_leaves_no_claude_code_word() {
        let all: String = all_base_sections()
            .iter()
            .flat_map(|s| [Some(s.content), s.compact_content])
            .flatten()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            all.contains(WAVE_DISPATCH_CLAUDE_CODE),
            "WAVE_DISPATCH_CLAUDE_CODE drifted from the full section"
        );
        assert!(
            all.contains(WAVE_DISPATCH_CLAUDE_CODE_COMPACT),
            "WAVE_DISPATCH_CLAUDE_CODE_COMPACT drifted from the compact section"
        );
        assert!(
            all.contains(EXPLORE_CLAUDE_CODE),
            "EXPLORE_CLAUDE_CODE drifted from the search section"
        );
        let agent = adapt_subagent_vocabulary(all.clone(), SubagentVocabulary::Agent);
        assert!(
            !agent.contains("TaskOutput")
                && !agent.contains("run_in_background")
                && !agent.contains("Task("),
            "a Claude Code sub-agent word survives on the Agent engine"
        );
        assert!(agent.contains(WAVE_DISPATCH_AGENT) && agent.contains(WAVE_DISPATCH_AGENT_COMPACT));
        assert_eq!(
            adapt_subagent_vocabulary(all.clone(), SubagentVocabulary::ClaudeCode),
            all,
            "Claude Code's vocabulary changes nothing"
        );
        assert_eq!(
            SubagentVocabulary::of_provider("claude-code"),
            SubagentVocabulary::ClaudeCode
        );
        assert_eq!(
            SubagentVocabulary::of_provider(""),
            SubagentVocabulary::ClaudeCode
        );
        assert_eq!(
            SubagentVocabulary::of_provider("deepseek"),
            SubagentVocabulary::Agent
        );
    }

    #[test]
    fn the_child_section_names_the_parent_and_tells_a_call_child_from_a_sub_conversation() {
        let call = child_section(&ChildFrame {
            parent_session_id: "p-1".into(),
            answers_a_call: true,
        });
        assert!(call.starts_with("## You are a sub-agent"));
        assert!(call.contains("`p-1`") && call.contains("nobody will answer a question"));
        let beside = child_section(&ChildFrame {
            parent_session_id: "p-2".into(),
            answers_a_call: false,
        });
        assert!(beside.starts_with("## You are a sub-conversation"));
        assert!(beside.contains("`p-2`") && !beside.contains("nobody will answer"));
        for s in [&call, &beside] {
            assert!(
                s.contains("sub-agents of your own"),
                "the depth rule is in both"
            );
        }
    }

    #[test]
    fn test_all_sections_count() {
        let sections = all_base_sections();
        assert_eq!(sections.len(), 17, "Expected 17 base sections");
    }

    #[test]
    fn test_section_ids_unique() {
        let sections = all_base_sections();
        let mut ids: Vec<_> = sections.iter().map(|s| s.id).collect();
        let original_len = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "Section IDs must be unique");
    }

    #[test]
    fn test_section_priorities_reasonable() {
        let sections = all_base_sections();
        for s in &sections {
            assert!(
                s.priority <= 20,
                "Priority {} too high for {:?}",
                s.priority,
                s.id
            );
        }
    }

    #[test]
    fn test_level0_full_activates_all() {
        let ctx = ComposerContext::level0_full();
        let selected = select_sections(&ctx);
        assert_eq!(
            selected.len(),
            17,
            "L0 with full context should activate all 17 sections, got {}",
            selected.len()
        );
    }

    #[test]
    fn test_level4_reduces_sections() {
        let ctx = ComposerContext {
            scaffolding_level: 4,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };
        let selected = select_sections(&ctx);
        // At L4: Identity(always), Megatools(always), BP Impact(always),
        // BP Knowledge(always), TaskExec(plan), PlanExec(plan), ToolRef(always)
        assert!(
            selected.len() < 16,
            "L4 should activate fewer sections, got {}",
            selected.len()
        );
        assert!(
            selected.len() >= 5,
            "L4 should still have at least 5 sections (always + plan), got {}",
            selected.len()
        );
    }

    #[test]
    fn test_critical_sections_never_excluded() {
        // Even at L4 without plan or protocol, critical sections must remain
        let ctx = ComposerContext {
            scaffolding_level: 4,
            has_active_plan: false,
            has_active_protocol: false,
            task_count: 0,
        };
        let selected = select_sections(&ctx);
        let ids: Vec<_> = selected.iter().map(|s| s.id).collect();

        assert!(
            ids.contains(&PromptSectionId::IdentityRole),
            "Identity always included"
        );
        assert!(
            ids.contains(&PromptSectionId::MegatoolsSyntax),
            "Megatools always included"
        );
        assert!(
            ids.contains(&PromptSectionId::BestPracticesImpact),
            "BP Impact always included"
        );
        assert!(
            ids.contains(&PromptSectionId::BestPracticesKnowledge),
            "BP Knowledge always included"
        );
        assert!(
            ids.contains(&PromptSectionId::ToolReference),
            "ToolRef always included"
        );
    }

    #[test]
    fn test_plan_dependent_sections_excluded_without_plan() {
        let ctx = ComposerContext {
            scaffolding_level: 0,
            has_active_plan: false,
            has_active_protocol: false,
            task_count: 0,
        };
        let selected = select_sections(&ctx);
        let ids: Vec<_> = selected.iter().map(|s| s.id).collect();

        assert!(
            !ids.contains(&PromptSectionId::TaskExecutionProtocol),
            "TaskExec excluded without plan"
        );
        assert!(
            !ids.contains(&PromptSectionId::PlanExecutionAutomation),
            "PlanExec excluded without plan"
        );
    }

    /// The contract of the frontend's ```iso-scene block: SCENE_KEYS, STATION_KEYS,
    /// MECHANISMS, TONES, LAYOUTS, OBJECTS, BUILD_MODES and limits are the engine's
    /// contract (`src/vendor/three-iso/`). SCENE_KEYS and KINDS were extended with
    /// `focal`, `lathe`, `capsule`, `repeat`, `torus` (frontend PRs #407/#408/#409).
    /// Re-copy it when the frontend's contract moves.
    mod iso_contract {
        #[allow(unused_imports)]
        use crate::neo4j::traits::*;
        pub const SCENE_KEYS: &[&str] = &[
            "title",
            "caption",
            "layout",
            "stations",
            "angles",
            "home",
            "stepSeconds",
            "build",
            "ast",
            "tone",
            "label",
            "focal",
        ];
        pub const STATION_KEYS: &[&str] = &["mech", "ast", "tone", "label", "after"];
        pub const MECHANISMS: &[&str] = &[
            "drawers",
            "dependency-graph",
            "parallel-lanes",
            "knowledge-graph",
            "local-vault",
            "skill-cartridge",
            "stair-procedure",
            "consent-gate",
            "model-selector",
            "reach-devices",
        ];
        pub const TONES: &[&str] = &["green", "indigo", "amber", "red", "violet", "sky"];
        /// Layouts and their station range (`STATION_RANGE`).
        pub const LAYOUTS: &[(&str, usize, usize)] = &[
            ("ring", 3, 12),
            ("stadium", 4, 12),
            ("arch", 3, 9),
            ("wings", 2, 10),
            ("corners", 4, 12),
            ("solo", 1, 1),
        ];
        pub const KINDS: &[&str] = &[
            "slab", "puck", "sphere", "tube", "ring", "cone", "lathe", "capsule", "repeat",
            "torus", "group", "use",
        ];
        pub const OBJECTS: &[&str] = &[
            "agent", "screen", "chip", "flag", "arrow", "server", "document", "node", "step",
        ];
        pub const BUILD_MODES: &[&str] = &["parallel", "sequential", "waves"];
        pub const MAX_LABEL: usize = 60;
        pub const MAX_DEPTH: usize = 8;
        pub const MAX_SHAPES: usize = 160;
    }

    /// The checks of `compile.ts` / `defineScene` / `checkAst` a model's scene meets
    /// first: known fields and names, one of `mech`/`ast`, station count, `after`, AST
    /// depth and shapes. Returns the problems (empty = accepted).
    fn iso_scene_problems(scene: &serde_json::Value) -> Vec<String> {
        use iso_contract::*;
        use serde_json::Value;
        let mut out = Vec::new();
        let Some(obj) = scene.as_object() else {
            return vec!["not an object".into()];
        };
        for k in obj.keys() {
            if !SCENE_KEYS.contains(&k.as_str()) {
                out.push(format!("unknown scene field {k}"));
            }
        }
        fn walk(n: &Value, depth: usize, shapes: &mut usize, out: &mut Vec<String>) {
            let kind = n.get("kind").and_then(Value::as_str).unwrap_or("");
            if !KINDS.contains(&kind) {
                out.push(format!("unknown kind {kind}"));
                return;
            }
            if depth > MAX_DEPTH {
                out.push("tree too deep".into());
            }
            if let Some(t) = n.get("tone").and_then(Value::as_str) {
                if t != "accent" && !TONES.contains(&t) {
                    out.push(format!("unknown node tone {t}"));
                }
            }
            match kind {
                "group" => match n.get("children").and_then(Value::as_array) {
                    Some(c) if !c.is_empty() => {
                        c.iter().for_each(|c| walk(c, depth + 1, shapes, out))
                    }
                    _ => out.push("a group needs children".into()),
                },
                "use" => {
                    let r = n.get("ref").and_then(Value::as_str).unwrap_or("");
                    if !OBJECTS.contains(&r) {
                        out.push(format!("unknown object {r}"));
                    }
                }
                "repeat" => match n.get("shape") {
                    Some(shape) => {
                        walk(shape, depth + 1, shapes, out);
                        *shapes += 1;
                    }
                    None => out.push("repeat needs a shape".into()),
                },
                _ => *shapes += 1,
            }
        }
        let trees: Vec<&Value> = match obj.get("stations").and_then(Value::as_array) {
            Some(stations) => {
                if obj.contains_key("ast") {
                    out.push("stations and ast".into());
                }
                let n = stations.len();
                if let Some(l) = obj.get("layout").and_then(Value::as_str) {
                    match LAYOUTS.iter().find(|(k, ..)| *k == l) {
                        Some((_, lo, hi)) if (*lo..=*hi).contains(&n) => {}
                        Some(_) => out.push(format!("{n} stations on {l}")),
                        None => out.push(format!("unknown layout {l}")),
                    }
                }
                if let Some(mode) = scene.pointer("/build/mode").and_then(Value::as_str) {
                    if !BUILD_MODES.contains(&mode) {
                        out.push(format!("unknown build mode {mode}"));
                    }
                }
                let mut trees = Vec::new();
                for (i, st) in stations.iter().enumerate() {
                    let Some(st) = st.as_object() else {
                        out.push(format!("station {i} not an object"));
                        continue;
                    };
                    for k in st.keys() {
                        if !STATION_KEYS.contains(&k.as_str()) {
                            out.push(format!("station {i}: unknown field {k}"));
                        }
                    }
                    match (st.get("mech"), st.get("ast")) {
                        (Some(m), None) => {
                            if !m.as_str().is_some_and(|m| MECHANISMS.contains(&m)) {
                                out.push(format!("station {i}: unknown mech {m}"));
                            }
                        }
                        (None, Some(a)) => trees.push(a),
                        _ => out.push(format!("station {i}: one of mech or ast")),
                    }
                    if let Some(t) = st.get("tone") {
                        if !t.as_str().is_some_and(|t| TONES.contains(&t)) {
                            out.push(format!("station {i}: unknown tone {t}"));
                        }
                    }
                    if let Some(l) = st.get("label") {
                        if l.as_str().is_none_or(|l| l.chars().count() > MAX_LABEL) {
                            out.push(format!("station {i}: bad label"));
                        }
                    }
                    for d in st
                        .get("after")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if !d
                            .as_u64()
                            .is_some_and(|d| (d as usize) < n && d as usize != i)
                        {
                            out.push(format!("station {i}: bad after {d}"));
                        }
                    }
                }
                trees
            }
            None => obj.get("ast").into_iter().collect(),
        };
        for t in trees {
            let mut shapes = 0;
            walk(t, 0, &mut shapes, &mut out);
            if shapes > MAX_SHAPES {
                out.push(format!("{shapes} shapes"));
            }
        }
        out
    }

    /// The v2 contract of the ```scene3d block (`frontend/docs/architecture/3d-general/01-scene-contract.md`):
    /// a free-camera scene graph with inline `mesh` geometry, PBR `material`s, `texture`s, `lights` and
    /// `assets`. A `scene3d` block may still carry a v1 payload (`version: "1"` — migration tier 2), which
    /// delegates to [`iso_scene_problems`]. The exact bounds are `scene3d_contract` (spec §3).
    mod scene3d_contract {
        #[allow(unused_imports)]
        use crate::neo4j::traits::*;
        pub const V1_KEYS: &[&str] = &[
            "layout",
            "stations",
            "angles",
            "home",
            "stepSeconds",
            "focal",
            "build",
            "ast",
            "tone",
            "label",
        ];
        pub const V2_KEYS: &[&str] = &[
            "camera",
            "lights",
            "textures",
            "materials",
            "assets",
            "nodes",
        ];
        pub const META_KEYS: &[&str] = &["version", "title", "caption"];
        pub const VERSIONS: &[&str] = &["1", "2"];

        pub const MAX_TITLE: usize = 120;
        pub const MAX_CAPTION: usize = 300;
        pub const MAX_NODES: usize = 512;
        pub const MAX_DEPTH: usize = 32;
        pub const MAX_ASSETS: usize = 256;
        pub const MAX_MATERIALS: usize = 64;
        pub const MAX_TEXTURES: usize = 32;
        pub const MAX_LIGHTS: usize = 16;
        pub const MAX_MESHES: usize = 256;

        pub const MAX_VERTICES: usize = 65_536;
        pub const MAX_TRIANGLES: usize = 131_072;
        pub const MAX_VERTICES_TOTAL: usize = 1_048_576;
        pub const MAX_TRIANGLES_TOTAL: usize = 2_097_152;

        pub const POS_LO: f64 = -1000.0;
        pub const POS_HI: f64 = 1000.0;
        pub const ROT_LO: f64 = -360.0;
        pub const ROT_HI: f64 = 360.0;
        pub const UV_LO: f64 = -64.0;
        pub const UV_HI: f64 = 64.0;
        pub const SCALE_LO: f64 = 1e-4;
        pub const SCALE_HI: f64 = 1000.0;

        pub const MAX_URL: usize = 2048;
        pub const MAX_DATA_BYTES: usize = 512 * 1024;

        pub const FOV_LO: f64 = 1.0;
        pub const FOV_HI: f64 = 120.0;
        pub const NEAR_HI: f64 = 10_000.0;
        pub const FAR_HI: f64 = 100_000.0;
        pub const INTENSITY_HI: f64 = 100.0;
        pub const DISTANCE_HI: f64 = 10_000.0;
        pub const DECAY_HI: f64 = 4.0;
        pub const EMISSIVE_HI: f64 = 64.0;
        pub const IOR_LO: f64 = 1.0;
        pub const IOR_HI: f64 = 3.0;

        pub const MAX_ID: usize = 64;
        pub const CAMERA_KINDS: &[&str] = &["perspective", "orthographic"];
        pub const TEXTURE_KINDS: &[&str] = &["image", "procedural"];
        pub const PROCEDURAL_TYPES: &[&str] = &["checker", "solid", "gradient"];
        pub const WRAPS: &[&str] = &["repeat", "clamp", "mirror"];
        pub const FILTERS: &[&str] = &["nearest", "linear"];
        pub const LIGHT_KINDS: &[&str] = &["ambient", "directional", "point"];
        pub const WINDINGS: &[&str] = &["ccw", "cw"];
    }

    /// The v2 checks of the ```scene3d block: a free-camera scene graph (`nodes`) with inline `mesh`
    /// geometry, PBR `material`s, `texture`s, `lights` and `assets`, and the exact bounds of the spec §3.
    /// A v1 payload (`version: "1"`, or v1 fields without v2 fields) delegates to [`iso_scene_problems`].
    fn scene3d_problems(scene: &serde_json::Value) -> Vec<String> {
        use scene3d_contract::*;
        use serde_json::Value;
        use std::collections::{HashMap, HashSet};

        fn is_id(s: &str) -> bool {
            !s.is_empty()
                && s.chars().count() <= MAX_ID
                && s.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
                && s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
                && s.chars().last().is_some_and(|c| c.is_ascii_alphanumeric())
        }
        fn num_in(v: &Value, lo: f64, hi: f64) -> bool {
            v.as_f64()
                .is_some_and(|x| x.is_finite() && x >= lo && x <= hi)
        }
        fn int_in(v: &Value, lo: i64, hi: i64) -> bool {
            v.as_i64().is_some_and(|x| x >= lo && x <= hi)
        }
        fn vec3(v: &Value, lo: f64, hi: f64) -> bool {
            v.as_array()
                .is_some_and(|a| a.len() == 3 && a.iter().all(|x| num_in(x, lo, hi)))
        }
        fn is_hex_color(s: &str) -> bool {
            let hex = s.strip_prefix('#').unwrap_or(s);
            matches!(hex.len(), 3 | 4 | 6 | 8) && hex.chars().all(|c| c.is_ascii_hexdigit())
        }
        fn is_color(v: &Value) -> bool {
            match v {
                Value::String(s) => is_hex_color(s),
                Value::Array(a) => {
                    (a.len() == 3 || a.len() == 4) && a.iter().all(|x| num_in(x, 0.0, 1.0))
                }
                _ => false,
            }
        }

        let mut out = Vec::new();
        let Some(obj) = scene.as_object() else {
            return vec!["not an object".into()];
        };

        let version = obj.get("version").and_then(Value::as_str);
        if let Some(v) = version {
            if !VERSIONS.contains(&v) {
                out.push(format!("unknown version {v}"));
                return out;
            }
        }
        let has = |keys: &[&str]| keys.iter().any(|k| obj.contains_key(*k));
        let has_v1 = has(V1_KEYS);
        let has_v2 = has(V2_KEYS);
        if has_v1 && has_v2 {
            out.push("mixed v1 and v2 fields".into());
            return out;
        }
        let is_v1 = match version {
            Some("1") => true,
            Some("2") => false,
            _ if has_v1 => true,
            _ if has_v2 => false,
            _ => return vec!["a scene needs stations/ast or nodes".into()],
        };
        if is_v1 {
            if has_v2 {
                out.push("version 1 with v2 fields".into());
                return out;
            }
            let mut v1 = obj.clone();
            v1.remove("version");
            return iso_scene_problems(&Value::Object(v1));
        }
        if has_v1 {
            out.push("version 2 with v1 fields".into());
            return out;
        }

        // v2 root fields.
        if let Some(t) = obj.get("title") {
            if t.as_str().is_none_or(|s| s.chars().count() > MAX_TITLE) {
                out.push("bad title".into());
            }
        }
        if let Some(c) = obj.get("caption") {
            if c.as_str().is_none_or(|s| s.chars().count() > MAX_CAPTION) {
                out.push("bad caption".into());
            }
        }
        for k in obj.keys() {
            if !V2_KEYS.contains(&k.as_str()) && !META_KEYS.contains(&k.as_str()) {
                out.push(format!("unknown scene field {k}"));
            }
        }

        // textures
        let mut texture_ids: HashSet<String> = HashSet::new();
        if let Some(list) = obj.get("textures").and_then(Value::as_array) {
            if list.len() > MAX_TEXTURES {
                out.push(format!("{} textures (at most {MAX_TEXTURES})", list.len()));
            }
            for (i, t) in list.iter().enumerate() {
                let Some(t) = t.as_object() else {
                    out.push(format!("texture {i} not an object"));
                    continue;
                };
                let Some(id) = t.get("id").and_then(Value::as_str) else {
                    out.push(format!("texture {i} needs an id"));
                    continue;
                };
                if !is_id(id) {
                    out.push(format!("texture {i}: bad id {id}"));
                }
                if !texture_ids.insert(id.to_string()) {
                    out.push(format!("duplicate texture id {id}"));
                }
                let kind = t.get("kind").and_then(Value::as_str).unwrap_or("");
                if !TEXTURE_KINDS.contains(&kind) {
                    out.push(format!("texture {id}: unknown kind {kind}"));
                    continue;
                }
                match kind {
                    "image" => {
                        match t.get("image") {
                            Some(Value::Object(img)) => {
                                if let Some(u) = img.get("url").and_then(Value::as_str) {
                                    if u.len() > MAX_URL {
                                        out.push(format!("texture {id}: url too long"));
                                    }
                                    if !(u.starts_with("https:") || u.starts_with("data:image/")) {
                                        out.push(format!("texture {id}: url scheme must be https: or data:image/"));
                                    }
                                    if u.starts_with("data:")
                                        && u.len().saturating_mul(3) / 4 > MAX_DATA_BYTES
                                    {
                                        out.push(format!("texture {id}: data URI too large"));
                                    }
                                }
                                for (key, allowed) in [
                                    ("wrapS", WRAPS),
                                    ("wrapT", WRAPS),
                                    ("minFilter", FILTERS),
                                    ("magFilter", FILTERS),
                                ] {
                                    if let Some(v) = img.get(key).and_then(Value::as_str) {
                                        if !allowed.contains(&v) {
                                            out.push(format!(
                                                "texture {id}: {key} {v} not in {}",
                                                allowed.join(",")
                                            ));
                                        }
                                    }
                                }
                            }
                            _ => out.push(format!("texture {id}: image must be an object")),
                        }
                    }
                    "procedural" => match t.get("procedural").and_then(Value::as_object) {
                        Some(p) => {
                            let ty = p.get("type").and_then(Value::as_str).unwrap_or("");
                            if !PROCEDURAL_TYPES.contains(&ty) {
                                out.push(format!("texture {id}: unknown procedural type {ty}"));
                            }
                            for key in ["colorA", "colorB"] {
                                if let Some(v) = p.get(key) {
                                    if !v.as_str().is_some_and(is_hex_color) {
                                        out.push(format!(
                                            "texture {id}: {key} must be a hex color"
                                        ));
                                    }
                                }
                            }
                            if let Some(s) = p.get("scale") {
                                if !int_in(s, 1, 64) {
                                    out.push(format!(
                                        "texture {id}: scale must be an integer 1..64"
                                    ));
                                }
                            }
                        }
                        None => out.push(format!("texture {id}: procedural must be an object")),
                    },
                    _ => {}
                }
            }
        }

        // materials
        let mut material_ids: HashSet<String> = HashSet::new();
        if let Some(list) = obj.get("materials").and_then(Value::as_array) {
            if list.len() > MAX_MATERIALS {
                out.push(format!(
                    "{} materials (at most {MAX_MATERIALS})",
                    list.len()
                ));
            }
            for (i, m) in list.iter().enumerate() {
                let Some(m) = m.as_object() else {
                    out.push(format!("material {i} not an object"));
                    continue;
                };
                let Some(id) = m.get("id").and_then(Value::as_str) else {
                    out.push(format!("material {i} needs an id"));
                    continue;
                };
                if !is_id(id) {
                    out.push(format!("material {i}: bad id {id}"));
                }
                if !material_ids.insert(id.to_string()) {
                    out.push(format!("duplicate material id {id}"));
                }
                for key in ["baseColor", "emissive"] {
                    if let Some(v) = m.get(key) {
                        if !is_color(v) {
                            out.push(format!("material {id}: bad {key}"));
                        }
                    }
                }
                for (key, lo, hi) in [
                    ("metallic", 0.0, 1.0),
                    ("roughness", 0.0, 1.0),
                    ("opacity", 0.0, 1.0),
                    ("alphaCutoff", 0.0, 1.0),
                ] {
                    if let Some(v) = m.get(key) {
                        if !num_in(v, lo, hi) {
                            out.push(format!("material {id}: {key} must be {lo}..{hi}"));
                        }
                    }
                }
                if let Some(v) = m.get("emissiveIntensity") {
                    if !num_in(v, 0.0, EMISSIVE_HI) {
                        out.push(format!("material {id}: emissiveIntensity out of range"));
                    }
                }
                if let Some(v) = m.get("ior") {
                    if !num_in(v, IOR_LO, IOR_HI) {
                        out.push(format!("material {id}: ior must be {IOR_LO}..{IOR_HI}"));
                    }
                }
                for key in [
                    "baseColorTexture",
                    "normalMap",
                    "metallicRoughnessTexture",
                    "occlusionTexture",
                ] {
                    if let Some(v) = m.get(key) {
                        if !v.is_null() && !v.as_str().is_some_and(|s| texture_ids.contains(s)) {
                            out.push(format!("material {id}: {key} does not reference a texture"));
                        }
                    }
                }
                if let Some(v) = m.get("doubleSided") {
                    if !v.is_boolean() {
                        out.push(format!("material {id}: doubleSided must be a boolean"));
                    }
                }
            }
        }

        // camera
        if let Some(cam) = obj.get("camera").and_then(Value::as_object) {
            let kind = cam
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("perspective");
            if !CAMERA_KINDS.contains(&kind) {
                out.push(format!("camera: unknown kind {kind}"));
            }
            for key in ["position", "target", "up"] {
                if let Some(v) = cam.get(key) {
                    if !vec3(v, POS_LO, POS_HI) {
                        out.push(format!(
                            "camera: {key} must be [x,y,z] in [{POS_LO},{POS_HI}]"
                        ));
                    }
                }
            }
            if let Some(n) = cam.get("near") {
                if !n.as_f64().is_some_and(|x| x > 0.0 && x <= NEAR_HI) {
                    out.push(format!("camera: near must be in (0, {NEAR_HI}]"));
                }
            }
            if let Some(f) = cam.get("far") {
                if !f.as_f64().is_some_and(|x| x.is_finite() && x <= FAR_HI) {
                    out.push(format!("camera: far must be <= {FAR_HI}"));
                }
            }
            if let (Some(n), Some(f)) = (cam.get("near"), cam.get("far")) {
                if let (Some(n), Some(f)) = (n.as_f64(), f.as_f64()) {
                    if f <= n {
                        out.push("camera: far must be > near".into());
                    }
                }
            }
            if kind == "perspective" {
                if let Some(fov) = cam.get("fov") {
                    if !num_in(fov, FOV_LO, FOV_HI) {
                        out.push(format!("camera: fov must be {FOV_LO}..{FOV_HI}"));
                    }
                }
            } else {
                if let Some(z) = cam.get("zoom") {
                    if !z.as_f64().is_some_and(|x| x > 0.0 && x <= 1000.0) {
                        out.push("camera: zoom must be in (0, 1000]".into());
                    }
                }
                if let (Some(l), Some(r)) = (
                    cam.get("left").and_then(Value::as_f64),
                    cam.get("right").and_then(Value::as_f64),
                ) {
                    if r <= l {
                        out.push("camera: right must be > left".into());
                    }
                }
                if let (Some(b), Some(t)) = (
                    cam.get("bottom").and_then(Value::as_f64),
                    cam.get("top").and_then(Value::as_f64),
                ) {
                    if t <= b {
                        out.push("camera: top must be > bottom".into());
                    }
                }
            }
        } else if obj.contains_key("camera") {
            out.push("camera must be an object".into());
        }

        // lights
        if let Some(list) = obj.get("lights").and_then(Value::as_array) {
            if list.len() > MAX_LIGHTS {
                out.push(format!("{} lights (at most {MAX_LIGHTS})", list.len()));
            }
            for (i, l) in list.iter().enumerate() {
                let Some(l) = l.as_object() else {
                    out.push(format!("light {i} not an object"));
                    continue;
                };
                let kind = l.get("kind").and_then(Value::as_str).unwrap_or("");
                if !LIGHT_KINDS.contains(&kind) {
                    out.push(format!("light {i}: unknown kind {kind}"));
                    continue;
                }
                if let Some(c) = l.get("color") {
                    if !is_color(c) {
                        out.push(format!("light {i}: bad color"));
                    }
                }
                if let Some(v) = l.get("intensity") {
                    if !num_in(v, 0.0, INTENSITY_HI) {
                        out.push(format!("light {i}: intensity out of range"));
                    }
                }
                match kind {
                    "directional" => {
                        if let Some(d) = l.get("direction") {
                            let nonzero = d.as_array().is_some_and(|a| {
                                a.iter().any(|x| x.as_f64().is_some_and(|y| y != 0.0))
                            });
                            if !vec3(d, POS_LO, POS_HI) || !nonzero {
                                out.push(format!(
                                    "light {i}: direction must be a non-zero [x,y,z]"
                                ));
                            }
                        }
                    }
                    "point" => {
                        if let Some(p) = l.get("position") {
                            if !vec3(p, POS_LO, POS_HI) {
                                out.push(format!("light {i}: position out of range"));
                            }
                        }
                        if let Some(d) = l.get("distance") {
                            if !num_in(d, 0.0, DISTANCE_HI) {
                                out.push(format!("light {i}: distance out of range"));
                            }
                        }
                        if let Some(d) = l.get("decay") {
                            if !num_in(d, 0.0, DECAY_HI) {
                                out.push(format!("light {i}: decay out of range"));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // assets
        let mut asset_ids: HashSet<String> = HashSet::new();
        if let Some(list) = obj.get("assets").and_then(Value::as_array) {
            if list.len() > MAX_ASSETS {
                out.push(format!("{} assets (at most {MAX_ASSETS})", list.len()));
            }
            for (i, a) in list.iter().enumerate() {
                let Some(a) = a.as_object() else {
                    out.push(format!("asset {i} not an object"));
                    continue;
                };
                let Some(id) = a.get("id").and_then(Value::as_str) else {
                    out.push(format!("asset {i} needs an id"));
                    continue;
                };
                if !is_id(id) {
                    out.push(format!("asset {i}: bad id {id}"));
                }
                if !asset_ids.insert(id.to_string()) {
                    out.push(format!("duplicate asset id {id}"));
                }
            }
        }

        // nodes
        fn check_mesh(
            mesh: &Value,
            material_ids: &HashSet<String>,
            where_: &str,
            out: &mut Vec<String>,
            verts: &mut usize,
            tris: &mut usize,
        ) {
            let Some(m) = mesh.as_object() else {
                out.push(format!("{where_}: mesh must be an object"));
                return;
            };
            let Some(positions) = m.get("positions").and_then(Value::as_array) else {
                out.push(format!("{where_}: mesh needs positions"));
                return;
            };
            if positions.is_empty() || positions.len() % 3 != 0 {
                out.push(format!(
                    "{where_}: positions length must be a positive multiple of 3"
                ));
                return;
            }
            let vcount = positions.len() / 3;
            if vcount > MAX_VERTICES {
                out.push(format!(
                    "{where_}: {vcount} vertices (at most {MAX_VERTICES})"
                ));
            }
            if positions.iter().any(|x| !num_in(x, POS_LO, POS_HI)) {
                out.push(format!(
                    "{where_}: a position component is outside [{POS_LO},{POS_HI}]"
                ));
            }
            if let Some(n) = m.get("normals").and_then(Value::as_array) {
                if n.len() != positions.len() {
                    out.push(format!(
                        "{where_}: normals length must equal positions length"
                    ));
                } else if n.iter().any(|x| !num_in(x, -1.0, 1.0)) {
                    out.push(format!("{where_}: a normal component is outside [-1,1]"));
                }
            }
            if let Some(u) = m.get("uvs").and_then(Value::as_array) {
                if u.len() != vcount * 2 {
                    out.push(format!("{where_}: uvs length must be 2 × vertex count"));
                } else if u.iter().any(|x| !num_in(x, UV_LO, UV_HI)) {
                    out.push(format!(
                        "{where_}: a uv component is outside [{UV_LO},{UV_HI}]"
                    ));
                }
            }
            let Some(indices) = m.get("indices").and_then(Value::as_array) else {
                out.push(format!("{where_}: mesh needs indices"));
                return;
            };
            if indices.is_empty() || indices.len() % 3 != 0 {
                out.push(format!(
                    "{where_}: indices length must be a positive multiple of 3"
                ));
                return;
            }
            if indices
                .iter()
                .any(|i| i.as_u64().is_none_or(|x| (x as usize) >= vcount))
            {
                out.push(format!("{where_}: an index is >= vertex count"));
            }
            let tcount = indices.len() / 3;
            if tcount > MAX_TRIANGLES {
                out.push(format!(
                    "{where_}: {tcount} triangles (at most {MAX_TRIANGLES})"
                ));
            }
            if let Some(w) = m.get("winding").and_then(Value::as_str) {
                if !WINDINGS.contains(&w) {
                    out.push(format!("{where_}: unknown winding {w}"));
                }
            }
            if let Some(mat) = m.get("material").and_then(Value::as_str) {
                if !material_ids.contains(mat) {
                    out.push(format!("{where_}: material {mat} not declared"));
                }
            }
            *verts += vcount;
            *tris += tcount;
        }

        let mut node_ids: HashSet<String> = HashSet::new();
        let mut parents: HashMap<String, Option<String>> = HashMap::new();
        let mut mesh_count = 0usize;
        let mut vertices_total = 0usize;
        let mut triangles_total = 0usize;

        if let Some(list) = obj.get("nodes").and_then(Value::as_array) {
            if list.len() > MAX_NODES {
                out.push(format!("{} nodes (at most {MAX_NODES})", list.len()));
            }
            for (i, n) in list.iter().enumerate() {
                let where_ = format!("node {i}");
                let Some(n) = n.as_object() else {
                    out.push(format!("{where_} not an object"));
                    continue;
                };
                let Some(id) = n.get("id").and_then(Value::as_str) else {
                    out.push(format!("{where_} needs an id"));
                    continue;
                };
                if !is_id(id) {
                    out.push(format!("{where_}: bad id {id}"));
                }
                if !node_ids.insert(id.to_string()) {
                    out.push(format!("{where_}: duplicate id {id}"));
                }
                parents.insert(
                    id.to_string(),
                    n.get("parent").and_then(Value::as_str).map(str::to_string),
                );

                if let Some(p) = n.get("position") {
                    if !vec3(p, POS_LO, POS_HI) {
                        out.push(format!(
                            "{where_}: position must be [x,y,z] in [{POS_LO},{POS_HI}]"
                        ));
                    }
                }
                if let Some(r) = n.get("rotation") {
                    if !vec3(r, ROT_LO, ROT_HI) {
                        out.push(format!(
                            "{where_}: rotation must be [x,y,z] in [{ROT_LO},{ROT_HI}]"
                        ));
                    }
                }
                if let Some(q) = n.get("quaternion") {
                    let norm = q.as_array().and_then(|a| {
                        if a.len() != 4 {
                            return None;
                        }
                        Some(
                            a.iter()
                                .filter_map(Value::as_f64)
                                .map(|x| x * x)
                                .sum::<f64>(),
                        )
                    });
                    match norm {
                        Some(s) if (s - 1.0).abs() <= 1e-4 => {}
                        _ => out.push(format!(
                            "{where_}: quaternion must be a normalized [x,y,z,w]"
                        )),
                    }
                }
                if n.contains_key("rotation") && n.contains_key("quaternion") {
                    out.push(format!("{where_}: rotation and quaternion are exclusive"));
                }
                if let Some(s) = n.get("scale") {
                    let ok = match s {
                        Value::Number(_) => num_in(s, SCALE_LO, SCALE_HI),
                        Value::Array(a) => {
                            a.len() == 3 && a.iter().all(|x| num_in(x, SCALE_LO, SCALE_HI))
                        }
                        _ => false,
                    };
                    if !ok {
                        out.push(format!(
                            "{where_}: scale must be a scalar or [x,y,z] in [{SCALE_LO},{SCALE_HI}]"
                        ));
                    }
                }

                let content = [
                    n.contains_key("primitive"),
                    n.contains_key("mesh"),
                    n.contains_key("instance"),
                ]
                .iter()
                .filter(|b| **b)
                .count();
                if content != 1 {
                    out.push(format!(
                        "{where_}: exactly one of primitive, mesh, instance"
                    ));
                }
                if let Some(m) = n.get("mesh") {
                    mesh_count += 1;
                    check_mesh(
                        m,
                        &material_ids,
                        &where_,
                        &mut out,
                        &mut vertices_total,
                        &mut triangles_total,
                    );
                }
                if let Some(inst) = n.get("instance").and_then(Value::as_object) {
                    let asset = inst.get("asset").and_then(Value::as_str).unwrap_or("");
                    let known = asset_ids.contains(asset)
                        || iso_contract::OBJECTS.contains(&asset)
                        || (asset.starts_with("mech:")
                            && iso_contract::MECHANISMS.contains(&&asset["mech:".len()..]));
                    if !known {
                        out.push(format!("{where_}: unknown asset {asset}"));
                    }
                    if let Some(mat) = inst.get("material").and_then(Value::as_str) {
                        if !material_ids.contains(mat) {
                            out.push(format!("{where_}: material {mat} not declared"));
                        }
                    }
                }
                if let Some(p) = n.get("primitive") {
                    if let Some(kind) = p.get("kind").and_then(Value::as_str) {
                        if !iso_contract::KINDS.contains(&kind) {
                            out.push(format!("{where_}: unknown primitive kind {kind}"));
                        }
                    } else {
                        out.push(format!("{where_}: primitive needs a kind"));
                    }
                }
                if let Some(mat) = n.get("material").and_then(Value::as_str) {
                    if !material_ids.contains(mat) {
                        out.push(format!("{where_}: material {mat} not declared"));
                    }
                }
                for (key, ok) in [
                    ("visible", n.get("visible").is_none_or(Value::is_boolean)),
                    (
                        "castsShadow",
                        n.get("castsShadow").is_none_or(Value::is_boolean),
                    ),
                    (
                        "receivesShadow",
                        n.get("receivesShadow").is_none_or(Value::is_boolean),
                    ),
                ] {
                    if !ok {
                        out.push(format!("{where_}: {key} must be a boolean"));
                    }
                }
            }
        }

        // resolve parents + acyclicity + depth
        for id in node_ids.iter() {
            let mut path: HashSet<&str> = HashSet::new();
            let mut cur: &str = id.as_str();
            let mut depth = 0usize;
            loop {
                if !path.insert(cur) {
                    out.push(format!("node {id}: parent cycle"));
                    break;
                }
                let Some(Some(p)) = parents.get(cur) else {
                    break;
                };
                if !node_ids.contains(p) {
                    out.push(format!("node {id}: parent {p} not declared"));
                    break;
                }
                cur = p.as_str();
                depth += 1;
                if depth > MAX_DEPTH {
                    out.push(format!("node {id}: graph too deep"));
                    break;
                }
            }
        }

        if mesh_count > MAX_MESHES {
            out.push(format!("{mesh_count} meshes (at most {MAX_MESHES})"));
        }
        if vertices_total > MAX_VERTICES_TOTAL {
            out.push(format!(
                "{vertices_total} vertices total (at most {MAX_VERTICES_TOTAL})"
            ));
        }
        if triangles_total > MAX_TRIANGLES_TOTAL {
            out.push(format!(
                "{triangles_total} triangles total (at most {MAX_TRIANGLES_TOTAL})"
            ));
        }

        out
    }

    #[test]
    fn scene3d_accepts_the_v2_example_and_refuses_a_bad_mesh() {
        // The §5 example of 01-scene-contract.md: free camera + inline meshes + PBR + texture.
        let example: serde_json::Value = serde_json::from_str(
            r##"{
              "version": "2",
              "title": "Plan métallique texturé",
              "camera": { "kind": "perspective", "position": [3.2, -2.6, 2.4], "target": [0, 0, 0.4], "fov": 42, "near": 0.1, "far": 100 },
              "lights": [
                { "kind": "directional", "color": "#ffffff", "intensity": 3, "direction": [0.6, -1, 0.8] },
                { "kind": "ambient", "color": "#3a4055", "intensity": 0.4 }
              ],
              "textures": [
                { "id": "checker", "kind": "procedural", "procedural": { "type": "checker", "colorA": "#dfe3ee", "colorB": "#1b1e2b", "scale": 8 } }
              ],
              "materials": [
                { "id": "metal", "baseColor": "#c89a4a", "metallic": 0.9, "roughness": 0.35 },
                { "id": "sol", "baseColor": "#ffffff", "baseColorTexture": "checker", "metallic": 0, "roughness": 0.8 }
              ],
              "nodes": [
                { "id": "dalle", "mesh": { "positions": [-2,0,-2, 2,0,-2, 2,0,2, -2,0,2], "uvs": [0,0, 1,0, 1,1, 0,1], "indices": [0,1,2, 0,2,3] }, "material": "sol" },
                { "id": "cube", "position": [0,0,0.6], "mesh": { "positions": [-0.5,-0.5,-0.5, 0.5,-0.5,-0.5, 0.5,0.5,-0.5, -0.5,0.5,-0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5, 0.5,0.5,0.5, -0.5,0.5,0.5], "uvs": [0,0, 1,0, 1,1, 0,1, 0,0, 1,0, 1,1, 0,1], "indices": [4,5,6, 4,6,7, 1,0,3, 1,3,2, 5,1,2, 5,2,6, 0,4,7, 0,7,3, 7,6,2, 7,2,3, 0,1,5, 0,5,4] }, "material": "metal" }
              ]
            }"##,
        )
        .unwrap();
        assert_eq!(scene3d_problems(&example), Vec::<String>::new());

        // A malformed mesh is refused: positions length not a multiple of 3.
        let bad: serde_json::Value = serde_json::from_str(
            r#"{"version":"2","nodes":[{"id":"m","mesh":{"positions":[0,0,0,1,1],"indices":[0,1,2]}}]}"#,
        )
        .unwrap();
        assert!(
            !scene3d_problems(&bad).is_empty(),
            "5 positions should be refused"
        );

        // An index beyond the vertex count is refused.
        let bad_idx: serde_json::Value = serde_json::from_str(
            r#"{"version":"2","nodes":[{"id":"m","mesh":{"positions":[0,0,0, 1,0,0, 0,1,0],"indices":[0,1,9]}}]}"#,
        )
        .unwrap();
        assert!(
            !scene3d_problems(&bad_idx).is_empty(),
            "index >= vertex count should be refused"
        );

        // A v1 payload is still accepted through the same entry point (Tier 2 migration).
        let v1: serde_json::Value = serde_json::from_str(
            r#"{"version":"1","title":"Release","stations":[{"mech":"drawers","tone":"green","label":"Plan"}],"build":{"mode":"sequential"}}"#,
        )
        .unwrap();
        assert_eq!(scene3d_problems(&v1), Vec::<String>::new());
    }

    // ── scene3d v2: every refusal branch of the checker ───────────────────
    // One malformed scene per family, each breaking many rules at once; the
    // assertions check that the branch emitted (substring), not its exact wording.

    fn s3d(v: &serde_json::Value) -> Vec<String> {
        scene3d_problems(v)
    }
    fn has(p: &[String], needle: &str) -> bool {
        p.iter().any(|x| x.contains(needle))
    }
    fn s3d_str(json: &str) -> Vec<String> {
        s3d(&serde_json::from_str(json).unwrap())
    }

    #[test]
    fn scene3d_refuses_bad_roots_and_versions() {
        assert!(has(&s3d(&serde_json::json!(42)), "not an object"));
        assert!(has(&s3d_str(r#"{"version":"9"}"#), "unknown version 9"));
        assert!(has(
            &s3d_str(r#"{"stations":[],"nodes":[]}"#),
            "mixed v1 and v2 fields"
        ));
        assert!(has(
            &s3d_str(r#"{"title":"x"}"#),
            "a scene needs stations/ast or nodes"
        ));
        assert!(has(
            &s3d_str(r#"{"version":"1","nodes":[]}"#),
            "version 1 with v2 fields"
        ));
        assert!(has(
            &s3d_str(r#"{"version":"2","stations":[]}"#),
            "version 2 with v1 fields"
        ));
        let long_caption = "c".repeat(301);
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [], "title": 123, "caption": long_caption, "bogus": 1
        }));
        assert!(has(&p, "bad title"), "{p:?}");
        assert!(has(&p, "bad caption"), "{p:?}");
        assert!(has(&p, "unknown scene field bogus"), "{p:?}");
    }

    #[test]
    fn scene3d_refuses_bad_textures() {
        let long_url = format!("http://{}", "a".repeat(2100));
        let big_data = format!("data:image/png;base64,{}", "A".repeat(700_000));
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "textures": [
                1,
                {"kind": "image"},
                {"id": "Bad_ID", "kind": "image", "image": {}},
                {"id": "dup", "kind": "procedural", "procedural": {"type": "solid"}},
                {"id": "dup", "kind": "procedural", "procedural": {"type": "solid"}},
                {"id": "k", "kind": "video"},
                {"id": "i1", "kind": "image", "image": "x"},
                {"id": "i2", "kind": "image", "image": {"url": long_url, "wrapS": "bogus", "minFilter": "cubic"}},
                {"id": "i3", "kind": "image", "image": {"url": big_data}},
                {"id": "p1", "kind": "procedural", "procedural": "x"},
                {"id": "p2", "kind": "procedural", "procedural": {"type": "noise", "colorA": "red", "scale": 0}}
            ]
        }));
        for needle in [
            "texture 0 not an object",
            "texture 1 needs an id",
            "bad id Bad_ID",
            "duplicate texture id dup",
            "unknown kind video",
            "image must be an object",
            "url too long",
            "url scheme must be https: or data:image/",
            "wrapS bogus not in",
            "minFilter cubic not in",
            "data URI too large",
            "procedural must be an object",
            "unknown procedural type noise",
            "colorA must be a hex color",
            "scale must be an integer 1..64",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        let many: Vec<serde_json::Value> = (0..33)
            .map(|i| serde_json::json!({"id": format!("t{i}"), "kind": "procedural", "procedural": {"type": "solid"}}))
            .collect();
        let p = s3d(&serde_json::json!({"version": "2", "nodes": [], "textures": many}));
        assert!(has(&p, "33 textures (at most 32)"), "{p:?}");
    }

    #[test]
    fn scene3d_refuses_bad_materials() {
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "materials": [
                1,
                {"metallic": 0.5},
                {"id": "Bad_ID"},
                {"id": "dup"}, {"id": "dup"},
                {"id": "m", "baseColor": "purple", "emissive": [1, 1, 1, 1, 1],
                 "metallic": 2, "roughness": -1, "opacity": 5, "alphaCutoff": 2,
                 "emissiveIntensity": 100, "ior": 5, "baseColorTexture": "nope",
                 "doubleSided": "yes"}
            ]
        }));
        for needle in [
            "material 0 not an object",
            "material 1 needs an id",
            "bad id Bad_ID",
            "duplicate material id dup",
            "bad baseColor",
            "bad emissive",
            "metallic must be 0..1",
            "roughness must be 0..1",
            "opacity must be 0..1",
            "alphaCutoff must be 0..1",
            "emissiveIntensity out of range",
            "ior must be 1..3",
            "baseColorTexture does not reference a texture",
            "doubleSided must be a boolean",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        let many: Vec<serde_json::Value> = (0..65)
            .map(|i| serde_json::json!({"id": format!("m{i}")}))
            .collect();
        let p = s3d(&serde_json::json!({"version": "2", "nodes": [], "materials": many}));
        assert!(has(&p, "65 materials (at most 64)"), "{p:?}");
    }

    #[test]
    fn scene3d_refuses_bad_cameras() {
        assert!(has(
            &s3d_str(r#"{"version":"2","nodes":[],"camera":5}"#),
            "camera must be an object"
        ));
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "camera": {"kind": "fisheye", "position": [1, 2], "near": 0, "far": 1e9, "fov": 500}
        }));
        for needle in [
            "camera: unknown kind fisheye",
            "camera: position must be [x,y,z]",
            "camera: near must be in (0,",
            "camera: far must be <=",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "camera": {"kind": "perspective", "near": 10, "far": 5, "fov": 500}
        }));
        assert!(has(&p, "camera: far must be > near"), "{p:?}");
        assert!(has(&p, "camera: fov must be"), "{p:?}");
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "camera": {"kind": "orthographic", "zoom": 0, "left": 5, "right": 1, "bottom": 5, "top": 1}
        }));
        for needle in [
            "camera: zoom must be in (0, 1000]",
            "camera: right must be > left",
            "camera: top must be > bottom",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
    }

    #[test]
    fn scene3d_refuses_bad_lights_and_assets() {
        let p = s3d(&serde_json::json!({
            "version": "2", "nodes": [],
            "lights": [
                1,
                {"kind": "spot"},
                {"kind": "ambient", "color": "nope", "intensity": 1000},
                {"kind": "directional", "direction": [0, 0, 0]},
                {"kind": "point", "position": [5000, 0, 0], "distance": 1e9, "decay": 10}
            ],
            "assets": [1, {"kind": "x"}, {"id": "Bad_ID"}, {"id": "dup"}, {"id": "dup"}]
        }));
        for needle in [
            "light 0 not an object",
            "light 1: unknown kind spot",
            "light 2: bad color",
            "light 2: intensity out of range",
            "light 3: direction must be a non-zero [x,y,z]",
            "light 4: position out of range",
            "light 4: distance out of range",
            "light 4: decay out of range",
            "asset 0 not an object",
            "asset 1 needs an id",
            "asset 2: bad id Bad_ID",
            "duplicate asset id dup",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        let lights: Vec<serde_json::Value> = (0..17)
            .map(|_| serde_json::json!({"kind": "ambient"}))
            .collect();
        let assets: Vec<serde_json::Value> = (0..257)
            .map(|i| serde_json::json!({"id": format!("a{i}")}))
            .collect();
        let p = s3d(
            &serde_json::json!({"version": "2", "nodes": [], "lights": lights, "assets": assets}),
        );
        assert!(has(&p, "17 lights (at most 16)"), "{p:?}");
        assert!(has(&p, "257 assets (at most 256)"), "{p:?}");
    }

    #[test]
    fn scene3d_refuses_bad_meshes() {
        let tri = |extra: serde_json::Value| {
            let mut m = serde_json::json!({"positions": [0,0,0, 1,0,0, 0,1,0], "indices": [0,1,2]});
            if let (Some(a), Some(b)) = (m.as_object_mut(), extra.as_object()) {
                for (k, v) in b {
                    a.insert(k.clone(), v.clone());
                }
            }
            m
        };
        let p = s3d(&serde_json::json!({
            "version": "2",
            "nodes": [
                {"id": "a", "mesh": "x"},
                {"id": "b", "mesh": {}},
                {"id": "c", "mesh": {"positions": []}},
                {"id": "d", "mesh": {"positions": [5000,0,0, 1,0,0, 0,1,0], "indices": [0,1,2]}},
                {"id": "e", "mesh": tri(serde_json::json!({"normals": [0, 0]}))},
                {"id": "f", "mesh": tri(serde_json::json!({"normals": [2,0,0, 0,0,0, 0,0,0]}))},
                {"id": "g", "mesh": tri(serde_json::json!({"uvs": [0, 0]}))},
                {"id": "h", "mesh": tri(serde_json::json!({"uvs": [100,0, 0,0, 0,0]}))},
                {"id": "i", "mesh": {"positions": [0,0,0, 1,0,0, 0,1,0]}},
                {"id": "j", "mesh": {"positions": [0,0,0, 1,0,0, 0,1,0], "indices": [0, 1]}},
                {"id": "k", "mesh": tri(serde_json::json!({"winding": "zigzag", "material": "ghost"}))}
            ]
        }));
        for needle in [
            "node 0: mesh must be an object",
            "node 1: mesh needs positions",
            "node 2: positions length must be a positive multiple of 3",
            "node 3: a position component is outside",
            "node 4: normals length must equal positions length",
            "node 5: a normal component is outside [-1,1]",
            "node 6: uvs length must be 2 × vertex count",
            "node 7: a uv component is outside",
            "node 8: mesh needs indices",
            "node 9: indices length must be a positive multiple of 3",
            "node 10: unknown winding zigzag",
            "node 10: material ghost not declared",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        // Per-mesh vertex cap: 65 537 vertices.
        let positions = vec![0.0f64; 65_537 * 3];
        let p = s3d(&serde_json::json!({
            "version": "2",
            "nodes": [{"id": "big", "mesh": {"positions": positions, "indices": [0, 1, 2]}}]
        }));
        assert!(has(&p, "vertices (at most 65536)"), "{p:?}");
    }

    #[test]
    fn scene3d_refuses_bad_nodes_and_parents() {
        let p = s3d(&serde_json::json!({
            "version": "2",
            "nodes": [
                1,
                {"mesh": {}},
                {"id": "Bad_ID", "primitive": {"kind": "group"}},
                {"id": "dup", "primitive": {"kind": "group"}},
                {"id": "dup", "primitive": {"kind": "group"}},
                {"id": "t", "primitive": {"kind": "group"}, "position": [1], "rotation": [400, 0, 0],
                 "quaternion": [1, 1, 1, 1], "scale": 0,
                 "visible": "yes", "castsShadow": 1, "receivesShadow": "no", "material": "ghost"},
                {"id": "none"},
                {"id": "both", "mesh": {"positions": [0,0,0, 1,0,0, 0,1,0], "indices": [0,1,2]}, "primitive": {"kind": "group"}},
                {"id": "inst", "instance": {"asset": "nope", "material": "ghost"}},
                {"id": "prim1", "primitive": {}},
                {"id": "prim2", "primitive": {"kind": "blob"}},
                {"id": "orphan", "primitive": {"kind": "group"}, "parent": "nope"},
                {"id": "x", "primitive": {"kind": "group"}, "parent": "y"},
                {"id": "y", "primitive": {"kind": "group"}, "parent": "x"}
            ]
        }));
        for needle in [
            "node 0 not an object",
            "node 1 needs an id",
            "node 2: bad id Bad_ID",
            "node 4: duplicate id dup",
            "node 5: position must be [x,y,z]",
            "node 5: rotation must be [x,y,z]",
            "node 5: quaternion must be a normalized [x,y,z,w]",
            "node 5: rotation and quaternion are exclusive",
            "node 5: scale must be a scalar or [x,y,z]",
            "node 5: visible must be a boolean",
            "node 5: castsShadow must be a boolean",
            "node 5: receivesShadow must be a boolean",
            "node 5: material ghost not declared",
            "node 6: exactly one of primitive, mesh, instance",
            "node 7: exactly one of primitive, mesh, instance",
            "node 8: unknown asset nope",
            "node 8: material ghost not declared",
            "node 9: primitive needs a kind",
            "node 10: unknown primitive kind blob",
            "node orphan: parent nope not declared",
            "parent cycle",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:?}");
        }
        // Too many nodes, too many meshes, and a graph too deep.
        let mut nodes: Vec<serde_json::Value> = (0..513)
            .map(|i| {
                serde_json::json!({"id": format!("n{i}"),
                    "mesh": {"positions": [0,0,0, 1,0,0, 0,1,0], "indices": [0,1,2]}})
            })
            .collect();
        for i in 1..35 {
            nodes.push(
                serde_json::json!({"id": format!("d{i}"), "primitive": {"kind": "group"},
                "parent": format!("d{}", i - 1)}),
            );
        }
        nodes.push(serde_json::json!({"id": "d0", "primitive": {"kind": "group"}}));
        let p = s3d(&serde_json::json!({"version": "2", "nodes": nodes}));
        assert!(has(&p, "nodes (at most 512)"), "{p:?}");
        assert!(has(&p, "meshes (at most 256)"), "{p:?}");
        assert!(has(&p, "graph too deep"), "{p:?}");
    }

    #[test]
    fn the_iso_scene_section_matches_the_frontend_contract() {
        use iso_contract::*;
        let section = SECTION_ISO_SCENE;
        // The example: the one ```iso-scene block of the section, accepted.
        let body = section
            .split_once("```iso-scene\n")
            .and_then(|(_, rest)| rest.split_once("\n```"))
            .map(|(json, _)| json)
            .expect("the section holds an ```iso-scene example");
        let example: serde_json::Value = serde_json::from_str(body).expect("valid JSON");
        assert_eq!(iso_scene_problems(&example), Vec::<String>::new());
        // The checker is not a rubber stamp: what the frontend refuses, it refuses.
        for bad in [
            r#"{"stations":[{"mech":"teleporter"},{"mech":"drawers"}]}"#,
            r#"{"stations":[{"mech":"drawers","script":"alert(1)"}]}"#,
            r#"{"code":"x","stations":[]}"#,
            r#"{"layout":"spiral","stations":[{"mech":"drawers"}]}"#,
            r#"{"layout":"solo","stations":[{"mech":"drawers"},{"mech":"drawers"}]}"#,
            r##"{"stations":[{"mech":"drawers","tone":"#ff00ff"}]}"##,
            r#"{"ast":{"kind":"use","ref":"spaceship"}}"#,
            r#"{"stations":[{"mech":"drawers","ast":{"kind":"use","ref":"flag"}}]}"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(bad).unwrap();
            assert!(
                !iso_scene_problems(&v).is_empty(),
                "{bad} should be refused"
            );
        }
        // Every name the frontend accepts and the section teaches is spelled as the
        // frontend spells it.
        for name in MECHANISMS
            .iter()
            .chain(TONES)
            .chain(KINDS)
            .chain(OBJECTS)
            .chain(BUILD_MODES)
        {
            assert!(section.contains(name), "the section names {name}");
        }
        for (layout, ..) in LAYOUTS
            .iter()
            .filter(|(l, ..)| ["arch", "ring", "wings"].contains(l))
        {
            assert!(section.contains(layout), "the section names {layout}");
        }
        for key in ["title", "caption", "layout", "stations", "build"] {
            assert!(SCENE_KEYS.contains(&key) && section.contains(key), "{key}");
        }
        for key in STATION_KEYS {
            assert!(
                section.contains(key),
                "the section names the station field {key}"
            );
        }
        assert!(section.contains(&format!("{MAX_LABEL} chars")));
        assert!(section.contains(&format!("{MAX_DEPTH} levels")));
        assert!(section.contains(&format!("{MAX_SHAPES} shapes")));
        // Concise: about 410 tokens (cl100k), well under 2 000 characters.
        assert!(section.len() < 1_700, "{} chars", section.len());
    }

    #[test]
    fn repeat_is_accepted_and_counts_its_shape_once() {
        // repeat + a 2-leaf shape = accepted; the shape is validated once, not per position.
        let scene: serde_json::Value = serde_json::from_str(
            r#"{"ast":{"kind":"group","children":[{"kind":"repeat","shape":{"kind":"group","children":[{"kind":"slab","size":[1,1,1]},{"kind":"puck","r":0.5,"h":1}]},"positions":[[0,0,0],[1,0,0]]}]}}"#,
        )
        .unwrap();
        assert_eq!(iso_scene_problems(&scene), Vec::<String>::new());
        // repeat without a shape is refused.
        let bad: serde_json::Value =
            serde_json::from_str(r#"{"ast":{"kind":"repeat","positions":[[0,0,0]]}}"#).unwrap();
        assert!(!iso_scene_problems(&bad).is_empty(), "repeat needs a shape");
    }

    #[test]
    fn test_section_content_non_empty() {
        let sections = all_base_sections();
        for s in &sections {
            if s.id != PromptSectionId::ToolReference {
                assert!(
                    !s.content.is_empty(),
                    "Section {:?} has empty content",
                    s.id
                );
                assert!(
                    s.content.len() > 50,
                    "Section {:?} content too short ({} chars)",
                    s.id,
                    s.content.len()
                );
            }
        }
    }

    #[test]
    fn test_assemble_produces_valid_markdown() {
        let ctx = ComposerContext::level0_full();
        let sections = select_sections(&ctx);
        let assembled = assemble_sections(&sections);

        assert!(
            assembled.contains("# Development Agent"),
            "Should start with title"
        );
        assert!(
            assembled.contains("## 1. Identity & Role"),
            "Should contain Identity section"
        );
        assert!(
            assembled.contains("## 2. Mega-tools"),
            "Should contain Megatools section"
        );
        assert!(
            assembled.contains("## 3. Data Model"),
            "Should contain Data Model section"
        );
        assert!(
            assembled.contains("## 4. Git Workflow"),
            "Should contain Git section"
        );
    }

    #[test]
    fn test_display_section_id() {
        assert_eq!(
            format!("{}", PromptSectionId::IdentityRole),
            "identity_role"
        );
        assert_eq!(
            format!("{}", PromptSectionId::ToolReference),
            "tool_reference"
        );
    }

    #[test]
    fn test_scaffolding_levels_progressive_reduction() {
        let section_counts: Vec<usize> = (0..=4)
            .map(|level| {
                let ctx = ComposerContext {
                    scaffolding_level: level,
                    has_active_plan: true,
                    has_active_protocol: true,
                    task_count: 10,
                };
                select_sections(&ctx).len()
            })
            .collect();

        // Each level should have <= sections than the previous
        for i in 1..section_counts.len() {
            assert!(
                section_counts[i] <= section_counts[i - 1],
                "L{} ({} sections) should have <= L{} ({} sections)",
                i,
                section_counts[i],
                i - 1,
                section_counts[i - 1]
            );
        }
    }

    // ── Tool Reference Group tests ───────────────────────────────────

    #[test]
    fn test_tool_groups_cover_every_mega_tool() {
        let mut all_tools: Vec<&str> = ToolRefGroupId::ALL
            .iter()
            .flat_map(|g| g.tool_names().iter().copied())
            .collect();
        all_tools.sort();
        all_tools.dedup();
        // Must cover all 32 mega-tools (see `mcp::tools::all_tools`)
        assert!(
            all_tools.len() >= 32,
            "Groups should cover at least 32 tools, got {}",
            all_tools.len()
        );
    }

    #[test]
    fn test_tool_groups_no_overlap() {
        let mut seen = std::collections::HashSet::new();
        for group in ToolRefGroupId::ALL {
            for tool in group.tool_names() {
                assert!(
                    seen.insert(*tool),
                    "Tool '{}' appears in multiple groups",
                    tool
                );
            }
        }
    }

    #[test]
    fn test_select_tool_groups_l0_all() {
        let ctx = ToolGroupSelectionContext::default(); // L0
        let groups = select_tool_groups(&ctx);
        assert_eq!(
            groups.len(),
            7,
            "L0 should include all 7 groups, got {}",
            groups.len()
        );
    }

    #[test]
    fn test_select_tool_groups_l4_minimal() {
        let ctx = ToolGroupSelectionContext {
            scaffolding_level: 4,
            ..Default::default()
        };
        let groups = select_tool_groups(&ctx);
        // At L4 without intent: Core + Knowledge only
        assert_eq!(
            groups.len(),
            2,
            "L4 without intent should have 2 groups, got {}",
            groups.len()
        );
        assert!(groups.contains(&ToolRefGroupId::Core));
        assert!(groups.contains(&ToolRefGroupId::Knowledge));
    }

    #[test]
    fn test_select_tool_groups_code_intent() {
        let ctx = ToolGroupSelectionContext {
            scaffolding_level: 4,
            user_intent_keywords: vec!["cherche la function X".to_string()],
            ..Default::default()
        };
        let groups = select_tool_groups(&ctx);
        assert!(
            groups.contains(&ToolRefGroupId::CodeExploration),
            "Code intent should include CodeExploration"
        );
    }

    #[test]
    fn test_select_tool_groups_fsm_whitelist() {
        let ctx = ToolGroupSelectionContext {
            scaffolding_level: 0,
            fsm_available_tools: vec!["code".to_string(), "note".to_string()],
            ..Default::default()
        };
        let groups = select_tool_groups(&ctx);
        // Core + Knowledge always, plus CodeExploration (has "code") and Knowledge (has "note")
        assert!(
            groups.contains(&ToolRefGroupId::Core),
            "Core always included"
        );
        assert!(
            groups.contains(&ToolRefGroupId::Knowledge),
            "Knowledge always included"
        );
        assert!(
            groups.contains(&ToolRefGroupId::CodeExploration),
            "CodeExploration has 'code'"
        );
        assert!(
            !groups.contains(&ToolRefGroupId::Workspace),
            "Workspace not in whitelist"
        );
    }

    #[test]
    fn test_extract_tool_reference_core_only() {
        use crate::chat::prompt::TOOL_REFERENCE;
        let groups = vec![ToolRefGroupId::Core];
        let extracted = extract_tool_reference(TOOL_REFERENCE, &groups);

        assert!(extracted.contains("## project"), "Should contain project");
        assert!(extracted.contains("## plan"), "Should contain plan");
        assert!(extracted.contains("## task"), "Should contain task");
        assert!(extracted.contains("## step"), "Should contain step");
        assert!(!extracted.contains("## code\n"), "Should NOT contain code");
        assert!(!extracted.contains("## admin"), "Should NOT contain admin");
    }

    #[test]
    fn test_extract_tool_reference_all_groups() {
        use crate::chat::prompt::TOOL_REFERENCE;
        let groups = ToolRefGroupId::ALL.to_vec();
        let extracted = extract_tool_reference(TOOL_REFERENCE, &groups);

        // Should contain all major tools
        assert!(extracted.contains("## project"), "Should contain project");
        assert!(extracted.contains("## code"), "Should contain code");
        assert!(extracted.contains("## admin"), "Should contain admin");
        assert!(extracted.contains("## protocol"), "Should contain protocol");
    }

    #[test]
    fn test_extract_tool_reference_smaller_than_full() {
        use crate::chat::prompt::TOOL_REFERENCE;

        let core_only = extract_tool_reference(
            TOOL_REFERENCE,
            &[ToolRefGroupId::Core, ToolRefGroupId::Knowledge],
        );
        let full = extract_tool_reference(TOOL_REFERENCE, ToolRefGroupId::ALL);

        assert!(
            core_only.len() < full.len(),
            "Core-only ({}) should be smaller than full ({})",
            core_only.len(),
            full.len()
        );
        // Significant reduction expected
        let ratio = core_only.len() as f64 / full.len() as f64;
        assert!(
            ratio < 0.6,
            "Core+Knowledge should be <60% of full reference, got {:.1}%",
            ratio * 100.0
        );
    }

    // ── Compact variant tests ───────────────────────────────────────

    #[test]
    fn test_compact_variants_size_under_60_percent() {
        let advanced_full = SECTION_BP_ADVANCED.len();
        let advanced_compact = SECTION_BP_ADVANCED_COMPACT.len();
        let personas_full = SECTION_BP_PERSONAS.len();
        let personas_compact = SECTION_BP_PERSONAS_COMPACT.len();

        let adv_ratio = advanced_compact as f64 / advanced_full as f64;
        let pers_ratio = personas_compact as f64 / personas_full as f64;

        assert!(
            adv_ratio <= 0.60,
            "SECTION_BP_ADVANCED_COMPACT should be ≤60% of original, got {:.1}%",
            adv_ratio * 100.0
        );
        assert!(
            pers_ratio <= 0.60,
            "SECTION_BP_PERSONAS_COMPACT should be ≤60% of original, got {:.1}%",
            pers_ratio * 100.0
        );
    }

    #[test]
    fn test_l0_uses_full_content() {
        let ctx = ComposerContext {
            scaffolding_level: 0,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };
        let sections = select_sections(&ctx);

        let advanced = sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesAdvanced)
            .expect("BestPracticesAdvanced should be present at L0");
        assert_eq!(
            advanced.content, SECTION_BP_ADVANCED,
            "L0 should use full BestPracticesAdvanced content"
        );

        let personas = sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesPersonas)
            .expect("BestPracticesPersonas should be present at L0");
        assert_eq!(
            personas.content, SECTION_BP_PERSONAS,
            "L0 should use full BestPracticesPersonas content"
        );
    }

    #[test]
    fn test_l1_uses_compact_variants() {
        let ctx = ComposerContext {
            scaffolding_level: 1,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };
        let sections = select_sections(&ctx);

        let advanced = sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesAdvanced)
            .expect("BestPracticesAdvanced should be present at L1");
        assert_eq!(
            advanced.content, SECTION_BP_ADVANCED_COMPACT,
            "L1 should use compact BestPracticesAdvanced content"
        );

        let personas = sections
            .iter()
            .find(|s| s.id == PromptSectionId::BestPracticesPersonas)
            .expect("BestPracticesPersonas should be present at L1");
        assert_eq!(
            personas.content, SECTION_BP_PERSONAS_COMPACT,
            "L1 should use compact BestPracticesPersonas content"
        );
    }

    #[test]
    fn test_l2_excludes_advanced_and_personas() {
        let ctx = ComposerContext {
            scaffolding_level: 2,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };
        let sections = select_sections(&ctx);
        let ids: Vec<_> = sections.iter().map(|s| s.id).collect();

        assert!(
            !ids.contains(&PromptSectionId::BestPracticesAdvanced),
            "BestPracticesAdvanced should be absent at L2"
        );
        assert!(
            !ids.contains(&PromptSectionId::BestPracticesPersonas),
            "BestPracticesPersonas should be absent at L2"
        );
    }

    #[test]
    fn test_l0_l1_differentiated_content_size() {
        let ctx_l0 = ComposerContext {
            scaffolding_level: 0,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };
        let ctx_l1 = ComposerContext {
            scaffolding_level: 1,
            has_active_plan: true,
            has_active_protocol: true,
            task_count: 10,
        };

        let sections_l0 = select_sections(&ctx_l0);
        let sections_l1 = select_sections(&ctx_l1);

        // Same number of sections (both include the same sections)
        assert_eq!(sections_l0.len(), sections_l1.len());

        // But L1 total content is smaller due to compact variants
        let total_l0: usize = sections_l0.iter().map(|s| s.content.len()).sum();
        let total_l1: usize = sections_l1.iter().map(|s| s.content.len()).sum();
        assert!(
            total_l1 < total_l0,
            "L1 total content ({}) should be smaller than L0 ({})",
            total_l1,
            total_l0
        );

        // The difference should be significant (~1400 tokens = ~5600 chars)
        let diff = total_l0 - total_l1;
        assert!(
            diff > 4000,
            "L0→L1 content reduction should be >4000 chars, got {}",
            diff
        );
    }

    #[test]
    fn test_use_compact_method() {
        let cond = ActivationCondition::scaffolding_range_with_compact(0, 1, 1, 1);
        assert!(!cond.use_compact(0), "L0 should NOT use compact");
        assert!(cond.use_compact(1), "L1 SHOULD use compact");

        let no_compact = ActivationCondition::scaffolding_range(0, 1);
        assert!(!no_compact.use_compact(0));
        assert!(!no_compact.use_compact(1));
    }

    // ── contains_word tests ───────────────────────────────────────

    #[test]
    fn test_contains_word_exact_match() {
        assert!(contains_word("read the file content", "file"));
        assert!(contains_word("file is here", "file"));
        assert!(contains_word("the file", "file"));
        assert!(contains_word("file", "file"));
    }

    #[test]
    fn test_contains_word_rejects_substring() {
        // "profile" should NOT match "file"
        assert!(!contains_word("check the user profile settings", "file"));
        // "findings" should NOT match "find"
        assert!(!contains_word("search for findings", "find"));
        // "import" should NOT match "port"
        assert!(!contains_word("import the module", "port"));
    }

    #[test]
    fn test_contains_word_with_punctuation() {
        assert!(contains_word("open file.", "file"));
        assert!(contains_word("file, folder", "file"));
        assert!(contains_word("(file)", "file"));
        assert!(contains_word("the file's content", "file"));
    }

    #[test]
    fn test_contains_word_multi_word_keyword() {
        assert!(contains_word("analyze the call graph", "call graph"));
        assert!(contains_word("state machine transitions", "state machine"));
        // Multi-word uses substring match, so embedded is OK
        assert!(contains_word("recall graph data", "call graph"));
    }

    #[test]
    fn test_contains_word_no_false_positive_code_keywords() {
        // "profile" should not trigger "file"
        let msg = "check the user profile settings";
        assert!(!CODE_KEYWORDS.iter().any(|kw| contains_word(msg, kw)));

        // "findings" should not trigger "find"
        let msg2 = "review the findings report";
        assert!(!CODE_KEYWORDS.iter().any(|kw| contains_word(msg2, kw)));
    }

    #[test]
    fn test_contains_word_accented_boundary() {
        // `is_ascii_alphanumeric` treated the leading byte of an accented char as a
        // boundary, so "partage" matched inside "partagée" and pulled the whole
        // Collaboration group into a Workspace question. Caught by
        // tests/routing_metamorphic.rs (invariant `lang`).
        assert!(!contains_word(
            "on a besoin d'une ressource partagée",
            "partage"
        ));
        assert!(contains_word("partage ce résultat", "partage"));
        assert!(!contains_word("il est noté absent", "note"));
        assert!(!contains_word("une clé étrangère", "cle"));
    }

    #[test]
    fn test_contains_word_plural_tolerance() {
        // A single trailing plural marker is tolerated...
        assert!(contains_word("list the open sessions", "session"));
        assert!(contains_word("quelles compétences", "compétence"));
        assert!(contains_word("read the files", "file"));
        // ...but ONLY for single-word keywords. A multi-word keyword is a plain
        // substring match, so "point chaud" does not match "points chauds" —
        // STRUCTURAL_KEYWORDS carries both forms for exactly this reason.
        assert!(!contains_word("les points chauds", "point chaud"));
        assert!(STRUCTURAL_KEYWORDS
            .iter()
            .any(|kw| contains_word("les points chauds du projet", kw)));
        // ...but it must not reopen the substring hole.
        assert!(!contains_word("review the findings report", "find"));
        assert!(!contains_word("check the user profile settings", "file"));
        assert!(!contains_word("the classifier output", "class"));
    }

    #[test]
    fn test_contains_word_true_positive_code_keywords() {
        let msg = "read the file content";
        assert!(CODE_KEYWORDS.iter().any(|kw| contains_word(msg, kw)));

        let msg2 = "find the function definition";
        assert!(CODE_KEYWORDS.iter().any(|kw| contains_word(msg2, kw)));
    }
}
