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
    /// §16 — Tool reference — the exhaustive MCP mega-tools reference (selective by group)
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
        }
    }

    /// Section is included only when protocols are active.
    pub const fn when_protocol_active() -> Self {
        Self {
            scaffolding_min: None,
            scaffolding_max: None,
            requires_active_plan: false,
            requires_active_protocol: true,
            min_task_count: None,
            always: false,
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
            has_active_plan: true,   // Assume plan active to include plan-dependent sections
            has_active_protocol: true, // Assume protocols to include protocol sections
            task_count: 10,          // Assume enough tasks for wave dispatch
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
You have **25 MCP mega-tools** covering the full project lifecycle: planning, execution, tracking, code exploration, knowledge management, neural skills, reasoning, behavioral patterns, living personas, episodic memory, sharing.

**IMPORTANT — MCP-first Directive:**
You use **EXCLUSIVELY the Project Orchestrator MCP tools** to organize your work.
You MUST **NOT** use Claude Code internal features for project management:
- ❌ Plan mode (EnterPlanMode / ExitPlanMode) — use `plan(action: "create")`, `task(action: "create")`, `step(action: "create")`
- ❌ TodoWrite — use `task(action: "update")`, `step(action: "update")` to track progress
- ❌ Any other internal planning tool

When asked to "plan", create an **MCP Plan** with Tasks and Steps.
When asked to "track progress", update **statuses via MCP tools**."#;

/// §2 — Mega-tools call syntax
pub const SECTION_MEGATOOLS_SYNTAX: &str = r#"## 2. Mega-tools — Call Syntax

Each tool has an `action` parameter that determines the operation:

```
tool_name(action: "<action>", param1: value1, param2: value2, ...)
```

The 25 mega-tools: `project`, `plan`, `task`, `step`, `decision`, `constraint`, `release`, `milestone`, `commit`, `note`, `workspace`, `workspace_milestone`, `resource`, `component`, `chat`, `feature_graph`, `code`, `reasoning`, `admin`, `skill`, `analysis_profile`, `protocol`, `persona`, `episode`, `sharing`"#;

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

// ============================================================================
// Section registry — all 16 base sections with their metadata
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
        },
        BasePromptSection {
            id: PromptSectionId::MegatoolsSyntax,
            content: SECTION_MEGATOOLS_SYNTAX,
            priority: 1,
            condition: ActivationCondition::always(),
        },
        BasePromptSection {
            id: PromptSectionId::DataModel,
            content: SECTION_DATA_MODEL,
            priority: 2,
            condition: ActivationCondition::scaffolding_range(0, 2),
        },
        BasePromptSection {
            id: PromptSectionId::TreeSitterSync,
            content: SECTION_TREE_SITTER,
            priority: 3,
            condition: ActivationCondition::scaffolding_range(0, 2),
        },
        BasePromptSection {
            id: PromptSectionId::GitWorkflow,
            content: SECTION_GIT_WORKFLOW,
            priority: 4,
            condition: ActivationCondition::scaffolding_range(0, 3),
        },
        BasePromptSection {
            id: PromptSectionId::TaskExecutionProtocol,
            content: SECTION_TASK_EXECUTION,
            priority: 5,
            condition: ActivationCondition::when_plan_active(),
        },
        BasePromptSection {
            id: PromptSectionId::PlanningProtocol,
            content: SECTION_PLANNING,
            priority: 6,
            // Always included since planning can be requested at any time
            condition: ActivationCondition::scaffolding_range(0, 3),
        },
        BasePromptSection {
            id: PromptSectionId::StatusManagement,
            content: SECTION_STATUS_MANAGEMENT,
            priority: 7,
            condition: ActivationCondition::scaffolding_range(0, 2),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesImpact,
            content: SECTION_BP_IMPACT,
            priority: 8,
            condition: ActivationCondition::always(),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesSearch,
            content: SECTION_BP_SEARCH,
            priority: 9,
            condition: ActivationCondition::scaffolding_range(0, 3),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesKnowledge,
            content: SECTION_BP_KNOWLEDGE,
            priority: 10,
            condition: ActivationCondition::always(),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesAdvanced,
            content: SECTION_BP_ADVANCED,
            priority: 11,
            condition: ActivationCondition::scaffolding_range(0, 1),
        },
        BasePromptSection {
            id: PromptSectionId::BestPracticesPersonas,
            content: SECTION_BP_PERSONAS,
            priority: 12,
            condition: ActivationCondition::scaffolding_range(0, 1),
        },
        BasePromptSection {
            id: PromptSectionId::DeepMaintenance,
            content: SECTION_DEEP_MAINTENANCE,
            priority: 13,
            condition: ActivationCondition::scaffolding_range(0, 1),
        },
        BasePromptSection {
            id: PromptSectionId::PlanExecutionAutomation,
            content: SECTION_PLAN_EXECUTION,
            priority: 14,
            condition: ActivationCondition::when_plan_active(),
        },
        // §16 — Tool Reference is handled separately (selective by group)
        // but included in the registry for completeness
        BasePromptSection {
            id: PromptSectionId::ToolReference,
            content: "", // Content comes from TOOL_REFERENCE or selected groups
            priority: 20,
            condition: ActivationCondition::always(),
        },
    ]
}

/// Select sections that are active given the current context.
///
/// Returns sections filtered by their activation conditions, sorted by priority.
pub fn select_sections(ctx: &ComposerContext) -> Vec<BasePromptSection> {
    all_base_sections()
        .into_iter()
        .filter(|s| s.condition.is_active(ctx))
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

// ============================================================================
// Tool Reference Groups — selective rendering of TOOL_REFERENCE
// ============================================================================

/// Identifies a group of related MCP tools for selective rendering.
///
/// Instead of always including the full ~600-line TOOL_REFERENCE, the system
/// selects relevant groups based on intent, FSM state, and scaffolding level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolRefGroupId {
    /// project, plan, task, step, constraint, release, milestone — always included
    Core,
    /// note, decision, commit — always included (compact)
    Knowledge,
    /// code, analysis_profile — included when code questions / file mentions
    CodeExploration,
    /// admin — included at L0-L1 or when GDS/maintenance requested
    Structural,
    /// protocol, skill, persona, episode — included when protocols active or L0-L1
    Behavioral,
    /// workspace, workspace_milestone, resource, component — included in multi-project
    Workspace,
    /// chat, feature_graph, sharing, reasoning — included on demand
    Collaboration,
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
    ];

    /// Tool names belonging to this group.
    pub fn tool_names(&self) -> &'static [&'static str] {
        match self {
            Self::Core => &["project", "plan", "task", "step", "constraint", "release", "milestone"],
            Self::Knowledge => &["note", "decision", "commit"],
            Self::CodeExploration => &["code", "analysis_profile"],
            Self::Structural => &["admin"],
            Self::Behavioral => &["protocol", "skill", "persona", "episode"],
            Self::Workspace => &["workspace", "workspace_milestone", "resource", "component"],
            Self::Collaboration => &["chat", "feature_graph", "sharing", "reasoning"],
        }
    }

    /// Display name for the group header.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Core => "Core (Project, Planning, Tracking)",
            Self::Knowledge => "Knowledge (Notes, Decisions, Commits)",
            Self::CodeExploration => "Code Exploration & Analytics",
            Self::Structural => "Admin & Sync",
            Self::Behavioral => "Behavioral (Protocols, Skills, Personas, Episodes)",
            Self::Workspace => "Workspace (Multi-project)",
            Self::Collaboration => "Collaboration (Chat, Features, Sharing, Reasoning)",
        }
    }
}

/// Context for tool group selection.
pub struct ToolGroupSelectionContext {
    pub scaffolding_level: u8,
    pub has_active_protocol: bool,
    pub is_multi_project: bool,
    /// Tool names whitelisted by the current FSM state's `available_tools`.
    /// Empty = no FSM restriction (include all).
    pub fsm_available_tools: Vec<String>,
    /// Keywords detected in the user message for intent matching.
    pub user_intent_keywords: Vec<String>,
}

impl Default for ToolGroupSelectionContext {
    fn default() -> Self {
        Self {
            scaffolding_level: 0,
            has_active_protocol: false,
            is_multi_project: false,
            fsm_available_tools: vec![],
            user_intent_keywords: vec![],
        }
    }
}

/// Intent keywords that trigger specific tool groups.
const CODE_KEYWORDS: &[&str] = &[
    "code", "function", "file", "import", "search", "find", "reference",
    "symbol", "architecture", "impact", "call graph", "dependency",
    "trait", "struct", "class", "cherche", "fonction", "fichier",
];
const STRUCTURAL_KEYWORDS: &[&str] = &[
    "admin", "sync", "maintenance", "health", "fabric", "gds",
    "neuron", "synapse", "energy", "staleness", "hotspot",
];
const BEHAVIORAL_KEYWORDS: &[&str] = &[
    "protocol", "skill", "persona", "episode", "fsm", "state machine",
    "transition", "protocole", "compétence",
];
const WORKSPACE_KEYWORDS: &[&str] = &[
    "workspace", "component", "resource", "multi-project", "topology",
    "cross-project",
];
const COLLAB_KEYWORDS: &[&str] = &[
    "chat", "session", "feature graph", "sharing", "reasoning",
    "reason", "conversation",
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
                    ctx.fsm_available_tools.iter().any(|allowed| allowed == tool)
                })
            })
            .collect();
    }

    let mut selected = vec![ToolRefGroupId::Core, ToolRefGroupId::Knowledge];
    let msg_lower: String = ctx.user_intent_keywords.join(" ").to_lowercase();

    // Intent-based inclusion
    let has_code_intent = CODE_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));
    let has_structural_intent = STRUCTURAL_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));
    let has_behavioral_intent = BEHAVIORAL_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));
    let has_workspace_intent = WORKSPACE_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));
    let has_collab_intent = COLLAB_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));

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

    selected
}

/// Extract tool sections from the full TOOL_REFERENCE text for a given set of groups.
///
/// Parses TOOL_REFERENCE by finding `## toolname` headers and extracting only
/// the sections whose tool names match the selected groups.
pub fn extract_tool_reference(
    tool_reference: &str,
    groups: &[ToolRefGroupId],
) -> String {
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
            include_current = wanted_tools.iter().any(|w| *w == tool_name);
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

    #[test]
    fn test_all_sections_count() {
        let sections = all_base_sections();
        assert_eq!(sections.len(), 16, "Expected 16 base sections");
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
            assert!(s.priority <= 20, "Priority {} too high for {:?}", s.priority, s.id);
        }
    }

    #[test]
    fn test_level0_full_activates_all() {
        let ctx = ComposerContext::level0_full();
        let selected = select_sections(&ctx);
        assert_eq!(
            selected.len(), 16,
            "L0 with full context should activate all 16 sections, got {}",
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

        assert!(ids.contains(&PromptSectionId::IdentityRole), "Identity always included");
        assert!(ids.contains(&PromptSectionId::MegatoolsSyntax), "Megatools always included");
        assert!(ids.contains(&PromptSectionId::BestPracticesImpact), "BP Impact always included");
        assert!(ids.contains(&PromptSectionId::BestPracticesKnowledge), "BP Knowledge always included");
        assert!(ids.contains(&PromptSectionId::ToolReference), "ToolRef always included");
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

        assert!(!ids.contains(&PromptSectionId::TaskExecutionProtocol), "TaskExec excluded without plan");
        assert!(!ids.contains(&PromptSectionId::PlanExecutionAutomation), "PlanExec excluded without plan");
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

        assert!(assembled.contains("# Development Agent"), "Should start with title");
        assert!(assembled.contains("## 1. Identity & Role"), "Should contain Identity section");
        assert!(assembled.contains("## 2. Mega-tools"), "Should contain Megatools section");
        assert!(assembled.contains("## 3. Data Model"), "Should contain Data Model section");
        assert!(assembled.contains("## 4. Git Workflow"), "Should contain Git section");
    }

    #[test]
    fn test_display_section_id() {
        assert_eq!(format!("{}", PromptSectionId::IdentityRole), "identity_role");
        assert_eq!(format!("{}", PromptSectionId::ToolReference), "tool_reference");
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
                i, section_counts[i], i - 1, section_counts[i - 1]
            );
        }
    }

    // ── Tool Reference Group tests ───────────────────────────────────

    #[test]
    fn test_tool_groups_cover_all_25_tools() {
        let mut all_tools: Vec<&str> = ToolRefGroupId::ALL
            .iter()
            .flat_map(|g| g.tool_names().iter().copied())
            .collect();
        all_tools.sort();
        all_tools.dedup();
        // 25 mega-tools + analysis_profile + neural_routing/trajectory = at least 25
        assert!(
            all_tools.len() >= 25,
            "Groups should cover at least 25 tools, got {}",
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
            groups.len(), 7,
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
        assert_eq!(groups.len(), 2, "L4 without intent should have 2 groups, got {}", groups.len());
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
        assert!(groups.contains(&ToolRefGroupId::Core), "Core always included");
        assert!(groups.contains(&ToolRefGroupId::Knowledge), "Knowledge always included");
        assert!(groups.contains(&ToolRefGroupId::CodeExploration), "CodeExploration has 'code'");
        assert!(!groups.contains(&ToolRefGroupId::Workspace), "Workspace not in whitelist");
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

        let core_only = extract_tool_reference(TOOL_REFERENCE, &[ToolRefGroupId::Core, ToolRefGroupId::Knowledge]);
        let full = extract_tool_reference(TOOL_REFERENCE, &ToolRefGroupId::ALL.to_vec());

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
}
