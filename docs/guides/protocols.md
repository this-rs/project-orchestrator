# Protocols — Building a Safe, Self-Aware Setup

How to use the Protocol FSM engine to build repeatable workflows that leverage
the full power of Project Orchestrator: health checks, impact analysis,
knowledge fabric, risk assessment, and neural maintenance.

**Time:** ~30 minutes to set up a production-ready protocol suite

---

## Why Protocols?

Without protocols, agents work ad-hoc — they might skip impact analysis,
forget to create notes, or modify high-risk code without checking the
knowledge fabric. Protocols encode **guardrails as state machines**:

- Each **state** represents a phase with specific MCP actions to perform
- Each **transition** requires an explicit trigger — no skipping steps
- **Guards** document preconditions (informational today, executable in future)
- **Hierarchical composition** lets you nest protocols (e.g., a code-review
  sub-protocol inside a feature-development parent)
- **Relevance vectors** let the system auto-suggest the right protocol for
  the current context

The goal: an agent that **knows what it knows, knows what it doesn't know,
and follows safe procedures automatically**.

---

## Table of Contents

1. [Quick Start — Your First Protocol](#1-quick-start--your-first-protocol)
2. [Project Onboarding (Assimilation)](#2-project-onboarding-assimilation)
3. [Safe Modification Protocol](#3-safe-modification-protocol)
4. [Knowledge Maintenance (System)](#4-knowledge-maintenance-system)
5. [Code Review Protocol](#5-code-review-protocol)
6. [Hierarchical Protocols](#6-hierarchical-protocols--feature-workflow)
7. [RFC Lifecycle](#7-rfc-lifecycle)
8. [Auto-Triggered Protocols](#8-auto-triggered-protocols)
9. [Routing & Simulation](#9-routing--simulation)
10. [Building Your Own](#10-building-your-own-protocols)

---

## 1. Quick Start — Your First Protocol

A protocol is created with a single `compose` call. Here's a minimal example:

```
protocol(action: "compose", project_id: "...",
  name: "pre-commit-check",
  category: "business",
  states: [
    { name: "analyze",  state_type: "start",       description: "Check what files changed and their risk level" },
    { name: "verify",   state_type: "intermediate", description: "Run tests and topology checks" },
    { name: "approved", state_type: "terminal",     description: "Changes are safe to commit" }
  ],
  transitions: [
    { from_state: "analyze", to_state: "verify",   trigger: "analysis_clean" },
    { from_state: "analyze", to_state: "analyze",   trigger: "risk_found",    guard: "Re-analyze after addressing risks" },
    { from_state: "verify",  to_state: "approved",  trigger: "tests_pass" }
  ]
)
```

This returns `{ protocol_id, skill_id, states_created: 3, transitions_created: 3 }`.

### Running it

```
# Start a run
protocol(action: "start_run", protocol_id: "...")
# → run_id, current_state: "analyze"

# After performing the analysis work, advance:
protocol(action: "transition", run_id: "...", trigger: "analysis_clean")
# → current_state: "verify"

protocol(action: "transition", run_id: "...", trigger: "tests_pass")
# → current_state: "approved", status: "completed"
```

### What makes this useful?

The protocol is **not just documentation** — it's a tracked FSM run. You can:
- Query `protocol(action: "get_run", run_id)` to see the full visit history
- See which state you're in, how long each phase took
- The skill associated with the protocol accumulates energy over activations

---

## 2. Project Onboarding (Assimilation)

When you register a new project, you need to build up the knowledge graph
from scratch. This protocol ensures nothing is missed.

### The Protocol

```
protocol(action: "compose", project_id: "...",
  name: "project-onboarding",
  category: "system",
  states: [
    { name: "sync",
      state_type: "start",
      description: "Parse source code with Tree-sitter and index in Meilisearch",
      action: "project(action: 'sync', slug: '{project_slug}')" },

    { name: "explore-structure",
      state_type: "intermediate",
      description: "Understand the architecture: top files, communities, entry points",
      action: "code(action: 'get_architecture') + code(action: 'get_communities') + code(action: 'get_entry_points')" },

    { name: "detect-risks",
      state_type: "intermediate",
      description: "Identify hotspots, god functions, knowledge gaps, and circular deps",
      action: "code(action: 'get_health') + code(action: 'get_risk_assessment')" },

    { name: "document-findings",
      state_type: "intermediate",
      description: "Create notes for every pattern, convention, and gotcha discovered",
      action: "note(action: 'create', ...) + note(action: 'link_to_entity', ...)" },

    { name: "bootstrap-fabric",
      state_type: "intermediate",
      description: "Initialize the Knowledge Fabric: synapses, co-change, GDS scores",
      action: "admin(action: 'bootstrap_knowledge_fabric')" },

    { name: "verify-health",
      state_type: "intermediate",
      description: "Run a final health check to confirm the fabric is healthy",
      action: "code(action: 'get_health') — verify pain_score < 0.4" },

    { name: "onboarded",
      state_type: "terminal",
      description: "Project is fully indexed, documented, and the fabric is active" }
  ],
  transitions: [
    { from_state: "sync",               to_state: "explore-structure", trigger: "sync_complete" },
    { from_state: "explore-structure",   to_state: "detect-risks",     trigger: "exploration_done" },
    { from_state: "detect-risks",       to_state: "document-findings", trigger: "risks_assessed" },
    { from_state: "document-findings",  to_state: "bootstrap-fabric",  trigger: "documented" },
    { from_state: "bootstrap-fabric",   to_state: "verify-health",     trigger: "fabric_ready" },
    { from_state: "verify-health",      to_state: "onboarded",         trigger: "health_ok" },
    { from_state: "verify-health",      to_state: "document-findings", trigger: "health_bad",
      guard: "pain_score > 0.4 — need more documentation to fill knowledge gaps" }
  ],
  relevance_vector: { phase: 0.0, structure: 0.8, domain: 0.5, resource: 0.5, lifecycle: 0.0 }
)
```

### What the agent does at each state

#### State: `sync`
```
project(action: "sync", slug: "my-project")
```
Wait for sync to complete. Check the output: files parsed, functions found,
structs indexed. If < 10 files, verify the `root_path` is correct.

#### State: `explore-structure`
```
# 1. Architecture overview — most connected files, language stats
code(action: "get_architecture", project_slug: "my-project")

# 2. Community detection — clusters of coupled files
code(action: "get_communities", project_slug: "my-project")

# 3. Entry points — main functions, HTTP handlers, CLI commands
code(action: "get_entry_points", project_slug: "my-project")

# 4. Business processes — BFS traces from entry points
code(action: "list_processes", project_slug: "my-project")
```

Read the communities carefully — they reveal the module boundaries
that aren't always obvious from the directory structure.

#### State: `detect-risks`
```
# Full health report
code(action: "get_health", project_slug: "my-project")

# Risk assessment — composite score per file
code(action: "get_risk_assessment", project_slug: "my-project")

# Knowledge gaps — files with zero notes/decisions
code(action: "get_knowledge_gaps", project_slug: "my-project")
```

Pay special attention to:
- **God functions** (high complexity, many callers)
- **Bridge files** (high betweenness — removing them disconnects the graph)
- **Hotspots** (high churn + high coupling = fragile)

#### State: `document-findings`
For every pattern, convention, or gotcha you discovered:

```
# Example: discovered that all handlers follow a pattern
note(action: "create",
  project_id: "...",
  note_type: "pattern",
  content: "All HTTP handlers follow extract→validate→execute→respond. The extractor does auth + parsing, the handler contains only business logic.",
  importance: "high",
  tags: ["architecture", "handlers", "convention"])
# → note_id

# Link to relevant files
note(action: "link_to_entity", note_id: "...",
  entity_type: "file", entity_id: "src/api/handlers.rs")

# Example: discovered a gotcha
note(action: "create",
  project_id: "...",
  note_type: "gotcha",
  content: "neo4rs returns BoltNull for missing optional rels — always use OPTIONAL MATCH and check .is_null() before unwrapping",
  importance: "critical",
  tags: ["neo4j", "null-safety", "gotcha"])
```

**Rule of thumb**: Create at least 1 note per community discovered.
Each high-risk file should have at least 1 note explaining its role.

#### State: `bootstrap-fabric`
```
admin(action: "bootstrap_knowledge_fabric", project_id: "...")
```

This computes:
- CO_CHANGED relations from commit history (TOUCHES)
- GDS scores (PageRank, betweenness, Louvain communities)
- Fabric fusion scores (multi-layer: IMPORTS + CO_CHANGED + AFFECTS + SYNAPSE)
- Note staleness and energy scores

#### State: `verify-health`
```
code(action: "get_health", project_slug: "my-project")
```

Check the `pain_score`. If it's > 0.4, transition back to `document-findings`
to fill knowledge gaps. Common causes:
- `note_density < 0.3` → Need more notes linked to files
- `decision_coverage < 10%` → Need architectural decisions with AFFECTS links
- `synapse_health < 0.2` → Run `admin(action: "backfill_synapses")`

### When to use this protocol

- First time registering a project
- After a major refactoring that restructured the codebase
- When onboarding a new team member (run it together to share knowledge)

---

## 3. Safe Modification Protocol

**The most important protocol.** Use it before any non-trivial code change.

### The Protocol

```
protocol(action: "compose", project_id: "...",
  name: "safe-modification",
  category: "business",
  states: [
    { name: "gather-context",
      state_type: "start",
      description: "Load all relevant knowledge: notes, decisions, propagated context, RFCs" },

    { name: "analyze-impact",
      state_type: "intermediate",
      description: "Run structural + temporal impact analysis on target files" },

    { name: "check-topology",
      state_type: "intermediate",
      description: "Verify that planned changes don't violate architectural rules" },

    { name: "check-risks",
      state_type: "intermediate",
      description: "Evaluate risk level of target files — PageRank, betweenness, churn" },

    { name: "implement",
      state_type: "intermediate",
      description: "Make the code changes with full awareness of impact and constraints" },

    { name: "verify",
      state_type: "intermediate",
      description: "Run tests, check topology again, verify no new violations" },

    { name: "document",
      state_type: "intermediate",
      description: "Create/update notes and decisions for the changes made" },

    { name: "done",
      state_type: "terminal",
      description: "Changes are safe, tested, and documented" }
  ],
  transitions: [
    { from_state: "gather-context",  to_state: "analyze-impact",  trigger: "context_loaded" },
    { from_state: "analyze-impact",  to_state: "check-topology",  trigger: "impact_assessed" },
    { from_state: "check-topology",  to_state: "check-risks",     trigger: "topology_ok" },
    { from_state: "check-topology",  to_state: "gather-context",  trigger: "topology_violation",
      guard: "New imports would violate architectural rules — rethink the approach" },
    { from_state: "check-risks",     to_state: "implement",       trigger: "risks_acceptable" },
    { from_state: "check-risks",     to_state: "gather-context",  trigger: "high_risk",
      guard: "Target file has critical risk score — need more context or a different approach" },
    { from_state: "implement",       to_state: "verify",          trigger: "changes_made" },
    { from_state: "verify",          to_state: "document",        trigger: "verification_passed" },
    { from_state: "verify",          to_state: "implement",       trigger: "verification_failed",
      guard: "Tests failed or new topology violations — fix before proceeding" },
    { from_state: "document",        to_state: "done",            trigger: "documented" }
  ],
  relevance_vector: { phase: 0.5, structure: 0.7, domain: 0.5, resource: 0.5, lifecycle: 0.5 }
)
```

### What the agent does at each state

#### State: `gather-context`

This is the **warm-up phase** — never skip it.

```
# 1. Semantic search for relevant notes
note(action: "search_semantic", query: "the feature or area you're about to modify")

# 2. Past architectural decisions
decision(action: "search_semantic", query: "relevant topic")

# 3. Contextual notes on the target file
note(action: "get_context", entity_type: "file", entity_id: "src/target/file.rs")

# 4. Propagated notes via Knowledge Fabric (import graph + co-change + AFFECTS)
note(action: "get_propagated", file_path: "src/target/file.rs", slug: "my-project")

# 5. Active RFCs — never conflict with an in-flight RFC
note(action: "list_rfcs", project_id: "...")

# 6. Decisions that AFFECT the target file
decision(action: "get_affecting", entity_type: "File", entity_id: "src/target/file.rs")
```

If any RFC is in `proposed` or `accepted` state and overlaps with your changes,
**stop and coordinate** — don't create conflicting work.

#### State: `analyze-impact`

```
# Structural impact — files and symbols affected
code(action: "analyze_impact", target: "src/target/file.rs")

# Temporal coupling — files that historically change together
code(action: "get_file_co_changers", file_path: "src/target/file.rs")

# Dependencies — who imports this file? what does it import?
code(action: "get_file_dependencies", file_path: "src/target/file.rs")

# For OOP: check the class hierarchy BEFORE modifying public/protected methods
code(action: "get_class_hierarchy", type_name: "TargetClass")
code(action: "find_subclasses", class_name: "TargetClass")
```

**Key insight**: `analyze_impact` gives you **structural** coupling (import graph).
`get_file_co_changers` gives you **temporal** coupling (commit history).
Files that appear in both are **high-confidence** impacted files.

#### State: `check-topology`

```
# Check if your planned new imports would violate rules
code(action: "check_file_topology",
  project_slug: "my-project",
  file_path: "src/target/file.rs",
  new_imports: ["src/other/module.rs", "src/shared/utils.rs"])

# Or check all existing violations
code(action: "check_topology", project_slug: "my-project")
```

Topology rules enforce architectural boundaries:
- `must_not_import`: "src/api/** must not import src/cli/**"
- `no_circular`: No circular dependencies between layers
- `max_fan_out`: A file shouldn't import more than N files

If violations are found, transition back to `gather-context` with `topology_violation`.

#### State: `check-risks`

```
# Node importance — PageRank, betweenness, bridge detection
code(action: "get_node_importance",
  project_slug: "my-project",
  node_path: "src/target/file.rs",
  node_type: "File")
```

The response includes a `risk_level` (critical/high/medium/low) and
an interpretive summary. For **critical** or **high** risk files:

- Ensure you have thorough test coverage
- Consider whether a less invasive approach exists
- Check if there are co-changers you might miss

#### State: `implement`

Now you have full context. Make changes knowing:
- What other files are affected (impact analysis)
- What rules you must respect (topology)
- What decisions were made before (AFFECTS decisions)
- What gotchas exist (propagated notes)

#### State: `verify`

```
# Run tests (language-specific)
cargo test  # or npm test, pytest, etc.

# Re-check topology after changes
code(action: "check_topology", project_slug: "my-project")

# Verify no new circular dependencies
```

#### State: `document`

```
# Record the decision if you made an architectural choice
decision(action: "add", task_id: "...",
  description: "Extracted shared validation logic into src/shared/validation.rs",
  rationale: "3 handlers duplicated the same validation — DRY + easier to test",
  alternatives: ["Keep duplicated — simpler but maintenance burden"],
  chosen_option: "Extract to shared module")

# Link decision to affected files
decision(action: "add_affects", decision_id: "...",
  entity_type: "File", entity_id: "src/shared/validation.rs")

# Create notes for any gotchas discovered
note(action: "create", project_id: "...",
  note_type: "gotcha",
  content: "validation.rs parse_email() silently accepts '+' aliases — intentional per RFC 5321",
  importance: "medium", tags: ["validation", "email"])

# Mark entities as discussed in this session
chat(action: "add_discussed", session_id: "...",
  entities: [
    { entity_type: "file", entity_id: "src/shared/validation.rs" },
    { entity_type: "file", entity_id: "src/api/handlers/user.rs" }
  ])
```

### When to use this protocol

- Before any change to a file with risk_level >= medium
- Before refactoring across module boundaries
- Before modifying public APIs or shared code
- **Always** for files with high betweenness centrality (bridge files)

---

## 4. Knowledge Maintenance (System)

A system protocol that runs periodically to keep the knowledge fabric healthy.

### The Protocol

```
protocol(action: "compose", project_id: "...",
  name: "knowledge-maintenance",
  category: "system",
  states: [
    { name: "audit",
      state_type: "start",
      description: "Audit knowledge graph for gaps: orphan notes, decisions without AFFECTS, unlinked commits" },

    { name: "health-check",
      state_type: "intermediate",
      description: "Run full health report — hotspots, risks, homeostasis, neural metrics" },

    { name: "decay-synapses",
      state_type: "intermediate",
      description: "Apply gentle decay to synapse weights, prune dead connections" },

    { name: "update-scores",
      state_type: "intermediate",
      description: "Recalculate staleness, energy, and fabric fusion scores" },

    { name: "review-stale",
      state_type: "intermediate",
      description: "Find stale or needs_review notes and flag them" },

    { name: "report",
      state_type: "terminal",
      description: "Persist health report as a note for historical tracking" }
  ],
  transitions: [
    { from_state: "audit",           to_state: "health-check",    trigger: "audit_done" },
    { from_state: "health-check",    to_state: "decay-synapses",  trigger: "health_assessed" },
    { from_state: "decay-synapses",  to_state: "update-scores",   trigger: "decay_applied" },
    { from_state: "update-scores",   to_state: "review-stale",    trigger: "scores_updated" },
    { from_state: "review-stale",    to_state: "report",          trigger: "review_done" }
  ],
  relevance_vector: { phase: 0.75, structure: 0.3, domain: 0.5, resource: 0.3, lifecycle: 0.8 }
)
```

### What the agent does at each state

#### State: `audit`
```
admin(action: "audit_gaps", project_id: "...")
```

Returns:
- **Orphan notes**: notes not linked to any entity (useless without context)
- **Decisions without AFFECTS**: architectural decisions not linked to code
- **Commits without TOUCHES**: commits that didn't register file changes
- **Skills without members**: empty skill shells

Fix the easy ones immediately:
```
# Link an orphan note to the file it's about
note(action: "link_to_entity", note_id: "...",
  entity_type: "file", entity_id: "src/relevant/file.rs")

# Add AFFECTS to a decision
decision(action: "add_affects", decision_id: "...",
  entity_type: "File", entity_id: "src/affected/file.rs")
```

#### State: `health-check`
```
code(action: "get_health", project_slug: "my-project")
```

Look for:
- `pain_score > 0.6` → Action needed
- `synapse_health > 3.0` → Network too dense, needs decay
- `synapse_health < 0.2` → Network too sparse, needs backfill
- `scar_load > 15%` → Too many scarred nodes, review old scars
- `dead_notes > 20%` → Notes losing energy, need reinforcement

#### State: `decay-synapses`
```
# Gentle decay — NEVER use decay_amount > 0.1 in a single pass
admin(action: "decay_synapses", decay_amount: 0.03, prune_threshold: 0.1)
```

> **Safety rule**: Always use small decay amounts (0.01-0.05).
> Large values (> 0.2) can destroy the entire synapse network in one pass.
> If the network needs significant cleanup, run multiple gentle passes
> with a health check between each.

If synapse_health was < 0.2 (too sparse), backfill instead:
```
admin(action: "backfill_synapses")
```

#### State: `update-scores`
```
admin(action: "update_staleness_scores", project_id: "...")
admin(action: "update_energy_scores", project_id: "...")
admin(action: "update_fabric_scores", project_id: "...")
```

#### State: `review-stale`
```
note(action: "get_needing_review", project_id: "...")
```

For each stale note:
- If still valid → `note(action: "confirm", note_id: "...")`
- If outdated → `note(action: "invalidate", note_id: "...")`
- If superseded → `note(action: "supersede", note_id: "...", superseded_by_id: "...")`

#### State: `report`
```
admin(action: "persist_health_report", project_id: "...")
```

This creates a note with type `observation` and tags `["health-check", "auto-generated"]`,
including a delta analysis vs. the previous health report.

### Scheduling

This protocol should run regularly. You can set it up as auto-triggered:

```
protocol(action: "update", protocol_id: "...",
  trigger_mode: "scheduled",
  trigger_config: { schedule: "weekly" })
```

Or run it manually after significant work:
```
protocol(action: "start_run", protocol_id: "...")
```

---

## 5. Code Review Protocol

A protocol for reviewing changes before they're merged. Works well as a
sub-protocol inside a feature workflow (see section 6).

### The Protocol

```
protocol(action: "compose", project_id: "...",
  name: "code-review",
  category: "business",
  states: [
    { name: "identify-changes",
      state_type: "start",
      description: "List all files changed, understand the scope" },

    { name: "check-impact",
      state_type: "intermediate",
      description: "Run impact analysis on each changed file" },

    { name: "check-conventions",
      state_type: "intermediate",
      description: "Verify changes follow documented patterns and guidelines" },

    { name: "check-knowledge-gaps",
      state_type: "intermediate",
      description: "Ensure changed files have adequate documentation" },

    { name: "approve",
      state_type: "terminal",
      description: "Changes pass review" },

    { name: "request-changes",
      state_type: "terminal",
      description: "Changes need revision" }
  ],
  transitions: [
    { from_state: "identify-changes",     to_state: "check-impact",         trigger: "scope_clear" },
    { from_state: "check-impact",         to_state: "check-conventions",    trigger: "impact_acceptable" },
    { from_state: "check-impact",         to_state: "request-changes",      trigger: "impact_too_wide",
      guard: "Changes affect too many files or cross community boundaries" },
    { from_state: "check-conventions",    to_state: "check-knowledge-gaps", trigger: "conventions_ok" },
    { from_state: "check-conventions",    to_state: "request-changes",      trigger: "convention_violation",
      guard: "Code doesn't follow documented patterns" },
    { from_state: "check-knowledge-gaps", to_state: "approve",              trigger: "well_documented" },
    { from_state: "check-knowledge-gaps", to_state: "request-changes",      trigger: "undocumented",
      guard: "Changed files have no notes/decisions — add documentation before merging" }
  ],
  relevance_vector: { phase: 0.75, structure: 0.6, domain: 0.5, resource: 0.5, lifecycle: 0.6 }
)
```

### What the agent does at each state

#### State: `check-conventions`

This is where the knowledge fabric shines. Search for guidelines that apply:

```
# Find guideline notes for the project
note(action: "search", query: "guideline convention")

# Get propagated notes for each changed file
note(action: "get_propagated", file_path: "src/changed/file.rs", slug: "my-project")

# Check if there's a pattern note for this kind of code
note(action: "search_semantic", query: "handler pattern error handling")
```

Compare the actual changes against documented patterns. If a guideline note
says "all handlers must validate input with the shared validator" but the
new handler does inline validation, that's a convention violation.

#### State: `check-knowledge-gaps`

```
# Check each changed file
code(action: "get_node_importance",
  project_slug: "my-project",
  node_path: "src/changed/file.rs",
  node_type: "File")

# If high importance + zero notes → knowledge gap
note(action: "get_context", entity_type: "file", entity_id: "src/changed/file.rs")
```

The review should require that any modified file with `risk_level >= medium`
has at least one note explaining its purpose and any gotchas.

---

## 6. Hierarchical Protocols — Feature Workflow

Combine protocols using **sub_protocol_id** on states. The parent waits
for the child to complete before advancing.

### Parent: Feature Development Workflow

```
protocol(action: "compose", project_id: "...",
  name: "feature-workflow",
  category: "business",
  states: [
    { name: "plan",
      state_type: "start",
      description: "Create plan with tasks and steps" },

    { name: "implement",
      state_type: "intermediate",
      description: "Execute tasks following the safe-modification protocol",
      sub_protocol_id: "<safe-modification-protocol-id>" },

    { name: "review",
      state_type: "intermediate",
      description: "Run code review protocol on all changes",
      sub_protocol_id: "<code-review-protocol-id>" },

    { name: "document",
      state_type: "intermediate",
      description: "Final documentation pass — decisions, notes, commit links" },

    { name: "complete",
      state_type: "terminal",
      description: "Feature is implemented, reviewed, and documented" }
  ],
  transitions: [
    { from_state: "plan",      to_state: "implement", trigger: "plan_approved" },
    { from_state: "implement", to_state: "review",    trigger: "child_completed" },
    { from_state: "review",    to_state: "document",  trigger: "child_completed" },
    { from_state: "review",    to_state: "implement", trigger: "child_failed",
      guard: "Review found issues — go back to implementation" },
    { from_state: "document",  to_state: "complete",  trigger: "documented" }
  ],
  relevance_vector: { phase: 0.5, structure: 0.8, domain: 0.5, resource: 0.5, lifecycle: 0.3 }
)
```

### How hierarchy works

1. When the parent transitions to `implement`, it **automatically spawns
   a child run** of the safe-modification protocol
2. The parent **pauses** on `implement` until the child reaches a terminal state
3. When the child completes → `child_completed` fires on the parent
4. When the child fails → `child_failed` fires on the parent

### Completion and failure strategies

Configure how the parent handles child outcomes:

```
# Default: wait for all children
{ name: "implement", sub_protocol_id: "...",
  completion_strategy: "all_complete" }

# Or: proceed when any child finishes
{ name: "implement", sub_protocol_id: "...",
  completion_strategy: "any_complete" }

# Failure: retry up to 2 times, then abort
{ name: "implement", sub_protocol_id: "...",
  on_failure_strategy: { retry: { max: 2 } } }

# Failure: skip the child and move on
{ name: "implement", sub_protocol_id: "...",
  on_failure_strategy: "skip" }
```

### Inspecting the hierarchy

```
# Full tree: parent + all nested children recursively
protocol(action: "get_run_tree", run_id: "<parent-run-id>")

# Direct children only
protocol(action: "get_run_children", run_id: "<parent-run-id>")
```

### Safety limits

- Maximum hierarchy depth: **5** (prevents infinite nesting)
- Cycle detection: a protocol can't appear twice in its ancestor chain
- Concurrency: max 1 running instance per protocol

---

## 7. RFC Lifecycle

RFCs are notes with `note_type: "rfc"`. Their lifecycle is managed by a
built-in protocol FSM.

### Creating an RFC

```
note(action: "create", project_id: "...",
  note_type: "rfc",
  content: "## Problem\n...\n\n## Proposed Solution\n...\n\n## Alternatives\n...\n\n## Impact\n...",
  importance: "high",
  tags: ["rfc", "api", "breaking-change"])
```

### Lifecycle

```
draft ──propose──> proposed ──accept──> accepted ──implement──> implemented
                      │
                      └──reject──> rejected
```

```
# Advance through states
note(action: "advance_rfc", note_id: "...", trigger: "propose")
note(action: "advance_rfc", note_id: "...", trigger: "accept")
note(action: "advance_rfc", note_id: "...", trigger: "implement")

# Check status
note(action: "get_rfc_status", note_id: "...")
# → { state: "accepted", protocol_run: { states_visited: [...] } }
```

### Integration with plans

When an RFC is accepted, link it to the implementation plan:

```
note(action: "link_to_entity", note_id: "<rfc-note-id>",
  entity_type: "plan", entity_id: "<plan-id>")
```

### Safety rule

**Always check active RFCs during warm-up** to avoid conflicting work:

```
note(action: "list_rfcs", project_id: "...")
```

If an RFC in `proposed` or `accepted` state overlaps with your planned
changes, coordinate before proceeding.

---

## 8. Auto-Triggered Protocols

Protocols can run automatically on events or schedules.

### Event-triggered

```
protocol(action: "compose", project_id: "...",
  name: "post-sync-enrichment",
  category: "system",
  trigger_mode: "event",
  trigger_config: { events: ["post_sync"] },
  states: [
    { name: "detect-changes",   state_type: "start",        description: "Identify what changed since last sync" },
    { name: "update-fabric",    state_type: "intermediate",  description: "Recompute fabric scores for changed files" },
    { name: "check-new-gaps",   state_type: "intermediate",  description: "Flag new knowledge gaps from new files" },
    { name: "done",             state_type: "terminal",      description: "Post-sync maintenance complete" }
  ],
  transitions: [
    { from_state: "detect-changes", to_state: "update-fabric",  trigger: "changes_detected" },
    { from_state: "update-fabric",  to_state: "check-new-gaps", trigger: "fabric_updated" },
    { from_state: "check-new-gaps", to_state: "done",           trigger: "gaps_flagged" }
  ]
)
```

Available events:
- `post_sync` — after `project(action: "sync")` or `admin(action: "sync_directory")`
- `post_import` — after `skill(action: "import")`

### Scheduled

```
protocol(action: "update", protocol_id: "...",
  trigger_mode: "scheduled",
  trigger_config: { schedule: "daily" })
```

Schedules: `hourly`, `daily`, `weekly`

### Auto (event + scheduled)

```
protocol(action: "update", protocol_id: "...",
  trigger_mode: "auto",
  trigger_config: { events: ["post_sync"], schedule: "weekly" })
```

### Cooldown

Auto-triggered protocols have a 5-minute cooldown (`MIN_TRIGGER_INTERVAL_SECS = 300`).
If a protocol was triggered less than 5 minutes ago, the new trigger is silently skipped.

---

## 9. Routing & Simulation

When you have multiple protocols, use **routing** to find the best one
for your current context.

### The 5 dimensions

Each protocol has a `relevance_vector` with 5 dimensions (all 0.0 - 1.0):

| Dimension | Meaning | 0.0 | 1.0 |
|-----------|---------|-----|-----|
| `phase` | Workflow phase | Warmup/onboarding | Closure/review |
| `structure` | Complexity | Simple task | Complex multi-file change |
| `domain` | Specificity | Domain-agnostic | Domain-specific |
| `resource` | Resources available | Constrained | Full resources |
| `lifecycle` | Project maturity | Greenfield | Mature/legacy |

### Routing

```
# Find the best protocol for the current context
protocol(action: "route", project_id: "...",
  phase: 0.5, structure: 0.8, domain: 0.5, resource: 0.5, lifecycle: 0.5)

# Or auto-build context from a plan
protocol(action: "route", project_id: "...", plan_id: "...")
```

Returns a ranked list with scores and explanations:
```
1. safe-modification: 92% — Strong match on structure (complex change)
2. feature-workflow:  85% — Good match on phase and structure
3. code-review:       45% — Phase mismatch (review, not execution)
```

### Simulation (dry-run)

Test a protocol's relevance vector without starting a run:

```
protocol(action: "simulate", protocol_id: "...",
  context: { phase: 0.5, structure: 0.9, domain: 0.3, resource: 0.5, lifecycle: 0.5 })
# → { score: 0.87, would_activate: true, dimensions: [...], explanation: "..." }
```

Use simulation to tune your `relevance_vector` until it matches
the contexts where you actually want the protocol to activate.

---

## 10. Building Your Own Protocols

### Design principles

1. **Each state = one clear phase** with specific MCP actions to perform.
   Don't create vague states like "do stuff" — name the exact tools.

2. **Add backward transitions** for failure recovery. A linear protocol
   (A→B→C) is fragile. Always include a path back:
   ```
   { from_state: "verify", to_state: "implement", trigger: "failed" }
   ```

3. **Use guards as documentation** even though they're not evaluated yet.
   They tell the agent (and future humans) *why* a transition exists:
   ```
   { guard: "pain_score > 0.4 — knowledge gaps need filling" }
   ```

4. **Set a meaningful relevance_vector** so routing can suggest your protocol
   in the right context. Use `simulate` to test it.

5. **Link notes to states** via the `notes` parameter in `compose`.
   This connects relevant knowledge directly to the protocol steps:
   ```
   notes: [
     { note_id: "<gotcha-note-id>", state_name: "implement" },
     { note_id: "<guideline-note-id>", state_name: "verify" }
   ]
   ```

6. **Keep protocols focused**. A protocol with 15 states is too complex.
   Use **hierarchical composition** instead — nest sub-protocols.

7. **System vs Business**: Use `category: "system"` for maintenance
   protocols that should run automatically. Use `category: "business"`
   for workflows the agent drives interactively.

### Template: custom protocol

```
protocol(action: "compose", project_id: "...",
  name: "<verb>-<noun>",           # e.g., "validate-migration", "audit-security"
  category: "business",            # or "system" for auto-triggered
  states: [
    { name: "<initial-action>",    state_type: "start",
      description: "<What to do — name the MCP tools>" },
    { name: "<main-work>",        state_type: "intermediate",
      description: "<Core work with specific tool calls>" },
    { name: "<verification>",     state_type: "intermediate",
      description: "<How to verify the work is correct>" },
    { name: "<success>",          state_type: "terminal",
      description: "<What 'done' looks like>" },
    { name: "<failure>",          state_type: "terminal",
      description: "<What failure looks like>" }
  ],
  transitions: [
    { from_state: "<initial>",     to_state: "<main>",     trigger: "<ready>" },
    { from_state: "<main>",        to_state: "<verify>",   trigger: "<work_done>" },
    { from_state: "<verify>",      to_state: "<success>",  trigger: "<checks_pass>" },
    { from_state: "<verify>",      to_state: "<main>",     trigger: "<checks_fail>",
      guard: "Why it failed — what to fix" },
    { from_state: "<verify>",      to_state: "<failure>",  trigger: "<unrecoverable>",
      guard: "When to give up" }
  ],
  relevance_vector: {
    phase: 0.5,        # When in the workflow? 0=start, 1=end
    structure: 0.5,    # How complex? 0=trivial, 1=very complex
    domain: 0.5,       # How domain-specific? 0.5=generic
    resource: 0.5,     # How resource-hungry? 0=light, 1=heavy
    lifecycle: 0.5     # Project maturity? 0=new, 1=legacy
  }
)
```

---

## Recommended Protocol Suite

For a production project, set up these 4 protocols:

| Protocol | Category | Trigger | Purpose |
|----------|----------|---------|---------|
| **project-onboarding** | system | manual | First-time setup, full knowledge graph bootstrap |
| **safe-modification** | business | manual | Guardrails for every non-trivial code change |
| **knowledge-maintenance** | system | weekly | Keep the fabric healthy, decay synapses, review stale notes |
| **code-review** | business | manual | Review changes before merge |

Optional additions:
| Protocol | Category | Trigger | Purpose |
|----------|----------|---------|---------|
| **feature-workflow** | business | manual | End-to-end feature development with nested sub-protocols |
| **post-sync-enrichment** | system | post_sync | Auto-update fabric after syncs |

This gives you a self-aware, self-maintaining setup where the agent always
knows what it knows, verifies before modifying, and documents what it learns.

---

## Next Steps

- [Advanced Knowledge Fabric](./advanced-knowledge-fabric.md) — Bio-inspired features (scars, homeostasis, scaffolding)
- [Multi-Agent Workflows](./multi-agent-workflow.md) — Coordinating multiple agents
- [MCP Tools Reference](../api/mcp-tools.md) — All 22 mega-tools
- [Getting Started](./getting-started.md) — Basic setup tutorial
