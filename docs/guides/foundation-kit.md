# Foundation Kit — Bootstrapping a Self-Aware Instance

A fresh Project Orchestrator instance is **blank** — no notes, no decisions,
no protocols, no understanding of what matters. This guide provides the
**foundational knowledge** to seed on day 1, giving the agent a reference
frame before it touches any code.

**Time:** ~20 minutes to bootstrap a production-ready knowledge base

---

## The Problem

Without seed knowledge, the agent:
- Doesn't know what the 22 mega-tools do or how they relate
- Doesn't know what's dangerous (e.g., aggressive synapse decay)
- Doesn't know the safe workflow (warm up → impact → implement → document)
- Doesn't know what to capture or how to structure notes
- Has no guidelines, no patterns, no gotchas — it starts from zero every session

The system prompt describes the tools, but that knowledge **disappears between
sessions**. Notes persist in the graph — they're the agent's long-term memory.

---

## How to Use This Guide

Run each code block in order. Each block creates a note and links it
appropriately. At the end, you'll have ~25 foundational notes covering:

1. **Workflow guidelines** — How to work safely
2. **Tool knowledge** — What each tool does and when to use it
3. **Safety rules** — What NOT to do
4. **Structural patterns** — How to organize knowledge
5. **Maintenance procedures** — How to keep the fabric healthy
6. **Bootstrap protocols** — Self-maintaining FSMs

> **Tip:** You can run this entire guide as a single session.
> Ask the agent: *"Bootstrap this project using the foundation kit"*
> and point it to this file.

---

## Prerequisites

Before starting, you need a project registered and synced:

```
project(action: "create", name: "my-project", root_path: "/path/to/code")
project(action: "sync", slug: "my-project")
```

Save the `project_id` — you'll use it for every note below.

---

## Phase 1 — Workflow Guidelines

These notes define **how the agent should work**. They're the most critical
because they shape every interaction.

### 1.1 The Warm-Up Rule

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Mandatory Warm-Up Before Any Work

Before starting any task, ALWAYS load existing knowledge to avoid redoing documented work or violating established conventions:

1. `note(action: \"search_semantic\", query: \"<topic you're about to work on>\")` — find related notes
2. `decision(action: \"search_semantic\", query: \"<topic>\")` — past architectural decisions
3. `note(action: \"get_context\", entity_type: \"file\", entity_id: \"<target file>\")` — notes on the file
4. `note(action: \"get_propagated\", file_path: \"<target file>\", slug: \"<project>\")` — notes from imports/co-change/AFFECTS
5. `note(action: \"list_rfcs\", project_id: \"...\")` — active RFCs to avoid conflicts
6. `decision(action: \"get_affecting\", entity_type: \"File\", entity_id: \"<target file>\")` — decisions that constrain this file

If an RFC in 'proposed' or 'accepted' state overlaps with your changes, STOP and coordinate — don't create conflicting work.

This warm-up prevents knowledge loss, redundant work, and convention violations.",
  importance: "critical",
  tags: ["workflow", "warm-up", "mandatory", "foundation"])
```

### 1.2 The Safe Modification Workflow

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Safe Modification Workflow

Every non-trivial code change MUST follow this sequence:

### 1. Gather Context (warm-up)
- Search notes and decisions semantically
- Load propagated notes via Knowledge Fabric
- Check active RFCs

### 2. Analyze Impact
- `code(action: \"analyze_impact\", target: \"<file>\")` — structural coupling (imports)
- `code(action: \"get_file_co_changers\", file_path: \"<file>\")` — temporal coupling (co-change history)
- `code(action: \"get_file_dependencies\", file_path: \"<file>\")` — who imports this? what does it import?
- Files appearing in BOTH structural AND temporal results are high-confidence impacted files

### 3. Check Topology
- `code(action: \"check_file_topology\", project_slug, file_path, new_imports: [...])` — verify new imports don't violate rules
- If violations found → rethink the approach

### 4. Check Risks
- `code(action: \"get_node_importance\", project_slug, node_path, node_type: \"File\")` — PageRank, betweenness, risk_level
- For critical/high risk files: ensure thorough test coverage, consider less invasive alternatives

### 5. Implement
- Make changes with full awareness of impact, constraints, and existing decisions

### 6. Verify
- Run tests
- Re-check topology for new violations

### 7. Document
- Create notes for patterns/gotchas discovered
- Record decisions with rationale and alternatives
- Link everything to code entities (AFFECTS, LINKED_TO)
- `chat(action: \"add_discussed\", session_id, entities)` for modified files

NEVER skip steps 1-2 for files with risk_level >= medium.",
  importance: "critical",
  tags: ["workflow", "safe-modification", "mandatory", "foundation"])
```

### 1.3 The Knowledge Capture Rule

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Mandatory Knowledge Capture

NEVER end a work session without capturing what you learned. Knowledge that stays only in the conversation context is LOST.

### When to create a note

| Situation | Note Type | Importance |
|-----------|-----------|------------|
| Found a bug root cause | gotcha | high |
| Discovered a code pattern | pattern | medium-high |
| Identified a convention | guideline | high |
| Found a useful trick | tip | medium |
| Observed behavior (perf, timing) | observation | low-medium |
| Made a non-trivial assertion | assertion | medium |

### How to make notes useful

1. **Always link to code**: `note(action: \"link_to_entity\", note_id, entity_type: \"file\", entity_id: \"<absolute_path>\")` — an unlinked note is nearly useless
2. **Use tags consistently**: use lowercase, hyphenated tags that describe the domain (e.g., 'error-handling', 'neo4j', 'api-design')
3. **Set importance accurately**: critical = must-know, high = should-know, medium = useful context, low = nice to have
4. **Be specific**: 'Neo4j returns BoltNull for optional rels — use OPTIONAL MATCH' is better than 'Be careful with null values'

### Architectural decisions

For every non-trivial choice:
```
decision(action: \"add\", task_id, description, rationale, alternatives, chosen_option)
decision(action: \"add_affects\", decision_id, entity_type: \"File\", entity_id: \"<absolute_path>\")
```

Always document:
- What alternatives were considered
- Why the chosen option was picked
- What files are affected (AFFECTS links)",
  importance: "critical",
  tags: ["workflow", "knowledge-capture", "mandatory", "foundation"])
```

### 1.4 The RFC Awareness Rule

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## RFC Awareness — Check Before You Build

RFCs (note_type: 'rfc') represent architectural proposals with a formal lifecycle: draft → proposed → accepted → implemented (or rejected).

### Before starting ANY significant work:
```
note(action: \"list_rfcs\", project_id: \"...\")
```

### If an RFC overlaps with your planned changes:
- **draft**: The author is still writing it — reach out before starting parallel work
- **proposed**: Under review — DO NOT implement conflicting changes, wait for resolution
- **accepted**: Approved — your implementation MUST align with it, or propose superseding it
- **implemented**: Done — your changes should be consistent with the decision made

### When to create an RFC:
- Architectural changes affecting multiple components or files
- New patterns that change how the team works
- API redesigns, data model migrations, breaking changes
- Any decision that benefits from structured review

### RFC content (required sections):
- **Problem**: What issue or need does this address?
- **Proposed Solution**: Detailed approach with implementation outline
- **Alternatives**: Other options and why they were rejected
- **Impact**: Affected files, components, breaking changes, migration path

### Advancing an RFC:
```
note(action: \"advance_rfc\", note_id, trigger: \"propose\")  # → proposed
note(action: \"advance_rfc\", note_id, trigger: \"accept\")   # → accepted
note(action: \"advance_rfc\", note_id, trigger: \"implement\") # → implemented
```",
  importance: "high",
  tags: ["workflow", "rfc", "coordination", "foundation"])
```

---

## Phase 2 — Tool Knowledge

These notes explain what tools to use and when. They help the agent make
the right choice instead of guessing.

### 2.1 Search Strategy

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Search Strategy — Which Tool for Which Question

### Hierarchy (most recommended → least recommended):

| Need | Tool | Why |
|------|------|-----|
| Find code by concept | `code(action: \"search\")` or `code(action: \"search_project\")` | MeiliSearch semantic search, cross-file, ranked |
| Find where X is used | `code(action: \"find_references\", symbol)` | Graph-based, understands imports/exports |
| Understand a flow | `code(action: \"get_call_graph\", function)` | Who calls this? What does it call? |
| Before modifying | `code(action: \"analyze_impact\", target)` | Structural + AFFECTS decisions |
| Overview of project | `code(action: \"get_architecture\")` | Most connected files, language stats |
| Symbols in a file | `code(action: \"get_file_symbols\", file_path)` | Functions, structs, traits, enums |
| Type system navigation | `code(action: \"find_trait_implementations\")` / `find_type_traits` / `get_impl_blocks` | |
| Find a note by concept | `note(action: \"search_semantic\", query)` | Vector similarity, finds conceptually close notes |
| Find a note by keyword | `note(action: \"search\", query)` | BM25, better for exact identifiers |
| Exact literal string | Grep / Read (Claude Code built-ins) | LAST RESORT only |

### Key rules:
- **ALWAYS use MCP search first** — only fall back to Grep for exact literal strings
- `search_semantic` (vector) finds conceptually related results even without keyword matches — use it for natural language queries
- `search` (BM25) is better for function names, identifiers, exact terms
- `get_file_co_changers` complements `analyze_impact` — temporal coupling vs structural coupling",
  importance: "high",
  tags: ["tools", "search", "strategy", "foundation"])
```

### 2.2 Code Exploration Toolkit

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Code Exploration — Understanding the Codebase

### Before exploring code, ALWAYS check if the project is synced:
The `last_synced` field on the project tells you when Tree-sitter last parsed the code. If absent, run `project(action: \"sync\", slug)` first.

### Structural analysis (use when you need the big picture):
- `code(action: \"get_architecture\")` — most connected files, language stats, project structure
- `code(action: \"get_communities\")` — Louvain clusters of coupled files/functions. Each community = a functional module. Use BEFORE refactoring to understand boundaries.
- `code(action: \"get_entry_points\")` — main functions, HTTP handlers, CLI commands, event handlers
- `code(action: \"list_processes\")` — business processes (BFS traces from entry points)

### Risk assessment (use before modifying critical code):
- `code(action: \"get_health\")` — hotspots, god functions, knowledge gaps, circular deps, neural metrics
- `code(action: \"get_node_importance\", node_path, node_type: \"File\")` — PageRank, betweenness, bridge detection, risk_level
- `code(action: \"get_risk_assessment\")` — composite risk score per file

### Inheritance (OOP languages — ALWAYS check before modifying public/protected methods):
- `code(action: \"get_class_hierarchy\", type_name)` — parents AND children
- `code(action: \"find_subclasses\", class_name)` — all transitive subclasses
- `code(action: \"find_interface_implementors\", interface_name)` — all implementations

A signature change in a parent class can silently break N subclasses.",
  importance: "high",
  tags: ["tools", "code-exploration", "foundation"])
```

### 2.3 Knowledge Fabric — What It Is and How to Feed It

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "pattern",
  content: "## Knowledge Fabric — The Agent's Long-Term Memory

The Knowledge Fabric is a multi-layer graph connecting all entities via semantic relations. It's what makes the agent 'remember' across sessions.

### The 5 relation layers:

| Layer | Type | How it's created |
|-------|------|-----------------|
| IMPORTS | File → File | Automatic (Tree-sitter sync) |
| TOUCHES | Commit → File | `commit(action: \"create\", files_changed: [...], project_id)` |
| CO_CHANGED | File ↔ File | Computed from TOUCHES — files that frequently change together |
| AFFECTS | Decision → File/Function | `decision(action: \"add_affects\", decision_id, entity_type, entity_id)` |
| SYNAPSE | Note ↔ Note | Neural connections — reinforced by co-activation, decay over time |
| LINKED_TO | Note → File/Function/Struct | `note(action: \"link_to_entity\", note_id, entity_type, entity_id)` |

### How to feed the fabric (the 3 essential inputs):

1. **Link notes to code** — every note should have at least one LINKED_TO relation
2. **Link decisions to files** — every decision should have AFFECTS relations to impacted files
3. **Register commits with files_changed** — enables CO_CHANGED detection and churn scoring

### What the fabric gives back:

- `note(action: \"get_propagated\", file_path, slug)` — notes from imports + co-change + AFFECTS
- `note(action: \"get_context\", entity_type, entity_id)` — direct contextual notes
- `code(action: \"analyze_impact\")` — includes AFFECTS decisions on impacted files
- `admin(action: \"search_neurons\")` — neural search follows synapse connections

### Without these inputs, the fabric is EMPTY and all propagation/context queries return nothing.",
  importance: "critical",
  tags: ["knowledge-fabric", "architecture", "foundation"])
```

---

## Phase 3 — Safety Rules

These notes prevent the agent from doing damage.

### 3.1 Synapse Safety

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "gotcha",
  content: "## DANGER: Synapse Decay Can Destroy the Network

### The rule:
NEVER use `decay_amount > 0.1` in a single pass of `admin(action: \"decay_synapses\")`.

### Why:
Decay is applied multiplicatively to ALL synapses. A value of 0.2 removes 20% of every synapse weight in one pass. After 5 passes at 0.2, a synapse at weight 1.0 drops to 0.33. At 0.5, three passes kills everything.

### Safe procedure:
```
# 1. Check current health
code(action: \"get_health\", project_slug: \"...\")
# Look at synapse_health — healthy range is 0.2 to 3.0

# 2. Apply GENTLE decay
admin(action: \"decay_synapses\", decay_amount: 0.03, prune_threshold: 0.1)

# 3. Re-check health
code(action: \"get_health\", project_slug: \"...\")

# 4. If synapse_health is still > 3.0, repeat with another gentle pass
```

### If the network is destroyed (synapse_health ≈ 0):
```
admin(action: \"backfill_synapses\")  # Recreate from LINKED_TO relations
admin(action: \"reinforce_neurons\", note_ids: [...], energy_boost: 0.3, synapse_boost: 0.4)
```

Knowledge scars decay 20x slower than synapses — they're nearly permanent.",
  importance: "critical",
  tags: ["safety", "synapses", "knowledge-fabric", "foundation"])
```

### 3.2 Entity ID Paths

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "gotcha",
  content: "## Entity IDs Must Be ABSOLUTE Paths

When linking notes or decisions to code entities (files, functions, structs), the entity_id MUST match the ABSOLUTE path stored in Neo4j by Tree-sitter sync.

### Correct:
```
note(action: \"link_to_entity\", note_id: \"...\",
  entity_type: \"file\",
  entity_id: \"/Users/me/projects/my-app/src/api/handlers.rs\")
```

### Wrong (silently ignored):
```
note(action: \"link_to_entity\", note_id: \"...\",
  entity_type: \"file\",
  entity_id: \"src/api/handlers.rs\")  # RELATIVE — won't match any node!
```

### How to find the correct path:
- `code(action: \"get_file_symbols\", file_path: \"src/api/handlers.rs\")` — the response shows the absolute path
- `code(action: \"search\", query: \"handlers\")` — results include full paths
- The project's `root_path` + relative path = absolute path

This applies to ALL entity linking: notes, decisions (AFFECTS), discussed entities.",
  importance: "critical",
  tags: ["safety", "entity-paths", "gotcha", "foundation"])
```

### 3.3 Commit Registration

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Always Register Commits with files_changed + project_id

When creating a commit record, ALWAYS include both `files_changed` and `project_id`:

### Correct:
```
commit(action: \"create\",
  sha: \"abc123\",
  message: \"feat(api): add user endpoint\",
  author: \"agent\",
  files_changed: [\"src/api/user.rs\", \"src/api/routes.rs\"],
  project_id: \"<uuid>\")
```

### Why both matter:
- `files_changed` → creates TOUCHES relations (Commit → File), enables CO_CHANGED detection and churn scoring
- `project_id` → triggers automatic incremental re-sync of modified files (no need for full `project(action: \"sync\")` after each commit)

### Without files_changed:
- No TOUCHES relations → no CO_CHANGED detection → no temporal coupling insights
- `get_file_co_changers` returns empty results
- Churn scores stay at zero → risk assessment is blind to change frequency

### Without project_id:
- No automatic re-sync → the knowledge graph becomes stale after code changes
- You'd have to manually call `project(action: \"sync\")` after each commit",
  importance: "high",
  tags: ["workflow", "commits", "knowledge-fabric", "foundation"])
```

---

## Phase 4 — Structural Patterns

These notes explain how to organize and structure knowledge.

### 4.1 Note Types and When to Use Them

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Note Types — Choosing the Right One

| Type | Purpose | Example | When |
|------|---------|---------|------|
| `guideline` | Rule to follow | 'Always use ULID for IDs' | When you identify a convention that should be respected |
| `gotcha` | Trap to avoid | 'Neo4j BoltNull breaks unwrap()' | After finding a bug root cause or a non-obvious behavior |
| `pattern` | Recurring design | 'Handlers follow extract→validate→execute→respond' | After recognizing a structural pattern in the code |
| `context` | Background info | 'This module was migrated from Python in Q1' | When historical context explains current design |
| `tip` | Useful trick | 'Use get_communities before refactoring' | After discovering an efficient workflow |
| `observation` | Measured behavior | 'Sync takes ~3min for repos > 50k LOC' | After observing performance or behavior characteristics |
| `assertion` | Claimed fact | 'This API always returns 200, never 204' | When making a testable claim about behavior |
| `rfc` | Architectural proposal | 'Standardize error format on RFC 7807' | When proposing a change that needs review |

### Importance levels:
- **critical**: Must-know — violating this causes bugs or data loss
- **high**: Should-know — violating this causes technical debt or inconsistency
- **medium**: Useful context that improves quality
- **low**: Nice-to-have, supplementary information

### Tags:
Use lowercase, hyphenated, descriptive tags. Consistent tagging enables effective search.
Examples: 'neo4j', 'api-design', 'error-handling', 'performance', 'security', 'convention'",
  importance: "high",
  tags: ["structure", "note-types", "foundation"])
```

### 4.2 Project-Scoped vs Global Notes

```
note(action: "create",
  note_type: "guideline",
  content: "## Project-Scoped vs Global Notes

### Project-scoped (with project_id):
Notes specific to ONE project — a gotcha in its API, a pattern in its codebase, a convention unique to it.

Example: 'This project's Neo4j client returns BoltNull for optional rels' → project-scoped

### Global (without project_id):
Cross-project conventions, workspace-wide patterns, universal best practices.

Example: 'Always use ULID for entity IDs across all projects' → global

### How to list each:
- `note(action: \"list_project\", slug: \"my-project\")` — project-specific notes
- `note(action: \"list\")` — includes global notes
- Global notes propagate to ALL projects during context queries

### Rule of thumb:
- If the note mentions a specific file path or function → project-scoped
- If the note is about a general principle or tooling pattern → global
- When in doubt, make it project-scoped — you can always promote it later",
  importance: "medium",
  tags: ["structure", "notes", "scope", "foundation"])
```

### 4.3 Decision Documentation Pattern

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "pattern",
  content: "## Decision Documentation Pattern

Every non-trivial architectural choice should be recorded as a Decision with AFFECTS links.

### The complete pattern:

```
# 1. Record the decision
decision(action: \"add\", task_id: \"...\",
  description: \"Use Axum over Actix-web for HTTP framework\",
  rationale: \"Better tokio integration, simpler middleware, growing ecosystem\",
  alternatives: [\"actix-web: mature but actor model adds complexity\", \"warp: filter-based API harder to read\"],
  chosen_option: \"axum\")
# → decision_id

# 2. Link to ALL affected files
decision(action: \"add_affects\", decision_id: \"...\",
  entity_type: \"File\",
  entity_id: \"/absolute/path/to/src/api/server.rs\",
  impact_description: \"Main HTTP server setup — framework choice defines all handler signatures\")

decision(action: \"add_affects\", decision_id: \"...\",
  entity_type: \"File\",
  entity_id: \"/absolute/path/to/src/api/handlers.rs\",
  impact_description: \"All handlers use Axum extractors and response types\")
```

### Why AFFECTS links matter:
- `code(action: \"analyze_impact\")` includes AFFECTS decisions on impacted files
- `note(action: \"get_propagated\")` traverses AFFECTS to surface related decisions
- `decision(action: \"get_affecting\", entity_type: \"File\", entity_id)` finds decisions constraining a file
- Without AFFECTS, decisions are orphaned — they exist but never surface in context

### When a decision is superseded:
```
decision(action: \"supersede\", decision_id: \"old\", superseded_by_id: \"new\")
```
The old decision is marked deprecated, the new one takes over.",
  importance: "high",
  tags: ["structure", "decisions", "pattern", "foundation"])
```

---

## Phase 5 — Maintenance Procedures

### 5.1 Knowledge Fabric Bootstrap Procedure

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Knowledge Fabric Bootstrap — When and How

### When to bootstrap:
- First time setting up a project (after sync)
- After a major refactoring that restructured the codebase
- When GDS metrics return empty results (communities, health, risk assessment)

### The procedure:
```
# 1. Ensure project is synced (creates File, Function, Struct, imports nodes)
project(action: \"sync\", slug: \"my-project\")

# 2. Bootstrap the full fabric
admin(action: \"bootstrap_knowledge_fabric\", project_id: \"...\")
```

### What bootstrap does:
1. Computes CO_CHANGED relations from commit history (TOUCHES)
2. Runs GDS algorithms: PageRank, betweenness centrality, Louvain communities
3. Computes fabric fusion scores (multi-layer: IMPORTS + CO_CHANGED + AFFECTS + SYNAPSE)
4. Calculates note staleness and energy scores
5. Backfills synapses from LINKED_TO relations

### After bootstrap, verify:
```
code(action: \"get_health\", project_slug: \"my-project\")
# Check: communities detected? hotspots identified? synapse_health > 0?
```

### If health shows issues:
- `note_density < 0.3` → create more notes linked to files (Phase 1-4 of this guide)
- `decision_coverage < 10%` → record architectural decisions with AFFECTS
- `synapse_health = 0` → `admin(action: \"backfill_synapses\")` then reinforce key clusters",
  importance: "high",
  tags: ["maintenance", "knowledge-fabric", "bootstrap", "foundation"])
```

### 5.2 Weekly Maintenance Checklist

```
note(action: "create", project_id: "<PROJECT_ID>",
  note_type: "guideline",
  content: "## Weekly Knowledge Maintenance Checklist

Run this sequence weekly (or after significant changes) to keep the knowledge fabric healthy.

### Step 1 — Audit gaps
```
admin(action: \"audit_gaps\", project_id: \"...\")
```
Fix: link orphan notes to entities, add AFFECTS to decisions, register missing commits.

### Step 2 — Health check
```
code(action: \"get_health\", project_slug: \"...\")
```
Look for: pain_score, synapse_health, scar_load, dead_notes %.

### Step 3 — Gentle synapse decay (if synapse_health > 3.0)
```
admin(action: \"decay_synapses\", decay_amount: 0.03, prune_threshold: 0.1)
```
NEVER > 0.1 in one pass. Run multiple gentle passes if needed.

### Step 4 — Update scores
```
admin(action: \"update_staleness_scores\", project_id: \"...\")
admin(action: \"update_energy_scores\", project_id: \"...\")
admin(action: \"update_fabric_scores\", project_id: \"...\")
```

### Step 5 — Review stale notes
```
note(action: \"get_needing_review\", project_id: \"...\")
```
For each: confirm (still valid), invalidate (outdated), or supersede (replaced).

### Step 6 — Persist health report
```
admin(action: \"persist_health_report\", project_id: \"...\")
```
Creates a note with delta vs. previous report — tracks health trends over time.

### Automate with a protocol:
See the knowledge-maintenance protocol in the Protocols guide for an FSM version of this checklist.",
  importance: "high",
  tags: ["maintenance", "weekly", "checklist", "foundation"])
```

---

## Phase 6 — Bootstrap Protocols

After creating all the notes above, set up the foundational protocols.

### 6.1 Create the Safe Modification Protocol

```
protocol(action: "compose", project_id: "<PROJECT_ID>",
  name: "safe-modification",
  category: "business",
  states: [
    { name: "gather-context",  state_type: "start",
      description: "Load notes, decisions, propagated context, active RFCs via warm-up procedure" },
    { name: "analyze-impact",  state_type: "intermediate",
      description: "analyze_impact + get_file_co_changers + get_file_dependencies on target files" },
    { name: "check-topology",  state_type: "intermediate",
      description: "check_file_topology — verify new imports don't violate rules" },
    { name: "check-risks",     state_type: "intermediate",
      description: "get_node_importance — PageRank, betweenness, risk_level assessment" },
    { name: "implement",       state_type: "intermediate",
      description: "Make changes with full context awareness" },
    { name: "verify",          state_type: "intermediate",
      description: "Run tests + re-check topology" },
    { name: "document",        state_type: "intermediate",
      description: "Create notes, record decisions, link AFFECTS, add_discussed" },
    { name: "done",            state_type: "terminal",
      description: "Changes are safe, tested, and documented" }
  ],
  transitions: [
    { from_state: "gather-context",  to_state: "analyze-impact",  trigger: "context_loaded" },
    { from_state: "analyze-impact",  to_state: "check-topology",  trigger: "impact_assessed" },
    { from_state: "check-topology",  to_state: "check-risks",     trigger: "topology_ok" },
    { from_state: "check-topology",  to_state: "gather-context",  trigger: "topology_violation",
      guard: "New imports violate architectural rules — rethink the approach" },
    { from_state: "check-risks",     to_state: "implement",       trigger: "risks_acceptable" },
    { from_state: "check-risks",     to_state: "gather-context",  trigger: "high_risk",
      guard: "Critical risk score — need a different approach or more context" },
    { from_state: "implement",       to_state: "verify",          trigger: "changes_made" },
    { from_state: "verify",          to_state: "document",        trigger: "verification_passed" },
    { from_state: "verify",          to_state: "implement",       trigger: "verification_failed" },
    { from_state: "document",        to_state: "done",            trigger: "documented" }
  ],
  relevance_vector: { phase: 0.5, structure: 0.7, domain: 0.5, resource: 0.5, lifecycle: 0.5 }
)
```

### 6.2 Create the Knowledge Maintenance Protocol

```
protocol(action: "compose", project_id: "<PROJECT_ID>",
  name: "knowledge-maintenance",
  category: "system",
  states: [
    { name: "audit",           state_type: "start",
      description: "audit_gaps — orphan notes, decisions without AFFECTS, unlinked commits" },
    { name: "health-check",    state_type: "intermediate",
      description: "get_health — hotspots, risks, homeostasis, neural metrics" },
    { name: "decay-synapses",  state_type: "intermediate",
      description: "Gentle decay (0.03, prune 0.1) — NEVER > 0.1 per pass" },
    { name: "update-scores",   state_type: "intermediate",
      description: "update_staleness + update_energy + update_fabric scores" },
    { name: "review-stale",    state_type: "intermediate",
      description: "get_needing_review — confirm, invalidate, or supersede stale notes" },
    { name: "report",          state_type: "terminal",
      description: "persist_health_report — note with delta vs. previous" }
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

---

## Phase 7 — Link Foundation Notes to Each Other

After creating all the notes, reinforce their connections so they surface
together during neural search:

```
# Gather the note IDs from phases 1-5 (you'll have ~15 notes)
# Group them thematically and reinforce:

# Workflow cluster (warm-up + safe-modification + knowledge-capture + rfc-awareness)
admin(action: "reinforce_neurons",
  note_ids: ["<warm-up-id>", "<safe-modification-id>", "<knowledge-capture-id>", "<rfc-awareness-id>"],
  energy_boost: 0.5, synapse_boost: 0.6)

# Safety cluster (synapse-safety + entity-paths + commit-registration)
admin(action: "reinforce_neurons",
  note_ids: ["<synapse-safety-id>", "<entity-paths-id>", "<commit-registration-id>"],
  energy_boost: 0.4, synapse_boost: 0.5)

# Knowledge fabric cluster (fabric-overview + bootstrap-procedure + maintenance-checklist)
admin(action: "reinforce_neurons",
  note_ids: ["<fabric-overview-id>", "<bootstrap-procedure-id>", "<maintenance-checklist-id>"],
  energy_boost: 0.5, synapse_boost: 0.6)

# Structure cluster (note-types + scoped-vs-global + decision-pattern)
admin(action: "reinforce_neurons",
  note_ids: ["<note-types-id>", "<scoped-vs-global-id>", "<decision-pattern-id>"],
  energy_boost: 0.3, synapse_boost: 0.4)
```

This creates synapses between related foundational notes. When the agent
searches for "how to create a decision", it will also surface the safety
rules about AFFECTS links and absolute paths.

---

## Verification

After completing all phases, verify the foundation is in place:

```
# 1. Count foundation notes
note(action: "search", query: "foundation")
# → Should return ~15 notes with tag "foundation"

# 2. Check fabric health
code(action: "get_health", project_slug: "my-project")
# → note_density should have improved, synapse_health > 0

# 3. Test propagation
note(action: "get_propagated", file_path: "<any_project_file>", slug: "my-project")
# → Should return relevant foundation notes

# 4. Test neural search
admin(action: "search_neurons", query: "safe modification workflow")
# → Should return the safe-modification guideline + linked notes

# 5. Test semantic search
note(action: "search_semantic", query: "how to modify code safely")
# → Should return the safe-modification workflow guideline
```

---

## What Happens Next

With the foundation in place, the agent now has:

| Capability | Without Foundation | With Foundation |
|------------|-------------------|-----------------|
| Warm-up | Skipped — no knowledge to load | Loads ~15 foundational notes + project-specific notes |
| Impact analysis | Done ad-hoc, sometimes skipped | Mandatory step in safe-modification protocol |
| Knowledge capture | Forgotten — knowledge lost between sessions | Mandatory rule — agent creates notes after every discovery |
| Safety awareness | Agent might destroy synapses or use wrong paths | Critical gotchas surface in every relevant context |
| Decision tracking | Decisions recorded but orphaned | Decisions always linked via AFFECTS to impacted files |
| Maintenance | Never done | Weekly protocol with clear checklist |

As the agent works, it creates MORE notes, MORE decisions, MORE synapses.
The foundation notes serve as the **seed crystal** — they attract and organize
all future knowledge around the patterns established here.

---

## Quick Reference: Foundation Note IDs

After running this guide, record the note IDs for reference:

| Note | ID | Tags |
|------|----|------|
| Warm-Up Rule | `...` | workflow, warm-up, mandatory |
| Safe Modification Workflow | `...` | workflow, safe-modification, mandatory |
| Knowledge Capture Rule | `...` | workflow, knowledge-capture, mandatory |
| RFC Awareness | `...` | workflow, rfc, coordination |
| Search Strategy | `...` | tools, search, strategy |
| Code Exploration Toolkit | `...` | tools, code-exploration |
| Knowledge Fabric Overview | `...` | knowledge-fabric, architecture |
| Synapse Safety | `...` | safety, synapses |
| Entity ID Paths | `...` | safety, entity-paths |
| Commit Registration | `...` | workflow, commits |
| Note Types Guide | `...` | structure, note-types |
| Scoped vs Global Notes | `...` | structure, notes, scope |
| Decision Documentation Pattern | `...` | structure, decisions, pattern |
| Fabric Bootstrap Procedure | `...` | maintenance, bootstrap |
| Weekly Maintenance Checklist | `...` | maintenance, weekly |

---

## Next Steps

- [Protocols Guide](./protocols.md) — Set up the full protocol suite (code review, feature workflow, etc.)
- [Advanced Knowledge Fabric](./advanced-knowledge-fabric.md) — Bio-inspired features (scars, homeostasis, scaffolding)
- [Getting Started](./getting-started.md) — Basic setup tutorial
- [MCP Tools Reference](../api/mcp-tools.md) — All 22 mega-tools
