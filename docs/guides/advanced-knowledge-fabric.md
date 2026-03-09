# Advanced Knowledge Fabric

How to leverage bio-inspired features for self-regulating knowledge graphs.

---

## What is the Knowledge Fabric?

The Knowledge Fabric is a multi-layer graph that connects all entities (notes, decisions, files, commits) via semantic relations. Beyond simple linking, it provides:

- **Neural synapses** between notes (weighted connections that strengthen with co-activation)
- **Knowledge scars** on notes/decisions (negative reinforcement from failed reasoning)
- **Homeostasis monitoring** (equilibrium metrics that detect imbalances)
- **Adaptive scaffolding** (injection volume adapts to project maturity)
- **Stagnation detection** (auto-triggers maintenance when the project stalls)

---

## 1. Discovering the Health of Your Project

### Quick health check

```
code(action: "get_homeostasis", project_slug: "my-project")
```

Returns 5 equilibrium ratios:

| Ratio | What it measures | Healthy range |
|-------|-----------------|---------------|
| `note_density` | Notes per file | 0.3 – 2.0 |
| `decision_coverage` | Files with architectural decisions | 10% – 80% |
| `synapse_health` | Active synapses per note | 0.2 – 3.0 |
| `churn_balance` | Hotspots covered by notes | 30% – 100% |
| `scar_load` | Scarred nodes ratio | 0% – 15% |

The `pain_score` (0.0 – 1.0) aggregates all ratios. Below 0.3 = healthy. Above 0.6 = action needed.

### Full health report

```
code(action: "get_health", project_slug: "my-project")
```

Adds hotspots, knowledge gaps, god functions, risk assessment, circular dependencies, and neural metrics.

---

## 2. Feeding the Knowledge Graph

The fabric needs three types of "food" to stay healthy.

### 2a. Link decisions to files (AFFECTS)

Decisions document *why* something was built a certain way. AFFECTS links connect them to the files they impact.

```
decision(action: "add_affects",
  decision_id: "uuid",
  entity_type: "File",
  entity_id: "/absolute/path/to/file.rs",
  impact_description: "Scar penalty applied in propagation score formula")
```

> ⚠️ **Gotcha**: Entity IDs must be **absolute paths** matching the Neo4j File nodes (as created by Tree-sitter sync). Relative paths are silently ignored.

**When**: After every architectural decision. Target: `decision_coverage ≥ 10%`.

### 2b. Link notes to code entities (LINKED_TO)

Notes become useful when they're attached to the code they describe.

```
note(action: "link_to_entity",
  note_id: "uuid",
  entity_type: "file",
  entity_id: "/absolute/path/to/file.rs")
```

Entity types: `file`, `function`, `struct`, `trait`, `task`, `plan`, `project`.

**Effect**: Linked notes propagate along the import graph, appear in `get_context`, and develop synapses with neighboring notes.

### 2c. Register commits (TOUCHES)

```
commit(action: "create",
  sha: "abc123",
  message: "feat(chat): add biomimicry stage",
  files_changed: ["src/chat/stages/biomimicry.rs", "src/chat/manager.rs"],
  project_id: "uuid")
```

TOUCHES relations enable co-change detection and churn scoring. When `project_id` is provided, modified files are automatically re-synced.

---

## 3. Knowledge Scars (Negative Reinforcement)

### How scars work

When a reasoning path fails (`reason_feedback(outcome: "failure")`), the involved notes receive a scar (+0.2 intensity, capped at 1.0). Scars penalize notes in search results and propagation:

```
final_score = raw_score × (1 - scar_intensity × 0.5)
```

A fully scarred note (1.0) still appears, but at 50% of its normal score. This prevents knowledge extinction while deprioritizing failed paths.

### Scars are persistent

Scars decay 20× slower than synapses during maintenance cycles. A full scar takes ~2000 decay cycles to heal naturally.

### Manual healing

```
admin(action: "heal_scars", node_id: "note-or-decision-uuid")
```

Use this when a previously bad pattern becomes valid again (e.g., a library bug was fixed).

### Checking scar load

```
code(action: "get_homeostasis")  →  scar_load ratio
```

If `scar_load > 15%`, too many nodes are scarred. Consider healing or reviewing old scars.

---

## 4. Adaptive Scaffolding

The system adapts how much knowledge it injects based on project maturity.

### How it works

A competence score is computed from 4 signals:
- Task success rate (weight 0.5)
- Average frustration (weight 0.2)
- Scar density (weight 0.15)
- Homeostasis pain (weight 0.15)

### 5 scaffolding levels

| Level | Competence | max_notes | max_propagated | max_content_chars |
|-------|-----------|-----------|----------------|-------------------|
| L0 Novice | < 0.3 | 8 | 5 | 5000 |
| L1 Beginner | < 0.5 | 6 | 4 | 4000 |
| L2 Intermediate | < 0.75 | 5 | 3 | 3000 |
| L3 Advanced | < 0.9 | 4 | 2 | 2000 |
| L4 Expert | ≥ 0.9 | 3 | 1 | 1500 |

### Activation

Set the environment variable:
```bash
ENRICHMENT_SCAFFOLDING=true
```

The scaffolding level is logged in debug output and visible in enrichment metadata.

### Checking current level

```
project(action: "get_scaffolding_level", slug: "my-project")
```

---

## 5. Stagnation Detection & Auto-Maintenance

### What triggers stagnation

4 signals are monitored (≥3 active = stagnation detected):
1. No tasks completed in 48h
2. Average frustration > 0.5 across in-progress tasks
3. Note energy trending negative
4. No commits in 48h

### Automatic response

When `ENRICHMENT_BIOMIMICRY=true`, the BiomimicryStage runs at the start of every chat interaction:

1. Detects stagnation (300ms timeout)
2. If ≥3 signals active → spawns `run_daily_maintenance` in background
3. Injects health warnings into the agent's context

### Manual stagnation check

```
admin(action: "detect_stagnation", project_id: "uuid")
```

### Activation

```bash
ENRICHMENT_BIOMIMICRY=true
```

---

## 6. Neural Synapses

### What are synapses?

Weighted connections between notes that strengthen through co-activation. When two notes are frequently relevant together, their synapse grows stronger.

### Searching via neural activation

```
admin(action: "search_neurons", query: "biomimicry scaffolding", limit: 10)
```

Unlike keyword search, neural search follows synapse connections — a query about "scaffolding" can surface notes about "homeostasis" if they're strongly connected.

### Reinforcing connections

When you discover that certain notes are thematically related:

```
admin(action: "reinforce_neurons",
  note_ids: ["note-1-uuid", "note-2-uuid", "note-3-uuid"],
  energy_boost: 0.3,
  synapse_boost: 0.4)
```

### Synapse maintenance

Synapses naturally decay during maintenance cycles. If the network becomes too dense:

```
admin(action: "decay_synapses", decay_amount: 0.05, prune_threshold: 0.15)
```

> ⚠️ **Never use `decay_amount > 0.2`** in a single pass — it can destroy the entire network. Use progressive decay with homeostasis checks between passes.

If the network is dead (0 synapses):

```
admin(action: "backfill_synapses")       # Recreates from LINKED_TO
admin(action: "reinforce_neurons", ...)  # Strengthen key clusters
```

---

## 7. Propagation & Context Discovery

### How notes propagate

When you work on a file, the system finds relevant notes by traversing:
1. **Direct links** — notes LINKED_TO this file
2. **Import graph** — notes linked to files that this file imports (or that import it)
3. **Co-change patterns** — notes linked to files that historically change alongside this one
4. **AFFECTS decisions** — architectural decisions impacting this file or its neighbors

The propagation score combines: distance, importance, path PageRank, relation type weight, and **scar penalty**.

### Querying propagated knowledge

```
note(action: "get_propagated",
  file_path: "src/chat/stages/knowledge_injection.rs",
  slug: "my-project")
```

### Cross-project propagation

In workspaces with multiple projects, notes propagate across projects weighted by coupling strength (structural twins, shared skills, tag overlap). Below 0.2 coupling, propagation is suppressed.

```
note(action: "get_propagated",
  file_path: "src/shared/utils.rs",
  slug: "my-project",
  source_project_id: "other-project-uuid",
  force_cross_project: true)  # Override the 0.2 threshold
```

---

## 8. Post-Implementation Enrichment Protocol

After completing a plan or significant PR, run this sequence to maximize knowledge retention:

### Phase 1 — Capture decisions
```
decision(action: "add", task_id, description, rationale, alternatives, chosen_option)
decision(action: "update", decision_id, status: "accepted")
decision(action: "add_affects", decision_id, entity_type: "File", entity_id: "/absolute/path")
```

### Phase 2 — Link notes to code
```
note(action: "search", query: "relevant topic")
note(action: "link_to_entity", note_id, entity_type: "file", entity_id: "/absolute/path")
```

### Phase 3 — Update fabric scores
```
admin(action: "update_fabric_scores", project_id)
```

### Phase 4 — Calibrate synapses
```
code(action: "get_homeostasis")  # Check synapse_health
# If > 3.0: admin(action: "decay_synapses", decay_amount: 0.05, prune_threshold: 0.15)
# If < 0.2: admin(action: "backfill_synapses")
```

### Phase 5 — Verify
```
code(action: "get_homeostasis")               # pain_score < 0.4?
note(action: "get_propagated", file_path: ...) # Notes propagating?
admin(action: "search_neurons", query: ...)    # Neural search working?
```

---

## 9. Thermal Noise (Stochastic Exploration)

Add randomness to search results to escape local optima:

```
note(action: "search_semantic", query: "error handling", temperature: 0.3)
decision(action: "search_semantic", query: "caching strategy", temperature: 0.5)
```

| Temperature | Behavior |
|------------|----------|
| 0.0 (default) | Deterministic — identical rankings every time |
| 0.1 – 0.3 | Mild exploration — top results shuffled slightly |
| 0.3 – 0.6 | Moderate — surprising results surface occasionally |
| 0.6 – 1.0 | Maximum exploration — near-random among candidates |

**Use case**: When you're stuck on a problem and the usual search results aren't helpful, increase temperature to discover non-obvious connections.

---

## 10. Feature Flags Summary

| Feature | Env Variable | Default | Latency Impact |
|---------|-------------|---------|----------------|
| Scar penalty in ranking | *Always active* | On | None (formula change) |
| Adaptive scaffolding | `ENRICHMENT_SCAFFOLDING` | Off | +100ms (Neo4j query) |
| Biomimicry stage | `ENRICHMENT_BIOMIMICRY` | Off | +300ms (2 Neo4j queries) |
| Thermal noise | `temperature` param | 0 | None (post-processing) |

Enable both for full bio-cognitive features:
```bash
export ENRICHMENT_BIOMIMICRY=true
export ENRICHMENT_SCAFFOLDING=true
```
