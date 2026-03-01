# MCP Mega-Tools Reference

Complete documentation for the **20 mega-tools** exposed by Project Orchestrator.

---

## How Mega-Tools Work

Each mega-tool uses an **`action` parameter** to select the operation. This replaces the previous 145+ individual tools with a cleaner, consolidated interface.

**Example call:**
```json
{
  "tool": "project",
  "arguments": {
    "action": "sync",
    "slug": "my-project"
  }
}
```

---

## Quick Reference

| Mega-Tool | Actions | Description |
|-----------|---------|-------------|
| [`project`](#project) | 8 | Project CRUD, sync, roadmap |
| [`plan`](#plan) | 10 | Plan lifecycle, dependency graph, critical path |
| [`task`](#task) | 13 | Task CRUD, dependencies, blockers, context |
| [`step`](#step) | 6 | Step CRUD, progress tracking |
| [`decision`](#decision) | 12 | Decisions, semantic search, affects tracking |
| [`constraint`](#constraint) | 5 | Plan constraints |
| [`release`](#release) | 8 | Release management |
| [`milestone`](#milestone) | 9 | Milestones with progress |
| [`commit`](#commit) | 7 | Git commit tracking, file history |
| [`note`](#note) | 20 | Knowledge notes, semantic search, propagation |
| [`workspace`](#workspace) | 10 | Multi-project workspaces |
| [`workspace_milestone`](#workspace_milestone) | 10 | Cross-project milestones |
| [`resource`](#resource) | 6 | Shared API contracts, schemas |
| [`component`](#component) | 8 | Service topology and dependencies |
| [`chat`](#chat) | 7 | Chat sessions, messages, delegation |
| [`feature_graph`](#feature_graph) | 6 | Feature graphs, auto-build |
| [`code`](#code) | 36 | Code search, analysis, health, processes, bridge, topology firewall |
| [`admin`](#admin) | 25 | Sync, watch, Knowledge Fabric, maintenance, skills |
| [`skill`](#skill) | 12 | Neural skills detection, activation |
| [`analysis_profile`](#analysis_profile) | 4 | Edge/fusion weight presets for analysis |

---

## project

Manage projects (codebases tracked by the orchestrator).

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List all projects | `search`, `limit`, `offset`, `sort_by`, `sort_order` |
| `create` | Register a new project | `name`, `root_path`, `description` |
| `get` | Get project by slug | `slug` |
| `update` | Update project details | `slug`, `name`, `description`, `root_path` |
| `delete` | Delete project and all data | `slug` |
| `sync` | Parse and index codebase | `slug` |
| `get_roadmap` | Aggregated roadmap view | `slug` |
| `list_plans` | List plans for a project | `slug` |

---

## plan

Manage development plans with tasks and constraints.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List plans with filters | `project_id`, `search`, `status`, `priority_min`, `priority_max`, `limit`, `offset` |
| `create` | Create a new plan | `title`, `description`, `priority`, `project_id` |
| `get` | Get plan with tasks and constraints | `plan_id` |
| `update_status` | Update plan status | `plan_id`, `status` (draft/approved/in_progress/completed/cancelled) |
| `delete` | Delete plan and all data | `plan_id` |
| `link_to_project` | Associate plan with project | `plan_id`, `project_id` |
| `unlink_from_project` | Dissociate from project | `plan_id`, `project_id` |
| `get_dependency_graph` | Task dependency DAG | `plan_id` |
| `get_critical_path` | Longest dependency chain | `plan_id` |
| `list_plans` | List all plans | (same as `list`) |

---

## task

Manage tasks within plans.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List tasks with filters | `plan_id`, `search`, `status`, `tags`, `assigned_to`, `limit`, `offset` |
| `create` | Create a task | `plan_id`, `title`, `description`, `priority`, `tags`, `acceptance_criteria`, `affected_files` |
| `get` | Get task details | `task_id` |
| `update` | Update task fields | `task_id`, `status`, `assigned_to`, `priority`, `tags` |
| `delete` | Delete task | `task_id` |
| `get_next` | Next unblocked task (highest priority) | `plan_id` |
| `add_dependencies` | Add task dependencies | `task_id`, `dependency_ids` |
| `remove_dependency` | Remove a dependency | `task_id`, `depends_on_task_id` |
| `get_blockers` | Tasks blocking this one | `task_id` |
| `get_blocked_by` | Tasks blocked by this one | `task_id` |
| `get_context` | Full context for agent work | `plan_id`, `task_id` |
| `get_prompt` | Generated agent prompt | `plan_id`, `task_id` |

---

## step

Manage steps (sub-tasks) within tasks.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List steps for a task | `task_id` |
| `create` | Add step to task | `task_id`, `description`, `verification` |
| `update` | Update step status | `step_id`, `status` (pending/in_progress/completed/skipped) |
| `get` | Get step by ID | `step_id` |
| `delete` | Delete step | `step_id` |
| `get_progress` | Step completion progress | `task_id` |

---

## decision

Manage architectural decisions with semantic search and impact tracking.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `add` | Record a decision | `task_id`, `description`, `rationale`, `alternatives`, `chosen_option` |
| `get` | Get decision by ID | `decision_id` |
| `update` | Update decision | `decision_id`, `description`, `chosen_option`, `status` |
| `delete` | Delete decision | `decision_id` |
| `search` | BM25 keyword search | `query` |
| `search_semantic` | Vector similarity search | `query`, `project_id` |
| `add_affects` | Link decision to impacted entity | `decision_id`, `entity_type`, `entity_id`, `impact_description` |
| `remove_affects` | Remove impact link | `decision_id`, `entity_type`, `entity_id` |
| `list_affects` | List impacted entities | `decision_id` |
| `get_affecting` | Decisions affecting an entity | `entity_type`, `entity_id` |
| `supersede` | Replace with new decision | `decision_id`, `superseded_by_id` |
| `get_timeline` | Decision timeline | `task_id`, `from`, `to` |

---

## constraint

Manage plan constraints.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List constraints for a plan | `plan_id` |
| `add` | Add constraint | `plan_id`, `constraint_type` (performance/security/style/compatibility/other), `description`, `severity` (must/should/nice_to_have) |
| `get` | Get constraint by ID | `constraint_id` |
| `update` | Update constraint | `constraint_id`, `description`, `constraint_type`, `enforced_by` |
| `delete` | Delete constraint | `constraint_id` |

---

## release

Manage releases with tasks and commits.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List project releases | `project_id` |
| `create` | Create release | `project_id`, `version`, `title`, `description`, `target_date` |
| `get` | Get release with tasks/commits | `release_id` |
| `update` | Update release | `release_id`, `status` (planned/in_progress/released/cancelled), `title`, `description`, `target_date` |
| `delete` | Delete release | `release_id` |
| `add_task` | Add task to release | `release_id`, `task_id` |
| `add_commit` | Add commit to release | `release_id`, `commit_sha` |
| `remove_commit` | Remove commit from release | `release_id`, `commit_sha` |

---

## milestone

Manage project milestones.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List project milestones | `project_id` |
| `create` | Create milestone | `project_id`, `title`, `description`, `target_date` |
| `get` | Get milestone with tasks | `milestone_id`, `include_tasks` |
| `update` | Update milestone | `milestone_id`, `title`, `status`, `target_date` |
| `delete` | Delete milestone | `milestone_id` |
| `get_progress` | Completion percentage | `milestone_id` |
| `add_task` | Add task to milestone | `milestone_id`, `task_id` |
| `link_plan` | Link plan to milestone | `milestone_id`, `plan_id` |
| `unlink_plan` | Unlink plan from milestone | `milestone_id`, `plan_id` |

---

## commit

Track git commits and link to tasks/plans.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `create` | Register a commit | `sha`, `message`, `author`, `files_changed`, `project_id` |
| `link_to_task` | Link commit → task | `task_id`, `commit_sha` |
| `link_to_plan` | Link commit → plan | `plan_id`, `commit_sha` |
| `get_task_commits` | Commits for a task | `task_id` |
| `get_plan_commits` | Commits for a plan | `plan_id` |
| `get_commit_files` | Files changed in commit | `sha` |
| `get_file_history` | Commit history for a file | `file_path`, `limit` |

---

## note

Manage knowledge notes with semantic search and graph propagation.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List notes with filters | `project_id`, `status`, `note_type`, `importance`, `limit`, `offset` |
| `create` | Create a note | `project_id`, `note_type`, `content`, `importance`, `tags` |
| `get` | Get note by ID | `note_id` |
| `update` | Update note | `note_id`, `content`, `importance`, `status`, `tags` |
| `delete` | Delete note | `note_id` |
| `search` | BM25 keyword search | `query` |
| `search_semantic` | Vector similarity search | `query`, `project_id` |
| `confirm` | Confirm note validity (reset staleness) | `note_id` |
| `invalidate` | Mark note as obsolete | `note_id` |
| `supersede` | Replace with new note | `note_id`, `superseded_by_id` |
| `link_to_entity` | Link note to entity | `note_id`, `entity_type`, `entity_id` |
| `unlink_from_entity` | Remove link | `note_id`, `entity_type`, `entity_id` |
| `get_context` | Contextual notes for entity | `entity_type`, `entity_id` |
| `get_needing_review` | Stale/needs_review notes | — |
| `list_project` | Notes for a project | `slug` |
| `get_propagated` | Notes propagated via graph | `slug`, `file_path` |
| `get_entity` | Notes directly on entity | `entity_type`, `entity_id` |
| `get_context_knowledge` | Context knowledge | `entity_type`, `entity_id` |
| `get_propagated_knowledge` | Propagated knowledge | `entity_type`, `entity_id` |

**Note Types:** `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion`

**Note Status:** `active`, `needs_review`, `stale`, `obsolete`, `archived`

**Importance Levels:** `critical`, `high`, `medium`, `low`

---

## workspace

Manage multi-project workspaces.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List workspaces | `limit`, `offset` |
| `create` | Create workspace | `name`, `description` |
| `get` | Get workspace by slug | `slug` |
| `update` | Update workspace | `slug`, `name`, `description` |
| `delete` | Delete workspace | `slug` |
| `get_overview` | Overview with projects, milestones, resources | `slug` |
| `list_projects` | Projects in workspace | `slug` |
| `add_project` | Add project to workspace | `slug`, `project_id`, `role` |
| `remove_project` | Remove project | `slug`, `project_id` |
| `get_topology` | Component topology graph | `slug` |

---

## workspace_milestone

Manage cross-project milestones.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list_all` | All workspace milestones | `workspace_id` |
| `list` | Milestones for a workspace | `slug`, `status` |
| `create` | Create workspace milestone | `slug`, `title`, `description`, `target_date` |
| `get` | Get milestone details | `milestone_id` |
| `update` | Update milestone | `milestone_id`, `title`, `status`, `target_date` |
| `delete` | Delete milestone | `milestone_id` |
| `add_task` | Add task from any project | `milestone_id`, `task_id` |
| `link_plan` | Link plan to milestone | `milestone_id`, `plan_id` |
| `unlink_plan` | Unlink plan | `milestone_id`, `plan_id` |
| `get_progress` | Completion percentage | `milestone_id` |

---

## resource

Manage shared resources (API contracts, schemas).

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List workspace resources | `slug` |
| `create` | Create resource | `slug`, `name`, `resource_type`, `description`, `file_path`, `url`, `version` |
| `get` | Get resource details | `id` |
| `update` | Update resource | `id`, `name`, `description`, `file_path`, `url`, `version` |
| `delete` | Delete resource | `id` |
| `link_to_project` | Link resource to project | `resource_id`, `project_id` |

**Resource Types:** `api_contract`, `schema`, `config`, `documentation`, `other`

---

## component

Manage workspace components and service topology.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List workspace components | `slug` |
| `create` | Create component | `slug`, `name`, `component_type`, `description`, `runtime`, `tags` |
| `get` | Get component details | `id` |
| `update` | Update component | `id`, `name`, `description`, `runtime`, `tags` |
| `delete` | Delete component | `id` |
| `add_dependency` | Add component dependency | `from_id`, `to_id`, `dependency_type` |
| `remove_dependency` | Remove dependency | `from_id`, `to_id` |
| `map_to_project` | Map component to project | `component_id`, `project_id` |

**Component Types:** `service`, `library`, `database`, `queue`, `external`

---

## chat

Manage chat sessions and messages.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list_sessions` | List chat sessions | `project_slug`, `limit`, `offset` |
| `get_session` | Get session details | `session_id` |
| `delete_session` | Delete session | `session_id` |
| `send_message` | Send message and wait for response | `message`, `cwd`, `project_slug`, `workspace_slug`, `model` |
| `list_messages` | List message history | `session_id`, `limit`, `offset` |
| `add_discussed` | Mark entities as discussed | `session_id`, `entities` |
| `get_session_entities` | Get entities discussed in session | `session_id`, `project_id` |

---

## feature_graph

Manage feature graphs for code analysis.

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `create` | Create feature graph | `project_id`, `name`, `description` |
| `get` | Get feature graph | `id` |
| `list` | List feature graphs | `project_id` |
| `add_entity` | Add entity to graph | `feature_graph_id`, `entity_id`, `entity_type`, `role` |
| `auto_build` | Auto-build from code analysis | `project_id`, `name`, `description`, `entry_function`, `depth`, `include_relations`, `filter_community` |
| `delete` | Delete feature graph | `id` |

---

## code

Code exploration, search, and analytics (36 actions).

### Search & Navigation

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `search` | Semantic code search | `query`, `path_prefix`, `limit` |
| `search_project` | Search within a project | `slug`, `query`, `limit` |
| `search_workspace` | Search across workspace | `workspace_slug`, `query`, `limit` |
| `get_file_symbols` | Functions, structs, traits in file | `file_path` |
| `find_references` | All usages of a symbol | `symbol` |
| `get_file_dependencies` | File imports and dependents | `file_path` |
| `get_call_graph` | Function call graph | `function`, `limit` |
| `analyze_impact` | Change impact analysis | `target` |
| `get_architecture` | Codebase overview | `project_slug` |
| `find_similar` | Find similar code snippets | `code_snippet` |

### Type System & Heritage

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `find_trait_implementations` | Types implementing a trait | `trait_name` |
| `find_type_traits` | Traits implemented by a type | `type_name` |
| `get_impl_blocks` | All impl blocks for a type | `type_name` |
| `get_class_hierarchy` | Full class hierarchy (parents + children) | `type_name`, `max_depth` |
| `find_subclasses` | All transitive subclasses | `class_name` |
| `find_interface_implementors` | All interface implementors | `interface_name` |

### Process Detection

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `get_entry_points` | Entry points (main, handlers, CLI) | `project_slug` |
| `list_processes` | All detected business processes | `project_slug` |
| `get_process` | Steps of a process | `process_id` |
| `detect_processes` | Run process detection | `project_slug` |

### Analytics & Health (GDS)

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `get_communities` | Louvain community clusters | `project_slug`, `min_size` |
| `enrich_communities` | Enrich community labels via LLM | `project_slug` |
| `get_health` | Full health report (hotspots, gaps, risk, neural) | `project_slug` |
| `get_node_importance` | PageRank, betweenness, bridge detection | `project_slug`, `node_path`, `node_type` |
| `plan_implementation` | AI-assisted implementation planning | `project_slug`, `description`, `entry_points`, `scope` |
| `get_co_change_graph` | Files that change together | `project_slug` |
| `get_file_co_changers` | Co-changers for a specific file | `file_path` |
| `get_hotspots` | High-churn files | `project_slug` |
| `get_knowledge_gaps` | Under-documented files | `project_slug` |
| `get_risk_assessment` | Composite risk scores | `project_slug` |

### Bridge Subgraph & Topology Firewall

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `get_bridge` | Bridge subgraph between two nodes (bottlenecks, bridge score) | `project_slug`, `source`, `target`, `max_hops`, `top_bottlenecks` |
| `check_topology` | Check all topology rule violations | `project_slug` |
| `list_topology_rules` | List topology rules | `project_slug` |
| `create_topology_rule` | Create topology rule | `project_slug`, `rule_type` (must_not_import/must_not_call/max_distance/max_fan_out/no_circular), `source_pattern`, `target_pattern`, `threshold`, `severity` (error/warning) |
| `delete_topology_rule` | Delete topology rule | `rule_id` |
| `check_file_topology` | Check if new imports would violate rules | `project_slug`, `file_path`, `new_imports` (array) |

---

## admin

Administrative operations: sync, watch, Knowledge Fabric, maintenance.

### Sync & Watch

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `sync_directory` | Manual directory sync | `path`, `project_id` |
| `start_watch` | Start file watcher | `path`, `project_id` |
| `stop_watch` | Stop file watcher | — |
| `watch_status` | Get watcher status | — |

### Meilisearch Maintenance

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `meilisearch_stats` | Code index statistics | — |
| `delete_meilisearch_orphans` | Clean orphan documents | — |

### Data Cleanup

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `cleanup_cross_project_calls` | Remove cross-project CALLS | — |
| `cleanup_builtin_calls` | Remove builtin function calls | — |
| `migrate_calls_confidence` | Migrate confidence scores | — |
| `cleanup_sync_data` | Clean stale sync data | — |

### Knowledge Fabric

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `bootstrap_knowledge_fabric` | Initialize all fabric relations | `project_id` |
| `update_fabric_scores` | Recalculate GDS multi-layer scores | `project_id` |
| `update_staleness_scores` | Recalculate note staleness | `project_id` |
| `update_energy_scores` | Recalculate neural energy | `project_id` |
| `backfill_touches` | Backfill TOUCHES from git history | `project_id` |
| `backfill_discussed` | Backfill DISCUSSED relations | — |
| `backfill_synapses` | Backfill SYNAPSE relations | — |
| `backfill_decision_embeddings` | Generate decision embeddings | — |

### Neural Maintenance

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `search_neurons` | Search neural notes | `query`, `min_strength`, `limit` |
| `reinforce_neurons` | Co-activate notes (boost synapses) | `note_ids`, `energy_boost`, `synapse_boost` |
| `decay_synapses` | Decay weak synapses | `decay_amount`, `prune_threshold` |
| `detect_skills` | Auto-detect neural skills | `project_id` |
| `install_hooks` | Install git hooks for auto-tracking | `project_id`, `cwd`, `port` |

---

## skill

Manage neural skills (emergent knowledge clusters).

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List skills | `project_id`, `limit`, `offset` |
| `create` | Create skill | `project_id`, `name`, `description`, `tags`, `trigger_patterns`, `context_template` |
| `get` | Get skill details | `skill_id` |
| `update` | Update skill | `skill_id`, `name`, `description`, `status`, `energy`, `cohesion`, `tags` |
| `delete` | Delete skill | `skill_id` |
| `get_members` | Get skill members (notes/decisions) | `skill_id` |
| `add_member` | Add member to skill | `skill_id`, `entity_id`, `entity_type` |
| `remove_member` | Remove member | `skill_id`, `entity_id`, `entity_type` |
| `activate` | Activate skill with query | `skill_id`, `query` |
| `export` | Export skill package | `skill_id`, `source_project_name` |
| `import` | Import skill package | `project_id`, `package`, `conflict_strategy` |
| `get_health` | Skill health metrics | `skill_id` |

**Skill Status:** `emerging`, `active`, `dormant`, `archived`, `imported`

**Trigger Pattern Types:** `regex`, `file_glob`, `semantic`

---

## analysis_profile

Manage analysis profiles (edge/fusion weight presets for GDS analytics).

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `list` | List analysis profiles | `project_id` |
| `create` | Create analysis profile | `name`, `project_slug`, `description`, `edge_weights` (object), `fusion_weights` (object) |
| `get` | Get analysis profile by ID | `id` |
| `delete` | Delete analysis profile | `id` |

**Edge Weights:** `{"IMPORTS": 0.7, "CALLS": 0.5, "CO_CHANGED": 0.3, ...}` — relative weight of each edge type in GDS computations.

**Fusion Weights:** `{"structural": 0.3, "temporal": 0.4, "semantic": 0.3}` — how to blend structural, temporal, and semantic signals.
