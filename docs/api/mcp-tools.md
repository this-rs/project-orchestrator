# MCP Tools Reference

Complete documentation for all 137 MCP tools exposed by Project Orchestrator.

---

## Quick Reference

| Category | Count | Tools |
|----------|-------|-------|
| [Project Management](#project-management-8-tools) | 8 | `list_projects`, `create_project`, `get_project`, `update_project`, `delete_project`, `sync_project`, `get_project_roadmap`, `list_project_plans` |
| [Plan Management](#plan-management-9-tools) | 9 | `list_plans`, `create_plan`, `get_plan`, `update_plan_status`, `link_plan_to_project`, `unlink_plan_from_project`, `get_dependency_graph`, `get_critical_path`, `delete_plan` |
| [Task Management](#task-management-13-tools) | 13 | `list_tasks`, `create_task`, `get_task`, `update_task`, `delete_task`, `get_next_task`, `add_task_dependencies`, `remove_task_dependency`, `get_task_blockers`, `get_tasks_blocked_by`, `get_task_context`, `get_task_prompt`, `add_decision` |
| [Step Management](#step-management-6-tools) | 6 | `list_steps`, `create_step`, `update_step`, `get_step`, `delete_step`, `get_step_progress` |
| [Decision Management](#decision-management-4-tools) | 4 | `get_decision`, `update_decision`, `delete_decision`, `search_decisions` |
| [Constraint Management](#constraint-management-5-tools) | 5 | `list_constraints`, `add_constraint`, `get_constraint`, `update_constraint`, `delete_constraint` |
| [Commit Tracking](#commit-tracking-5-tools) | 5 | `create_commit`, `link_commit_to_task`, `link_commit_to_plan`, `get_task_commits`, `get_plan_commits` |
| [Release Management](#release-management-7-tools) | 7 | `list_releases`, `create_release`, `get_release`, `update_release`, `delete_release`, `add_task_to_release`, `add_commit_to_release` |
| [Milestone Management](#milestone-management-7-tools) | 7 | `list_milestones`, `create_milestone`, `get_milestone`, `update_milestone`, `delete_milestone`, `get_milestone_progress`, `add_task_to_milestone` |
| [Code Exploration](#code-exploration-13-tools) | 13 | `search_code`, `search_project_code`, `search_workspace_code`, `get_file_symbols`, `find_references`, `get_file_dependencies`, `get_call_graph`, `analyze_impact`, `get_architecture`, `find_similar_code`, `find_trait_implementations`, `find_type_traits`, `get_impl_blocks` |
| [Notes (Knowledge Base)](#notes-knowledge-base-17-tools) | 17 | `list_notes`, `create_note`, `get_note`, `update_note`, `delete_note`, `search_notes`, `confirm_note`, `invalidate_note`, `supersede_note`, `link_note_to_entity`, `unlink_note_from_entity`, `get_context_notes`, `get_propagated_notes`, `get_entity_notes`, `get_notes_needing_review`, `update_staleness_scores`, `list_project_notes` |
| [Sync & Watch](#sync--watch-4-tools) | 4 | `sync_directory`, `start_watch`, `stop_watch`, `watch_status` |
| [Workspace Management](#workspace-management-9-tools) | 9 | `list_workspaces`, `create_workspace`, `get_workspace`, `update_workspace`, `delete_workspace`, `get_workspace_overview`, `list_workspace_projects`, `add_project_to_workspace`, `remove_project_from_workspace` |
| [Workspace Milestones](#workspace-milestones-8-tools) | 8 | `list_all_workspace_milestones`, `list_workspace_milestones`, `create_workspace_milestone`, `get_workspace_milestone`, `update_workspace_milestone`, `delete_workspace_milestone`, `add_task_to_workspace_milestone`, `get_workspace_milestone_progress` |
| [Resources](#resources-6-tools) | 6 | `list_resources`, `create_resource`, `get_resource`, `update_resource`, `delete_resource`, `link_resource_to_project` |
| [Components & Topology](#components--topology-9-tools) | 9 | `list_components`, `create_component`, `get_component`, `update_component`, `delete_component`, `add_component_dependency`, `remove_component_dependency`, `map_component_to_project`, `get_workspace_topology` |
| [Chat](#chat-5-tools) | 5 | `list_chat_sessions`, `get_chat_session`, `delete_chat_session`, `list_chat_messages`, `chat_send_message` |
| [Meilisearch Maintenance](#meilisearch-maintenance-2-tools) | 2 | `get_meilisearch_stats`, `delete_meilisearch_orphans` |

---

## Project Management (8 tools)

### list_projects

List all projects with optional search and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `search` | string | No | Search in name/description |
| `limit` | integer | No | Max items (default 50, max 100) |
| `offset` | integer | No | Items to skip |
| `sort_by` | string | No | `name` or `created_at` |
| `sort_order` | string | No | `asc` or `desc` |

---

### create_project

Create a new project to track a codebase.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | **Yes** | Project name |
| `root_path` | string | **Yes** | Path to codebase root |
| `slug` | string | No | URL-safe identifier (auto-generated) |
| `description` | string | No | Project description |

---

### get_project

Get project details by slug.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

---

### update_project

Update a project's name, description, or root_path.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |
| `name` | string | No | New project name |
| `description` | string | No | New description |
| `root_path` | string | No | New root path |

---

### delete_project

Delete a project and all associated data.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

---

### sync_project

Sync a project's codebase (parse files, update graph).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

---

### get_project_roadmap

Get aggregated roadmap view with milestones, releases, and progress.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |

---

### list_project_plans

List all plans for a specific project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_slug` | string | **Yes** | Project slug |
| `status` | string | No | Filter by status |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

## Plan Management (9 tools)

### list_plans

List plans with optional filters and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | No | Filter by project UUID |
| `status` | string | No | Comma-separated: `draft,approved,in_progress,completed,cancelled` |
| `priority_min` | integer | No | Minimum priority |
| `priority_max` | integer | No | Maximum priority |
| `search` | string | No | Search in title/description |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |
| `sort_by` | string | No | `created_at`, `priority`, or `title` |
| `sort_order` | string | No | `asc` or `desc` |

---

### create_plan

Create a new development plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `title` | string | **Yes** | Plan title |
| `description` | string | **Yes** | Plan description |
| `priority` | integer | No | Priority (higher = more important) |
| `project_id` | string | No | Optional project UUID to link |

---

### get_plan

Get plan details including tasks, constraints, and decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

### update_plan_status

Update a plan's status.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `status` | string | **Yes** | `draft`, `approved`, `in_progress`, `completed`, `cancelled` |

---

### link_plan_to_project

Link a plan to a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `project_id` | string | **Yes** | Project UUID |

---

### unlink_plan_from_project

Unlink a plan from its project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

### get_dependency_graph

Get the task dependency graph for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Returns:** Graph with nodes (tasks) and edges (dependencies).

---

### get_critical_path

Get the critical path (longest dependency chain) for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Returns:** Ordered list of tasks in the critical path.

---

### delete_plan

Delete a plan and all its related data (tasks, steps, decisions, constraints).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

## Task Management (13 tools)

### list_tasks

List all tasks across plans with filters.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | No | Filter by plan UUID |
| `status` | string | No | Comma-separated: `pending,in_progress,blocked,completed,failed` |
| `priority_min` | integer | No | Minimum priority |
| `priority_max` | integer | No | Maximum priority |
| `tags` | string | No | Comma-separated tags |
| `assigned_to` | string | No | Filter by assignee |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |
| `sort_by` | string | No | Sort field |
| `sort_order` | string | No | `asc` or `desc` |

---

### create_task

Add a new task to a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `description` | string | **Yes** | Task description |
| `title` | string | No | Short title |
| `priority` | integer | No | Priority (higher = more important) |
| `tags` | array | No | Tags for categorization |
| `acceptance_criteria` | array | No | Conditions for completion |
| `affected_files` | array | No | Files to be modified |
| `dependencies` | array | No | Task UUIDs this depends on |

---

### get_task

Get task details including steps and decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### update_task

Update a task's status, assignee, or other fields.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `status` | string | No | `pending`, `in_progress`, `blocked`, `completed`, `failed` |
| `assigned_to` | string | No | Assignee name |
| `priority` | integer | No | New priority |
| `tags` | array | No | New tags |

---

### delete_task

Delete a task and all its steps and decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### get_next_task

Get the next available task from a plan (unblocked, highest priority).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

### add_task_dependencies

Add dependencies to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `dependency_ids` | array | **Yes** | Task UUIDs to depend on |

---

### remove_task_dependency

Remove a dependency from a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `dependency_id` | string | **Yes** | Dependency task UUID to remove |

---

### get_task_blockers

Get tasks that are blocking this task (uncompleted dependencies).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### get_tasks_blocked_by

Get tasks that are blocked by this task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### get_task_context

Get full context for a task (for agent execution).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `task_id` | string | **Yes** | Task UUID |

**Returns:** Rich context including plan, constraints, related code, and decisions.

---

### get_task_prompt

Get generated prompt for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `task_id` | string | **Yes** | Task UUID |

**Returns:** Ready-to-use prompt with all context embedded.

---

### add_decision

Record an architectural decision for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `description` | string | **Yes** | Decision description |
| `rationale` | string | **Yes** | Why this decision was made |
| `alternatives` | array | No | Alternatives considered |
| `chosen_option` | string | No | The chosen option |

---

## Step Management (6 tools)

### list_steps

List all steps for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### create_step

Add a step to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `description` | string | **Yes** | Step description |
| `verification` | string | No | How to verify completion |

---

### update_step

Update a step's status.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `step_id` | string | **Yes** | Step UUID |
| `status` | string | **Yes** | `pending`, `in_progress`, `completed`, `skipped` |

---

### get_step

Get a step by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `step_id` | string | **Yes** | Step UUID |

---

### delete_step

Delete a step.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `step_id` | string | **Yes** | Step UUID |

---

### get_step_progress

Get step completion progress for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Returns:** `{completed: N, total: M, percentage: X}`

---

## Decision Management (4 tools)

### get_decision

Get a decision by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `decision_id` | string | **Yes** | Decision UUID |

---

### update_decision

Update a decision's description, rationale, or chosen_option.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `decision_id` | string | **Yes** | Decision UUID |
| `description` | string | No | New description |
| `rationale` | string | No | New rationale |
| `chosen_option` | string | No | New chosen option |

---

### delete_decision

Delete a decision.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `decision_id` | string | **Yes** | Decision UUID |

---

### search_decisions

Search architectural decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | **Yes** | Search query |
| `limit` | integer | No | Max results |
| `project_slug` | string | No | Filter by project slug |

---

## Constraint Management (5 tools)

### list_constraints

List constraints for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

### add_constraint

Add a constraint to a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `constraint_type` | string | **Yes** | `performance`, `security`, `style`, `compatibility`, `other` |
| `description` | string | **Yes** | Constraint description |
| `severity` | string | No | `low`, `medium`, `high`, `critical` |

---

### get_constraint

Get a constraint by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `constraint_id` | string | **Yes** | Constraint UUID |

---

### update_constraint

Update a constraint's description, type, or enforced_by.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `constraint_id` | string | **Yes** | Constraint UUID |
| `description` | string | No | New description |
| `constraint_type` | string | No | New type: `performance`, `security`, `style`, `compatibility`, `other` |
| `enforced_by` | string | No | New enforced_by |

---

### delete_constraint

Delete a constraint.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `constraint_id` | string | **Yes** | Constraint UUID |

---

## Commit Tracking (5 tools)

### create_commit

Register a git commit.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `sha` | string | **Yes** | Commit SHA |
| `message` | string | **Yes** | Commit message |
| `author` | string | No | Author name |
| `files_changed` | array | No | Files changed |
| `project_id` | string | No | Project UUID -- enables incremental sync of changed files |

---

### link_commit_to_task

Link a commit to a task (RESOLVED_BY relationship).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `commit_sha` | string | **Yes** | Commit SHA |

---

### link_commit_to_plan

Link a commit to a plan (RESULTED_IN relationship).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `commit_sha` | string | **Yes** | Commit SHA |

---

### get_task_commits

Get commits linked to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

---

### get_plan_commits

Get commits linked to a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

---

## Release Management (7 tools)

### list_releases

List releases for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `status` | string | No | Filter by status |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

### create_release

Create a new release for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `version` | string | **Yes** | Version string (e.g., 1.0.0) |
| `title` | string | No | Release title |
| `description` | string | No | Release notes |
| `target_date` | string | No | Target date (ISO 8601) |

---

### get_release

Get release details with tasks and commits.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |

---

### update_release

Update a release.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |
| `status` | string | No | `planned`, `in_progress`, `released`, `cancelled` |
| `target_date` | string | No | New target date |
| `released_at` | string | No | Actual release date |
| `title` | string | No | New title |
| `description` | string | No | New description |

---

### delete_release

Delete a release.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |

---

### add_task_to_release

Add a task to a release.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |
| `task_id` | string | **Yes** | Task UUID |

---

### add_commit_to_release

Add a commit to a release.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |
| `commit_sha` | string | **Yes** | Commit SHA |

---

## Milestone Management (7 tools)

### list_milestones

List milestones for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `status` | string | No | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

### create_milestone

Create a new milestone for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `title` | string | **Yes** | Milestone title |
| `description` | string | No | Milestone description |
| `target_date` | string | No | Target date (ISO 8601) |

---

### get_milestone

Get milestone details with tasks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

---

### update_milestone

Update a milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |
| `status` | string | No | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `target_date` | string | No | New target date |
| `closed_at` | string | No | Closure date |
| `title` | string | No | New title |
| `description` | string | No | New description |

---

### delete_milestone

Delete a milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

---

### get_milestone_progress

Get milestone completion progress.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

**Returns:** `{completed: N, total: M, percentage: X}`

---

### add_task_to_milestone

Add a task to a milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |
| `task_id` | string | **Yes** | Task UUID |

---

## Code Exploration (13 tools)

### search_code

Search code semantically across all projects.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | **Yes** | Search query |
| `limit` | integer | No | Max results (default 10) |
| `language` | string | No | Filter by language |
| `project_slug` | string | No | Filter by project slug |
| `path_prefix` | string | No | Filter by path prefix (e.g. `src/mcp/`) |

---

### search_project_code

Search code within a specific project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_slug` | string | **Yes** | Project slug |
| `query` | string | **Yes** | Search query |
| `limit` | integer | No | Max results |
| `language` | string | No | Filter by language |
| `path_prefix` | string | No | Filter by path prefix (e.g. `src/mcp/`) |

---

### search_workspace_code

Search code across all projects in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `query` | string | **Yes** | Search query |
| `language` | string | No | Filter by language |
| `limit` | integer | No | Max results (default 10) |
| `path_prefix` | string | No | Filter by path prefix (e.g. `src/mcp/`) |

---

### get_file_symbols

Get all symbols (functions, structs, traits) in a file.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | **Yes** | File path |

---

### find_references

Find all references to a symbol.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | **Yes** | Symbol name |
| `limit` | integer | No | Max results |

---

### get_file_dependencies

Get file imports and files that depend on it.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | **Yes** | File path |

---

### get_call_graph

Get the call graph for a function.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `function` | string | **Yes** | Function name |
| `limit` | integer | No | Max depth/results |

---

### analyze_impact

Analyze the impact of changing a file or symbol.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `target` | string | **Yes** | File path or symbol name |

**Returns:** Directly affected, transitively affected, test files, risk level.

---

### get_architecture

Get codebase architecture overview (most connected files).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_slug` | string | No | Filter by project slug |

---

### find_similar_code

Find code similar to a given snippet.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code_snippet` | string | **Yes** | Code to find similar matches for |
| `limit` | integer | No | Max results |
| `language` | string | No | Filter by language |
| `project_slug` | string | No | Filter by project slug |

---

### find_trait_implementations

Find all implementations of a trait.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `trait_name` | string | **Yes** | Trait name |
| `limit` | integer | No | Max results |

---

### find_type_traits

Find all traits implemented by a type.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `type_name` | string | **Yes** | Type name (struct/enum) |
| `limit` | integer | No | Max results |

---

### get_impl_blocks

Get all impl blocks for a type.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `type_name` | string | **Yes** | Type name (struct/enum) |
| `limit` | integer | No | Max results |

---

## Notes (Knowledge Base) (17 tools)

Knowledge Notes capture contextual knowledge about your codebase -- guidelines, gotchas, patterns, and tips that propagate through the code graph.

See the [Knowledge Notes Guide](../guides/knowledge-notes.md) for detailed usage instructions.

### list_notes

List notes with optional filters and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | No | Filter by project UUID |
| `note_type` | string | No | Filter by type: `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion` |
| `status` | string | No | Comma-separated: `active,needs_review,stale,obsolete,archived` |
| `importance` | string | No | `critical`, `high`, `medium`, `low` |
| `tags` | string | No | Comma-separated tags |
| `min_staleness` | number | No | Minimum staleness score (0.0-1.0) |
| `max_staleness` | number | No | Maximum staleness score (0.0-1.0) |
| `search` | string | No | Search in content |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |

---

### create_note

Create a new knowledge note.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `note_type` | string | **Yes** | `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion` |
| `content` | string | **Yes** | Note content |
| `importance` | string | No | `critical`, `high`, `medium`, `low` (default: medium) |
| `tags` | array | No | Tags for categorization |
| `scope` | object | No | Scope: `{type: "file", path: "src/auth.rs"}` |
| `anchors` | array | No | Initial anchors to code entities |

---

### get_note

Get a note by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

---

### update_note

Update a note's content, importance, status, or tags.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `content` | string | No | New content |
| `importance` | string | No | New importance level |
| `status` | string | No | New status |
| `tags` | array | No | New tags |

---

### delete_note

Delete a note.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

---

### search_notes

Search notes using semantic search.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | **Yes** | Search query |
| `project_slug` | string | No | Filter by project slug |
| `note_type` | string | No | Filter by note type |
| `status` | string | No | Filter by status |
| `importance` | string | No | Filter by importance |
| `limit` | integer | No | Max results (default 20) |

---

### confirm_note

Confirm a note is still valid (resets staleness score).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

---

### invalidate_note

Mark a note as obsolete with a reason.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `reason` | string | **Yes** | Reason for invalidation |

---

### supersede_note

Replace an old note with a new one (preserves history).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `old_note_id` | string | **Yes** | ID of note to supersede |
| `project_id` | string | **Yes** | Project UUID |
| `note_type` | string | **Yes** | Type of new note |
| `content` | string | **Yes** | Content of new note |
| `importance` | string | No | Importance of new note |
| `tags` | array | No | Tags for new note |

---

### link_note_to_entity

Link a note to a code entity (file, function, struct, etc.).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `entity_type` | string | **Yes** | `file`, `function`, `struct`, `trait`, `task`, `plan`, etc. |
| `entity_id` | string | **Yes** | Entity ID (file path or UUID) |

---

### unlink_note_from_entity

Remove a link between a note and an entity.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `entity_type` | string | **Yes** | Entity type |
| `entity_id` | string | **Yes** | Entity ID |

---

### get_context_notes

Get contextual notes for an entity (direct + propagated through graph).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `entity_type` | string | **Yes** | `file`, `function`, `struct`, `task`, etc. |
| `entity_id` | string | **Yes** | Entity ID |
| `max_depth` | integer | No | Max traversal depth (default 3) |
| `min_score` | number | No | Min relevance score (default 0.1) |

**Returns:** Direct notes and propagated notes with relevance scores.

---

### get_propagated_notes

Get notes propagated through the graph (not directly attached).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `entity_type` | string | **Yes** | `file`, `function`, `struct`, `task`, etc. |
| `entity_id` | string | **Yes** | Entity ID |
| `max_depth` | integer | No | Max traversal depth (default 3) |
| `min_score` | number | No | Min relevance score (default 0.1) |

---

### get_entity_notes

Get notes directly attached to an entity.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `entity_type` | string | **Yes** | `file`, `function`, `struct`, `trait`, `task`, `plan`, etc. |
| `entity_id` | string | **Yes** | Entity ID (file path or UUID) |

---

### get_notes_needing_review

Get notes that need human review (stale or needs_review status).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | No | Optional project UUID filter |

---

### update_staleness_scores

Update staleness scores for all notes based on time decay.

**Parameters:** None

---

### list_project_notes

List notes for a specific project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `note_type` | string | No | Filter by type |
| `status` | string | No | Filter by status |
| `importance` | string | No | Filter by importance |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |

---

## Sync & Watch (4 tools)

### sync_directory

Manually sync a directory to the knowledge graph.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | **Yes** | Directory path |
| `project_id` | string | No | Optional project UUID |

---

### start_watch

Start auto-sync file watcher.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | **Yes** | Directory to watch |
| `project_id` | string | No | Optional project UUID |

---

### stop_watch

Stop the file watcher.

**Parameters:** None

---

### watch_status

Get file watcher status.

**Parameters:** None

---

## Workspace Management (9 tools)

Workspaces group related projects together, enabling shared context, cross-project milestones, and deployment topology modeling.

See the [Workspaces Guide](../guides/workspaces.md) for detailed usage instructions.

### list_workspaces

List all workspaces with optional search and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `search` | string | No | Search in name/description |
| `limit` | integer | No | Max items (default 50, max 100) |
| `offset` | integer | No | Items to skip |
| `sort_by` | string | No | `name` or `created_at` |
| `sort_order` | string | No | `asc` or `desc` |

---

### create_workspace

Create a new workspace to group related projects.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | **Yes** | Workspace name |
| `slug` | string | No | URL-safe identifier (auto-generated) |
| `description` | string | No | Workspace description |
| `metadata` | object | No | Optional metadata |

---

### get_workspace

Get workspace details by slug.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

---

### update_workspace

Update a workspace's name, description, or metadata.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `name` | string | No | New name |
| `description` | string | No | New description |
| `metadata` | object | No | New metadata |

---

### delete_workspace

Delete a workspace (does not delete associated projects).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

---

### get_workspace_overview

Get workspace overview with projects, milestones, resources, and progress.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Returns:** Projects, milestones, resources, components, and progress stats.

---

### list_workspace_projects

List all projects in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

---

### add_project_to_workspace

Add an existing project to a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `project_id` | string | **Yes** | Project UUID to add |

---

### remove_project_from_workspace

Remove a project from a workspace (does not delete the project).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `project_id` | string | **Yes** | Project UUID to remove |

---

## Workspace Milestones (8 tools)

Workspace milestones coordinate tasks across multiple projects within a workspace.

### list_all_workspace_milestones

List all workspace milestones across all workspaces with optional filters and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_id` | string | No | Filter by workspace UUID |
| `status` | string | No | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |

---

### list_workspace_milestones

List milestones for a workspace (cross-project milestones).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `status` | string | No | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

### create_workspace_milestone

Create a cross-project milestone in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `title` | string | **Yes** | Milestone title |
| `description` | string | No | Milestone description |
| `target_date` | string | No | Target date (ISO 8601) |
| `tags` | array | No | Tags for categorization |

---

### get_workspace_milestone

Get workspace milestone details with linked tasks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Workspace milestone UUID |

---

### update_workspace_milestone

Update a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Workspace milestone UUID |
| `title` | string | No | New title |
| `description` | string | No | New description |
| `status` | string | No | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `target_date` | string | No | New target date |
| `closed_at` | string | No | Closure date |

---

### delete_workspace_milestone

Delete a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Workspace milestone UUID |

---

### add_task_to_workspace_milestone

Add a task from any project to a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Workspace milestone UUID |
| `task_id` | string | **Yes** | Task UUID (from any project) |

---

### get_workspace_milestone_progress

Get completion progress for a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Workspace milestone UUID |

**Returns:** `{total: N, completed: M, in_progress: P, pending: Q, by_project: {...}}`

---

## Resources (6 tools)

Resources are shared contracts, schemas, or specifications referenced by multiple projects.

### list_resources

List resources (API contracts, schemas) in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `resource_type` | string | No | Filter by type: `ApiContract`, `Protobuf`, `GraphqlSchema`, `JsonSchema`, `DatabaseSchema`, `SharedTypes`, `Config`, `Documentation`, `Other` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

### create_resource

Create a shared resource reference (API contract, schema file).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `name` | string | **Yes** | Resource name |
| `resource_type` | string | **Yes** | `ApiContract`, `Protobuf`, `GraphqlSchema`, `JsonSchema`, `DatabaseSchema`, `SharedTypes`, `Config`, `Documentation`, `Other` |
| `file_path` | string | **Yes** | Path to the resource file |
| `url` | string | No | External URL |
| `format` | string | No | Format: `openapi`, `protobuf`, `graphql` |
| `version` | string | No | Version string |
| `description` | string | No | Resource description |
| `metadata` | object | No | Additional metadata |

---

### get_resource

Get resource details.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Resource UUID |

---

### update_resource

Update a resource's name, file_path, url, version, or description.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Resource UUID |
| `name` | string | No | New name |
| `file_path` | string | No | New file path |
| `url` | string | No | New URL |
| `version` | string | No | New version |
| `description` | string | No | New description |

---

### delete_resource

Delete a resource.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Resource UUID |

---

### link_resource_to_project

Link a resource to a project (implements or uses).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Resource UUID |
| `project_id` | string | **Yes** | Project UUID |
| `link_type` | string | **Yes** | `implements` or `uses` |

---

## Components & Topology (9 tools)

Components model the deployment topology of your system.

### list_components

List components (services, databases, etc.) in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `component_type` | string | No | Filter by type: `Service`, `Frontend`, `Worker`, `Database`, `MessageQueue`, `Cache`, `Gateway`, `External`, `Other` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

---

### create_component

Create a component in the workspace topology.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `name` | string | **Yes** | Component name |
| `component_type` | string | **Yes** | `Service`, `Frontend`, `Worker`, `Database`, `MessageQueue`, `Cache`, `Gateway`, `External`, `Other` |
| `description` | string | No | Component description |
| `runtime` | string | No | Runtime: `docker`, `kubernetes`, `lambda` |
| `config` | object | No | Configuration (env vars, ports, etc.) |
| `tags` | array | No | Tags for categorization |

---

### get_component

Get component details.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Component UUID |

---

### update_component

Update a component's name, description, runtime, config, or tags.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Component UUID |
| `name` | string | No | New name |
| `description` | string | No | New description |
| `runtime` | string | No | New runtime |
| `config` | object | No | New configuration |
| `tags` | array | No | New tags |

---

### delete_component

Delete a component.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Component UUID |

---

### add_component_dependency

Add a dependency between components.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Source component UUID |
| `depends_on_id` | string | **Yes** | Target component UUID |
| `protocol` | string | No | Communication protocol: `http`, `grpc`, `amqp`, etc. |
| `required` | boolean | No | Whether dependency is required |

---

### remove_component_dependency

Remove a dependency between components.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Source component UUID |
| `dep_id` | string | **Yes** | Target component UUID to remove |

---

### map_component_to_project

Map a component to a project (link source code).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `id` | string | **Yes** | Component UUID |
| `project_id` | string | **Yes** | Project UUID |

---

### get_workspace_topology

Get the full topology graph of a workspace (components and dependencies).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Returns:** All components with dependencies, protocols, and mapped projects.

---

## Chat (5 tools)

Chat tools enable conversational interactions with AI agents, with session persistence and project association.

### list_chat_sessions

List chat sessions with optional project filter and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_slug` | string | No | Filter by project slug |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |

---

### get_chat_session

Get chat session details by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `session_id` | string | **Yes** | Session UUID |

---

### delete_chat_session

Delete a chat session.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `session_id` | string | **Yes** | Session UUID |

---

### list_chat_messages

List message history for a chat session (chronological order).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `session_id` | string | **Yes** | Session UUID |
| `limit` | integer | No | Max messages to retrieve (default 50) |
| `offset` | integer | No | Messages to skip for pagination (default 0) |

---

### chat_send_message

Send a chat message and wait for the complete response (non-streaming). Creates a new session or resumes an existing one.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `message` | string | **Yes** | The user message to send |
| `cwd` | string | **Yes** | Working directory for Claude Code CLI |
| `session_id` | string | No | Session ID to resume (creates new session if omitted) |
| `project_slug` | string | No | Project slug to associate with the session |
| `model` | string | No | Model override (default: from config) |

---

## Meilisearch Maintenance (2 tools)

### get_meilisearch_stats

Get Meilisearch code index statistics.

**Parameters:** None

---

### delete_meilisearch_orphans

Delete documents without project_id from Meilisearch.

**Parameters:** None
