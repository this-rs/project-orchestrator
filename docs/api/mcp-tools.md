# MCP Tools Reference

Complete documentation for all 62 MCP tools exposed by Project Orchestrator.

---

## Quick Reference

| Category | Tools |
|----------|-------|
| [Project Management](#project-management-6-tools) | `list_projects`, `create_project`, `get_project`, `delete_project`, `sync_project`, `get_project_roadmap` |
| [Plan Management](#plan-management-8-tools) | `list_plans`, `create_plan`, `get_plan`, `update_plan_status`, `link_plan_to_project`, `unlink_plan_from_project`, `get_dependency_graph`, `get_critical_path` |
| [Task Management](#task-management-12-tools) | `list_tasks`, `create_task`, `get_task`, `update_task`, `get_next_task`, `add_task_dependencies`, `remove_task_dependency`, `get_task_blockers`, `get_tasks_blocked_by`, `get_task_context`, `get_task_prompt`, `add_decision` |
| [Step Management](#step-management-4-tools) | `list_steps`, `create_step`, `update_step`, `get_step_progress` |
| [Constraint Management](#constraint-management-3-tools) | `list_constraints`, `add_constraint`, `delete_constraint` |
| [Release Management](#release-management-5-tools) | `list_releases`, `create_release`, `get_release`, `update_release`, `add_task_to_release` |
| [Milestone Management](#milestone-management-5-tools) | `list_milestones`, `create_milestone`, `get_milestone`, `update_milestone`, `get_milestone_progress` |
| [Commit Tracking](#commit-tracking-4-tools) | `create_commit`, `link_commit_to_task`, `link_commit_to_plan`, `get_task_commits` |
| [Code Exploration](#code-exploration-10-tools) | `search_code`, `search_project_code`, `get_file_symbols`, `find_references`, `get_file_dependencies`, `get_call_graph`, `analyze_impact`, `get_architecture`, `find_similar_code`, `find_trait_implementations` |
| [Decision Search](#decision-search-1-tool) | `search_decisions` |
| [Sync & Watch](#sync--watch-4-tools) | `sync_directory`, `start_watch`, `stop_watch`, `watch_status` |

---

## Project Management (6 tools)

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

**Example:**
```
List all projects that contain "api" in their name
```

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

**Example:**
```
Create a project named "My App" at /Users/me/projects/myapp
```

---

### get_project

Get project details by slug.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

**Example:**
```
Get details for project "my-app"
```

---

### delete_project

Delete a project and all associated data.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

**Example:**
```
Delete the project "old-project"
```

---

### sync_project

Sync a project's codebase (parse files, update graph).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Project slug |

**Example:**
```
Sync the my-app project to update the code index
```

---

### get_project_roadmap

Get aggregated roadmap view with milestones, releases, and progress.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |

**Example:**
```
Show me the roadmap for project abc-123
```

---

## Plan Management (8 tools)

### list_plans

List plans with optional filters and pagination.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `status` | string | No | Comma-separated: `draft,approved,in_progress,completed,cancelled` |
| `priority_min` | integer | No | Minimum priority |
| `priority_max` | integer | No | Maximum priority |
| `search` | string | No | Search in title/description |
| `limit` | integer | No | Max items (default 50) |
| `offset` | integer | No | Items to skip |
| `sort_by` | string | No | `created_at`, `priority`, or `title` |
| `sort_order` | string | No | `asc` or `desc` |

**Example:**
```
List all in-progress plans with priority above 5
```

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

**Example:**
```
Create a plan called "Add OAuth Support" with priority 10
```

---

### get_plan

Get plan details including tasks, constraints, and decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Example:**
```
Get details for plan abc-123
```

---

### update_plan_status

Update a plan's status.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `status` | string | **Yes** | `draft`, `approved`, `in_progress`, `completed`, `cancelled` |

**Example:**
```
Mark plan abc-123 as in_progress
```

---

### link_plan_to_project

Link a plan to a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `project_id` | string | **Yes** | Project UUID |

**Example:**
```
Link plan abc-123 to project xyz-456
```

---

### unlink_plan_from_project

Unlink a plan from its project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Example:**
```
Unlink plan abc-123 from its project
```

---

### get_dependency_graph

Get the task dependency graph for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Returns:** Graph with nodes (tasks) and edges (dependencies).

**Example:**
```
Show the dependency graph for plan abc-123
```

---

### get_critical_path

Get the critical path (longest dependency chain) for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Returns:** Ordered list of tasks in the critical path.

**Example:**
```
What's the critical path for plan abc-123?
```

---

## Task Management (12 tools)

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

**Example:**
```
List all tasks assigned to agent-1 that are in progress
```

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

**Example:**
```
Add a task "Implement login endpoint" to plan abc-123 with tags [backend, auth]
```

---

### get_task

Get task details including steps and decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
Get details for task xyz-789
```

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

**Example:**
```
Mark task xyz-789 as completed
```

---

### get_next_task

Get the next available task from a plan (unblocked, highest priority).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Example:**
```
What's the next task I should work on for plan abc-123?
```

---

### add_task_dependencies

Add dependencies to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `dependency_ids` | array | **Yes** | Task UUIDs to depend on |

**Example:**
```
Make task B depend on tasks A and C
```

---

### remove_task_dependency

Remove a dependency from a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `dependency_id` | string | **Yes** | Dependency task UUID to remove |

**Example:**
```
Remove the dependency of task B on task A
```

---

### get_task_blockers

Get tasks that are blocking this task (uncompleted dependencies).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
What tasks are blocking task xyz-789?
```

---

### get_tasks_blocked_by

Get tasks that are blocked by this task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
What tasks are waiting on task xyz-789?
```

---

### get_task_context

Get full context for a task (for agent execution).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `task_id` | string | **Yes** | Task UUID |

**Returns:** Rich context including plan, constraints, related code, and decisions.

**Example:**
```
Get the full context for working on task xyz-789
```

---

### get_task_prompt

Get generated prompt for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `task_id` | string | **Yes** | Task UUID |

**Returns:** Ready-to-use prompt with all context embedded.

**Example:**
```
Generate a prompt for task xyz-789
```

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

**Example:**
```
Record a decision: We chose JWT over sessions because it's stateless
```

---

## Step Management (4 tools)

### list_steps

List all steps for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
Show all steps for task xyz-789
```

---

### create_step

Add a step to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `description` | string | **Yes** | Step description |
| `verification` | string | No | How to verify completion |

**Example:**
```
Add step "Write unit tests" to task xyz-789
```

---

### update_step

Update a step's status.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `step_id` | string | **Yes** | Step UUID |
| `status` | string | **Yes** | `pending`, `in_progress`, `completed`, `skipped` |

**Example:**
```
Mark step abc as completed
```

---

### get_step_progress

Get step completion progress for a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Returns:** `{completed: N, total: M, percentage: X}`

**Example:**
```
How many steps are done for task xyz-789?
```

---

## Constraint Management (3 tools)

### list_constraints

List constraints for a plan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |

**Example:**
```
What constraints apply to plan abc-123?
```

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

**Example:**
```
Add a security constraint: no plaintext passwords, severity critical
```

---

### delete_constraint

Delete a constraint.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `constraint_id` | string | **Yes** | Constraint UUID |

**Example:**
```
Remove constraint xyz-456
```

---

## Release Management (5 tools)

### list_releases

List releases for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `status` | string | No | Filter by status |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

**Example:**
```
List all releases for project my-app
```

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

**Example:**
```
Create release v1.0.0 for project my-app with target date March 1st
```

---

### get_release

Get release details with tasks and commits.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |

**Example:**
```
Get details for release xyz-789
```

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

**Example:**
```
Mark release xyz-789 as released
```

---

### add_task_to_release

Add a task to a release.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `release_id` | string | **Yes** | Release UUID |
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
Add task abc to release v1.0.0
```

---

## Milestone Management (5 tools)

### list_milestones

List milestones for a project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | **Yes** | Project UUID |
| `status` | string | No | `open` or `closed` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

**Example:**
```
List open milestones for project my-app
```

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

**Example:**
```
Create milestone "MVP Complete" with target February 28th
```

---

### get_milestone

Get milestone details with tasks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

**Example:**
```
Get details for milestone xyz-789
```

---

### update_milestone

Update a milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |
| `status` | string | No | `open` or `closed` |
| `target_date` | string | No | New target date |
| `closed_at` | string | No | Closure date |
| `title` | string | No | New title |
| `description` | string | No | New description |

**Example:**
```
Close milestone xyz-789
```

---

### get_milestone_progress

Get milestone completion progress.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

**Returns:** `{completed: N, total: M, percentage: X}`

**Example:**
```
What's the progress on milestone xyz-789?
```

---

## Commit Tracking (4 tools)

### create_commit

Register a git commit.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `sha` | string | **Yes** | Commit SHA |
| `message` | string | **Yes** | Commit message |
| `author` | string | No | Author name |
| `files_changed` | array | No | Files changed |

**Example:**
```
Register commit abc123 with message "feat: add auth"
```

---

### link_commit_to_task

Link a commit to a task (RESOLVED_BY relationship).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |
| `commit_sha` | string | **Yes** | Commit SHA |

**Example:**
```
Link commit abc123 to task xyz-789
```

---

### link_commit_to_plan

Link a commit to a plan (RESULTED_IN relationship).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `plan_id` | string | **Yes** | Plan UUID |
| `commit_sha` | string | **Yes** | Commit SHA |

**Example:**
```
Link commit abc123 to plan xyz-789
```

---

### get_task_commits

Get commits linked to a task.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | string | **Yes** | Task UUID |

**Example:**
```
What commits resolved task xyz-789?
```

---

## Code Exploration (10 tools)

### search_code

Search code semantically across all projects.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | **Yes** | Search query |
| `limit` | integer | No | Max results (default 10) |
| `language` | string | No | Filter by language |

**Example:**
```
Search for code related to "error handling"
```

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

**Example:**
```
Search for "authentication" in project my-app
```

---

### get_file_symbols

Get all symbols (functions, structs, traits) in a file.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | **Yes** | File path |

**Example:**
```
Show me all symbols in src/lib.rs
```

---

### find_references

Find all references to a symbol.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | **Yes** | Symbol name |
| `limit` | integer | No | Max results |

**Example:**
```
Find all references to AppState
```

---

### get_file_dependencies

Get file imports and files that depend on it.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | **Yes** | File path |

**Example:**
```
What files depend on src/models/user.rs?
```

---

### get_call_graph

Get the call graph for a function.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `function` | string | **Yes** | Function name |
| `limit` | integer | No | Max depth/results |

**Example:**
```
Show the call graph for function handle_request
```

---

### analyze_impact

Analyze the impact of changing a file or symbol.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `target` | string | **Yes** | File path or symbol name |

**Returns:** Directly affected, transitively affected, test files, risk level.

**Example:**
```
What would be impacted if I change UserService?
```

---

### get_architecture

Get codebase architecture overview (most connected files).

**Parameters:** None

**Example:**
```
Show me the architecture overview
```

---

### find_similar_code

Find code similar to a given snippet.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code_snippet` | string | **Yes** | Code to find similar matches for |
| `limit` | integer | No | Max results |

**Example:**
```
Find code similar to "async fn handle_error"
```

---

### find_trait_implementations

Find all implementations of a trait.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `trait_name` | string | **Yes** | Trait name |
| `limit` | integer | No | Max results |

**Example:**
```
Find all types that implement Handler
```

---

## Decision Search (1 tool)

### search_decisions

Search architectural decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | **Yes** | Search query |
| `limit` | integer | No | Max results |

**Example:**
```
Search for decisions about authentication
```

---

## Sync & Watch (4 tools)

### sync_directory

Manually sync a directory to the knowledge graph.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | **Yes** | Directory path |
| `project_id` | string | No | Optional project UUID |

**Example:**
```
Sync directory /path/to/code
```

---

### start_watch

Start auto-sync file watcher.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | **Yes** | Directory to watch |
| `project_id` | string | No | Optional project UUID |

**Example:**
```
Start watching /path/to/project for changes
```

---

### stop_watch

Stop the file watcher.

**Parameters:** None

**Example:**
```
Stop the file watcher
```

---

### watch_status

Get file watcher status.

**Parameters:** None

**Example:**
```
Is the file watcher running?
```
