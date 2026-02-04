# MCP Tools Reference

Complete documentation for all 113 MCP tools exposed by Project Orchestrator.

---

## Quick Reference

| Category | Tools |
|----------|-------|
| [Project Management](#project-management-6-tools) | `list_projects`, `create_project`, `get_project`, `delete_project`, `sync_project`, `get_project_roadmap` |
| [Workspace Management](#workspace-management-9-tools) | `list_workspaces`, `create_workspace`, `get_workspace`, `update_workspace`, `delete_workspace`, `get_workspace_overview`, `get_workspace_context`, `add_project_to_workspace`, `remove_project_from_workspace` |
| [Workspace Milestones](#workspace-milestones-6-tools) | `list_workspace_milestones`, `create_workspace_milestone`, `get_workspace_milestone`, `update_workspace_milestone`, `add_task_to_workspace_milestone`, `get_workspace_milestone_progress` |
| [Resources](#resources-6-tools) | `list_resources`, `create_resource`, `get_resource`, `update_resource`, `delete_resource`, `link_resource_to_project` |
| [Components & Topology](#components--topology-8-tools) | `list_components`, `create_component`, `get_component`, `update_component`, `delete_component`, `add_component_dependency`, `map_component_to_project`, `get_workspace_topology` |
| [Plan Management](#plan-management-8-tools) | `list_plans`, `create_plan`, `get_plan`, `update_plan_status`, `link_plan_to_project`, `unlink_plan_from_project`, `get_dependency_graph`, `get_critical_path` |
| [Task Management](#task-management-12-tools) | `list_tasks`, `create_task`, `get_task`, `update_task`, `get_next_task`, `add_task_dependencies`, `remove_task_dependency`, `get_task_blockers`, `get_tasks_blocked_by`, `get_task_context`, `get_task_prompt`, `add_decision` |
| [Step Management](#step-management-4-tools) | `list_steps`, `create_step`, `update_step`, `get_step_progress` |
| [Constraint Management](#constraint-management-3-tools) | `list_constraints`, `add_constraint`, `delete_constraint` |
| [Release Management](#release-management-6-tools) | `list_releases`, `create_release`, `get_release`, `update_release`, `add_task_to_release`, `add_commit_to_release` |
| [Milestone Management](#milestone-management-6-tools) | `list_milestones`, `create_milestone`, `get_milestone`, `update_milestone`, `get_milestone_progress`, `add_task_to_milestone` |
| [Commit Tracking](#commit-tracking-5-tools) | `create_commit`, `link_commit_to_task`, `link_commit_to_plan`, `get_task_commits`, `get_plan_commits` |
| [Code Exploration](#code-exploration-13-tools) | `search_code`, `search_project_code`, `get_file_symbols`, `find_references`, `get_file_dependencies`, `get_call_graph`, `analyze_impact`, `get_architecture`, `find_similar_code`, `find_trait_implementations`, `find_type_traits`, `get_impl_blocks` |
| [Knowledge Notes](#knowledge-notes-14-tools) | `list_notes`, `create_note`, `get_note`, `update_note`, `delete_note`, `search_notes`, `confirm_note`, `invalidate_note`, `supersede_note`, `link_note_to_entity`, `unlink_note_from_entity`, `get_context_notes`, `get_notes_needing_review`, `update_staleness_scores` |
| [Decision Search](#decision-search-1-tool) | `search_decisions` |
| [Sync & Watch](#sync--watch-4-tools) | `sync_directory`, `start_watch`, `stop_watch`, `watch_status` |
| [Meilisearch Admin](#meilisearch-admin-2-tools) | `get_meilisearch_stats`, `delete_meilisearch_orphans` |

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

**Example:**
```
List all workspaces containing "microservices"
```

---

### create_workspace

Create a new workspace to group related projects.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | **Yes** | Workspace name |
| `slug` | string | No | URL-safe identifier (auto-generated) |
| `description` | string | No | Workspace description |

**Example:**
```
Create a workspace named "E-Commerce Platform" for our API and frontend projects
```

---

### get_workspace

Get workspace details by slug.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Example:**
```
Get details for workspace "e-commerce-platform"
```

---

### update_workspace

Update a workspace's details.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |
| `name` | string | No | New name |
| `description` | string | No | New description |

**Example:**
```
Update workspace description to include new scope
```

---

### delete_workspace

Delete a workspace and remove project associations.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Example:**
```
Delete workspace "old-project-group"
```

---

### get_workspace_overview

Get a comprehensive overview of the workspace including projects, milestones, resources, and components.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Returns:** Projects, milestones, resources, components, and progress stats.

**Example:**
```
Show me the overview for workspace "e-commerce-platform"
```

---

### get_workspace_context

Get rich context for agents working on workspace-level tasks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `slug` | string | **Yes** | Workspace slug |

**Returns:** Full context including projects, resources, topology, and notes.

**Example:**
```
Get workspace context for coordinated development
```

---

### add_project_to_workspace

Add an existing project to a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `project_id` | string | **Yes** | Project UUID to add |

**Example:**
```
Add the "api-service" project to workspace "e-commerce-platform"
```

---

### remove_project_from_workspace

Remove a project from a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `project_id` | string | **Yes** | Project UUID to remove |

**Example:**
```
Remove project "legacy-api" from workspace
```

---

## Workspace Milestones (6 tools)

Workspace milestones coordinate tasks across multiple projects within a workspace.

### list_workspace_milestones

List milestones for a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `status` | string | No | `open` or `closed` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

**Example:**
```
List open workspace milestones for "e-commerce-platform"
```

---

### create_workspace_milestone

Create a cross-project milestone in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `title` | string | **Yes** | Milestone title |
| `description` | string | No | Milestone description |
| `target_date` | string | No | Target date (ISO 8601) |
| `tags` | array | No | Tags for categorization |

**Example:**
```
Create milestone "Q1 Launch" with target March 31st
```

---

### get_workspace_milestone

Get workspace milestone details with associated tasks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

**Example:**
```
Get details for workspace milestone xyz-789
```

---

### update_workspace_milestone

Update a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |
| `title` | string | No | New title |
| `description` | string | No | New description |
| `status` | string | No | `open` or `closed` |
| `target_date` | string | No | New target date |

**Example:**
```
Close workspace milestone "Q1 Launch"
```

---

### add_task_to_workspace_milestone

Add a task from any project to a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |
| `task_id` | string | **Yes** | Task UUID (from any project) |

**Example:**
```
Add the API authentication task to the Q1 Launch milestone
```

---

### get_workspace_milestone_progress

Get aggregated progress across all projects for a workspace milestone.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `milestone_id` | string | **Yes** | Milestone UUID |

**Returns:** `{total: N, completed: M, in_progress: P, pending: Q, by_project: {...}}`

**Example:**
```
What's the progress on workspace milestone "Q1 Launch"?
```

---

## Resources (6 tools)

Resources are shared contracts, schemas, or specifications referenced by multiple projects.

### list_resources

List resources in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `resource_type` | string | No | Filter by type: `api_contract`, `protobuf`, `graphql_schema`, `json_schema`, `database_schema`, `shared_types`, `config`, `documentation`, `other` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

**Example:**
```
List all API contracts in workspace "e-commerce-platform"
```

---

### create_resource

Create a shared resource in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `name` | string | **Yes** | Resource name |
| `resource_type` | string | **Yes** | Resource type (see list_resources) |
| `file_path` | string | **Yes** | Path to the spec file |
| `url` | string | No | External URL |
| `format` | string | No | Format: `openapi`, `protobuf`, `graphql` |
| `version` | string | No | Version string |
| `description` | string | No | Resource description |

**Example:**
```
Create an API contract "User Service API" pointing to openapi/users.yaml
```

---

### get_resource

Get resource details.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `resource_id` | string | **Yes** | Resource UUID |

**Example:**
```
Get details for resource xyz-789
```

---

### update_resource

Update a resource.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `resource_id` | string | **Yes** | Resource UUID |
| `name` | string | No | New name |
| `file_path` | string | No | New file path |
| `version` | string | No | New version |
| `description` | string | No | New description |

**Example:**
```
Update resource version to 2.0.0
```

---

### delete_resource

Delete a resource.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `resource_id` | string | **Yes** | Resource UUID |

**Example:**
```
Delete deprecated resource xyz-789
```

---

### link_resource_to_project

Link a resource to a project as implementer or consumer.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `resource_id` | string | **Yes** | Resource UUID |
| `project_id` | string | **Yes** | Project UUID |
| `relationship` | string | **Yes** | `implements` (provider) or `uses` (consumer) |

**Example:**
```
Link api-service as implementer of "User Service API" contract
```

---

## Components & Topology (8 tools)

Components model the deployment topology of your system.

### list_components

List components in a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `component_type` | string | No | Filter by type: `service`, `frontend`, `worker`, `database`, `message_queue`, `cache`, `gateway`, `external`, `other` |
| `limit` | integer | No | Max items |
| `offset` | integer | No | Items to skip |

**Example:**
```
List all services in workspace "e-commerce-platform"
```

---

### create_component

Create a deployment component.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |
| `name` | string | **Yes** | Component name |
| `component_type` | string | **Yes** | Component type (see list_components) |
| `description` | string | No | Component description |
| `runtime` | string | No | Runtime: `docker`, `kubernetes`, `lambda` |
| `config` | object | No | Configuration (ports, env vars) |
| `tags` | array | No | Tags for categorization |

**Example:**
```
Create a service component "API Gateway" running on Kubernetes
```

---

### get_component

Get component details.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `component_id` | string | **Yes** | Component UUID |

**Example:**
```
Get details for component xyz-789
```

---

### update_component

Update a component.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `component_id` | string | **Yes** | Component UUID |
| `name` | string | No | New name |
| `description` | string | No | New description |
| `runtime` | string | No | New runtime |
| `config` | object | No | New configuration |

**Example:**
```
Update component runtime to kubernetes
```

---

### delete_component

Delete a component.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `component_id` | string | **Yes** | Component UUID |

**Example:**
```
Delete component xyz-789
```

---

### add_component_dependency

Add a dependency between components.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `component_id` | string | **Yes** | Component UUID |
| `depends_on_id` | string | **Yes** | UUID of component this depends on |
| `protocol` | string | No | Protocol: `http`, `grpc`, `amqp`, `redis` |
| `required` | boolean | No | Whether dependency is required (default: true) |

**Example:**
```
API Gateway depends on User Service via gRPC
```

---

### map_component_to_project

Map a component to its source code project.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `component_id` | string | **Yes** | Component UUID |
| `project_id` | string | **Yes** | Project UUID |

**Example:**
```
Map "User Service" component to project "user-api"
```

---

### get_workspace_topology

Get the full deployment topology graph for a workspace.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `workspace_slug` | string | **Yes** | Workspace slug |

**Returns:** All components with dependencies, protocols, and mapped projects.

**Example:**
```
Show the deployment topology for workspace "e-commerce-platform"
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

---

## Knowledge Notes (14 tools)

Knowledge Notes capture contextual knowledge about your codebase - guidelines, gotchas, patterns, and tips that propagate through the code graph.

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

**Example:**
```
List all active guideline notes for project my-app
```

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

**Example:**
```
Create a gotcha note: "Don't use unwrap() in async contexts - it will panic the runtime"
```

---

### get_note

Get a note by ID.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

**Example:**
```
Get details for note abc-123
```

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

**Example:**
```
Update note abc-123 to importance critical
```

---

### delete_note

Delete a note.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

**Example:**
```
Delete note abc-123
```

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

**Example:**
```
Search for notes about error handling
```

---

### confirm_note

Confirm a note is still valid (resets staleness score).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |

**Example:**
```
Confirm that note abc-123 is still valid
```

---

### invalidate_note

Mark a note as obsolete with a reason.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `reason` | string | **Yes** | Reason for invalidation |

**Example:**
```
Invalidate note abc-123 because the auth system was refactored
```

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

**Example:**
```
Supersede note abc-123 with updated guidance about OAuth tokens
```

---

### link_note_to_entity

Link a note to a code entity (file, function, struct, etc.).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `entity_type` | string | **Yes** | `file`, `function`, `struct`, `trait`, `task`, `plan`, etc. |
| `entity_id` | string | **Yes** | Entity ID (file path or UUID) |

**Example:**
```
Link note abc-123 to the validate_user function
```

---

### unlink_note_from_entity

Remove a link between a note and an entity.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `note_id` | string | **Yes** | Note UUID |
| `entity_type` | string | **Yes** | Entity type |
| `entity_id` | string | **Yes** | Entity ID |

**Example:**
```
Unlink note abc-123 from file src/auth.rs
```

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

**Example:**
```
Get all relevant notes for file src/auth/jwt.rs
```

---

### get_notes_needing_review

Get notes that need human review (stale or needs_review status).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | No | Optional project UUID filter |

**Example:**
```
What notes need review?
```

---

### update_staleness_scores

Update staleness scores for all notes based on time decay.

**Parameters:** None

**Example:**
```
Recalculate staleness scores for all notes
```

---

## Meilisearch Admin (2 tools)

### get_meilisearch_stats

Get Meilisearch code index statistics.

**Parameters:** None

**Example:**
```
Show Meilisearch index stats
```

---

### delete_meilisearch_orphans

Delete documents without project_id from Meilisearch.

**Parameters:** None

**Example:**
```
Clean up orphaned documents in Meilisearch
```
