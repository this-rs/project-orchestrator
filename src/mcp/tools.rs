//! MCP Tool definitions
//!
//! Defines all 113 tools exposed by the MCP server.

use super::protocol::{InputSchema, ToolDefinition};
use serde_json::json;

/// Generate all tool definitions
pub fn all_tools() -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    tools.extend(project_tools());
    tools.extend(plan_tools());
    tools.extend(task_tools());
    tools.extend(step_tools());
    tools.extend(constraint_tools());
    tools.extend(release_tools());
    tools.extend(milestone_tools());
    tools.extend(commit_tools());
    tools.extend(code_tools());
    tools.extend(decision_tools());
    tools.extend(sync_tools());
    tools.extend(meilisearch_tools());
    tools.extend(note_tools());
    tools.extend(workspace_tools());
    tools.extend(chat_tools());
    tools
}

// ============================================================================
// Project Tools (6)
// ============================================================================

fn project_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_projects".to_string(),
            description: "List all projects with optional search and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "search": {"type": "string", "description": "Search in name/description"},
                    "limit": {"type": "integer", "description": "Max items (default 50, max 100)"},
                    "offset": {"type": "integer", "description": "Items to skip"},
                    "sort_by": {"type": "string", "description": "Sort field (name, created_at)"},
                    "sort_order": {"type": "string", "description": "asc or desc"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "create_project".to_string(),
            description: "Create a new project to track a codebase".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "name": {"type": "string", "description": "Project name"},
                    "root_path": {"type": "string", "description": "Path to codebase root"},
                    "slug": {"type": "string", "description": "URL-safe identifier (auto-generated if not provided)"},
                    "description": {"type": "string", "description": "Project description"}
                })),
                required: Some(vec!["name".to_string(), "root_path".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_project".to_string(),
            description: "Get project details by slug".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Project slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_project".to_string(),
            description: "Delete a project and all associated data".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Project slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "sync_project".to_string(),
            description: "Sync a project's codebase (parse files, update graph)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Project slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_project".to_string(),
            description: "Update a project's name, description, or root_path".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Project slug"},
                    "name": {"type": "string", "description": "New project name"},
                    "description": {"type": "string", "description": "New description"},
                    "root_path": {"type": "string", "description": "New root path"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_project_roadmap".to_string(),
            description: "Get aggregated roadmap view with milestones, releases, and progress"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"}
                })),
                required: Some(vec!["project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "list_project_plans".to_string(),
            description: "List all plans for a specific project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Project slug"},
                    "status": {"type": "string", "description": "Filter by status"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["project_slug".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Plan Tools (9)
// ============================================================================

fn plan_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_plans".to_string(),
            description: "List plans with optional filters and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Filter by project UUID"},
                    "status": {"type": "string", "description": "Filter by status (comma-separated: draft,approved,in_progress,completed,cancelled)"},
                    "priority_min": {"type": "integer", "description": "Minimum priority"},
                    "priority_max": {"type": "integer", "description": "Maximum priority"},
                    "search": {"type": "string", "description": "Search in title/description"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"},
                    "sort_by": {"type": "string", "description": "Sort field (created_at, priority, title)"},
                    "sort_order": {"type": "string", "description": "asc or desc"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "create_plan".to_string(),
            description: "Create a new development plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "title": {"type": "string", "description": "Plan title"},
                    "description": {"type": "string", "description": "Plan description"},
                    "priority": {"type": "integer", "description": "Priority (higher = more important)"},
                    "project_id": {"type": "string", "description": "Optional project UUID to link"}
                })),
                required: Some(vec!["title".to_string(), "description".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_plan".to_string(),
            description: "Get plan details including tasks, constraints, and decisions".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_plan_status".to_string(),
            description: "Update a plan's status".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "status": {"type": "string", "description": "New status (draft, approved, in_progress, completed, cancelled)"}
                })),
                required: Some(vec!["plan_id".to_string(), "status".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_plan_to_project".to_string(),
            description: "Link a plan to a project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "project_id": {"type": "string", "description": "Project UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "unlink_plan_from_project".to_string(),
            description: "Unlink a plan from its project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_dependency_graph".to_string(),
            description: "Get the task dependency graph for a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_critical_path".to_string(),
            description: "Get the critical path (longest dependency chain) for a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_plan".to_string(),
            description:
                "Delete a plan and all its related data (tasks, steps, decisions, constraints)"
                    .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Task Tools (12)
// ============================================================================

fn task_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_tasks".to_string(),
            description: "List all tasks across plans with filters".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Filter by plan UUID"},
                    "status": {"type": "string", "description": "Filter by status (comma-separated: pending,in_progress,blocked,completed,failed)"},
                    "priority_min": {"type": "integer", "description": "Minimum priority"},
                    "priority_max": {"type": "integer", "description": "Maximum priority"},
                    "tags": {"type": "string", "description": "Filter by tags (comma-separated)"},
                    "assigned_to": {"type": "string", "description": "Filter by assignee"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"},
                    "sort_by": {"type": "string", "description": "Sort field"},
                    "sort_order": {"type": "string", "description": "asc or desc"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "create_task".to_string(),
            description: "Add a new task to a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "description": {"type": "string", "description": "Task description"},
                    "title": {"type": "string", "description": "Short title"},
                    "priority": {"type": "integer", "description": "Priority (higher = more important)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "Conditions for completion"},
                    "affected_files": {"type": "array", "items": {"type": "string"}, "description": "Files to be modified"},
                    "dependencies": {"type": "array", "items": {"type": "string"}, "description": "Task UUIDs this depends on"}
                })),
                required: Some(vec!["plan_id".to_string(), "description".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_task".to_string(),
            description: "Get task details including steps and decisions".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_task".to_string(),
            description: "Update a task's status, assignee, or other fields".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "status": {"type": "string", "description": "New status (pending, in_progress, blocked, completed, failed)"},
                    "assigned_to": {"type": "string", "description": "Assignee name"},
                    "priority": {"type": "integer", "description": "New priority"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_task".to_string(),
            description: "Delete a task and all its steps and decisions".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_next_task".to_string(),
            description: "Get the next available task from a plan (unblocked, highest priority)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_task_dependencies".to_string(),
            description: "Add dependencies to a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "dependency_ids": {"type": "array", "items": {"type": "string"}, "description": "Task UUIDs to depend on"}
                })),
                required: Some(vec!["task_id".to_string(), "dependency_ids".to_string()]),
            },
        },
        ToolDefinition {
            name: "remove_task_dependency".to_string(),
            description: "Remove a dependency from a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "dependency_id": {"type": "string", "description": "Dependency task UUID to remove"}
                })),
                required: Some(vec!["task_id".to_string(), "dependency_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_task_blockers".to_string(),
            description: "Get tasks that are blocking this task (uncompleted dependencies)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_tasks_blocked_by".to_string(),
            description: "Get tasks that are blocked by this task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_task_context".to_string(),
            description: "Get full context for a task (for agent execution)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_task_prompt".to_string(),
            description: "Get generated prompt for a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_decision".to_string(),
            description: "Record an architectural decision for a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "description": {"type": "string", "description": "Decision description"},
                    "rationale": {"type": "string", "description": "Why this decision was made"},
                    "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Alternatives considered"},
                    "chosen_option": {"type": "string", "description": "The chosen option"}
                })),
                required: Some(vec![
                    "task_id".to_string(),
                    "description".to_string(),
                    "rationale".to_string(),
                ]),
            },
        },
    ]
}

// ============================================================================
// Step Tools (4)
// ============================================================================

fn step_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_steps".to_string(),
            description: "List all steps for a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_step".to_string(),
            description: "Add a step to a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "description": {"type": "string", "description": "Step description"},
                    "verification": {"type": "string", "description": "How to verify completion"}
                })),
                required: Some(vec!["task_id".to_string(), "description".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_step".to_string(),
            description: "Update a step's status".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "step_id": {"type": "string", "description": "Step UUID"},
                    "status": {"type": "string", "description": "New status (pending, in_progress, completed, skipped)"}
                })),
                required: Some(vec!["step_id".to_string(), "status".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_step".to_string(),
            description: "Get a step by ID".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "step_id": {"type": "string", "description": "Step UUID"}
                })),
                required: Some(vec!["step_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_step".to_string(),
            description: "Delete a step".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "step_id": {"type": "string", "description": "Step UUID"}
                })),
                required: Some(vec!["step_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_step_progress".to_string(),
            description: "Get step completion progress for a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Constraint Tools (3)
// ============================================================================

fn constraint_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_constraints".to_string(),
            description: "List constraints for a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_constraint".to_string(),
            description: "Add a constraint to a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "constraint_type": {"type": "string", "description": "Type (performance, security, style, compatibility, other)"},
                    "description": {"type": "string", "description": "Constraint description"},
                    "severity": {"type": "string", "description": "Severity (low, medium, high, critical)"}
                })),
                required: Some(vec![
                    "plan_id".to_string(),
                    "constraint_type".to_string(),
                    "description".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "get_constraint".to_string(),
            description: "Get a constraint by ID".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "constraint_id": {"type": "string", "description": "Constraint UUID"}
                })),
                required: Some(vec!["constraint_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_constraint".to_string(),
            description: "Update a constraint's description, type, or enforced_by".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "constraint_id": {"type": "string", "description": "Constraint UUID"},
                    "description": {"type": "string", "description": "New description"},
                    "constraint_type": {"type": "string", "description": "New type (performance, security, style, compatibility, other)"},
                    "enforced_by": {"type": "string", "description": "New enforced_by"}
                })),
                required: Some(vec!["constraint_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_constraint".to_string(),
            description: "Delete a constraint".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "constraint_id": {"type": "string", "description": "Constraint UUID"}
                })),
                required: Some(vec!["constraint_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Release Tools (5)
// ============================================================================

fn release_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_releases".to_string(),
            description: "List releases for a project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "status": {"type": "string", "description": "Filter by status"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_release".to_string(),
            description: "Create a new release for a project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "version": {"type": "string", "description": "Version string (e.g., 1.0.0)"},
                    "title": {"type": "string", "description": "Release title"},
                    "description": {"type": "string", "description": "Release notes"},
                    "target_date": {"type": "string", "description": "Target date (ISO 8601)"}
                })),
                required: Some(vec!["project_id".to_string(), "version".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_release".to_string(),
            description: "Get release details with tasks and commits".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "release_id": {"type": "string", "description": "Release UUID"}
                })),
                required: Some(vec!["release_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_release".to_string(),
            description: "Update a release".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "release_id": {"type": "string", "description": "Release UUID"},
                    "status": {"type": "string", "description": "New status (planned, in_progress, released, cancelled)"},
                    "target_date": {"type": "string", "description": "New target date"},
                    "released_at": {"type": "string", "description": "Actual release date"},
                    "title": {"type": "string", "description": "New title"},
                    "description": {"type": "string", "description": "New description"}
                })),
                required: Some(vec!["release_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_release".to_string(),
            description: "Delete a release".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "release_id": {"type": "string", "description": "Release UUID"}
                })),
                required: Some(vec!["release_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_task_to_release".to_string(),
            description: "Add a task to a release".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "release_id": {"type": "string", "description": "Release UUID"},
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["release_id".to_string(), "task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_commit_to_release".to_string(),
            description: "Add a commit to a release".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "release_id": {"type": "string", "description": "Release UUID"},
                    "commit_sha": {"type": "string", "description": "Commit SHA"}
                })),
                required: Some(vec!["release_id".to_string(), "commit_sha".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Milestone Tools (5)
// ============================================================================

fn milestone_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_milestones".to_string(),
            description: "List milestones for a project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "status": {"type": "string", "description": "Filter by status (planned, open, in_progress, completed, closed)"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_milestone".to_string(),
            description: "Create a new milestone for a project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "title": {"type": "string", "description": "Milestone title"},
                    "description": {"type": "string", "description": "Milestone description"},
                    "target_date": {"type": "string", "description": "Target date (ISO 8601)"}
                })),
                required: Some(vec!["project_id".to_string(), "title".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_milestone".to_string(),
            description: "Get milestone details with tasks".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "milestone_id": {"type": "string", "description": "Milestone UUID"}
                })),
                required: Some(vec!["milestone_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_milestone".to_string(),
            description: "Update a milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "milestone_id": {"type": "string", "description": "Milestone UUID"},
                    "status": {"type": "string", "description": "New status (planned, open, in_progress, completed, closed)"},
                    "target_date": {"type": "string", "description": "New target date"},
                    "closed_at": {"type": "string", "description": "Closure date"},
                    "title": {"type": "string", "description": "New title"},
                    "description": {"type": "string", "description": "New description"}
                })),
                required: Some(vec!["milestone_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_milestone".to_string(),
            description: "Delete a milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "milestone_id": {"type": "string", "description": "Milestone UUID"}
                })),
                required: Some(vec!["milestone_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_milestone_progress".to_string(),
            description: "Get milestone completion progress".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "milestone_id": {"type": "string", "description": "Milestone UUID"}
                })),
                required: Some(vec!["milestone_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_task_to_milestone".to_string(),
            description: "Add a task to a milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "milestone_id": {"type": "string", "description": "Milestone UUID"},
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["milestone_id".to_string(), "task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_plan_to_milestone".to_string(),
            description: "Link a plan to a project milestone (TARGETS_MILESTONE relationship)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "milestone_id": {"type": "string", "description": "Milestone UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "milestone_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "unlink_plan_from_milestone".to_string(),
            description: "Unlink a plan from a project milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "milestone_id": {"type": "string", "description": "Milestone UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "milestone_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Commit Tools (4)
// ============================================================================

fn commit_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "create_commit".to_string(),
            description: "Register a git commit".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "sha": {"type": "string", "description": "Commit SHA"},
                    "message": {"type": "string", "description": "Commit message"},
                    "author": {"type": "string", "description": "Author name"},
                    "files_changed": {"type": "array", "items": {"type": "string"}, "description": "Files changed"},
                    "project_id": {"type": "string", "description": "Project UUID — enables incremental sync of changed files"}
                })),
                required: Some(vec!["sha".to_string(), "message".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_commit_to_task".to_string(),
            description: "Link a commit to a task (RESOLVED_BY relationship)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "commit_sha": {"type": "string", "description": "Commit SHA"}
                })),
                required: Some(vec!["task_id".to_string(), "commit_sha".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_commit_to_plan".to_string(),
            description: "Link a commit to a plan (RESULTED_IN relationship)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "commit_sha": {"type": "string", "description": "Commit SHA"}
                })),
                required: Some(vec!["plan_id".to_string(), "commit_sha".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_task_commits".to_string(),
            description: "Get commits linked to a task".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "task_id": {"type": "string", "description": "Task UUID"}
                })),
                required: Some(vec!["task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_plan_commits".to_string(),
            description: "Get commits linked to a plan".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"}
                })),
                required: Some(vec!["plan_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Code Exploration Tools (10)
// ============================================================================

fn code_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_code".to_string(),
            description: "Search code semantically across all projects".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                    "language": {"type": "string", "description": "Filter by language"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"},
                    "path_prefix": {"type": "string", "description": "Filter by path prefix (e.g. 'src/mcp/')"}
                })),
                required: Some(vec!["query".to_string()]),
            },
        },
        ToolDefinition {
            name: "search_project_code".to_string(),
            description: "Search code within a specific project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Project slug"},
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                    "language": {"type": "string", "description": "Filter by language"},
                    "path_prefix": {"type": "string", "description": "Filter by path prefix (e.g. 'src/mcp/')"}
                })),
                required: Some(vec!["project_slug".to_string(), "query".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_file_symbols".to_string(),
            description: "Get all symbols (functions, structs, traits) in a file".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {"type": "string", "description": "File path"}
                })),
                required: Some(vec!["file_path".to_string()]),
            },
        },
        ToolDefinition {
            name: "find_references".to_string(),
            description: "Find all references to a symbol".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "symbol": {"type": "string", "description": "Symbol name"},
                    "limit": {"type": "integer", "description": "Max results"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: Some(vec!["symbol".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_file_dependencies".to_string(),
            description: "Get file imports and files that depend on it".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {"type": "string", "description": "File path"}
                })),
                required: Some(vec!["file_path".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_call_graph".to_string(),
            description: "Get the call graph for a function".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "function": {"type": "string", "description": "Function name"},
                    "limit": {"type": "integer", "description": "Max depth/results"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: Some(vec!["function".to_string()]),
            },
        },
        ToolDefinition {
            name: "analyze_impact".to_string(),
            description: "Analyze the impact of changing a file or symbol".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "target": {"type": "string", "description": "File path or symbol name"},
                    "target_type": {"type": "string", "description": "Target type: 'file' or 'symbol' (auto-detected if omitted)"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: Some(vec!["target".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_architecture".to_string(),
            description: "Get codebase architecture overview (most connected files)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "find_similar_code".to_string(),
            description: "Find code similar to a given snippet".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "code_snippet": {"type": "string", "description": "Code to find similar matches for"},
                    "limit": {"type": "integer", "description": "Max results"},
                    "language": {"type": "string", "description": "Filter by language"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: Some(vec!["code_snippet".to_string()]),
            },
        },
        ToolDefinition {
            name: "find_trait_implementations".to_string(),
            description: "Find all implementations of a trait".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "trait_name": {"type": "string", "description": "Trait name"},
                    "limit": {"type": "integer", "description": "Max results"}
                })),
                required: Some(vec!["trait_name".to_string()]),
            },
        },
        ToolDefinition {
            name: "find_type_traits".to_string(),
            description: "Find all traits implemented by a type".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "type_name": {"type": "string", "description": "Type name (struct/enum)"},
                    "limit": {"type": "integer", "description": "Max results"}
                })),
                required: Some(vec!["type_name".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_impl_blocks".to_string(),
            description: "Get all impl blocks for a type".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "type_name": {"type": "string", "description": "Type name (struct/enum)"},
                    "limit": {"type": "integer", "description": "Max results"}
                })),
                required: Some(vec!["type_name".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Decision Tools (1)
// ============================================================================

fn decision_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "get_decision".to_string(),
            description: "Get a decision by ID".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "decision_id": {"type": "string", "description": "Decision UUID"}
                })),
                required: Some(vec!["decision_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_decision".to_string(),
            description: "Update a decision's description, rationale, or chosen_option".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "decision_id": {"type": "string", "description": "Decision UUID"},
                    "description": {"type": "string", "description": "New description"},
                    "rationale": {"type": "string", "description": "New rationale"},
                    "chosen_option": {"type": "string", "description": "New chosen option"}
                })),
                required: Some(vec!["decision_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_decision".to_string(),
            description: "Delete a decision".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "decision_id": {"type": "string", "description": "Decision UUID"}
                })),
                required: Some(vec!["decision_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "search_decisions".to_string(),
            description: "Search architectural decisions".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"}
                })),
                required: Some(vec!["query".to_string()]),
            },
        },
        ToolDefinition {
            name: "search_workspace_code".to_string(),
            description: "Search code across all projects in a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "workspace_slug": {"type": "string", "description": "Workspace slug"},
                    "query": {"type": "string", "description": "Search query"},
                    "language": {"type": "string", "description": "Filter by language"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                    "path_prefix": {"type": "string", "description": "Filter by path prefix (e.g. 'src/mcp/')"}
                })),
                required: Some(vec!["workspace_slug".to_string(), "query".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Sync & Watch Tools (4)
// ============================================================================

fn sync_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "sync_directory".to_string(),
            description: "Manually sync a directory to the knowledge graph".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "path": {"type": "string", "description": "Directory path"},
                    "project_id": {"type": "string", "description": "Optional project UUID"}
                })),
                required: Some(vec!["path".to_string()]),
            },
        },
        ToolDefinition {
            name: "start_watch".to_string(),
            description: "Start auto-sync file watcher".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "path": {"type": "string", "description": "Directory to watch"},
                    "project_id": {"type": "string", "description": "Optional project UUID"}
                })),
                required: Some(vec!["path".to_string()]),
            },
        },
        ToolDefinition {
            name: "stop_watch".to_string(),
            description: "Stop the file watcher".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
        ToolDefinition {
            name: "watch_status".to_string(),
            description: "Get file watcher status".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
    ]
}

// ============================================================================
// Meilisearch Maintenance Tools (2)
// ============================================================================

fn meilisearch_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "get_meilisearch_stats".to_string(),
            description: "Get Meilisearch code index statistics".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
        ToolDefinition {
            name: "delete_meilisearch_orphans".to_string(),
            description: "Delete documents without project_id from Meilisearch".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
        ToolDefinition {
            name: "cleanup_cross_project_calls".to_string(),
            description: "Delete CALLS relationships where caller and callee belong to different projects. Returns the number of deleted relationships.".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
    ]
}

// ============================================================================
// Knowledge Notes Tools (14)
// ============================================================================

fn note_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_notes".to_string(),
            description: "List notes with optional filters and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Filter by project UUID"},
                    "note_type": {"type": "string", "description": "Filter by type (guideline, gotcha, pattern, context, tip, observation, assertion)"},
                    "status": {"type": "string", "description": "Filter by status (comma-separated: active,needs_review,stale,obsolete,archived)"},
                    "importance": {"type": "string", "description": "Filter by importance (critical, high, medium, low)"},
                    "min_staleness": {"type": "number", "description": "Minimum staleness score (0.0-1.0)"},
                    "max_staleness": {"type": "number", "description": "Maximum staleness score (0.0-1.0)"},
                    "tags": {"type": "string", "description": "Filter by tags (comma-separated)"},
                    "search": {"type": "string", "description": "Search in content"},
                    "global_only": {"type": "boolean", "description": "Filter global notes only (no project_id)"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "create_note".to_string(),
            description: "Create a new knowledge note".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID (optional — omit for global/cross-project notes)"},
                    "note_type": {"type": "string", "description": "Type: guideline, gotcha, pattern, context, tip, observation, assertion"},
                    "content": {"type": "string", "description": "Note content"},
                    "importance": {"type": "string", "description": "Importance: critical, high, medium, low"},
                    "scope": {"type": "object", "description": "Scope (project, module, file, function, struct, trait)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                    "anchors": {"type": "array", "description": "Initial anchors to code entities"}
                })),
                required: Some(vec!["note_type".to_string(), "content".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_note".to_string(),
            description: "Get a note by ID".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"}
                })),
                required: Some(vec!["note_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_note".to_string(),
            description: "Update a note's content, importance, status, or tags".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"},
                    "content": {"type": "string", "description": "New content"},
                    "importance": {"type": "string", "description": "New importance level"},
                    "status": {"type": "string", "description": "New status"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags"}
                })),
                required: Some(vec!["note_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_note".to_string(),
            description: "Delete a note".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"}
                })),
                required: Some(vec!["note_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "search_notes".to_string(),
            description: "Search notes using semantic search".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "query": {"type": "string", "description": "Search query"},
                    "project_slug": {"type": "string", "description": "Filter by project slug"},
                    "note_type": {"type": "string", "description": "Filter by note type"},
                    "status": {"type": "string", "description": "Filter by status"},
                    "importance": {"type": "string", "description": "Filter by importance"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"}
                })),
                required: Some(vec!["query".to_string()]),
            },
        },
        ToolDefinition {
            name: "confirm_note".to_string(),
            description: "Confirm a note is still valid (resets staleness)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"}
                })),
                required: Some(vec!["note_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "invalidate_note".to_string(),
            description: "Mark a note as obsolete with a reason".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"},
                    "reason": {"type": "string", "description": "Reason for invalidation"}
                })),
                required: Some(vec!["note_id".to_string(), "reason".to_string()]),
            },
        },
        ToolDefinition {
            name: "supersede_note".to_string(),
            description: "Replace an old note with a new one".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "old_note_id": {"type": "string", "description": "ID of note to supersede"},
                    "project_id": {"type": "string", "description": "Project UUID (optional for global notes)"},
                    "note_type": {"type": "string", "description": "Type of new note"},
                    "content": {"type": "string", "description": "Content of new note"},
                    "importance": {"type": "string", "description": "Importance of new note"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for new note"}
                })),
                required: Some(vec![
                    "old_note_id".to_string(),
                    "note_type".to_string(),
                    "content".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "link_note_to_entity".to_string(),
            description: "Link a note to a code entity (file, function, struct, etc.)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"},
                    "entity_type": {"type": "string", "description": "Entity type: file, function, struct, trait, task, plan, etc."},
                    "entity_id": {"type": "string", "description": "Entity ID (file path or UUID)"}
                })),
                required: Some(vec![
                    "note_id".to_string(),
                    "entity_type".to_string(),
                    "entity_id".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "unlink_note_from_entity".to_string(),
            description: "Remove a link between a note and an entity".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "note_id": {"type": "string", "description": "Note UUID"},
                    "entity_type": {"type": "string", "description": "Entity type"},
                    "entity_id": {"type": "string", "description": "Entity ID"}
                })),
                required: Some(vec![
                    "note_id".to_string(),
                    "entity_type".to_string(),
                    "entity_id".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "get_context_notes".to_string(),
            description: "Get contextual notes for an entity (direct + propagated through graph)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "entity_type": {"type": "string", "description": "Entity type: file, function, struct, task, etc."},
                    "entity_id": {"type": "string", "description": "Entity ID"},
                    "max_depth": {"type": "integer", "description": "Max traversal depth (default 3)"},
                    "min_score": {"type": "number", "description": "Min relevance score (default 0.1)"}
                })),
                required: Some(vec!["entity_type".to_string(), "entity_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_notes_needing_review".to_string(),
            description: "Get notes that need human review (stale or needs_review status)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Optional project UUID filter"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "update_staleness_scores".to_string(),
            description: "Update staleness scores for all notes based on time decay".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({})),
                required: None,
            },
        },
        ToolDefinition {
            name: "list_project_notes".to_string(),
            description: "List notes for a specific project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "note_type": {"type": "string", "description": "Filter by type"},
                    "status": {"type": "string", "description": "Filter by status"},
                    "importance": {"type": "string", "description": "Filter by importance"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_propagated_notes".to_string(),
            description: "Get notes propagated through the graph (not directly attached)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "entity_type": {"type": "string", "description": "Entity type: file, function, struct, task, etc."},
                    "entity_id": {"type": "string", "description": "Entity ID"},
                    "max_depth": {"type": "integer", "description": "Max traversal depth (default 3)"},
                    "min_score": {"type": "number", "description": "Min relevance score (default 0.1)"}
                })),
                required: Some(vec!["entity_type".to_string(), "entity_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_entity_notes".to_string(),
            description: "Get notes directly attached to an entity".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "entity_type": {"type": "string", "description": "Entity type: file, function, struct, trait, task, plan, etc."},
                    "entity_id": {"type": "string", "description": "Entity ID (file path or UUID)"}
                })),
                required: Some(vec!["entity_type".to_string(), "entity_id".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Workspace Tools (30)
// ============================================================================

fn workspace_tools() -> Vec<ToolDefinition> {
    vec![
        // --- Workspace CRUD (5) ---
        ToolDefinition {
            name: "list_workspaces".to_string(),
            description: "List all workspaces with optional search and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "search": {"type": "string", "description": "Search in name/description"},
                    "limit": {"type": "integer", "description": "Max items (default 50, max 100)"},
                    "offset": {"type": "integer", "description": "Items to skip"},
                    "sort_by": {"type": "string", "description": "Sort field (name, created_at)"},
                    "sort_order": {"type": "string", "description": "asc or desc"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "create_workspace".to_string(),
            description: "Create a new workspace to group related projects".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "name": {"type": "string", "description": "Workspace name"},
                    "slug": {"type": "string", "description": "URL-safe identifier (auto-generated if not provided)"},
                    "description": {"type": "string", "description": "Workspace description"},
                    "metadata": {"type": "object", "description": "Optional metadata"}
                })),
                required: Some(vec!["name".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_workspace".to_string(),
            description: "Get workspace details by slug".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_workspace".to_string(),
            description: "Update a workspace's name, description, or metadata".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "name": {"type": "string", "description": "New name"},
                    "description": {"type": "string", "description": "New description"},
                    "metadata": {"type": "object", "description": "New metadata"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_workspace".to_string(),
            description: "Delete a workspace (does not delete associated projects)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        // --- Workspace Overview (1) ---
        ToolDefinition {
            name: "get_workspace_overview".to_string(),
            description:
                "Get workspace overview with projects, milestones, resources, and progress"
                    .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        // --- Workspace-Project Association (3) ---
        ToolDefinition {
            name: "list_workspace_projects".to_string(),
            description: "List all projects in a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_project_to_workspace".to_string(),
            description: "Add an existing project to a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "project_id": {"type": "string", "description": "Project UUID to add"}
                })),
                required: Some(vec!["slug".to_string(), "project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "remove_project_from_workspace".to_string(),
            description: "Remove a project from a workspace (does not delete the project)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "project_id": {"type": "string", "description": "Project UUID to remove"}
                })),
                required: Some(vec!["slug".to_string(), "project_id".to_string()]),
            },
        },
        // --- Workspace Milestones (7) ---
        ToolDefinition {
            name: "list_all_workspace_milestones".to_string(),
            description: "List all workspace milestones across all workspaces with optional filters and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "workspace_id": {"type": "string", "description": "Filter by workspace UUID"},
                    "status": {"type": "string", "description": "Filter by status (planned, open, in_progress, completed, closed)"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "list_workspace_milestones".to_string(),
            description: "List milestones for a workspace (cross-project milestones)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "status": {"type": "string", "description": "Filter by status (planned, open, in_progress, completed, closed)"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_workspace_milestone".to_string(),
            description: "Create a cross-project milestone in a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "title": {"type": "string", "description": "Milestone title"},
                    "description": {"type": "string", "description": "Milestone description"},
                    "target_date": {"type": "string", "description": "Target date (ISO 8601)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"}
                })),
                required: Some(vec!["slug".to_string(), "title".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_workspace_milestone".to_string(),
            description: "Get workspace milestone details with linked tasks".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Workspace milestone UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_workspace_milestone".to_string(),
            description: "Update a workspace milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Workspace milestone UUID"},
                    "title": {"type": "string", "description": "New title"},
                    "description": {"type": "string", "description": "New description"},
                    "status": {"type": "string", "description": "New status (planned, open, in_progress, completed, closed)"},
                    "target_date": {"type": "string", "description": "New target date"},
                    "closed_at": {"type": "string", "description": "Closure date"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_workspace_milestone".to_string(),
            description: "Delete a workspace milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Workspace milestone UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_task_to_workspace_milestone".to_string(),
            description: "Add a task from any project to a workspace milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Workspace milestone UUID"},
                    "task_id": {"type": "string", "description": "Task UUID to add"}
                })),
                required: Some(vec!["id".to_string(), "task_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_plan_to_workspace_milestone".to_string(),
            description: "Link a plan to a workspace milestone (TARGETS_MILESTONE relationship)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "id": {"type": "string", "description": "Workspace milestone UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "id".to_string()]),
            },
        },
        ToolDefinition {
            name: "unlink_plan_from_workspace_milestone".to_string(),
            description: "Unlink a plan from a workspace milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "plan_id": {"type": "string", "description": "Plan UUID"},
                    "id": {"type": "string", "description": "Workspace milestone UUID"}
                })),
                required: Some(vec!["plan_id".to_string(), "id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_workspace_milestone_progress".to_string(),
            description: "Get completion progress for a workspace milestone".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Workspace milestone UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        // --- Resources (5) ---
        ToolDefinition {
            name: "list_resources".to_string(),
            description: "List resources (API contracts, schemas) in a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "resource_type": {"type": "string", "description": "Filter by type (ApiContract, Protobuf, GraphqlSchema, etc.)"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_resource".to_string(),
            description: "Create a shared resource reference (API contract, schema file)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "name": {"type": "string", "description": "Resource name"},
                    "resource_type": {"type": "string", "description": "Type: ApiContract, Protobuf, GraphqlSchema, JsonSchema, DatabaseSchema, SharedTypes, Config, Documentation, Other"},
                    "file_path": {"type": "string", "description": "Path to the resource file"},
                    "url": {"type": "string", "description": "External URL (optional)"},
                    "format": {"type": "string", "description": "Format (openapi, protobuf, graphql)"},
                    "version": {"type": "string", "description": "Version string"},
                    "description": {"type": "string", "description": "Resource description"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                })),
                required: Some(vec![
                    "slug".to_string(),
                    "name".to_string(),
                    "resource_type".to_string(),
                    "file_path".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "get_resource".to_string(),
            description: "Get resource details".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Resource UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_resource".to_string(),
            description: "Update a resource's name, file_path, url, version, or description"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Resource UUID"},
                    "name": {"type": "string", "description": "New name"},
                    "file_path": {"type": "string", "description": "New file path"},
                    "url": {"type": "string", "description": "New URL"},
                    "version": {"type": "string", "description": "New version"},
                    "description": {"type": "string", "description": "New description"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_resource".to_string(),
            description: "Delete a resource".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Resource UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "link_resource_to_project".to_string(),
            description: "Link a resource to a project (implements or uses)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Resource UUID"},
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "link_type": {"type": "string", "description": "Link type: implements or uses"}
                })),
                required: Some(vec![
                    "id".to_string(),
                    "project_id".to_string(),
                    "link_type".to_string(),
                ]),
            },
        },
        // --- Components (7) ---
        ToolDefinition {
            name: "list_components".to_string(),
            description: "List components (services, databases, etc.) in a workspace".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "component_type": {"type": "string", "description": "Filter by type (Service, Frontend, Worker, Database, etc.)"},
                    "limit": {"type": "integer", "description": "Max items"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
        ToolDefinition {
            name: "create_component".to_string(),
            description: "Create a component in the workspace topology".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"},
                    "name": {"type": "string", "description": "Component name"},
                    "component_type": {"type": "string", "description": "Type: Service, Frontend, Worker, Database, MessageQueue, Cache, Gateway, External, Other"},
                    "description": {"type": "string", "description": "Component description"},
                    "runtime": {"type": "string", "description": "Runtime (docker, kubernetes, lambda)"},
                    "config": {"type": "object", "description": "Configuration (env vars, ports, etc.)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"}
                })),
                required: Some(vec![
                    "slug".to_string(),
                    "name".to_string(),
                    "component_type".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "get_component".to_string(),
            description: "Get component details".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Component UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "update_component".to_string(),
            description: "Update a component's name, description, runtime, config, or tags"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Component UUID"},
                    "name": {"type": "string", "description": "New name"},
                    "description": {"type": "string", "description": "New description"},
                    "runtime": {"type": "string", "description": "New runtime"},
                    "config": {"type": "object", "description": "New configuration"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_component".to_string(),
            description: "Delete a component".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Component UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "add_component_dependency".to_string(),
            description: "Add a dependency between components".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Source component UUID"},
                    "depends_on_id": {"type": "string", "description": "Target component UUID"},
                    "protocol": {"type": "string", "description": "Communication protocol (http, grpc, amqp, etc.)"},
                    "required": {"type": "boolean", "description": "Whether dependency is required"}
                })),
                required: Some(vec!["id".to_string(), "depends_on_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "remove_component_dependency".to_string(),
            description: "Remove a dependency between components".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Source component UUID"},
                    "dep_id": {"type": "string", "description": "Target component UUID to remove"}
                })),
                required: Some(vec!["id".to_string(), "dep_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "map_component_to_project".to_string(),
            description: "Map a component to a project (link source code)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Component UUID"},
                    "project_id": {"type": "string", "description": "Project UUID"}
                })),
                required: Some(vec!["id".to_string(), "project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_workspace_topology".to_string(),
            description: "Get the full topology graph of a workspace (components and dependencies)"
                .to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "slug": {"type": "string", "description": "Workspace slug"}
                })),
                required: Some(vec!["slug".to_string()]),
            },
        },
    ]
}

// ============================================================================
// Chat Tools (5)
// ============================================================================

fn chat_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_chat_messages".to_string(),
            description: "List message history for a chat session (chronological order)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "session_id": {"type": "string", "description": "Session UUID"},
                    "limit": {"type": "integer", "description": "Max messages to retrieve (default 50)"},
                    "offset": {"type": "integer", "description": "Messages to skip for pagination (default 0)"}
                })),
                required: Some(vec!["session_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "list_chat_sessions".to_string(),
            description: "List chat sessions with optional project filter and pagination".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Filter by project slug"},
                    "limit": {"type": "integer", "description": "Max items (default 50)"},
                    "offset": {"type": "integer", "description": "Items to skip"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "get_chat_session".to_string(),
            description: "Get chat session details by ID".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "session_id": {"type": "string", "description": "Session UUID"}
                })),
                required: Some(vec!["session_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "delete_chat_session".to_string(),
            description: "Delete a chat session".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "session_id": {"type": "string", "description": "Session UUID"}
                })),
                required: Some(vec!["session_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "chat_send_message".to_string(),
            description: "Send a chat message and wait for the complete response (non-streaming). Creates a new session or resumes an existing one.".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "message": {"type": "string", "description": "The user message to send"},
                    "cwd": {"type": "string", "description": "Working directory for Claude Code CLI"},
                    "session_id": {"type": "string", "description": "Session ID to resume (optional — creates new session if omitted)"},
                    "project_slug": {"type": "string", "description": "Project slug to associate with the session"},
                    "model": {"type": "string", "description": "Model override (default: from config)"},
                    "workspace_slug": {"type": "string", "description": "Workspace slug — resolves all project root_paths as --add-dir automatically"},
                    "add_dirs": {"type": "array", "items": {"type": "string"}, "description": "Additional directories to expose to Claude CLI (--add-dir)"}
                })),
                required: Some(vec!["message".to_string(), "cwd".to_string()]),
            },
        },
        // ================================================================
        // Feature Graph tools (6)
        // ================================================================
        ToolDefinition {
            name: "create_feature_graph".to_string(),
            description: "Create a named feature graph to capture code entities related to a feature".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "name": {"type": "string", "description": "Feature graph name (e.g. 'SSO', 'chat-streaming')"},
                    "description": {"type": "string", "description": "Description of what this feature covers"},
                    "project_id": {"type": "string", "description": "Project UUID"}
                })),
                required: Some(vec!["name".to_string(), "project_id".to_string()]),
            },
        },
        ToolDefinition {
            name: "get_feature_graph".to_string(),
            description: "Get feature graph details with all included entities (functions, files, structs)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Feature graph UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        ToolDefinition {
            name: "list_feature_graphs".to_string(),
            description: "List feature graphs, optionally filtered by project".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_id": {"type": "string", "description": "Optional project UUID filter"}
                })),
                required: None,
            },
        },
        ToolDefinition {
            name: "add_to_feature_graph".to_string(),
            description: "Add a code entity (file, function, struct, trait, enum) to a feature graph".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "feature_graph_id": {"type": "string", "description": "Feature graph UUID"},
                    "entity_type": {"type": "string", "description": "Entity type: file, function, struct, trait, enum"},
                    "entity_id": {"type": "string", "description": "Entity identifier (file path or symbol name)"},
                    "role": {"type": "string", "description": "Role of the entity in the feature: entry_point, core_logic, data_model, trait_contract, api_surface, support"}
                })),
                required: Some(vec![
                    "feature_graph_id".to_string(),
                    "entity_type".to_string(),
                    "entity_id".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "auto_build_feature_graph".to_string(),
            description: "Automatically build a feature graph from a function entry point by traversing the call graph (callers + callees)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "name": {"type": "string", "description": "Name for the feature graph"},
                    "description": {"type": "string", "description": "Optional description"},
                    "project_id": {"type": "string", "description": "Project UUID"},
                    "entry_function": {"type": "string", "description": "Function name to start traversal from"},
                    "depth": {"type": "integer", "description": "Traversal depth (1-5, default 2)"},
                    "include_relations": {"type": "array", "items": {"type": "string"}, "description": "Relation types to traverse (default: all). Options: calls, implements_trait, implements_for, imports"},
                    "filter_community": {"type": "boolean", "description": "Filter out transitive functions from different communities (default: true). Set to false to include all traversed functions."}
                })),
                required: Some(vec![
                    "name".to_string(),
                    "project_id".to_string(),
                    "entry_function".to_string(),
                ]),
            },
        },
        ToolDefinition {
            name: "delete_feature_graph".to_string(),
            description: "Delete a feature graph and all its entity relationships".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "id": {"type": "string", "description": "Feature graph UUID"}
                })),
                required: Some(vec!["id".to_string()]),
            },
        },
        // ================================================================
        // Structural Analytics (3)
        // ================================================================
        ToolDefinition {
            name: "get_code_communities".to_string(),
            description: "Identify code communities (clusters of tightly coupled files) in a project using structural analysis (Louvain clustering)".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Project slug"},
                    "min_size": {"type": "integer", "description": "Minimum community size to return (default: 2)"}
                })),
                required: Some(vec!["project_slug".to_string()]),
            },
        },
        // ================================================================
        // Implementation Planner (1)
        // ================================================================
        ToolDefinition {
            name: "plan_implementation".to_string(),
            description: "Analyze the knowledge graph to produce a DAG of implementation phases (sequential + parallel branches). Given entry points or a description, identifies relevant code zones, expands dependencies, and computes an ordered plan of modifications with risk levels and test files.".to_string(),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "project_slug": {"type": "string", "description": "Project slug to analyze"},
                    "description": {"type": "string", "description": "Human description of what to implement"},
                    "entry_points": {"type": "array", "items": {"type": "string"}, "description": "Explicit entry points (file paths or function names). If omitted, uses semantic search."},
                    "scope": {"type": "string", "enum": ["file", "module", "project"], "description": "Scope of analysis (default: module)"},
                    "auto_create_plan": {"type": "boolean", "description": "If true, auto-create a Plan MCP with Tasks/Steps from the result"}
                })),
                required: Some(vec![
                    "project_slug".to_string(),
                    "description".to_string(),
                ]),
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tools_count() {
        let tools = all_tools();
        assert_eq!(tools.len(), 150, "Expected 150 tools, got {}", tools.len());
    }

    #[test]
    fn test_tool_names_unique() {
        let tools = all_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "Tool names must be unique");
    }

    #[test]
    fn test_tool_serialization() {
        let tools = all_tools();
        for tool in &tools {
            let json = serde_json::to_string(tool).unwrap();
            assert!(json.contains(&tool.name));
            assert!(json.contains("inputSchema"));
        }
    }

    #[test]
    fn test_workspace_tools_count() {
        let tools = workspace_tools();
        assert_eq!(
            tools.len(),
            34,
            "Expected 34 workspace tools, got {}",
            tools.len()
        );
    }

    #[test]
    fn test_workspace_tools_names() {
        let tools = workspace_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // Workspace management (9)
        assert!(names.contains(&"list_workspaces"));
        assert!(names.contains(&"create_workspace"));
        assert!(names.contains(&"get_workspace"));
        assert!(names.contains(&"update_workspace"));
        assert!(names.contains(&"delete_workspace"));
        assert!(names.contains(&"get_workspace_overview"));
        assert!(names.contains(&"list_workspace_projects"));
        assert!(names.contains(&"add_project_to_workspace"));
        assert!(names.contains(&"remove_project_from_workspace"));

        // Workspace milestones (9)
        assert!(names.contains(&"list_workspace_milestones"));
        assert!(names.contains(&"create_workspace_milestone"));
        assert!(names.contains(&"get_workspace_milestone"));
        assert!(names.contains(&"update_workspace_milestone"));
        assert!(names.contains(&"delete_workspace_milestone"));
        assert!(names.contains(&"add_task_to_workspace_milestone"));
        assert!(names.contains(&"link_plan_to_workspace_milestone"));
        assert!(names.contains(&"unlink_plan_from_workspace_milestone"));
        assert!(names.contains(&"get_workspace_milestone_progress"));

        // Resources (5)
        assert!(names.contains(&"list_resources"));
        assert!(names.contains(&"create_resource"));
        assert!(names.contains(&"get_resource"));
        assert!(names.contains(&"delete_resource"));
        assert!(names.contains(&"link_resource_to_project"));

        // Components (8)
        assert!(names.contains(&"list_components"));
        assert!(names.contains(&"create_component"));
        assert!(names.contains(&"get_component"));
        assert!(names.contains(&"delete_component"));
        assert!(names.contains(&"add_component_dependency"));
        assert!(names.contains(&"remove_component_dependency"));
        assert!(names.contains(&"map_component_to_project"));
        assert!(names.contains(&"get_workspace_topology"));
    }

    #[test]
    fn test_workspace_tools_have_descriptions() {
        let tools = workspace_tools();
        for tool in &tools {
            assert!(
                !tool.description.is_empty(),
                "Tool {} has empty description",
                tool.name
            );
        }
    }

    #[test]
    fn test_workspace_tools_required_params() {
        let tools = workspace_tools();

        // Check create_workspace requires name
        let create_ws = tools.iter().find(|t| t.name == "create_workspace").unwrap();
        let required = create_ws.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"name".to_string()));

        // Check get_workspace requires slug
        let get_ws = tools.iter().find(|t| t.name == "get_workspace").unwrap();
        let required = get_ws.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"slug".to_string()));

        // Check create_resource requires slug, name, resource_type, file_path
        let create_res = tools.iter().find(|t| t.name == "create_resource").unwrap();
        let required = create_res.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"slug".to_string()));
        assert!(required.contains(&"name".to_string()));
        assert!(required.contains(&"resource_type".to_string()));
        assert!(required.contains(&"file_path".to_string()));

        // Check create_component requires slug, name, component_type
        let create_comp = tools.iter().find(|t| t.name == "create_component").unwrap();
        let required = create_comp.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"slug".to_string()));
        assert!(required.contains(&"name".to_string()));
        assert!(required.contains(&"component_type".to_string()));
    }

    #[test]
    fn test_all_tools_have_valid_input_schema() {
        let tools = all_tools();
        for tool in &tools {
            assert_eq!(
                tool.input_schema.schema_type, "object",
                "Tool {} input_schema type is not 'object'",
                tool.name
            );
        }
    }

    #[test]
    fn test_new_crud_tools_exist() {
        let tools = all_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // All 13 new tools from the CRUD completion plan
        let new_tools = [
            "update_project",
            "delete_task",
            "get_step",
            "delete_step",
            "get_constraint",
            "update_constraint",
            "delete_release",
            "delete_milestone",
            "get_decision",
            "update_decision",
            "delete_decision",
            "update_resource",
            "update_component",
        ];

        for tool_name in &new_tools {
            assert!(
                names.contains(tool_name),
                "Missing new CRUD tool: {}",
                tool_name
            );
        }
    }

    #[test]
    fn test_new_crud_tools_required_params() {
        let tools = all_tools();

        // update_project requires slug
        let t = tools.iter().find(|t| t.name == "update_project").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"slug".to_string()));

        // delete_task requires task_id
        let t = tools.iter().find(|t| t.name == "delete_task").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"task_id".to_string()));

        // get_step requires step_id
        let t = tools.iter().find(|t| t.name == "get_step").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"step_id".to_string()));

        // delete_step requires step_id
        let t = tools.iter().find(|t| t.name == "delete_step").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"step_id".to_string()));

        // get_constraint requires constraint_id
        let t = tools.iter().find(|t| t.name == "get_constraint").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"constraint_id".to_string()));

        // update_constraint requires constraint_id
        let t = tools
            .iter()
            .find(|t| t.name == "update_constraint")
            .unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"constraint_id".to_string()));

        // delete_release requires release_id
        let t = tools.iter().find(|t| t.name == "delete_release").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"release_id".to_string()));

        // delete_milestone requires milestone_id
        let t = tools.iter().find(|t| t.name == "delete_milestone").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"milestone_id".to_string()));

        // get_decision requires decision_id
        let t = tools.iter().find(|t| t.name == "get_decision").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"decision_id".to_string()));

        // update_decision requires decision_id
        let t = tools.iter().find(|t| t.name == "update_decision").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"decision_id".to_string()));

        // delete_decision requires decision_id
        let t = tools.iter().find(|t| t.name == "delete_decision").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"decision_id".to_string()));

        // update_resource requires id
        let t = tools.iter().find(|t| t.name == "update_resource").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"id".to_string()));

        // update_component requires id
        let t = tools.iter().find(|t| t.name == "update_component").unwrap();
        let req = t.input_schema.required.as_ref().unwrap();
        assert!(req.contains(&"id".to_string()));
    }

    #[test]
    fn test_new_crud_tools_have_properties() {
        let tools = all_tools();

        // update_project should have slug, name, description, root_path
        let t = tools.iter().find(|t| t.name == "update_project").unwrap();
        let props = t.input_schema.properties.as_ref().unwrap();
        assert!(props.get("slug").is_some());
        assert!(props.get("name").is_some());
        assert!(props.get("description").is_some());
        assert!(props.get("root_path").is_some());

        // update_constraint should have constraint_id, description, constraint_type, enforced_by
        let t = tools
            .iter()
            .find(|t| t.name == "update_constraint")
            .unwrap();
        let props = t.input_schema.properties.as_ref().unwrap();
        assert!(props.get("constraint_id").is_some());
        assert!(props.get("description").is_some());
        assert!(props.get("constraint_type").is_some());
        assert!(props.get("enforced_by").is_some());

        // update_decision should have decision_id, description, rationale, chosen_option
        let t = tools.iter().find(|t| t.name == "update_decision").unwrap();
        let props = t.input_schema.properties.as_ref().unwrap();
        assert!(props.get("decision_id").is_some());
        assert!(props.get("description").is_some());
        assert!(props.get("rationale").is_some());
        assert!(props.get("chosen_option").is_some());

        // update_resource should have id, name, file_path, url, version, description
        let t = tools.iter().find(|t| t.name == "update_resource").unwrap();
        let props = t.input_schema.properties.as_ref().unwrap();
        assert!(props.get("id").is_some());
        assert!(props.get("name").is_some());
        assert!(props.get("file_path").is_some());
        assert!(props.get("url").is_some());
        assert!(props.get("version").is_some());
        assert!(props.get("description").is_some());

        // update_component should have id, name, description, runtime, config, tags
        let t = tools.iter().find(|t| t.name == "update_component").unwrap();
        let props = t.input_schema.properties.as_ref().unwrap();
        assert!(props.get("id").is_some());
        assert!(props.get("name").is_some());
        assert!(props.get("description").is_some());
        assert!(props.get("runtime").is_some());
        assert!(props.get("config").is_some());
        assert!(props.get("tags").is_some());
    }

    #[test]
    fn test_workspace_tools_include_update_resource_and_component() {
        let tools = workspace_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"update_resource"));
        assert!(names.contains(&"update_component"));
    }

    #[test]
    fn test_list_all_workspace_milestones_tool_exists() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "list_all_workspace_milestones");
        assert!(
            tool.is_some(),
            "list_all_workspace_milestones tool must exist"
        );
    }

    #[test]
    fn test_list_all_workspace_milestones_tool_properties() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "list_all_workspace_milestones")
            .unwrap();

        let props = tool.input_schema.properties.as_ref().unwrap();
        assert!(props.get("workspace_id").is_some());
        assert!(props.get("status").is_some());
        assert!(props.get("limit").is_some());
        assert!(props.get("offset").is_some());

        // No required params - all optional
        assert!(tool.input_schema.required.is_none());
    }

    #[test]
    fn test_list_all_workspace_milestones_in_workspace_tools() {
        let tools = workspace_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"list_all_workspace_milestones"),
            "list_all_workspace_milestones should be in workspace tools"
        );
    }

    #[test]
    fn test_list_plans_tool_has_project_id() {
        let tools = all_tools();
        let tool = tools.iter().find(|t| t.name == "list_plans").unwrap();

        let props = tool.input_schema.properties.as_ref().unwrap();
        assert!(
            props.get("project_id").is_some(),
            "list_plans tool must have project_id parameter"
        );

        // project_id should be a string type
        let project_id_schema = props.get("project_id").unwrap();
        assert_eq!(project_id_schema["type"], "string");
    }

    #[test]
    fn test_list_workspace_milestones_tool_has_pagination() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "list_workspace_milestones")
            .unwrap();

        let props = tool.input_schema.properties.as_ref().unwrap();
        assert!(props.get("slug").is_some());
        assert!(props.get("status").is_some());
        assert!(props.get("limit").is_some());
        assert!(props.get("offset").is_some());

        // slug is required
        let required = tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"slug".to_string()));
    }
}
