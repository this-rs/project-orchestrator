//! MCP Tool definitions — Mega-tools architecture
//!
//! Instead of 160 individual tools, we expose ~18 mega-tools with an `action` parameter.
//! Each mega-tool groups all operations for a domain (e.g., project, plan, task).
//! The `action` parameter selects the specific operation; additional parameters vary by action.
//!
//! This reduces the tool count from 160 to 18, cutting LLM token overhead by ~60%.

use super::protocol::{InputSchema, ToolDefinition};
use serde_json::json;

/// Generate all tool definitions (mega-tools architecture)
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        project_tool(),
        plan_tool(),
        task_tool(),
        step_tool(),
        decision_tool(),
        constraint_tool(),
        release_tool(),
        milestone_tool(),
        commit_tool(),
        note_tool(),
        workspace_tool(),
        workspace_milestone_tool(),
        resource_tool(),
        component_tool(),
        chat_tool(),
        feature_graph_tool(),
        code_tool(),
        admin_tool(),
    ]
}

// ============================================================================
// Backward-compatible aliases → map old tool names to (mega_tool, action)
// ============================================================================

/// Map a legacy tool name to (mega_tool_name, action).
/// Returns None if the name is already a mega-tool or unknown.
pub fn resolve_legacy_alias(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        // Project
        "list_projects" => Some(("project", "list")),
        "create_project" => Some(("project", "create")),
        "get_project" => Some(("project", "get")),
        "update_project" => Some(("project", "update")),
        "delete_project" => Some(("project", "delete")),
        "sync_project" => Some(("project", "sync")),
        "get_project_roadmap" => Some(("project", "get_roadmap")),
        "list_project_plans" => Some(("project", "list_plans")),

        // Plan
        "list_plans" => Some(("plan", "list")),
        "create_plan" => Some(("plan", "create")),
        "get_plan" => Some(("plan", "get")),
        "update_plan_status" => Some(("plan", "update_status")),
        "delete_plan" => Some(("plan", "delete")),
        "link_plan_to_project" => Some(("plan", "link_to_project")),
        "unlink_plan_from_project" => Some(("plan", "unlink_from_project")),
        "get_dependency_graph" => Some(("plan", "get_dependency_graph")),
        "get_critical_path" => Some(("plan", "get_critical_path")),

        // Task
        "list_tasks" => Some(("task", "list")),
        "create_task" => Some(("task", "create")),
        "get_task" => Some(("task", "get")),
        "update_task" => Some(("task", "update")),
        "delete_task" => Some(("task", "delete")),
        "get_next_task" => Some(("task", "get_next")),
        "add_task_dependencies" => Some(("task", "add_dependencies")),
        "remove_task_dependency" => Some(("task", "remove_dependency")),
        "get_task_blockers" => Some(("task", "get_blockers")),
        "get_tasks_blocked_by" => Some(("task", "get_blocked_by")),
        "get_task_context" => Some(("task", "get_context")),
        "get_task_prompt" => Some(("task", "get_prompt")),

        // Step
        "list_steps" => Some(("step", "list")),
        "create_step" => Some(("step", "create")),
        "update_step" => Some(("step", "update")),
        "get_step" => Some(("step", "get")),
        "delete_step" => Some(("step", "delete")),
        "get_step_progress" => Some(("step", "get_progress")),

        // Decision
        "add_decision" => Some(("decision", "add")),
        "get_decision" => Some(("decision", "get")),
        "update_decision" => Some(("decision", "update")),
        "delete_decision" => Some(("decision", "delete")),
        "search_decisions" => Some(("decision", "search")),
        "search_decisions_semantic" => Some(("decision", "search_semantic")),
        "add_decision_affects" => Some(("decision", "add_affects")),
        "remove_decision_affects" => Some(("decision", "remove_affects")),
        "list_decision_affects" => Some(("decision", "list_affects")),
        "get_decisions_affecting" => Some(("decision", "get_affecting")),
        "supersede_decision" => Some(("decision", "supersede")),
        "get_decision_timeline" => Some(("decision", "get_timeline")),

        // Constraint
        "list_constraints" => Some(("constraint", "list")),
        "add_constraint" => Some(("constraint", "add")),
        "get_constraint" => Some(("constraint", "get")),
        "update_constraint" => Some(("constraint", "update")),
        "delete_constraint" => Some(("constraint", "delete")),

        // Release
        "list_releases" => Some(("release", "list")),
        "create_release" => Some(("release", "create")),
        "get_release" => Some(("release", "get")),
        "update_release" => Some(("release", "update")),
        "delete_release" => Some(("release", "delete")),
        "add_task_to_release" => Some(("release", "add_task")),
        "add_commit_to_release" => Some(("release", "add_commit")),
        "remove_commit_from_release" => Some(("release", "remove_commit")),

        // Milestone
        "list_milestones" => Some(("milestone", "list")),
        "create_milestone" => Some(("milestone", "create")),
        "get_milestone" => Some(("milestone", "get")),
        "update_milestone" => Some(("milestone", "update")),
        "delete_milestone" => Some(("milestone", "delete")),
        "get_milestone_progress" => Some(("milestone", "get_progress")),
        "add_task_to_milestone" => Some(("milestone", "add_task")),
        "link_plan_to_milestone" => Some(("milestone", "link_plan")),
        "unlink_plan_from_milestone" => Some(("milestone", "unlink_plan")),

        // Commit
        "create_commit" => Some(("commit", "create")),
        "link_commit_to_task" => Some(("commit", "link_to_task")),
        "link_commit_to_plan" => Some(("commit", "link_to_plan")),
        "get_task_commits" => Some(("commit", "get_task_commits")),
        "get_plan_commits" => Some(("commit", "get_plan_commits")),

        // Note
        "list_notes" => Some(("note", "list")),
        "create_note" => Some(("note", "create")),
        "get_note" => Some(("note", "get")),
        "update_note" => Some(("note", "update")),
        "delete_note" => Some(("note", "delete")),
        "search_notes" => Some(("note", "search")),
        "search_notes_semantic" => Some(("note", "search_semantic")),
        "confirm_note" => Some(("note", "confirm")),
        "invalidate_note" => Some(("note", "invalidate")),
        "supersede_note" => Some(("note", "supersede")),
        "link_note_to_entity" => Some(("note", "link_to_entity")),
        "unlink_note_from_entity" => Some(("note", "unlink_from_entity")),
        "get_context_notes" => Some(("note", "get_context")),
        "get_notes_needing_review" => Some(("note", "get_needing_review")),
        "list_project_notes" => Some(("note", "list_project")),
        "get_propagated_notes" => Some(("note", "get_propagated")),
        "get_propagated_knowledge" => Some(("note", "get_propagated_knowledge")),
        "get_entity_notes" => Some(("note", "get_entity")),

        // Workspace
        "list_workspaces" => Some(("workspace", "list")),
        "create_workspace" => Some(("workspace", "create")),
        "get_workspace" => Some(("workspace", "get")),
        "update_workspace" => Some(("workspace", "update")),
        "delete_workspace" => Some(("workspace", "delete")),
        "get_workspace_overview" => Some(("workspace", "get_overview")),
        "list_workspace_projects" => Some(("workspace", "list_projects")),
        "add_project_to_workspace" => Some(("workspace", "add_project")),
        "remove_project_from_workspace" => Some(("workspace", "remove_project")),
        "get_workspace_topology" => Some(("workspace", "get_topology")),

        // Workspace Milestone
        "list_all_workspace_milestones" => Some(("workspace_milestone", "list_all")),
        "list_workspace_milestones" => Some(("workspace_milestone", "list")),
        "create_workspace_milestone" => Some(("workspace_milestone", "create")),
        "get_workspace_milestone" => Some(("workspace_milestone", "get")),
        "update_workspace_milestone" => Some(("workspace_milestone", "update")),
        "delete_workspace_milestone" => Some(("workspace_milestone", "delete")),
        "add_task_to_workspace_milestone" => Some(("workspace_milestone", "add_task")),
        "link_plan_to_workspace_milestone" => Some(("workspace_milestone", "link_plan")),
        "unlink_plan_from_workspace_milestone" => Some(("workspace_milestone", "unlink_plan")),
        "get_workspace_milestone_progress" => Some(("workspace_milestone", "get_progress")),

        // Resource
        "list_resources" => Some(("resource", "list")),
        "create_resource" => Some(("resource", "create")),
        "get_resource" => Some(("resource", "get")),
        "update_resource" => Some(("resource", "update")),
        "delete_resource" => Some(("resource", "delete")),
        "link_resource_to_project" => Some(("resource", "link_to_project")),

        // Component
        "list_components" => Some(("component", "list")),
        "create_component" => Some(("component", "create")),
        "get_component" => Some(("component", "get")),
        "update_component" => Some(("component", "update")),
        "delete_component" => Some(("component", "delete")),
        "add_component_dependency" => Some(("component", "add_dependency")),
        "remove_component_dependency" => Some(("component", "remove_dependency")),
        "map_component_to_project" => Some(("component", "map_to_project")),

        // Chat
        "list_chat_sessions" => Some(("chat", "list_sessions")),
        "get_chat_session" => Some(("chat", "get_session")),
        "delete_chat_session" => Some(("chat", "delete_session")),
        "chat_send_message" => Some(("chat", "send_message")),
        "list_chat_messages" => Some(("chat", "list_messages")),

        // Feature Graph
        "create_feature_graph" => Some(("feature_graph", "create")),
        "get_feature_graph" => Some(("feature_graph", "get")),
        "list_feature_graphs" => Some(("feature_graph", "list")),
        "add_to_feature_graph" => Some(("feature_graph", "add_entity")),
        "auto_build_feature_graph" => Some(("feature_graph", "auto_build")),
        "delete_feature_graph" => Some(("feature_graph", "delete")),

        // Code
        "search_code" => Some(("code", "search")),
        "search_project_code" => Some(("code", "search_project")),
        "search_workspace_code" => Some(("code", "search_workspace")),
        "get_file_symbols" => Some(("code", "get_file_symbols")),
        "find_references" => Some(("code", "find_references")),
        "get_file_dependencies" => Some(("code", "get_file_dependencies")),
        "get_call_graph" => Some(("code", "get_call_graph")),
        "analyze_impact" => Some(("code", "analyze_impact")),
        "get_architecture" => Some(("code", "get_architecture")),
        "find_similar_code" => Some(("code", "find_similar")),
        "find_trait_implementations" => Some(("code", "find_trait_implementations")),
        "find_type_traits" => Some(("code", "find_type_traits")),
        "get_impl_blocks" => Some(("code", "get_impl_blocks")),
        "get_code_communities" => Some(("code", "get_communities")),
        "get_code_health" => Some(("code", "get_health")),
        "get_node_importance" => Some(("code", "get_node_importance")),
        "plan_implementation" => Some(("code", "plan_implementation")),

        // Admin
        "sync_directory" => Some(("admin", "sync_directory")),
        "start_watch" => Some(("admin", "start_watch")),
        "stop_watch" => Some(("admin", "stop_watch")),
        "watch_status" => Some(("admin", "watch_status")),
        "get_meilisearch_stats" => Some(("admin", "meilisearch_stats")),
        "delete_meilisearch_orphans" => Some(("admin", "delete_meilisearch_orphans")),
        "cleanup_cross_project_calls" => Some(("admin", "cleanup_cross_project_calls")),
        "cleanup_sync_data" => Some(("admin", "cleanup_sync_data")),
        "update_staleness_scores" => Some(("admin", "update_staleness_scores")),
        "update_energy_scores" => Some(("admin", "update_energy_scores")),
        "search_neurons" => Some(("admin", "search_neurons")),
        "reinforce_neurons" => Some(("admin", "reinforce_neurons")),
        "decay_synapses" => Some(("admin", "decay_synapses")),
        "backfill_synapses" => Some(("admin", "backfill_synapses")),
        "backfill_decision_embeddings" => Some(("admin", "backfill_decision_embeddings")),

        _ => None,
    }
}

// ============================================================================
// Mega-tool Definitions
// ============================================================================

fn project_tool() -> ToolDefinition {
    ToolDefinition {
        name: "project".to_string(),
        description: "Manage projects. Actions: list, create, get, update, delete, sync, get_roadmap, list_plans".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "sync", "get_roadmap", "list_plans"],
                    "description": "Operation to perform"
                },
                "slug": {"type": "string", "description": "Project slug (get/update/delete/sync/get_roadmap/list_plans)"},
                "name": {"type": "string", "description": "Project name (create/update)"},
                "root_path": {"type": "string", "description": "Path to codebase root (create/update)"},
                "description": {"type": "string", "description": "Project description (create/update)"},
                "search": {"type": "string", "description": "Search filter (list)"},
                "limit": {"type": "integer", "description": "Max items (list)"},
                "offset": {"type": "integer", "description": "Skip items (list)"},
                "sort_by": {"type": "string", "description": "Sort field (list)"},
                "sort_order": {"type": "string", "description": "asc or desc (list)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn plan_tool() -> ToolDefinition {
    ToolDefinition {
        name: "plan".to_string(),
        description: "Manage plans. Actions: list, create, get, update_status, delete, link_to_project, unlink_from_project, get_dependency_graph, get_critical_path".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update_status", "delete", "link_to_project", "unlink_from_project", "get_dependency_graph", "get_critical_path"],
                    "description": "Operation to perform"
                },
                "plan_id": {"type": "string", "description": "Plan UUID (get/update_status/delete/link_to_project/unlink_from_project/get_dependency_graph/get_critical_path)"},
                "project_id": {"type": "string", "description": "Project UUID (create/link_to_project/unlink_from_project/list)"},
                "title": {"type": "string", "description": "Plan title (create)"},
                "description": {"type": "string", "description": "Plan description (create)"},
                "priority": {"type": "integer", "description": "Priority 1-100 (create)"},
                "status": {"type": "string", "description": "New status (update_status): draft, approved, in_progress, completed, cancelled"},
                "search": {"type": "string", "description": "Search filter (list)"},
                "limit": {"type": "integer", "description": "Max items (list)"},
                "offset": {"type": "integer", "description": "Skip items (list)"},
                "sort_by": {"type": "string", "description": "Sort field (list)"},
                "sort_order": {"type": "string", "description": "asc or desc (list)"},
                "priority_min": {"type": "integer", "description": "Min priority filter (list)"},
                "priority_max": {"type": "integer", "description": "Max priority filter (list)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn task_tool() -> ToolDefinition {
    ToolDefinition {
        name: "task".to_string(),
        description: "Manage tasks. Actions: list, create, get, update, delete, get_next, add_dependencies, remove_dependency, get_blockers, get_blocked_by, get_context, get_prompt".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "get_next", "add_dependencies", "remove_dependency", "get_blockers", "get_blocked_by", "get_context", "get_prompt"],
                    "description": "Operation to perform"
                },
                "task_id": {"type": "string", "description": "Task UUID"},
                "plan_id": {"type": "string", "description": "Plan UUID (list/create/get_next/get_context/get_prompt)"},
                "title": {"type": "string", "description": "Task title (create)"},
                "description": {"type": "string", "description": "Task description (create)"},
                "priority": {"type": "integer", "description": "Priority (create/update)"},
                "status": {"type": "string", "description": "Status (update): pending, in_progress, blocked, completed, failed"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags (create/update)"},
                "assigned_to": {"type": "string", "description": "Assignee (update)"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "Criteria (create)"},
                "affected_files": {"type": "array", "items": {"type": "string"}, "description": "Files (create)"},
                "dependency_ids": {"type": "array", "items": {"type": "string"}, "description": "Task UUIDs to depend on (add_dependencies)"},
                "depends_on_task_id": {"type": "string", "description": "Dependency to remove (remove_dependency)"},
                "search": {"type": "string", "description": "Search filter (list)"},
                "limit": {"type": "integer", "description": "Max items (list)"},
                "offset": {"type": "integer", "description": "Skip items (list)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn step_tool() -> ToolDefinition {
    ToolDefinition {
        name: "step".to_string(),
        description:
            "Manage steps within tasks. Actions: list, create, update, get, delete, get_progress"
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "update", "get", "delete", "get_progress"],
                    "description": "Operation to perform"
                },
                "step_id": {"type": "string", "description": "Step UUID (update/get/delete)"},
                "task_id": {"type": "string", "description": "Task UUID (list/create/get_progress)"},
                "description": {"type": "string", "description": "Step description (create)"},
                "verification": {"type": "string", "description": "How to verify (create)"},
                "status": {"type": "string", "description": "New status (update): pending, in_progress, completed, skipped"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn decision_tool() -> ToolDefinition {
    ToolDefinition {
        name: "decision".to_string(),
        description: "Manage architectural decisions. Actions: add, get, update, delete, search, search_semantic, add_affects, remove_affects, list_affects, get_affecting, supersede, get_timeline"
            .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["add", "get", "update", "delete", "search", "search_semantic", "add_affects", "remove_affects", "list_affects", "get_affecting", "supersede", "get_timeline"],
                    "description": "Operation to perform"
                },
                "decision_id": {"type": "string", "description": "Decision UUID (get/update/delete/add_affects/remove_affects/list_affects)"},
                "task_id": {"type": "string", "description": "Task UUID (add/get_timeline)"},
                "description": {"type": "string", "description": "Decision description (add/update)"},
                "rationale": {"type": "string", "description": "Rationale (add/update)"},
                "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Alternatives considered (add)"},
                "chosen_option": {"type": "string", "description": "Chosen option (add/update)"},
                "status": {"type": "string", "description": "New status (update): proposed, accepted, deprecated, superseded"},
                "query": {"type": "string", "description": "Search query (search/search_semantic)"},
                "project_id": {"type": "string", "description": "Project UUID filter (search_semantic — post-query filtering)"},
                "entity_type": {"type": "string", "description": "Entity type (add_affects/remove_affects/get_affecting)"},
                "entity_id": {"type": "string", "description": "Entity identifier (add_affects/remove_affects)"},
                "impact_description": {"type": "string", "description": "Description of how the decision impacts the entity (add_affects)"},
                "superseded_by_id": {"type": "string", "description": "Decision UUID being superseded (supersede)"},
                "from": {"type": "string", "description": "Start date ISO filter (get_timeline)"},
                "to": {"type": "string", "description": "End date ISO filter (get_timeline)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn constraint_tool() -> ToolDefinition {
    ToolDefinition {
        name: "constraint".to_string(),
        description: "Manage plan constraints. Actions: list, add, get, update, delete".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "get", "update", "delete"],
                    "description": "Operation to perform"
                },
                "constraint_id": {"type": "string", "description": "Constraint UUID (get/update/delete)"},
                "plan_id": {"type": "string", "description": "Plan UUID (list/add)"},
                "constraint_type": {"type": "string", "description": "Type (add/update): performance, security, style, compatibility, other"},
                "description": {"type": "string", "description": "Description (add/update)"},
                "severity": {"type": "string", "description": "Severity (add): must, should, nice_to_have"},
                "enforced_by": {"type": "string", "description": "Enforcement (update)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn release_tool() -> ToolDefinition {
    ToolDefinition {
        name: "release".to_string(),
        description: "Manage releases. Actions: list, create, get, update, delete, add_task, add_commit, remove_commit".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "add_task", "add_commit", "remove_commit"],
                    "description": "Operation to perform"
                },
                "release_id": {"type": "string", "description": "Release UUID"},
                "project_id": {"type": "string", "description": "Project UUID (list/create)"},
                "version": {"type": "string", "description": "Version (create/update)"},
                "title": {"type": "string", "description": "Title (create/update)"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "status": {"type": "string", "description": "Status (update): planned, in_progress, released, cancelled"},
                "target_date": {"type": "string", "description": "Target date ISO (create/update)"},
                "task_id": {"type": "string", "description": "Task UUID (add_task)"},
                "commit_sha": {"type": "string", "description": "Commit SHA (add_commit/remove_commit)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn milestone_tool() -> ToolDefinition {
    ToolDefinition {
        name: "milestone".to_string(),
        description: "Manage milestones. Actions: list, create, get, update, delete, get_progress, add_task, link_plan, unlink_plan".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "get_progress", "add_task", "link_plan", "unlink_plan"],
                    "description": "Operation to perform"
                },
                "milestone_id": {"type": "string", "description": "Milestone UUID"},
                "project_id": {"type": "string", "description": "Project UUID (list/create)"},
                "title": {"type": "string", "description": "Title (create/update)"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "status": {"type": "string", "description": "Status (update)"},
                "target_date": {"type": "string", "description": "Target date ISO (create/update)"},
                "task_id": {"type": "string", "description": "Task UUID (add_task)"},
                "plan_id": {"type": "string", "description": "Plan UUID (link_plan/unlink_plan)"},
                "include_tasks": {"type": "boolean", "description": "Include tasks/plans/steps in get response (default false)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn commit_tool() -> ToolDefinition {
    ToolDefinition {
        name: "commit".to_string(),
        description: "Register and link git commits. Actions: create, link_to_task, link_to_plan, get_task_commits, get_plan_commits, get_commit_files, get_file_history".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["create", "link_to_task", "link_to_plan", "get_task_commits", "get_plan_commits", "get_commit_files", "get_file_history"],
                    "description": "Operation to perform"
                },
                "sha": {"type": "string", "description": "Commit SHA (create/link_to_task/link_to_plan/get_commit_files)"},
                "message": {"type": "string", "description": "Commit message (create)"},
                "author": {"type": "string", "description": "Author name (create)"},
                "files_changed": {"type": "array", "items": {"type": "string"}, "description": "Files changed (create)"},
                "project_id": {"type": "string", "description": "Project UUID for incremental sync (create)"},
                "task_id": {"type": "string", "description": "Task UUID (link_to_task/get_task_commits)"},
                "plan_id": {"type": "string", "description": "Plan UUID (link_to_plan/get_plan_commits)"},
                "commit_sha": {"type": "string", "description": "Alias for sha (link_to_task/link_to_plan)"},
                "file_path": {"type": "string", "description": "File path (get_file_history)"},
                "limit": {"type": "integer", "description": "Max results (get_file_history)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn note_tool() -> ToolDefinition {
    ToolDefinition {
        name: "note".to_string(),
        description: "Manage knowledge notes. Actions: list, create, get, update, delete, search, search_semantic, confirm, invalidate, supersede, link_to_entity, unlink_from_entity, get_context, get_needing_review, list_project, get_propagated, get_entity, get_context_knowledge, get_propagated_knowledge".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "search", "search_semantic", "confirm", "invalidate", "supersede", "link_to_entity", "unlink_from_entity", "get_context", "get_needing_review", "list_project", "get_propagated", "get_entity", "get_context_knowledge", "get_propagated_knowledge"],
                    "description": "Operation to perform"
                },
                "note_id": {"type": "string", "description": "Note UUID"},
                "project_id": {"type": "string", "description": "Project UUID"},
                "note_type": {"type": "string", "description": "Type: guideline, gotcha, pattern, context, tip, observation, assertion"},
                "content": {"type": "string", "description": "Note content (create/update)"},
                "importance": {"type": "string", "description": "Importance: critical, high, medium, low"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                "status": {"type": "string", "description": "Status filter (list)"},
                "query": {"type": "string", "description": "Search query (search/search_semantic)"},
                "superseded_by_id": {"type": "string", "description": "New note UUID (supersede)"},
                "entity_type": {"type": "string", "description": "Entity type (link_to_entity/unlink_from_entity/get_context/get_entity)"},
                "entity_id": {"type": "string", "description": "Entity identifier (link_to_entity/unlink_from_entity/get_context/get_entity)"},
                "slug": {"type": "string", "description": "Project slug (list_project/get_propagated)"},
                "file_path": {"type": "string", "description": "File path (get_propagated)"},
                "limit": {"type": "integer", "description": "Max items"},
                "offset": {"type": "integer", "description": "Skip items"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn workspace_tool() -> ToolDefinition {
    ToolDefinition {
        name: "workspace".to_string(),
        description: "Manage workspaces. Actions: list, create, get, update, delete, get_overview, list_projects, add_project, remove_project, get_topology".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "get_overview", "list_projects", "add_project", "remove_project", "get_topology"],
                    "description": "Operation to perform"
                },
                "slug": {"type": "string", "description": "Workspace slug"},
                "name": {"type": "string", "description": "Workspace name (create/update)"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "project_id": {"type": "string", "description": "Project UUID (add_project/remove_project)"},
                "role": {"type": "string", "description": "Project role in workspace (add_project)"},
                "limit": {"type": "integer", "description": "Max items (list)"},
                "offset": {"type": "integer", "description": "Skip items (list)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn workspace_milestone_tool() -> ToolDefinition {
    ToolDefinition {
        name: "workspace_milestone".to_string(),
        description: "Manage workspace milestones. Actions: list_all, list, create, get, update, delete, add_task, link_plan, unlink_plan, get_progress".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list_all", "list", "create", "get", "update", "delete", "add_task", "link_plan", "unlink_plan", "get_progress"],
                    "description": "Operation to perform"
                },
                "milestone_id": {"type": "string", "description": "Workspace milestone UUID"},
                "slug": {"type": "string", "description": "Workspace slug (list/create)"},
                "workspace_id": {"type": "string", "description": "Workspace UUID (list_all)"},
                "title": {"type": "string", "description": "Title (create/update)"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "status": {"type": "string", "description": "Status (update/list filter)"},
                "target_date": {"type": "string", "description": "Target date ISO (create/update)"},
                "task_id": {"type": "string", "description": "Task UUID (add_task)"},
                "plan_id": {"type": "string", "description": "Plan UUID (link_plan/unlink_plan)"},
                "limit": {"type": "integer", "description": "Max items"},
                "offset": {"type": "integer", "description": "Skip items"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn resource_tool() -> ToolDefinition {
    ToolDefinition {
        name: "resource".to_string(),
        description: "Manage workspace resources (API contracts, schemas). Actions: list, create, get, update, delete, link_to_project".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "link_to_project"],
                    "description": "Operation to perform"
                },
                "id": {"type": "string", "description": "Resource UUID (get/update/delete)"},
                "slug": {"type": "string", "description": "Workspace slug (list/create)"},
                "name": {"type": "string", "description": "Resource name (create/update)"},
                "resource_type": {"type": "string", "description": "Type (create): api_contract, schema, config, documentation, other"},
                "file_path": {"type": "string", "description": "File path (create/update)"},
                "url": {"type": "string", "description": "URL (create/update)"},
                "version": {"type": "string", "description": "Version (create/update)"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "project_id": {"type": "string", "description": "Project UUID (link_to_project)"},
                "resource_id": {"type": "string", "description": "Resource UUID (link_to_project)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn component_tool() -> ToolDefinition {
    ToolDefinition {
        name: "component".to_string(),
        description: "Manage workspace components (services, modules). Actions: list, create, get, update, delete, add_dependency, remove_dependency, map_to_project".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "update", "delete", "add_dependency", "remove_dependency", "map_to_project"],
                    "description": "Operation to perform"
                },
                "id": {"type": "string", "description": "Component UUID (get/update/delete)"},
                "slug": {"type": "string", "description": "Workspace slug (list/create)"},
                "name": {"type": "string", "description": "Component name (create/update)"},
                "component_type": {"type": "string", "description": "Type (create): service, library, database, queue, external"},
                "description": {"type": "string", "description": "Description (create/update)"},
                "runtime": {"type": "string", "description": "Runtime (create/update)"},
                "config": {"type": "object", "description": "Config (create/update)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags (create/update)"},
                "from_id": {"type": "string", "description": "Source component UUID (add_dependency/remove_dependency)"},
                "to_id": {"type": "string", "description": "Target component UUID (add_dependency/remove_dependency)"},
                "dependency_type": {"type": "string", "description": "Dependency type (add_dependency)"},
                "component_id": {"type": "string", "description": "Component UUID (map_to_project)"},
                "project_id": {"type": "string", "description": "Project UUID (map_to_project)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn chat_tool() -> ToolDefinition {
    ToolDefinition {
        name: "chat".to_string(),
        description: "Manage chat sessions. Actions: list_sessions, get_session, delete_session, send_message, list_messages, add_discussed, get_session_entities".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["list_sessions", "get_session", "delete_session", "send_message", "list_messages", "add_discussed", "get_session_entities"],
                    "description": "Operation to perform"
                },
                "session_id": {"type": "string", "description": "Session UUID"},
                "message": {"type": "string", "description": "Message to send (send_message)"},
                "cwd": {"type": "string", "description": "Working directory (send_message)"},
                "project_slug": {"type": "string", "description": "Project filter (list_sessions/send_message)"},
                "model": {"type": "string", "description": "Model (send_message)"},
                "permission_mode": {"type": "string", "description": "Permission mode (send_message)"},
                "workspace_slug": {"type": "string", "description": "Workspace slug (send_message)"},
                "add_dirs": {"type": "array", "items": {"type": "string"}, "description": "Additional directories (send_message)"},
                "entities": {"type": "array", "items": {"type": "object"}, "description": "Entities to mark as discussed (add_discussed): [{entity_type, entity_id}]"},
                "project_id": {"type": "string", "description": "Project UUID (get_session_entities — scoping filter)"},
                "limit": {"type": "integer", "description": "Max items"},
                "offset": {"type": "integer", "description": "Skip items"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn feature_graph_tool() -> ToolDefinition {
    ToolDefinition {
        name: "feature_graph".to_string(),
        description:
            "Manage feature graphs. Actions: create, get, list, add_entity, auto_build, delete"
                .to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "list", "add_entity", "auto_build", "delete"],
                    "description": "Operation to perform"
                },
                "id": {"type": "string", "description": "Feature graph UUID (get/delete)"},
                "feature_graph_id": {"type": "string", "description": "Feature graph UUID (add_entity)"},
                "project_id": {"type": "string", "description": "Project UUID (create/list/auto_build)"},
                "name": {"type": "string", "description": "Name (create/auto_build)"},
                "description": {"type": "string", "description": "Description (create/auto_build)"},
                "entity_type": {"type": "string", "description": "Entity type (add_entity)"},
                "entity_id": {"type": "string", "description": "Entity identifier (add_entity)"},
                "role": {"type": "string", "description": "Entity role (add_entity)"},
                "entry_function": {"type": "string", "description": "Entry function (auto_build)"},
                "depth": {"type": "integer", "description": "Traversal depth (auto_build)"},
                "include_relations": {"type": "array", "items": {"type": "string"}, "description": "Relation types to include (auto_build)"},
                "filter_community": {"type": "boolean", "description": "Filter by community (auto_build)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn code_tool() -> ToolDefinition {
    ToolDefinition {
        name: "code".to_string(),
        description: "Explore and analyze code. Actions: search, search_project, search_workspace, get_file_symbols, find_references, get_file_dependencies, get_call_graph, analyze_impact, get_architecture, find_similar, find_trait_implementations, find_type_traits, get_impl_blocks, get_communities, get_health, get_node_importance, plan_implementation, get_co_change_graph, get_file_co_changers".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["search", "search_project", "search_workspace", "get_file_symbols", "find_references", "get_file_dependencies", "get_call_graph", "analyze_impact", "get_architecture", "find_similar", "find_trait_implementations", "find_type_traits", "get_impl_blocks", "get_communities", "get_health", "get_node_importance", "plan_implementation", "get_co_change_graph", "get_file_co_changers"],
                    "description": "Operation to perform"
                },
                "query": {"type": "string", "description": "Search query (search/search_project/search_workspace)"},
                "slug": {"type": "string", "description": "Project slug (search_project)"},
                "workspace_slug": {"type": "string", "description": "Workspace slug (search_workspace)"},
                "project_slug": {"type": "string", "description": "Project slug (get_communities/get_health/get_node_importance/plan_implementation/get_architecture)"},
                "file_path": {"type": "string", "description": "File path (get_file_symbols/get_file_dependencies)"},
                "symbol": {"type": "string", "description": "Symbol name (find_references)"},
                "function": {"type": "string", "description": "Function name (get_call_graph)"},
                "target": {"type": "string", "description": "Target for impact analysis (analyze_impact)"},
                "code_snippet": {"type": "string", "description": "Code to find similar (find_similar)"},
                "trait_name": {"type": "string", "description": "Trait name (find_trait_implementations)"},
                "type_name": {"type": "string", "description": "Type name (find_type_traits/get_impl_blocks)"},
                "node_path": {"type": "string", "description": "Node path (get_node_importance)"},
                "node_type": {"type": "string", "description": "Node type (get_node_importance)"},
                "description": {"type": "string", "description": "Implementation description (plan_implementation)"},
                "entry_points": {"type": "array", "items": {"type": "string"}, "description": "Entry points (plan_implementation)"},
                "scope": {"type": "string", "description": "Scope: file, module, project (plan_implementation)"},
                "auto_create_plan": {"type": "boolean", "description": "Auto-create plan (plan_implementation)"},
                "path_prefix": {"type": "string", "description": "Path prefix filter (search)"},
                "min_size": {"type": "integer", "description": "Min community size (get_communities)"},
                "limit": {"type": "integer", "description": "Max results / depth (search/get_call_graph)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

fn admin_tool() -> ToolDefinition {
    ToolDefinition {
        name: "admin".to_string(),
        description: "Admin operations. Actions: sync_directory, start_watch, stop_watch, watch_status, meilisearch_stats, delete_meilisearch_orphans, cleanup_cross_project_calls, cleanup_sync_data, update_staleness_scores, update_energy_scores, search_neurons, reinforce_neurons, decay_synapses, backfill_synapses, backfill_decision_embeddings, backfill_touches, backfill_discussed, update_fabric_scores, bootstrap_knowledge_fabric".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Some(json!({
                "action": {
                    "type": "string",
                    "enum": ["sync_directory", "start_watch", "stop_watch", "watch_status", "meilisearch_stats", "delete_meilisearch_orphans", "cleanup_cross_project_calls", "cleanup_sync_data", "update_staleness_scores", "update_energy_scores", "search_neurons", "reinforce_neurons", "decay_synapses", "backfill_synapses", "backfill_decision_embeddings", "backfill_touches", "backfill_discussed", "update_fabric_scores", "bootstrap_knowledge_fabric"],
                    "description": "Operation to perform"
                },
                "path": {"type": "string", "description": "Directory path (sync_directory/start_watch)"},
                "project_id": {"type": "string", "description": "Project UUID (sync_directory/start_watch/update_staleness_scores/update_energy_scores/update_fabric_scores/bootstrap_knowledge_fabric)"},
                "query": {"type": "string", "description": "Search query (search_neurons)"},
                "note_ids": {"type": "array", "items": {"type": "string"}, "description": "Note UUIDs to co-activate (reinforce_neurons, min 2)"},
                "energy_boost": {"type": "number", "description": "Energy boost amount 0-1 (reinforce_neurons, default 0.2)"},
                "synapse_boost": {"type": "number", "description": "Synapse weight boost 0-1 (reinforce_neurons, default 0.05)"},
                "min_strength": {"type": "number", "description": "Min strength filter (search_neurons)"},
                "decay_amount": {"type": "number", "description": "Amount to subtract from each synapse weight (decay_synapses, default 0.01)"},
                "prune_threshold": {"type": "number", "description": "Prune synapses below this weight (decay_synapses, default 0.1)"},
                "limit": {"type": "integer", "description": "Max items (search_neurons)"}
            })),
            required: Some(vec!["action".to_string()]),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tools_count() {
        let tools = all_tools();
        assert_eq!(
            tools.len(),
            18,
            "Expected 18 mega-tools, got {}",
            tools.len()
        );
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
    fn test_all_tools_have_action_parameter() {
        let tools = all_tools();
        for tool in &tools {
            let props = tool.input_schema.properties.as_ref().unwrap();
            assert!(
                props.get("action").is_some(),
                "Tool {} must have an 'action' parameter",
                tool.name
            );
            let required = tool.input_schema.required.as_ref().unwrap();
            assert!(
                required.contains(&"action".to_string()),
                "Tool {} must require 'action'",
                tool.name
            );
        }
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
    fn test_legacy_alias_coverage() {
        // Verify all 160 old tool names are mapped
        let old_names = vec![
            "list_projects",
            "create_project",
            "get_project",
            "update_project",
            "delete_project",
            "sync_project",
            "get_project_roadmap",
            "list_project_plans",
            "list_plans",
            "create_plan",
            "get_plan",
            "update_plan_status",
            "delete_plan",
            "link_plan_to_project",
            "unlink_plan_from_project",
            "get_dependency_graph",
            "get_critical_path",
            "list_tasks",
            "create_task",
            "get_task",
            "update_task",
            "delete_task",
            "get_next_task",
            "add_task_dependencies",
            "remove_task_dependency",
            "get_task_blockers",
            "get_tasks_blocked_by",
            "get_task_context",
            "get_task_prompt",
            "add_decision",
            "list_steps",
            "create_step",
            "update_step",
            "get_step",
            "delete_step",
            "get_step_progress",
            "list_constraints",
            "add_constraint",
            "get_constraint",
            "update_constraint",
            "delete_constraint",
            "list_releases",
            "create_release",
            "get_release",
            "update_release",
            "delete_release",
            "add_task_to_release",
            "add_commit_to_release",
            "remove_commit_from_release",
            "list_milestones",
            "create_milestone",
            "get_milestone",
            "update_milestone",
            "delete_milestone",
            "get_milestone_progress",
            "add_task_to_milestone",
            "link_plan_to_milestone",
            "unlink_plan_from_milestone",
            "create_commit",
            "link_commit_to_task",
            "link_commit_to_plan",
            "get_task_commits",
            "get_plan_commits",
            "search_code",
            "search_project_code",
            "get_file_symbols",
            "find_references",
            "get_file_dependencies",
            "get_call_graph",
            "analyze_impact",
            "get_architecture",
            "find_similar_code",
            "find_trait_implementations",
            "find_type_traits",
            "get_impl_blocks",
            "get_decision",
            "update_decision",
            "delete_decision",
            "search_decisions",
            "search_decisions_semantic",
            "search_workspace_code",
            "sync_directory",
            "start_watch",
            "stop_watch",
            "watch_status",
            "get_meilisearch_stats",
            "delete_meilisearch_orphans",
            "cleanup_cross_project_calls",
            "cleanup_sync_data",
            "list_notes",
            "create_note",
            "get_note",
            "update_note",
            "delete_note",
            "search_notes",
            "search_notes_semantic",
            "confirm_note",
            "invalidate_note",
            "supersede_note",
            "link_note_to_entity",
            "unlink_note_from_entity",
            "get_context_notes",
            "get_notes_needing_review",
            "update_staleness_scores",
            "update_energy_scores",
            "search_neurons",
            "reinforce_neurons",
            "decay_synapses",
            "backfill_synapses",
            "list_project_notes",
            "get_propagated_notes",
            "get_entity_notes",
            "list_workspaces",
            "create_workspace",
            "get_workspace",
            "update_workspace",
            "delete_workspace",
            "get_workspace_overview",
            "list_workspace_projects",
            "add_project_to_workspace",
            "remove_project_from_workspace",
            "list_all_workspace_milestones",
            "list_workspace_milestones",
            "create_workspace_milestone",
            "get_workspace_milestone",
            "update_workspace_milestone",
            "delete_workspace_milestone",
            "add_task_to_workspace_milestone",
            "link_plan_to_workspace_milestone",
            "unlink_plan_from_workspace_milestone",
            "get_workspace_milestone_progress",
            "list_resources",
            "create_resource",
            "get_resource",
            "update_resource",
            "delete_resource",
            "link_resource_to_project",
            "list_components",
            "create_component",
            "get_component",
            "update_component",
            "delete_component",
            "add_component_dependency",
            "remove_component_dependency",
            "map_component_to_project",
            "get_workspace_topology",
            "list_chat_messages",
            "list_chat_sessions",
            "get_chat_session",
            "delete_chat_session",
            "chat_send_message",
            "create_feature_graph",
            "get_feature_graph",
            "list_feature_graphs",
            "add_to_feature_graph",
            "auto_build_feature_graph",
            "delete_feature_graph",
            "get_code_communities",
            "get_code_health",
            "get_node_importance",
            "plan_implementation",
        ];

        for name in &old_names {
            assert!(
                resolve_legacy_alias(name).is_some(),
                "Legacy tool '{}' has no alias mapping",
                name
            );
        }
    }

    #[test]
    fn test_legacy_alias_returns_none_for_mega_tools() {
        let mega_names = vec![
            "project",
            "plan",
            "task",
            "step",
            "decision",
            "constraint",
            "release",
            "milestone",
            "commit",
            "note",
            "workspace",
            "workspace_milestone",
            "resource",
            "component",
            "chat",
            "feature_graph",
            "code",
            "admin",
        ];
        for name in &mega_names {
            assert!(
                resolve_legacy_alias(name).is_none(),
                "Mega-tool '{}' should not be in legacy aliases",
                name
            );
        }
    }
}
