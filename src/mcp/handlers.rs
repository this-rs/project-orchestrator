//! MCP Tool handlers
//!
//! All tool calls are proxied to the REST API via McpHttpClient.
//! Mega-tool names (e.g. "project") are resolved to legacy names
//! (e.g. "list_projects") before HTTP routing.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::http_client::{extract_id, extract_optional_string, extract_string, McpHttpClient};

/// Handles MCP tool calls by proxying to the REST API.
pub struct ToolHandler {
    client: McpHttpClient,
}

impl ToolHandler {
    /// Create a new ToolHandler with an HTTP client.
    pub fn new(http_client: McpHttpClient) -> Self {
        Self {
            client: http_client,
        }
    }

    /// Get the HTTP client.
    fn http(&self) -> &McpHttpClient {
        &self.client
    }

    /// Resolve mega-tool names to legacy names for backward-compatible routing.
    ///
    /// If `name` is a mega-tool (e.g. "project"), extracts the `action` param and
    /// maps to the legacy tool name (e.g. "list_projects"). Legacy names pass through.
    fn resolve_mega_tool(&self, name: &str, args: &Value) -> Result<(String, Value)> {
        use super::tools::resolve_legacy_alias;

        // Check if already a legacy name → pass through
        if resolve_legacy_alias(name).is_some() {
            return Ok((name.to_string(), args.clone()));
        }

        // Mega-tool names: project, plan, task, step, decision, constraint,
        // release, milestone, commit, note, workspace, workspace_milestone,
        // resource, component, chat, feature_graph, code, admin
        let mega_tools: &[&str] = &[
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

        if !mega_tools.contains(&name) {
            // Unknown tool — return as-is, let downstream handle the error
            return Ok((name.to_string(), args.clone()));
        }

        // Extract action parameter
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Mega-tool '{}' requires an 'action' parameter", name))?;

        // Build the legacy tool name from (mega_tool, action)
        let legacy_name = self.mega_tool_to_legacy(name, action)?;

        // Return resolved name + original args (action param is ignored by handlers)
        Ok((legacy_name, args.clone()))
    }

    /// Map (mega_tool, action) → legacy tool name
    fn mega_tool_to_legacy(&self, tool: &str, action: &str) -> Result<String> {
        let name = match (tool, action) {
            // Project
            ("project", "list") => "list_projects",
            ("project", "create") => "create_project",
            ("project", "get") => "get_project",
            ("project", "update") => "update_project",
            ("project", "delete") => "delete_project",
            ("project", "sync") => "sync_project",
            ("project", "get_roadmap") => "get_project_roadmap",
            ("project", "list_plans") => "list_project_plans",

            // Plan
            ("plan", "list") => "list_plans",
            ("plan", "create") => "create_plan",
            ("plan", "get") => "get_plan",
            ("plan", "update_status") => "update_plan_status",
            ("plan", "delete") => "delete_plan",
            ("plan", "link_to_project") => "link_plan_to_project",
            ("plan", "unlink_from_project") => "unlink_plan_from_project",
            ("plan", "get_dependency_graph") => "get_dependency_graph",
            ("plan", "get_critical_path") => "get_critical_path",

            // Task
            ("task", "list") => "list_tasks",
            ("task", "create") => "create_task",
            ("task", "get") => "get_task",
            ("task", "update") => "update_task",
            ("task", "delete") => "delete_task",
            ("task", "get_next") => "get_next_task",
            ("task", "add_dependencies") => "add_task_dependencies",
            ("task", "remove_dependency") => "remove_task_dependency",
            ("task", "get_blockers") => "get_task_blockers",
            ("task", "get_blocked_by") => "get_tasks_blocked_by",
            ("task", "get_context") => "get_task_context",
            ("task", "get_prompt") => "get_task_prompt",

            // Step
            ("step", "list") => "list_steps",
            ("step", "create") => "create_step",
            ("step", "update") => "update_step",
            ("step", "get") => "get_step",
            ("step", "delete") => "delete_step",
            ("step", "get_progress") => "get_step_progress",

            // Decision
            ("decision", "add") => "add_decision",
            ("decision", "get") => "get_decision",
            ("decision", "update") => "update_decision",
            ("decision", "delete") => "delete_decision",
            ("decision", "search") => "search_decisions",
            ("decision", "search_semantic") => "search_decisions_semantic",
            ("decision", "add_affects") => "add_decision_affects",
            ("decision", "remove_affects") => "remove_decision_affects",
            ("decision", "list_affects") => "list_decision_affects",
            ("decision", "get_affecting") => "get_decisions_affecting",
            ("decision", "supersede") => "supersede_decision",
            ("decision", "get_timeline") => "get_decision_timeline",

            // Constraint
            ("constraint", "list") => "list_constraints",
            ("constraint", "add") => "add_constraint",
            ("constraint", "get") => "get_constraint",
            ("constraint", "update") => "update_constraint",
            ("constraint", "delete") => "delete_constraint",

            // Release
            ("release", "list") => "list_releases",
            ("release", "create") => "create_release",
            ("release", "get") => "get_release",
            ("release", "update") => "update_release",
            ("release", "delete") => "delete_release",
            ("release", "add_task") => "add_task_to_release",
            ("release", "add_commit") => "add_commit_to_release",
            ("release", "remove_commit") => "remove_commit_from_release",

            // Milestone
            ("milestone", "list") => "list_milestones",
            ("milestone", "create") => "create_milestone",
            ("milestone", "get") => "get_milestone",
            ("milestone", "update") => "update_milestone",
            ("milestone", "delete") => "delete_milestone",
            ("milestone", "get_progress") => "get_milestone_progress",
            ("milestone", "add_task") => "add_task_to_milestone",
            ("milestone", "link_plan") => "link_plan_to_milestone",
            ("milestone", "unlink_plan") => "unlink_plan_from_milestone",

            // Commit
            ("commit", "create") => "create_commit",
            ("commit", "link_to_task") => "link_commit_to_task",
            ("commit", "link_to_plan") => "link_commit_to_plan",
            ("commit", "get_task_commits") => "get_task_commits",
            ("commit", "get_plan_commits") => "get_plan_commits",
            ("commit", "get_commit_files") => "get_commit_files",
            ("commit", "get_file_history") => "get_file_history",

            // Note
            ("note", "list") => "list_notes",
            ("note", "create") => "create_note",
            ("note", "get") => "get_note",
            ("note", "update") => "update_note",
            ("note", "delete") => "delete_note",
            ("note", "search") => "search_notes",
            ("note", "search_semantic") => "search_notes_semantic",
            ("note", "confirm") => "confirm_note",
            ("note", "invalidate") => "invalidate_note",
            ("note", "supersede") => "supersede_note",
            ("note", "link_to_entity") => "link_note_to_entity",
            ("note", "unlink_from_entity") => "unlink_note_from_entity",
            ("note", "get_context") => "get_context_notes",
            ("note", "get_needing_review") => "get_notes_needing_review",
            ("note", "list_project") => "list_project_notes",
            ("note", "get_propagated") => "get_propagated_notes",
            ("note", "get_propagated_knowledge") => "get_propagated_knowledge",
            ("note", "get_context_knowledge") => "get_context_knowledge",
            ("note", "get_entity") => "get_entity_notes",

            // Workspace
            ("workspace", "list") => "list_workspaces",
            ("workspace", "create") => "create_workspace",
            ("workspace", "get") => "get_workspace",
            ("workspace", "update") => "update_workspace",
            ("workspace", "delete") => "delete_workspace",
            ("workspace", "get_overview") => "get_workspace_overview",
            ("workspace", "list_projects") => "list_workspace_projects",
            ("workspace", "add_project") => "add_project_to_workspace",
            ("workspace", "remove_project") => "remove_project_from_workspace",
            ("workspace", "get_topology") => "get_workspace_topology",

            // Workspace Milestone
            ("workspace_milestone", "list_all") => "list_all_workspace_milestones",
            ("workspace_milestone", "list") => "list_workspace_milestones",
            ("workspace_milestone", "create") => "create_workspace_milestone",
            ("workspace_milestone", "get") => "get_workspace_milestone",
            ("workspace_milestone", "update") => "update_workspace_milestone",
            ("workspace_milestone", "delete") => "delete_workspace_milestone",
            ("workspace_milestone", "add_task") => "add_task_to_workspace_milestone",
            ("workspace_milestone", "link_plan") => "link_plan_to_workspace_milestone",
            ("workspace_milestone", "unlink_plan") => "unlink_plan_from_workspace_milestone",
            ("workspace_milestone", "get_progress") => "get_workspace_milestone_progress",

            // Resource
            ("resource", "list") => "list_resources",
            ("resource", "create") => "create_resource",
            ("resource", "get") => "get_resource",
            ("resource", "update") => "update_resource",
            ("resource", "delete") => "delete_resource",
            ("resource", "link_to_project") => "link_resource_to_project",

            // Component
            ("component", "list") => "list_components",
            ("component", "create") => "create_component",
            ("component", "get") => "get_component",
            ("component", "update") => "update_component",
            ("component", "delete") => "delete_component",
            ("component", "add_dependency") => "add_component_dependency",
            ("component", "remove_dependency") => "remove_component_dependency",
            ("component", "map_to_project") => "map_component_to_project",

            // Chat
            ("chat", "list_sessions") => "list_chat_sessions",
            ("chat", "get_session") => "get_chat_session",
            ("chat", "delete_session") => "delete_chat_session",
            ("chat", "send_message") => "chat_send_message",
            ("chat", "list_messages") => "list_chat_messages",

            // Feature Graph
            ("feature_graph", "create") => "create_feature_graph",
            ("feature_graph", "get") => "get_feature_graph",
            ("feature_graph", "list") => "list_feature_graphs",
            ("feature_graph", "add_entity") => "add_to_feature_graph",
            ("feature_graph", "auto_build") => "auto_build_feature_graph",
            ("feature_graph", "delete") => "delete_feature_graph",

            // Code
            ("code", "search") => "search_code",
            ("code", "search_project") => "search_project_code",
            ("code", "search_workspace") => "search_workspace_code",
            ("code", "get_file_symbols") => "get_file_symbols",
            ("code", "find_references") => "find_references",
            ("code", "get_file_dependencies") => "get_file_dependencies",
            ("code", "get_call_graph") => "get_call_graph",
            ("code", "analyze_impact") => "analyze_impact",
            ("code", "get_architecture") => "get_architecture",
            ("code", "find_similar") => "find_similar_code",
            ("code", "find_trait_implementations") => "find_trait_implementations",
            ("code", "find_type_traits") => "find_type_traits",
            ("code", "get_impl_blocks") => "get_impl_blocks",
            ("code", "get_communities") => "get_code_communities",
            ("code", "get_health") => "get_code_health",
            ("code", "get_node_importance") => "get_node_importance",
            ("code", "plan_implementation") => "plan_implementation",
            ("code", "get_co_change_graph") => "get_co_change_graph",
            ("code", "get_file_co_changers") => "get_file_co_changers",

            // Admin
            ("admin", "sync_directory") => "sync_directory",
            ("admin", "start_watch") => "start_watch",
            ("admin", "stop_watch") => "stop_watch",
            ("admin", "watch_status") => "watch_status",
            ("admin", "meilisearch_stats") => "get_meilisearch_stats",
            ("admin", "delete_meilisearch_orphans") => "delete_meilisearch_orphans",
            ("admin", "cleanup_cross_project_calls") => "cleanup_cross_project_calls",
            ("admin", "cleanup_sync_data") => "cleanup_sync_data",
            ("admin", "update_staleness_scores") => "update_staleness_scores",
            ("admin", "update_energy_scores") => "update_energy_scores",
            ("admin", "search_neurons") => "search_neurons",
            ("admin", "reinforce_neurons") => "reinforce_neurons",
            ("admin", "decay_synapses") => "decay_synapses",
            ("admin", "backfill_synapses") => "backfill_synapses",
            ("admin", "backfill_decision_embeddings") => "backfill_decision_embeddings",
            ("admin", "backfill_touches") => "backfill_touches",

            _ => {
                return Err(anyhow!(
                    "Unknown action '{}' for mega-tool '{}'",
                    action,
                    tool
                ))
            }
        };
        Ok(name.to_string())
    }

    /// Handle a tool call and return the result as JSON.
    ///
    /// All calls are routed through the REST API via `try_handle_http()`.
    /// Mega-tool names (e.g. "project") are first resolved to legacy names
    /// (e.g. "list_projects") for backward-compatible routing.
    pub async fn handle(&self, name: &str, args: Option<Value>) -> Result<Value> {
        let args = args.unwrap_or(json!({}));

        // ── Mega-tool resolution ────────────────────────────────────────
        let (resolved_name, resolved_args) = self.resolve_mega_tool(name, &args)?;
        let name = resolved_name.as_str();
        let args = resolved_args;

        // ── HTTP routing ────────────────────────────────────────────────
        if let Some(result) = self.try_handle_http(name, &args).await? {
            return Ok(result);
        }

        Err(anyhow!("Unknown tool: '{}'", name))
    }

    // ========================================================================
    // HTTP-mode routing (migrated tools)
    // ========================================================================

    /// Try to handle a tool call via HTTP proxy.
    /// Returns `Ok(Some(value))` if the tool is migrated, `Ok(None)` if not yet migrated.
    async fn try_handle_http(&self, name: &str, args: &Value) -> Result<Option<Value>> {
        let http = self.http();

        match name {
            // ── P2: Projects (8 tools) ──────────────────────────────────
            "list_projects" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/projects").await?
                } else {
                    http.get_with_query("/api/projects", &query).await?
                };
                Ok(Some(result))
            }

            "create_project" => {
                let result = http.post("/api/projects", args).await?;
                Ok(Some(result))
            }

            "get_project" => {
                let slug = extract_string(args, "slug")?;
                let result = http.get(&format!("/api/projects/{}", slug)).await?;
                Ok(Some(result))
            }

            "update_project" => {
                let slug = extract_string(args, "slug")?;
                // Build PATCH body — only include fields that are present
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("root_path") {
                    body.insert("root_path".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/projects/{}", slug), &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "delete_project" => {
                let slug = extract_string(args, "slug")?;
                let result = http.delete(&format!("/api/projects/{}", slug)).await?;
                Ok(Some(result))
            }

            "sync_project" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(force) = args.get("force").and_then(|v| v.as_bool()) {
                    if force {
                        query.push(("force".to_string(), "true".to_string()));
                    }
                }
                let result = if query.is_empty() {
                    http.post(&format!("/api/projects/{}/sync", slug), &json!({}))
                        .await?
                } else {
                    // POST with query params — use get_with_query pattern but POST
                    let url = format!("/api/projects/{}/sync?force=true", slug);
                    http.post(&url, &json!({})).await?
                };
                Ok(Some(result))
            }

            "get_project_roadmap" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .get(&format!("/api/projects/{}/roadmap", project_id))
                    .await?;
                Ok(Some(result))
            }

            "list_project_plans" => {
                let slug = extract_string(args, "project_slug")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/plans", slug)).await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/plans", slug), &query)
                        .await?
                };
                Ok(Some(result))
            }

            // ── P3: Plans (9 tools) ──────────────────────────────────────
            "list_plans" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(v) = args.get("priority_min").and_then(|v| v.as_i64()) {
                    query.push(("priority_min".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("priority_max").and_then(|v| v.as_i64()) {
                    query.push(("priority_max".to_string(), v.to_string()));
                }
                if let Some(s) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/plans").await?
                } else {
                    http.get_with_query("/api/plans", &query).await?
                };
                Ok(Some(result))
            }

            "create_plan" => {
                let result = http.post("/api/plans", args).await?;
                Ok(Some(result))
            }

            "get_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}", plan_id)).await?;
                Ok(Some(result))
            }

            "update_plan_status" => {
                let plan_id = extract_id(args, "plan_id")?;
                let status = extract_string(args, "status")?;
                let body = json!({"status": status});
                let result = http
                    .patch(&format!("/api/plans/{}", plan_id), &body)
                    .await?;
                // REST returns 204 → null, normalize to {"updated": true}
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.delete(&format!("/api/plans/{}", plan_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "link_plan_to_project" => {
                let plan_id = extract_id(args, "plan_id")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .put(&format!("/api/plans/{}/project", plan_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_project" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!("/api/plans/{}/project", plan_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            "get_dependency_graph" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/dependency-graph", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "get_critical_path" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/critical-path", plan_id))
                    .await?;
                Ok(Some(result))
            }

            // ── P3: Constraints (5 tools) ───────────────────────────────────
            "list_constraints" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/constraints", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "add_constraint" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/constraints", plan_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let result = http
                    .get(&format!("/api/constraints/{}", constraint_id))
                    .await?;
                Ok(Some(result))
            }

            "update_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("constraint_type") {
                    body.insert("constraint_type".to_string(), v.clone());
                }
                if let Some(v) = args.get("enforced_by") {
                    body.insert("enforced_by".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/constraints/{}", constraint_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let result = http
                    .delete(&format!("/api/constraints/{}", constraint_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // ── P4: Tasks (13 tools) ────────────────────────────────────
            "list_tasks" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("plan_id").and_then(|v| v.as_str()) {
                    query.push(("plan_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    query.push(("workspace_slug".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(v) = args.get("priority_min").and_then(|v| v.as_i64()) {
                    query.push(("priority_min".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("priority_max").and_then(|v| v.as_i64()) {
                    query.push(("priority_max".to_string(), v.to_string()));
                }
                if let Some(s) = args.get("tags").and_then(|v| v.as_str()) {
                    query.push(("tags".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("assigned_to").and_then(|v| v.as_str()) {
                    query.push(("assigned_to".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/tasks").await?
                } else {
                    http.get_with_query("/api/tasks", &query).await?
                };
                Ok(Some(result))
            }

            "create_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/tasks", plan_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_task" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}", task_id)).await?;
                Ok(Some(result))
            }

            "update_task" => {
                let task_id = extract_id(args, "task_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("assigned_to") {
                    body.insert("assigned_to".to_string(), v.clone());
                }
                if let Some(v) = args.get("priority") {
                    body.insert("priority".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/tasks/{}", task_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_task" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.delete(&format!("/api/tasks/{}", task_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_next_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/next-task", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "add_task_dependencies" => {
                let task_id = extract_id(args, "task_id")?;
                // REST expects { "depends_on": [...] }, MCP sends { "dependency_ids": [...] }
                let dep_ids = args.get("dependency_ids").cloned().unwrap_or(json!([]));
                let body = json!({"depends_on": dep_ids});
                let result = http
                    .post(&format!("/api/tasks/{}/dependencies", task_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_task_dependency" => {
                let task_id = extract_id(args, "task_id")?;
                let dependency_id = extract_id(args, "dependency_id")?;
                let result = http
                    .delete(&format!(
                        "/api/tasks/{}/dependencies/{}",
                        task_id, dependency_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "get_task_blockers" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/blockers", task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_tasks_blocked_by" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/blocking", task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_task_context" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/tasks/{}/context", plan_id, task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_task_prompt" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/tasks/{}/prompt", plan_id, task_id))
                    .await?;
                Ok(Some(result))
            }

            "add_decision" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(&format!("/api/tasks/{}/decisions", task_id), args)
                    .await?;
                Ok(Some(result))
            }

            // ── P4: Steps (6 tools) ────────────────────────────────────
            "list_steps" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}/steps", task_id)).await?;
                Ok(Some(result))
            }

            "create_step" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(&format!("/api/tasks/{}/steps", task_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_step" => {
                let step_id = extract_id(args, "step_id")?;
                let result = http.get(&format!("/api/steps/{}", step_id)).await?;
                Ok(Some(result))
            }

            "update_step" => {
                let step_id = extract_id(args, "step_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/steps/{}", step_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_step" => {
                let step_id = extract_id(args, "step_id")?;
                let result = http.delete(&format!("/api/steps/{}", step_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_step_progress" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/steps/progress", task_id))
                    .await?;
                Ok(Some(result))
            }

            // ── P4: Decisions (4 tools) ────────────────────────────────
            "get_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http.get(&format!("/api/decisions/{}", decision_id)).await?;
                Ok(Some(result))
            }

            "update_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("rationale") {
                    body.insert("rationale".to_string(), v.clone());
                }
                if let Some(v) = args.get("chosen_option") {
                    body.insert("chosen_option".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/decisions/{}", decision_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http
                    .delete(&format!("/api/decisions/{}", decision_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "search_decisions" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("q".to_string(), query_str)];
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(s) = extract_optional_string(args, "project_slug") {
                    query.push(("project_slug".to_string(), s));
                }
                let result = http.get_with_query("/api/decisions/search", &query).await?;
                Ok(Some(result))
            }

            "search_decisions_semantic" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                let result = http
                    .get_with_query("/api/decisions/search-semantic", &query)
                    .await?;
                Ok(Some(result))
            }

            "add_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let mut body = serde_json::Map::new();
                body.insert(
                    "entity_type".to_string(),
                    Value::String(extract_string(args, "entity_type")?),
                );
                body.insert(
                    "entity_id".to_string(),
                    Value::String(extract_string(args, "entity_id")?),
                );
                if let Some(v) = args.get("impact_description") {
                    body.insert("impact_description".to_string(), v.clone());
                }
                let result = http
                    .post(
                        &format!("/api/decisions/{}/affects", decision_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "remove_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let result = http
                    .delete(&format!(
                        "/api/decisions/{}/affects/{}/{}",
                        decision_id, entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "list_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http
                    .get(&format!("/api/decisions/{}/affects", decision_id))
                    .await?;
                Ok(Some(result))
            }

            "get_decisions_affecting" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(s) = extract_optional_string(args, "status") {
                    query.push(("status".to_string(), s));
                }
                let result = http
                    .get_with_query("/api/decisions/affecting", &query)
                    .await?;
                Ok(Some(result))
            }

            "supersede_decision" => {
                let new_id = extract_id(args, "decision_id")?;
                let old_id = extract_id(args, "superseded_by_id")?;
                let result = http
                    .post(
                        &format!("/api/decisions/{}/supersedes/{}", new_id, old_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "get_decision_timeline" => {
                let mut query = Vec::new();
                if let Some(tid) = extract_optional_string(args, "task_id") {
                    query.push(("task_id".to_string(), tid));
                }
                if let Some(f) = extract_optional_string(args, "from") {
                    query.push(("from".to_string(), f));
                }
                if let Some(t) = extract_optional_string(args, "to") {
                    query.push(("to".to_string(), t));
                }
                let result = http
                    .get_with_query("/api/decisions/timeline", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── P5: Milestones (9 tools) ────────────────────────────────
            "list_milestones" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let path = format!("/api/projects/{}/milestones", project_id);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_milestone" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(&format!("/api/projects/{}/milestones", project_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let result = http
                    .get(&format!("/api/milestones/{}", milestone_id))
                    .await?;
                Ok(Some(result))
            }

            "update_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                if let Some(v) = args.get("closed_at") {
                    body.insert("closed_at".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/milestones/{}", milestone_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let result = http
                    .delete(&format!("/api/milestones/{}", milestone_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_milestone_progress" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let result = http
                    .get(&format!("/api/milestones/{}/progress", milestone_id))
                    .await?;
                Ok(Some(result))
            }

            "add_task_to_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/milestones/{}/tasks", milestone_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "link_plan_to_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let body = json!({"plan_id": plan_id});
                let result = http
                    .post(&format!("/api/milestones/{}/plans", milestone_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!(
                        "/api/milestones/{}/plans/{}",
                        milestone_id, plan_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            // ── P5: Releases (8 tools) ─────────────────────────────────
            "list_releases" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let path = format!("/api/projects/{}/releases", project_id);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_release" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(&format!("/api/projects/{}/releases", project_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_release" => {
                let release_id = extract_id(args, "release_id")?;
                let result = http.get(&format!("/api/releases/{}", release_id)).await?;
                Ok(Some(result))
            }

            "update_release" => {
                let release_id = extract_id(args, "release_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                if let Some(v) = args.get("released_at") {
                    body.insert("released_at".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/releases/{}", release_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_release" => {
                let release_id = extract_id(args, "release_id")?;
                let result = http
                    .delete(&format!("/api/releases/{}", release_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_task_to_release" => {
                let release_id = extract_id(args, "release_id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/releases/{}/tasks", release_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "add_commit_to_release" => {
                let release_id = extract_id(args, "release_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/releases/{}/commits", release_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_commit_from_release" => {
                let release_id = extract_id(args, "release_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                let result = http
                    .delete(&format!(
                        "/api/releases/{}/commits/{}",
                        release_id, commit_sha
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            // ── P6: Commits (5 tools) ───────────────────────────────────
            "create_commit" => {
                // Map MCP field names to REST field names: sha→hash
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("sha") {
                    body.insert("hash".to_string(), v.clone());
                }
                if let Some(v) = args.get("message") {
                    body.insert("message".to_string(), v.clone());
                }
                if let Some(v) = args.get("author") {
                    body.insert("author".to_string(), v.clone());
                }
                if let Some(v) = args.get("files_changed") {
                    body.insert("files_changed".to_string(), v.clone());
                }
                if let Some(v) = args.get("project_id") {
                    body.insert("project_id".to_string(), v.clone());
                }
                let result = http.post("/api/commits", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            "link_commit_to_task" => {
                let task_id = extract_id(args, "task_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/tasks/{}/commits", task_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "link_commit_to_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/plans/{}/commits", plan_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "get_task_commits" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}/commits", task_id)).await?;
                Ok(Some(result))
            }

            "get_plan_commits" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}/commits", plan_id)).await?;
                Ok(Some(result))
            }

            "get_commit_files" => {
                let sha = args
                    .get("sha")
                    .or_else(|| args.get("commit_sha"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("sha is required"))?;
                let result = http.get(&format!("/api/commits/{}/files", sha)).await?;
                Ok(Some(result))
            }

            "get_file_history" => {
                let file_path = args
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("file_path is required"))?;
                let mut query_params = vec![("path".to_string(), file_path.to_string())];
                if let Some(limit) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), limit.to_string()));
                }
                let result = http.get_with_query("/api/files/history", &query_params).await?;
                Ok(Some(result))
            }

            // ── P7: Notes & Knowledge (23 tools) ──────────────────────────

            // --- CRUD (5) ---
            "list_notes" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("note_type").and_then(|v| v.as_str()) {
                    query.push(("note_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("importance").and_then(|v| v.as_str()) {
                    query.push(("importance".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_staleness").and_then(|v| v.as_f64()) {
                    query.push(("min_staleness".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_staleness").and_then(|v| v.as_f64()) {
                    query.push(("max_staleness".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("tags").and_then(|v| v.as_str()) {
                    query.push(("tags".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("global_only").and_then(|v| v.as_bool()) {
                    query.push(("global_only".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/notes").await?
                } else {
                    http.get_with_query("/api/notes", &query).await?
                };
                Ok(Some(result))
            }

            "create_note" => {
                // Forward full args as body (REST ignores unknown fields via serde)
                let result = http.post("/api/notes", args).await?;
                Ok(Some(result))
            }

            "get_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http.get(&format!("/api/notes/{}", note_id)).await?;
                Ok(Some(result))
            }

            "update_note" => {
                let note_id = extract_id(args, "note_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("content") {
                    body.insert("content".to_string(), v.clone());
                }
                if let Some(v) = args.get("importance") {
                    body.insert("importance".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/notes/{}", note_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http.delete(&format!("/api/notes/{}", note_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // --- Search (3) ---
            "search_notes" => {
                // MCP param "query" → REST param "q"
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("q".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("note_type").and_then(|v| v.as_str()) {
                    query.push(("note_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("importance").and_then(|v| v.as_str()) {
                    query.push(("importance".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/search", &query).await?;
                Ok(Some(result))
            }

            "search_notes_semantic" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    query.push(("workspace_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/search-semantic", &query)
                    .await?;
                Ok(Some(result))
            }

            "search_neurons" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_results").and_then(|v| v.as_i64()) {
                    query.push(("max_results".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_hops").and_then(|v| v.as_i64()) {
                    query.push(("max_hops".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/neurons/search", &query)
                    .await?;
                Ok(Some(result))
            }

            // --- Lifecycle (3) ---
            "confirm_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http
                    .post(&format!("/api/notes/{}/confirm", note_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "invalidate_note" => {
                let note_id = extract_id(args, "note_id")?;
                let reason = extract_string(args, "reason")?;
                let body = json!({"reason": reason});
                let result = http
                    .post(&format!("/api/notes/{}/invalidate", note_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "supersede_note" => {
                let old_note_id = extract_id(args, "old_note_id")?;
                // Forward remaining args as body (old_note_id extracted for URL path)
                let result = http
                    .post(&format!("/api/notes/{}/supersede", old_note_id), args)
                    .await?;
                Ok(Some(result))
            }

            // --- Linking (2) ---
            "link_note_to_entity" => {
                let note_id = extract_id(args, "note_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let body = json!({"entity_type": entity_type, "entity_id": entity_id});
                let result = http
                    .post(&format!("/api/notes/{}/links", note_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_note_from_entity" => {
                let note_id = extract_id(args, "note_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let result = http
                    .delete(&format!(
                        "/api/notes/{}/links/{}/{}",
                        note_id, entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            // --- Retrieval (5) ---
            "get_context_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/context", &query).await?;
                Ok(Some(result))
            }

            "get_notes_needing_review" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/notes/needs-review").await?
                } else {
                    http.get_with_query("/api/notes/needs-review", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "get_propagated_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                // Forward relation_types as comma-separated string
                if let Some(v) = args.get("relation_types").and_then(|v| v.as_str()) {
                    query.push(("relation_types".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/propagated", &query).await?;
                Ok(Some(result))
            }

            "get_context_knowledge" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/context-knowledge", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_propagated_knowledge" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("relation_types").and_then(|v| v.as_str()) {
                    query.push(("relation_types".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/propagated-knowledge", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_entity_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let result = http
                    .get(&format!(
                        "/api/entities/{}/{}/notes",
                        entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(result))
            }

            "list_project_notes" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/notes", project_id))
                        .await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/notes", project_id), &query)
                        .await?
                };
                Ok(Some(result))
            }

            // --- Admin (5) ---
            "update_staleness_scores" => {
                let result = http.post("/api/notes/update-staleness", &json!({})).await?;
                Ok(Some(result))
            }

            "update_energy_scores" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("half_life") {
                    body.insert("half_life".to_string(), v.clone());
                }
                let result = http
                    .post("/api/notes/update-energy", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "reinforce_neurons" => {
                // Forward full args as body (note_ids, energy_boost, synapse_boost)
                let result = http.post("/api/notes/neurons/reinforce", args).await?;
                Ok(Some(result))
            }

            "decay_synapses" => {
                // Forward full args as body (decay_amount, prune_threshold)
                let result = http.post("/api/notes/neurons/decay", args).await?;
                Ok(Some(result))
            }

            "backfill_synapses" => {
                // REST runs async (202 Accepted), MCP Direct was sync.
                // We forward to the async endpoint and return the 202 response.
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("batch_size") {
                    body.insert("batch_size".to_string(), v.clone());
                }
                if let Some(v) = args.get("min_similarity") {
                    body.insert("min_similarity".to_string(), v.clone());
                }
                if let Some(v) = args.get("max_neighbors") {
                    body.insert("max_neighbors".to_string(), v.clone());
                }
                let result = http
                    .post("/api/admin/backfill-synapses", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "backfill_decision_embeddings" => {
                let result = http
                    .post("/api/admin/backfill-decision-embeddings", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "backfill_touches" => {
                let project_slug = extract_string(args, "project_slug")
                    .or_else(|_| extract_string(args, "slug"))?;
                let result = http
                    .post(
                        &format!("/api/projects/{}/backfill-touches", project_slug),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            // ── P8: Workspaces (34 tools) ──────────────────────────────────

            // --- Workspace CRUD (5) ---
            "list_workspaces" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/workspaces").await?
                } else {
                    http.get_with_query("/api/workspaces", &query).await?
                };
                Ok(Some(result))
            }

            "create_workspace" => {
                let result = http.post("/api/workspaces", args).await?;
                Ok(Some(result))
            }

            "get_workspace" => {
                let slug = extract_string(args, "slug")?;
                let result = http.get(&format!("/api/workspaces/{}", slug)).await?;
                Ok(Some(result))
            }

            "update_workspace" => {
                let slug = extract_string(args, "slug")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("metadata") {
                    body.insert("metadata".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/workspaces/{}", slug), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_workspace" => {
                let slug = extract_string(args, "slug")?;
                let result = http.delete(&format!("/api/workspaces/{}", slug)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // --- Workspace Overview & Projects (4) ---
            "get_workspace_overview" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/overview", slug))
                    .await?;
                Ok(Some(result))
            }

            "list_workspace_projects" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/projects", slug))
                    .await?;
                Ok(Some(result))
            }

            "add_project_to_workspace" => {
                let slug = extract_string(args, "slug")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .post(&format!("/api/workspaces/{}/projects", slug), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_project_from_workspace" => {
                let slug = extract_string(args, "slug")?;
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .delete(&format!("/api/workspaces/{}/projects/{}", slug, project_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            // --- Workspace Topology (1) ---
            "get_workspace_topology" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/topology", slug))
                    .await?;
                Ok(Some(result))
            }

            // --- Workspace Milestones (10) ---
            "list_all_workspace_milestones" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("workspace_id").and_then(|v| v.as_str()) {
                    query.push(("workspace_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/workspace-milestones").await?
                } else {
                    http.get_with_query("/api/workspace-milestones", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "list_workspace_milestones" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/milestones", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_workspace_milestone" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/milestones", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let result = http
                    .get(&format!("/api/workspace-milestones/{}", id))
                    .await?;
                Ok(Some(result))
            }

            "update_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/workspace-milestones/{}", id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let result = http
                    .delete(&format!("/api/workspace-milestones/{}", id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_task_to_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/workspace-milestones/{}/tasks", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "link_plan_to_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let body = json!({"plan_id": plan_id});
                let result = http
                    .post(&format!("/api/workspace-milestones/{}/plans", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_workspace_milestone" => {
                let id = extract_id(args, "id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!(
                        "/api/workspace-milestones/{}/plans/{}",
                        id, plan_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            "get_workspace_milestone_progress" => {
                let id = extract_id(args, "id")?;
                let result = http
                    .get(&format!("/api/workspace-milestones/{}/progress", id))
                    .await?;
                Ok(Some(result))
            }

            // --- Resources (6) ---
            "list_resources" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("resource_type").and_then(|v| v.as_str()) {
                    query.push(("resource_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/resources", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_resource" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/resources", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_resource" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/resources/{}", id)).await?;
                Ok(Some(result))
            }

            "update_resource" => {
                let id = extract_id(args, "id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("file_path") {
                    body.insert("file_path".to_string(), v.clone());
                }
                if let Some(v) = args.get("url") {
                    body.insert("url".to_string(), v.clone());
                }
                if let Some(v) = args.get("version") {
                    body.insert("version".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/resources/{}", id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_resource" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/resources/{}", id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "link_resource_to_project" => {
                let id = extract_id(args, "id")?;
                let project_id = extract_id(args, "project_id")?;
                let link_type = extract_string(args, "link_type")?;
                let body = json!({"project_id": project_id, "link_type": link_type});
                let result = http
                    .post(&format!("/api/resources/{}/projects", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            // --- Components (8) ---
            "list_components" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("component_type").and_then(|v| v.as_str()) {
                    query.push(("component_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/components", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_component" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/components", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_component" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/components/{}", id)).await?;
                Ok(Some(result))
            }

            "update_component" => {
                let id = extract_id(args, "id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("runtime") {
                    body.insert("runtime".to_string(), v.clone());
                }
                if let Some(v) = args.get("config") {
                    body.insert("config".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/components/{}", id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_component" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/components/{}", id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_component_dependency" => {
                let id = extract_id(args, "id")?;
                let depends_on_id = extract_id(args, "depends_on_id")?;
                let mut body = json!({"depends_on_id": depends_on_id});
                if let Some(v) = args.get("protocol") {
                    body["protocol"] = v.clone();
                }
                if let Some(v) = args.get("required") {
                    body["required"] = v.clone();
                }
                let result = http
                    .post(&format!("/api/components/{}/dependencies", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_component_dependency" => {
                let id = extract_id(args, "id")?;
                let dep_id = extract_id(args, "dep_id")?;
                let result = http
                    .delete(&format!("/api/components/{}/dependencies/{}", id, dep_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "map_component_to_project" => {
                let id = extract_id(args, "id")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .put(&format!("/api/components/{}/project", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"mapped": true})
                } else {
                    result
                }))
            }

            // ── P9: Code Exploration & Analytics (16 tools) ────────────────

            // --- Search (3) ---
            "search_code" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/search", &query).await?;
                Ok(Some(result))
            }

            "search_project_code" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(
                        &format!("/api/projects/{}/code/search", project_slug),
                        &query,
                    )
                    .await?;
                Ok(Some(result))
            }

            "search_workspace_code" => {
                // REST search_code supports workspace_slug as query param
                let workspace_slug = extract_string(args, "workspace_slug")?;
                let query_str = extract_string(args, "query")?;
                let mut query = vec![
                    ("query".to_string(), query_str),
                    ("workspace_slug".to_string(), workspace_slug),
                ];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/search", &query).await?;
                Ok(Some(result))
            }

            // --- Graph Traversal (4) ---
            "get_file_symbols" => {
                let file_path = extract_string(args, "file_path")?;
                let result = http
                    .get(&format!("/api/code/symbols/{}", file_path))
                    .await?;
                Ok(Some(result))
            }

            "find_references" => {
                let symbol = extract_string(args, "symbol")?;
                let mut query = vec![("symbol".to_string(), symbol)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/references", &query).await?;
                Ok(Some(result))
            }

            "get_file_dependencies" => {
                let file_path = extract_string(args, "file_path")?;
                let result = http
                    .get(&format!("/api/code/dependencies/{}", file_path))
                    .await?;
                Ok(Some(result))
            }

            "get_call_graph" => {
                let function = extract_string(args, "function")?;
                let mut query = vec![("function".to_string(), function)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/callgraph", &query).await?;
                Ok(Some(result))
            }

            // --- Analysis (3) ---
            "analyze_impact" => {
                let target = extract_string(args, "target")?;
                let mut query = vec![("target".to_string(), target)];
                if let Some(v) = args.get("target_type").and_then(|v| v.as_str()) {
                    query.push(("target_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/impact", &query).await?;
                Ok(Some(result))
            }

            "get_architecture" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/code/architecture").await?
                } else {
                    http.get_with_query("/api/code/architecture", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "find_similar_code" => {
                // POST because of body (code_snippet)
                let result = http.post("/api/code/similar", args).await?;
                Ok(Some(result))
            }

            // --- Types (3) ---
            "find_trait_implementations" => {
                let trait_name = extract_string(args, "trait_name")?;
                let query = vec![("trait_name".to_string(), trait_name)];
                let result = http.get_with_query("/api/code/trait-impls", &query).await?;
                Ok(Some(result))
            }

            "find_type_traits" => {
                let type_name = extract_string(args, "type_name")?;
                let query = vec![("type_name".to_string(), type_name)];
                let result = http.get_with_query("/api/code/type-traits", &query).await?;
                Ok(Some(result))
            }

            "get_impl_blocks" => {
                let type_name = extract_string(args, "type_name")?;
                let query = vec![("type_name".to_string(), type_name)];
                let result = http.get_with_query("/api/code/impl-blocks", &query).await?;
                Ok(Some(result))
            }

            // --- Analytics GDS (3) ---
            "get_code_communities" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("min_size").and_then(|v| v.as_i64()) {
                    query.push(("min_size".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/communities", &query).await?;
                Ok(Some(result))
            }

            "get_code_health" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("god_function_threshold").and_then(|v| v.as_i64()) {
                    query.push(("god_function_threshold".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/health", &query).await?;
                Ok(Some(result))
            }

            "get_node_importance" => {
                let project_slug = extract_string(args, "project_slug")?;
                let node_path = extract_string(args, "node_path")?;
                let mut query = vec![
                    ("project_slug".to_string(), project_slug),
                    ("node_path".to_string(), node_path),
                ];
                if let Some(v) = args.get("node_type").and_then(|v| v.as_str()) {
                    query.push(("node_type".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/node-importance", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── P10: Chat, Feature Graphs & Misc (14 tools) ─────────────

            // --- Chat CRUD (5) ---
            "list_chat_sessions" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/chat/sessions", &query).await?;
                Ok(Some(result))
            }

            "get_chat_session" => {
                let id = extract_id(args, "session_id")?;
                let result = http.get(&format!("/api/chat/sessions/{}", id)).await?;
                Ok(Some(result))
            }

            "delete_chat_session" => {
                let id = extract_id(args, "session_id")?;
                let result = http.delete(&format!("/api/chat/sessions/{}", id)).await?;
                Ok(Some(result))
            }

            "chat_send_message" => {
                // REST POST /api/chat/sessions creates a session + sends the first message
                let mut body = serde_json::Map::new();
                body.insert(
                    "message".to_string(),
                    json!(extract_string(args, "message")?),
                );
                body.insert("cwd".to_string(), json!(extract_string(args, "cwd")?));
                if let Some(v) = args.get("session_id").and_then(|v| v.as_str()) {
                    body.insert("session_id".to_string(), json!(v));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    body.insert("project_slug".to_string(), json!(v));
                }
                if let Some(v) = args.get("model").and_then(|v| v.as_str()) {
                    body.insert("model".to_string(), json!(v));
                }
                if let Some(v) = args.get("permission_mode").and_then(|v| v.as_str()) {
                    body.insert("permission_mode".to_string(), json!(v));
                }
                if let Some(v) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    body.insert("workspace_slug".to_string(), json!(v));
                }
                if let Some(v) = args.get("add_dirs") {
                    body.insert("add_dirs".to_string(), v.clone());
                }
                let result = http
                    .post("/api/chat/sessions", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "list_chat_messages" => {
                let id = extract_id(args, "session_id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(&format!("/api/chat/sessions/{}/messages", id), &query)
                    .await?;
                Ok(Some(result))
            }

            // --- Feature Graphs (6) ---
            "create_feature_graph" => {
                let mut body = serde_json::Map::new();
                body.insert("name".to_string(), json!(extract_string(args, "name")?));
                body.insert(
                    "project_id".to_string(),
                    json!(extract_id(args, "project_id")?),
                );
                if let Some(v) = args.get("description").and_then(|v| v.as_str()) {
                    body.insert("description".to_string(), json!(v));
                }
                let result = http
                    .post("/api/feature-graphs", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "list_feature_graphs" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/feature-graphs", &query).await?;
                Ok(Some(result))
            }

            "get_feature_graph" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/feature-graphs/{}", id)).await?;
                Ok(Some(result))
            }

            "delete_feature_graph" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/feature-graphs/{}", id)).await?;
                Ok(Some(result))
            }

            "add_to_feature_graph" => {
                let id = extract_id(args, "feature_graph_id")?;
                let mut body = serde_json::Map::new();
                body.insert(
                    "entity_type".to_string(),
                    json!(extract_string(args, "entity_type")?),
                );
                body.insert(
                    "entity_id".to_string(),
                    json!(extract_string(args, "entity_id")?),
                );
                if let Some(v) = args.get("role").and_then(|v| v.as_str()) {
                    body.insert("role".to_string(), json!(v));
                }
                let result = http
                    .post(
                        &format!("/api/feature-graphs/{}/entities", id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "auto_build_feature_graph" => {
                let mut body = serde_json::Map::new();
                body.insert("name".to_string(), json!(extract_string(args, "name")?));
                body.insert(
                    "project_id".to_string(),
                    json!(extract_id(args, "project_id")?),
                );
                body.insert(
                    "entry_function".to_string(),
                    json!(extract_string(args, "entry_function")?),
                );
                if let Some(v) = args.get("description").and_then(|v| v.as_str()) {
                    body.insert("description".to_string(), json!(v));
                }
                if let Some(v) = args.get("depth").and_then(|v| v.as_u64()) {
                    body.insert("depth".to_string(), json!(v));
                }
                if let Some(v) = args.get("include_relations") {
                    body.insert("include_relations".to_string(), v.clone());
                }
                if let Some(v) = args.get("filter_community").and_then(|v| v.as_bool()) {
                    body.insert("filter_community".to_string(), json!(v));
                }
                let result = http
                    .post("/api/feature-graphs/auto-build", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            // --- Misc (3) ---
            "plan_implementation" => {
                let mut body = serde_json::Map::new();
                body.insert(
                    "project_slug".to_string(),
                    json!(extract_string(args, "project_slug")?),
                );
                body.insert(
                    "description".to_string(),
                    json!(extract_string(args, "description")?),
                );
                if let Some(v) = args.get("entry_points") {
                    body.insert("entry_points".to_string(), v.clone());
                }
                if let Some(v) = args.get("scope").and_then(|v| v.as_str()) {
                    body.insert("scope".to_string(), json!(v));
                }
                if let Some(v) = args.get("auto_create_plan").and_then(|v| v.as_bool()) {
                    body.insert("auto_create_plan".to_string(), json!(v));
                }
                let result = http
                    .post("/api/code/plan-implementation", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "get_co_change_graph" => {
                let project_slug = extract_string(args, "project_slug")?;
                // Resolve project UUID from slug
                let project_info = http.get(&format!("/api/projects/{}", project_slug)).await?;
                let project_id = project_info
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Project not found: {}", project_slug))?;
                let mut query_params = Vec::new();
                if let Some(v) = args.get("min_count").and_then(|v| v.as_i64()) {
                    query_params.push(("min_count".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), v.to_string()));
                }
                let result = if query_params.is_empty() {
                    http.get(&format!("/api/projects/{}/co-changes", project_id)).await?
                } else {
                    http.get_with_query(
                        &format!("/api/projects/{}/co-changes", project_id),
                        &query_params,
                    )
                    .await?
                };
                Ok(Some(result))
            }

            "get_file_co_changers" => {
                let file_path = extract_string(args, "file_path")?;
                let mut query_params = vec![("path".to_string(), file_path)];
                if let Some(v) = args.get("min_count").and_then(|v| v.as_i64()) {
                    query_params.push(("min_count".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/files/co-changers", &query_params)
                    .await?;
                Ok(Some(result))
            }

            "get_meilisearch_stats" => {
                let result = http.get("/api/meilisearch/stats").await?;
                Ok(Some(result))
            }

            "delete_meilisearch_orphans" => {
                let result = http.delete("/api/meilisearch/orphans").await?;
                Ok(Some(result))
            }

            // ── P11: Admin & Sync (6 tools) ─────────────────────────────

            // --- Sync (1) ---
            "sync_directory" => {
                let mut body = serde_json::Map::new();
                body.insert("path".to_string(), json!(extract_string(args, "path")?));
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), json!(v));
                }
                let result = http.post("/api/sync", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            // --- Watch (3) ---
            "start_watch" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("path").and_then(|v| v.as_str()) {
                    body.insert("path".to_string(), json!(v));
                }
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), json!(v));
                }
                let result = http.post("/api/watch", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            "stop_watch" => {
                let result = http.delete("/api/watch").await?;
                Ok(Some(result))
            }

            "watch_status" => {
                let result = http.get("/api/watch").await?;
                Ok(Some(result))
            }

            // --- Admin Cleanup (2) ---
            "cleanup_cross_project_calls" => {
                let result = http
                    .post("/api/admin/cleanup-cross-project-calls", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "cleanup_sync_data" => {
                let result = http
                    .post("/api/admin/cleanup-sync-data", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            // ── All tools migrated ──────────────────────────────────────
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========================================================================
    // Mega-tool resolution tests
    // ========================================================================

    fn make_handler() -> ToolHandler {
        // Create a handler with a dummy HTTP client (no actual server needed for resolution tests)
        let client = McpHttpClient::new("http://localhost:0".to_string(), None);
        ToolHandler::new(client)
    }

    #[test]
    fn test_resolve_mega_tool_project_list() {
        let handler = make_handler();
        let args = json!({"action": "list"});
        let (name, _resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "list_projects");
    }

    #[test]
    fn test_resolve_mega_tool_project_create() {
        let handler = make_handler();
        let args = json!({"action": "create", "name": "My Project", "root_path": "/tmp"});
        let (name, resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "create_project");
        assert_eq!(
            resolved_args.get("name").unwrap().as_str().unwrap(),
            "My Project"
        );
    }

    #[test]
    fn test_resolve_mega_tool_plan_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_plans"),
            ("create", "create_plan"),
            ("get", "get_plan"),
            ("update_status", "update_plan_status"),
            ("link_to_project", "link_plan_to_project"),
            ("get_dependency_graph", "get_dependency_graph"),
            ("get_critical_path", "get_critical_path"),
            ("delete", "delete_plan"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("plan", &args).unwrap();
            assert_eq!(
                name, expected,
                "plan action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_task_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_tasks"),
            ("create", "create_task"),
            ("get", "get_task"),
            ("update", "update_task"),
            ("delete", "delete_task"),
            ("get_next", "get_next_task"),
            ("add_dependencies", "add_task_dependencies"),
            ("get_blockers", "get_task_blockers"),
            ("get_context", "get_task_context"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("task", &args).unwrap();
            assert_eq!(
                name, expected,
                "task action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_note_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_notes"),
            ("create", "create_note"),
            ("search", "search_notes"),
            ("search_semantic", "search_notes_semantic"),
            ("link_to_entity", "link_note_to_entity"),
            ("get_context", "get_context_notes"),
            ("get_context_knowledge", "get_context_knowledge"),
            ("get_propagated_knowledge", "get_propagated_knowledge"),
            ("confirm", "confirm_note"),
            ("invalidate", "invalidate_note"),
            ("supersede", "supersede_note"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("note", &args).unwrap();
            assert_eq!(
                name, expected,
                "note action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_code_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("search", "search_code"),
            ("search_project", "search_project_code"),
            ("get_file_symbols", "get_file_symbols"),
            ("find_references", "find_references"),
            ("get_call_graph", "get_call_graph"),
            ("analyze_impact", "analyze_impact"),
            ("get_architecture", "get_architecture"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("code", &args).unwrap();
            assert_eq!(
                name, expected,
                "code action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_legacy_passthrough() {
        let handler = make_handler();
        // Legacy tool names should pass through unchanged
        let args = json!({"plan_id": "some-id"});
        let (name, resolved_args) = handler.resolve_mega_tool("list_plans", &args).unwrap();
        assert_eq!(name, "list_plans");
        assert_eq!(resolved_args, args);
    }

    #[test]
    fn test_resolve_mega_tool_unknown_action() {
        let handler = make_handler();
        let args = json!({"action": "nonexistent"});
        let result = handler.resolve_mega_tool("project", &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown action"));
    }

    #[test]
    fn test_resolve_mega_tool_missing_action() {
        let handler = make_handler();
        let args = json!({"name": "test"});
        let result = handler.resolve_mega_tool("project", &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("action"));
    }

    #[test]
    fn test_resolve_mega_tool_preserves_all_args() {
        let handler = make_handler();
        let args = json!({"action": "list", "search": "test", "limit": 10});
        let (name, resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "list_projects");
        // action param is kept in resolved args (handlers ignore it)
        assert_eq!(
            resolved_args.get("action").unwrap().as_str().unwrap(),
            "list"
        );
        assert_eq!(
            resolved_args.get("search").unwrap().as_str().unwrap(),
            "test"
        );
        assert_eq!(resolved_args.get("limit").unwrap().as_u64().unwrap(), 10);
    }

    #[test]
    fn test_resolve_all_18_mega_tools() {
        let handler = make_handler();
        // Every mega-tool should have at least "list" or "search" or some base action
        let mega_tools_with_action: Vec<(&str, &str)> = vec![
            ("project", "list"),
            ("plan", "list"),
            ("task", "list"),
            ("step", "list"),
            ("constraint", "list"),
            ("release", "list"),
            ("milestone", "list"),
            ("commit", "create"),
            ("code", "search"),
            ("decision", "get"),
            ("note", "list"),
            ("workspace", "list"),
            ("workspace_milestone", "list_all"),
            ("resource", "list"),
            ("component", "list"),
            ("chat", "list_sessions"),
            ("feature_graph", "list"),
            ("admin", "meilisearch_stats"),
        ];

        for (tool, action) in mega_tools_with_action {
            let args = json!({"action": action});
            let result = handler.resolve_mega_tool(tool, &args);
            assert!(
                result.is_ok(),
                "mega-tool '{}' with action '{}' should resolve successfully, got: {:?}",
                tool,
                action,
                result.err()
            );
        }
    }

    // ========================================================================
    // HTTP routing tests (mock axum server)
    // ========================================================================

    use axum::{
        body::Body, extract::Request, http::StatusCode, response::IntoResponse, routing::any,
        Router,
    };
    use tokio::net::TcpListener;

    /// Echo handler: returns {method, path, query, body} for every request.
    async fn echo_handler(req: Request<Body>) -> impl IntoResponse {
        let method = req.method().to_string();
        let path = req.uri().path().to_string();
        let query = req.uri().query().unwrap_or("").to_string();
        let body_bytes = axum::body::to_bytes(req.into_body(), 1024 * 1024)
            .await
            .unwrap_or_default();
        let body_str = String::from_utf8_lossy(&body_bytes).to_string();
        let body_json: Value = serde_json::from_str(&body_str).unwrap_or(Value::Null);

        (
            StatusCode::OK,
            axum::Json(json!({
                "_echo": true,
                "method": method,
                "path": path,
                "query": query,
                "body": body_json,
            })),
        )
    }

    /// Spin up a temporary axum server and return a ToolHandler connected to it.
    async fn make_http_handler() -> (ToolHandler, String) {
        let app = Router::new().fallback(any(echo_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Give the server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let client = McpHttpClient::new(base_url.clone(), Some("test-token".to_string()));
        (ToolHandler::new(client), base_url)
    }

    // -- Projects ----------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_projects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_projects", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
    }

    #[tokio::test]
    async fn test_http_list_projects_with_query() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_projects",
                Some(json!({"search": "test", "limit": 10})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
        let query = result["query"].as_str().unwrap();
        assert!(
            query.contains("search=test"),
            "query should contain search=test, got: {}",
            query
        );
        assert!(
            query.contains("limit=10"),
            "query should contain limit=10, got: {}",
            query
        );
    }

    #[tokio::test]
    async fn test_http_create_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_project",
                Some(json!({"name": "My Project", "root_path": "/tmp/proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/projects");
        assert_eq!(result["body"]["name"], "My Project");
    }

    #[tokio::test]
    async fn test_http_get_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_project", Some(json!({"slug": "my-project"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects/my-project");
    }

    #[tokio::test]
    async fn test_http_update_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_project",
                Some(json!({"slug": "my-proj", "name": "New Name"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert_eq!(result["path"], "/api/projects/my-proj");
        assert_eq!(result["body"]["name"], "New Name");
    }

    #[tokio::test]
    async fn test_http_delete_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_project", Some(json!({"slug": "old-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(result["path"], "/api/projects/old-proj");
    }

    // -- Plans -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_plan",
                Some(json!({"title": "Test Plan", "description": "A plan"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/plans");
        assert_eq!(result["body"]["title"], "Test Plan");
    }

    #[tokio::test]
    async fn test_http_update_plan_status() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_plan_status",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "in_progress"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("550e8400-e29b-41d4-a716-446655440000"));
        assert_eq!(result["body"]["status"], "in_progress");
    }

    #[tokio::test]
    async fn test_http_link_plan_to_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_plan_to_project",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "project_id": "660e8400-e29b-41d4-a716-446655440000"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PUT");
        assert!(result["path"].as_str().unwrap().ends_with("/project"));
    }

    #[tokio::test]
    async fn test_http_delete_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "delete_plan",
                Some(json!({"plan_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/plans/"));
    }

    // -- Tasks -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_tasks_with_filters() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_tasks",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "pending",
                    "limit": 20
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/tasks");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("plan_id="));
        assert!(query.contains("status=pending"));
    }

    #[tokio::test]
    async fn test_http_create_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "title": "Test Task"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/tasks"));
        assert_eq!(result["body"]["title"], "Test Task");
    }

    // -- Steps -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_step",
                Some(json!({
                    "task_id": "550e8400-e29b-41d4-a716-446655440000",
                    "description": "Do something"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().contains("/steps"));
    }

    #[tokio::test]
    async fn test_http_update_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_step",
                Some(json!({
                    "step_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"].as_str().unwrap().contains("/steps/"));
    }

    // -- Notes -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_note",
                Some(json!({
                    "content": "Important note",
                    "note_type": "guideline",
                    "importance": "high"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes");
    }

    #[tokio::test]
    async fn test_http_search_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_notes", Some(json!({"query": "architecture"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/search");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("q=architecture"));
    }

    // -- Commits -----------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_commit() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_commit",
                Some(json!({
                    "sha": "abc1234",
                    "message": "feat: something",
                    "author": "test"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/commits");
        // MCP field "sha" maps to REST "hash"
        assert_eq!(result["body"]["hash"], "abc1234");
    }

    // -- Code --------------------------------------------------------------

    #[tokio::test]
    async fn test_http_search_code() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_code", Some(json!({"query": "ToolHandler"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/search");
        let query = result["query"].as_str().unwrap();
        assert!(
            query.contains("query=ToolHandler"),
            "query should contain query=ToolHandler, got: {}",
            query
        );
    }

    // -- Workspaces --------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_workspace",
                Some(json!({"name": "Test WS", "slug": "test-ws"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/workspaces");
    }

    // -- Mega-tool → HTTP integration -------------------------------------

    #[tokio::test]
    async fn test_mega_tool_project_routes_to_http() {
        let (handler, _) = make_http_handler().await;
        // Calling mega-tool "project" with action "list" should resolve and route via HTTP
        let result = handler
            .handle("project", Some(json!({"action": "list"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
    }

    #[tokio::test]
    async fn test_mega_tool_plan_create_routes_to_http() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "plan",
                Some(json!({"action": "create", "title": "New Plan"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/plans");
    }

    #[tokio::test]
    async fn test_mega_tool_note_search_semantic() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "note",
                Some(json!({"action": "search_semantic", "query": "how to"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/search-semantic");
    }

    // -- Error paths -------------------------------------------------------

    #[tokio::test]
    async fn test_handle_unknown_tool_returns_error() {
        let (handler, _) = make_http_handler().await;
        let result = handler.handle("totally_unknown_tool", None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_handle_missing_required_arg() {
        let (handler, _) = make_http_handler().await;
        // get_project requires "slug" — call without it
        let result = handler.handle("get_project", Some(json!({}))).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("slug"));
    }
}
