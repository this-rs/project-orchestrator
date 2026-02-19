//! MCP Tool handlers
//!
//! Implements the actual logic for each MCP tool.

use crate::chat::ChatManager;
use crate::meilisearch::SearchStore;
use crate::neo4j::models::*;
use crate::neo4j::GraphStore;
use crate::orchestrator::Orchestrator;
use crate::plan::models::*;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

use crate::expand_tilde;

/// Handles MCP tool calls
pub struct ToolHandler {
    orchestrator: Arc<Orchestrator>,
    chat_manager: Option<Arc<ChatManager>>,
}

impl ToolHandler {
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self {
            orchestrator,
            chat_manager: None,
        }
    }

    pub fn with_chat_manager(mut self, chat_manager: Option<Arc<ChatManager>>) -> Self {
        self.chat_manager = chat_manager;
        self
    }

    fn neo4j(&self) -> &dyn GraphStore {
        self.orchestrator.neo4j()
    }

    fn meili(&self) -> &dyn SearchStore {
        self.orchestrator.meili()
    }

    /// Handle a tool call and return the result as JSON
    pub async fn handle(&self, name: &str, args: Option<Value>) -> Result<Value> {
        let args = args.unwrap_or(json!({}));

        match name {
            // Projects
            "list_projects" => self.list_projects(args).await,
            "create_project" => self.create_project(args).await,
            "get_project" => self.get_project(args).await,
            "delete_project" => self.delete_project(args).await,
            "sync_project" => self.sync_project(args).await,
            "get_project_roadmap" => self.get_project_roadmap(args).await,
            "list_project_plans" => self.list_project_plans(args).await,

            "update_project" => self.update_project(args).await,

            // Plans
            "list_plans" => self.list_plans(args).await,
            "create_plan" => self.create_plan(args).await,
            "get_plan" => self.get_plan(args).await,
            "update_plan_status" => self.update_plan_status(args).await,
            "link_plan_to_project" => self.link_plan_to_project(args).await,
            "unlink_plan_from_project" => self.unlink_plan_from_project(args).await,
            "get_dependency_graph" => self.get_dependency_graph(args).await,
            "get_critical_path" => self.get_critical_path(args).await,
            "delete_plan" => self.delete_plan(args).await,

            // Tasks
            "list_tasks" => self.list_tasks(args).await,
            "create_task" => self.create_task(args).await,
            "get_task" => self.get_task(args).await,
            "update_task" => self.update_task(args).await,
            "delete_task" => self.delete_task(args).await,
            "get_next_task" => self.get_next_task(args).await,
            "add_task_dependencies" => self.add_task_dependencies(args).await,
            "remove_task_dependency" => self.remove_task_dependency(args).await,
            "get_task_blockers" => self.get_task_blockers(args).await,
            "get_tasks_blocked_by" => self.get_tasks_blocked_by(args).await,
            "get_task_context" => self.get_task_context(args).await,
            "get_task_prompt" => self.get_task_prompt(args).await,
            "add_decision" => self.add_decision(args).await,

            // Steps
            "list_steps" => self.list_steps(args).await,
            "create_step" => self.create_step(args).await,
            "get_step" => self.get_step(args).await,
            "delete_step" => self.delete_step_handler(args).await,
            "update_step" => self.update_step(args).await,
            "get_step_progress" => self.get_step_progress(args).await,

            // Constraints
            "list_constraints" => self.list_constraints(args).await,
            "add_constraint" => self.add_constraint(args).await,
            "get_constraint" => self.get_constraint(args).await,
            "update_constraint" => self.update_constraint(args).await,
            "delete_constraint" => self.delete_constraint(args).await,

            // Releases
            "list_releases" => self.list_releases(args).await,
            "create_release" => self.create_release(args).await,
            "get_release" => self.get_release(args).await,
            "update_release" => self.update_release(args).await,
            "delete_release" => self.delete_release(args).await,
            "add_task_to_release" => self.add_task_to_release(args).await,
            "add_commit_to_release" => self.add_commit_to_release(args).await,

            // Milestones
            "list_milestones" => self.list_milestones(args).await,
            "create_milestone" => self.create_milestone(args).await,
            "get_milestone" => self.get_milestone(args).await,
            "update_milestone" => self.update_milestone(args).await,
            "delete_milestone" => self.delete_milestone(args).await,
            "get_milestone_progress" => self.get_milestone_progress(args).await,
            "add_task_to_milestone" => self.add_task_to_milestone(args).await,

            // Commits
            "create_commit" => self.create_commit(args).await,
            "link_commit_to_task" => self.link_commit_to_task(args).await,
            "link_commit_to_plan" => self.link_commit_to_plan(args).await,
            "get_task_commits" => self.get_task_commits(args).await,
            "get_plan_commits" => self.get_plan_commits(args).await,

            // Code
            "search_code" => self.search_code(args).await,
            "search_project_code" => self.search_project_code(args).await,
            "search_workspace_code" => self.search_workspace_code(args).await,
            "get_file_symbols" => self.get_file_symbols(args).await,
            "find_references" => self.find_references(args).await,
            "get_file_dependencies" => self.get_file_dependencies(args).await,
            "get_call_graph" => self.get_call_graph(args).await,
            "analyze_impact" => self.analyze_impact(args).await,
            "get_architecture" => self.get_architecture(args).await,
            "find_similar_code" => self.find_similar_code(args).await,
            "find_trait_implementations" => self.find_trait_implementations(args).await,
            "find_type_traits" => self.find_type_traits(args).await,
            "get_impl_blocks" => self.get_impl_blocks(args).await,

            // Decisions
            "get_decision" => self.get_decision(args).await,
            "update_decision" => self.update_decision(args).await,
            "delete_decision" => self.delete_decision(args).await,
            "search_decisions" => self.search_decisions(args).await,

            // Sync
            "sync_directory" => self.sync_directory(args).await,
            "start_watch" => self.start_watch(args).await,
            "stop_watch" => self.stop_watch(args).await,
            "watch_status" => self.watch_status(args).await,

            // Meilisearch
            "get_meilisearch_stats" => self.get_meilisearch_stats(args).await,
            "delete_meilisearch_orphans" => self.delete_meilisearch_orphans(args).await,
            "cleanup_cross_project_calls" => self.cleanup_cross_project_calls(args).await,

            // Notes
            "list_notes" => self.list_notes(args).await,
            "create_note" => self.create_note(args).await,
            "get_note" => self.get_note(args).await,
            "update_note" => self.update_note(args).await,
            "delete_note" => self.delete_note(args).await,
            "search_notes" => self.search_notes(args).await,
            "confirm_note" => self.confirm_note(args).await,
            "invalidate_note" => self.invalidate_note(args).await,
            "supersede_note" => self.supersede_note(args).await,
            "link_note_to_entity" => self.link_note_to_entity(args).await,
            "unlink_note_from_entity" => self.unlink_note_from_entity(args).await,
            "get_context_notes" => self.get_context_notes(args).await,
            "get_notes_needing_review" => self.get_notes_needing_review(args).await,
            "update_staleness_scores" => self.update_staleness_scores(args).await,
            "list_project_notes" => self.list_project_notes(args).await,
            "get_propagated_notes" => self.get_propagated_notes(args).await,
            "get_entity_notes" => self.get_entity_notes(args).await,

            // Workspaces
            "list_workspaces" => self.list_workspaces(args).await,
            "create_workspace" => self.create_workspace(args).await,
            "get_workspace" => self.get_workspace(args).await,
            "update_workspace" => self.update_workspace(args).await,
            "delete_workspace" => self.delete_workspace(args).await,
            "get_workspace_overview" => self.get_workspace_overview(args).await,
            "list_workspace_projects" => self.list_workspace_projects(args).await,
            "add_project_to_workspace" => self.add_project_to_workspace(args).await,
            "remove_project_from_workspace" => self.remove_project_from_workspace(args).await,

            // Workspace Milestones
            "list_all_workspace_milestones" => self.list_all_workspace_milestones(args).await,
            "list_workspace_milestones" => self.list_workspace_milestones(args).await,
            "create_workspace_milestone" => self.create_workspace_milestone(args).await,
            "get_workspace_milestone" => self.get_workspace_milestone(args).await,
            "update_workspace_milestone" => self.update_workspace_milestone(args).await,
            "delete_workspace_milestone" => self.delete_workspace_milestone(args).await,
            "add_task_to_workspace_milestone" => self.add_task_to_workspace_milestone(args).await,
            "get_workspace_milestone_progress" => self.get_workspace_milestone_progress(args).await,

            // Resources
            "list_resources" => self.list_resources(args).await,
            "create_resource" => self.create_resource(args).await,
            "get_resource" => self.get_resource(args).await,
            "update_resource" => self.update_resource(args).await,
            "delete_resource" => self.delete_resource(args).await,
            "link_resource_to_project" => self.link_resource_to_project(args).await,

            // Components
            "list_components" => self.list_components(args).await,
            "create_component" => self.create_component(args).await,
            "get_component" => self.get_component(args).await,
            "update_component" => self.update_component(args).await,
            "delete_component" => self.delete_component(args).await,
            "add_component_dependency" => self.add_component_dependency(args).await,
            "remove_component_dependency" => self.remove_component_dependency(args).await,
            "map_component_to_project" => self.map_component_to_project(args).await,
            "get_workspace_topology" => self.get_workspace_topology(args).await,

            // Chat
            "list_chat_sessions" => self.list_chat_sessions(args).await,
            "get_chat_session" => self.get_chat_session(args).await,
            "delete_chat_session" => self.delete_chat_session(args).await,
            "chat_send_message" => self.chat_send_message(args).await,
            "list_chat_messages" => self.list_chat_messages(args).await,

            // Feature Graph
            "create_feature_graph" => self.create_feature_graph(args).await,
            "get_feature_graph" => self.get_feature_graph(args).await,
            "list_feature_graphs" => self.list_feature_graphs(args).await,
            "add_to_feature_graph" => self.add_to_feature_graph(args).await,
            "auto_build_feature_graph" => self.auto_build_feature_graph(args).await,
            "delete_feature_graph" => self.delete_feature_graph(args).await,

            // Implementation Planner
            "plan_implementation" => self.plan_implementation(args).await,

            _ => Err(anyhow!("Unknown tool: {}", name)),
        }
    }

    // ========================================================================
    // Project Handlers
    // ========================================================================

    async fn list_projects(&self, args: Value) -> Result<Value> {
        let search = args.get("search").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let sort_by = args.get("sort_by").and_then(|v| v.as_str());
        let sort_order = args
            .get("sort_order")
            .and_then(|v| v.as_str())
            .unwrap_or("desc");

        let (projects, total) = self
            .neo4j()
            .list_projects_filtered(search, limit, offset, sort_by, sort_order)
            .await?;

        Ok(json!({
            "items": projects,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_project(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let root_path = args
            .get("root_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("root_path is required"))?;
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| slugify(name));
        let description = args.get("description").and_then(|v| v.as_str());

        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            slug,
            root_path: expand_tilde(root_path),
            description: description.map(|s| s.to_string()),
            created_at: chrono::Utc::now(),
            last_synced: None,
        };

        self.orchestrator.create_project(&project).await?;
        Ok(serde_json::to_value(project)?)
    }

    async fn get_project(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let project = self
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Project not found: {}", slug))?;

        Ok(serde_json::to_value(project)?)
    }

    async fn update_project(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let project = self
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Project not found: {}", slug))?;

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .map(|v| v.as_str().map(|s| s.to_string()));
        let root_path = args
            .get("root_path")
            .and_then(|v| v.as_str())
            .map(expand_tilde);

        self.orchestrator
            .update_project(project.id, name, description, root_path)
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn delete_project(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let project = self
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Project not found: {}", slug))?;

        self.orchestrator.delete_project(project.id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn sync_project(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let project = self
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Project not found: {}", slug))?;

        let expanded = expand_tilde(&project.root_path);
        let path = std::path::Path::new(&expanded);
        let result = self
            .orchestrator
            .sync_directory_for_project(path, Some(project.id), Some(&project.slug))
            .await?;

        self.neo4j().update_project_synced(project.id).await?;

        // Refresh auto-built feature graphs in background (best-effort)
        self.orchestrator.spawn_refresh_feature_graphs(project.id);

        Ok(json!({
            "files_synced": result.files_synced,
            "files_skipped": result.files_skipped,
            "files_deleted": result.files_deleted,
            "symbols_deleted": result.symbols_deleted,
            "errors": result.errors
        }))
    }

    async fn get_project_roadmap(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;

        let milestones = self.neo4j().list_project_milestones(project_id).await?;
        let releases = self.neo4j().list_project_releases(project_id).await?;
        let (total, completed, in_progress, pending) =
            self.neo4j().get_project_progress(project_id).await?;

        Ok(json!({
            "milestones": milestones,
            "releases": releases,
            "progress": {
                "total": total,
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending
            }
        }))
    }

    async fn list_project_plans(&self, args: Value) -> Result<Value> {
        let project_slug = args
            .get("project_slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("project_slug is required"))?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        // Get project by slug
        let project = self
            .neo4j()
            .get_project_by_slug(project_slug)
            .await?
            .ok_or_else(|| anyhow!("Project not found: {}", project_slug))?;

        // List plans for this project
        let (plans, total) = self
            .neo4j()
            .list_plans_for_project(project.id, status, limit, offset)
            .await?;

        Ok(json!({
            "items": plans,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    // ========================================================================
    // Plan Handlers
    // ========================================================================

    async fn list_plans(&self, args: Value) -> Result<Value> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .map_err(|_| anyhow!("Invalid project_id UUID"))?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let priority_min = args
            .get("priority_min")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let priority_max = args
            .get("priority_max")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let search = args.get("search").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let sort_by = args.get("sort_by").and_then(|v| v.as_str());
        let sort_order = args
            .get("sort_order")
            .and_then(|v| v.as_str())
            .unwrap_or("desc");

        let (plans, total) = self
            .neo4j()
            .list_plans_filtered(
                project_id,
                None,
                status,
                priority_min,
                priority_max,
                search,
                limit,
                offset,
                sort_by,
                sort_order,
            )
            .await?;

        Ok(json!({
            "items": plans,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_plan(&self, args: Value) -> Result<Value> {
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("title is required"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;
        let priority = args.get("priority").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let req = CreatePlanRequest {
            title: title.to_string(),
            description: description.to_string(),
            priority: Some(priority),
            project_id,
            constraints: None,
        };

        let plan = self
            .orchestrator
            .plan_manager()
            .create_plan(req, "mcp")
            .await?;
        Ok(serde_json::to_value(plan)?)
    }

    async fn get_plan(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        let details = self
            .orchestrator
            .plan_manager()
            .get_plan_details(plan_id)
            .await?
            .ok_or_else(|| anyhow!("Plan not found"))?;

        Ok(serde_json::to_value(details)?)
    }

    async fn update_plan_status(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let status_str = args
            .get("status")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("status is required"))?;

        let status: PlanStatus = serde_json::from_str(&format!("\"{}\"", status_str))?;
        self.orchestrator
            .plan_manager()
            .update_plan_status(plan_id, status)
            .await?;

        Ok(json!({"updated": true}))
    }

    async fn link_plan_to_project(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let project_id = parse_uuid(&args, "project_id")?;

        self.orchestrator
            .link_plan_to_project(plan_id, project_id)
            .await?;
        Ok(json!({"linked": true}))
    }

    async fn unlink_plan_from_project(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        self.orchestrator.unlink_plan_from_project(plan_id).await?;
        Ok(json!({"unlinked": true}))
    }

    async fn delete_plan(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        self.orchestrator
            .plan_manager()
            .delete_plan(plan_id)
            .await?;
        Ok(json!({"deleted": true}))
    }

    async fn get_dependency_graph(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        // Get all tasks for the plan with their dependencies
        let details = self
            .orchestrator
            .plan_manager()
            .get_plan_details(plan_id)
            .await?
            .ok_or_else(|| anyhow!("Plan not found"))?;

        // Build a simple graph structure
        let nodes: Vec<Value> = details
            .tasks
            .iter()
            .map(|t| {
                json!({
                    "id": t.task.id.to_string(),
                    "title": t.task.title.clone().unwrap_or_else(|| t.task.description.chars().take(50).collect()),
                    "status": format!("{:?}", t.task.status),
                    "priority": t.task.priority
                })
            })
            .collect();

        let edges: Vec<Value> = details
            .tasks
            .iter()
            .flat_map(|t| {
                t.depends_on
                    .iter()
                    .map(|dep_id| {
                        json!({
                            "from": dep_id.to_string(),
                            "to": t.task.id.to_string()
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(json!({
            "nodes": nodes,
            "edges": edges
        }))
    }

    async fn get_critical_path(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        // Get all tasks for the plan
        let details = self
            .orchestrator
            .plan_manager()
            .get_plan_details(plan_id)
            .await?
            .ok_or_else(|| anyhow!("Plan not found"))?;

        // Build dependency map
        let task_map: std::collections::HashMap<Uuid, &TaskDetails> =
            details.tasks.iter().map(|t| (t.task.id, t)).collect();

        // Simple critical path: find the longest chain
        // This is a simplified implementation - a proper one would use topological sort
        let mut longest_path: Vec<Uuid> = Vec::new();

        for task in &details.tasks {
            if task.task.status != TaskStatus::Completed {
                let mut path = vec![task.task.id];
                let mut current = task;

                // Follow dependencies
                while !current.depends_on.is_empty() {
                    // Find an uncompleted dependency
                    if let Some(dep_id) = current.depends_on.iter().find(|id| {
                        task_map
                            .get(id)
                            .map(|t| t.task.status != TaskStatus::Completed)
                            .unwrap_or(false)
                    }) {
                        if let Some(dep_task) = task_map.get(dep_id) {
                            path.push(*dep_id);
                            current = dep_task;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if path.len() > longest_path.len() {
                    longest_path = path;
                }
            }
        }

        // Reverse to get tasks in order (dependencies first)
        longest_path.reverse();

        let path_tasks: Vec<Value> = longest_path
            .iter()
            .filter_map(|id| {
                task_map.get(id).map(|t| {
                    json!({
                        "id": t.task.id.to_string(),
                        "title": t.task.title.clone().unwrap_or_else(|| t.task.description.chars().take(50).collect::<String>()),
                        "status": format!("{:?}", t.task.status)
                    })
                })
            })
            .collect();

        Ok(json!({
            "path": path_tasks,
            "length": path_tasks.len()
        }))
    }

    // ========================================================================
    // Task Handlers
    // ========================================================================

    async fn list_tasks(&self, args: Value) -> Result<Value> {
        let plan_id = args
            .get("plan_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let priority_min = args
            .get("priority_min")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let priority_max = args
            .get("priority_max")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let tags = args
            .get("tags")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let assigned_to = args.get("assigned_to").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let sort_by = args.get("sort_by").and_then(|v| v.as_str());
        let sort_order = args
            .get("sort_order")
            .and_then(|v| v.as_str())
            .unwrap_or("desc");

        let (tasks, total) = self
            .neo4j()
            .list_all_tasks_filtered(
                plan_id,
                None,
                None,
                status,
                priority_min,
                priority_max,
                tags,
                assigned_to,
                limit,
                offset,
                sort_by,
                sort_order,
            )
            .await?;

        Ok(json!({
            "items": tasks,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_task(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let priority = args
            .get("priority")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let tags = args.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });
        let acceptance_criteria = args
            .get("acceptance_criteria")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let affected_files = args
            .get("affected_files")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let depends_on = args
            .get("dependencies")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().and_then(|s| Uuid::parse_str(s).ok()))
                    .collect()
            });

        let req = CreateTaskRequest {
            description: description.to_string(),
            title,
            priority,
            tags,
            acceptance_criteria,
            affected_files,
            depends_on,
            steps: None,
            estimated_complexity: None,
        };

        let task = self
            .orchestrator
            .plan_manager()
            .add_task(plan_id, req)
            .await?;
        Ok(serde_json::to_value(task)?)
    }

    async fn get_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let details = self
            .orchestrator
            .plan_manager()
            .get_task_details(task_id)
            .await?
            .ok_or_else(|| anyhow!("Task not found"))?;

        Ok(serde_json::to_value(details)?)
    }

    async fn delete_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        self.orchestrator
            .plan_manager()
            .delete_task(task_id)
            .await?;
        Ok(json!({"deleted": true}))
    }

    async fn update_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let status = match args.get("status").and_then(|v| v.as_str()) {
            Some(s) => Some(
                serde_json::from_str::<TaskStatus>(&format!("\"{}\"", s)).map_err(|_| {
                    anyhow!(
                        "Invalid task status '{}'. Valid: pending, in_progress, blocked, completed, failed",
                        s
                    )
                })?,
            ),
            None => None,
        };
        let assigned_to = args
            .get("assigned_to")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let priority = args
            .get("priority")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        let tags = args.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

        let req = UpdateTaskRequest {
            status,
            assigned_to,
            priority,
            tags,
            ..Default::default()
        };

        self.orchestrator
            .plan_manager()
            .update_task(task_id, req)
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn get_next_task(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        let task = self
            .orchestrator
            .plan_manager()
            .get_next_available_task(plan_id)
            .await?;
        Ok(serde_json::to_value(task)?)
    }

    async fn add_task_dependencies(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let dependency_ids: Vec<Uuid> = args
            .get("dependency_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().and_then(|s| Uuid::parse_str(s).ok()))
                    .collect()
            })
            .ok_or_else(|| anyhow!("dependency_ids is required"))?;

        for dep_id in dependency_ids {
            self.orchestrator
                .add_task_dependency(task_id, dep_id)
                .await?;
        }
        Ok(json!({"added": true}))
    }

    async fn remove_task_dependency(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let dependency_id = parse_uuid(&args, "dependency_id")?;

        self.orchestrator
            .remove_task_dependency(task_id, dependency_id)
            .await?;
        Ok(json!({"removed": true}))
    }

    async fn get_task_blockers(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let blockers = self.neo4j().get_task_blockers(task_id).await?;
        Ok(serde_json::to_value(blockers)?)
    }

    async fn get_tasks_blocked_by(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let blocked = self.neo4j().get_tasks_blocked_by(task_id).await?;
        Ok(serde_json::to_value(blocked)?)
    }

    async fn get_task_context(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        let context = self
            .orchestrator
            .context_builder()
            .build_context(task_id, plan_id)
            .await?;
        Ok(serde_json::to_value(context)?)
    }

    async fn get_task_prompt(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        let context = self
            .orchestrator
            .context_builder()
            .build_context(task_id, plan_id)
            .await?;
        let prompt = self
            .orchestrator
            .context_builder()
            .generate_prompt(&context);

        Ok(json!({"prompt": prompt}))
    }

    async fn add_decision(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;
        let rationale = args
            .get("rationale")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("rationale is required"))?;
        let alternatives = args
            .get("alternatives")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let chosen_option = args
            .get("chosen_option")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let req = CreateDecisionRequest {
            description: description.to_string(),
            rationale: rationale.to_string(),
            alternatives,
            chosen_option,
        };

        let decision = self
            .orchestrator
            .plan_manager()
            .add_decision(task_id, req, "mcp")
            .await?;
        Ok(serde_json::to_value(decision)?)
    }

    // ========================================================================
    // Step Handlers
    // ========================================================================

    async fn list_steps(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let steps = self.neo4j().get_task_steps(task_id).await?;
        Ok(serde_json::to_value(steps)?)
    }

    async fn create_step(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;
        let verification = args
            .get("verification")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Get current step count for ordering
        let steps = self.neo4j().get_task_steps(task_id).await?;
        let order = steps.len() as u32;

        let step = StepNode::new(order, description.to_string(), verification);
        self.orchestrator
            .plan_manager()
            .add_step(task_id, &step)
            .await?;
        Ok(serde_json::to_value(step)?)
    }

    async fn update_step(&self, args: Value) -> Result<Value> {
        let step_id = parse_uuid(&args, "step_id")?;
        let status_str = args
            .get("status")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("status is required"))?;

        let status: StepStatus = serde_json::from_str(&format!("\"{}\"", status_str))?;
        self.orchestrator
            .plan_manager()
            .update_step_status(step_id, status)
            .await?;

        Ok(json!({"updated": true}))
    }

    async fn get_step_progress(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let (completed, total) = self.neo4j().get_task_step_progress(task_id).await?;
        let percentage = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(json!({
            "completed": completed,
            "total": total,
            "percentage": percentage
        }))
    }

    // ========================================================================
    // Constraint Handlers
    // ========================================================================

    async fn list_constraints(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        let constraints = self.neo4j().get_plan_constraints(plan_id).await?;
        Ok(serde_json::to_value(constraints)?)
    }

    async fn add_constraint(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let constraint_type = args
            .get("constraint_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("constraint_type is required"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;

        let constraint = ConstraintNode::new(
            serde_json::from_str(&format!("\"{}\"", constraint_type))?,
            description.to_string(),
            None,
        );

        self.orchestrator
            .plan_manager()
            .add_constraint(plan_id, &constraint)
            .await?;
        Ok(serde_json::to_value(constraint)?)
    }

    async fn delete_constraint(&self, args: Value) -> Result<Value> {
        let constraint_id = parse_uuid(&args, "constraint_id")?;

        self.orchestrator.delete_constraint(constraint_id).await?;
        Ok(json!({"deleted": true}))
    }

    // ========================================================================
    // Release Handlers
    // ========================================================================

    async fn list_releases(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let (releases, total) = self
            .neo4j()
            .list_releases_filtered(project_id, status, limit, offset, None, "desc")
            .await?;

        Ok(json!({
            "items": releases,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_release(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;
        let version = args
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("version is required"))?;
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let release = ReleaseNode {
            id: Uuid::new_v4(),
            version: version.to_string(),
            title,
            description,
            status: ReleaseStatus::Planned,
            target_date,
            released_at: None,
            created_at: chrono::Utc::now(),
            project_id,
        };

        self.orchestrator.create_release(&release).await?;
        Ok(serde_json::to_value(release)?)
    }

    async fn get_release(&self, args: Value) -> Result<Value> {
        let release_id = parse_uuid(&args, "release_id")?;

        let release = self
            .neo4j()
            .get_release(release_id)
            .await?
            .ok_or_else(|| anyhow!("Release not found"))?;
        let tasks = self.neo4j().get_release_tasks(release_id).await?;

        Ok(json!({
            "release": release,
            "tasks": tasks
        }))
    }

    async fn update_release(&self, args: Value) -> Result<Value> {
        let release_id = parse_uuid(&args, "release_id")?;
        let status = match args.get("status").and_then(|v| v.as_str()) {
            Some(s) => Some(
                serde_json::from_str::<ReleaseStatus>(&format!("\"{}\"", s)).map_err(|_| {
                    anyhow!(
                        "Invalid release status '{}'. Valid: planned, in_progress, released, cancelled",
                        s
                    )
                })?,
            ),
            None => None,
        };
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let released_at = args
            .get("released_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.orchestrator
            .update_release(
                release_id,
                status,
                target_date,
                released_at,
                title,
                description,
            )
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn add_task_to_release(&self, args: Value) -> Result<Value> {
        let release_id = parse_uuid(&args, "release_id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        self.orchestrator
            .add_task_to_release(release_id, task_id)
            .await?;
        Ok(json!({"added": true}))
    }

    async fn add_commit_to_release(&self, args: Value) -> Result<Value> {
        let release_id = parse_uuid(&args, "release_id")?;
        let commit_sha = args
            .get("commit_sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("commit_sha is required"))?;

        self.orchestrator
            .add_commit_to_release(release_id, commit_sha)
            .await?;
        Ok(json!({"added": true}))
    }

    // ========================================================================
    // Milestone Handlers
    // ========================================================================

    async fn list_milestones(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let (milestones, total) = self
            .neo4j()
            .list_milestones_filtered(project_id, status, limit, offset, None, "asc")
            .await?;

        Ok(json!({
            "items": milestones,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_milestone(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("title is required"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let milestone = MilestoneNode {
            id: Uuid::new_v4(),
            title: title.to_string(),
            description,
            status: MilestoneStatus::Open,
            target_date,
            closed_at: None,
            created_at: chrono::Utc::now(),
            project_id,
        };

        self.orchestrator.create_milestone(&milestone).await?;
        Ok(serde_json::to_value(milestone)?)
    }

    async fn get_milestone(&self, args: Value) -> Result<Value> {
        let milestone_id = parse_uuid(&args, "milestone_id")?;

        let milestone = self
            .neo4j()
            .get_milestone(milestone_id)
            .await?
            .ok_or_else(|| anyhow!("Milestone not found"))?;
        let tasks = self.neo4j().get_milestone_tasks(milestone_id).await?;

        Ok(json!({
            "milestone": milestone,
            "tasks": tasks
        }))
    }

    async fn update_milestone(&self, args: Value) -> Result<Value> {
        let milestone_id = parse_uuid(&args, "milestone_id")?;
        let status = match args.get("status").and_then(|v| v.as_str()) {
            Some(s) => Some(
                serde_json::from_str::<MilestoneStatus>(&format!("\"{}\"", s)).map_err(|_| {
                    anyhow!("Invalid milestone status '{}'. Valid: planned, open, in_progress, completed, closed", s)
                })?,
            ),
            None => None,
        };
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let closed_at = args
            .get("closed_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.orchestrator
            .update_milestone(
                milestone_id,
                status,
                target_date,
                closed_at,
                title,
                description,
            )
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn get_milestone_progress(&self, args: Value) -> Result<Value> {
        let milestone_id = parse_uuid(&args, "milestone_id")?;

        let (completed, total) = self.neo4j().get_milestone_progress(milestone_id).await?;
        let percentage = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(json!({
            "completed": completed,
            "total": total,
            "percentage": percentage
        }))
    }

    async fn add_task_to_milestone(&self, args: Value) -> Result<Value> {
        let milestone_id = parse_uuid(&args, "milestone_id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        self.orchestrator
            .add_task_to_milestone(milestone_id, task_id)
            .await?;
        Ok(json!({"added": true}))
    }

    // ========================================================================
    // Commit Handlers
    // ========================================================================

    async fn create_commit(&self, args: Value) -> Result<Value> {
        let hash = args
            .get("sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sha is required"))?;
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("message is required"))?;
        let author = args
            .get("author")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let files_changed: Vec<String> = args
            .get("files_changed")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let commit = CommitNode {
            hash: hash.to_string(),
            message: message.to_string(),
            author: author.to_string(),
            timestamp: chrono::Utc::now(),
        };

        self.orchestrator.create_commit(&commit).await?;

        // Trigger incremental sync if files_changed and project_id are provided
        let sync_triggered = !files_changed.is_empty() && project_id.is_some();
        if sync_triggered {
            let project_id = project_id.unwrap();
            let orchestrator = self.orchestrator.clone();

            // Resolve project slug for MeiliSearch indexing
            let project_slug = self.neo4j().get_project(project_id).await?.map(|p| p.slug);

            tokio::spawn(async move {
                for file_path in &files_changed {
                    let path = std::path::Path::new(file_path);
                    if path.exists() {
                        if let Err(e) = orchestrator
                            .sync_file_for_project(path, Some(project_id), project_slug.as_deref())
                            .await
                        {
                            tracing::warn!("Incremental sync failed for {}: {}", file_path, e);
                        }
                    }
                }
                // Update last_synced timestamp
                if let Err(e) = orchestrator.neo4j().update_project_synced(project_id).await {
                    tracing::warn!("Failed to update last_synced: {}", e);
                }
            });
        }

        let mut result = serde_json::to_value(&commit)?;
        result
            .as_object_mut()
            .unwrap()
            .insert("sync_triggered".to_string(), json!(sync_triggered));
        Ok(result)
    }

    async fn link_commit_to_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let commit_sha = args
            .get("commit_sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("commit_sha is required"))?;

        self.orchestrator
            .link_commit_to_task(commit_sha, task_id)
            .await?;
        Ok(json!({"linked": true}))
    }

    async fn link_commit_to_plan(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;
        let commit_sha = args
            .get("commit_sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("commit_sha is required"))?;

        self.orchestrator
            .link_commit_to_plan(commit_sha, plan_id)
            .await?;
        Ok(json!({"linked": true}))
    }

    async fn get_task_commits(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;

        let commits = self.neo4j().get_task_commits(task_id).await?;
        Ok(serde_json::to_value(commits)?)
    }

    async fn get_plan_commits(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        let commits = self.neo4j().get_plan_commits(plan_id).await?;
        Ok(serde_json::to_value(commits)?)
    }

    // ========================================================================
    // Code Exploration Handlers
    // ========================================================================

    async fn search_code(&self, args: Value) -> Result<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("query is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let language = args.get("language").and_then(|v| v.as_str());
        let project_slug = args.get("project_slug").and_then(|v| v.as_str());
        let path_prefix = args.get("path_prefix").and_then(|v| v.as_str());

        let results = self
            .meili()
            .search_code_with_scores(query, limit, language, project_slug, path_prefix)
            .await?;

        Ok(serde_json::to_value(results)?)
    }

    async fn search_project_code(&self, args: Value) -> Result<Value> {
        let project_slug = args
            .get("project_slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("project_slug is required"))?;
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("query is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let language = args.get("language").and_then(|v| v.as_str());
        let path_prefix = args.get("path_prefix").and_then(|v| v.as_str());

        let results = self
            .meili()
            .search_code_with_scores(query, limit, language, Some(project_slug), path_prefix)
            .await?;

        Ok(serde_json::to_value(results)?)
    }

    async fn search_workspace_code(&self, args: Value) -> Result<Value> {
        let workspace_slug = args
            .get("workspace_slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("workspace_slug is required"))?;
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("query is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let language = args.get("language").and_then(|v| v.as_str());
        let path_prefix = args.get("path_prefix").and_then(|v| v.as_str());

        // Resolve workspace to project slugs
        let workspace = self
            .neo4j()
            .get_workspace_by_slug(workspace_slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found: {}", workspace_slug))?;
        let projects = self.neo4j().list_workspace_projects(workspace.id).await?;

        if projects.is_empty() {
            return Ok(json!([]));
        }

        // Search across all projects and merge results
        let mut all_hits = Vec::new();
        for project in &projects {
            let hits = self
                .meili()
                .search_code_with_scores(query, limit, language, Some(&project.slug), path_prefix)
                .await
                .unwrap_or_default();
            all_hits.extend(hits);
        }

        // Sort by score descending and truncate to limit
        all_hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hits.truncate(limit);

        Ok(serde_json::to_value(all_hits)?)
    }

    async fn get_file_symbols(&self, args: Value) -> Result<Value> {
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("file_path is required"))?;

        let symbols = self.neo4j().get_file_symbol_names(file_path).await?;

        Ok(json!({
            "file_path": file_path,
            "functions": symbols.functions,
            "structs": symbols.structs,
            "traits": symbols.traits,
            "enums": symbols.enums
        }))
    }

    async fn find_references(&self, args: Value) -> Result<Value> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("symbol is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let project_id = if let Some(slug) = args.get("project_slug").and_then(|v| v.as_str()) {
            Some(
                self.neo4j()
                    .get_project_by_slug(slug)
                    .await?
                    .ok_or_else(|| anyhow!("Project not found: {}", slug))?
                    .id,
            )
        } else {
            None
        };

        let refs = self
            .neo4j()
            .find_symbol_references(symbol, limit, project_id)
            .await?;
        let references: Vec<Value> = refs
            .into_iter()
            .map(|r| {
                json!({
                    "file_path": r.file_path,
                    "line": r.line,
                    "context": r.context,
                    "type": r.reference_type
                })
            })
            .collect();

        Ok(json!({
            "symbol": symbol,
            "references": references
        }))
    }

    async fn get_file_dependencies(&self, args: Value) -> Result<Value> {
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("file_path is required"))?;

        // Get files that depend on this file
        let dependents = self
            .neo4j()
            .find_dependent_files(file_path, 3, None)
            .await?;

        // Get files this file imports
        let direct_imports = self.neo4j().get_file_direct_imports(file_path).await?;
        let imports: Vec<String> = direct_imports.into_iter().map(|i| i.path).collect();

        Ok(json!({
            "imports": imports,
            "dependents": dependents
        }))
    }

    async fn get_call_graph(&self, args: Value) -> Result<Value> {
        let function = args
            .get("function")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("function is required"))?;
        let depth = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(2) as u32;

        let project_id = if let Some(slug) = args.get("project_slug").and_then(|v| v.as_str()) {
            Some(
                self.neo4j()
                    .get_project_by_slug(slug)
                    .await?
                    .ok_or_else(|| anyhow!("Project not found: {}", slug))?
                    .id,
            )
        } else {
            None
        };

        let callers = self
            .neo4j()
            .get_function_callers_by_name(function, depth, project_id)
            .await?;
        let callees = self
            .neo4j()
            .get_function_callees_by_name(function, depth, project_id)
            .await?;

        Ok(json!({
            "function": function,
            "callers": callers,
            "callees": callees
        }))
    }

    async fn analyze_impact(&self, args: Value) -> Result<Value> {
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target is required"))?;

        let project_id = if let Some(slug) = args.get("project_slug").and_then(|v| v.as_str()) {
            Some(
                self.neo4j()
                    .get_project_by_slug(slug)
                    .await?
                    .ok_or_else(|| anyhow!("Project not found: {}", slug))?
                    .id,
            )
        } else {
            None
        };

        // Auto-detect target_type: contains '/' or '.' → file, otherwise symbol
        let target_type = args
            .get("target_type")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                if target.contains('/') || target.contains('.') {
                    "file"
                } else {
                    "symbol"
                }
            });

        let (directly_affected, transitively_affected, caller_count) = if target_type == "file" {
            let direct = self
                .neo4j()
                .find_dependent_files(target, 1, project_id)
                .await?;
            let transitive = self
                .neo4j()
                .find_dependent_files(target, 3, project_id)
                .await?;
            let caller_count = direct.len() as i64;
            (direct, transitive, caller_count)
        } else {
            // Symbol mode: use find_callers for direct impact
            let callers = self.neo4j().find_callers(target, project_id).await?;
            let direct: Vec<String> = callers.iter().map(|f| f.file_path.clone()).collect();
            let caller_count = self
                .neo4j()
                .get_function_caller_count(target, project_id)
                .await?;
            (direct.clone(), direct, caller_count)
        };

        // Filter test files
        let test_files: Vec<String> = transitively_affected
            .iter()
            .filter(|p| p.contains("test") || p.contains("_test") || p.ends_with("_tests.rs"))
            .cloned()
            .collect();

        // Compute risk level
        let risk_level = if transitively_affected.len() > 10 || caller_count > 10 {
            "high"
        } else if transitively_affected.len() > 3 || caller_count > 3 {
            "medium"
        } else {
            "low"
        };

        // Generate suggestion
        let suggestion = format!(
            "Run tests in: {}. Consider reviewing: {}",
            if test_files.is_empty() {
                "(no test files found)".to_string()
            } else {
                test_files.join(", ")
            },
            directly_affected
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(json!({
            "target": target,
            "target_type": target_type,
            "directly_affected": directly_affected,
            "transitively_affected": transitively_affected,
            "test_files_affected": test_files,
            "caller_count": caller_count,
            "risk_level": risk_level,
            "suggestion": suggestion,
            // Backward compat aliases
            "dependent_files": transitively_affected,
            "impact_level": risk_level
        }))
    }

    async fn get_architecture(&self, args: Value) -> Result<Value> {
        let project_slug = args.get("project_slug").and_then(|v| v.as_str());

        let (lang_stats, connected) = if let Some(slug) = project_slug {
            let project = self
                .neo4j()
                .get_project_by_slug(slug)
                .await?
                .ok_or_else(|| anyhow!("Project not found: {}", slug))?;
            let project_id = project.id;
            let stats = self
                .neo4j()
                .get_language_stats_for_project(project_id)
                .await?;
            let files = self
                .neo4j()
                .get_most_connected_files_for_project(project_id, 10)
                .await?;
            (stats, files)
        } else {
            let stats = self.neo4j().get_language_stats().await?;
            let files = self.neo4j().get_most_connected_files_detailed(10).await?;
            (stats, files)
        };

        let languages: Vec<Value> = lang_stats
            .into_iter()
            .map(|s| json!({"language": s.language, "file_count": s.file_count}))
            .collect();

        let key_files: Vec<Value> = connected
            .into_iter()
            .map(|f| json!({"path": f.path, "imports": f.imports, "dependents": f.dependents}))
            .collect();

        Ok(json!({
            "languages": languages,
            "key_files": key_files
        }))
    }

    async fn find_similar_code(&self, args: Value) -> Result<Value> {
        let code_snippet = args
            .get("code_snippet")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("code_snippet is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let language = args.get("language").and_then(|v| v.as_str());
        let project_slug = args.get("project_slug").and_then(|v| v.as_str());

        // Use meilisearch to find similar code by searching for the snippet
        let results = self
            .meili()
            .search_code_with_scores(code_snippet, limit, language, project_slug, None)
            .await?;
        Ok(serde_json::to_value(results)?)
    }

    async fn find_trait_implementations(&self, args: Value) -> Result<Value> {
        let trait_name = args
            .get("trait_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("trait_name is required"))?;

        let impls = self.neo4j().find_trait_implementors(trait_name).await?;
        Ok(json!({
            "trait_name": trait_name,
            "implementors": impls
        }))
    }

    async fn find_type_traits(&self, args: Value) -> Result<Value> {
        let type_name = args
            .get("type_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("type_name is required"))?;

        let traits = self.neo4j().get_type_traits(type_name).await?;
        Ok(json!({
            "type_name": type_name,
            "traits": traits
        }))
    }

    async fn get_impl_blocks(&self, args: Value) -> Result<Value> {
        let type_name = args
            .get("type_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("type_name is required"))?;

        let impl_blocks = self.neo4j().get_impl_blocks(type_name).await?;
        Ok(json!({
            "type_name": type_name,
            "impl_blocks": impl_blocks
        }))
    }

    // ========================================================================
    // Decision Handlers
    // ========================================================================

    async fn search_decisions(&self, args: Value) -> Result<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("query is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let project_slug = args.get("project_slug").and_then(|v| v.as_str());

        let decisions = self
            .meili()
            .search_decisions_in_project(query, limit, project_slug)
            .await?;
        Ok(serde_json::to_value(decisions)?)
    }

    // ========================================================================
    // Sync Handlers
    // ========================================================================

    async fn sync_directory(&self, args: Value) -> Result<Value> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("path is required"))?;
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        // Resolve project slug when project_id is provided (needed for Meilisearch indexing)
        let project_slug = if let Some(pid) = project_id {
            self.neo4j().get_project(pid).await?.map(|p| p.slug)
        } else {
            None
        };

        let expanded = expand_tilde(path_str);
        let path = std::path::Path::new(&expanded);
        let result = self
            .orchestrator
            .sync_directory_for_project(path, project_id, project_slug.as_deref())
            .await?;

        // Update last_synced when project context is available
        if let Some(pid) = project_id {
            self.neo4j().update_project_synced(pid).await?;
        }

        Ok(json!({
            "files_synced": result.files_synced,
            "files_skipped": result.files_skipped,
            "files_deleted": result.files_deleted,
            "symbols_deleted": result.symbols_deleted,
            "errors": result.errors
        }))
    }

    async fn start_watch(&self, _args: Value) -> Result<Value> {
        // Note: This requires access to the watcher which is managed by ServerState
        // For now, return a message indicating it should be done via HTTP API
        Ok(json!({"message": "Use HTTP API POST /api/watch to start watcher"}))
    }

    async fn stop_watch(&self, _args: Value) -> Result<Value> {
        Ok(json!({"message": "Use HTTP API DELETE /api/watch to stop watcher"}))
    }

    async fn watch_status(&self, _args: Value) -> Result<Value> {
        Ok(json!({"message": "Use HTTP API GET /api/watch to get watcher status"}))
    }

    // ========================================================================
    // Meilisearch Handlers
    // ========================================================================

    async fn get_meilisearch_stats(&self, _args: Value) -> Result<Value> {
        let stats = self.meili().get_code_stats().await?;
        Ok(serde_json::to_value(stats)?)
    }

    async fn delete_meilisearch_orphans(&self, _args: Value) -> Result<Value> {
        self.meili().delete_orphan_code_documents().await?;
        Ok(json!({"deleted": true}))
    }

    async fn cleanup_cross_project_calls(&self, _args: Value) -> Result<Value> {
        let deleted = self.neo4j().cleanup_cross_project_calls().await?;
        Ok(json!({"deleted_count": deleted}))
    }

    // ========================================================================
    // Note Handlers
    // ========================================================================

    async fn list_notes(&self, args: Value) -> Result<Value> {
        use crate::notes::{NoteFilters, NoteImportance, NoteStatus, NoteType};

        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let filters = NoteFilters {
            note_type: args
                .get("note_type")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteType>().ok())
                .map(|t| vec![t]),
            status: args.get("status").and_then(|v| v.as_str()).map(|s| {
                s.split(',')
                    .filter_map(|s| s.trim().parse::<NoteStatus>().ok())
                    .collect()
            }),
            importance: args
                .get("importance")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteImportance>().ok())
                .map(|i| vec![i]),
            min_staleness: args.get("min_staleness").and_then(|v| v.as_f64()),
            max_staleness: args.get("max_staleness").and_then(|v| v.as_f64()),
            tags: args
                .get("tags")
                .and_then(|v| v.as_str())
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect()),
            search: args
                .get("search")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            limit: args.get("limit").and_then(|v| v.as_i64()),
            offset: args.get("offset").and_then(|v| v.as_i64()),
            global_only: args
                .get("global_only")
                .and_then(|v| v.as_bool())
                .or_else(|| {
                    args.get("global_only")
                        .and_then(|v| v.as_str())
                        .map(|s| s == "true")
                }),
            scope_type: None,
            sort_by: None,
            sort_order: None,
        };

        let (notes, total) = self
            .orchestrator
            .note_manager()
            .list_notes(project_id, None, &filters)
            .await?;

        Ok(json!({
            "items": notes,
            "total": total,
            "limit": filters.limit.unwrap_or(50),
            "offset": filters.offset.unwrap_or(0)
        }))
    }

    async fn create_note(&self, args: Value) -> Result<Value> {
        use crate::notes::{CreateNoteRequest, NoteImportance, NoteType};

        let project_id = parse_uuid(&args, "project_id").ok();
        let note_type: NoteType = args
            .get("note_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("note_type is required"))?
            .parse()
            .map_err(|_| anyhow!("Invalid note_type"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("content is required"))?
            .to_string();
        let importance = args
            .get("importance")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<NoteImportance>().ok());
        let tags = args.get("tags").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
        });

        let request = CreateNoteRequest {
            project_id,
            note_type,
            content,
            importance,
            scope: None,
            tags,
            anchors: None,
            assertion_rule: None,
        };

        let note = self
            .orchestrator
            .note_manager()
            .create_note(request, "mcp")
            .await?;

        Ok(serde_json::to_value(note)?)
    }

    async fn get_note(&self, args: Value) -> Result<Value> {
        let note_id = parse_uuid(&args, "note_id")?;

        let note = self
            .orchestrator
            .note_manager()
            .get_note(note_id)
            .await?
            .ok_or_else(|| anyhow!("Note not found"))?;

        Ok(serde_json::to_value(note)?)
    }

    async fn update_note(&self, args: Value) -> Result<Value> {
        use crate::notes::{NoteImportance, NoteStatus, UpdateNoteRequest};

        let note_id = parse_uuid(&args, "note_id")?;

        let request = UpdateNoteRequest {
            content: args
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            importance: args
                .get("importance")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteImportance>().ok()),
            status: args
                .get("status")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteStatus>().ok()),
            tags: args.get("tags").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
            }),
        };

        let note = self
            .orchestrator
            .note_manager()
            .update_note(note_id, request)
            .await?
            .ok_or_else(|| anyhow!("Note not found"))?;

        Ok(serde_json::to_value(note)?)
    }

    async fn delete_note(&self, args: Value) -> Result<Value> {
        let note_id = parse_uuid(&args, "note_id")?;

        let deleted = self
            .orchestrator
            .note_manager()
            .delete_note(note_id)
            .await?;

        Ok(json!({"deleted": deleted}))
    }

    async fn search_notes(&self, args: Value) -> Result<Value> {
        use crate::notes::{NoteFilters, NoteImportance, NoteStatus, NoteType};

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("query is required"))?;

        let filters = NoteFilters {
            note_type: args
                .get("note_type")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteType>().ok())
                .map(|t| vec![t]),
            status: args.get("status").and_then(|v| v.as_str()).map(|s| {
                s.split(',')
                    .filter_map(|s| s.trim().parse::<NoteStatus>().ok())
                    .collect()
            }),
            importance: args
                .get("importance")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<NoteImportance>().ok())
                .map(|i| vec![i]),
            search: args
                .get("project_slug")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            limit: args.get("limit").and_then(|v| v.as_i64()),
            ..Default::default()
        };

        let hits = self
            .orchestrator
            .note_manager()
            .search_notes(query, &filters)
            .await?;

        Ok(serde_json::to_value(hits)?)
    }

    async fn confirm_note(&self, args: Value) -> Result<Value> {
        let note_id = parse_uuid(&args, "note_id")?;

        let note = self
            .orchestrator
            .note_manager()
            .confirm_note(note_id, "mcp")
            .await?
            .ok_or_else(|| anyhow!("Note not found"))?;

        Ok(serde_json::to_value(note)?)
    }

    async fn invalidate_note(&self, args: Value) -> Result<Value> {
        let note_id = parse_uuid(&args, "note_id")?;
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("reason is required"))?;

        let note = self
            .orchestrator
            .note_manager()
            .invalidate_note(note_id, reason, "mcp")
            .await?
            .ok_or_else(|| anyhow!("Note not found"))?;

        Ok(serde_json::to_value(note)?)
    }

    async fn supersede_note(&self, args: Value) -> Result<Value> {
        use crate::notes::{CreateNoteRequest, NoteImportance, NoteType};

        let old_note_id = parse_uuid(&args, "old_note_id")?;
        let project_id = parse_uuid(&args, "project_id").ok();
        let note_type: NoteType = args
            .get("note_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("note_type is required"))?
            .parse()
            .map_err(|_| anyhow!("Invalid note_type"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("content is required"))?
            .to_string();
        let importance = args
            .get("importance")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<NoteImportance>().ok());
        let tags = args.get("tags").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
        });

        let request = CreateNoteRequest {
            project_id,
            note_type,
            content,
            importance,
            scope: None,
            tags,
            anchors: None,
            assertion_rule: None,
        };

        let new_note = self
            .orchestrator
            .note_manager()
            .supersede_note(old_note_id, request, "mcp")
            .await?;

        Ok(serde_json::to_value(new_note)?)
    }

    async fn link_note_to_entity(&self, args: Value) -> Result<Value> {
        use crate::notes::{EntityType, LinkNoteRequest};

        let note_id = parse_uuid(&args, "note_id")?;
        let entity_type: EntityType = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?
            .parse()
            .map_err(|_| anyhow!("Invalid entity_type"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?
            .to_string();

        let request = LinkNoteRequest {
            entity_type,
            entity_id,
        };

        self.orchestrator
            .note_manager()
            .link_note_to_entity(note_id, &request)
            .await?;

        Ok(json!({"linked": true}))
    }

    async fn unlink_note_from_entity(&self, args: Value) -> Result<Value> {
        use crate::notes::EntityType;

        let note_id = parse_uuid(&args, "note_id")?;
        let entity_type: EntityType = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?
            .parse()
            .map_err(|_| anyhow!("Invalid entity_type"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?;

        self.orchestrator
            .note_manager()
            .unlink_note_from_entity(note_id, &entity_type, entity_id)
            .await?;

        Ok(json!({"unlinked": true}))
    }

    async fn get_context_notes(&self, args: Value) -> Result<Value> {
        use crate::notes::EntityType;

        let entity_type: EntityType = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?
            .parse()
            .map_err(|_| anyhow!("Invalid entity_type"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?;
        let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as u32;
        let min_score = args
            .get("min_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        let response = self
            .orchestrator
            .note_manager()
            .get_context_notes(&entity_type, entity_id, max_depth, min_score)
            .await?;

        Ok(serde_json::to_value(response)?)
    }

    async fn get_notes_needing_review(&self, args: Value) -> Result<Value> {
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let notes = self
            .orchestrator
            .note_manager()
            .get_notes_needing_review(project_id)
            .await?;

        Ok(serde_json::to_value(notes)?)
    }

    async fn update_staleness_scores(&self, _args: Value) -> Result<Value> {
        let count = self
            .orchestrator
            .note_manager()
            .update_staleness_scores()
            .await?;

        Ok(json!({"notes_updated": count}))
    }

    async fn list_project_notes(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id")?;
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(50);
        let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

        let filters = crate::notes::NoteFilters {
            limit: Some(limit),
            offset: Some(offset),
            ..Default::default()
        };

        let (notes, total) = self
            .orchestrator
            .note_manager()
            .list_project_notes(project_id, &filters)
            .await?;

        Ok(json!({
            "items": notes,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn get_propagated_notes(&self, args: Value) -> Result<Value> {
        let entity_type_str = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?;
        let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as u32;
        let min_score = args
            .get("min_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        let entity_type: crate::notes::EntityType = entity_type_str
            .parse()
            .map_err(|_| anyhow!("Invalid entity type: {}", entity_type_str))?;

        let notes = self
            .orchestrator
            .note_manager()
            .get_propagated_notes(&entity_type, entity_id, max_depth, min_score)
            .await?;

        Ok(serde_json::to_value(notes)?)
    }

    async fn get_entity_notes(&self, args: Value) -> Result<Value> {
        let entity_type_str = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?;

        let entity_type: crate::notes::EntityType = entity_type_str
            .parse()
            .map_err(|_| anyhow!("Invalid entity type: {}", entity_type_str))?;

        let notes = self
            .orchestrator
            .neo4j()
            .get_notes_for_entity(&entity_type, entity_id)
            .await?;

        Ok(serde_json::to_value(notes)?)
    }

    // ========================================================================
    // Workspace Handlers
    // ========================================================================

    async fn list_workspaces(&self, args: Value) -> Result<Value> {
        let search = args.get("search").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        // Get all workspaces and filter/paginate in memory
        let all_workspaces = self.neo4j().list_workspaces().await?;

        // Filter by search if provided
        let filtered: Vec<_> = if let Some(search_term) = search {
            let search_lower = search_term.to_lowercase();
            all_workspaces
                .into_iter()
                .filter(|w| {
                    w.name.to_lowercase().contains(&search_lower)
                        || w.description
                            .as_ref()
                            .map(|d| d.to_lowercase().contains(&search_lower))
                            .unwrap_or(false)
                })
                .collect()
        } else {
            all_workspaces
        };

        let total = filtered.len();
        let items: Vec<_> = filtered.into_iter().skip(offset).take(limit).collect();

        Ok(json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_workspace(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| slugify(name));
        let description = args.get("description").and_then(|v| v.as_str());
        let metadata = args
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));

        let workspace = WorkspaceNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            slug,
            description: description.map(|s| s.to_string()),
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata,
        };

        self.orchestrator.create_workspace(&workspace).await?;
        Ok(serde_json::to_value(workspace)?)
    }

    async fn get_workspace(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        Ok(serde_json::to_value(workspace)?)
    }

    async fn update_workspace(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let metadata = args.get("metadata").cloned();

        // Get workspace by slug first
        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        self.orchestrator
            .update_workspace(workspace.id, name, description, metadata)
            .await?;

        // Fetch updated workspace
        let updated = self
            .neo4j()
            .get_workspace(workspace.id)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        Ok(serde_json::to_value(updated)?)
    }

    async fn delete_workspace(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        // Get workspace by slug first
        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        self.orchestrator.delete_workspace(workspace.id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn get_workspace_overview(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let projects = self.neo4j().list_workspace_projects(workspace.id).await?;
        let milestones = self.neo4j().list_workspace_milestones(workspace.id).await?;
        let resources = self.neo4j().list_workspace_resources(workspace.id).await?;
        let components = self.neo4j().list_components(workspace.id).await?;

        // Calculate progress from all projects' tasks
        let mut total_tasks = 0u32;
        let mut completed_tasks = 0u32;
        for project in &projects {
            let (t, c, _, _) = self.neo4j().get_project_progress(project.id).await?;
            total_tasks += t;
            completed_tasks += c;
        }

        Ok(json!({
            "workspace": workspace,
            "projects": projects,
            "milestones": milestones,
            "resources": resources,
            "components": components,
            "progress": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "percentage": if total_tasks > 0 { (completed_tasks as f64 / total_tasks as f64 * 100.0).round() } else { 0.0 }
            }
        }))
    }

    async fn list_workspace_projects(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let projects = self.neo4j().list_workspace_projects(workspace.id).await?;
        Ok(serde_json::to_value(projects)?)
    }

    async fn add_project_to_workspace(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let project_id = parse_uuid(&args, "project_id")?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        self.orchestrator
            .add_project_to_workspace(workspace.id, project_id)
            .await?;

        Ok(json!({"added": true}))
    }

    async fn remove_project_from_workspace(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let project_id = parse_uuid(&args, "project_id")?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        self.orchestrator
            .remove_project_from_workspace(workspace.id, project_id)
            .await?;

        Ok(json!({"removed": true}))
    }

    // ========================================================================
    // Workspace Milestone Handlers
    // ========================================================================

    async fn list_all_workspace_milestones(&self, args: Value) -> Result<Value> {
        let workspace_id = args
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .map_err(|_| anyhow!("Invalid workspace_id UUID"))?;
        let status_filter = args.get("status").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let total = self
            .neo4j()
            .count_all_workspace_milestones(workspace_id, status_filter)
            .await?;

        let results = self
            .neo4j()
            .list_all_workspace_milestones_filtered(workspace_id, status_filter, limit, offset)
            .await?;

        let items: Vec<Value> = results
            .into_iter()
            .map(|(m, wid, wname, wslug)| {
                let mut v = serde_json::to_value(&m).unwrap_or_default();
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("workspace_id".to_string(), json!(wid));
                    obj.insert("workspace_name".to_string(), json!(wname));
                    obj.insert("workspace_slug".to_string(), json!(wslug));
                }
                v
            })
            .collect();

        Ok(json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn list_workspace_milestones(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let status_filter = args.get("status").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let (items, total) = self
            .neo4j()
            .list_workspace_milestones_filtered(workspace.id, status_filter, limit, offset)
            .await?;

        Ok(json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_workspace_milestone(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("title is required"))?;
        let description = args.get("description").and_then(|v| v.as_str());
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let milestone = WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: workspace.id,
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            status: MilestoneStatus::Open,
            target_date,
            closed_at: None,
            created_at: chrono::Utc::now(),
            tags,
        };

        self.orchestrator
            .create_workspace_milestone(&milestone)
            .await?;
        Ok(serde_json::to_value(milestone)?)
    }

    async fn get_workspace_milestone(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        let milestone = self
            .neo4j()
            .get_workspace_milestone(id)
            .await?
            .ok_or_else(|| anyhow!("Workspace milestone not found"))?;

        let tasks = self.neo4j().get_workspace_milestone_tasks(id).await?;

        Ok(json!({
            "milestone": milestone,
            "tasks": tasks
        }))
    }

    async fn update_workspace_milestone(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let status = match args.get("status").and_then(|v| v.as_str()) {
            Some(s) => Some(
                serde_json::from_str::<MilestoneStatus>(&format!("\"{}\"", s)).map_err(|_| {
                    anyhow!("Invalid milestone status '{}'. Valid: planned, open, in_progress, completed, closed", s)
                })?,
            ),
            None => None,
        };
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        self.orchestrator
            .update_workspace_milestone(id, title, description, status, target_date)
            .await?;

        let milestone = self
            .neo4j()
            .get_workspace_milestone(id)
            .await?
            .ok_or_else(|| anyhow!("Workspace milestone not found"))?;

        Ok(serde_json::to_value(milestone)?)
    }

    async fn delete_workspace_milestone(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        self.orchestrator.delete_workspace_milestone(id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn add_task_to_workspace_milestone(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        self.orchestrator
            .add_task_to_workspace_milestone(id, task_id)
            .await?;

        Ok(json!({"added": true}))
    }

    async fn get_workspace_milestone_progress(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        let (total, completed, in_progress, pending) =
            self.neo4j().get_workspace_milestone_progress(id).await?;

        let percentage = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round()
        } else {
            0.0
        };

        Ok(json!({
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "percentage": percentage
        }))
    }

    // ========================================================================
    // Resource Handlers
    // ========================================================================

    async fn list_resources(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let resource_type_filter = args.get("resource_type").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        // Get all resources and filter in memory
        let all_resources = self.neo4j().list_workspace_resources(workspace.id).await?;

        // Filter by type if provided
        let filtered: Vec<_> = if let Some(type_str) = resource_type_filter {
            if let Ok(rt) = type_str.parse::<ResourceType>() {
                all_resources
                    .into_iter()
                    .filter(|r| r.resource_type == rt)
                    .collect()
            } else {
                all_resources
            }
        } else {
            all_resources
        };

        let total = filtered.len();
        let items: Vec<_> = filtered.into_iter().skip(offset).take(limit).collect();

        Ok(json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_resource(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let resource_type_str = args
            .get("resource_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("resource_type is required"))?;
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("file_path is required"))?;
        let url = args.get("url").and_then(|v| v.as_str());
        let format = args.get("format").and_then(|v| v.as_str());
        let version = args.get("version").and_then(|v| v.as_str());
        let description = args.get("description").and_then(|v| v.as_str());
        let metadata = args
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let resource_type: ResourceType = resource_type_str
            .parse()
            .map_err(|_| anyhow!("Invalid resource_type"))?;

        let resource = ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(workspace.id),
            project_id: None,
            name: name.to_string(),
            resource_type,
            file_path: file_path.to_string(),
            url: url.map(|s| s.to_string()),
            format: format.map(|s| s.to_string()),
            version: version.map(|s| s.to_string()),
            description: description.map(|s| s.to_string()),
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata,
        };

        self.orchestrator.create_resource(&resource).await?;
        Ok(serde_json::to_value(resource)?)
    }

    async fn get_resource(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        let resource = self
            .neo4j()
            .get_resource(id)
            .await?
            .ok_or_else(|| anyhow!("Resource not found"))?;

        Ok(serde_json::to_value(resource)?)
    }

    async fn delete_resource(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        self.orchestrator.delete_resource(id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn link_resource_to_project(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let project_id = parse_uuid(&args, "project_id")?;
        let link_type = args
            .get("link_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("link_type is required (implements or uses)"))?;

        match link_type.to_lowercase().as_str() {
            "implements" => {
                self.orchestrator
                    .link_project_implements_resource(project_id, id)
                    .await?;
            }
            "uses" => {
                self.orchestrator
                    .link_project_uses_resource(project_id, id)
                    .await?;
            }
            _ => return Err(anyhow!("link_type must be 'implements' or 'uses'")),
        }

        Ok(json!({"linked": true, "link_type": link_type}))
    }

    // ========================================================================
    // Component Handlers
    // ========================================================================

    async fn list_components(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let component_type_filter = args.get("component_type").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        // Get all components and filter in memory
        let all_components = self.neo4j().list_components(workspace.id).await?;

        // Filter by type if provided
        let filtered: Vec<_> = if let Some(type_str) = component_type_filter {
            if let Ok(ct) = type_str.parse::<ComponentType>() {
                all_components
                    .into_iter()
                    .filter(|c| c.component_type == ct)
                    .collect()
            } else {
                all_components
            }
        } else {
            all_components
        };

        let total = filtered.len();
        let items: Vec<_> = filtered.into_iter().skip(offset).take(limit).collect();

        Ok(json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn create_component(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let component_type_str = args
            .get("component_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("component_type is required"))?;
        let description = args.get("description").and_then(|v| v.as_str());
        let runtime = args.get("runtime").and_then(|v| v.as_str());
        let config = args
            .get("config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let component_type: ComponentType = component_type_str
            .parse()
            .map_err(|_| anyhow!("Invalid component_type"))?;

        let component = ComponentNode {
            id: Uuid::new_v4(),
            workspace_id: workspace.id,
            name: name.to_string(),
            component_type,
            description: description.map(|s| s.to_string()),
            runtime: runtime.map(|s| s.to_string()),
            config,
            created_at: chrono::Utc::now(),
            tags,
        };

        self.orchestrator.create_component(&component).await?;
        Ok(serde_json::to_value(component)?)
    }

    async fn get_component(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        let component = self
            .neo4j()
            .get_component(id)
            .await?
            .ok_or_else(|| anyhow!("Component not found"))?;

        Ok(serde_json::to_value(component)?)
    }

    async fn delete_component(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;

        self.orchestrator.delete_component(id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn add_component_dependency(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let depends_on_id = parse_uuid(&args, "depends_on_id")?;
        let protocol = args
            .get("protocol")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let required = args
            .get("required")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        self.orchestrator
            .add_component_dependency(id, depends_on_id, protocol, required)
            .await?;

        Ok(json!({"added": true}))
    }

    async fn remove_component_dependency(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let dep_id = parse_uuid(&args, "dep_id")?;

        self.orchestrator
            .remove_component_dependency(id, dep_id)
            .await?;

        Ok(json!({"removed": true}))
    }

    async fn map_component_to_project(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let project_id = parse_uuid(&args, "project_id")?;

        self.orchestrator
            .map_component_to_project(id, project_id)
            .await?;

        Ok(json!({"mapped": true}))
    }

    async fn get_workspace_topology(&self, args: Value) -> Result<Value> {
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"))?;

        let workspace = self
            .neo4j()
            .get_workspace_by_slug(slug)
            .await?
            .ok_or_else(|| anyhow!("Workspace not found"))?;

        let topology = self.neo4j().get_workspace_topology(workspace.id).await?;
        Ok(serde_json::to_value(topology)?)
    }

    // ========================================================================
    // Additional CRUD Handlers (get_step, delete_step, decisions, constraints,
    // delete_release, delete_milestone, update_resource, update_component)
    // ========================================================================

    async fn get_step(&self, args: Value) -> Result<Value> {
        let step_id = parse_uuid(&args, "step_id")?;
        let step = self
            .neo4j()
            .get_step(step_id)
            .await?
            .ok_or_else(|| anyhow!("Step not found"))?;
        Ok(serde_json::to_value(step)?)
    }

    async fn delete_step_handler(&self, args: Value) -> Result<Value> {
        let step_id = parse_uuid(&args, "step_id")?;
        self.orchestrator.delete_step(step_id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn get_constraint(&self, args: Value) -> Result<Value> {
        let constraint_id = parse_uuid(&args, "constraint_id")?;
        let constraint = self
            .neo4j()
            .get_constraint(constraint_id)
            .await?
            .ok_or_else(|| anyhow!("Constraint not found"))?;
        Ok(serde_json::to_value(constraint)?)
    }

    async fn update_constraint(&self, args: Value) -> Result<Value> {
        let constraint_id = parse_uuid(&args, "constraint_id")?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let constraint_type = match args.get("constraint_type").and_then(|v| v.as_str()) {
            Some(s) => Some(
                serde_json::from_str::<ConstraintType>(&format!("\"{}\"", s)).map_err(|_| {
                    anyhow!(
                        "Invalid constraint type '{}'. Valid: performance, compatibility, security, style, testing, other",
                        s
                    )
                })?,
            ),
            None => None,
        };
        let enforced_by = args
            .get("enforced_by")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.orchestrator
            .update_constraint(constraint_id, description, constraint_type, enforced_by)
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn delete_release(&self, args: Value) -> Result<Value> {
        let release_id = parse_uuid(&args, "release_id")?;
        self.orchestrator.delete_release(release_id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn delete_milestone(&self, args: Value) -> Result<Value> {
        let milestone_id = parse_uuid(&args, "milestone_id")?;
        self.orchestrator.delete_milestone(milestone_id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn get_decision(&self, args: Value) -> Result<Value> {
        let decision_id = parse_uuid(&args, "decision_id")?;
        let decision = self
            .neo4j()
            .get_decision(decision_id)
            .await?
            .ok_or_else(|| anyhow!("Decision not found"))?;
        Ok(serde_json::to_value(decision)?)
    }

    async fn update_decision(&self, args: Value) -> Result<Value> {
        let decision_id = parse_uuid(&args, "decision_id")?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let rationale = args
            .get("rationale")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let chosen_option = args
            .get("chosen_option")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.orchestrator
            .update_decision(decision_id, description, rationale, chosen_option)
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn delete_decision(&self, args: Value) -> Result<Value> {
        let decision_id = parse_uuid(&args, "decision_id")?;
        self.orchestrator.delete_decision(decision_id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn update_resource(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let version = args
            .get("version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.orchestrator
            .update_resource(id, name, file_path, url, version, description)
            .await?;
        Ok(json!({"updated": true}))
    }

    async fn update_component(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let runtime = args
            .get("runtime")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let config = args.get("config").cloned();
        let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

        self.orchestrator
            .update_component(id, name, description, runtime, config, tags)
            .await?;
        Ok(json!({"updated": true}))
    }

    // ========================================================================
    // Chat
    // ========================================================================

    async fn list_chat_sessions(&self, args: Value) -> Result<Value> {
        let project_slug = args.get("project_slug").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let (sessions, total) = self
            .neo4j()
            .list_chat_sessions(project_slug, None, limit, offset)
            .await?;

        Ok(json!({
            "items": sessions,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }

    async fn get_chat_session(&self, args: Value) -> Result<Value> {
        let session_id = parse_uuid(&args, "session_id")?;
        let session = self
            .neo4j()
            .get_chat_session(session_id)
            .await?
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        Ok(serde_json::to_value(session)?)
    }

    async fn delete_chat_session(&self, args: Value) -> Result<Value> {
        let session_id = parse_uuid(&args, "session_id")?;

        // Close active session if chat manager is available
        if let Some(cm) = &self.chat_manager {
            let _ = cm.close_session(&session_id.to_string()).await;
        }

        let deleted = self.neo4j().delete_chat_session(session_id).await?;

        if deleted {
            Ok(json!({"deleted": true}))
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }

    async fn chat_send_message(&self, args: Value) -> Result<Value> {
        let cm = self
            .chat_manager
            .as_ref()
            .ok_or_else(|| anyhow!("Chat manager not initialized"))?;

        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("message is required"))?;
        let cwd = args
            .get("cwd")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("cwd is required"))?;
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let project_slug = args
            .get("project_slug")
            .and_then(|v| v.as_str())
            .map(String::from);
        let model = args.get("model").and_then(|v| v.as_str()).map(String::from);
        let permission_mode = args
            .get("permission_mode")
            .and_then(|v| v.as_str())
            .map(String::from);
        let workspace_slug = args
            .get("workspace_slug")
            .and_then(|v| v.as_str())
            .map(String::from);
        let add_dirs: Option<Vec<String>> =
            args.get("add_dirs").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });

        let request = crate::chat::ChatRequest {
            message: message.to_string(),
            session_id,
            cwd: cwd.to_string(),
            project_slug,
            model,
            permission_mode,
            add_dirs,
            workspace_slug,
        };

        // Create session and wait for it to complete (non-streaming for MCP)
        let response = cm.create_session(&request).await?;

        // Subscribe and collect all events until Result
        let rx = cm.subscribe(&response.session_id).await?;
        let mut stream = tokio_stream::wrappers::BroadcastStream::new(rx);

        let mut text_parts: Vec<String> = Vec::new();
        let mut result_session_id = response.session_id.clone();
        let mut cost_usd = None;
        let mut duration_ms = 0u64;

        use tokio_stream::StreamExt;
        while let Some(Ok(event)) = stream.next().await {
            match event {
                crate::chat::ChatEvent::AssistantText { content, .. } => {
                    text_parts.push(content);
                }
                crate::chat::ChatEvent::Result {
                    session_id: sid,
                    duration_ms: dur,
                    cost_usd: cost,
                    ..
                } => {
                    result_session_id = sid;
                    duration_ms = dur;
                    cost_usd = cost;
                    break;
                }
                crate::chat::ChatEvent::Error { message, .. } => {
                    return Err(anyhow!("Chat error: {}", message));
                }
                _ => {} // Skip tool_use, tool_result, thinking, etc.
            }
        }

        Ok(json!({
            "session_id": result_session_id,
            "response": text_parts.join(""),
            "duration_ms": duration_ms,
            "cost_usd": cost_usd
        }))
    }

    async fn list_chat_messages(&self, args: Value) -> Result<Value> {
        let cm = self
            .chat_manager
            .as_ref()
            .ok_or_else(|| anyhow!("Chat manager not initialized"))?;

        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("session_id is required"))?;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let loaded = cm.get_session_messages(session_id, limit, offset).await?;

        let events: Vec<serde_json::Value> = loaded
            .events
            .iter()
            .map(|e| {
                let mut obj = serde_json::from_str::<serde_json::Value>(&e.data)
                    .unwrap_or_else(|_| json!({ "type": e.event_type, "raw": e.data }));
                if let Some(map) = obj.as_object_mut() {
                    map.insert("seq".to_string(), json!(e.seq));
                    map.insert("created_at".to_string(), json!(e.created_at.to_rfc3339()));
                }
                obj
            })
            .collect();

        Ok(json!({
            "events": events,
            "total_count": loaded.total_count,
            "has_more": loaded.has_more,
            "offset": loaded.offset,
            "limit": loaded.limit,
        }))
    }

    // ====================================================================
    // Feature Graph handlers
    // ====================================================================

    async fn create_feature_graph(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let project_id = parse_uuid(&args, "project_id")?;
        let description = args.get("description").and_then(|v| v.as_str());

        let fg = FeatureGraphNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: description.map(|d| d.to_string()),
            project_id,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            entity_count: None,
            entry_function: None,
            build_depth: None,
            include_relations: None,
        };

        self.neo4j().create_feature_graph(&fg).await?;

        Ok(json!({
            "id": fg.id.to_string(),
            "name": fg.name,
            "project_id": fg.project_id.to_string(),
            "created_at": fg.created_at.to_rfc3339(),
        }))
    }

    async fn get_feature_graph(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let detail = self
            .neo4j()
            .get_feature_graph_detail(id)
            .await?
            .ok_or_else(|| anyhow!("Feature graph not found: {}", id))?;

        let entities: Vec<Value> = detail
            .entities
            .iter()
            .map(|e| {
                json!({
                    "entity_type": e.entity_type,
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "role": e.role,
                })
            })
            .collect();

        Ok(json!({
            "id": detail.graph.id.to_string(),
            "name": detail.graph.name,
            "description": detail.graph.description,
            "project_id": detail.graph.project_id.to_string(),
            "created_at": detail.graph.created_at.to_rfc3339(),
            "updated_at": detail.graph.updated_at.to_rfc3339(),
            "entities": entities,
            "entity_count": entities.len(),
        }))
    }

    async fn list_feature_graphs(&self, args: Value) -> Result<Value> {
        let project_id = parse_uuid(&args, "project_id").ok();
        let graphs = self.neo4j().list_feature_graphs(project_id).await?;

        let items: Vec<Value> = graphs
            .iter()
            .map(|fg| {
                json!({
                    "id": fg.id.to_string(),
                    "name": fg.name,
                    "description": fg.description,
                    "project_id": fg.project_id.to_string(),
                    "created_at": fg.created_at.to_rfc3339(),
                })
            })
            .collect();

        Ok(json!({
            "feature_graphs": items,
            "count": items.len(),
        }))
    }

    async fn add_to_feature_graph(&self, args: Value) -> Result<Value> {
        let feature_graph_id = parse_uuid(&args, "feature_graph_id")?;
        let entity_type = args
            .get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_type is required"))?;
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entity_id is required"))?;

        let role = args.get("role").and_then(|v| v.as_str());

        self.neo4j()
            .add_entity_to_feature_graph(feature_graph_id, entity_type, entity_id, role)
            .await?;

        Ok(json!({
            "added": true,
            "feature_graph_id": feature_graph_id.to_string(),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "role": role,
        }))
    }

    async fn auto_build_feature_graph(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("name is required"))?;
        let project_id = parse_uuid(&args, "project_id")?;
        let entry_function = args
            .get("entry_function")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("entry_function is required"))?;
        let description = args.get("description").and_then(|v| v.as_str());
        let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
        let include_relations: Option<Vec<String>> = args
            .get("include_relations")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            });

        let detail = self
            .neo4j()
            .auto_build_feature_graph(
                name,
                description,
                project_id,
                entry_function,
                depth,
                include_relations.as_deref(),
            )
            .await?;

        let entities: Vec<Value> = detail
            .entities
            .iter()
            .map(|e| {
                json!({
                    "entity_type": e.entity_type,
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "role": e.role,
                })
            })
            .collect();

        Ok(json!({
            "id": detail.graph.id.to_string(),
            "name": detail.graph.name,
            "description": detail.graph.description,
            "project_id": detail.graph.project_id.to_string(),
            "entities": entities,
            "entity_count": entities.len(),
        }))
    }

    async fn delete_feature_graph(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let deleted = self.neo4j().delete_feature_graph(id).await?;
        Ok(json!({ "deleted": deleted }))
    }

    // ========================================================================
    // Implementation Planner
    // ========================================================================

    async fn plan_implementation(&self, args: Value) -> Result<Value> {
        use crate::orchestrator::planner::{PlanRequest, PlanScope};

        let project_slug = args
            .get("project_slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("project_slug is required"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("description is required"))?;

        // Resolve project slug → project ID
        let project = self
            .neo4j()
            .get_project_by_slug(project_slug)
            .await?
            .ok_or_else(|| anyhow!("Project '{}' not found", project_slug))?;

        let entry_points: Option<Vec<String>> = args
            .get("entry_points")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });

        let scope: Option<PlanScope> = args
            .get("scope")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());

        let auto_create_plan = args.get("auto_create_plan").and_then(|v| v.as_bool());

        let request = PlanRequest {
            project_id: project.id,
            project_slug: Some(project_slug.to_string()),
            description: description.to_string(),
            entry_points,
            scope,
            auto_create_plan,
        };

        let plan = self
            .orchestrator
            .planner()
            .plan_implementation(request)
            .await?;

        Ok(serde_json::to_value(&plan)?)
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn parse_uuid(args: &Value, field: &str) -> Result<Uuid> {
    args.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("{} is required", field))?
        .parse()
        .map_err(|_| anyhow!("{} must be a valid UUID", field))
}

fn slugify(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;
    use serde_json::json;

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_parse_uuid_valid() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let args = json!({"id": uuid_str});
        let result = parse_uuid(&args, "id");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().to_string(), uuid_str);
    }

    #[test]
    fn test_parse_uuid_missing_field() {
        let args = json!({});
        let result = parse_uuid(&args, "id");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("id is required"));
    }

    #[test]
    fn test_parse_uuid_invalid_format() {
        let args = json!({"id": "not-a-uuid"});
        let result = parse_uuid(&args, "id");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be a valid UUID"));
    }

    #[test]
    fn test_parse_uuid_null_value() {
        let args = json!({"id": null});
        let result = parse_uuid(&args, "id");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_uuid_number_value() {
        let args = json!({"id": 12345});
        let result = parse_uuid(&args, "id");
        assert!(result.is_err());
    }

    #[test]
    fn test_slugify_simple() {
        assert_eq!(slugify("My Project"), "my-project");
    }

    #[test]
    fn test_slugify_special_chars() {
        assert_eq!(slugify("Project @#$ Name!"), "project-name");
    }

    #[test]
    fn test_slugify_multiple_spaces() {
        assert_eq!(slugify("Multiple   Spaces   Here"), "multiple-spaces-here");
    }

    #[test]
    fn test_slugify_already_slug() {
        assert_eq!(slugify("already-a-slug"), "already-a-slug");
    }

    #[test]
    fn test_slugify_uppercase() {
        assert_eq!(slugify("UPPERCASE"), "uppercase");
    }

    #[test]
    fn test_slugify_numbers() {
        assert_eq!(slugify("Project 123"), "project-123");
    }

    #[test]
    fn test_slugify_leading_trailing_special() {
        assert_eq!(slugify("---Project---"), "project");
    }

    #[test]
    fn test_slugify_empty() {
        assert_eq!(slugify(""), "");
    }

    #[test]
    fn test_slugify_unicode() {
        // Unicode alphanumeric chars are preserved (is_alphanumeric includes Unicode letters)
        assert_eq!(slugify("Projet été"), "projet-été");
    }

    // ========================================================================
    // Argument extraction tests (verify parsing logic)
    // ========================================================================

    #[test]
    fn test_workspace_args_extraction() {
        let args = json!({
            "name": "Test Workspace",
            "slug": "test-workspace",
            "description": "A test workspace",
            "metadata": {"key": "value"}
        });

        // Test name extraction
        let name = args.get("name").and_then(|v| v.as_str());
        assert_eq!(name, Some("Test Workspace"));

        // Test slug extraction
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| slugify("Test Workspace"));
        assert_eq!(slug, "test-workspace");

        // Test description extraction
        let description = args.get("description").and_then(|v| v.as_str());
        assert_eq!(description, Some("A test workspace"));

        // Test metadata extraction
        let metadata = args
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        assert_eq!(metadata, json!({"key": "value"}));
    }

    #[test]
    fn test_workspace_args_defaults() {
        let args = json!({
            "name": "My Workspace"
        });

        // Slug defaults to slugified name
        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| slugify("My Workspace"));
        assert_eq!(slug, "my-workspace");

        // Description defaults to None
        let description = args.get("description").and_then(|v| v.as_str());
        assert!(description.is_none());

        // Metadata defaults to empty object
        let metadata = args
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        assert_eq!(metadata, json!({}));
    }

    #[test]
    fn test_pagination_args_extraction() {
        let args = json!({
            "limit": 25,
            "offset": 10,
            "search": "test query"
        });

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let search = args.get("search").and_then(|v| v.as_str());

        assert_eq!(limit, 25);
        assert_eq!(offset, 10);
        assert_eq!(search, Some("test query"));
    }

    #[test]
    fn test_pagination_args_defaults() {
        let args = json!({});

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let search = args.get("search").and_then(|v| v.as_str());

        assert_eq!(limit, 50);
        assert_eq!(offset, 0);
        assert!(search.is_none());
    }

    #[test]
    fn test_milestone_args_extraction() {
        let args = json!({
            "slug": "my-workspace",
            "title": "Q1 Release",
            "description": "First quarter release",
            "target_date": "2024-03-31T00:00:00Z",
            "tags": ["release", "q1"]
        });

        let slug = args.get("slug").and_then(|v| v.as_str());
        assert_eq!(slug, Some("my-workspace"));

        let title = args.get("title").and_then(|v| v.as_str());
        assert_eq!(title, Some("Q1 Release"));

        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        assert!(target_date.is_some());

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(tags, vec!["release", "q1"]);
    }

    #[test]
    fn test_milestone_status_parsing() {
        // Test status parsing logic
        let parse_status = |s: &str| -> Result<MilestoneStatus, _> {
            serde_json::from_str(&format!("\"{}\"", s.to_lowercase()))
        };

        assert_eq!(parse_status("open").unwrap(), MilestoneStatus::Open);
        assert_eq!(parse_status("planned").unwrap(), MilestoneStatus::Planned);
        assert_eq!(
            parse_status("in_progress").unwrap(),
            MilestoneStatus::InProgress
        );
        assert_eq!(
            parse_status("completed").unwrap(),
            MilestoneStatus::Completed
        );
        assert_eq!(parse_status("closed").unwrap(), MilestoneStatus::Closed);
        assert!(parse_status("invalid").is_err());
    }

    #[test]
    fn test_resource_args_extraction() {
        let args = json!({
            "slug": "my-workspace",
            "name": "User API",
            "resource_type": "api_contract",
            "file_path": "specs/openapi/users.yaml",
            "url": "https://api.example.com/docs",
            "format": "openapi",
            "version": "1.0.0",
            "description": "User management API",
            "metadata": {"owner": "team-a"}
        });

        let name = args.get("name").and_then(|v| v.as_str());
        assert_eq!(name, Some("User API"));

        let resource_type_str = args.get("resource_type").and_then(|v| v.as_str());
        assert_eq!(resource_type_str, Some("api_contract"));

        let resource_type: Result<ResourceType, _> = "api_contract".parse();
        assert!(resource_type.is_ok());
        assert_eq!(resource_type.unwrap(), ResourceType::ApiContract);

        let file_path = args.get("file_path").and_then(|v| v.as_str());
        assert_eq!(file_path, Some("specs/openapi/users.yaml"));

        let url = args.get("url").and_then(|v| v.as_str());
        assert_eq!(url, Some("https://api.example.com/docs"));
    }

    #[test]
    fn test_component_args_extraction() {
        let args = json!({
            "slug": "my-workspace",
            "name": "API Gateway",
            "component_type": "gateway",
            "description": "Main API gateway",
            "runtime": "kubernetes",
            "config": {"replicas": 3, "port": 8080},
            "tags": ["infrastructure", "gateway"]
        });

        let name = args.get("name").and_then(|v| v.as_str());
        assert_eq!(name, Some("API Gateway"));

        let component_type_str = args.get("component_type").and_then(|v| v.as_str());
        assert_eq!(component_type_str, Some("gateway"));

        let component_type: Result<ComponentType, _> = "gateway".parse();
        assert!(component_type.is_ok());
        assert_eq!(component_type.unwrap(), ComponentType::Gateway);

        let runtime = args.get("runtime").and_then(|v| v.as_str());
        assert_eq!(runtime, Some("kubernetes"));

        let config = args
            .get("config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        assert_eq!(config["replicas"], 3);
        assert_eq!(config["port"], 8080);
    }

    #[test]
    fn test_link_type_validation() {
        let validate_link_type =
            |link_type: &str| matches!(link_type.to_lowercase().as_str(), "implements" | "uses");

        assert!(validate_link_type("implements"));
        assert!(validate_link_type("IMPLEMENTS"));
        assert!(validate_link_type("uses"));
        assert!(validate_link_type("USES"));
        assert!(!validate_link_type("invalid"));
        assert!(!validate_link_type(""));
    }

    #[test]
    fn test_dependency_args_extraction() {
        let args = json!({
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "depends_on_id": "660e8400-e29b-41d4-a716-446655440001",
            "protocol": "http",
            "required": true
        });

        let id = parse_uuid(&args, "id");
        assert!(id.is_ok());

        let depends_on_id = parse_uuid(&args, "depends_on_id");
        assert!(depends_on_id.is_ok());

        let protocol = args
            .get("protocol")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(protocol, Some("http".to_string()));

        let required = args
            .get("required")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        assert!(required);
    }

    #[test]
    fn test_dependency_args_defaults() {
        let args = json!({
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "depends_on_id": "660e8400-e29b-41d4-a716-446655440001"
        });

        let protocol = args
            .get("protocol")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert!(protocol.is_none());

        let required = args
            .get("required")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        assert!(required); // Defaults to true
    }

    // ========================================================================
    // Response structure tests
    // ========================================================================

    #[test]
    fn test_paginated_response_structure() {
        // Simulate a paginated response
        let items = vec![json!({"id": "1", "name": "item1"})];
        let total = 10;
        let limit = 5;
        let offset = 0;

        let response = json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        });

        assert!(response.get("items").is_some());
        assert_eq!(response["total"], 10);
        assert_eq!(response["limit"], 5);
        assert_eq!(response["offset"], 0);
    }

    #[test]
    fn test_workspace_overview_response_structure() {
        // Simulate workspace overview response structure
        let total_tasks = 10u32;
        let completed_tasks = 5u32;

        let response = json!({
            "workspace": {"id": "123", "name": "Test"},
            "projects": [],
            "milestones": [],
            "resources": [],
            "components": [],
            "progress": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "percentage": if total_tasks > 0 {
                    (completed_tasks as f64 / total_tasks as f64 * 100.0).round()
                } else {
                    0.0
                }
            }
        });

        assert!(response.get("workspace").is_some());
        assert!(response.get("projects").is_some());
        assert!(response.get("milestones").is_some());
        assert!(response.get("resources").is_some());
        assert!(response.get("components").is_some());
        assert!(response.get("progress").is_some());
        assert_eq!(response["progress"]["percentage"], 50.0);
    }

    #[test]
    fn test_milestone_progress_response_structure() {
        let total = 10u32;
        let completed = 6u32;
        let in_progress = 2u32;
        let pending = 2u32;

        let percentage = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round()
        } else {
            0.0
        };

        let response = json!({
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "percentage": percentage
        });

        assert_eq!(response["total"], 10);
        assert_eq!(response["completed"], 6);
        assert_eq!(response["in_progress"], 2);
        assert_eq!(response["pending"], 2);
        assert_eq!(response["percentage"], 60.0);
    }

    #[test]
    fn test_milestone_progress_empty() {
        let total = 0u32;
        let completed = 0u32;

        let percentage = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round()
        } else {
            0.0
        };

        assert_eq!(percentage, 0.0);
    }

    #[test]
    fn test_boolean_response_structures() {
        // Test various boolean response structures used by handlers
        assert_eq!(json!({"deleted": true})["deleted"], true);
        assert_eq!(json!({"added": true})["added"], true);
        assert_eq!(json!({"removed": true})["removed"], true);
        assert_eq!(json!({"linked": true})["linked"], true);
        assert_eq!(json!({"mapped": true})["mapped"], true);
        assert_eq!(json!({"updated": true})["updated"], true);
    }

    #[test]
    fn test_link_response_with_type() {
        let link_type = "implements";
        let response = json!({"linked": true, "link_type": link_type});

        assert_eq!(response["linked"], true);
        assert_eq!(response["link_type"], "implements");
    }

    // ========================================================================
    // Filter logic tests
    // ========================================================================

    #[test]
    fn test_search_filter_logic() {
        let workspaces = vec![
            json!({"name": "Test Project", "description": "A test workspace"}),
            json!({"name": "Production", "description": "Production environment"}),
            json!({"name": "Development", "description": "Dev workspace"}),
        ];

        let search_term = "test";
        let search_lower = search_term.to_lowercase();

        let filtered: Vec<_> = workspaces
            .into_iter()
            .filter(|w| {
                let name = w["name"].as_str().unwrap_or("").to_lowercase();
                let desc = w["description"].as_str().unwrap_or("").to_lowercase();
                name.contains(&search_lower) || desc.contains(&search_lower)
            })
            .collect();

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["name"], "Test Project");
    }

    #[test]
    fn test_search_filter_description() {
        let workspaces = vec![
            json!({"name": "Project A", "description": "Contains test data"}),
            json!({"name": "Project B", "description": "Production only"}),
        ];

        let search_term = "test";
        let search_lower = search_term.to_lowercase();

        let filtered: Vec<_> = workspaces
            .into_iter()
            .filter(|w| {
                let name = w["name"].as_str().unwrap_or("").to_lowercase();
                let desc = w["description"].as_str().unwrap_or("").to_lowercase();
                name.contains(&search_lower) || desc.contains(&search_lower)
            })
            .collect();

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["name"], "Project A");
    }

    #[test]
    fn test_type_filter_logic() {
        // Simulating resource type filtering
        let resources = vec![
            (ResourceType::ApiContract, "User API"),
            (ResourceType::Protobuf, "Events Proto"),
            (ResourceType::ApiContract, "Order API"),
        ];

        let filter_type = ResourceType::ApiContract;
        let filtered: Vec<_> = resources
            .into_iter()
            .filter(|(rt, _)| *rt == filter_type)
            .collect();

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].1, "User API");
        assert_eq!(filtered[1].1, "Order API");
    }

    #[test]
    fn test_pagination_logic() {
        let items: Vec<i32> = (1..=20).collect();
        let limit = 5;
        let offset = 10;

        let paginated: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

        assert_eq!(paginated.len(), 5);
        assert_eq!(paginated, vec![11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_pagination_beyond_bounds() {
        let items: Vec<i32> = (1..=10).collect();
        let limit = 5;
        let offset = 8;

        let paginated: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

        assert_eq!(paginated.len(), 2);
        assert_eq!(paginated, vec![9, 10]);
    }

    #[test]
    fn test_pagination_empty_offset() {
        let items: Vec<i32> = (1..=10).collect();
        let limit = 5;
        let offset = 20;

        let paginated: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

        assert!(paginated.is_empty());
    }

    // ========================================================================
    // Date parsing tests
    // ========================================================================

    #[test]
    fn test_date_parsing_rfc3339() {
        let date_str = "2024-03-31T00:00:00Z";
        let parsed = chrono::DateTime::parse_from_rfc3339(date_str);
        assert!(parsed.is_ok());

        let utc = parsed.unwrap().with_timezone(&chrono::Utc);
        assert_eq!(utc.year(), 2024);
        assert_eq!(utc.month(), 3);
        assert_eq!(utc.day(), 31);
    }

    #[test]
    fn test_date_parsing_with_offset() {
        let date_str = "2024-03-31T12:00:00+02:00";
        let parsed = chrono::DateTime::parse_from_rfc3339(date_str);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_date_parsing_invalid() {
        let date_str = "2024-03-31"; // Not RFC3339
        let parsed = chrono::DateTime::parse_from_rfc3339(date_str);
        assert!(parsed.is_err());
    }

    #[test]
    fn test_date_parsing_optional() {
        let args = json!({});
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        assert!(target_date.is_none());
    }

    // ========================================================================
    // Tags parsing tests
    // ========================================================================

    #[test]
    fn test_tags_parsing() {
        let args = json!({
            "tags": ["tag1", "tag2", "tag3"]
        });

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        assert_eq!(tags, vec!["tag1", "tag2", "tag3"]);
    }

    #[test]
    fn test_tags_parsing_empty() {
        let args = json!({});

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        assert!(tags.is_empty());
    }

    #[test]
    fn test_tags_parsing_mixed_types() {
        let args = json!({
            "tags": ["valid", 123, "another", null]
        });

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Only strings are extracted
        assert_eq!(tags, vec!["valid", "another"]);
    }

    // ========================================================================
    // New CRUD handler arg extraction tests
    // ========================================================================

    #[test]
    fn test_update_project_args_extraction() {
        let args = json!({
            "slug": "my-project",
            "name": "New Name",
            "description": "New desc",
            "root_path": "/new/path"
        });

        let slug = args.get("slug").and_then(|v| v.as_str());
        assert_eq!(slug, Some("my-project"));

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(name, Some("New Name".to_string()));

        let description = args
            .get("description")
            .map(|v| v.as_str().map(|s| s.to_string()));
        assert_eq!(description, Some(Some("New desc".to_string())));

        let root_path = args
            .get("root_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(root_path, Some("/new/path".to_string()));
    }

    #[test]
    fn test_update_project_args_partial() {
        // Only slug, all others missing
        let args = json!({"slug": "my-project"});

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(name, None);

        let description = args
            .get("description")
            .map(|v| v.as_str().map(|s| s.to_string()));
        assert_eq!(description, None);

        let root_path = args
            .get("root_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(root_path, None);
    }

    #[test]
    fn test_update_project_args_null_description() {
        // Explicit null description (to clear it)
        let args = json!({
            "slug": "my-project",
            "description": null
        });

        let description = args
            .get("description")
            .map(|v| v.as_str().map(|s| s.to_string()));
        // description key exists but value is null -> Some(None)
        assert_eq!(description, Some(None));
    }

    #[test]
    fn test_update_constraint_args_with_enum() {
        let args = json!({
            "constraint_id": "550e8400-e29b-41d4-a716-446655440000",
            "description": "Must be fast",
            "constraint_type": "performance",
            "enforced_by": "benchmark"
        });

        let constraint_id = parse_uuid(&args, "constraint_id");
        assert!(constraint_id.is_ok());

        let constraint_type: Option<crate::neo4j::models::ConstraintType> = args
            .get("constraint_type")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());
        assert!(constraint_type.is_some());

        let enforced_by = args
            .get("enforced_by")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(enforced_by, Some("benchmark".to_string()));
    }

    #[test]
    fn test_update_constraint_args_invalid_enum() {
        let args = json!({
            "constraint_id": "550e8400-e29b-41d4-a716-446655440000",
            "constraint_type": "invalid_type"
        });

        let constraint_type: Option<crate::neo4j::models::ConstraintType> = args
            .get("constraint_type")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());
        // Invalid enum variant -> None (graceful)
        assert!(constraint_type.is_none());
    }

    #[test]
    fn test_update_decision_args_extraction() {
        let args = json!({
            "decision_id": "550e8400-e29b-41d4-a716-446655440000",
            "description": "Updated desc",
            "rationale": "New rationale",
            "chosen_option": "Option B"
        });

        let decision_id = parse_uuid(&args, "decision_id");
        assert!(decision_id.is_ok());

        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(description, Some("Updated desc".to_string()));

        let rationale = args
            .get("rationale")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(rationale, Some("New rationale".to_string()));

        let chosen_option = args
            .get("chosen_option")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(chosen_option, Some("Option B".to_string()));
    }

    #[test]
    fn test_update_resource_args_extraction() {
        let args = json!({
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "API v2",
            "file_path": "/api/v2.yaml",
            "url": "https://example.com/api",
            "version": "2.0.0",
            "description": "Updated API contract"
        });

        let id = parse_uuid(&args, "id");
        assert!(id.is_ok());

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(name, Some("API v2".to_string()));

        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(url, Some("https://example.com/api".to_string()));

        let version = args
            .get("version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert_eq!(version, Some("2.0.0".to_string()));
    }

    #[test]
    fn test_update_component_args_with_config_and_tags() {
        let args = json!({
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "Auth Service",
            "runtime": "rust",
            "config": {"port": 8080, "workers": 4},
            "tags": ["auth", "core", "security"]
        });

        let id = parse_uuid(&args, "id");
        assert!(id.is_ok());

        let config = args.get("config").cloned();
        assert!(config.is_some());
        assert_eq!(config.as_ref().unwrap()["port"], 8080);

        let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });
        assert_eq!(
            tags,
            Some(vec![
                "auth".to_string(),
                "core".to_string(),
                "security".to_string()
            ])
        );
    }

    #[test]
    fn test_update_component_args_partial() {
        let args = json!({
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "New Name"
        });

        let config = args.get("config").cloned();
        assert!(config.is_none());

        let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });
        assert!(tags.is_none());

        let runtime = args
            .get("runtime")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        assert!(runtime.is_none());
    }

    #[test]
    fn test_delete_handlers_uuid_parsing() {
        // All delete handlers just need a valid UUID
        let args = json!({"task_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "task_id").is_ok());

        let args = json!({"step_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "step_id").is_ok());

        let args = json!({"release_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "release_id").is_ok());

        let args = json!({"milestone_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "milestone_id").is_ok());

        let args = json!({"decision_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "decision_id").is_ok());

        let args = json!({"constraint_id": "550e8400-e29b-41d4-a716-446655440000"});
        assert!(parse_uuid(&args, "constraint_id").is_ok());
    }

    #[test]
    fn test_delete_handlers_missing_uuid() {
        assert!(parse_uuid(&json!({}), "task_id").is_err());
        assert!(parse_uuid(&json!({}), "step_id").is_err());
        assert!(parse_uuid(&json!({}), "release_id").is_err());
        assert!(parse_uuid(&json!({}), "milestone_id").is_err());
        assert!(parse_uuid(&json!({}), "decision_id").is_err());
        assert!(parse_uuid(&json!({}), "constraint_id").is_err());
    }

    // ========================================================================
    // list_all_workspace_milestones argument extraction tests
    // ========================================================================

    #[test]
    fn test_list_all_workspace_milestones_args_full() {
        let args = json!({
            "workspace_id": "b37351e3-6c90-4a53-bc4f-8cbd024cecb7",
            "status": "open",
            "limit": 25,
            "offset": 10
        });

        let workspace_id = args
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .unwrap();
        assert!(workspace_id.is_some());
        assert_eq!(
            workspace_id.unwrap().to_string(),
            "b37351e3-6c90-4a53-bc4f-8cbd024cecb7"
        );

        let status_filter = args.get("status").and_then(|v| v.as_str());
        assert_eq!(status_filter, Some("open"));

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        assert_eq!(limit, 25);

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        assert_eq!(offset, 10);
    }

    #[test]
    fn test_list_all_workspace_milestones_args_defaults() {
        let args = json!({});

        let workspace_id = args
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .unwrap();
        assert!(workspace_id.is_none());

        let status_filter = args.get("status").and_then(|v| v.as_str());
        assert!(status_filter.is_none());

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        assert_eq!(limit, 50);

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_list_all_workspace_milestones_args_invalid_uuid() {
        let args = json!({
            "workspace_id": "not-a-uuid"
        });

        let result = args
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose();
        assert!(result.is_err());
    }

    #[test]
    fn test_list_all_workspace_milestones_args_status_only() {
        let args = json!({
            "status": "closed"
        });

        let workspace_id = args
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .unwrap();
        assert!(workspace_id.is_none());

        let status_filter = args.get("status").and_then(|v| v.as_str());
        assert_eq!(status_filter, Some("closed"));
    }

    #[test]
    fn test_list_all_workspace_milestones_response_structure() {
        let items = vec![json!({
            "id": "test-id",
            "title": "Milestone 1",
            "workspace_id": "ws-1",
            "workspace_name": "Workspace One",
            "workspace_slug": "workspace-one"
        })];
        let total = 1usize;
        let limit = 50usize;
        let offset = 0usize;

        let response = json!({
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        });

        assert!(response["items"].is_array());
        assert_eq!(response["items"].as_array().unwrap().len(), 1);
        assert_eq!(response["total"], 1);
        assert_eq!(response["limit"], 50);
        assert_eq!(response["offset"], 0);
        assert_eq!(response["items"][0]["workspace_slug"], "workspace-one");
    }

    // ========================================================================
    // list_plans project_id argument extraction tests
    // ========================================================================

    #[test]
    fn test_list_plans_args_with_project_id() {
        let args = json!({
            "project_id": "e83b0663-9600-450d-9f63-234e857394df",
            "status": "draft,in_progress",
            "limit": 10,
            "offset": 0
        });

        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .unwrap();
        assert!(project_id.is_some());
        assert_eq!(
            project_id.unwrap().to_string(),
            "e83b0663-9600-450d-9f63-234e857394df"
        );

        let status = args.get("status").and_then(|v| v.as_str()).map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            status,
            Some(vec!["draft".to_string(), "in_progress".to_string()])
        );
    }

    #[test]
    fn test_list_plans_args_without_project_id() {
        let args = json!({
            "status": "completed"
        });

        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose()
            .unwrap();
        assert!(project_id.is_none());
    }

    #[test]
    fn test_list_plans_args_invalid_project_id() {
        let args = json!({
            "project_id": "invalid-uuid"
        });

        let result = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(Uuid::parse_str)
            .transpose();
        assert!(result.is_err());
    }

    // ========================================================================
    // list_workspace_milestones (per-workspace) argument extraction tests
    // ========================================================================

    #[test]
    fn test_list_workspace_milestones_args_full() {
        let args = json!({
            "slug": "my-workspace",
            "status": "open",
            "limit": 20,
            "offset": 5
        });

        let slug = args.get("slug").and_then(|v| v.as_str());
        assert_eq!(slug, Some("my-workspace"));

        let status_filter = args.get("status").and_then(|v| v.as_str());
        assert_eq!(status_filter, Some("open"));

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        assert_eq!(limit, 20);

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        assert_eq!(offset, 5);
    }

    #[test]
    fn test_list_workspace_milestones_args_slug_only() {
        let args = json!({
            "slug": "prod-workspace"
        });

        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"));
        assert!(slug.is_ok());
        assert_eq!(slug.unwrap(), "prod-workspace");

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        assert_eq!(limit, 50);
    }

    #[test]
    fn test_list_workspace_milestones_args_missing_slug() {
        let args = json!({});

        let slug = args
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("slug is required"));
        assert!(slug.is_err());
        assert!(slug.unwrap_err().to_string().contains("slug is required"));
    }

    // ========================================================================
    // Async integration tests (mock backends)
    // ========================================================================

    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::mock_app_state;

    async fn create_handler() -> ToolHandler {
        let state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(state).await.unwrap());
        ToolHandler::new(orchestrator)
    }

    // -- Basic tool routing -------------------------------------------------

    #[tokio::test]
    async fn test_unknown_tool() {
        let handler = create_handler().await;
        let result = handler.handle("nonexistent_tool", None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown tool: nonexistent_tool"));
    }

    #[tokio::test]
    async fn test_list_plans_empty() {
        let handler = create_handler().await;
        let result = handler.handle("list_plans", Some(json!({}))).await.unwrap();
        assert!(result.is_object());
        let items = result.get("items").unwrap().as_array().unwrap();
        assert!(items.is_empty());
        assert_eq!(result.get("total").unwrap().as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_create_and_get_plan() {
        let handler = create_handler().await;

        // Create a plan
        let create_result = handler
            .handle(
                "create_plan",
                Some(json!({
                    "title": "Integration Test Plan",
                    "description": "Testing plan creation via MCP handler"
                })),
            )
            .await
            .unwrap();

        assert!(create_result.is_object());
        let plan_id = create_result.get("id").unwrap().as_str().unwrap();
        assert!(!plan_id.is_empty());
        assert_eq!(
            create_result.get("title").unwrap().as_str().unwrap(),
            "Integration Test Plan"
        );

        // Get the plan back
        let get_result = handler
            .handle("get_plan", Some(json!({"plan_id": plan_id})))
            .await
            .unwrap();

        assert!(get_result.is_object());
        let plan = get_result.get("plan").unwrap();
        assert_eq!(
            plan.get("title").unwrap().as_str().unwrap(),
            "Integration Test Plan"
        );
    }

    // -- Project tools ------------------------------------------------------

    #[tokio::test]
    async fn test_create_project() {
        let handler = create_handler().await;

        let result = handler
            .handle(
                "create_project",
                Some(json!({
                    "name": "My Test Project",
                    "slug": "my-test-project",
                    "root_path": "/tmp/my-test-project"
                })),
            )
            .await
            .unwrap();

        assert!(result.is_object());
        assert_eq!(
            result.get("name").unwrap().as_str().unwrap(),
            "My Test Project"
        );
        assert_eq!(
            result.get("slug").unwrap().as_str().unwrap(),
            "my-test-project"
        );
        assert_eq!(
            result.get("root_path").unwrap().as_str().unwrap(),
            "/tmp/my-test-project"
        );
    }

    #[tokio::test]
    async fn test_get_project_not_found() {
        let handler = create_handler().await;

        let result = handler
            .handle("get_project", Some(json!({"slug": "nonexistent"})))
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Project not found"));
    }

    // -- Task tools ---------------------------------------------------------

    #[tokio::test]
    async fn test_create_task() {
        let handler = create_handler().await;

        // Create plan first
        let plan = handler
            .handle(
                "create_plan",
                Some(json!({
                    "title": "Plan for Tasks",
                    "description": "Plan to hold tasks"
                })),
            )
            .await
            .unwrap();
        let plan_id = plan.get("id").unwrap().as_str().unwrap();

        // Create task
        let task = handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": plan_id,
                    "title": "First Task",
                    "description": "Implement feature X"
                })),
            )
            .await
            .unwrap();

        assert!(task.is_object());
        assert_eq!(task.get("title").unwrap().as_str().unwrap(), "First Task");
        assert_eq!(
            task.get("description").unwrap().as_str().unwrap(),
            "Implement feature X"
        );
        assert_eq!(task.get("status").unwrap().as_str().unwrap(), "pending");
    }

    #[tokio::test]
    async fn test_get_next_task() {
        let handler = create_handler().await;

        // Create plan
        let plan = handler
            .handle(
                "create_plan",
                Some(json!({
                    "title": "Plan with Tasks",
                    "description": "Plan for next-task test"
                })),
            )
            .await
            .unwrap();
        let plan_id = plan.get("id").unwrap().as_str().unwrap();

        // Create two tasks
        handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": plan_id,
                    "title": "Task A",
                    "description": "First task"
                })),
            )
            .await
            .unwrap();

        handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": plan_id,
                    "title": "Task B",
                    "description": "Second task"
                })),
            )
            .await
            .unwrap();

        // Get next task — should return one of the pending tasks
        let next = handler
            .handle("get_next_task", Some(json!({"plan_id": plan_id})))
            .await
            .unwrap();

        // get_next_task returns a task or null
        assert!(next.is_object() || next.is_null());
        if next.is_object() {
            assert_eq!(next.get("status").unwrap().as_str().unwrap(), "pending");
        }
    }

    // -- Step tools ---------------------------------------------------------

    #[tokio::test]
    async fn test_create_and_list_steps() {
        let handler = create_handler().await;

        // Create plan + task
        let plan = handler
            .handle(
                "create_plan",
                Some(json!({
                    "title": "Step Test Plan",
                    "description": "Plan for step tests"
                })),
            )
            .await
            .unwrap();
        let plan_id = plan.get("id").unwrap().as_str().unwrap();

        let task = handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": plan_id,
                    "title": "Task with Steps",
                    "description": "A task that has steps"
                })),
            )
            .await
            .unwrap();
        let task_id = task.get("id").unwrap().as_str().unwrap();

        // Create two steps
        let step1 = handler
            .handle(
                "create_step",
                Some(json!({
                    "task_id": task_id,
                    "description": "Step one: setup"
                })),
            )
            .await
            .unwrap();
        assert_eq!(
            step1.get("description").unwrap().as_str().unwrap(),
            "Step one: setup"
        );
        assert_eq!(step1.get("order").unwrap().as_u64().unwrap(), 0);

        let step2 = handler
            .handle(
                "create_step",
                Some(json!({
                    "task_id": task_id,
                    "description": "Step two: implement"
                })),
            )
            .await
            .unwrap();
        assert_eq!(step2.get("order").unwrap().as_u64().unwrap(), 1);

        // List steps
        let steps = handler
            .handle("list_steps", Some(json!({"task_id": task_id})))
            .await
            .unwrap();

        let steps_arr = steps.as_array().unwrap();
        assert_eq!(steps_arr.len(), 2);
    }

    // -- Error handling -----------------------------------------------------

    #[tokio::test]
    async fn test_missing_required_arg() {
        let handler = create_handler().await;

        // create_plan requires "title" and "description"
        let result = handler
            .handle("create_plan", Some(json!({"title": "No desc"})))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("description is required"));

        // create_task requires "plan_id" and "description"
        let result = handler
            .handle(
                "create_task",
                Some(json!({"description": "Missing plan_id"})),
            )
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("plan_id is required"));

        // create_project requires "name" and "root_path"
        let result = handler
            .handle("create_project", Some(json!({"name": "test"})))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("root_path is required"));
    }

    // -- Chat tools ---------------------------------------------------------

    #[tokio::test]
    async fn test_list_chat_sessions_empty() {
        let handler = create_handler().await;
        let result = handler
            .handle("list_chat_sessions", Some(json!({})))
            .await
            .unwrap();
        assert!(result.is_object());
        let items = result.get("items").unwrap().as_array().unwrap();
        assert!(items.is_empty());
        assert_eq!(result.get("total").unwrap().as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_get_chat_session_not_found() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "get_chat_session",
                Some(json!({"session_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_delete_chat_session_not_found() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "delete_chat_session",
                Some(json!({"session_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_chat_send_message_no_chat_manager() {
        let handler = create_handler().await;
        // chat_manager is None by default, so chat_send_message should error
        let result = handler
            .handle(
                "chat_send_message",
                Some(json!({
                    "message": "Hello",
                    "cwd": "/tmp"
                })),
            )
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Chat manager not initialized"));
    }

    #[tokio::test]
    async fn test_chat_send_message_missing_args() {
        let handler = create_handler().await;
        // Missing "message" field
        let result = handler
            .handle("chat_send_message", Some(json!({"cwd": "/tmp"})))
            .await;
        assert!(result.is_err());
        // Should fail because chat_manager is None (checked first)
        assert!(result.unwrap_err().to_string().contains("not initialized"));
    }

    #[tokio::test]
    async fn test_list_chat_sessions_with_project_filter() {
        let handler = create_handler().await;

        // Create a chat session via neo4j directly
        use crate::test_helpers::test_chat_session;
        let session = test_chat_session(Some("my-project"));
        handler.neo4j().create_chat_session(&session).await.unwrap();

        // List with matching filter
        let result = handler
            .handle(
                "list_chat_sessions",
                Some(json!({"project_slug": "my-project"})),
            )
            .await
            .unwrap();
        let items = result.get("items").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 1);

        // List with non-matching filter
        let result = handler
            .handle(
                "list_chat_sessions",
                Some(json!({"project_slug": "other-project"})),
            )
            .await
            .unwrap();
        let items = result.get("items").unwrap().as_array().unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_chat_session_crud_via_mcp() {
        let handler = create_handler().await;

        // Create a session via neo4j
        use crate::test_helpers::test_chat_session;
        let session = test_chat_session(Some("test-proj"));
        let session_id = session.id.to_string();
        handler.neo4j().create_chat_session(&session).await.unwrap();

        // Get it
        let result = handler
            .handle("get_chat_session", Some(json!({"session_id": session_id})))
            .await
            .unwrap();
        assert_eq!(
            result.get("project_slug").unwrap().as_str().unwrap(),
            "test-proj"
        );

        // Delete it
        let result = handler
            .handle(
                "delete_chat_session",
                Some(json!({"session_id": session_id})),
            )
            .await
            .unwrap();
        assert!(result.get("deleted").unwrap().as_bool().unwrap());

        // Verify it's gone
        let result = handler
            .handle("get_chat_session", Some(json!({"session_id": session_id})))
            .await;
        assert!(result.is_err());
    }

    // -- chat_send_message arg parsing (workspace_slug, add_dirs) ------------

    #[tokio::test]
    async fn test_chat_send_message_parses_workspace_slug() {
        // chat_send_message requires a ChatManager — without one it fails,
        // but we can verify the arg parsing logic inline.
        let args = json!({
            "message": "Hello",
            "cwd": "/tmp",
            "workspace_slug": "my-ws",
            "add_dirs": ["/dir/a", "/dir/b"]
        });

        let workspace_slug = args
            .get("workspace_slug")
            .and_then(|v| v.as_str())
            .map(String::from);
        assert_eq!(workspace_slug.as_deref(), Some("my-ws"));

        let add_dirs: Option<Vec<String>> =
            args.get("add_dirs").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        assert_eq!(
            add_dirs.as_deref(),
            Some(vec!["/dir/a".to_string(), "/dir/b".to_string()].as_slice())
        );
    }

    #[tokio::test]
    async fn test_chat_send_message_workspace_slug_defaults() {
        let args = json!({
            "message": "Hello",
            "cwd": "/tmp"
        });

        let workspace_slug = args
            .get("workspace_slug")
            .and_then(|v| v.as_str())
            .map(String::from);
        assert!(workspace_slug.is_none());

        let add_dirs: Option<Vec<String>> =
            args.get("add_dirs").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        assert!(add_dirs.is_none());
    }

    #[tokio::test]
    async fn test_chat_send_message_no_chat_manager_with_workspace() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "chat_send_message",
                Some(json!({
                    "message": "Hello",
                    "cwd": "/tmp",
                    "workspace_slug": "my-ws",
                    "add_dirs": ["/extra/dir"]
                })),
            )
            .await;
        // No chat_manager → error, but parsing succeeded
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Chat manager not initialized"));
    }

    #[tokio::test]
    async fn test_chat_session_crud_with_workspace_fields() {
        let handler = create_handler().await;

        use crate::test_helpers::test_chat_session;
        let mut session = test_chat_session(Some("test-proj"));
        session.workspace_slug = Some("test-ws".into());
        session.add_dirs = Some(vec!["/extra/a".into(), "/extra/b".into()]);
        let session_id = session.id.to_string();
        handler.neo4j().create_chat_session(&session).await.unwrap();

        // Get it and verify workspace fields are present
        let result = handler
            .handle("get_chat_session", Some(json!({"session_id": session_id})))
            .await
            .unwrap();
        assert_eq!(
            result.get("workspace_slug").unwrap().as_str().unwrap(),
            "test-ws"
        );
        let dirs = result.get("add_dirs").unwrap().as_array().unwrap();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0].as_str().unwrap(), "/extra/a");
        assert_eq!(dirs[1].as_str().unwrap(), "/extra/b");
    }

    #[tokio::test]
    async fn test_list_chat_sessions_includes_workspace() {
        let handler = create_handler().await;

        use crate::test_helpers::test_chat_session;
        let mut session = test_chat_session(Some("proj"));
        session.workspace_slug = Some("ws-slug".into());
        handler.neo4j().create_chat_session(&session).await.unwrap();

        let result = handler
            .handle("list_chat_sessions", Some(json!({})))
            .await
            .unwrap();
        let items = result.get("items").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0].get("workspace_slug").unwrap().as_str().unwrap(),
            "ws-slug"
        );
    }

    // -- list_chat_messages --------------------------------------------------

    #[tokio::test]
    async fn test_list_chat_messages_no_chat_manager() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "list_chat_messages",
                Some(json!({"session_id": Uuid::new_v4().to_string()})),
            )
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Chat manager not initialized"));
    }

    #[tokio::test]
    async fn test_list_chat_messages_missing_session_id() {
        let handler = create_handler().await;
        let result = handler.handle("list_chat_messages", Some(json!({}))).await;
        assert!(result.is_err());
        // Even though chat_manager is None, session_id check comes after
        // so we get "Chat manager not initialized" first
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Chat manager not initialized"));
    }

    #[tokio::test]
    async fn test_list_chat_messages_args_extraction() {
        // Verify argument parsing logic for list_chat_messages
        let args = json!({
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "limit": 25,
            "offset": 10
        });

        let session_id = args.get("session_id").and_then(|v| v.as_str());
        assert_eq!(session_id, Some("550e8400-e29b-41d4-a716-446655440000"));

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        assert_eq!(limit, Some(25));

        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        assert_eq!(offset, Some(10));
    }

    #[tokio::test]
    async fn test_list_chat_messages_args_defaults() {
        let args = json!({
            "session_id": "550e8400-e29b-41d4-a716-446655440000"
        });

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        assert!(limit.is_none());

        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        assert!(offset.is_none());
    }

    // ========================================================================
    // Search scoping tests (project_slug, path_prefix, workspace)
    // ========================================================================

    #[tokio::test]
    async fn test_search_code_without_project_slug() {
        let handler = create_handler().await;
        let result = handler
            .handle("search_code", Some(json!({"query": "nonexistent_xyz"})))
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_search_code_with_project_slug() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "search_code",
                Some(json!({"query": "test", "project_slug": "my-project"})),
            )
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_search_code_with_path_prefix() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "search_code",
                Some(json!({"query": "test", "path_prefix": "src/mcp/"})),
            )
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_find_similar_code_with_filters() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "find_similar_code",
                Some(json!({
                    "code_snippet": "fn main()",
                    "project_slug": "my-project",
                    "language": "rust"
                })),
            )
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_find_similar_code_without_filters() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "find_similar_code",
                Some(json!({"code_snippet": "fn main()"})),
            )
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_search_decisions_without_project_slug() {
        let handler = create_handler().await;
        let result = handler
            .handle("search_decisions", Some(json!({"query": "architecture"})))
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_search_decisions_with_project_slug() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "search_decisions",
                Some(json!({"query": "architecture", "project_slug": "my-project"})),
            )
            .await
            .unwrap();
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn test_get_architecture_without_project_slug() {
        let handler = create_handler().await;
        let result = handler
            .handle("get_architecture", Some(json!({})))
            .await
            .unwrap();
        assert!(result.is_object());
        assert!(result.get("languages").is_some());
        assert!(result.get("key_files").is_some());
    }

    #[tokio::test]
    async fn test_get_architecture_with_nonexistent_project() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "get_architecture",
                Some(json!({"project_slug": "nonexistent"})),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_commit_without_sync() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "create_commit",
                Some(json!({
                    "sha": "abc1234",
                    "message": "test commit"
                })),
            )
            .await
            .unwrap();
        assert!(result.is_object());
        assert!(!result.get("sync_triggered").unwrap().as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_create_commit_with_files_but_no_project() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "create_commit",
                Some(json!({
                    "sha": "def5678",
                    "message": "test commit",
                    "files_changed": ["src/main.rs"]
                })),
            )
            .await
            .unwrap();
        // No project_id → no sync
        assert!(!result.get("sync_triggered").unwrap().as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_search_workspace_code_nonexistent_workspace() {
        let handler = create_handler().await;
        let result = handler
            .handle(
                "search_workspace_code",
                Some(json!({
                    "workspace_slug": "nonexistent",
                    "query": "test"
                })),
            )
            .await;
        assert!(result.is_err());
    }

    // ========================================================================
    // Call graph / references / impact — project scoping tests
    // ========================================================================

    use crate::meilisearch::mock::MockSearchStore;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{FileNode, FunctionNode, Visibility};
    use crate::test_helpers::{mock_app_state_with, test_project_named};

    /// Seed a file + its functions into the MockGraphStore (direct field access).
    async fn seed_mock_file(
        store: &MockGraphStore,
        project_id: uuid::Uuid,
        file_path: &str,
        fn_names: &[&str],
    ) {
        // Insert file
        let file = FileNode {
            path: file_path.to_string(),
            language: "rust".to_string(),
            hash: "h".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project_id),
        };
        store.upsert_file(&file).await.unwrap();

        // Register file in project_files
        store
            .project_files
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(file_path.to_string());

        // Insert functions
        for (i, name) in fn_names.iter().enumerate() {
            let func = FunctionNode {
                name: name.to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: file_path.to_string(),
                line_start: (i * 10) as u32,
                line_end: (i * 10 + 5) as u32,
                docstring: None,
            };
            store.upsert_function(&func).await.unwrap();
        }
    }

    /// Create a handler pre-seeded with two projects that have homonymous functions.
    ///
    /// Seeds the MockGraphStore *before* wrapping in Arc, avoiding the need
    /// for trait downcasting.
    async fn create_handler_with_call_data() -> ToolHandler {
        let graph = MockGraphStore::new();

        // Create two projects
        let project_a = test_project_named("alpha");
        let project_b = test_project_named("beta");
        graph.create_project(&project_a).await.unwrap();
        graph.create_project(&project_b).await.unwrap();

        // Seed files + functions
        seed_mock_file(
            &graph,
            project_a.id,
            "a/src/lib.rs",
            &["handler", "validate"],
        )
        .await;
        seed_mock_file(
            &graph,
            project_b.id,
            "b/src/lib.rs",
            &["handler", "validate"],
        )
        .await;

        // Create CALLS: handler -> validate in each project (unscoped for test setup)
        graph
            .create_call_relationship("a/src/lib.rs::handler", "validate", None)
            .await
            .unwrap();
        graph
            .create_call_relationship("b/src/lib.rs::handler", "validate", None)
            .await
            .unwrap();

        let state = mock_app_state_with(graph, MockSearchStore::new());
        let orchestrator = Arc::new(Orchestrator::new(state).await.unwrap());
        ToolHandler::new(orchestrator)
    }

    #[tokio::test]
    async fn test_get_call_graph_handler_unscoped() {
        let handler = create_handler_with_call_data().await;

        let result = handler
            .handle("get_call_graph", Some(json!({"function": "validate"})))
            .await
            .unwrap();

        let callers = result.get("callers").unwrap().as_array().unwrap();
        // Unscoped: both handler functions from both projects call validate
        assert_eq!(callers.len(), 2);
    }

    #[tokio::test]
    async fn test_get_call_graph_handler_scoped() {
        let handler = create_handler_with_call_data().await;

        let result = handler
            .handle(
                "get_call_graph",
                Some(json!({"function": "validate", "project_slug": "alpha"})),
            )
            .await
            .unwrap();

        let callers = result.get("callers").unwrap().as_array().unwrap();
        // Scoped to alpha: only 1 caller
        assert_eq!(callers.len(), 1);
        assert!(callers[0].as_str().unwrap().contains("a/src/lib.rs"));
    }

    #[tokio::test]
    async fn test_analyze_impact_handler_scoped() {
        let handler = create_handler_with_call_data().await;

        // Unscoped
        let result = handler
            .handle("analyze_impact", Some(json!({"target": "validate"})))
            .await
            .unwrap();
        let caller_count = result.get("caller_count").unwrap().as_i64().unwrap();
        assert_eq!(caller_count, 2); // Both projects

        // Scoped to beta
        let result = handler
            .handle(
                "analyze_impact",
                Some(json!({"target": "validate", "project_slug": "beta"})),
            )
            .await
            .unwrap();
        let caller_count = result.get("caller_count").unwrap().as_i64().unwrap();
        assert_eq!(caller_count, 1); // Only beta
    }

    #[tokio::test]
    async fn test_find_references_handler_scoped() {
        let handler = create_handler_with_call_data().await;

        // Unscoped
        let result = handler
            .handle("find_references", Some(json!({"symbol": "validate"})))
            .await
            .unwrap();
        let refs = result.get("references").unwrap().as_array().unwrap();
        assert_eq!(refs.len(), 2); // Both projects

        // Scoped to alpha
        let result = handler
            .handle(
                "find_references",
                Some(json!({"symbol": "validate", "project_slug": "alpha"})),
            )
            .await
            .unwrap();
        let refs = result.get("references").unwrap().as_array().unwrap();
        assert_eq!(refs.len(), 1);
    }

    // ========================================================================
    // Enriched analyze_impact handler tests
    // ========================================================================

    /// Create a handler with file imports for analyze_impact file mode tests
    async fn create_handler_with_import_data() -> ToolHandler {
        let graph = MockGraphStore::new();

        let project_a = test_project_named("alpha");
        graph.create_project(&project_a).await.unwrap();

        seed_mock_file(&graph, project_a.id, "a/src/lib.rs", &["main_fn"]).await;
        seed_mock_file(&graph, project_a.id, "a/src/handler.rs", &["handle"]).await;
        seed_mock_file(&graph, project_a.id, "a/src/utils.rs", &["util_fn"]).await;
        seed_mock_file(&graph, project_a.id, "a/tests/test_lib.rs", &["test_main"]).await;

        // handler.rs imports lib.rs, utils.rs imports lib.rs, test_lib.rs imports lib.rs
        graph
            .create_import_relationship("a/src/handler.rs", "a/src/lib.rs", "lib")
            .await
            .unwrap();
        graph
            .create_import_relationship("a/src/utils.rs", "a/src/lib.rs", "lib")
            .await
            .unwrap();
        graph
            .create_import_relationship("a/tests/test_lib.rs", "a/src/lib.rs", "lib")
            .await
            .unwrap();

        let state = mock_app_state_with(graph, MockSearchStore::new());
        let orchestrator = Arc::new(Orchestrator::new(state).await.unwrap());
        ToolHandler::new(orchestrator)
    }

    #[tokio::test]
    async fn test_analyze_impact_file_mode() {
        let handler = create_handler_with_import_data().await;

        let result = handler
            .handle(
                "analyze_impact",
                Some(json!({"target": "a/src/lib.rs", "project_slug": "alpha"})),
            )
            .await
            .unwrap();

        // target_type should be auto-detected as "file" (contains '/')
        assert_eq!(result.get("target_type").unwrap().as_str().unwrap(), "file");

        // Should have dependents
        let directly = result.get("directly_affected").unwrap().as_array().unwrap();
        let transitively = result
            .get("transitively_affected")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(!directly.is_empty());
        assert!(!transitively.is_empty());

        // caller_count = number of direct dependents
        let caller_count = result.get("caller_count").unwrap().as_i64().unwrap();
        assert_eq!(caller_count, directly.len() as i64);

        // test_files_affected should contain the test file
        let test_files = result
            .get("test_files_affected")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(test_files
            .iter()
            .any(|f| f.as_str().unwrap().contains("test")));

        // risk_level and suggestion should be present
        assert!(result.get("risk_level").is_some());
        assert!(result.get("suggestion").is_some());
    }

    #[tokio::test]
    async fn test_analyze_impact_symbol_mode() {
        let handler = create_handler_with_call_data().await;

        // "validate" has no '/' or '.' → auto-detected as symbol
        let result = handler
            .handle(
                "analyze_impact",
                Some(json!({"target": "validate", "project_slug": "alpha"})),
            )
            .await
            .unwrap();

        assert_eq!(
            result.get("target_type").unwrap().as_str().unwrap(),
            "symbol"
        );

        let directly = result.get("directly_affected").unwrap().as_array().unwrap();
        // In alpha, only "handler" calls "validate" → 1 caller file
        assert_eq!(directly.len(), 1);
        assert!(directly[0].as_str().unwrap().contains("a/src/lib.rs"));

        let caller_count = result.get("caller_count").unwrap().as_i64().unwrap();
        assert_eq!(caller_count, 1);
    }

    #[tokio::test]
    async fn test_analyze_impact_explicit_target_type() {
        let handler = create_handler_with_call_data().await;

        // Force target_type=symbol even though target has no special chars
        let result = handler
            .handle(
                "analyze_impact",
                Some(json!({"target": "validate", "target_type": "symbol"})),
            )
            .await
            .unwrap();

        assert_eq!(
            result.get("target_type").unwrap().as_str().unwrap(),
            "symbol"
        );
        // Unscoped: both projects
        let caller_count = result.get("caller_count").unwrap().as_i64().unwrap();
        assert_eq!(caller_count, 2);
    }

    #[tokio::test]
    async fn test_analyze_impact_backward_compat_fields() {
        let handler = create_handler_with_import_data().await;

        let result = handler
            .handle(
                "analyze_impact",
                Some(json!({"target": "a/src/lib.rs", "project_slug": "alpha"})),
            )
            .await
            .unwrap();

        // Backward compat: dependent_files = alias of transitively_affected
        let dep_files = result.get("dependent_files").unwrap().as_array().unwrap();
        let transitively = result
            .get("transitively_affected")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(dep_files, transitively);

        // Backward compat: impact_level = alias of risk_level
        let impact = result.get("impact_level").unwrap().as_str().unwrap();
        let risk = result.get("risk_level").unwrap().as_str().unwrap();
        assert_eq!(impact, risk);

        // Original fields still present
        assert!(result.get("target").is_some());
        assert!(result.get("caller_count").is_some());
    }
}
