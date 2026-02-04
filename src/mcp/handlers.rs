//! MCP Tool handlers
//!
//! Implements the actual logic for each MCP tool.

use crate::meilisearch::MeiliClient;
use crate::neo4j::models::*;
use crate::neo4j::Neo4jClient;
use crate::orchestrator::Orchestrator;
use crate::plan::models::*;
use anyhow::{anyhow, Result};
use neo4rs;
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

/// Handles MCP tool calls
pub struct ToolHandler {
    orchestrator: Arc<Orchestrator>,
}

impl ToolHandler {
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self { orchestrator }
    }

    fn neo4j(&self) -> &Neo4jClient {
        self.orchestrator.neo4j()
    }

    fn meili(&self) -> &MeiliClient {
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

            // Plans
            "list_plans" => self.list_plans(args).await,
            "create_plan" => self.create_plan(args).await,
            "get_plan" => self.get_plan(args).await,
            "update_plan_status" => self.update_plan_status(args).await,
            "link_plan_to_project" => self.link_plan_to_project(args).await,
            "unlink_plan_from_project" => self.unlink_plan_from_project(args).await,
            "get_dependency_graph" => self.get_dependency_graph(args).await,
            "get_critical_path" => self.get_critical_path(args).await,

            // Tasks
            "list_tasks" => self.list_tasks(args).await,
            "create_task" => self.create_task(args).await,
            "get_task" => self.get_task(args).await,
            "update_task" => self.update_task(args).await,
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
            "update_step" => self.update_step(args).await,
            "get_step_progress" => self.get_step_progress(args).await,

            // Constraints
            "list_constraints" => self.list_constraints(args).await,
            "add_constraint" => self.add_constraint(args).await,
            "delete_constraint" => self.delete_constraint(args).await,

            // Releases
            "list_releases" => self.list_releases(args).await,
            "create_release" => self.create_release(args).await,
            "get_release" => self.get_release(args).await,
            "update_release" => self.update_release(args).await,
            "add_task_to_release" => self.add_task_to_release(args).await,
            "add_commit_to_release" => self.add_commit_to_release(args).await,

            // Milestones
            "list_milestones" => self.list_milestones(args).await,
            "create_milestone" => self.create_milestone(args).await,
            "get_milestone" => self.get_milestone(args).await,
            "update_milestone" => self.update_milestone(args).await,
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
            "search_decisions" => self.search_decisions(args).await,

            // Sync
            "sync_directory" => self.sync_directory(args).await,
            "start_watch" => self.start_watch(args).await,
            "stop_watch" => self.stop_watch(args).await,
            "watch_status" => self.watch_status(args).await,

            // Meilisearch
            "get_meilisearch_stats" => self.get_meilisearch_stats(args).await,
            "delete_meilisearch_orphans" => self.delete_meilisearch_orphans(args).await,

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
            "delete_resource" => self.delete_resource(args).await,
            "link_resource_to_project" => self.link_resource_to_project(args).await,

            // Components
            "list_components" => self.list_components(args).await,
            "create_component" => self.create_component(args).await,
            "get_component" => self.get_component(args).await,
            "delete_component" => self.delete_component(args).await,
            "add_component_dependency" => self.add_component_dependency(args).await,
            "remove_component_dependency" => self.remove_component_dependency(args).await,
            "map_component_to_project" => self.map_component_to_project(args).await,
            "get_workspace_topology" => self.get_workspace_topology(args).await,

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
            root_path: root_path.to_string(),
            description: description.map(|s| s.to_string()),
            created_at: chrono::Utc::now(),
            last_synced: None,
        };

        self.neo4j().create_project(&project).await?;
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

        self.neo4j().delete_project(project.id).await?;
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

        let path = std::path::Path::new(&project.root_path);
        let result = self
            .orchestrator
            .sync_directory_for_project(path, Some(project.id), Some(&project.slug))
            .await?;

        self.neo4j().update_project_synced(project.id).await?;

        Ok(json!({
            "files_synced": result.files_synced,
            "files_skipped": result.files_skipped,
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

        self.neo4j()
            .link_plan_to_project(plan_id, project_id)
            .await?;
        Ok(json!({"linked": true}))
    }

    async fn unlink_plan_from_project(&self, args: Value) -> Result<Value> {
        let plan_id = parse_uuid(&args, "plan_id")?;

        self.neo4j().unlink_plan_from_project(plan_id).await?;
        Ok(json!({"unlinked": true}))
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

    async fn update_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());
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
            self.neo4j().add_task_dependency(task_id, dep_id).await?;
        }
        Ok(json!({"added": true}))
    }

    async fn remove_task_dependency(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let dependency_id = parse_uuid(&args, "dependency_id")?;

        self.neo4j()
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

        self.neo4j().delete_constraint(constraint_id).await?;
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

        self.neo4j().create_release(&release).await?;
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
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());
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

        self.neo4j()
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

        self.neo4j()
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

        self.neo4j()
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

        self.neo4j().create_milestone(&milestone).await?;
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
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());
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

        self.neo4j()
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

        self.neo4j()
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

        let commit = CommitNode {
            hash: hash.to_string(),
            message: message.to_string(),
            author: author.to_string(),
            timestamp: chrono::Utc::now(),
        };

        self.neo4j().create_commit(&commit).await?;
        Ok(serde_json::to_value(commit)?)
    }

    async fn link_commit_to_task(&self, args: Value) -> Result<Value> {
        let task_id = parse_uuid(&args, "task_id")?;
        let commit_sha = args
            .get("commit_sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("commit_sha is required"))?;

        self.neo4j()
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

        self.neo4j()
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

        let results = self
            .meili()
            .search_code_with_scores(query, limit, language, None)
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

        let results = self
            .meili()
            .search_code_in_project(query, limit, language, Some(project_slug))
            .await?;

        Ok(serde_json::to_value(results)?)
    }

    async fn get_file_symbols(&self, args: Value) -> Result<Value> {
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("file_path is required"))?;

        // Get all symbols in the file
        let q = neo4rs::query(
            r#"
            MATCH (f:File {path: $path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
            OPTIONAL MATCH (f)-[:CONTAINS]->(st:Struct)
            OPTIONAL MATCH (f)-[:CONTAINS]->(tr:Trait)
            OPTIONAL MATCH (f)-[:CONTAINS]->(en:Enum)
            RETURN
                collect(DISTINCT func.name) AS functions,
                collect(DISTINCT st.name) AS structs,
                collect(DISTINCT tr.name) AS traits,
                collect(DISTINCT en.name) AS enums
            "#,
        )
        .param("path", file_path.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let row = rows.first().ok_or_else(|| anyhow!("File not found"))?;

        let functions: Vec<String> = row.get("functions").unwrap_or_default();
        let structs: Vec<String> = row.get("structs").unwrap_or_default();
        let traits: Vec<String> = row.get("traits").unwrap_or_default();
        let enums: Vec<String> = row.get("enums").unwrap_or_default();

        Ok(json!({
            "file_path": file_path,
            "functions": functions,
            "structs": structs,
            "traits": traits,
            "enums": enums
        }))
    }

    async fn find_references(&self, args: Value) -> Result<Value> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("symbol is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20);

        // Find files that reference this symbol (functions, structs, traits)
        let q = neo4rs::query(&format!(
            r#"
            MATCH (s)
            WHERE (s:Function OR s:Struct OR s:Trait OR s:Enum) AND s.name = $name
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(s)
            OPTIONAL MATCH (f:File)-[:CONTAINS]->(s)
            RETURN DISTINCT
                s.name AS name,
                labels(s)[0] AS type,
                s.file_path AS definition_file,
                collect(DISTINCT caller.name)[0..{}] AS callers,
                f.path AS file_path
            "#,
            limit
        ))
        .param("name", symbol.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let references: Vec<Value> = rows
            .into_iter()
            .filter_map(|row| {
                let name: String = row.get("name").ok()?;
                let symbol_type: String = row.get("type").ok()?;
                let def_file: String = row.get("definition_file").ok().unwrap_or_default();
                let callers: Vec<String> = row.get("callers").ok().unwrap_or_default();
                Some(json!({
                    "name": name,
                    "type": symbol_type,
                    "definition_file": def_file,
                    "callers": callers
                }))
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
        let dependents = self.neo4j().find_dependent_files(file_path, 3).await?;

        // Get files this file imports
        let q = neo4rs::query(
            r#"
            MATCH (f:File {path: $path})-[:IMPORTS]->(imported:File)
            RETURN imported.path AS path
            "#,
        )
        .param("path", file_path.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let imports: Vec<String> = rows
            .into_iter()
            .filter_map(|row| row.get("path").ok())
            .collect();

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

        // Find functions that call this function
        let q = neo4rs::query(&format!(
            r#"
            MATCH (f:Function {{name: $name}})
            MATCH (caller:Function)-[:CALLS*1..{}]->(f)
            RETURN DISTINCT caller.name AS name
            "#,
            depth
        ))
        .param("name", function.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let callers: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r.get::<String>("name").ok())
            .collect();

        // Find functions this function calls
        let q = neo4rs::query(&format!(
            r#"
            MATCH (f:Function {{name: $name}})
            MATCH (f)-[:CALLS*1..{}]->(callee:Function)
            RETURN DISTINCT callee.name AS name
            "#,
            depth
        ))
        .param("name", function.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let callees: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r.get::<String>("name").ok())
            .collect();

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

        // Find all files that depend on this target (file or symbol)
        let dependents = self.neo4j().find_dependent_files(target, 3).await?;

        // If target is a function, find callers
        let q = neo4rs::query(
            r#"
            MATCH (f:Function {name: $name})
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
            RETURN count(caller) AS caller_count, f.file_path AS file_path
            "#,
        )
        .param("name", target.to_string());

        let rows = self.neo4j().execute_with_params(q).await?;
        let caller_count: i64 = rows
            .first()
            .and_then(|r| r.get("caller_count").ok())
            .unwrap_or(0);

        Ok(json!({
            "target": target,
            "dependent_files": dependents,
            "caller_count": caller_count,
            "impact_level": if dependents.len() > 5 || caller_count > 10 { "high" } else if dependents.len() > 2 || caller_count > 3 { "medium" } else { "low" }
        }))
    }

    async fn get_architecture(&self, _args: Value) -> Result<Value> {
        // Get file count by language
        let q = neo4rs::query(
            r#"
            MATCH (f:File)
            RETURN f.language AS language, count(f) AS count
            ORDER BY count DESC
            "#,
        );

        let rows = self.neo4j().execute_with_params(q).await?;
        let languages: Vec<Value> = rows
            .into_iter()
            .filter_map(|r| {
                let lang: String = r.get("language").ok()?;
                let count: i64 = r.get("count").ok()?;
                Some(json!({"language": lang, "file_count": count}))
            })
            .collect();

        // Get most connected files (highest import relationships)
        let q = neo4rs::query(
            r#"
            MATCH (f:File)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
            OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f)
            WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents
            RETURN f.path AS path, imports, dependents, imports + dependents AS connections
            ORDER BY connections DESC
            LIMIT 10
            "#,
        );

        let rows = self.neo4j().execute_with_params(q).await?;
        let key_files: Vec<Value> = rows
            .into_iter()
            .filter_map(|r| {
                let path: String = r.get("path").ok()?;
                let imports: i64 = r.get("imports").ok().unwrap_or(0);
                let dependents: i64 = r.get("dependents").ok().unwrap_or(0);
                Some(json!({"path": path, "imports": imports, "dependents": dependents}))
            })
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

        // Use meilisearch to find similar code by searching for the snippet
        let results = self
            .meili()
            .search_code_with_scores(code_snippet, limit, None, None)
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

        let decisions = self.meili().search_decisions(query, limit).await?;
        Ok(serde_json::to_value(decisions)?)
    }

    // ========================================================================
    // Sync Handlers
    // ========================================================================

    async fn sync_directory(&self, args: Value) -> Result<Value> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("path is required"))?;
        let project_id = args
            .get("project_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let path = std::path::Path::new(path);
        let result = self
            .orchestrator
            .sync_directory_for_project(path, project_id, None)
            .await?;

        Ok(json!({
            "files_synced": result.files_synced,
            "files_skipped": result.files_skipped,
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
            scope_type: None,
            sort_by: None,
            sort_order: None,
        };

        let (notes, total) = self
            .orchestrator
            .note_manager()
            .list_notes(project_id, &filters)
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

        let project_id = parse_uuid(&args, "project_id")?;
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
        let project_id = parse_uuid(&args, "project_id")?;
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

        self.neo4j().create_workspace(&workspace).await?;
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

        self.neo4j()
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

        self.neo4j().delete_workspace(workspace.id).await?;
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

        self.neo4j()
            .add_project_to_workspace(project_id, workspace.id)
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

        self.neo4j()
            .remove_project_from_workspace(project_id, workspace.id)
            .await?;

        Ok(json!({"removed": true}))
    }

    // ========================================================================
    // Workspace Milestone Handlers
    // ========================================================================

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

        // Get all milestones and filter in memory
        let all_milestones = self.neo4j().list_workspace_milestones(workspace.id).await?;

        // Filter by status if provided
        let filtered: Vec<_> = if let Some(status) = status_filter {
            let status_lower = status.to_lowercase();
            all_milestones
                .into_iter()
                .filter(|m| {
                    let m_status = match m.status {
                        MilestoneStatus::Open => "open",
                        MilestoneStatus::Closed => "closed",
                    };
                    m_status == status_lower
                })
                .collect()
        } else {
            all_milestones
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

        self.neo4j().create_workspace_milestone(&milestone).await?;
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
        let status = args.get("status").and_then(|v| v.as_str()).and_then(|s| {
            match s.to_lowercase().as_str() {
                "open" => Some(MilestoneStatus::Open),
                "closed" => Some(MilestoneStatus::Closed),
                _ => None,
            }
        });
        let target_date = args
            .get("target_date")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        self.neo4j()
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

        self.neo4j().delete_workspace_milestone(id).await?;
        Ok(json!({"deleted": true}))
    }

    async fn add_task_to_workspace_milestone(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let task_id = parse_uuid(&args, "task_id")?;

        self.neo4j()
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

        self.neo4j().create_resource(&resource).await?;
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

        self.neo4j().delete_resource(id).await?;
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
                self.neo4j()
                    .link_project_implements_resource(project_id, id)
                    .await?;
            }
            "uses" => {
                self.neo4j()
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

        self.neo4j().create_component(&component).await?;
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

        self.neo4j().delete_component(id).await?;
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

        self.neo4j()
            .add_component_dependency(id, depends_on_id, protocol, required)
            .await?;

        Ok(json!({"added": true}))
    }

    async fn remove_component_dependency(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let dep_id = parse_uuid(&args, "dep_id")?;

        self.neo4j().remove_component_dependency(id, dep_id).await?;

        Ok(json!({"removed": true}))
    }

    async fn map_component_to_project(&self, args: Value) -> Result<Value> {
        let id = parse_uuid(&args, "id")?;
        let project_id = parse_uuid(&args, "project_id")?;

        self.neo4j()
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
