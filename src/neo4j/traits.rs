//! GraphStore trait definition
//!
//! Defines the abstract interface for all Neo4j graph operations.
//! This trait mirrors all public async methods of `Neo4jClient`,
//! enabling testing with mock implementations and future backend swaps.

use crate::neo4j::models::*;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteFilters, NoteImportance, NoteStatus, PropagatedNote,
};
use crate::plan::models::{TaskDetails, UpdateTaskRequest};
use anyhow::Result;
use async_trait::async_trait;
use uuid::Uuid;

/// Abstract interface for all graph database operations.
///
/// Every public async method of `Neo4jClient` (excluding `new`, `init_schema`,
/// `execute`, `execute_with_params`, and private helpers) is represented here.
#[async_trait]
pub trait GraphStore: Send + Sync {
    // ========================================================================
    // Project operations
    // ========================================================================

    /// Create a new project
    async fn create_project(&self, project: &ProjectNode) -> Result<()>;

    /// Get a project by ID
    async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>>;

    /// Get a project by slug
    async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>>;

    /// List all projects
    async fn list_projects(&self) -> Result<Vec<ProjectNode>>;

    /// Update project fields (name, description, root_path)
    async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()>;

    /// Update project last_synced timestamp
    async fn update_project_synced(&self, id: Uuid) -> Result<()>;

    /// Delete a project and all its data
    async fn delete_project(&self, id: Uuid) -> Result<()>;

    // ========================================================================
    // Workspace operations
    // ========================================================================

    /// Create a new workspace
    async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()>;

    /// Get a workspace by ID
    async fn get_workspace(&self, id: Uuid) -> Result<Option<WorkspaceNode>>;

    /// Get a workspace by slug
    async fn get_workspace_by_slug(&self, slug: &str) -> Result<Option<WorkspaceNode>>;

    /// List all workspaces
    async fn list_workspaces(&self) -> Result<Vec<WorkspaceNode>>;

    /// Update a workspace
    async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()>;

    /// Delete a workspace and all its data
    async fn delete_workspace(&self, id: Uuid) -> Result<()>;

    /// Add a project to a workspace
    async fn add_project_to_workspace(&self, workspace_id: Uuid, project_id: Uuid) -> Result<()>;

    /// Remove a project from a workspace
    async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()>;

    /// List all projects in a workspace
    async fn list_workspace_projects(&self, workspace_id: Uuid) -> Result<Vec<ProjectNode>>;

    /// Get the workspace a project belongs to
    async fn get_project_workspace(&self, project_id: Uuid) -> Result<Option<WorkspaceNode>>;

    // ========================================================================
    // Workspace Milestone operations
    // ========================================================================

    /// Create a workspace milestone
    async fn create_workspace_milestone(&self, milestone: &WorkspaceMilestoneNode) -> Result<()>;

    /// Get a workspace milestone by ID
    async fn get_workspace_milestone(&self, id: Uuid) -> Result<Option<WorkspaceMilestoneNode>>;

    /// List workspace milestones (unpaginated, used internally)
    async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<WorkspaceMilestoneNode>>;

    /// List workspace milestones with pagination and status filter
    ///
    /// Returns (milestones, total_count)
    async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<WorkspaceMilestoneNode>, usize)>;

    /// List all workspace milestones across all workspaces with filters and pagination
    ///
    /// Returns (milestones_with_workspace_info, total_count)
    async fn list_all_workspace_milestones_filtered(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(WorkspaceMilestoneNode, String, String, String)>>;

    /// Count all workspace milestones across workspaces with optional filters
    async fn count_all_workspace_milestones(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
    ) -> Result<usize>;

    /// Update a workspace milestone
    async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()>;

    /// Delete a workspace milestone
    async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()>;

    /// Add a task to a workspace milestone
    async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()>;

    /// Remove a task from a workspace milestone
    async fn remove_task_from_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()>;

    /// Get workspace milestone progress
    async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)>;

    /// Get tasks linked to a workspace milestone
    async fn get_workspace_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>>;

    // ========================================================================
    // Resource operations
    // ========================================================================

    /// Create a resource
    async fn create_resource(&self, resource: &ResourceNode) -> Result<()>;

    /// Get a resource by ID
    async fn get_resource(&self, id: Uuid) -> Result<Option<ResourceNode>>;

    /// List workspace resources
    async fn list_workspace_resources(&self, workspace_id: Uuid) -> Result<Vec<ResourceNode>>;

    /// Update a resource
    async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()>;

    /// Delete a resource
    async fn delete_resource(&self, id: Uuid) -> Result<()>;

    /// Link a project as implementing a resource
    async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()>;

    /// Link a project as using a resource
    async fn link_project_uses_resource(&self, project_id: Uuid, resource_id: Uuid) -> Result<()>;

    /// Get projects that implement a resource
    async fn get_resource_implementers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>>;

    /// Get projects that use a resource
    async fn get_resource_consumers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>>;

    // ========================================================================
    // Component operations (Topology)
    // ========================================================================

    /// Create a component
    async fn create_component(&self, component: &ComponentNode) -> Result<()>;

    /// Get a component by ID
    async fn get_component(&self, id: Uuid) -> Result<Option<ComponentNode>>;

    /// List components in a workspace
    async fn list_components(&self, workspace_id: Uuid) -> Result<Vec<ComponentNode>>;

    /// Update a component
    async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()>;

    /// Delete a component
    async fn delete_component(&self, id: Uuid) -> Result<()>;

    /// Add a dependency between components
    async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()>;

    /// Remove a dependency between components
    async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()>;

    /// Map a component to a project
    async fn map_component_to_project(&self, component_id: Uuid, project_id: Uuid) -> Result<()>;

    /// Get the workspace topology (all components with their dependencies)
    async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>>;

    // ========================================================================
    // File operations
    // ========================================================================

    /// Get all file paths for a project
    async fn get_project_file_paths(&self, project_id: Uuid) -> Result<Vec<String>>;

    /// Delete a file and all its symbols
    async fn delete_file(&self, path: &str) -> Result<()>;

    /// Delete files that are no longer on the filesystem
    /// Returns the number of files and symbols deleted
    async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize)>;

    /// Link a file to a project (create CONTAINS relationship)
    async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> Result<()>;

    /// Create or update a file node
    async fn upsert_file(&self, file: &FileNode) -> Result<()>;

    /// Get a file by path
    async fn get_file(&self, path: &str) -> Result<Option<FileNode>>;

    /// List files for a project
    async fn list_project_files(&self, project_id: Uuid) -> Result<Vec<FileNode>>;

    // ========================================================================
    // Symbol operations
    // ========================================================================

    /// Create or update a function node
    async fn upsert_function(&self, func: &FunctionNode) -> Result<()>;

    /// Create or update a struct node
    async fn upsert_struct(&self, s: &StructNode) -> Result<()>;

    /// Create or update a trait node
    async fn upsert_trait(&self, t: &TraitNode) -> Result<()>;

    /// Find a trait by name (searches across all files)
    async fn find_trait_by_name(&self, name: &str) -> Result<Option<String>>;

    /// Create or update an enum node
    async fn upsert_enum(&self, e: &EnumNode) -> Result<()>;

    /// Create or update an impl block node
    async fn upsert_impl(&self, impl_node: &ImplNode) -> Result<()>;

    /// Create an import relationship between files
    async fn create_import_relationship(
        &self,
        from_file: &str,
        to_file: &str,
        import_path: &str,
    ) -> Result<()>;

    /// Store an import node (for tracking even unresolved imports)
    async fn upsert_import(&self, import: &ImportNode) -> Result<()>;

    /// Create a CALLS relationship between functions, scoped to the same project.
    /// When project_id is provided, the callee is matched only within the same project
    /// to prevent cross-project CALLS pollution.
    async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<()>;

    /// Delete all CALLS relationships where caller and callee belong to different projects.
    /// Returns the number of deleted relationships.
    async fn cleanup_cross_project_calls(&self) -> Result<i64>;

    /// Get all functions called by a function
    async fn get_callees(&self, function_id: &str, depth: u32) -> Result<Vec<FunctionNode>>;

    /// Create a USES_TYPE relationship from a function to a type
    async fn create_uses_type_relationship(&self, function_id: &str, type_name: &str)
        -> Result<()>;

    /// Find types that implement a specific trait
    async fn find_trait_implementors(&self, trait_name: &str) -> Result<Vec<String>>;

    /// Get all traits implemented by a type
    async fn get_type_traits(&self, type_name: &str) -> Result<Vec<String>>;

    /// Get all impl blocks for a type
    async fn get_impl_blocks(&self, type_name: &str) -> Result<Vec<serde_json::Value>>;

    // ========================================================================
    // Code exploration queries
    // ========================================================================

    /// Get the language of a file by path
    async fn get_file_language(&self, path: &str) -> Result<Option<String>>;

    /// Get function summaries for a file
    async fn get_file_functions_summary(&self, path: &str) -> Result<Vec<FunctionSummaryNode>>;

    /// Get struct summaries for a file
    async fn get_file_structs_summary(&self, path: &str) -> Result<Vec<StructSummaryNode>>;

    /// Get import paths for a file
    async fn get_file_import_paths_list(&self, path: &str) -> Result<Vec<String>>;

    /// Find references to a symbol (function callers, struct importers, file importers).
    /// When project_id is provided, results are scoped to the same project.
    async fn find_symbol_references(
        &self,
        symbol: &str,
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<SymbolReferenceNode>>;

    /// Get files directly imported by a file
    async fn get_file_direct_imports(&self, path: &str) -> Result<Vec<FileImportNode>>;

    /// Get callers chain for a function name (by name, variable depth).
    /// When project_id is provided, scopes start/end to the same project.
    async fn get_function_callers_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>>;

    /// Get callees chain for a function name (by name, variable depth).
    /// When project_id is provided, scopes start/end to the same project.
    async fn get_function_callees_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>>;

    /// Get language statistics across all files
    async fn get_language_stats(&self) -> Result<Vec<LanguageStatsNode>>;

    /// Get language statistics for a specific project
    async fn get_language_stats_for_project(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<LanguageStatsNode>>;

    /// Get most connected files (highest in-degree from imports)
    async fn get_most_connected_files(&self, limit: usize) -> Result<Vec<String>>;

    /// Get most connected files with import/dependent counts
    async fn get_most_connected_files_detailed(
        &self,
        limit: usize,
    ) -> Result<Vec<ConnectedFileNode>>;

    /// Get most connected files with import/dependent counts for a specific project
    async fn get_most_connected_files_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<ConnectedFileNode>>;

    /// Get aggregated symbol names for a file (functions, structs, traits, enums)
    async fn get_file_symbol_names(&self, path: &str) -> Result<FileSymbolNamesNode>;

    /// Get the number of callers for a function by name.
    /// When project_id is provided, only counts callers from the same project.
    async fn get_function_caller_count(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<i64>;

    /// Get trait info (is_external, source)
    async fn get_trait_info(&self, trait_name: &str) -> Result<Option<TraitInfoNode>>;

    /// Get trait implementors with file locations
    async fn get_trait_implementors_detailed(
        &self,
        trait_name: &str,
    ) -> Result<Vec<TraitImplementorNode>>;

    /// Get all traits implemented by a type, with details
    async fn get_type_trait_implementations(
        &self,
        type_name: &str,
    ) -> Result<Vec<TypeTraitInfoNode>>;

    /// Get all impl blocks for a type with methods
    async fn get_type_impl_blocks_detailed(
        &self,
        type_name: &str,
    ) -> Result<Vec<ImplBlockDetailNode>>;

    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    async fn create_plan(&self, plan: &PlanNode) -> Result<()>;

    /// Get a plan by ID
    async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>>;

    /// List all active plans
    async fn list_active_plans(&self) -> Result<Vec<PlanNode>>;

    /// List active plans for a specific project
    async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>>;

    /// List plans for a project with filters
    async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)>;

    /// Update plan status
    async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()>;

    /// Link a plan to a project (creates HAS_PLAN relationship)
    async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()>;

    /// Unlink a plan from its project
    async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()>;

    /// Delete a plan and all its related data (tasks, steps, decisions, constraints)
    async fn delete_plan(&self, plan_id: Uuid) -> Result<()>;

    // ========================================================================
    // Task operations
    // ========================================================================

    /// Create a task for a plan
    async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> Result<()>;

    /// Get tasks for a plan
    async fn get_plan_tasks(&self, plan_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get full task details including steps, decisions, dependencies, and modified files
    async fn get_task_with_full_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>>;

    /// Analyze the impact of a task on the codebase (files it modifies + their dependents)
    async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>>;

    /// Find pending tasks in a plan that are blocked by uncompleted dependencies
    async fn find_blocked_tasks(&self, plan_id: Uuid) -> Result<Vec<(TaskNode, Vec<TaskNode>)>>;

    /// Update task status
    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> Result<()>;

    /// Assign task to an agent
    async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> Result<()>;

    /// Add task dependency
    async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()>;

    /// Remove task dependency
    async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()>;

    /// Get tasks that block this task (dependencies that are not completed)
    async fn get_task_blockers(&self, task_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get tasks blocked by this task (tasks depending on this one)
    async fn get_tasks_blocked_by(&self, task_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get all dependencies for a task (all tasks it depends on, regardless of status)
    async fn get_task_dependencies(&self, task_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get dependency graph for a plan (all tasks and their dependencies)
    async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)>;

    /// Find critical path in a plan (longest chain of dependencies)
    async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get next available task (no unfinished dependencies)
    async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>>;

    /// Get a single task by ID
    async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>>;

    /// Update a task with new values
    async fn update_task(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> Result<()>;

    /// Delete a task and all its related data (steps, decisions)
    async fn delete_task(&self, task_id: Uuid) -> Result<()>;

    // ========================================================================
    // Step operations
    // ========================================================================

    /// Create a step for a task
    async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()>;

    /// Get steps for a task
    async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>>;

    /// Update step status
    async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()>;

    /// Get count of completed steps for a task
    async fn get_task_step_progress(&self, task_id: Uuid) -> Result<(u32, u32)>;

    /// Get a single step by ID
    async fn get_step(&self, step_id: Uuid) -> Result<Option<StepNode>>;

    /// Delete a step
    async fn delete_step(&self, step_id: Uuid) -> Result<()>;

    // ========================================================================
    // Constraint operations
    // ========================================================================

    /// Create a constraint for a plan
    async fn create_constraint(&self, plan_id: Uuid, constraint: &ConstraintNode) -> Result<()>;

    /// Get constraints for a plan
    async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>>;

    /// Get a single constraint by ID
    async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>>;

    /// Update a constraint
    async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()>;

    /// Delete a constraint
    async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()>;

    // ========================================================================
    // Decision operations
    // ========================================================================

    /// Record a decision
    async fn create_decision(&self, task_id: Uuid, decision: &DecisionNode) -> Result<()>;

    /// Get a single decision by ID
    async fn get_decision(&self, decision_id: Uuid) -> Result<Option<DecisionNode>>;

    /// Update a decision
    async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> Result<()>;

    /// Delete a decision
    async fn delete_decision(&self, decision_id: Uuid) -> Result<()>;

    // ========================================================================
    // Dependency analysis
    // ========================================================================

    /// Find all files that depend on a given file.
    /// When project_id is provided, only return dependents from the same project.
    async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>>;

    /// Find all functions that call a given function
    /// When project_id is provided, only return callers from the same project.
    async fn find_callers(
        &self,
        function_id: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FunctionNode>>;

    // ========================================================================
    // Task-file linking
    // ========================================================================

    /// Link a task to files it modifies
    async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> Result<()>;

    // ========================================================================
    // Commit operations
    // ========================================================================

    /// Create a commit node
    async fn create_commit(&self, commit: &CommitNode) -> Result<()>;

    /// Get a commit by hash
    async fn get_commit(&self, hash: &str) -> Result<Option<CommitNode>>;

    /// Link a commit to a task (RESOLVED_BY relationship)
    async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()>;

    /// Link a commit to a plan (RESULTED_IN relationship)
    async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()>;

    /// Get commits for a task
    async fn get_task_commits(&self, task_id: Uuid) -> Result<Vec<CommitNode>>;

    /// Get commits for a plan
    async fn get_plan_commits(&self, plan_id: Uuid) -> Result<Vec<CommitNode>>;

    /// Delete a commit
    async fn delete_commit(&self, hash: &str) -> Result<()>;

    // ========================================================================
    // Release operations
    // ========================================================================

    /// Create a release
    async fn create_release(&self, release: &ReleaseNode) -> Result<()>;

    /// Get a release by ID
    async fn get_release(&self, id: Uuid) -> Result<Option<ReleaseNode>>;

    /// List releases for a project
    async fn list_project_releases(&self, project_id: Uuid) -> Result<Vec<ReleaseNode>>;

    /// Update a release
    async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()>;

    /// Add a task to a release
    async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()>;

    /// Add a commit to a release
    async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()>;

    /// Get release details with tasks and commits
    async fn get_release_details(
        &self,
        release_id: Uuid,
    ) -> Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>>;

    /// Delete a release
    async fn delete_release(&self, release_id: Uuid) -> Result<()>;

    // ========================================================================
    // Milestone operations
    // ========================================================================

    /// Create a milestone
    async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()>;

    /// Get a milestone by ID
    async fn get_milestone(&self, id: Uuid) -> Result<Option<MilestoneNode>>;

    /// List milestones for a project
    async fn list_project_milestones(&self, project_id: Uuid) -> Result<Vec<MilestoneNode>>;

    /// Update a milestone
    async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()>;

    /// Add a task to a milestone
    async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()>;

    /// Get milestone details with tasks
    async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>>;

    /// Get milestone progress (completed tasks / total tasks)
    async fn get_milestone_progress(&self, milestone_id: Uuid) -> Result<(u32, u32)>;

    /// Delete a milestone
    async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()>;

    /// Get tasks for a milestone
    async fn get_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Get tasks for a release
    async fn get_release_tasks(&self, release_id: Uuid) -> Result<Vec<TaskNode>>;

    // ========================================================================
    // Project stats
    // ========================================================================

    /// Get project progress stats
    async fn get_project_progress(&self, project_id: Uuid) -> Result<(u32, u32, u32, u32)>;

    /// Get all task dependencies for a project (across all plans)
    async fn get_project_task_dependencies(&self, project_id: Uuid) -> Result<Vec<(Uuid, Uuid)>>;

    /// Get all tasks for a project (across all plans)
    async fn get_project_tasks(&self, project_id: Uuid) -> Result<Vec<TaskNode>>;

    // ========================================================================
    // Filtered list operations with pagination
    // ========================================================================

    /// List plans with filters and pagination
    ///
    /// Returns (plans, total_count)
    #[allow(clippy::too_many_arguments)]
    async fn list_plans_filtered(
        &self,
        project_id: Option<Uuid>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)>;

    /// List all tasks across all plans with filters and pagination
    ///
    /// Returns (tasks_with_plan_info, total_count)
    #[allow(clippy::too_many_arguments)]
    async fn list_all_tasks_filtered(
        &self,
        plan_id: Option<Uuid>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        tags: Option<Vec<String>>,
        assigned_to: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<TaskWithPlan>, usize)>;

    /// List project releases with filters and pagination
    async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ReleaseNode>, usize)>;

    /// List project milestones with filters and pagination
    async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<MilestoneNode>, usize)>;

    /// List projects with search and pagination
    async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ProjectNode>, usize)>;

    // ========================================================================
    // Knowledge Note operations
    // ========================================================================

    /// Create a new note
    async fn create_note(&self, note: &Note) -> Result<()>;

    /// Get a note by ID
    async fn get_note(&self, id: Uuid) -> Result<Option<Note>>;

    /// Update a note
    async fn update_note(
        &self,
        id: Uuid,
        content: Option<String>,
        importance: Option<NoteImportance>,
        status: Option<NoteStatus>,
        tags: Option<Vec<String>>,
        staleness_score: Option<f64>,
    ) -> Result<Option<Note>>;

    /// Delete a note
    async fn delete_note(&self, id: Uuid) -> Result<bool>;

    /// List notes with filters and pagination
    async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)>;

    /// Link a note to an entity (File, Function, Task, etc.)
    async fn link_note_to_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> Result<()>;

    /// Unlink a note from an entity
    async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()>;

    /// Get all notes attached to an entity
    async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<Note>>;

    /// Get propagated notes for an entity (traversing the graph)
    async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
    ) -> Result<Vec<PropagatedNote>>;

    /// Get workspace-level notes for a project (propagated from parent workspace)
    async fn get_workspace_notes_for_project(
        &self,
        project_id: Uuid,
        propagation_factor: f64,
    ) -> Result<Vec<PropagatedNote>>;

    /// Mark a note as superseded by another
    async fn supersede_note(&self, old_note_id: Uuid, new_note_id: Uuid) -> Result<()>;

    /// Confirm a note is still valid
    async fn confirm_note(&self, note_id: Uuid, confirmed_by: &str) -> Result<Option<Note>>;

    /// Get notes that need review (stale or needs_review status)
    async fn get_notes_needing_review(&self, project_id: Option<Uuid>) -> Result<Vec<Note>>;

    /// Update staleness scores for all active notes
    async fn update_staleness_scores(&self) -> Result<usize>;

    /// Get anchors for a note
    async fn get_note_anchors(&self, note_id: Uuid) -> Result<Vec<NoteAnchor>>;

    // ========================================================================
    // Chat session operations
    // ========================================================================

    /// Create a new chat session
    async fn create_chat_session(&self, session: &ChatSessionNode) -> Result<()>;

    /// Get a chat session by ID
    async fn get_chat_session(&self, id: Uuid) -> Result<Option<ChatSessionNode>>;

    /// List chat sessions with optional project_slug filter and pagination
    ///
    /// Returns (sessions, total_count)
    async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ChatSessionNode>, usize)>;

    /// Update a chat session (partial update, None fields are skipped)
    #[allow(clippy::too_many_arguments)]
    async fn update_chat_session(
        &self,
        id: Uuid,
        cli_session_id: Option<String>,
        title: Option<String>,
        message_count: Option<i64>,
        total_cost_usd: Option<f64>,
        conversation_id: Option<String>,
        preview: Option<String>,
    ) -> Result<Option<ChatSessionNode>>;

    /// Backfill title and preview for sessions that don't have them yet
    async fn backfill_chat_session_previews(&self) -> Result<usize>;

    /// Delete a chat session
    async fn delete_chat_session(&self, id: Uuid) -> Result<bool>;

    // ========================================================================
    // Chat event operations (WebSocket replay & persistence)
    // ========================================================================

    /// Store a batch of chat events for a session
    async fn store_chat_events(&self, session_id: Uuid, events: Vec<ChatEventRecord>)
        -> Result<()>;

    /// Get chat events for a session after a given sequence number (for replay)
    async fn get_chat_events(
        &self,
        session_id: Uuid,
        after_seq: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>>;

    /// Get chat events for a session with offset-based pagination (for REST/MCP).
    async fn get_chat_events_paginated(
        &self,
        session_id: Uuid,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>>;

    /// Count total chat events for a session.
    async fn count_chat_events(&self, session_id: Uuid) -> Result<i64>;

    /// Get the latest sequence number for a session (0 if no events)
    async fn get_latest_chat_event_seq(&self, session_id: Uuid) -> Result<i64>;

    /// Delete all chat events for a session
    async fn delete_chat_events(&self, session_id: Uuid) -> Result<()>;

    // ========================================================================
    // User / Auth operations
    // ========================================================================

    /// Upsert a user (create or update). Returns the resulting UserNode.
    ///
    /// For OIDC users: MERGE on (auth_provider + external_id).
    /// For password users: MERGE on (auth_provider + email).
    async fn upsert_user(&self, user: &UserNode) -> Result<UserNode>;

    /// Get a user by internal ID
    async fn get_user_by_id(&self, id: Uuid) -> Result<Option<UserNode>>;

    /// Get a user by provider and external ID (for OIDC lookups)
    async fn get_user_by_provider_id(
        &self,
        provider: &str,
        external_id: &str,
    ) -> Result<Option<UserNode>>;

    /// Get a user by email and auth provider
    async fn get_user_by_email_and_provider(
        &self,
        email: &str,
        provider: &str,
    ) -> Result<Option<UserNode>>;

    /// Get a user by email (any provider)
    async fn get_user_by_email(&self, email: &str) -> Result<Option<UserNode>>;

    /// Create a password-authenticated user
    async fn create_password_user(
        &self,
        email: &str,
        name: &str,
        password_hash: &str,
    ) -> Result<UserNode>;

    /// List all users
    async fn list_users(&self) -> Result<Vec<UserNode>>;

    // ================================================================
    // Feature Graphs
    // ================================================================

    /// Create a feature graph
    async fn create_feature_graph(&self, graph: &FeatureGraphNode) -> Result<()>;

    /// Get a feature graph by ID (without entities)
    async fn get_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphNode>>;

    /// Get a feature graph with all its included entities
    async fn get_feature_graph_detail(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>>;

    /// List feature graphs, optionally filtered by project
    async fn list_feature_graphs(&self, project_id: Option<Uuid>) -> Result<Vec<FeatureGraphNode>>;

    /// Delete a feature graph and all its INCLUDES_ENTITY relationships
    async fn delete_feature_graph(&self, id: Uuid) -> Result<bool>;

    /// Add an entity (file, function, struct, trait) to a feature graph
    async fn add_entity_to_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        role: Option<&str>,
    ) -> Result<()>;

    /// Remove an entity from a feature graph
    async fn remove_entity_from_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<bool>;

    /// Automatically build a feature graph from a function entry point.
    /// Uses the call graph (callers + callees) to discover related functions and files,
    /// creates a FeatureGraph, and populates it with the discovered entities.
    async fn auto_build_feature_graph(
        &self,
        name: &str,
        description: Option<&str>,
        project_id: Uuid,
        entry_function: &str,
        depth: u32,
        include_relations: Option<&[String]>,
    ) -> Result<FeatureGraphDetail>;

    /// Refresh an auto-built feature graph by re-running the BFS with the
    /// same parameters (entry_function, depth, include_relations).
    /// Returns None if the graph was manually created (no entry_function).
    /// Returns Err if the graph doesn't exist.
    async fn refresh_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>>;
}
