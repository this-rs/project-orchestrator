//! `GraphStore` implementation for `Neo4jClient`.
//!
//! Every method simply delegates to the corresponding inherent method on `Neo4jClient`.

use async_trait::async_trait;
use uuid::Uuid;

use super::client::Neo4jClient;
use super::models::*;
use super::traits::GraphStore;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteFilters, NoteImportance, NoteStatus, PropagatedNote,
};
use crate::plan::models::{TaskDetails, UpdateTaskRequest};

#[async_trait]
impl GraphStore for Neo4jClient {
    // ========================================================================
    // Project operations
    // ========================================================================

    async fn create_project(&self, project: &ProjectNode) -> anyhow::Result<()> {
        self.create_project(project).await
    }

    async fn get_project(&self, id: Uuid) -> anyhow::Result<Option<ProjectNode>> {
        self.get_project(id).await
    }

    async fn get_project_by_slug(&self, slug: &str) -> anyhow::Result<Option<ProjectNode>> {
        self.get_project_by_slug(slug).await
    }

    async fn list_projects(&self) -> anyhow::Result<Vec<ProjectNode>> {
        self.list_projects().await
    }

    async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_project(id, name, description, root_path).await
    }

    async fn update_project_synced(&self, id: Uuid) -> anyhow::Result<()> {
        self.update_project_synced(id).await
    }

    async fn delete_project(&self, id: Uuid) -> anyhow::Result<()> {
        self.delete_project(id).await
    }

    // ========================================================================
    // Workspace operations
    // ========================================================================

    async fn create_workspace(&self, workspace: &WorkspaceNode) -> anyhow::Result<()> {
        self.create_workspace(workspace).await
    }

    async fn get_workspace(&self, id: Uuid) -> anyhow::Result<Option<WorkspaceNode>> {
        self.get_workspace(id).await
    }

    async fn get_workspace_by_slug(&self, slug: &str) -> anyhow::Result<Option<WorkspaceNode>> {
        self.get_workspace_by_slug(slug).await
    }

    async fn list_workspaces(&self) -> anyhow::Result<Vec<WorkspaceNode>> {
        self.list_workspaces().await
    }

    async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        self.update_workspace(id, name, description, metadata).await
    }

    async fn delete_workspace(&self, id: Uuid) -> anyhow::Result<()> {
        self.delete_workspace(id).await
    }

    async fn add_project_to_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_project_to_workspace(workspace_id, project_id)
            .await
    }

    async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_project_from_workspace(workspace_id, project_id)
            .await
    }

    async fn list_workspace_projects(
        &self,
        workspace_id: Uuid,
    ) -> anyhow::Result<Vec<ProjectNode>> {
        self.list_workspace_projects(workspace_id).await
    }

    async fn get_project_workspace(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Option<WorkspaceNode>> {
        self.get_project_workspace(project_id).await
    }

    // ========================================================================
    // Workspace Milestone operations
    // ========================================================================

    async fn create_workspace_milestone(
        &self,
        milestone: &WorkspaceMilestoneNode,
    ) -> anyhow::Result<()> {
        self.create_workspace_milestone(milestone).await
    }

    async fn get_workspace_milestone(
        &self,
        id: Uuid,
    ) -> anyhow::Result<Option<WorkspaceMilestoneNode>> {
        self.get_workspace_milestone(id).await
    }

    async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> anyhow::Result<Vec<WorkspaceMilestoneNode>> {
        self.list_workspace_milestones(workspace_id).await
    }

    async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<WorkspaceMilestoneNode>, usize)> {
        self.list_workspace_milestones_filtered(workspace_id, status, limit, offset)
            .await
    }

    async fn list_all_workspace_milestones_filtered(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<Vec<(WorkspaceMilestoneNode, String, String, String)>> {
        self.list_all_workspace_milestones_filtered(workspace_id, status, limit, offset)
            .await
    }

    async fn count_all_workspace_milestones(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
    ) -> anyhow::Result<usize> {
        self.count_all_workspace_milestones(workspace_id, status)
            .await
    }

    async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> anyhow::Result<()> {
        self.update_workspace_milestone(id, title, description, status, target_date)
            .await
    }

    async fn delete_workspace_milestone(&self, id: Uuid) -> anyhow::Result<()> {
        self.delete_workspace_milestone(id).await
    }

    async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_task_to_workspace_milestone(milestone_id, task_id)
            .await
    }

    async fn remove_task_from_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_task_from_workspace_milestone(milestone_id, task_id)
            .await
    }

    async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<(u32, u32, u32, u32)> {
        self.get_workspace_milestone_progress(milestone_id).await
    }

    async fn get_workspace_milestone_tasks(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<Vec<TaskNode>> {
        self.get_workspace_milestone_tasks(milestone_id).await
    }

    // ========================================================================
    // Resource operations
    // ========================================================================

    async fn create_resource(&self, resource: &ResourceNode) -> anyhow::Result<()> {
        self.create_resource(resource).await
    }

    async fn get_resource(&self, id: Uuid) -> anyhow::Result<Option<ResourceNode>> {
        self.get_resource(id).await
    }

    async fn list_workspace_resources(
        &self,
        workspace_id: Uuid,
    ) -> anyhow::Result<Vec<ResourceNode>> {
        self.list_workspace_resources(workspace_id).await
    }

    async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_resource(id, name, file_path, url, version, description)
            .await
    }

    async fn delete_resource(&self, id: Uuid) -> anyhow::Result<()> {
        self.delete_resource(id).await
    }

    async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_project_implements_resource(project_id, resource_id)
            .await
    }

    async fn link_project_uses_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_project_uses_resource(project_id, resource_id)
            .await
    }

    async fn get_resource_implementers(
        &self,
        resource_id: Uuid,
    ) -> anyhow::Result<Vec<ProjectNode>> {
        self.get_resource_implementers(resource_id).await
    }

    async fn get_resource_consumers(
        &self,
        resource_id: Uuid,
    ) -> anyhow::Result<Vec<ProjectNode>> {
        self.get_resource_consumers(resource_id).await
    }

    // ========================================================================
    // Component operations (Topology)
    // ========================================================================

    async fn create_component(&self, component: &ComponentNode) -> anyhow::Result<()> {
        self.create_component(component).await
    }

    async fn get_component(&self, id: Uuid) -> anyhow::Result<Option<ComponentNode>> {
        self.get_component(id).await
    }

    async fn list_components(&self, workspace_id: Uuid) -> anyhow::Result<Vec<ComponentNode>> {
        self.list_components(workspace_id).await
    }

    async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        self.update_component(id, name, description, runtime, config, tags)
            .await
    }

    async fn delete_component(&self, id: Uuid) -> anyhow::Result<()> {
        self.delete_component(id).await
    }

    async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> anyhow::Result<()> {
        self.add_component_dependency(component_id, depends_on_id, protocol, required)
            .await
    }

    async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_component_dependency(component_id, depends_on_id)
            .await
    }

    async fn map_component_to_project(
        &self,
        component_id: Uuid,
        project_id: Uuid,
    ) -> anyhow::Result<()> {
        self.map_component_to_project(component_id, project_id)
            .await
    }

    async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> anyhow::Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>> {
        self.get_workspace_topology(workspace_id).await
    }

    // ========================================================================
    // File operations
    // ========================================================================

    async fn get_project_file_paths(&self, project_id: Uuid) -> anyhow::Result<Vec<String>> {
        self.get_project_file_paths(project_id).await
    }

    async fn delete_file(&self, path: &str) -> anyhow::Result<()> {
        self.delete_file(path).await
    }

    async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> anyhow::Result<(usize, usize)> {
        self.delete_stale_files(project_id, valid_paths).await
    }

    async fn link_file_to_project(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_file_to_project(file_path, project_id).await
    }

    async fn upsert_file(&self, file: &FileNode) -> anyhow::Result<()> {
        self.upsert_file(file).await
    }

    async fn get_file(&self, path: &str) -> anyhow::Result<Option<FileNode>> {
        self.get_file(path).await
    }

    async fn list_project_files(&self, project_id: Uuid) -> anyhow::Result<Vec<FileNode>> {
        self.list_project_files(project_id).await
    }

    // ========================================================================
    // Symbol operations
    // ========================================================================

    async fn upsert_function(&self, func: &FunctionNode) -> anyhow::Result<()> {
        self.upsert_function(func).await
    }

    async fn upsert_struct(&self, s: &StructNode) -> anyhow::Result<()> {
        self.upsert_struct(s).await
    }

    async fn upsert_trait(&self, t: &TraitNode) -> anyhow::Result<()> {
        self.upsert_trait(t).await
    }

    async fn find_trait_by_name(&self, name: &str) -> anyhow::Result<Option<String>> {
        self.find_trait_by_name(name).await
    }

    async fn upsert_enum(&self, e: &EnumNode) -> anyhow::Result<()> {
        self.upsert_enum(e).await
    }

    async fn upsert_impl(&self, impl_node: &ImplNode) -> anyhow::Result<()> {
        self.upsert_impl(impl_node).await
    }

    async fn create_import_relationship(
        &self,
        from_file: &str,
        to_file: &str,
        import_path: &str,
    ) -> anyhow::Result<()> {
        self.create_import_relationship(from_file, to_file, import_path)
            .await
    }

    async fn upsert_import(&self, import: &ImportNode) -> anyhow::Result<()> {
        self.upsert_import(import).await
    }

    async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
    ) -> anyhow::Result<()> {
        self.create_call_relationship(caller_id, callee_name).await
    }

    async fn get_callees(
        &self,
        function_id: &str,
        depth: u32,
    ) -> anyhow::Result<Vec<FunctionNode>> {
        self.get_callees(function_id, depth).await
    }

    async fn create_uses_type_relationship(
        &self,
        function_id: &str,
        type_name: &str,
    ) -> anyhow::Result<()> {
        self.create_uses_type_relationship(function_id, type_name)
            .await
    }

    async fn find_trait_implementors(&self, trait_name: &str) -> anyhow::Result<Vec<String>> {
        self.find_trait_implementors(trait_name).await
    }

    async fn get_type_traits(&self, type_name: &str) -> anyhow::Result<Vec<String>> {
        self.get_type_traits(type_name).await
    }

    async fn get_impl_blocks(&self, type_name: &str) -> anyhow::Result<Vec<serde_json::Value>> {
        self.get_impl_blocks(type_name).await
    }

    // ========================================================================
    // Code exploration queries
    // ========================================================================

    async fn get_file_language(&self, path: &str) -> anyhow::Result<Option<String>> {
        self.get_file_language(path).await
    }

    async fn get_file_functions_summary(
        &self,
        path: &str,
    ) -> anyhow::Result<Vec<FunctionSummaryNode>> {
        self.get_file_functions_summary(path).await
    }

    async fn get_file_structs_summary(
        &self,
        path: &str,
    ) -> anyhow::Result<Vec<StructSummaryNode>> {
        self.get_file_structs_summary(path).await
    }

    async fn get_file_import_paths_list(&self, path: &str) -> anyhow::Result<Vec<String>> {
        self.get_file_import_paths_list(path).await
    }

    async fn find_symbol_references(
        &self,
        symbol: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<SymbolReferenceNode>> {
        self.find_symbol_references(symbol, limit).await
    }

    async fn get_file_direct_imports(&self, path: &str) -> anyhow::Result<Vec<FileImportNode>> {
        self.get_file_direct_imports(path).await
    }

    async fn get_function_callers_by_name(
        &self,
        function_name: &str,
        depth: u32,
    ) -> anyhow::Result<Vec<String>> {
        self.get_function_callers_by_name(function_name, depth)
            .await
    }

    async fn get_function_callees_by_name(
        &self,
        function_name: &str,
        depth: u32,
    ) -> anyhow::Result<Vec<String>> {
        self.get_function_callees_by_name(function_name, depth)
            .await
    }

    async fn get_language_stats(&self) -> anyhow::Result<Vec<LanguageStatsNode>> {
        self.get_language_stats().await
    }

    async fn get_most_connected_files(&self, limit: usize) -> anyhow::Result<Vec<String>> {
        self.get_most_connected_files(limit).await
    }

    async fn get_most_connected_files_detailed(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ConnectedFileNode>> {
        self.get_most_connected_files_detailed(limit).await
    }

    async fn get_file_symbol_names(&self, path: &str) -> anyhow::Result<FileSymbolNamesNode> {
        self.get_file_symbol_names(path).await
    }

    async fn get_function_caller_count(&self, function_name: &str) -> anyhow::Result<i64> {
        self.get_function_caller_count(function_name).await
    }

    async fn get_trait_info(&self, trait_name: &str) -> anyhow::Result<Option<TraitInfoNode>> {
        self.get_trait_info(trait_name).await
    }

    async fn get_trait_implementors_detailed(
        &self,
        trait_name: &str,
    ) -> anyhow::Result<Vec<TraitImplementorNode>> {
        self.get_trait_implementors_detailed(trait_name).await
    }

    async fn get_type_trait_implementations(
        &self,
        type_name: &str,
    ) -> anyhow::Result<Vec<TypeTraitInfoNode>> {
        self.get_type_trait_implementations(type_name).await
    }

    async fn get_type_impl_blocks_detailed(
        &self,
        type_name: &str,
    ) -> anyhow::Result<Vec<ImplBlockDetailNode>> {
        self.get_type_impl_blocks_detailed(type_name).await
    }

    // ========================================================================
    // Plan operations
    // ========================================================================

    async fn create_plan(&self, plan: &PlanNode) -> anyhow::Result<()> {
        self.create_plan(plan).await
    }

    async fn get_plan(&self, id: Uuid) -> anyhow::Result<Option<PlanNode>> {
        self.get_plan(id).await
    }

    async fn list_active_plans(&self) -> anyhow::Result<Vec<PlanNode>> {
        self.list_active_plans().await
    }

    async fn list_project_plans(&self, project_id: Uuid) -> anyhow::Result<Vec<PlanNode>> {
        self.list_project_plans(project_id).await
    }

    async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<PlanNode>, usize)> {
        self.list_plans_for_project(project_id, status_filter, limit, offset)
            .await
    }

    async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> anyhow::Result<()> {
        self.update_plan_status(id, status).await
    }

    async fn link_plan_to_project(
        &self,
        plan_id: Uuid,
        project_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_plan_to_project(plan_id, project_id).await
    }

    async fn unlink_plan_from_project(&self, plan_id: Uuid) -> anyhow::Result<()> {
        self.unlink_plan_from_project(plan_id).await
    }

    async fn delete_plan(&self, plan_id: Uuid) -> anyhow::Result<()> {
        self.delete_plan(plan_id).await
    }

    // ========================================================================
    // Task operations
    // ========================================================================

    async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> anyhow::Result<()> {
        self.create_task(plan_id, task).await
    }

    async fn get_plan_tasks(&self, plan_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_plan_tasks(plan_id).await
    }

    async fn get_task_with_full_details(
        &self,
        task_id: Uuid,
    ) -> anyhow::Result<Option<TaskDetails>> {
        self.get_task_with_full_details(task_id).await
    }

    async fn analyze_task_impact(&self, task_id: Uuid) -> anyhow::Result<Vec<String>> {
        self.analyze_task_impact(task_id).await
    }

    async fn find_blocked_tasks(
        &self,
        plan_id: Uuid,
    ) -> anyhow::Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        self.find_blocked_tasks(plan_id).await
    }

    async fn update_task_status(
        &self,
        task_id: Uuid,
        status: TaskStatus,
    ) -> anyhow::Result<()> {
        self.update_task_status(task_id, status).await
    }

    async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> anyhow::Result<()> {
        self.assign_task(task_id, agent_id).await
    }

    async fn add_task_dependency(
        &self,
        task_id: Uuid,
        depends_on_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_task_dependency(task_id, depends_on_id).await
    }

    async fn remove_task_dependency(
        &self,
        task_id: Uuid,
        depends_on_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_task_dependency(task_id, depends_on_id).await
    }

    async fn get_task_blockers(&self, task_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_task_blockers(task_id).await
    }

    async fn get_tasks_blocked_by(&self, task_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_tasks_blocked_by(task_id).await
    }

    async fn get_task_dependencies(&self, task_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_task_dependencies(task_id).await
    }

    async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> anyhow::Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        self.get_plan_dependency_graph(plan_id).await
    }

    async fn get_plan_critical_path(&self, plan_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_plan_critical_path(plan_id).await
    }

    async fn get_next_available_task(
        &self,
        plan_id: Uuid,
    ) -> anyhow::Result<Option<TaskNode>> {
        self.get_next_available_task(plan_id).await
    }

    async fn get_task(&self, task_id: Uuid) -> anyhow::Result<Option<TaskNode>> {
        self.get_task(task_id).await
    }

    async fn update_task(
        &self,
        task_id: Uuid,
        updates: &UpdateTaskRequest,
    ) -> anyhow::Result<()> {
        self.update_task(task_id, updates).await
    }

    async fn delete_task(&self, task_id: Uuid) -> anyhow::Result<()> {
        self.delete_task(task_id).await
    }

    // ========================================================================
    // Step operations
    // ========================================================================

    async fn create_step(&self, task_id: Uuid, step: &StepNode) -> anyhow::Result<()> {
        self.create_step(task_id, step).await
    }

    async fn get_task_steps(&self, task_id: Uuid) -> anyhow::Result<Vec<StepNode>> {
        self.get_task_steps(task_id).await
    }

    async fn update_step_status(
        &self,
        step_id: Uuid,
        status: StepStatus,
    ) -> anyhow::Result<()> {
        self.update_step_status(step_id, status).await
    }

    async fn get_task_step_progress(&self, task_id: Uuid) -> anyhow::Result<(u32, u32)> {
        self.get_task_step_progress(task_id).await
    }

    async fn get_step(&self, step_id: Uuid) -> anyhow::Result<Option<StepNode>> {
        self.get_step(step_id).await
    }

    async fn delete_step(&self, step_id: Uuid) -> anyhow::Result<()> {
        self.delete_step(step_id).await
    }

    // ========================================================================
    // Constraint operations
    // ========================================================================

    async fn create_constraint(
        &self,
        plan_id: Uuid,
        constraint: &ConstraintNode,
    ) -> anyhow::Result<()> {
        self.create_constraint(plan_id, constraint).await
    }

    async fn get_plan_constraints(
        &self,
        plan_id: Uuid,
    ) -> anyhow::Result<Vec<ConstraintNode>> {
        self.get_plan_constraints(plan_id).await
    }

    async fn get_constraint(
        &self,
        constraint_id: Uuid,
    ) -> anyhow::Result<Option<ConstraintNode>> {
        self.get_constraint(constraint_id).await
    }

    async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_constraint(constraint_id, description, constraint_type, enforced_by)
            .await
    }

    async fn delete_constraint(&self, constraint_id: Uuid) -> anyhow::Result<()> {
        self.delete_constraint(constraint_id).await
    }

    // ========================================================================
    // Decision operations
    // ========================================================================

    async fn create_decision(
        &self,
        task_id: Uuid,
        decision: &DecisionNode,
    ) -> anyhow::Result<()> {
        self.create_decision(task_id, decision).await
    }

    async fn get_decision(
        &self,
        decision_id: Uuid,
    ) -> anyhow::Result<Option<DecisionNode>> {
        self.get_decision(decision_id).await
    }

    async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_decision(decision_id, description, rationale, chosen_option)
            .await
    }

    async fn delete_decision(&self, decision_id: Uuid) -> anyhow::Result<()> {
        self.delete_decision(decision_id).await
    }

    // ========================================================================
    // Dependency analysis
    // ========================================================================

    async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
    ) -> anyhow::Result<Vec<String>> {
        self.find_dependent_files(file_path, depth).await
    }

    async fn find_callers(&self, function_id: &str) -> anyhow::Result<Vec<FunctionNode>> {
        self.find_callers(function_id).await
    }

    // ========================================================================
    // Task-file linking
    // ========================================================================

    async fn link_task_to_files(
        &self,
        task_id: Uuid,
        file_paths: &[String],
    ) -> anyhow::Result<()> {
        self.link_task_to_files(task_id, file_paths).await
    }

    // ========================================================================
    // Commit operations
    // ========================================================================

    async fn create_commit(&self, commit: &CommitNode) -> anyhow::Result<()> {
        self.create_commit(commit).await
    }

    async fn get_commit(&self, hash: &str) -> anyhow::Result<Option<CommitNode>> {
        self.get_commit(hash).await
    }

    async fn link_commit_to_task(
        &self,
        commit_hash: &str,
        task_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_commit_to_task(commit_hash, task_id).await
    }

    async fn link_commit_to_plan(
        &self,
        commit_hash: &str,
        plan_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_commit_to_plan(commit_hash, plan_id).await
    }

    async fn get_task_commits(&self, task_id: Uuid) -> anyhow::Result<Vec<CommitNode>> {
        self.get_task_commits(task_id).await
    }

    async fn get_plan_commits(&self, plan_id: Uuid) -> anyhow::Result<Vec<CommitNode>> {
        self.get_plan_commits(plan_id).await
    }

    async fn delete_commit(&self, hash: &str) -> anyhow::Result<()> {
        self.delete_commit(hash).await
    }

    // ========================================================================
    // Release operations
    // ========================================================================

    async fn create_release(&self, release: &ReleaseNode) -> anyhow::Result<()> {
        self.create_release(release).await
    }

    async fn get_release(&self, id: Uuid) -> anyhow::Result<Option<ReleaseNode>> {
        self.get_release(id).await
    }

    async fn list_project_releases(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<ReleaseNode>> {
        self.list_project_releases(project_id).await
    }

    async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_release(id, status, target_date, released_at, title, description)
            .await
    }

    async fn add_task_to_release(
        &self,
        release_id: Uuid,
        task_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_task_to_release(release_id, task_id).await
    }

    async fn add_commit_to_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> anyhow::Result<()> {
        self.add_commit_to_release(release_id, commit_hash).await
    }

    async fn get_release_details(
        &self,
        release_id: Uuid,
    ) -> anyhow::Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>> {
        self.get_release_details(release_id).await
    }

    async fn delete_release(&self, release_id: Uuid) -> anyhow::Result<()> {
        self.delete_release(release_id).await
    }

    // ========================================================================
    // Milestone operations
    // ========================================================================

    async fn create_milestone(&self, milestone: &MilestoneNode) -> anyhow::Result<()> {
        self.create_milestone(milestone).await
    }

    async fn get_milestone(&self, id: Uuid) -> anyhow::Result<Option<MilestoneNode>> {
        self.get_milestone(id).await
    }

    async fn list_project_milestones(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<MilestoneNode>> {
        self.list_project_milestones(project_id).await
    }

    async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_milestone(id, status, target_date, closed_at, title, description)
            .await
    }

    async fn add_task_to_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_task_to_milestone(milestone_id, task_id).await
    }

    async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<Option<(MilestoneNode, Vec<TaskNode>)>> {
        self.get_milestone_details(milestone_id).await
    }

    async fn get_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<(u32, u32)> {
        self.get_milestone_progress(milestone_id).await
    }

    async fn delete_milestone(&self, milestone_id: Uuid) -> anyhow::Result<()> {
        self.delete_milestone(milestone_id).await
    }

    async fn get_milestone_tasks(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<Vec<TaskNode>> {
        self.get_milestone_tasks(milestone_id).await
    }

    async fn get_release_tasks(&self, release_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_release_tasks(release_id).await
    }

    // ========================================================================
    // Project stats
    // ========================================================================

    async fn get_project_progress(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<(u32, u32, u32, u32)> {
        self.get_project_progress(project_id).await
    }

    async fn get_project_task_dependencies(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, Uuid)>> {
        self.get_project_task_dependencies(project_id).await
    }

    async fn get_project_tasks(&self, project_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_project_tasks(project_id).await
    }

    // ========================================================================
    // Filtered list operations with pagination
    // ========================================================================

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
    ) -> anyhow::Result<(Vec<PlanNode>, usize)> {
        self.list_plans_filtered(
            project_id,
            statuses,
            priority_min,
            priority_max,
            search,
            limit,
            offset,
            sort_by,
            sort_order,
        )
        .await
    }

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
    ) -> anyhow::Result<(Vec<TaskWithPlan>, usize)> {
        self.list_all_tasks_filtered(
            plan_id,
            statuses,
            priority_min,
            priority_max,
            tags,
            assigned_to,
            limit,
            offset,
            sort_by,
            sort_order,
        )
        .await
    }

    async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> anyhow::Result<(Vec<ReleaseNode>, usize)> {
        self.list_releases_filtered(project_id, statuses, limit, offset, sort_by, sort_order)
            .await
    }

    async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> anyhow::Result<(Vec<MilestoneNode>, usize)> {
        self.list_milestones_filtered(project_id, statuses, limit, offset, sort_by, sort_order)
            .await
    }

    async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> anyhow::Result<(Vec<ProjectNode>, usize)> {
        self.list_projects_filtered(search, limit, offset, sort_by, sort_order)
            .await
    }

    // ========================================================================
    // Knowledge Note operations
    // ========================================================================

    async fn create_note(&self, note: &Note) -> anyhow::Result<()> {
        self.create_note(note).await
    }

    async fn get_note(&self, id: Uuid) -> anyhow::Result<Option<Note>> {
        self.get_note(id).await
    }

    async fn update_note(
        &self,
        id: Uuid,
        content: Option<String>,
        importance: Option<NoteImportance>,
        status: Option<NoteStatus>,
        tags: Option<Vec<String>>,
        staleness_score: Option<f64>,
    ) -> anyhow::Result<Option<Note>> {
        self.update_note(id, content, importance, status, tags, staleness_score)
            .await
    }

    async fn delete_note(&self, id: Uuid) -> anyhow::Result<bool> {
        self.delete_note(id).await
    }

    async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        filters: &NoteFilters,
    ) -> anyhow::Result<(Vec<Note>, usize)> {
        self.list_notes(project_id, filters).await
    }

    async fn link_note_to_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> anyhow::Result<()> {
        self.link_note_to_entity(note_id, entity_type, entity_id, signature_hash, body_hash)
            .await
    }

    async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> anyhow::Result<()> {
        self.unlink_note_from_entity(note_id, entity_type, entity_id)
            .await
    }

    async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> anyhow::Result<Vec<Note>> {
        self.get_notes_for_entity(entity_type, entity_id).await
    }

    async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
    ) -> anyhow::Result<Vec<PropagatedNote>> {
        self.get_propagated_notes(entity_type, entity_id, max_depth, min_score)
            .await
    }

    async fn get_workspace_notes_for_project(
        &self,
        project_id: Uuid,
        propagation_factor: f64,
    ) -> anyhow::Result<Vec<PropagatedNote>> {
        self.get_workspace_notes_for_project(project_id, propagation_factor)
            .await
    }

    async fn supersede_note(
        &self,
        old_note_id: Uuid,
        new_note_id: Uuid,
    ) -> anyhow::Result<()> {
        self.supersede_note(old_note_id, new_note_id).await
    }

    async fn confirm_note(
        &self,
        note_id: Uuid,
        confirmed_by: &str,
    ) -> anyhow::Result<Option<Note>> {
        self.confirm_note(note_id, confirmed_by).await
    }

    async fn get_notes_needing_review(
        &self,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<Note>> {
        self.get_notes_needing_review(project_id).await
    }

    async fn update_staleness_scores(&self) -> anyhow::Result<usize> {
        self.update_staleness_scores().await
    }

    async fn get_note_anchors(&self, note_id: Uuid) -> anyhow::Result<Vec<NoteAnchor>> {
        self.get_note_anchors(note_id).await
    }
}
