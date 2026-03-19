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
use crate::plan::models::{TaskDetails, UpdatePlanRequest, UpdateStepRequest, UpdateTaskRequest};

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

    async fn update_project_analytics_timestamp(&self, id: Uuid) -> anyhow::Result<()> {
        self.update_project_analytics_timestamp(id).await
    }

    async fn delete_project(&self, id: Uuid, project_name: &str) -> anyhow::Result<()> {
        self.delete_project(id, project_name).await
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
        slug: Option<String>,
    ) -> anyhow::Result<()> {
        self.update_workspace(id, name, description, metadata, slug)
            .await
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

    async fn compute_coupling_matrix(
        &self,
        workspace_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::models::CouplingMatrix> {
        self.compute_coupling_matrix(workspace_id).await
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

    async fn link_plan_to_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_plan_to_workspace_milestone(plan_id, milestone_id)
            .await
    }

    async fn unlink_plan_from_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> anyhow::Result<()> {
        self.unlink_plan_from_workspace_milestone(plan_id, milestone_id)
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
    ) -> anyhow::Result<Vec<TaskWithPlan>> {
        self.get_workspace_milestone_tasks(milestone_id).await
    }

    async fn get_workspace_milestone_steps(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        self.get_workspace_milestone_steps(milestone_id).await
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

    async fn get_resource_consumers(&self, resource_id: Uuid) -> anyhow::Result<Vec<ProjectNode>> {
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
    ) -> anyhow::Result<(usize, usize, Vec<String>)> {
        self.delete_stale_files(project_id, valid_paths).await
    }

    async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> anyhow::Result<()> {
        self.link_file_to_project(file_path, project_id).await
    }

    async fn upsert_file(&self, file: &FileNode) -> anyhow::Result<()> {
        self.upsert_file(file).await
    }

    async fn batch_upsert_files(&self, files: &[FileNode]) -> anyhow::Result<()> {
        self.batch_upsert_files(files).await
    }

    async fn get_file(&self, path: &str) -> anyhow::Result<Option<FileNode>> {
        self.get_file(path).await
    }

    async fn list_project_files(&self, project_id: Uuid) -> anyhow::Result<Vec<FileNode>> {
        self.list_project_files(project_id).await
    }

    async fn count_project_files(&self, project_id: Uuid) -> anyhow::Result<i64> {
        self.count_project_files(project_id).await
    }

    async fn invalidate_computed_properties(
        &self,
        project_id: Uuid,
        paths: &[String],
    ) -> anyhow::Result<u64> {
        self.invalidate_computed_properties(project_id, paths).await
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

    async fn create_imports_symbol_relationship(
        &self,
        import_id: &str,
        symbol_name: &str,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<()> {
        self.create_imports_symbol_relationship(import_id, symbol_name, project_id)
            .await
    }

    async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
        confidence: f64,
        reason: &str,
    ) -> anyhow::Result<()> {
        self.create_call_relationship(caller_id, callee_name, project_id, confidence, reason)
            .await
    }

    // ========================================================================
    // Batch upsert operations (UNWIND)
    // ========================================================================

    async fn batch_upsert_functions(&self, functions: &[FunctionNode]) -> anyhow::Result<()> {
        self.batch_upsert_functions(functions).await
    }

    async fn batch_upsert_structs(&self, structs: &[StructNode]) -> anyhow::Result<()> {
        self.batch_upsert_structs(structs).await
    }

    async fn batch_upsert_traits(&self, traits: &[TraitNode]) -> anyhow::Result<()> {
        self.batch_upsert_traits(traits).await
    }

    async fn batch_upsert_enums(&self, enums: &[EnumNode]) -> anyhow::Result<()> {
        self.batch_upsert_enums(enums).await
    }

    async fn batch_upsert_impls(&self, impls: &[ImplNode]) -> anyhow::Result<()> {
        self.batch_upsert_impls(impls).await
    }

    async fn batch_upsert_imports(&self, imports: &[ImportNode]) -> anyhow::Result<()> {
        self.batch_upsert_imports(imports).await
    }

    async fn batch_create_import_relationships(
        &self,
        relationships: &[(String, String, String)],
    ) -> anyhow::Result<()> {
        self.batch_create_import_relationships(relationships).await
    }

    async fn batch_create_imports_symbol_relationships(
        &self,
        relationships: &[(String, String, Option<uuid::Uuid>)],
    ) -> anyhow::Result<()> {
        self.batch_create_imports_symbol_relationships(relationships)
            .await
    }

    async fn batch_create_call_relationships(
        &self,
        calls: &[crate::parser::FunctionCall],
        project_id: Option<uuid::Uuid>,
    ) -> anyhow::Result<()> {
        self.batch_create_call_relationships(calls, project_id)
            .await
    }

    async fn batch_create_extends_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> anyhow::Result<()> {
        self.batch_create_extends_relationships(rels).await
    }

    async fn batch_create_implements_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> anyhow::Result<()> {
        self.batch_create_implements_relationships(rels).await
    }

    async fn cleanup_cross_project_calls(&self) -> anyhow::Result<i64> {
        self.cleanup_cross_project_calls().await
    }

    async fn cleanup_builtin_calls(&self) -> anyhow::Result<i64> {
        self.cleanup_builtin_calls().await
    }

    async fn migrate_calls_confidence(&self) -> anyhow::Result<i64> {
        self.migrate_calls_confidence().await
    }

    async fn cleanup_sync_data(&self) -> anyhow::Result<i64> {
        self.cleanup_sync_data().await
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
    // Heritage navigation queries
    // ========================================================================

    async fn get_class_hierarchy(
        &self,
        type_name: &str,
        max_depth: u32,
    ) -> anyhow::Result<serde_json::Value> {
        self.get_class_hierarchy(type_name, max_depth).await
    }

    async fn find_subclasses(&self, class_name: &str) -> anyhow::Result<Vec<serde_json::Value>> {
        self.find_subclasses(class_name).await
    }

    async fn find_interface_implementors(
        &self,
        interface_name: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        self.find_interface_implementors(interface_name).await
    }

    // ========================================================================
    // Process queries
    // ========================================================================

    async fn list_processes(
        &self,
        project_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        self.list_processes(project_id).await
    }

    async fn get_process_detail(
        &self,
        process_id: &str,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        self.get_process_detail(process_id).await
    }

    async fn get_entry_points(
        &self,
        project_id: uuid::Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        self.get_entry_points(project_id, limit).await
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

    async fn get_file_structs_summary(&self, path: &str) -> anyhow::Result<Vec<StructSummaryNode>> {
        self.get_file_structs_summary(path).await
    }

    async fn get_file_import_paths_list(&self, path: &str) -> anyhow::Result<Vec<String>> {
        self.get_file_import_paths_list(path).await
    }

    async fn find_symbol_references(
        &self,
        symbol: &str,
        limit: usize,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<SymbolReferenceNode>> {
        self.find_symbol_references(symbol, limit, project_id).await
    }

    async fn get_file_direct_imports(&self, path: &str) -> anyhow::Result<Vec<FileImportNode>> {
        self.get_file_direct_imports(path).await
    }

    async fn get_function_callers_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<String>> {
        self.get_function_callers_by_name(function_name, depth, project_id)
            .await
    }

    async fn get_function_callees_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<String>> {
        self.get_function_callees_by_name(function_name, depth, project_id)
            .await
    }

    async fn get_callers_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<(String, String, f64, String)>> {
        self.get_callers_with_confidence(function_name, project_id)
            .await
    }

    async fn get_callees_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<(String, String, f64, String)>> {
        self.get_callees_with_confidence(function_name, project_id)
            .await
    }

    async fn get_language_stats(&self) -> anyhow::Result<Vec<LanguageStatsNode>> {
        self.get_language_stats().await
    }

    async fn get_language_stats_for_project(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<LanguageStatsNode>> {
        self.get_language_stats_for_project(project_id).await
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

    async fn get_most_connected_files_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<ConnectedFileNode>> {
        self.get_most_connected_files_for_project(project_id, limit)
            .await
    }

    async fn get_project_communities(&self, project_id: Uuid) -> anyhow::Result<Vec<CommunityRow>> {
        self.get_project_communities(project_id).await
    }

    async fn get_node_analytics(
        &self,
        identifier: &str,
        node_type: &str,
    ) -> anyhow::Result<Option<NodeAnalyticsRow>> {
        self.get_node_analytics(identifier, node_type).await
    }

    async fn get_affected_communities(&self, file_paths: &[String]) -> anyhow::Result<Vec<String>> {
        self.get_affected_communities(file_paths).await
    }

    async fn get_code_health_report(
        &self,
        project_id: Uuid,
        god_function_threshold: usize,
    ) -> anyhow::Result<crate::neo4j::models::CodeHealthReport> {
        self.get_code_health_report(project_id, god_function_threshold)
            .await
    }

    async fn compute_maintenance_snapshot(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::models::MaintenanceSnapshot> {
        self.compute_maintenance_snapshot(project_id).await
    }

    async fn compute_scaffolding_level(
        &self,
        project_id: Uuid,
        scaffolding_override: Option<u8>,
    ) -> anyhow::Result<crate::neo4j::models::ScaffoldingLevel> {
        self.compute_scaffolding_level(project_id, scaffolding_override)
            .await
    }

    async fn set_scaffolding_override(
        &self,
        project_id: Uuid,
        level: Option<u8>,
    ) -> anyhow::Result<()> {
        self.set_scaffolding_override(project_id, level).await
    }

    async fn set_default_note_energy(
        &self,
        project_id: Uuid,
        energy: Option<f64>,
    ) -> anyhow::Result<()> {
        self.set_default_note_energy(project_id, energy).await
    }

    async fn detect_global_stagnation(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::models::StagnationReport> {
        self.detect_global_stagnation(project_id).await
    }

    async fn get_circular_dependencies(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<Vec<String>>> {
        self.get_circular_dependencies(project_id).await
    }

    async fn get_node_gds_metrics(
        &self,
        node_path: &str,
        node_type: &str,
        project_id: Uuid,
    ) -> anyhow::Result<Option<NodeGdsMetrics>> {
        self.get_node_gds_metrics(node_path, node_type, project_id)
            .await
    }

    async fn get_project_percentiles(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<ProjectPercentiles> {
        self.get_project_percentiles(project_id).await
    }

    async fn get_top_bridges_by_betweenness(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<BridgeFile>> {
        self.get_top_bridges_by_betweenness(project_id, limit).await
    }

    async fn get_file_symbol_names(&self, path: &str) -> anyhow::Result<FileSymbolNamesNode> {
        self.get_file_symbol_names(path).await
    }

    async fn get_function_caller_count(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<i64> {
        self.get_function_caller_count(function_name, project_id)
            .await
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

    async fn count_project_plans(&self, project_id: Uuid) -> anyhow::Result<i64> {
        self.count_project_plans(project_id).await
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

    async fn update_plan(&self, id: Uuid, updates: &UpdatePlanRequest) -> anyhow::Result<()> {
        self.update_plan(id, updates).await
    }

    async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> anyhow::Result<()> {
        self.update_plan_status(id, status).await
    }

    async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> anyhow::Result<()> {
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

    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> anyhow::Result<()> {
        self.update_task_status(task_id, status).await
    }

    async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> anyhow::Result<()> {
        self.assign_task(task_id, agent_id).await
    }

    async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> anyhow::Result<()> {
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

    async fn get_task_enrichment_counts(
        &self,
        task_ids: &[String],
    ) -> anyhow::Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentCounts>>
    {
        self.get_task_enrichment_counts(task_ids).await
    }

    async fn get_task_enrichment_data(
        &self,
        task_ids: &[String],
    ) -> anyhow::Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentData>>
    {
        self.get_task_enrichment_data(task_ids).await
    }

    async fn get_plan_critical_path(&self, plan_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_plan_critical_path(plan_id).await
    }

    async fn compute_waves(
        &self,
        plan_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::plan::WaveComputationResult> {
        self.compute_waves(plan_id).await
    }

    async fn get_next_available_task(&self, plan_id: Uuid) -> anyhow::Result<Option<TaskNode>> {
        self.get_next_available_task(plan_id).await
    }

    async fn get_task(&self, task_id: Uuid) -> anyhow::Result<Option<TaskNode>> {
        self.get_task(task_id).await
    }

    async fn update_task(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> anyhow::Result<()> {
        self.update_task(task_id, updates).await
    }

    async fn update_task_enrichment(
        &self,
        task_id: Uuid,
        execution_context: Option<&str>,
        persona: Option<&str>,
        prompt_cache: Option<&str>,
    ) -> anyhow::Result<()> {
        self.update_task_enrichment(task_id, execution_context, persona, prompt_cache)
            .await
    }

    async fn delete_task(&self, task_id: Uuid) -> anyhow::Result<()> {
        self.delete_task(task_id).await
    }

    async fn get_project_for_task(&self, task_id: Uuid) -> anyhow::Result<Option<ProjectNode>> {
        self.get_project_for_task(task_id).await
    }

    async fn get_plan_id_for_task(&self, task_id: Uuid) -> anyhow::Result<Option<Uuid>> {
        self.get_plan_id_for_task(task_id).await
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

    async fn update_step(&self, step_id: Uuid, updates: &UpdateStepRequest) -> anyhow::Result<()> {
        self.update_step(step_id, updates).await
    }

    async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> anyhow::Result<()> {
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

    async fn get_plan_constraints(&self, plan_id: Uuid) -> anyhow::Result<Vec<ConstraintNode>> {
        self.get_plan_constraints(plan_id).await
    }

    async fn get_constraint(&self, constraint_id: Uuid) -> anyhow::Result<Option<ConstraintNode>> {
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

    async fn create_decision(&self, task_id: Uuid, decision: &DecisionNode) -> anyhow::Result<()> {
        self.create_decision(task_id, decision).await
    }

    async fn get_decision(&self, decision_id: Uuid) -> anyhow::Result<Option<DecisionNode>> {
        self.get_decision(decision_id).await
    }

    async fn get_decision_project_id(&self, decision_id: Uuid) -> anyhow::Result<Option<String>> {
        self.get_decision_project_id(decision_id).await
    }

    async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
        status: Option<DecisionStatus>,
    ) -> anyhow::Result<()> {
        self.update_decision(decision_id, description, rationale, chosen_option, status)
            .await
    }

    async fn delete_decision(&self, decision_id: Uuid) -> anyhow::Result<()> {
        self.delete_decision(decision_id).await
    }

    async fn get_decisions_for_entity(
        &self,
        entity_type: &str,
        entity_id: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<DecisionNode>> {
        self.get_decisions_for_entity(entity_type, entity_id, limit)
            .await
    }

    async fn set_decision_embedding(
        &self,
        decision_id: Uuid,
        embedding: &[f32],
        model: &str,
    ) -> anyhow::Result<()> {
        self.set_decision_embedding(decision_id, embedding, model)
            .await
    }

    async fn get_decision_embedding(&self, decision_id: Uuid) -> anyhow::Result<Option<Vec<f32>>> {
        self.get_decision_embedding(decision_id).await
    }

    async fn get_all_decisions_with_task_id(&self) -> anyhow::Result<Vec<(DecisionNode, Uuid)>> {
        self.get_all_decisions_with_task_id().await
    }

    async fn get_decisions_without_embedding(&self) -> anyhow::Result<Vec<(Uuid, String, String)>> {
        self.get_decisions_without_embedding().await
    }

    async fn search_decisions_by_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_id: Option<&str>,
    ) -> anyhow::Result<Vec<(DecisionNode, f64)>> {
        self.search_decisions_by_vector(query_embedding, limit, project_id)
            .await
    }

    async fn get_decisions_affecting(
        &self,
        entity_type: &str,
        entity_id: &str,
        status_filter: Option<&str>,
    ) -> anyhow::Result<Vec<DecisionNode>> {
        self.get_decisions_affecting(entity_type, entity_id, status_filter)
            .await
    }

    async fn add_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        impact_description: Option<&str>,
    ) -> anyhow::Result<()> {
        self.add_decision_affects(decision_id, entity_type, entity_id, impact_description)
            .await
    }

    async fn remove_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> anyhow::Result<()> {
        self.remove_decision_affects(decision_id, entity_type, entity_id)
            .await
    }

    async fn list_decision_affects(
        &self,
        decision_id: Uuid,
    ) -> anyhow::Result<Vec<AffectsRelation>> {
        self.list_decision_affects(decision_id).await
    }

    async fn supersede_decision(
        &self,
        new_decision_id: Uuid,
        old_decision_id: Uuid,
    ) -> anyhow::Result<()> {
        self.supersede_decision(new_decision_id, old_decision_id)
            .await
    }

    async fn get_decision_timeline(
        &self,
        task_id: Option<Uuid>,
        from: Option<&str>,
        to: Option<&str>,
    ) -> anyhow::Result<Vec<DecisionTimelineEntry>> {
        self.get_decision_timeline(task_id, from, to).await
    }

    // ========================================================================
    // Dependency analysis
    // ========================================================================

    async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<String>> {
        self.find_dependent_files(file_path, depth, project_id)
            .await
    }

    async fn find_impacted_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<String>> {
        self.find_impacted_files(file_path, depth, project_id).await
    }

    async fn find_callers(
        &self,
        function_id: &str,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<FunctionNode>> {
        self.find_callers(function_id, project_id).await
    }

    // ========================================================================
    // Task-file linking
    // ========================================================================

    async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> anyhow::Result<()> {
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

    async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> anyhow::Result<()> {
        self.link_commit_to_task(commit_hash, task_id).await
    }

    async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> anyhow::Result<()> {
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
    // TOUCHES operations (Commit → File)
    // ========================================================================

    async fn create_commit_touches(
        &self,
        commit_hash: &str,
        files: &[FileChangedInfo],
    ) -> anyhow::Result<()> {
        self.create_commit_touches(commit_hash, files).await
    }

    async fn get_commit_files(&self, commit_hash: &str) -> anyhow::Result<Vec<CommitFileInfo>> {
        self.get_commit_files(commit_hash).await
    }

    async fn get_file_history(
        &self,
        file_path: &str,
        limit: Option<i64>,
    ) -> anyhow::Result<Vec<FileHistoryEntry>> {
        self.get_file_history(file_path, limit).await
    }

    async fn ping_freshness_for_files(&self, file_paths: &[String]) -> anyhow::Result<usize> {
        self.ping_freshness_for_files(file_paths).await
    }

    // ========================================================================
    // CO_CHANGED operations (File ↔ File)
    // ========================================================================

    async fn compute_co_changed(
        &self,
        project_id: Uuid,
        since: Option<chrono::DateTime<chrono::Utc>>,
        min_count: i64,
        max_relations: i64,
    ) -> anyhow::Result<i64> {
        self.compute_co_changed(project_id, since, min_count, max_relations)
            .await
    }

    async fn update_project_co_change_timestamp(&self, id: Uuid) -> anyhow::Result<()> {
        self.update_project_co_change_timestamp(id).await
    }

    async fn get_co_change_graph(
        &self,
        project_id: Uuid,
        min_count: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<CoChangePair>> {
        self.get_co_change_graph(project_id, min_count, limit).await
    }

    async fn get_file_co_changers(
        &self,
        file_path: &str,
        min_count: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<CoChanger>> {
        self.get_file_co_changers(file_path, min_count, limit).await
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

    async fn list_project_releases(&self, project_id: Uuid) -> anyhow::Result<Vec<ReleaseNode>> {
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

    async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> anyhow::Result<()> {
        self.add_task_to_release(release_id, task_id).await
    }

    async fn add_commit_to_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> anyhow::Result<()> {
        self.add_commit_to_release(release_id, commit_hash).await
    }

    async fn remove_commit_from_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> anyhow::Result<()> {
        self.remove_commit_from_release(release_id, commit_hash)
            .await
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

    async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> anyhow::Result<()> {
        self.add_task_to_milestone(milestone_id, task_id).await
    }

    async fn link_plan_to_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> anyhow::Result<()> {
        self.link_plan_to_milestone(plan_id, milestone_id).await
    }

    async fn unlink_plan_from_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> anyhow::Result<()> {
        self.unlink_plan_from_milestone(plan_id, milestone_id).await
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
    ) -> anyhow::Result<(u32, u32, u32, u32)> {
        self.get_milestone_progress(milestone_id).await
    }

    async fn get_milestone_tasks_with_plans(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<Vec<TaskWithPlan>> {
        self.get_milestone_tasks_with_plans(milestone_id).await
    }

    async fn get_milestone_steps_batch(
        &self,
        milestone_id: Uuid,
    ) -> anyhow::Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        self.get_milestone_steps_batch(milestone_id).await
    }

    async fn delete_milestone(&self, milestone_id: Uuid) -> anyhow::Result<()> {
        self.delete_milestone(milestone_id).await
    }

    async fn get_milestone_tasks(&self, milestone_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_milestone_tasks(milestone_id).await
    }

    async fn get_release_tasks(&self, release_id: Uuid) -> anyhow::Result<Vec<TaskNode>> {
        self.get_release_tasks(release_id).await
    }

    // ========================================================================
    // Project stats
    // ========================================================================

    async fn get_project_progress(&self, project_id: Uuid) -> anyhow::Result<(u32, u32, u32, u32)> {
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
        workspace_slug: Option<&str>,
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
            workspace_slug,
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
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
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
            project_id,
            workspace_slug,
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
        workspace_slug: Option<&str>,
        filters: &NoteFilters,
    ) -> anyhow::Result<(Vec<Note>, usize)> {
        self.list_notes(project_id, workspace_slug, filters).await
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

    async fn propagate_structural_links(&self, project_id: Uuid) -> anyhow::Result<usize> {
        self.propagate_structural_links(project_id).await
    }

    async fn propagate_high_level_links(&self, project_id: Uuid) -> anyhow::Result<usize> {
        self.propagate_high_level_links(project_id).await
    }

    async fn propagate_note_via_feature_graph(
        &self,
        note_id: Uuid,
        feature_graph_id: &str,
    ) -> anyhow::Result<usize> {
        self.propagate_note_via_feature_graph(note_id, feature_graph_id)
            .await
    }

    async fn propagate_note_via_skill(
        &self,
        note_id: Uuid,
        skill_id: &str,
    ) -> anyhow::Result<usize> {
        self.propagate_note_via_skill(note_id, skill_id).await
    }

    async fn propagate_note_via_protocol(
        &self,
        note_id: Uuid,
        protocol_id: &str,
    ) -> anyhow::Result<usize> {
        self.propagate_note_via_protocol(note_id, protocol_id).await
    }

    async fn propagate_semantic_links(
        &self,
        project_id: Uuid,
        min_similarity: f64,
    ) -> anyhow::Result<usize> {
        self.propagate_semantic_links(project_id, min_similarity)
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
        relation_types: Option<&[String]>,
        source_project_id: Option<Uuid>,
        force_cross_project: bool,
    ) -> anyhow::Result<Vec<PropagatedNote>> {
        self.get_propagated_notes(
            entity_type,
            entity_id,
            max_depth,
            min_score,
            relation_types,
            source_project_id,
            force_cross_project,
        )
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

    async fn supersede_note(&self, old_note_id: Uuid, new_note_id: Uuid) -> anyhow::Result<()> {
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

    async fn set_note_embedding(
        &self,
        note_id: Uuid,
        embedding: &[f32],
        model: &str,
    ) -> anyhow::Result<()> {
        self.set_note_embedding(note_id, embedding, model).await
    }

    async fn get_note_embedding(&self, note_id: Uuid) -> anyhow::Result<Option<Vec<f32>>> {
        self.get_note_embedding(note_id).await
    }

    async fn vector_search_notes(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        min_similarity: Option<f64>,
    ) -> anyhow::Result<Vec<(Note, f64)>> {
        self.vector_search_notes(embedding, limit, project_id, workspace_slug, min_similarity)
            .await
    }

    async fn list_notes_without_embedding(
        &self,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<Note>, usize)> {
        self.list_notes_without_embedding(limit, offset).await
    }

    // ========================================================================
    // Code embedding operations (File & Function vector search)
    // ========================================================================

    async fn set_file_embedding(
        &self,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> anyhow::Result<()> {
        self.set_file_embedding(file_path, embedding, model).await
    }

    async fn set_function_embedding(
        &self,
        function_name: &str,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> anyhow::Result<()> {
        self.set_function_embedding(function_name, file_path, embedding, model)
            .await
    }

    async fn vector_search_files(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        self.vector_search_files(embedding, limit, project_id).await
    }

    async fn vector_search_functions(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<(String, String, f64)>> {
        self.vector_search_functions(embedding, limit, project_id)
            .await
    }

    // ========================================================================
    // Synapse operations (Phase 2 — Neural Network)
    // ========================================================================

    async fn create_synapses(
        &self,
        note_id: Uuid,
        neighbors: &[(Uuid, f64)],
    ) -> anyhow::Result<usize> {
        self.create_synapses(note_id, neighbors).await
    }

    async fn get_synapses(&self, note_id: Uuid) -> anyhow::Result<Vec<(Uuid, f64)>> {
        self.get_synapses(note_id).await
    }

    async fn delete_synapses(&self, note_id: Uuid) -> anyhow::Result<usize> {
        self.delete_synapses(note_id).await
    }

    async fn get_project_note_entity_links(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String, String)>> {
        self.get_project_note_entity_links(project_id).await
    }

    async fn get_project_note_synapses(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> anyhow::Result<Vec<(String, String, f64)>> {
        self.get_project_note_synapses(project_id, min_weight).await
    }

    async fn get_project_decisions_for_graph(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(DecisionNode, Vec<AffectsRelation>)>> {
        self.get_project_decisions_for_graph(project_id).await
    }

    // ========================================================================
    // Energy operations (Phase 2 — Neural Network)
    // ========================================================================

    async fn update_energy_scores(&self, half_life_days: f64) -> anyhow::Result<usize> {
        self.update_energy_scores(half_life_days).await
    }

    async fn boost_energy(&self, note_id: Uuid, amount: f64) -> anyhow::Result<()> {
        self.boost_energy(note_id, amount).await
    }

    async fn track_reactivation(&self, note_ids: &[Uuid]) -> anyhow::Result<usize> {
        self.track_reactivation(note_ids).await
    }

    async fn reinforce_synapses(&self, note_ids: &[Uuid], boost: f64) -> anyhow::Result<usize> {
        self.reinforce_synapses(note_ids, boost).await
    }

    async fn decay_synapses(
        &self,
        decay_amount: f64,
        prune_threshold: f64,
    ) -> anyhow::Result<(usize, usize)> {
        self.decay_synapses(decay_amount, prune_threshold).await
    }

    async fn apply_scars(&self, node_ids: &[Uuid], increment: f64) -> anyhow::Result<usize> {
        self.apply_scars(node_ids, increment).await
    }

    async fn heal_scars(&self, node_id: Uuid) -> anyhow::Result<bool> {
        self.heal_scars(node_id).await
    }

    async fn consolidate_memory(&self) -> anyhow::Result<(usize, usize)> {
        self.consolidate_memory().await
    }

    async fn compute_homeostasis(
        &self,
        project_id: Uuid,
        custom_ranges: Option<&[(String, f64, f64)]>,
    ) -> anyhow::Result<crate::neo4j::models::HomeostasisReport> {
        self.compute_homeostasis(project_id, custom_ranges).await
    }

    async fn compute_structural_drift(
        &self,
        project_id: Uuid,
        warning_threshold: Option<f64>,
        critical_threshold: Option<f64>,
    ) -> anyhow::Result<crate::neo4j::models::StructuralDriftReport> {
        self.compute_structural_drift(project_id, warning_threshold, critical_threshold)
            .await
    }

    async fn increment_frustration(&self, task_id: Uuid, delta: f64) -> anyhow::Result<f64> {
        self.increment_frustration(task_id, delta).await
    }

    async fn decrement_frustration(&self, task_id: Uuid, delta: f64) -> anyhow::Result<f64> {
        self.decrement_frustration(task_id, delta).await
    }

    async fn get_frustration(&self, task_id: Uuid) -> anyhow::Result<f64> {
        self.get_frustration(task_id).await
    }

    async fn get_step_parent_task_id(&self, step_id: Uuid) -> anyhow::Result<Option<Uuid>> {
        self.get_step_parent_task_id(step_id).await
    }

    async fn init_note_energy(&self) -> anyhow::Result<usize> {
        self.init_note_energy().await
    }

    async fn list_notes_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<crate::notes::Note>, usize)> {
        self.list_notes_needing_synapses(limit, offset).await
    }

    async fn create_cross_entity_synapses(
        &self,
        source_id: Uuid,
        neighbors: &[(Uuid, f64)],
    ) -> anyhow::Result<usize> {
        self.create_cross_entity_synapses(source_id, neighbors)
            .await
    }

    async fn get_cross_entity_synapses(
        &self,
        node_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, f64, String)>> {
        self.get_cross_entity_synapses(node_id).await
    }

    async fn list_decisions_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<DecisionNode>, usize)> {
        self.list_decisions_needing_synapses(limit, offset).await
    }

    async fn get_all_synapse_weights(&self, project_id: Option<Uuid>) -> anyhow::Result<Vec<f64>> {
        self.get_all_synapse_weights(project_id).await
    }

    // ========================================================================
    // Chat session operations
    // ========================================================================

    async fn create_chat_session(&self, session: &ChatSessionNode) -> anyhow::Result<()> {
        self.create_chat_session(session).await
    }

    async fn get_chat_session(&self, id: Uuid) -> anyhow::Result<Option<ChatSessionNode>> {
        self.get_chat_session(id).await
    }

    async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
        include_detached: bool,
    ) -> anyhow::Result<(Vec<ChatSessionNode>, usize)> {
        self.list_chat_sessions(
            project_slug,
            workspace_slug,
            limit,
            offset,
            include_detached,
        )
        .await
    }

    async fn get_session_children(&self, parent_id: Uuid) -> anyhow::Result<Vec<ChatSessionNode>> {
        self.get_session_children(parent_id).await
    }

    async fn create_spawned_by_relation(
        &self,
        child_session_id: &str,
        parent_session_id: &str,
        spawn_type: &str,
        run_id: Option<Uuid>,
        task_id: Option<Uuid>,
    ) -> anyhow::Result<()> {
        self.create_spawned_by_relation(
            child_session_id,
            parent_session_id,
            spawn_type,
            run_id,
            task_id,
        )
        .await
    }

    async fn get_session_tree(&self, session_id: &str) -> anyhow::Result<Vec<SessionTreeNode>> {
        self.get_session_tree(session_id).await
    }

    async fn get_session_root(&self, session_id: &str) -> anyhow::Result<Option<String>> {
        self.get_session_root(session_id).await
    }

    async fn get_run_sessions(&self, run_id: Uuid) -> anyhow::Result<Vec<SessionInfo>> {
        self.get_run_sessions(run_id).await
    }

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
    ) -> anyhow::Result<Option<ChatSessionNode>> {
        self.update_chat_session(
            id,
            cli_session_id,
            title,
            message_count,
            total_cost_usd,
            conversation_id,
            preview,
        )
        .await
    }

    async fn update_chat_session_permission_mode(
        &self,
        id: Uuid,
        mode: &str,
    ) -> anyhow::Result<()> {
        self.update_chat_session_permission_mode(id, mode).await
    }

    async fn set_session_auto_continue(&self, id: Uuid, enabled: bool) -> anyhow::Result<()> {
        self.set_session_auto_continue(id, enabled).await
    }

    async fn get_session_auto_continue(&self, id: Uuid) -> anyhow::Result<bool> {
        self.get_session_auto_continue(id).await
    }

    async fn backfill_chat_session_previews(&self) -> anyhow::Result<usize> {
        self.backfill_chat_session_previews().await
    }

    async fn delete_chat_session(&self, id: Uuid) -> anyhow::Result<bool> {
        self.delete_chat_session(id).await
    }

    // Chat event operations

    async fn store_chat_events(
        &self,
        session_id: Uuid,
        events: Vec<ChatEventRecord>,
    ) -> anyhow::Result<()> {
        self.store_chat_events(session_id, events).await
    }

    async fn get_chat_events(
        &self,
        session_id: Uuid,
        after_seq: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<ChatEventRecord>> {
        self.get_chat_events(session_id, after_seq, limit).await
    }

    async fn get_chat_events_paginated(
        &self,
        session_id: Uuid,
        offset: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<ChatEventRecord>> {
        self.get_chat_events_paginated(session_id, offset, limit)
            .await
    }

    async fn count_chat_events(&self, session_id: Uuid) -> anyhow::Result<i64> {
        self.count_chat_events(session_id).await
    }

    async fn get_latest_chat_event_seq(&self, session_id: Uuid) -> anyhow::Result<i64> {
        self.get_latest_chat_event_seq(session_id).await
    }

    async fn delete_chat_events(&self, session_id: Uuid) -> anyhow::Result<()> {
        self.delete_chat_events(session_id).await
    }

    // ========================================================================
    // Chat DISCUSSED relations
    // ========================================================================

    async fn add_discussed(
        &self,
        session_id: Uuid,
        entities: &[(String, String)],
    ) -> anyhow::Result<usize> {
        self.add_discussed(session_id, entities).await
    }

    async fn get_session_entities(
        &self,
        session_id: Uuid,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<DiscussedEntity>> {
        self.get_session_entities(session_id, project_id).await
    }

    async fn get_discussed_co_changers(
        &self,
        project_id: Uuid,
        max_sessions: i64,
        max_results: i64,
    ) -> anyhow::Result<Vec<CoChanger>> {
        self.get_discussed_co_changers(project_id, max_sessions, max_results)
            .await
    }

    async fn backfill_discussed(&self) -> anyhow::Result<(usize, usize, usize)> {
        self.backfill_discussed().await
    }

    // ========================================================================
    // Graph visualization queries (PM + Chat layers)
    // ========================================================================

    async fn get_pm_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<(Vec<PmGraphNode>, Vec<PmGraphEdge>)> {
        self.get_pm_graph_data(project_id, limit).await
    }

    async fn get_chat_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<(Vec<ChatGraphSession>, Vec<ChatGraphDiscussed>)> {
        self.get_chat_graph_data(project_id, limit).await
    }

    // ========================================================================
    // User / Auth operations
    // ========================================================================

    async fn upsert_user(&self, user: &UserNode) -> anyhow::Result<UserNode> {
        self.upsert_user(user).await
    }

    async fn get_user_by_id(&self, id: Uuid) -> anyhow::Result<Option<UserNode>> {
        self.get_user_by_id(id).await
    }

    async fn get_user_by_provider_id(
        &self,
        provider: &str,
        external_id: &str,
    ) -> anyhow::Result<Option<UserNode>> {
        self.get_user_by_provider_id(provider, external_id).await
    }

    async fn get_user_by_email_and_provider(
        &self,
        email: &str,
        provider: &str,
    ) -> anyhow::Result<Option<UserNode>> {
        self.get_user_by_email_and_provider(email, provider).await
    }

    async fn get_user_by_email(&self, email: &str) -> anyhow::Result<Option<UserNode>> {
        self.get_user_by_email(email).await
    }

    async fn create_password_user(
        &self,
        email: &str,
        name: &str,
        password_hash: &str,
    ) -> anyhow::Result<UserNode> {
        self.create_password_user(email, name, password_hash).await
    }

    async fn list_users(&self) -> anyhow::Result<Vec<UserNode>> {
        self.list_users().await
    }

    // Refresh Tokens
    async fn create_refresh_token(
        &self,
        user_id: Uuid,
        token_hash: &str,
        expires_at: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<()> {
        self.create_refresh_token(user_id, token_hash, expires_at)
            .await
    }

    async fn validate_refresh_token(
        &self,
        token_hash: &str,
    ) -> anyhow::Result<Option<crate::neo4j::models::RefreshTokenNode>> {
        self.validate_refresh_token(token_hash).await
    }

    async fn revoke_refresh_token(&self, token_hash: &str) -> anyhow::Result<bool> {
        self.revoke_refresh_token(token_hash).await
    }

    async fn revoke_all_user_tokens(&self, user_id: Uuid) -> anyhow::Result<u64> {
        self.revoke_all_user_tokens(user_id).await
    }

    // Feature Graphs
    async fn create_feature_graph(&self, graph: &FeatureGraphNode) -> anyhow::Result<()> {
        self.create_feature_graph(graph).await
    }
    async fn get_feature_graph(&self, id: Uuid) -> anyhow::Result<Option<FeatureGraphNode>> {
        self.get_feature_graph(id).await
    }
    async fn get_feature_graph_detail(
        &self,
        id: Uuid,
    ) -> anyhow::Result<Option<FeatureGraphDetail>> {
        self.get_feature_graph_detail(id).await
    }
    async fn list_feature_graphs(
        &self,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<Vec<FeatureGraphNode>> {
        self.list_feature_graphs(project_id).await
    }
    async fn delete_feature_graph(&self, id: Uuid) -> anyhow::Result<bool> {
        self.delete_feature_graph(id).await
    }
    async fn add_entity_to_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        role: Option<&str>,
        project_id: Option<Uuid>,
    ) -> anyhow::Result<()> {
        self.add_entity_to_feature_graph(feature_graph_id, entity_type, entity_id, role, project_id)
            .await
    }
    async fn remove_entity_from_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> anyhow::Result<bool> {
        self.remove_entity_from_feature_graph(feature_graph_id, entity_type, entity_id)
            .await
    }

    async fn auto_build_feature_graph(
        &self,
        name: &str,
        description: Option<&str>,
        project_id: Uuid,
        entry_function: &str,
        depth: u32,
        include_relations: Option<&[String]>,
        filter_community: Option<bool>,
    ) -> anyhow::Result<FeatureGraphDetail> {
        self.auto_build_feature_graph(
            name,
            description,
            project_id,
            entry_function,
            depth,
            include_relations,
            filter_community,
        )
        .await
    }

    async fn refresh_feature_graph(&self, id: Uuid) -> anyhow::Result<Option<FeatureGraphDetail>> {
        self.refresh_feature_graph(id).await
    }

    async fn get_top_entry_functions(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<String>> {
        self.get_top_entry_functions(project_id, limit).await
    }

    async fn get_feature_graph_statistics(
        &self,
        id: Uuid,
    ) -> anyhow::Result<Option<FeatureGraphStatistics>> {
        self.get_feature_graph_statistics(id).await
    }

    async fn compare_feature_graphs(
        &self,
        id_a: Uuid,
        id_b: Uuid,
    ) -> anyhow::Result<Option<FeatureGraphComparison>> {
        self.compare_feature_graphs(id_a, id_b).await
    }

    async fn find_overlapping_feature_graphs(
        &self,
        id: Uuid,
        min_overlap: f64,
    ) -> anyhow::Result<Vec<FeatureGraphOverlap>> {
        self.find_overlapping_feature_graphs(id, min_overlap).await
    }

    async fn get_project_import_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String)>> {
        self.get_project_import_edges(project_id).await
    }

    async fn get_project_call_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String)>> {
        self.get_project_call_edges(project_id).await
    }

    async fn get_project_extends_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String)>> {
        self.get_project_extends_edges(project_id).await
    }

    async fn get_project_implements_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String)>> {
        self.get_project_implements_edges(project_id).await
    }

    async fn batch_update_file_analytics(
        &self,
        updates: &[crate::graph::models::FileAnalyticsUpdate],
    ) -> anyhow::Result<()> {
        self.batch_update_file_analytics(updates).await
    }

    async fn batch_update_function_analytics(
        &self,
        updates: &[crate::graph::models::FunctionAnalyticsUpdate],
    ) -> anyhow::Result<()> {
        self.batch_update_function_analytics(updates).await
    }

    async fn batch_update_fabric_file_analytics(
        &self,
        updates: &[crate::graph::models::FabricFileAnalyticsUpdate],
    ) -> anyhow::Result<()> {
        self.batch_update_fabric_file_analytics(updates).await
    }

    async fn batch_update_structural_dna(
        &self,
        updates: &[crate::graph::models::StructuralDnaUpdate],
    ) -> anyhow::Result<()> {
        self.batch_update_structural_dna(updates).await
    }

    async fn write_predicted_links(
        &self,
        project_id: &str,
        links: &[crate::graph::models::LinkPrediction],
    ) -> anyhow::Result<()> {
        self.write_predicted_links(project_id, links).await
    }

    async fn get_project_structural_dna(
        &self,
        project_id: &str,
    ) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
        self.get_project_structural_dna(project_id).await
    }

    async fn batch_update_structural_fingerprints(
        &self,
        updates: &[crate::graph::models::StructuralFingerprintUpdate],
    ) -> anyhow::Result<()> {
        self.batch_update_structural_fingerprints(updates).await
    }

    async fn get_project_structural_fingerprints(
        &self,
        project_id: &str,
    ) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
        self.get_project_structural_fingerprints(project_id).await
    }

    async fn get_project_file_signals(
        &self,
        project_id: &str,
    ) -> anyhow::Result<Vec<crate::graph::models::FileSignalRecord>> {
        self.get_project_file_signals(project_id).await
    }

    async fn get_project_synapse_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String, f64)>> {
        self.get_project_synapse_edges(project_id).await
    }

    async fn get_neural_metrics(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::models::NeuralMetrics> {
        self.get_neural_metrics(project_id).await
    }

    // T5.5 — Churn score
    async fn compute_churn_scores(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::models::FileChurnScore>> {
        self.compute_churn_scores(project_id).await
    }

    async fn batch_update_churn_scores(
        &self,
        updates: &[crate::neo4j::models::FileChurnScore],
    ) -> anyhow::Result<()> {
        self.batch_update_churn_scores(updates).await
    }

    async fn get_top_hotspots(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::neo4j::models::FileChurnScore>> {
        self.get_top_hotspots(project_id, limit).await
    }

    // T5.6 — Knowledge density
    async fn compute_knowledge_density(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::models::FileKnowledgeDensity>> {
        self.compute_knowledge_density(project_id).await
    }

    async fn batch_update_knowledge_density(
        &self,
        updates: &[crate::neo4j::models::FileKnowledgeDensity],
    ) -> anyhow::Result<()> {
        self.batch_update_knowledge_density(updates).await
    }

    async fn get_top_knowledge_gaps(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::neo4j::models::FileKnowledgeDensity>> {
        self.get_top_knowledge_gaps(project_id, limit).await
    }

    // T5.7 — Risk score composite
    async fn compute_risk_scores(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::models::FileRiskScore>> {
        self.compute_risk_scores(project_id).await
    }

    async fn batch_update_risk_scores(
        &self,
        updates: &[crate::neo4j::models::FileRiskScore],
    ) -> anyhow::Result<()> {
        self.batch_update_risk_scores(updates).await
    }

    async fn get_risk_summary(&self, project_id: Uuid) -> anyhow::Result<serde_json::Value> {
        self.get_risk_summary(project_id).await
    }

    async fn batch_upsert_processes(&self, processes: &[ProcessNode]) -> anyhow::Result<()> {
        self.batch_upsert_processes(processes).await
    }

    async fn batch_create_step_relationships(
        &self,
        steps: &[(String, String, u32)],
    ) -> anyhow::Result<()> {
        self.batch_create_step_relationships(steps).await
    }

    async fn delete_project_processes(&self, project_id: Uuid) -> anyhow::Result<u64> {
        self.delete_project_processes(project_id).await
    }

    // ========================================================================
    // Persona operations (delegates to neo4j/persona.rs)
    // ========================================================================

    async fn create_persona(&self, persona: &PersonaNode) -> anyhow::Result<()> {
        self.create_persona(persona).await
    }

    async fn get_persona(&self, id: Uuid) -> anyhow::Result<Option<PersonaNode>> {
        self.get_persona(id).await
    }

    async fn update_persona(&self, persona: &PersonaNode) -> anyhow::Result<()> {
        self.update_persona(persona).await
    }

    async fn delete_persona(&self, id: Uuid) -> anyhow::Result<bool> {
        self.delete_persona(id).await
    }

    async fn list_personas(
        &self,
        project_id: Uuid,
        status: Option<PersonaStatus>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<PersonaNode>, usize)> {
        self.list_personas(project_id, status, limit, offset).await
    }

    async fn list_global_personas(&self) -> anyhow::Result<Vec<PersonaNode>> {
        self.list_global_personas().await
    }

    async fn add_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> anyhow::Result<()> {
        self.add_persona_skill(persona_id, skill_id).await
    }

    async fn remove_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> anyhow::Result<()> {
        self.remove_persona_skill(persona_id, skill_id).await
    }

    async fn add_persona_protocol(
        &self,
        persona_id: Uuid,
        protocol_id: Uuid,
    ) -> anyhow::Result<()> {
        self.add_persona_protocol(persona_id, protocol_id).await
    }

    async fn remove_persona_protocol(
        &self,
        persona_id: Uuid,
        protocol_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_persona_protocol(persona_id, protocol_id).await
    }

    async fn increment_persona_activation(&self, persona_id: Uuid) -> anyhow::Result<()> {
        self.increment_persona_activation(persona_id).await
    }

    async fn set_persona_feature_graph(
        &self,
        persona_id: Uuid,
        feature_graph_id: Uuid,
    ) -> anyhow::Result<()> {
        self.set_persona_feature_graph(persona_id, feature_graph_id)
            .await
    }

    async fn remove_persona_feature_graph(&self, persona_id: Uuid) -> anyhow::Result<()> {
        self.remove_persona_feature_graph(persona_id).await
    }

    async fn add_persona_file(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.add_persona_file(persona_id, file_path, weight).await
    }

    async fn remove_persona_file(&self, persona_id: Uuid, file_path: &str) -> anyhow::Result<()> {
        self.remove_persona_file(persona_id, file_path).await
    }

    async fn add_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.add_persona_function(persona_id, function_id, weight)
            .await
    }

    async fn remove_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
    ) -> anyhow::Result<()> {
        self.remove_persona_function(persona_id, function_id).await
    }

    async fn add_persona_note(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.add_persona_note(persona_id, note_id, weight).await
    }

    async fn remove_persona_note(&self, persona_id: Uuid, note_id: Uuid) -> anyhow::Result<()> {
        self.remove_persona_note(persona_id, note_id).await
    }

    async fn add_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.add_persona_decision(persona_id, decision_id, weight)
            .await
    }

    async fn remove_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
    ) -> anyhow::Result<()> {
        self.remove_persona_decision(persona_id, decision_id).await
    }

    async fn add_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> anyhow::Result<()> {
        self.add_persona_extends(child_id, parent_id).await
    }

    async fn remove_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> anyhow::Result<()> {
        self.remove_persona_extends(child_id, parent_id).await
    }

    async fn get_persona_subgraph(&self, persona_id: Uuid) -> anyhow::Result<PersonaSubgraph> {
        self.get_persona_subgraph(persona_id).await
    }

    async fn find_personas_for_file(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(PersonaNode, f64)>> {
        self.find_personas_for_file(file_path, project_id).await
    }

    async fn get_all_persona_knows(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(PersonaNode, String, f64)>> {
        self.get_all_persona_knows(project_id).await
    }

    async fn auto_scope_to_feature_graphs(&self, project_id: Uuid) -> anyhow::Result<usize> {
        self.auto_scope_to_feature_graphs(project_id).await
    }

    async fn compute_adaptive_thresholds(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::persona::AdaptivePersonaThresholds> {
        self.compute_adaptive_thresholds(project_id).await
    }

    async fn maintain_personas(&self, project_id: Uuid) -> anyhow::Result<(usize, usize, usize)> {
        self.maintain_personas(project_id).await
    }

    async fn detect_personas(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::persona::PersonaProposal>> {
        self.detect_personas(project_id).await
    }

    async fn find_adjacent_personas(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, String)>> {
        self.find_adjacent_personas(file_path, project_id).await
    }

    async fn auto_grow_file_knows(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.auto_grow_file_knows(persona_id, file_path, weight)
            .await
    }

    async fn find_relevant_personas_for_note(
        &self,
        file_paths: &[String],
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, f64)>> {
        self.find_relevant_personas_for_note(file_paths, project_id)
            .await
    }

    async fn find_relevant_personas_for_decision(
        &self,
        decision_id: Uuid,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, f64)>> {
        self.find_relevant_personas_for_decision(decision_id, project_id)
            .await
    }

    async fn auto_link_note_to_persona(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.auto_link_note_to_persona(persona_id, note_id, weight)
            .await
    }

    async fn auto_link_decision_to_persona(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.auto_link_decision_to_persona(persona_id, decision_id, weight)
            .await
    }

    async fn auto_link_file_to_persona(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> anyhow::Result<()> {
        self.auto_link_file_to_persona(persona_id, file_path, weight)
            .await
    }

    // ========================================================================
    // Persona learning methods
    // ========================================================================

    async fn propagate_knows_via_co_change(
        &self,
        persona_id: Uuid,
        file_path: &str,
        base_weight: f64,
    ) -> anyhow::Result<usize> {
        self.propagate_knows_via_co_change(persona_id, file_path, base_weight)
            .await
    }

    async fn compute_persona_affinity(
        &self,
        persona_a: Uuid,
        persona_b: Uuid,
    ) -> anyhow::Result<crate::neo4j::persona::PersonaAffinityScore> {
        self.compute_persona_affinity(persona_a, persona_b).await
    }

    async fn merge_personas(&self, keep_id: Uuid, merge_id: Uuid) -> anyhow::Result<()> {
        self.merge_personas(keep_id, merge_id).await
    }

    async fn find_synapse_linked_personas(
        &self,
        persona_id: Uuid,
    ) -> anyhow::Result<Vec<(Uuid, String, f64)>> {
        self.find_synapse_linked_personas(persona_id).await
    }

    async fn rate_limited_energy_boost(
        &self,
        persona_id: Uuid,
        boost: f64,
        max_per_cycle: f64,
    ) -> anyhow::Result<bool> {
        self.rate_limited_energy_boost(persona_id, boost, max_per_cycle)
            .await
    }

    async fn get_learning_health(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::analytics::LearningHealthReport> {
        self.get_learning_health(project_id).await
    }

    // ========================================================================
    // Analysis Profile operations (delegates to neo4j/profile.rs)
    // ========================================================================

    async fn create_analysis_profile(
        &self,
        profile: &crate::graph::models::AnalysisProfile,
    ) -> anyhow::Result<()> {
        self.create_analysis_profile(profile).await
    }

    async fn list_analysis_profiles(
        &self,
        project_id: Option<&str>,
    ) -> anyhow::Result<Vec<crate::graph::models::AnalysisProfile>> {
        self.list_analysis_profiles(project_id).await
    }

    async fn get_analysis_profile(
        &self,
        id: &str,
    ) -> anyhow::Result<Option<crate::graph::models::AnalysisProfile>> {
        self.get_analysis_profile(id).await
    }

    async fn delete_analysis_profile(&self, id: &str) -> anyhow::Result<()> {
        self.delete_analysis_profile(id).await
    }

    async fn get_knowledge_density(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> anyhow::Result<f64> {
        self.get_knowledge_density(file_path, project_id).await
    }

    async fn get_node_pagerank(&self, file_path: &str, project_id: &str) -> anyhow::Result<f64> {
        self.get_node_pagerank(file_path, project_id).await
    }

    async fn get_bridge_proximity(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        self.get_bridge_proximity(file_path, project_id).await
    }

    async fn find_bridge_subgraph(
        &self,
        source: &str,
        target: &str,
        max_hops: u32,
        relation_types: &[String],
        project_id: &str,
    ) -> anyhow::Result<(
        Vec<crate::graph::models::BridgeRawNode>,
        Vec<crate::graph::models::BridgeRawEdge>,
    )> {
        self.find_bridge_subgraph(source, target, max_hops, relation_types, project_id)
            .await
    }

    async fn get_avg_multi_signal_score(&self, project_id: Uuid) -> anyhow::Result<f64> {
        self.get_avg_multi_signal_score(project_id).await
    }

    // ========================================================================
    // Topology Firewall (delegates to neo4j/topology.rs)
    // ========================================================================

    async fn create_topology_rule(
        &self,
        rule: &crate::graph::models::TopologyRule,
    ) -> anyhow::Result<()> {
        self.create_topology_rule(rule).await
    }

    async fn list_topology_rules(
        &self,
        project_id: &str,
    ) -> anyhow::Result<Vec<crate::graph::models::TopologyRule>> {
        self.list_topology_rules(project_id).await
    }

    async fn delete_topology_rule(&self, rule_id: &str) -> anyhow::Result<()> {
        self.delete_topology_rule(rule_id).await
    }

    async fn check_topology_rules(
        &self,
        project_id: &str,
    ) -> anyhow::Result<Vec<crate::graph::models::TopologyViolation>> {
        self.check_topology_rules(project_id).await
    }

    async fn check_file_topology(
        &self,
        project_id: &str,
        file_path: &str,
        new_imports: &[String],
    ) -> anyhow::Result<Vec<crate::graph::models::TopologyViolation>> {
        self.check_file_topology(project_id, file_path, new_imports)
            .await
    }

    async fn health_check(&self) -> anyhow::Result<bool> {
        match self.execute("RETURN 1 AS ping").await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    // ========================================================================
    // Skill operations (delegates to neo4j/skill.rs)
    // ========================================================================

    async fn create_skill(&self, skill: &crate::skills::SkillNode) -> anyhow::Result<()> {
        self.create_skill(skill).await
    }

    async fn get_skill(&self, id: uuid::Uuid) -> anyhow::Result<Option<crate::skills::SkillNode>> {
        self.get_skill(id).await
    }

    async fn update_skill(&self, skill: &crate::skills::SkillNode) -> anyhow::Result<()> {
        self.update_skill(skill).await
    }

    async fn delete_skill(&self, id: uuid::Uuid) -> anyhow::Result<bool> {
        self.delete_skill(id).await
    }

    async fn list_skills(
        &self,
        project_id: uuid::Uuid,
        status: Option<crate::skills::SkillStatus>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<crate::skills::SkillNode>, usize)> {
        self.list_skills(project_id, status, limit, offset).await
    }

    async fn get_skill_members(
        &self,
        skill_id: uuid::Uuid,
    ) -> anyhow::Result<(
        Vec<crate::notes::Note>,
        Vec<crate::neo4j::models::DecisionNode>,
    )> {
        self.get_skill_members(skill_id).await
    }

    async fn add_skill_member(
        &self,
        skill_id: uuid::Uuid,
        entity_type: &str,
        entity_id: uuid::Uuid,
    ) -> anyhow::Result<()> {
        self.add_skill_member(skill_id, entity_type, entity_id)
            .await
    }

    async fn remove_skill_member(
        &self,
        skill_id: uuid::Uuid,
        entity_type: &str,
        entity_id: uuid::Uuid,
    ) -> anyhow::Result<bool> {
        self.remove_skill_member(skill_id, entity_type, entity_id)
            .await
    }

    async fn remove_all_skill_members(&self, skill_id: uuid::Uuid) -> anyhow::Result<i64> {
        self.remove_all_skill_members(skill_id).await
    }

    async fn get_skills_for_note(
        &self,
        note_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::skills::SkillNode>> {
        self.get_skills_for_note(note_id).await
    }

    async fn get_skills_for_project(
        &self,
        project_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::skills::SkillNode>> {
        self.get_skills_for_project(project_id).await
    }

    async fn activate_skill(
        &self,
        skill_id: uuid::Uuid,
        query: &str,
    ) -> anyhow::Result<crate::skills::ActivatedSkillContext> {
        self.activate_skill(skill_id, query).await
    }

    async fn increment_skill_activation(&self, skill_id: uuid::Uuid) -> anyhow::Result<()> {
        self.increment_skill_activation(skill_id).await
    }

    async fn match_skills_by_trigger(
        &self,
        project_id: uuid::Uuid,
        input: &str,
    ) -> anyhow::Result<Vec<(crate::skills::SkillNode, f64)>> {
        self.match_skills_by_trigger(project_id, input).await
    }

    async fn get_synapse_graph(
        &self,
        project_id: uuid::Uuid,
        min_weight: f64,
    ) -> anyhow::Result<Vec<(String, String, f64)>> {
        self.get_synapse_graph(project_id, min_weight).await
    }

    async fn batch_save_context_cards(
        &self,
        cards: &[crate::graph::models::ContextCard],
    ) -> anyhow::Result<()> {
        self.batch_save_context_cards(cards).await
    }

    async fn invalidate_context_cards(
        &self,
        paths: &[String],
        project_id: &str,
    ) -> anyhow::Result<()> {
        self.invalidate_context_cards(paths, project_id).await
    }

    async fn get_context_card(
        &self,
        path: &str,
        project_id: &str,
    ) -> anyhow::Result<Option<crate::graph::models::ContextCard>> {
        self.get_context_card(path, project_id).await
    }

    async fn get_context_cards_batch(
        &self,
        paths: &[String],
        project_id: &str,
    ) -> anyhow::Result<Vec<crate::graph::models::ContextCard>> {
        self.get_context_cards_batch(paths, project_id).await
    }

    async fn find_isomorphic_groups(
        &self,
        project_id: &str,
        min_group_size: usize,
    ) -> anyhow::Result<Vec<crate::graph::models::IsomorphicGroup>> {
        self.find_isomorphic_groups(project_id, min_group_size)
            .await
    }

    async fn has_context_cards(&self, project_id: &str) -> anyhow::Result<bool> {
        self.has_context_cards(project_id).await
    }

    async fn get_note_embeddings_for_project(
        &self,
        project_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<super::models::NoteEmbeddingPoint>> {
        self.get_note_embeddings_for_project(project_id).await
    }

    // ========================================================================
    // Protocol operations (Pattern Federation)
    // ========================================================================

    async fn upsert_protocol(&self, protocol: &crate::protocol::Protocol) -> anyhow::Result<()> {
        self.upsert_protocol(protocol).await
    }

    async fn get_protocol(
        &self,
        id: uuid::Uuid,
    ) -> anyhow::Result<Option<crate::protocol::Protocol>> {
        self.get_protocol(id).await
    }

    async fn list_protocols(
        &self,
        project_id: uuid::Uuid,
        category: Option<crate::protocol::ProtocolCategory>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<crate::protocol::Protocol>, usize)> {
        self.list_protocols(project_id, category, limit, offset)
            .await
    }

    async fn delete_protocol(&self, id: uuid::Uuid) -> anyhow::Result<bool> {
        self.delete_protocol(id).await
    }

    async fn upsert_protocol_state(
        &self,
        state: &crate::protocol::ProtocolState,
    ) -> anyhow::Result<()> {
        self.upsert_protocol_state(state).await
    }

    async fn get_protocol_states(
        &self,
        protocol_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::protocol::ProtocolState>> {
        self.get_protocol_states(protocol_id).await
    }

    async fn get_protocol_by_name_and_project(
        &self,
        name: &str,
        project_id: uuid::Uuid,
    ) -> anyhow::Result<Option<uuid::Uuid>> {
        self.get_protocol_by_name_and_project(name, project_id)
            .await
    }

    async fn delete_protocol_state(&self, state_id: uuid::Uuid) -> anyhow::Result<bool> {
        self.delete_protocol_state(state_id).await
    }

    async fn upsert_protocol_transition(
        &self,
        transition: &crate::protocol::ProtocolTransition,
    ) -> anyhow::Result<()> {
        self.upsert_protocol_transition(transition).await
    }

    async fn get_protocol_transitions(
        &self,
        protocol_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::protocol::ProtocolTransition>> {
        self.get_protocol_transitions(protocol_id).await
    }

    async fn delete_protocol_transition(&self, transition_id: uuid::Uuid) -> anyhow::Result<bool> {
        self.delete_protocol_transition(transition_id).await
    }

    // ========================================================================
    // ProtocolRun operations (FSM Runtime)
    // ========================================================================

    async fn create_protocol_run(&self, run: &crate::protocol::ProtocolRun) -> anyhow::Result<()> {
        self.create_protocol_run(run).await
    }

    async fn get_protocol_run(
        &self,
        run_id: uuid::Uuid,
    ) -> anyhow::Result<Option<crate::protocol::ProtocolRun>> {
        self.get_protocol_run(run_id).await
    }

    async fn update_protocol_run(
        &self,
        run: &mut crate::protocol::ProtocolRun,
    ) -> anyhow::Result<()> {
        self.update_protocol_run(run).await
    }

    async fn list_protocol_runs(
        &self,
        protocol_id: uuid::Uuid,
        status: Option<crate::protocol::RunStatus>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<crate::protocol::ProtocolRun>, usize)> {
        self.list_protocol_runs(protocol_id, status, limit, offset)
            .await
    }

    async fn list_child_runs(
        &self,
        parent_run_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::protocol::ProtocolRun>> {
        self.list_child_runs(parent_run_id).await
    }

    async fn count_child_runs(&self, parent_run_id: uuid::Uuid) -> anyhow::Result<usize> {
        self.count_child_runs(parent_run_id).await
    }

    async fn get_run_tree(
        &self,
        root_run_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::protocol::ProtocolRun>> {
        self.get_run_tree(root_run_id).await
    }

    async fn delete_protocol_run(&self, run_id: uuid::Uuid) -> anyhow::Result<bool> {
        self.delete_protocol_run(run_id).await
    }

    async fn create_produced_during(
        &self,
        entity_type: &str,
        entity_id: uuid::Uuid,
        run_id: uuid::Uuid,
    ) -> anyhow::Result<bool> {
        self.create_produced_during(entity_type, entity_id, run_id)
            .await
    }

    async fn get_run_outcomes(
        &self,
        run_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::protocol::ProducedArtefact>> {
        self.get_run_outcomes(run_id).await
    }

    async fn find_active_run_for_project(
        &self,
        project_id: uuid::Uuid,
    ) -> anyhow::Result<Option<uuid::Uuid>> {
        self.find_active_run_for_project(project_id).await
    }

    async fn persist_reasoning_tree(
        &self,
        tree: &crate::reasoning::ReasoningTree,
        linked_entity_type: Option<&str>,
        linked_entity_id: Option<uuid::Uuid>,
    ) -> anyhow::Result<uuid::Uuid> {
        self.persist_reasoning_tree(tree, linked_entity_type, linked_entity_id)
            .await
    }

    async fn get_run_reasoning_tree_id(
        &self,
        run_id: uuid::Uuid,
    ) -> anyhow::Result<Option<uuid::Uuid>> {
        self.get_run_reasoning_tree_id(run_id).await
    }

    async fn list_completed_runs_for_project(
        &self,
        project_id: uuid::Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::protocol::models::ProtocolRun>> {
        self.list_completed_runs_for_project(project_id, limit)
            .await
    }

    // ========================================================================
    // RuntimeState operations (Generator-produced dynamic states)
    // ========================================================================

    async fn create_runtime_state(
        &self,
        state: &crate::protocol::RuntimeState,
    ) -> anyhow::Result<()> {
        self.create_runtime_state(state).await
    }

    async fn get_runtime_states(
        &self,
        run_id: uuid::Uuid,
    ) -> anyhow::Result<Vec<crate::protocol::RuntimeState>> {
        self.get_runtime_states(run_id).await
    }

    async fn delete_runtime_states(&self, run_id: uuid::Uuid) -> anyhow::Result<()> {
        self.delete_runtime_states(run_id).await
    }

    // SI — System Inference: audit knowledge gaps
    async fn audit_knowledge_gaps(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<crate::neo4j::models::AuditGapsReport> {
        self.audit_knowledge_gaps(project_id).await
    }

    // ========================================================================
    // Registry operations (Skill Registry)
    // ========================================================================

    async fn upsert_published_skill(
        &self,
        published: &crate::skills::registry::PublishedSkill,
    ) -> anyhow::Result<()> {
        self.upsert_published_skill(published).await
    }

    async fn get_published_skill(
        &self,
        id: Uuid,
    ) -> anyhow::Result<Option<crate::skills::registry::PublishedSkill>> {
        self.get_published_skill(id).await
    }

    async fn search_published_skills(
        &self,
        search_query: Option<&str>,
        min_trust: Option<f64>,
        tags: Option<&[String]>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<crate::skills::registry::PublishedSkill>, usize)> {
        self.search_published_skills(search_query, min_trust, tags, limit, offset)
            .await
    }

    async fn increment_published_skill_imports(&self, id: Uuid) -> anyhow::Result<()> {
        self.increment_published_skill_imports(id).await
    }

    // ========================================================================
    // Graph visualization helpers
    // ========================================================================

    async fn list_project_symbols(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, String, String, String, Option<String>, Option<i64>)>> {
        self.list_project_symbols(project_id, limit).await
    }

    async fn get_project_inheritance_edges(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(String, String, String)>> {
        self.get_project_inheritance_edges(project_id).await
    }

    async fn get_project_constraints(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Vec<(crate::neo4j::models::ConstraintNode, Uuid)>> {
        self.get_project_constraints(project_id).await
    }

    // ========================================================================
    // PlanRun operations (Runner)
    // ========================================================================

    async fn create_plan_run(&self, state: &crate::runner::RunnerState) -> anyhow::Result<()> {
        self.create_plan_run(state).await
    }

    async fn update_plan_run(&self, state: &crate::runner::RunnerState) -> anyhow::Result<()> {
        self.update_plan_run(state).await
    }

    async fn get_plan_run(
        &self,
        run_id: Uuid,
    ) -> anyhow::Result<Option<crate::runner::RunnerState>> {
        self.get_plan_run(run_id).await
    }

    async fn list_active_plan_runs(&self) -> anyhow::Result<Vec<crate::runner::RunnerState>> {
        self.list_active_plan_runs_impl().await
    }

    async fn list_all_plan_runs(
        &self,
        limit: i64,
        offset: i64,
        status: Option<&str>,
        workspace_slug: Option<&str>,
    ) -> anyhow::Result<Vec<crate::runner::RunnerState>> {
        self.list_all_plan_runs_impl(limit, offset, status, workspace_slug).await
    }

    async fn list_plan_runs(
        &self,
        plan_id: Uuid,
        limit: i64,
    ) -> anyhow::Result<Vec<crate::runner::RunnerState>> {
        self.list_plan_runs_impl(plan_id, limit).await
    }

    // ── Triggers ──────────────────────────────────────────────────────────

    async fn create_trigger(
        &self,
        trigger: &crate::runner::Trigger,
    ) -> anyhow::Result<crate::runner::Trigger> {
        self.create_trigger_impl(trigger).await
    }

    async fn get_trigger(
        &self,
        trigger_id: Uuid,
    ) -> anyhow::Result<Option<crate::runner::Trigger>> {
        self.get_trigger_impl(trigger_id).await
    }

    async fn list_triggers(&self, plan_id: Uuid) -> anyhow::Result<Vec<crate::runner::Trigger>> {
        self.list_triggers_impl(plan_id).await
    }

    async fn list_all_triggers(
        &self,
        trigger_type: Option<&str>,
    ) -> anyhow::Result<Vec<crate::runner::Trigger>> {
        self.list_all_triggers_impl(trigger_type).await
    }

    async fn update_trigger(
        &self,
        trigger_id: Uuid,
        enabled: Option<bool>,
        config: Option<serde_json::Value>,
        cooldown_secs: Option<u64>,
    ) -> anyhow::Result<Option<crate::runner::Trigger>> {
        self.update_trigger_impl(trigger_id, enabled, config, cooldown_secs)
            .await
    }

    async fn delete_trigger(&self, trigger_id: Uuid) -> anyhow::Result<()> {
        self.delete_trigger_impl(trigger_id).await
    }

    async fn record_trigger_firing(
        &self,
        firing: &crate::runner::TriggerFiring,
    ) -> anyhow::Result<()> {
        self.record_trigger_firing_impl(firing).await
    }

    async fn list_trigger_firings(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> anyhow::Result<Vec<crate::runner::TriggerFiring>> {
        self.list_trigger_firings_impl(trigger_id, limit).await
    }

    // ── AgentExecution ──────────────────────────────────────────────────────

    async fn create_agent_execution(
        &self,
        ae: &crate::neo4j::agent_execution::AgentExecutionNode,
    ) -> anyhow::Result<()> {
        self.create_agent_execution_impl(ae).await
    }

    async fn update_agent_execution(
        &self,
        ae: &crate::neo4j::agent_execution::AgentExecutionNode,
    ) -> anyhow::Result<()> {
        self.update_agent_execution_impl(ae).await
    }

    async fn get_agent_executions_for_run(
        &self,
        run_id: Uuid,
    ) -> anyhow::Result<Vec<crate::neo4j::agent_execution::AgentExecutionNode>> {
        self.get_agent_executions_for_run_impl(run_id).await
    }

    async fn create_used_skill_relation(
        &self,
        agent_execution_id: Uuid,
        skill_id: Uuid,
        result: &str,
    ) -> anyhow::Result<()> {
        self.create_used_skill_relation_impl(agent_execution_id, skill_id, result)
            .await
    }

    async fn get_all_pagerank_values(&self, project_id: Uuid) -> anyhow::Result<Vec<f64>> {
        self.get_all_pagerank_values(project_id).await
    }

    async fn get_community_risk_vectors(&self, project_id: Uuid) -> anyhow::Result<Vec<Vec<f64>>> {
        self.get_community_risk_vectors(project_id).await
    }

    async fn get_all_risk_score_values(&self, project_id: Uuid) -> anyhow::Result<Vec<f64>> {
        self.get_all_risk_score_values(project_id).await
    }

    // ========================================================================
    // Sharing & Privacy operations
    // ========================================================================

    async fn get_sharing_policy(
        &self,
        project_id: Uuid,
    ) -> anyhow::Result<Option<crate::episodes::distill_models::SharingPolicy>> {
        self.get_sharing_policy(project_id).await
    }

    async fn update_sharing_policy(
        &self,
        project_id: Uuid,
        policy: &crate::episodes::distill_models::SharingPolicy,
    ) -> anyhow::Result<()> {
        self.update_sharing_policy(project_id, policy).await
    }

    async fn get_sharing_consent(
        &self,
        note_id: Uuid,
    ) -> anyhow::Result<crate::episodes::distill_models::SharingConsent> {
        self.get_sharing_consent(note_id).await
    }

    async fn update_sharing_consent(
        &self,
        note_id: Uuid,
        consent: &crate::episodes::distill_models::SharingConsent,
    ) -> anyhow::Result<()> {
        self.update_sharing_consent(note_id, consent).await
    }

    async fn create_sharing_event(
        &self,
        event: &crate::episodes::distill_models::SharingEvent,
    ) -> anyhow::Result<()> {
        self.create_sharing_event(event).await
    }

    async fn list_sharing_events(
        &self,
        project_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> anyhow::Result<Vec<crate::episodes::distill_models::SharingEvent>> {
        self.list_sharing_events(project_id, limit, offset).await
    }

    async fn persist_tombstone(
        &self,
        tombstone: &crate::reception::anchor::SignedTombstone,
    ) -> anyhow::Result<()> {
        self.persist_tombstone(tombstone).await
    }

    async fn list_tombstones(
        &self,
    ) -> anyhow::Result<Vec<crate::reception::anchor::SignedTombstone>> {
        self.list_tombstones().await
    }

    async fn is_tombstoned(&self, content_hash: &str) -> anyhow::Result<bool> {
        self.is_tombstoned(content_hash).await
    }

    // ========================================================================
    // UserProfile operations
    // ========================================================================

    async fn create_or_get_user_profile(
        &self,
        user_id: &str,
    ) -> anyhow::Result<crate::profile::UserProfile> {
        self.create_or_get_user_profile(user_id).await
    }

    async fn update_user_profile(
        &self,
        profile: &crate::profile::UserProfile,
    ) -> anyhow::Result<()> {
        self.update_user_profile(profile).await
    }

    async fn get_user_profile(
        &self,
        user_id: &str,
    ) -> anyhow::Result<Option<crate::profile::UserProfile>> {
        self.get_user_profile(user_id).await
    }

    async fn upsert_works_on(&self, user_id: &str, project_id: uuid::Uuid) -> anyhow::Result<()> {
        self.upsert_works_on(user_id, project_id).await
    }

    async fn get_works_on(
        &self,
        user_id: &str,
    ) -> anyhow::Result<Vec<crate::profile::WorksOnRelation>> {
        self.get_works_on(user_id).await
    }

    // ========================================================================
    // Alert operations (Heartbeat Engine)
    // ========================================================================

    async fn create_alert(&self, alert: &AlertNode) -> anyhow::Result<()> {
        self.create_alert_node(alert).await
    }

    async fn list_pending_alerts(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
    ) -> anyhow::Result<Vec<AlertNode>> {
        self.list_pending_alerts_impl(project_id, limit).await
    }

    async fn acknowledge_alert(&self, alert_id: Uuid, acknowledged_by: &str) -> anyhow::Result<()> {
        self.acknowledge_alert_impl(alert_id, acknowledged_by).await
    }

    async fn get_alert(&self, alert_id: Uuid) -> anyhow::Result<Option<AlertNode>> {
        self.get_alert_impl(alert_id).await
    }

    async fn list_alerts(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<(Vec<AlertNode>, usize)> {
        self.list_alerts_impl(project_id, limit, offset).await
    }

    // ========================================================================
    // EventTrigger operations
    // ========================================================================

    async fn list_event_triggers(
        &self,
        project_scope: Option<Uuid>,
        enabled_only: bool,
    ) -> anyhow::Result<Vec<crate::events::trigger::EventTrigger>> {
        self.list_event_triggers(project_scope, enabled_only).await
    }

    async fn create_event_trigger(
        &self,
        trigger: &crate::events::trigger::EventTrigger,
    ) -> anyhow::Result<Uuid> {
        self.create_event_trigger(trigger).await
    }

    async fn get_event_trigger(
        &self,
        id: Uuid,
    ) -> anyhow::Result<Option<crate::events::trigger::EventTrigger>> {
        self.get_event_trigger(id).await
    }

    async fn update_event_trigger(
        &self,
        id: Uuid,
        enabled: Option<bool>,
        name: Option<String>,
        entity_type_pattern: Option<Option<String>>,
        action_pattern: Option<Option<String>>,
        payload_conditions: Option<Option<serde_json::Value>>,
        cooldown_secs: Option<u32>,
        project_scope: Option<Option<Uuid>>,
    ) -> anyhow::Result<bool> {
        self.update_event_trigger(
            id,
            enabled,
            name,
            entity_type_pattern,
            action_pattern,
            payload_conditions,
            cooldown_secs,
            project_scope,
        )
        .await
    }

    async fn delete_event_trigger(&self, id: Uuid) -> anyhow::Result<bool> {
        self.delete_event_trigger(id).await
    }
}
