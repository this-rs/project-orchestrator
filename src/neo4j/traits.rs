//! GraphStore trait definition
//!
//! Defines the abstract interface for all Neo4j graph operations.
//! This trait mirrors all public async methods of `Neo4jClient`,
//! enabling testing with mock implementations and future backend swaps.

use crate::graph::models::{
    AnalysisProfile, FabricFileAnalyticsUpdate, FileAnalyticsUpdate, FunctionAnalyticsUpdate,
    TopologyRule, TopologyViolation,
};
use crate::neo4j::models::*;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteFilters, NoteImportance, NoteStatus, PropagatedNote,
};
use crate::parser::FunctionCall;
use crate::plan::models::{TaskDetails, UpdatePlanRequest, UpdateStepRequest, UpdateTaskRequest};
use crate::protocol::{
    Protocol, ProtocolRun, ProtocolState, ProtocolTransition, RunStatus, RuntimeState,
};
use crate::skills::{ActivatedSkillContext, SkillNode, SkillStatus};
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

    /// Update project analytics_computed_at timestamp
    async fn update_project_analytics_timestamp(&self, id: Uuid) -> Result<()>;

    /// Delete a project and all its data.
    /// `project_name` is used to tag archived notes/decisions with the source project.
    async fn delete_project(&self, id: Uuid, project_name: &str) -> Result<()>;

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
        slug: Option<String>,
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

    /// Compute the P2P coupling matrix for all projects in a workspace.
    async fn compute_coupling_matrix(
        &self,
        workspace_id: Uuid,
    ) -> Result<crate::neo4j::models::CouplingMatrix>;

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

    /// Link a plan to a workspace milestone
    async fn link_plan_to_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()>;

    /// Unlink a plan from a workspace milestone
    async fn unlink_plan_from_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()>;

    /// Get workspace milestone progress
    async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)>;

    /// Get tasks linked to a workspace milestone (with plan info)
    async fn get_workspace_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskWithPlan>>;

    /// Get all steps for all tasks linked to a workspace milestone (batch)
    async fn get_workspace_milestone_steps(
        &self,
        milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>>;

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

    /// Delete files that are no longer on the filesystem.
    /// Returns `(files_deleted, symbols_deleted, deleted_paths)`.
    async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize, Vec<String>)>;

    /// Link a file to a project (create CONTAINS relationship)
    async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> Result<()>;

    /// Create or update a file node
    async fn upsert_file(&self, file: &FileNode) -> Result<()>;

    /// Batch create or update file nodes (UNWIND)
    async fn batch_upsert_files(&self, files: &[FileNode]) -> Result<()>;

    /// Get a file by path
    async fn get_file(&self, path: &str) -> Result<Option<FileNode>>;

    /// List files for a project
    async fn list_project_files(&self, project_id: Uuid) -> Result<Vec<FileNode>>;

    /// Count files for a project (lightweight, no data transfer)
    async fn count_project_files(&self, project_id: Uuid) -> Result<i64>;

    /// Invalidate pre-computed GraIL properties on changed files and their neighbors.
    ///
    /// Sets `*_version = -1` on:
    /// - Direct files: `cc_version`, `structural_dna_version`, `wl_hash_version`
    /// - 1-2 hop neighbors: `cc_version` (1-hop), `wl_hash_version` (2-hop)
    ///
    /// Returns the total number of File nodes marked as stale.
    async fn invalidate_computed_properties(
        &self,
        project_id: Uuid,
        paths: &[String],
    ) -> Result<u64>;

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

    /// Create an IMPORTS_SYMBOL relationship from an Import to a matching Struct/Enum/Trait
    async fn create_imports_symbol_relationship(
        &self,
        import_id: &str,
        symbol_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<()>;

    /// Create a CALLS relationship between functions, scoped to the same project.
    /// When project_id is provided, the callee is matched only within the same project
    /// to prevent cross-project CALLS pollution.
    /// Sets `confidence` (0.0-1.0) and `reason` properties on the CALLS relationship.
    async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
        confidence: f64,
        reason: &str,
    ) -> Result<()>;

    // ========================================================================
    // Batch upsert operations (UNWIND)
    // ========================================================================

    /// Batch upsert functions using UNWIND for a single Neo4j transaction per call.
    async fn batch_upsert_functions(&self, functions: &[FunctionNode]) -> Result<()>;

    /// Batch upsert structs using UNWIND.
    async fn batch_upsert_structs(&self, structs: &[StructNode]) -> Result<()>;

    /// Batch upsert traits using UNWIND.
    async fn batch_upsert_traits(&self, traits: &[TraitNode]) -> Result<()>;

    /// Batch upsert enums using UNWIND.
    async fn batch_upsert_enums(&self, enums: &[EnumNode]) -> Result<()>;

    /// Batch upsert impl blocks using UNWIND — 3 phases matching upsert_impl behavior.
    async fn batch_upsert_impls(&self, impls: &[ImplNode]) -> Result<()>;

    /// Batch upsert imports using UNWIND.
    async fn batch_upsert_imports(&self, imports: &[ImportNode]) -> Result<()>;

    /// Batch create File→IMPORTS→File relationships using UNWIND.
    async fn batch_create_import_relationships(
        &self,
        relationships: &[(String, String, String)],
    ) -> Result<()>;

    /// Batch create Import→IMPORTS_SYMBOL→(Struct|Enum|Trait) relationships using UNWIND.
    async fn batch_create_imports_symbol_relationships(
        &self,
        relationships: &[(String, String, Option<Uuid>)],
    ) -> Result<()>;

    /// Batch create CALLS relationships using UNWIND — 2-phase strategy.
    async fn batch_create_call_relationships(
        &self,
        calls: &[FunctionCall],
        project_id: Option<Uuid>,
    ) -> Result<()>;

    /// Batch create EXTENDS relationships (class inheritance) using UNWIND.
    /// Takes (child_name, child_file_path, parent_name, project_id) tuples.
    /// Phase 1: same-file match. Phase 2: project-scoped fallback.
    async fn batch_create_extends_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> Result<()>;

    /// Batch create IMPLEMENTS relationships (interface/protocol implementation) using UNWIND.
    /// Takes (struct_name, struct_file_path, interface_name, project_id) tuples.
    async fn batch_create_implements_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> Result<()>;

    /// Delete all CALLS relationships where caller and callee belong to different projects.
    /// Returns the number of deleted relationships.
    async fn cleanup_cross_project_calls(&self) -> Result<i64>;

    /// Delete all CALLS relationships where the callee function name is a known built-in.
    /// Returns the number of deleted relationships.
    async fn cleanup_builtin_calls(&self) -> Result<i64>;

    /// Migrate existing CALLS: set default confidence=0.50 and reason='fuzzy-global'
    /// for relationships missing these properties.
    async fn migrate_calls_confidence(&self) -> Result<i64>;

    /// Clean up ALL sync-generated data (File, Function, Struct, Trait, Enum, Impl, Import, FeatureGraph).
    /// Returns the total number of deleted entities/relationships.
    async fn cleanup_sync_data(&self) -> Result<i64>;

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
    // Heritage navigation queries (EXTENDS + IMPLEMENTS)
    // ========================================================================

    /// Get the full class hierarchy (parents + children) for a type
    async fn get_class_hierarchy(
        &self,
        type_name: &str,
        max_depth: u32,
    ) -> Result<serde_json::Value>;

    /// Find all subclasses of a given class (via EXTENDS)
    async fn find_subclasses(&self, class_name: &str) -> Result<Vec<serde_json::Value>>;

    /// Find all classes that implement a given interface (via IMPLEMENTS)
    async fn find_interface_implementors(
        &self,
        interface_name: &str,
    ) -> Result<Vec<serde_json::Value>>;

    // ========================================================================
    // Process queries
    // ========================================================================

    /// List all detected processes for a project
    async fn list_processes(&self, project_id: uuid::Uuid) -> Result<Vec<serde_json::Value>>;

    /// Get details of a specific process including ordered steps
    async fn get_process_detail(&self, process_id: &str) -> Result<Option<serde_json::Value>>;

    /// Get scored entry points for a project
    async fn get_entry_points(
        &self,
        project_id: uuid::Uuid,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>>;

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

    /// Get direct callers with confidence scores (depth 1 only).
    /// Returns (name, file_path, confidence, reason) tuples.
    async fn get_callers_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>>;

    /// Get direct callees with confidence scores (depth 1 only).
    async fn get_callees_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>>;

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

    /// Get distinct communities for a project (from graph analytics Louvain clustering)
    async fn get_project_communities(&self, project_id: Uuid) -> Result<Vec<CommunityRow>>;

    /// Get GDS analytics properties for a node (File by path, or Function by name).
    /// Returns None if the node doesn't exist or has no analytics properties.
    async fn get_node_analytics(
        &self,
        identifier: &str,
        node_type: &str,
    ) -> Result<Option<NodeAnalyticsRow>>;

    /// Get distinct community labels for a list of file paths.
    /// Returns only non-null community_label values.
    async fn get_affected_communities(&self, file_paths: &[String]) -> Result<Vec<String>>;

    /// Get a structural health report: god functions, orphan files, coupling metrics.
    async fn get_code_health_report(
        &self,
        project_id: Uuid,
        god_function_threshold: usize,
    ) -> Result<crate::neo4j::models::CodeHealthReport>;

    /// Capture a lightweight maintenance snapshot (biomimicry T11).
    async fn compute_maintenance_snapshot(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::MaintenanceSnapshot>;

    /// Compute scaffolding level for adaptive task complexity (biomimicry T8).
    async fn compute_scaffolding_level(
        &self,
        project_id: Uuid,
        scaffolding_override: Option<u8>,
    ) -> Result<crate::neo4j::models::ScaffoldingLevel>;

    /// Set or clear the scaffolding level override on a project (biomimicry T8).
    async fn set_scaffolding_override(&self, project_id: Uuid, level: Option<u8>) -> Result<()>;

    /// Detect global stagnation across a project (biomimicry T12).
    async fn detect_global_stagnation(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::StagnationReport>;

    /// Detect circular dependencies between files (import cycles).
    async fn get_circular_dependencies(&self, project_id: Uuid) -> Result<Vec<Vec<String>>>;

    /// Get GDS metrics for a specific node (file or function) in a project.
    async fn get_node_gds_metrics(
        &self,
        node_path: &str,
        node_type: &str,
        project_id: Uuid,
    ) -> Result<Option<NodeGdsMetrics>>;

    /// Get statistical percentiles for GDS metrics across all nodes in a project.
    async fn get_project_percentiles(&self, project_id: Uuid) -> Result<ProjectPercentiles>;

    /// Get top N files by betweenness centrality (bridge files).
    async fn get_top_bridges_by_betweenness(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<BridgeFile>>;

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

    /// Count plans for a project (lightweight, no data transfer)
    async fn count_project_plans(&self, project_id: Uuid) -> Result<i64>;

    /// List plans for a project with filters
    async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)>;

    /// Update plan fields (title, description, priority)
    async fn update_plan(&self, id: Uuid, updates: &UpdatePlanRequest) -> Result<()>;

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

    /// Batch-fetch step counts, note counts, and decision counts for a list of task IDs.
    async fn get_task_enrichment_counts(
        &self,
        task_ids: &[String],
    ) -> Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentCounts>>;

    /// Batch-fetch full enrichment data (counts + individual step details) for dependency graph.
    async fn get_task_enrichment_data(
        &self,
        task_ids: &[String],
    ) -> Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentData>>;

    /// Find critical path in a plan (longest chain of dependencies)
    async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>>;

    /// Compute execution waves for a plan (topological sort + level grouping)
    async fn compute_waves(
        &self,
        plan_id: Uuid,
    ) -> Result<crate::neo4j::plan::WaveComputationResult>;

    /// Get next available task (no unfinished dependencies)
    async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>>;

    /// Get a single task by ID
    async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>>;

    /// Update a task with new values
    async fn update_task(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> Result<()>;

    /// Update pre-enrichment fields on a task (execution_context, persona, prompt_cache).
    async fn update_task_enrichment(
        &self,
        task_id: Uuid,
        execution_context: Option<&str>,
        persona: Option<&str>,
        prompt_cache: Option<&str>,
    ) -> Result<()>;

    /// Delete a task and all its related data (steps, decisions)
    async fn delete_task(&self, task_id: Uuid) -> Result<()>;

    /// Get the project that owns a task (via Task→Plan→Project chain).
    ///
    /// Traverses: `(Task)-[:PART_OF]->(Plan)-[:BELONGS_TO]->(Project)`
    /// Returns `None` if the task doesn't exist or has no linked project.
    async fn get_project_for_task(&self, task_id: Uuid) -> Result<Option<ProjectNode>>;

    // ========================================================================
    // Step operations
    // ========================================================================

    /// Create a step for a task
    async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()>;

    /// Get steps for a task
    async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>>;

    /// Update step fields (description, verification)
    async fn update_step(&self, step_id: Uuid, updates: &UpdateStepRequest) -> Result<()>;

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
        status: Option<DecisionStatus>,
    ) -> Result<()>;

    /// Delete a decision
    async fn delete_decision(&self, decision_id: Uuid) -> Result<()>;

    /// Get decisions related to an entity (via AFFECTS or task linkage)
    async fn get_decisions_for_entity(
        &self,
        entity_type: &str,
        entity_id: &str,
        limit: u32,
    ) -> Result<Vec<DecisionNode>>;

    /// Store a vector embedding on a Decision node
    async fn set_decision_embedding(
        &self,
        decision_id: Uuid,
        embedding: &[f32],
        model: &str,
    ) -> Result<()>;

    /// Retrieve the stored vector embedding for a Decision node
    async fn get_decision_embedding(&self, decision_id: Uuid) -> Result<Option<Vec<f32>>>;

    /// Get all decisions with their linked task_id (for MeiliSearch reindex)
    async fn get_all_decisions_with_task_id(&self) -> Result<Vec<(DecisionNode, Uuid)>>;

    /// Get all Decision IDs that have no embedding yet (for backfill)
    async fn get_decisions_without_embedding(&self) -> Result<Vec<(Uuid, String, String)>>;

    /// Semantic search over Decision embeddings using Neo4j vector index.
    /// When `project_id` is provided, filters results to that project (post-query).
    async fn search_decisions_by_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_id: Option<&str>,
    ) -> Result<Vec<(DecisionNode, f64)>>;

    /// Get decisions that AFFECT a given entity (reverse AFFECTS lookup)
    async fn get_decisions_affecting(
        &self,
        entity_type: &str,
        entity_id: &str,
        status_filter: Option<&str>,
    ) -> Result<Vec<DecisionNode>>;

    /// Create an AFFECTS relation from a Decision to any entity in the graph
    async fn add_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        impact_description: Option<&str>,
    ) -> Result<()>;

    /// Remove an AFFECTS relation from a Decision to an entity
    async fn remove_decision_affects(
        &self,
        decision_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<()>;

    /// List all entities affected by a Decision
    async fn list_decision_affects(&self, decision_id: Uuid) -> Result<Vec<AffectsRelation>>;

    /// Mark a decision as superseded by a newer decision
    async fn supersede_decision(&self, new_decision_id: Uuid, old_decision_id: Uuid) -> Result<()>;

    /// Get a timeline of decisions, optionally filtered by task and date range
    async fn get_decision_timeline(
        &self,
        task_id: Option<Uuid>,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Vec<DecisionTimelineEntry>>;

    // ========================================================================
    // Dependency analysis
    // ========================================================================

    /// Find all files that depend on a given file (IMPORTS-only traversal).
    /// When project_id is provided, only return dependents from the same project.
    async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>>;

    /// Find all files impacted by a change to a given file.
    /// Combines IMPORTS traversal (file→file) and CALLS traversal
    /// (functions in target ← called by functions in other files).
    /// When project_id is provided, only return files from the same project.
    async fn find_impacted_files(
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
    // TOUCHES operations (Commit → File)
    // ========================================================================

    /// Create TOUCHES relations between a Commit and its changed Files (batch UNWIND).
    /// Files that don't exist as nodes are silently skipped.
    async fn create_commit_touches(
        &self,
        commit_hash: &str,
        files: &[FileChangedInfo],
    ) -> Result<()>;

    /// Get all files touched by a commit
    async fn get_commit_files(&self, commit_hash: &str) -> Result<Vec<CommitFileInfo>>;

    /// Get the commit history for a specific file
    async fn get_file_history(
        &self,
        file_path: &str,
        limit: Option<i64>,
    ) -> Result<Vec<FileHistoryEntry>>;

    // ========================================================================
    // CO_CHANGED operations (File ↔ File)
    // ========================================================================

    /// Compute CO_CHANGED relations from TOUCHES history (incremental).
    /// Returns the number of relations created/updated.
    async fn compute_co_changed(
        &self,
        project_id: Uuid,
        since: Option<chrono::DateTime<chrono::Utc>>,
        min_count: i64,
        max_relations: i64,
    ) -> Result<i64>;

    /// Update project last_co_change_computed_at timestamp
    async fn update_project_co_change_timestamp(&self, id: Uuid) -> Result<()>;

    /// Get the co-change graph for a project
    async fn get_co_change_graph(
        &self,
        project_id: Uuid,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChangePair>>;

    /// Get files that co-change with a given file
    async fn get_file_co_changers(
        &self,
        file_path: &str,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChanger>>;

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

    /// Remove a commit from a release
    async fn remove_commit_from_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()>;

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

    /// Link a plan to a project milestone
    async fn link_plan_to_milestone(&self, plan_id: Uuid, milestone_id: Uuid) -> Result<()>;

    /// Unlink a plan from a project milestone
    async fn unlink_plan_from_milestone(&self, plan_id: Uuid, milestone_id: Uuid) -> Result<()>;

    /// Get milestone details with tasks
    async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>>;

    /// Get milestone progress (total, completed, in_progress, pending)
    async fn get_milestone_progress(&self, milestone_id: Uuid) -> Result<(u32, u32, u32, u32)>;

    /// Get tasks linked to a project milestone (with plan info)
    async fn get_milestone_tasks_with_plans(&self, milestone_id: Uuid)
        -> Result<Vec<TaskWithPlan>>;

    /// Get all steps for all tasks linked to a project milestone (batch)
    async fn get_milestone_steps_batch(
        &self,
        milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>>;

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
        workspace_slug: Option<&str>,
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
        workspace_slug: Option<&str>,
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

    /// Propagate LINKED_TO relationships via structural code relations (IMPORTS, CALLS, CO_CHANGED).
    /// Returns the number of propagated links created.
    async fn propagate_structural_links(&self, project_id: Uuid) -> Result<usize>;

    /// Propagate LINKED_TO via high-level entities (FeatureGraph, Skill, Protocol).
    /// Returns the number of propagated links created.
    async fn propagate_high_level_links(&self, project_id: Uuid) -> Result<usize>;

    /// Propagate a single note linked to a FeatureGraph to its member files (event-driven).
    async fn propagate_note_via_feature_graph(
        &self,
        note_id: Uuid,
        feature_graph_id: &str,
    ) -> Result<usize>;

    /// Propagate a single note linked to a Skill to files of member notes (event-driven).
    async fn propagate_note_via_skill(&self, note_id: Uuid, skill_id: &str) -> Result<usize>;

    /// Propagate a single note linked to a Protocol via Skill to files (event-driven).
    async fn propagate_note_via_protocol(&self, note_id: Uuid, protocol_id: &str) -> Result<usize>;

    /// Propagate LINKED_TO relationships via semantic similarity (embeddings).
    /// For each File with an embedding, queries the note HNSW index for top-K
    /// nearest notes and creates links above `min_similarity` threshold.
    /// Returns the number of semantic links created.
    async fn propagate_semantic_links(
        &self,
        project_id: Uuid,
        min_similarity: f64,
    ) -> Result<usize>;

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

    /// Get propagated notes for an entity (traversing the graph).
    ///
    /// `relation_types` controls which graph relations to traverse.
    /// - `None` → default (CONTAINS|IMPORTS|CALLS) — backward compatible
    /// - `Some(&["CO_CHANGED", "IMPLEMENTS_TRAIT", ...])` → custom traversal
    /// Only whitelisted relation types are accepted (see `ALLOWED_PROPAGATION_RELATIONS`).
    #[allow(clippy::too_many_arguments)]
    async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
        relation_types: Option<&[String]>,
        source_project_id: Option<Uuid>,
        force_cross_project: bool,
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

    /// Store a vector embedding on a Note node.
    ///
    /// Uses `db.create.setNodeVectorProperty` to ensure the correct type
    /// for the HNSW vector index. Also stores the model name for traceability.
    ///
    /// This is a separate method from `create_note`/`update_note` to:
    /// - Keep the CRUD API backward compatible (no signature changes)
    /// - Allow the NoteManager to call it after creation (T1.4)
    /// - Support the backfill use case (T1.5)
    async fn set_note_embedding(&self, note_id: Uuid, embedding: &[f32], model: &str)
        -> Result<()>;

    /// Retrieve the stored embedding vector for a note.
    ///
    /// Returns `None` if the note has no embedding yet.
    /// Used by the synapse backfill to get the vector and feed it
    /// into `vector_search_notes` for finding nearest neighbours.
    async fn get_note_embedding(&self, note_id: Uuid) -> Result<Option<Vec<f32>>>;

    /// Search notes by vector similarity using the HNSW index.
    ///
    /// Returns notes ordered by descending cosine similarity score,
    /// filtered by optional project_id or workspace_slug for data isolation.
    /// Only returns notes with status 'active' or 'needs_review'.
    ///
    /// Filtering priority: `project_id` > `workspace_slug` > global (no filter).
    async fn vector_search_notes(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        min_similarity: Option<f64>,
    ) -> Result<Vec<(Note, f64)>>;

    /// List notes that don't have an embedding yet.
    ///
    /// Used by the backfill process (T1.5) to find notes that need embedding.
    /// Returns (notes, total_count) where total_count is the total number of
    /// notes without embeddings. Results are ordered by created_at ASC.
    async fn list_notes_without_embedding(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Note>, usize)>;

    // ========================================================================
    // Code embedding operations (File & Function vector search)
    // ========================================================================

    /// Store a vector embedding on a File node.
    /// Uses `db.create.setNodeVectorProperty` to ensure the correct type
    /// for the HNSW vector index.
    async fn set_file_embedding(
        &self,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()>;

    /// Store a vector embedding on a Function node.
    /// Identifies the function by name + file_path for uniqueness.
    async fn set_function_embedding(
        &self,
        function_name: &str,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()>;

    /// Search files by vector similarity using the HNSW index.
    /// Returns file paths with cosine similarity scores, ordered descending.
    /// Optionally filtered by project_id.
    async fn vector_search_files(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, f64)>>;

    /// Search functions by vector similarity using the HNSW index.
    /// Returns (function_name, file_path, score) tuples, ordered by score descending.
    /// Optionally filtered by project_id.
    async fn vector_search_functions(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64)>>;

    // ========================================================================
    // Synapse operations (Phase 2 — Neural Network)
    // ========================================================================

    /// Create bidirectional SYNAPSE relationships between a note and its neighbors.
    ///
    /// Each neighbor is a tuple (neighbor_note_id, weight) where weight is the
    /// cosine similarity score (0.0 - 1.0). Uses MERGE for idempotence — calling
    /// this twice with the same data does not create duplicate relationships.
    ///
    /// Creates edges in both directions: (source)-[:SYNAPSE]->(neighbor)
    /// AND (neighbor)-[:SYNAPSE]->(source) with the same weight.
    async fn create_synapses(&self, note_id: Uuid, neighbors: &[(Uuid, f64)]) -> Result<usize>;

    /// Get all SYNAPSE relationships for a note.
    ///
    /// Returns a list of (neighbor_note_id, weight) tuples sorted by weight
    /// descending. Includes synapses in both directions (outgoing + incoming).
    async fn get_synapses(&self, note_id: Uuid) -> Result<Vec<(Uuid, f64)>>;

    /// Delete all SYNAPSE relationships for a note (both directions).
    ///
    /// Called when a note is deleted or when its content changes (before
    /// re-creating synapses with the updated embedding). Returns the number
    /// of deleted relationships.
    async fn delete_synapses(&self, note_id: Uuid) -> Result<usize>;

    // ========================================================================
    // Graph visualization batch queries (Intelligence Visualization)
    // ========================================================================

    /// Get all LINKED_TO edges from notes in a project to code entities.
    /// Returns Vec<(note_id_string, entity_type, entity_id)>.
    /// Used by the graph visualization endpoint to build the Knowledge layer.
    async fn get_project_note_entity_links(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>>;

    /// Get all SYNAPSE edges between notes in a project.
    /// Returns Vec<(source_note_id, target_note_id, weight)>.
    /// Deduplicates bidirectional synapses (only returns one direction).
    /// Used by the graph visualization endpoint to build the Neural layer.
    async fn get_project_note_synapses(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>>;

    /// Get all decisions scoped to a project (via Plan→Task→Decision chain).
    /// Returns decisions with their AFFECTS relations.
    /// Used by the graph visualization endpoint to build the Knowledge layer.
    async fn get_project_decisions_for_graph(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(DecisionNode, Vec<AffectsRelation>)>>;

    // ========================================================================
    // Energy operations (Phase 2 — Neural Network)
    // ========================================================================

    /// Apply exponential energy decay to all active notes.
    ///
    /// For each note, computes: `energy = energy × exp(-days_idle / half_life)`
    /// where `days_idle = (now - last_activated).days()`.
    ///
    /// This formula is **temporally idempotent**: calling it once after 30 days
    /// gives the same result as calling it 30 times daily, because each call
    /// recomputes from `last_activated` (absolute reference) rather than
    /// multiplying a running value.
    ///
    /// Notes that decay below 0.05 are floored to 0.0 ("dead neuron").
    /// Returns the number of notes updated.
    async fn update_energy_scores(&self, half_life_days: f64) -> Result<usize>;

    /// Boost a note's energy by a given amount (capped at 1.0) and set
    /// `last_activated` to now. Used when a note is retrieved, confirmed, or
    /// reinforced through spreading activation.
    async fn boost_energy(&self, note_id: Uuid, amount: f64) -> Result<()>;

    /// Reinforce synapses between co-activated notes (Hebbian learning).
    ///
    /// For every pair (i, j) in `note_ids`, MERGE a bidirectional SYNAPSE:
    /// - **ON CREATE**: set weight = 0.5 (new connection)
    /// - **ON MATCH**: set weight = min(weight + `boost`, 1.0)
    ///
    /// Returns the number of synapses reinforced (created + updated).
    async fn reinforce_synapses(&self, note_ids: &[Uuid], boost: f64) -> Result<usize>;

    /// Apply decay to all synapses and prune weak ones.
    ///
    /// 1. Subtract `decay_amount` from every synapse weight
    /// 2. Delete synapses where weight < `prune_threshold`
    ///
    /// Returns (decayed_count, pruned_count).
    async fn decay_synapses(
        &self,
        decay_amount: f64,
        prune_threshold: f64,
    ) -> Result<(usize, usize)>;

    /// Apply knowledge scars to notes/decisions after failed reasoning feedback.
    /// Biomimicry: Elun HypersphereIdentity.Scar — penalizes nodes in search scoring.
    /// `increment` is added to existing scar_intensity, capped at 1.0.
    /// Returns the number of nodes scarred.
    async fn apply_scars(&self, node_ids: &[Uuid], increment: f64) -> Result<usize>;

    /// Reset scar_intensity to 0.0 for a given note or decision node.
    /// Returns true if the node was found and healed.
    async fn heal_scars(&self, node_id: Uuid) -> Result<bool>;

    /// Consolidate memory: evaluate all non-consolidated notes for promotion,
    /// and archive stale ephemeral notes (>48h without reactivation).
    /// Returns (promoted_count, archived_count).
    async fn consolidate_memory(&self) -> Result<(usize, usize)>;

    /// Compute homeostasis report: 5 ratios measuring knowledge graph equilibrium.
    async fn compute_homeostasis(
        &self,
        project_id: Uuid,
        custom_ranges: Option<&[(String, f64, f64)]>,
    ) -> Result<crate::neo4j::models::HomeostasisReport>;

    // ========================================================================
    // Identity Manifold — Structural drift (Biomimicry)
    // ========================================================================

    /// Compute structural drift report for all files in a project.
    /// Returns community centroids and per-file drift distances.
    async fn compute_structural_drift(
        &self,
        project_id: Uuid,
        warning_threshold: Option<f64>,
        critical_threshold: Option<f64>,
    ) -> Result<crate::neo4j::models::StructuralDriftReport>;

    // ========================================================================
    // Frustration-Catharsis (Biomimicry)
    // ========================================================================

    /// Increment a task's frustration_score by delta (clamped to [0, 1]).
    /// Returns the new frustration_score.
    async fn increment_frustration(&self, task_id: Uuid, delta: f64) -> Result<f64>;

    /// Decrement a task's frustration_score by delta (clamped to [0, 1]).
    /// Returns the new frustration_score.
    async fn decrement_frustration(&self, task_id: Uuid, delta: f64) -> Result<f64>;

    /// Get the frustration_score for a task.
    async fn get_frustration(&self, task_id: Uuid) -> Result<f64>;

    /// Get the parent task ID for a step (via HAS_STEP relationship).
    async fn get_step_parent_task_id(&self, step_id: Uuid) -> Result<Option<Uuid>>;

    /// Initialize energy for all notes that don't have it set.
    /// Sets energy = 1.0 and last_activated = coalesce(last_confirmed_at, created_at).
    /// Idempotent. Returns the number of notes initialized.
    async fn init_note_energy(&self) -> Result<usize>;

    /// List notes that have an embedding but no outgoing SYNAPSE.
    /// Used for synapse backfill. Returns (notes, total_count).
    async fn list_notes_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<crate::notes::Note>, usize)>;

    // ========================================================================
    // Cross-entity SYNAPSE operations (Decision ↔ Note)
    // ========================================================================

    /// Create bidirectional SYNAPSE relationships between any two nodes (Note or Decision).
    /// Enables cross-entity neural linking for the Knowledge Fabric.
    async fn create_cross_entity_synapses(
        &self,
        source_id: Uuid,
        neighbors: &[(Uuid, f64)],
    ) -> Result<usize>;

    /// Get all SYNAPSE neighbors for any node (Note or Decision).
    /// Returns (neighbor_id, weight, entity_type) where entity_type is "Note" or "Decision".
    async fn get_cross_entity_synapses(&self, node_id: Uuid) -> Result<Vec<(Uuid, f64, String)>>;

    /// List Decision nodes that have embeddings but no SYNAPSE relationships.
    /// Used for cross-entity synapse backfill.
    async fn list_decisions_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DecisionNode>, usize)>;

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
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
        include_detached: bool,
    ) -> Result<(Vec<ChatSessionNode>, usize)>;

    /// Get child sessions spawned by a parent session
    async fn get_session_children(&self, parent_id: Uuid) -> Result<Vec<ChatSessionNode>>;

    /// Create a SPAWNED_BY relation between two chat sessions
    async fn create_spawned_by_relation(
        &self,
        child_session_id: &str,
        parent_session_id: &str,
        spawn_type: &str,
        run_id: Option<Uuid>,
        task_id: Option<Uuid>,
    ) -> Result<()>;

    /// Get the full session tree rooted at a session (recursive SPAWNED_BY traversal, max 10 levels)
    async fn get_session_tree(&self, session_id: &str) -> Result<Vec<SessionTreeNode>>;

    /// Follow SPAWNED_BY upward to find the root session
    async fn get_session_root(&self, session_id: &str) -> Result<Option<String>>;

    /// Get all sessions for a PlanRun via SPAWNED_BY relation metadata
    async fn get_run_sessions(&self, run_id: Uuid) -> Result<Vec<SessionInfo>>;

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

    /// Update the permission_mode field on a chat session node
    async fn update_chat_session_permission_mode(&self, id: Uuid, mode: &str) -> Result<()>;

    /// Set the auto_continue flag on a chat session node
    async fn set_session_auto_continue(&self, id: Uuid, enabled: bool) -> Result<()>;

    /// Get the auto_continue flag from a chat session node (false if not set)
    async fn get_session_auto_continue(&self, id: Uuid) -> Result<bool>;

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
    // Chat DISCUSSED relations (ChatSession → Entity)
    // ========================================================================

    /// Add DISCUSSED relations between a chat session and entities (MERGE-based, idempotent).
    /// Each entity is `(entity_type, entity_id)`.
    async fn add_discussed(&self, session_id: Uuid, entities: &[(String, String)])
        -> Result<usize>;

    /// Get all entities discussed in a chat session, optionally scoped by project_id.
    async fn get_session_entities(
        &self,
        session_id: Uuid,
        project_id: Option<Uuid>,
    ) -> Result<Vec<DiscussedEntity>>;

    /// WorldModel predictive context (biomimicry T7):
    /// Get co-changers of files discussed in the last N sessions.
    async fn get_discussed_co_changers(
        &self,
        project_id: Uuid,
        max_sessions: i64,
        max_results: i64,
    ) -> Result<Vec<CoChanger>>;

    /// Backfill DISCUSSED relations on all existing sessions.
    /// Returns `(sessions_processed, entities_found, relations_created)`.
    async fn backfill_discussed(&self) -> Result<(usize, usize, usize)>;

    // ========================================================================
    // Graph visualization queries (PM + Chat layers)
    // ========================================================================

    /// Get all PM entities (plans, tasks, steps, milestones, releases) and their
    /// inter-entity edges for the graph visualization layer.
    /// Returns (nodes, edges) where nodes are tuples of (id, type, label, attributes_json)
    /// and edges are tuples of (source_id, target_id, rel_type, attributes_json).
    async fn get_pm_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<PmGraphNode>, Vec<PmGraphEdge>)>;

    /// Get chat sessions that have DISCUSSED relations, along with those edges.
    /// Only returns sessions with at least 1 DISCUSSED relation.
    /// Returns (sessions, discussed_edges).
    async fn get_chat_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<ChatGraphSession>, Vec<ChatGraphDiscussed>)>;

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
    // Refresh Tokens
    // ================================================================

    /// Store a new refresh token (hashed) linked to a user.
    async fn create_refresh_token(
        &self,
        user_id: Uuid,
        token_hash: &str,
        expires_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()>;

    /// Validate a refresh token by its hash. Returns the token if valid
    /// (not expired, not revoked).
    async fn validate_refresh_token(
        &self,
        token_hash: &str,
    ) -> Result<Option<crate::neo4j::models::RefreshTokenNode>>;

    /// Revoke a single refresh token by its hash.
    async fn revoke_refresh_token(&self, token_hash: &str) -> Result<bool>;

    /// Revoke all refresh tokens for a given user (e.g. on password change).
    async fn revoke_all_user_tokens(&self, user_id: Uuid) -> Result<u64>;

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

    /// Add an entity (file, function, struct, trait) to a feature graph.
    /// When `project_id` is provided, MATCH queries for Function/Struct/Trait/Enum
    /// are scoped to that project to avoid cross-project contamination.
    async fn add_entity_to_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        role: Option<&str>,
        project_id: Option<Uuid>,
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
    #[allow(clippy::too_many_arguments)]
    async fn auto_build_feature_graph(
        &self,
        name: &str,
        description: Option<&str>,
        project_id: Uuid,
        entry_function: &str,
        depth: u32,
        include_relations: Option<&[String]>,
        filter_community: Option<bool>,
    ) -> Result<FeatureGraphDetail>;

    /// Refresh an auto-built feature graph by re-running the BFS with the
    /// same parameters (entry_function, depth, include_relations).
    /// Returns None if the graph was manually created (no entry_function).
    /// Returns Err if the graph doesn't exist.
    async fn refresh_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>>;

    /// Get the top N most connected functions for a project, ranked by
    /// (callers + callees). Used for auto-generating feature graphs after sync.
    async fn get_top_entry_functions(&self, project_id: Uuid, limit: usize) -> Result<Vec<String>>;

    /// Compute statistics for a feature graph: coupling, cohesion, complexity metrics.
    async fn get_feature_graph_statistics(
        &self,
        id: Uuid,
    ) -> Result<Option<FeatureGraphStatistics>>;

    /// Compare two feature graphs: shared entities, unique entities, similarity score.
    async fn compare_feature_graphs(
        &self,
        id_a: Uuid,
        id_b: Uuid,
    ) -> Result<Option<FeatureGraphComparison>>;

    /// Find feature graphs that overlap with a reference graph (same project).
    async fn find_overlapping_feature_graphs(
        &self,
        id: Uuid,
        min_overlap: f64,
    ) -> Result<Vec<FeatureGraphOverlap>>;

    // ========================================================================
    // Bulk graph extraction (for graph analytics)
    // ========================================================================

    /// Get all IMPORTS edges between files in a project as (source_path, target_path) pairs.
    /// Single bulk query — used by the graph analytics engine for extraction.
    async fn get_project_import_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>>;

    /// Get all CALLS edges between functions in a project as (caller_id, callee_id) pairs.
    /// Scoped to the same project (no cross-project calls).
    /// Single bulk query — used by the graph analytics engine for extraction.
    async fn get_project_call_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>>;

    /// Get all EXTENDS edges between structs/classes in a project as (child_file, parent_file) pairs.
    /// Returns file-level edges for the graph analytics engine.
    async fn get_project_extends_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>>;

    /// Get all IMPLEMENTS edges between structs and traits in a project as (struct_file, trait_file) pairs.
    /// Returns file-level edges for the graph analytics engine.
    async fn get_project_implements_edges(&self, project_id: Uuid)
        -> Result<Vec<(String, String)>>;

    /// Batch-update analytics scores on File nodes.
    /// Uses UNWIND for single-query efficiency.
    async fn batch_update_file_analytics(&self, updates: &[FileAnalyticsUpdate]) -> Result<()>;

    /// Batch-update analytics scores on Function nodes.
    /// Uses UNWIND for single-query efficiency.
    async fn batch_update_function_analytics(
        &self,
        updates: &[FunctionAnalyticsUpdate],
    ) -> Result<()>;

    /// Batch-update **fabric** analytics scores on File nodes.
    /// Writes to `fabric_pagerank`, `fabric_betweenness`, `fabric_community_id`, etc.
    /// These are separate from the code-only scores written by `batch_update_file_analytics`.
    async fn batch_update_fabric_file_analytics(
        &self,
        updates: &[FabricFileAnalyticsUpdate],
    ) -> Result<()>;

    /// Batch-update structural DNA vectors on File nodes.
    /// Writes the `structural_dna` property (Vec<f64>) via UNWIND in chunks of 1000.
    /// DNA = K-dimensional distance vector to anchor nodes, normalized [0,1].
    async fn batch_update_structural_dna(
        &self,
        updates: &[crate::graph::models::StructuralDnaUpdate],
    ) -> Result<()>;

    /// Write predicted missing links for a project.
    /// Persists top-N link predictions as PREDICTED_LINK relationships in Neo4j.
    async fn write_predicted_links(
        &self,
        project_id: &str,
        links: &[crate::graph::models::LinkPrediction],
    ) -> Result<()>;

    /// Read structural DNA vectors for all File nodes in a project.
    /// Returns (file_path, dna_vector) pairs.
    async fn get_project_structural_dna(&self, project_id: &str)
        -> Result<Vec<(String, Vec<f64>)>>;

    /// Batch-update structural fingerprint vectors on File nodes.
    /// Writes the `structural_fingerprint` property (Vec<f64>, 17 dims) via UNWIND in chunks of 1000.
    /// Fingerprint = universal feature vector with fixed semantics, project-independent.
    async fn batch_update_structural_fingerprints(
        &self,
        updates: &[crate::graph::models::StructuralFingerprintUpdate],
    ) -> Result<()>;

    /// Read structural fingerprint vectors for all File nodes in a project.
    /// Returns (file_path, fingerprint_vector) pairs.
    async fn get_project_structural_fingerprints(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f64>)>>;

    /// Read all file signals needed for multi-signal structural similarity.
    ///
    /// Returns fingerprint, WL hash, and function count for each file in one query.
    /// Used by `find_cross_project_twins` and `find_structural_twins` handlers.
    async fn get_project_file_signals(
        &self,
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::FileSignalRecord>>;

    // ========================================================================
    // SYNAPSE (neural connections bridged to file-level)
    // ========================================================================

    /// Get SYNAPSE edges bridged from Note-level to File-level.
    /// Returns (source_file_path, target_file_path, avg_weight) tuples.
    async fn get_project_synapse_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, f64)>>;

    /// Get neural network metrics for a project's SYNAPSE layer.
    async fn get_neural_metrics(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::NeuralMetrics>;

    // ========================================================================
    // T5.5 — Churn score (commit frequency per file)
    // ========================================================================

    /// Compute churn metrics per file via TOUCHES relations.
    async fn compute_churn_scores(&self, project_id: Uuid) -> Result<Vec<FileChurnScore>>;

    /// Batch-write churn scores to File nodes.
    async fn batch_update_churn_scores(&self, updates: &[FileChurnScore]) -> Result<()>;

    /// Get top N files by churn_score (pre-computed on File nodes).
    async fn get_top_hotspots(&self, project_id: Uuid, limit: usize)
        -> Result<Vec<FileChurnScore>>;

    // ========================================================================
    // T5.6 — Knowledge density score
    // ========================================================================

    /// Compute knowledge density per file based on linked notes and decisions.
    async fn compute_knowledge_density(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<FileKnowledgeDensity>>;

    /// Batch-write knowledge density scores to File nodes.
    async fn batch_update_knowledge_density(&self, updates: &[FileKnowledgeDensity]) -> Result<()>;

    /// Get top N files with lowest knowledge_density (knowledge gaps).
    async fn get_top_knowledge_gaps(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<FileKnowledgeDensity>>;

    // ========================================================================
    // T5.7 — Risk score composite
    // ========================================================================

    /// Compute composite risk scores for all files in a project.
    async fn compute_risk_scores(&self, project_id: Uuid) -> Result<Vec<FileRiskScore>>;

    /// Batch-write composite risk scores to File nodes.
    async fn batch_update_risk_scores(&self, updates: &[FileRiskScore]) -> Result<()>;

    /// Get risk assessment summary stats for a project.
    async fn get_risk_summary(&self, project_id: Uuid) -> Result<serde_json::Value>;

    // ========================================================================
    // Process detection
    // ========================================================================

    /// Batch upsert Process nodes using UNWIND.
    async fn batch_upsert_processes(&self, processes: &[ProcessNode]) -> Result<()>;

    /// Batch create STEP_IN_PROCESS relationships.
    /// Takes (process_id, function_id, step_number) tuples.
    async fn batch_create_step_relationships(&self, steps: &[(String, String, u32)]) -> Result<()>;

    /// Delete all Process nodes and their STEP_IN_PROCESS relationships for a project.
    /// Returns the number of deleted entities.
    async fn delete_project_processes(&self, project_id: Uuid) -> Result<u64>;

    // ========================================================================
    // Skill operations (Neural Skills)
    // ========================================================================

    /// Create a new skill node with its BELONGS_TO relationship to project.
    async fn create_skill(&self, skill: &SkillNode) -> Result<()>;

    /// Get a skill by ID.
    async fn get_skill(&self, id: Uuid) -> Result<Option<SkillNode>>;

    /// Update a skill node (replaces all mutable fields).
    async fn update_skill(&self, skill: &SkillNode) -> Result<()>;

    /// Delete a skill and all its relationships (CONTAINS, COVERS, BELONGS_TO).
    /// Returns true if the skill existed and was deleted.
    async fn delete_skill(&self, id: Uuid) -> Result<bool>;

    /// List skills for a project with optional status filter and pagination.
    /// Returns (skills, total_count).
    async fn list_skills(
        &self,
        project_id: Uuid,
        status: Option<SkillStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<SkillNode>, usize)>;

    // ========================================================================
    // Skill membership operations
    // ========================================================================

    /// Get all member notes and decisions for a skill.
    async fn get_skill_members(&self, skill_id: Uuid) -> Result<(Vec<Note>, Vec<DecisionNode>)>;

    /// Add a note or decision as a member of a skill.
    /// `entity_type` must be "note" or "decision".
    /// Uses MERGE for idempotence — calling twice is safe.
    async fn add_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<()>;

    /// Remove a member (note or decision) from a skill.
    /// Returns true if the relationship existed and was removed.
    async fn remove_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<bool>;

    /// Remove all members (notes and decisions) from a skill.
    /// Used during evolution to clean up relationships before archiving.
    /// Returns the number of relationships removed.
    async fn remove_all_skill_members(&self, skill_id: Uuid) -> Result<i64>;

    // ========================================================================
    // Skill query & activation operations
    // ========================================================================

    /// Get all skills that contain a given note as member.
    async fn get_skills_for_note(&self, note_id: Uuid) -> Result<Vec<SkillNode>>;

    /// Get all skills belonging to a project (convenience for list without pagination).
    async fn get_skills_for_project(&self, project_id: Uuid) -> Result<Vec<SkillNode>>;

    /// Activate a skill: collect member notes (above min energy) and relevant
    /// decisions, assemble context text. Returns the full activation payload.
    async fn activate_skill(&self, skill_id: Uuid, query: &str) -> Result<ActivatedSkillContext>;

    /// Increment a skill's activation_count and update last_activated.
    /// Used by hook activation paths (fire-and-forget, best-effort).
    async fn increment_skill_activation(&self, skill_id: Uuid) -> Result<()>;

    /// Match skills by evaluating trigger patterns against an input string.
    /// Returns matching skills with their confidence scores, sorted descending.
    /// Only matches Active/Emerging skills with reliable triggers.
    async fn match_skills_by_trigger(
        &self,
        project_id: Uuid,
        input: &str,
    ) -> Result<Vec<(SkillNode, f64)>>;

    // ========================================================================
    // Skill Detection — Graph Extraction
    // ========================================================================

    /// Extract the SYNAPSE graph for a project: all Note↔Note edges with weight > min_weight.
    /// Returns Vec<(from_note_id, to_note_id, weight)>.
    async fn get_synapse_graph(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>>;

    // ========================================================================
    // Persona operations (Living Personas)
    // ========================================================================

    /// Create a new persona node. If project_id is Some, links it via BELONGS_TO.
    async fn create_persona(&self, persona: &PersonaNode) -> Result<()>;

    /// Get a persona by ID.
    async fn get_persona(&self, id: Uuid) -> Result<Option<PersonaNode>>;

    /// Update a persona node (replaces all mutable fields).
    async fn update_persona(&self, persona: &PersonaNode) -> Result<()>;

    /// Delete a persona and all its relationships.
    /// Returns true if the persona existed and was deleted.
    async fn delete_persona(&self, id: Uuid) -> Result<bool>;

    /// List personas for a project with optional status filter and pagination.
    /// Returns (personas, total_count).
    async fn list_personas(
        &self,
        project_id: Uuid,
        status: Option<PersonaStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PersonaNode>, usize)>;

    /// List global personas (project_id IS NULL) — portable personas.
    async fn list_global_personas(&self) -> Result<Vec<PersonaNode>>;

    // ========================================================================
    // Persona relation operations
    // ========================================================================

    /// Add MASTERS relation: Persona -> Skill
    async fn add_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()>;

    /// Remove MASTERS relation: Persona -> Skill
    async fn remove_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()>;

    /// Add FOLLOWS relation: Persona -> Protocol
    async fn add_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()>;

    /// Remove FOLLOWS relation: Persona -> Protocol
    async fn remove_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()>;

    /// Add SCOPED_TO relation: Persona -> FeatureGraph
    async fn set_persona_feature_graph(
        &self,
        persona_id: Uuid,
        feature_graph_id: Uuid,
    ) -> Result<()>;

    /// Remove SCOPED_TO relation: Persona -> FeatureGraph
    async fn remove_persona_feature_graph(&self, persona_id: Uuid) -> Result<()>;

    /// Add KNOWS relation: Persona -> File (with weight)
    async fn add_persona_file(&self, persona_id: Uuid, file_path: &str, weight: f64) -> Result<()>;

    /// Remove KNOWS relation: Persona -> File
    async fn remove_persona_file(&self, persona_id: Uuid, file_path: &str) -> Result<()>;

    /// Add KNOWS relation: Persona -> Function (with weight)
    async fn add_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
        weight: f64,
    ) -> Result<()>;

    /// Remove KNOWS relation: Persona -> Function
    async fn remove_persona_function(&self, persona_id: Uuid, function_id: &str) -> Result<()>;

    /// Add USES relation: Persona -> Note (with weight)
    async fn add_persona_note(&self, persona_id: Uuid, note_id: Uuid, weight: f64) -> Result<()>;

    /// Remove USES relation: Persona -> Note
    async fn remove_persona_note(&self, persona_id: Uuid, note_id: Uuid) -> Result<()>;

    /// Add USES relation: Persona -> Decision (with weight)
    async fn add_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()>;

    /// Remove USES relation: Persona -> Decision
    async fn remove_persona_decision(&self, persona_id: Uuid, decision_id: Uuid) -> Result<()>;

    /// Add EXTENDS relation: child Persona -> parent Persona
    async fn add_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()>;

    /// Remove EXTENDS relation: child Persona -> parent Persona
    async fn remove_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()>;

    // ========================================================================
    // Persona subgraph (knowledge assembly)
    // ========================================================================

    /// Assemble the full knowledge subgraph for a persona.
    /// Includes all KNOWS, USES, MASTERS, FOLLOWS, SCOPED_TO, EXTENDS entities.
    async fn get_persona_subgraph(&self, persona_id: Uuid) -> Result<PersonaSubgraph>;

    /// Find personas that KNOW a given file (for activation by file match).
    /// Returns (persona, weight) sorted by weight descending.
    async fn find_personas_for_file(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, f64)>>;

    /// Maintain all personas for a project: decay weights, prune, recalculate cohesion.
    /// Returns (decayed_count, pruned_count, personas_updated).
    async fn maintain_personas(&self, project_id: Uuid) -> Result<(usize, usize, usize)>;

    /// Detect potential personas from Louvain communities.
    async fn detect_personas(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<crate::neo4j::persona::PersonaProposal>>;

    /// Find personas relevant to a note based on KNOWS file overlap.
    async fn find_relevant_personas_for_note(
        &self,
        file_paths: &[String],
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>>;

    /// Find personas relevant to a decision based on KNOWS overlap with AFFECTS entities.
    async fn find_relevant_personas_for_decision(
        &self,
        decision_id: Uuid,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>>;

    /// Auto-link a note to an active persona (growth hook).
    async fn auto_link_note_to_persona(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> Result<()>;

    /// Auto-link a decision to an active persona (growth hook).
    async fn auto_link_decision_to_persona(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()>;

    /// Auto-link a file to an active persona (growth hook).
    async fn auto_link_file_to_persona(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()>;

    // ========================================================================
    // Analysis Profile operations
    // ========================================================================

    /// Create or update an analysis profile.
    ///
    /// Uses MERGE on `id` so built-in profiles can be upserted idempotently.
    /// If `project_id` is Some, links the profile to the project via HAS_PROFILE.
    async fn create_analysis_profile(&self, profile: &AnalysisProfile) -> Result<()>;

    /// List analysis profiles visible to a project.
    ///
    /// Returns global profiles (project_id IS NULL) + project-specific profiles.
    /// If `project_id` is None, returns only global profiles.
    async fn list_analysis_profiles(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<AnalysisProfile>>;

    /// Get a single analysis profile by id.
    async fn get_analysis_profile(&self, id: &str) -> Result<Option<AnalysisProfile>>;

    /// Delete an analysis profile by id.
    ///
    /// Returns an error if the profile is built-in (`is_builtin = true`).
    async fn delete_analysis_profile(&self, id: &str) -> Result<()>;

    // ========================================================================
    // Bridge subgraph extraction (GraIL Plan 1)
    // ========================================================================

    /// Extract the enclosing bridge subgraph between two nodes via bidirectional
    /// BFS intersection. Returns raw nodes and edges; double-radius labeling
    /// and bottleneck detection are done in Rust (see graph/algorithms.rs).
    ///
    /// - `source`/`target`: file paths within the project
    /// - `max_hops`: BFS radius (1..=5)
    /// - `relation_types`: edge types to traverse (e.g. ["IMPORTS", "CALLS"])
    /// - `project_id`: project UUID for scoping
    async fn find_bridge_subgraph(
        &self,
        source: &str,
        target: &str,
        max_hops: u32,
        relation_types: &[String],
        project_id: &str,
    ) -> Result<(
        Vec<crate::graph::models::BridgeRawNode>,
        Vec<crate::graph::models::BridgeRawEdge>,
    )>;

    // ========================================================================
    // Multi-signal impact queries
    // ========================================================================

    /// Get the knowledge density for a file: count of notes LINKED_TO + decisions AFFECTS,
    /// normalized by max count across all files in the project. Returns f64 in [0, 1].
    async fn get_knowledge_density(&self, file_path: &str, project_id: &str) -> Result<f64>;

    /// Get the PageRank score for a file node.
    /// Reads the `cc_pagerank` property set by GDS projection. Falls back to 0.0.
    async fn get_node_pagerank(&self, file_path: &str, project_id: &str) -> Result<f64>;

    /// Get bridge proximity scores for a file's top co-changers.
    /// For each co-changer, computes 1.0 / shortestPath distance to the target.
    /// Returns Vec<(path, score)> sorted by score descending.
    async fn get_bridge_proximity(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> Result<Vec<(String, f64)>>;

    /// Compute the average multi-signal impact score across the top-10 files
    /// by PageRank. Uses GDS properties (pagerank, betweenness, churn_score,
    /// knowledge_density) to approximate the 5-signal fusion in a single query.
    /// Returns 0.0 if no files have GDS metrics computed.
    async fn get_avg_multi_signal_score(&self, project_id: Uuid) -> Result<f64>;

    // ========================================================================
    // Topology Firewall (GraIL Plan 3)
    // ========================================================================

    /// Create a topology rule.
    ///
    /// Stores the rule as a `TopologyRule` node in Neo4j with the given
    /// project_id, rule_type, source/target patterns, threshold, and severity.
    async fn create_topology_rule(&self, rule: &TopologyRule) -> Result<()>;

    /// List all topology rules for a project.
    async fn list_topology_rules(&self, project_id: &str) -> Result<Vec<TopologyRule>>;

    /// Delete a topology rule by id.
    async fn delete_topology_rule(&self, rule_id: &str) -> Result<()>;

    /// Check all topology rules for a project and return violations.
    ///
    /// Iterates over all rules for the project and runs type-specific Cypher
    /// queries to detect violations:
    /// - `MustNotImport`: files matching source_pattern that IMPORTS files matching target_pattern
    /// - `MustNotCall`: functions matching source_pattern that CALLS functions matching target_pattern
    /// - `MaxFanOut`: files matching source_pattern with more than threshold IMPORTS
    /// - `NoCircular`: circular import chains (depth 2..6) among files matching source_pattern
    /// - `MaxDistance`: shortest path between source and target patterns >= threshold
    async fn check_topology_rules(&self, project_id: &str) -> Result<Vec<TopologyViolation>>;

    /// Check if a specific file's new imports would violate any topology rules.
    ///
    /// Designed for real-time pre-write validation (<50ms target).
    /// Only checks `MustNotImport` rules where `file_path` matches the source pattern
    /// and any of `new_imports` matches the target pattern.
    async fn check_file_topology(
        &self,
        project_id: &str,
        file_path: &str,
        new_imports: &[String],
    ) -> Result<Vec<TopologyViolation>>;

    // ========================================================================
    // Health check
    // ========================================================================

    /// Check connectivity to the graph database.
    /// Returns Ok(true) if the database is reachable, Ok(false) if not.
    async fn health_check(&self) -> Result<bool>;

    // ========================================================================
    // Context Cards persistence
    // ========================================================================

    /// Batch-write context cards as cc_* properties on File nodes.
    async fn batch_save_context_cards(
        &self,
        cards: &[crate::graph::models::ContextCard],
    ) -> Result<()>;

    /// Invalidate context cards for given file paths + their 1-hop neighbors.
    /// Sets cc_version = -1 on the target files and their direct neighbors.
    async fn invalidate_context_cards(&self, paths: &[String], project_id: &str) -> Result<()>;

    /// Read a single context card from Neo4j cc_* properties.
    /// Returns None if the file doesn't exist or has no cc_* properties.
    async fn get_context_card(
        &self,
        path: &str,
        project_id: &str,
    ) -> Result<Option<crate::graph::models::ContextCard>>;

    /// Batch-read context cards for multiple files.
    async fn get_context_cards_batch(
        &self,
        paths: &[String],
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::ContextCard>>;

    /// Find groups of files with identical WL hash (isomorphic neighborhoods).
    /// Returns groups with at least `min_group_size` members.
    async fn find_isomorphic_groups(
        &self,
        project_id: &str,
        min_group_size: usize,
    ) -> Result<Vec<crate::graph::models::IsomorphicGroup>>;

    /// Check if any file in the project has GraIL analytics (context cards) computed.
    /// Used by staleness check to detect pre-GraIL projects needing first computation.
    async fn has_context_cards(&self, project_id: &str) -> Result<bool>;

    // ========================================================================
    // Embedding projection (UMAP visualization)
    // ========================================================================

    /// Get all note embeddings for a project in a single batch query.
    /// Returns (note_id, embedding_vec, note_type, importance, energy, tags, content_preview).
    /// Only returns notes that have embeddings. Used for UMAP 2D projection.
    async fn get_note_embeddings_for_project(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<NoteEmbeddingPoint>>;

    // ========================================================================
    // Protocol operations (Pattern Federation)
    // ========================================================================

    /// Create a new protocol with its BELONGS_TO relationship to project.
    /// Also creates HAS_STATE and HAS_TRANSITION relationships for sub-entities.
    async fn upsert_protocol(&self, protocol: &Protocol) -> Result<()>;

    /// Get a protocol by ID, including its states and transitions.
    async fn get_protocol(&self, id: Uuid) -> Result<Option<Protocol>>;

    /// List protocols for a project with optional category filter and pagination.
    /// Returns (protocols, total_count).
    async fn list_protocols(
        &self,
        project_id: Uuid,
        category: Option<crate::protocol::ProtocolCategory>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Protocol>, usize)>;

    /// Delete a protocol and all its states, transitions, and relationships.
    /// Returns true if the protocol existed and was deleted.
    async fn delete_protocol(&self, id: Uuid) -> Result<bool>;

    /// Upsert a protocol state and create its HAS_STATE relationship.
    async fn upsert_protocol_state(&self, state: &ProtocolState) -> Result<()>;

    /// Get all states for a protocol.
    async fn get_protocol_states(&self, protocol_id: Uuid) -> Result<Vec<ProtocolState>>;

    /// Delete a protocol state and its relationships.
    async fn delete_protocol_state(&self, state_id: Uuid) -> Result<bool>;

    /// Upsert a protocol transition and create its HAS_TRANSITION relationship.
    async fn upsert_protocol_transition(&self, transition: &ProtocolTransition) -> Result<()>;

    /// Get all transitions for a protocol.
    async fn get_protocol_transitions(&self, protocol_id: Uuid) -> Result<Vec<ProtocolTransition>>;

    /// Delete a protocol transition.
    async fn delete_protocol_transition(&self, transition_id: Uuid) -> Result<bool>;

    // ========================================================================
    // ProtocolRun operations (FSM Runtime)
    // ========================================================================

    /// Create a new protocol run node with INSTANCE_OF relationship to its protocol.
    async fn create_protocol_run(&self, run: &ProtocolRun) -> Result<()>;

    /// Get a protocol run by ID.
    async fn get_protocol_run(&self, run_id: Uuid) -> Result<Option<ProtocolRun>>;

    /// Update an existing protocol run (current_state, states_visited, status, etc.).
    async fn update_protocol_run(&self, run: &ProtocolRun) -> Result<()>;

    /// List protocol runs for a protocol with optional status filter and pagination.
    /// Returns (runs, total_count).
    async fn list_protocol_runs(
        &self,
        protocol_id: Uuid,
        status: Option<RunStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ProtocolRun>, usize)>;

    /// List child runs for a given parent run.
    async fn list_child_runs(&self, parent_run_id: Uuid) -> Result<Vec<ProtocolRun>>;

    /// Count children for a given run.
    async fn count_child_runs(&self, parent_run_id: Uuid) -> Result<usize>;

    /// Get the full run tree rooted at a given run (recursive, depth limit 5).
    /// Returns root + all descendants, ordered by depth then started_at.
    async fn get_run_tree(&self, root_run_id: Uuid) -> Result<Vec<ProtocolRun>>;

    /// Delete a protocol run.
    /// Returns true if the run existed and was deleted.
    async fn delete_protocol_run(&self, run_id: Uuid) -> Result<bool>;

    /// Create a PRODUCED_DURING relation between a Note/Decision and a ProtocolRun.
    /// entity_type must be "Note" or "Decision".
    async fn create_produced_during(
        &self,
        entity_type: &str,
        entity_id: Uuid,
        run_id: Uuid,
    ) -> Result<bool>;

    /// Get all notes and decisions produced during a protocol run.
    async fn get_run_outcomes(
        &self,
        run_id: Uuid,
    ) -> Result<Vec<crate::neo4j::protocol::ProducedArtefact>>;

    /// Find the most recently started active (running) protocol run for a project.
    async fn find_active_run_for_project(&self, project_id: Uuid) -> Result<Option<Uuid>>;

    /// Persist a ReasoningTree to Neo4j as a PersistedReasoningTree node.
    /// Optionally links it to an entity via REASONING_FOR relation.
    async fn persist_reasoning_tree(
        &self,
        tree: &crate::reasoning::ReasoningTree,
        linked_entity_type: Option<&str>,
        linked_entity_id: Option<Uuid>,
    ) -> Result<Uuid>;

    /// Get the ID of a persisted reasoning tree linked to a protocol run.
    async fn get_run_reasoning_tree_id(&self, run_id: Uuid) -> Result<Option<Uuid>>;

    /// List completed protocol runs for a project (most recent first).
    async fn list_completed_runs_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<crate::protocol::models::ProtocolRun>>;

    // ========================================================================
    // RuntimeState operations (Generator-produced dynamic states)
    // ========================================================================

    /// Create a new RuntimeState node with HAS_RUNTIME_STATE and GENERATED_BY relationships.
    async fn create_runtime_state(&self, state: &RuntimeState) -> Result<()>;

    /// Get all RuntimeStates for a protocol run, ordered by index.
    async fn get_runtime_states(&self, run_id: Uuid) -> Result<Vec<RuntimeState>>;

    /// Delete all RuntimeStates for a protocol run.
    async fn delete_runtime_states(&self, run_id: Uuid) -> Result<()>;

    // ========================================================================
    // System Inference — Audit Gaps
    // ========================================================================

    /// Audit the knowledge graph for gaps: orphan notes, decisions without AFFECTS,
    /// commits without TOUCHES, skills without members.
    async fn audit_knowledge_gaps(&self, project_id: Uuid) -> Result<AuditGapsReport>;

    // ========================================================================
    // Skill Registry
    // ========================================================================

    /// Create or update a published skill in the registry.
    async fn upsert_published_skill(
        &self,
        published: &crate::skills::registry::PublishedSkill,
    ) -> Result<()>;

    /// Get a published skill by ID.
    async fn get_published_skill(
        &self,
        id: Uuid,
    ) -> Result<Option<crate::skills::registry::PublishedSkill>>;

    /// Search published skills with optional filters.
    /// Returns (matching entries, total count).
    async fn search_published_skills(
        &self,
        query: Option<&str>,
        min_trust: Option<f64>,
        tags: Option<&[String]>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<crate::skills::registry::PublishedSkill>, usize)>;

    /// Increment the import count for a published skill.
    async fn increment_published_skill_imports(&self, id: Uuid) -> Result<()>;

    // ========================================================================
    // Graph visualization helpers (sub-file symbols, inheritance, constraints)
    // ========================================================================

    /// List all sub-file symbols (Function, Struct, Trait, Enum) for a project.
    /// Returns tuples of (id, name, symbol_type, file_path, visibility, line_start).
    async fn list_project_symbols(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<(String, String, String, String, Option<String>, Option<i64>)>>;

    /// List EXTENDS and IMPLEMENTS edges between structs/traits in a project.
    /// Returns (source_id, target_id, relation_type).
    async fn get_project_inheritance_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>>;

    /// Get all constraints for a project (via Project → Plan → Constraint).
    /// Returns (constraint, plan_id) pairs for graph visualization.
    async fn get_project_constraints(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(crate::neo4j::models::ConstraintNode, Uuid)>>;

    // ========================================================================
    // PlanRun operations (Runner execution state)
    // ========================================================================

    /// Create a new PlanRun node linked to a Plan via (:PlanRun)-[:RUNS]->(:Plan).
    async fn create_plan_run(&self, state: &crate::runner::RunnerState) -> Result<()>;

    /// Update an existing PlanRun with current execution state.
    async fn update_plan_run(&self, state: &crate::runner::RunnerState) -> Result<()>;

    /// Get a PlanRun by its run_id.
    async fn get_plan_run(&self, run_id: Uuid) -> Result<Option<crate::runner::RunnerState>>;

    /// List all PlanRuns with status=Running (for crash recovery at boot).
    async fn list_active_plan_runs(&self) -> Result<Vec<crate::runner::RunnerState>>;

    /// List all PlanRuns for a given plan, ordered by started_at desc.
    async fn list_plan_runs(
        &self,
        plan_id: Uuid,
        limit: i64,
    ) -> Result<Vec<crate::runner::RunnerState>>;

    // ── Triggers ──────────────────────────────────────────────────────────

    /// Create a trigger node linked to a plan via (:Trigger)-[:TRIGGERS]->(:Plan).
    async fn create_trigger(
        &self,
        trigger: &crate::runner::Trigger,
    ) -> Result<crate::runner::Trigger>;

    /// Get a trigger by its UUID.
    async fn get_trigger(&self, trigger_id: Uuid) -> Result<Option<crate::runner::Trigger>>;

    /// List all triggers for a given plan.
    async fn list_triggers(&self, plan_id: Uuid) -> Result<Vec<crate::runner::Trigger>>;

    /// Update a trigger (enabled, config, cooldown_secs).
    async fn update_trigger(
        &self,
        trigger_id: Uuid,
        enabled: Option<bool>,
        config: Option<serde_json::Value>,
        cooldown_secs: Option<u64>,
    ) -> Result<Option<crate::runner::Trigger>>;

    /// Delete a trigger by its UUID.
    async fn delete_trigger(&self, trigger_id: Uuid) -> Result<()>;

    /// Record a trigger firing event.
    async fn record_trigger_firing(&self, firing: &crate::runner::TriggerFiring) -> Result<()>;

    /// List all triggers across all plans, optionally filtered by type.
    async fn list_all_triggers(
        &self,
        trigger_type: Option<&str>,
    ) -> Result<Vec<crate::runner::Trigger>>;

    /// List trigger firings for a given trigger, ordered by fired_at desc.
    async fn list_trigger_firings(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> Result<Vec<crate::runner::TriggerFiring>>;

    // ── AgentExecution ──────────────────────────────────────────────────────

    /// Create an AgentExecution node linked to a PlanRun and Task.
    async fn create_agent_execution(
        &self,
        ae: &crate::neo4j::agent_execution::AgentExecutionNode,
    ) -> Result<()>;

    /// Update an existing AgentExecution with final results.
    async fn update_agent_execution(
        &self,
        ae: &crate::neo4j::agent_execution::AgentExecutionNode,
    ) -> Result<()>;

    /// Get all AgentExecution nodes for a given PlanRun.
    async fn get_agent_executions_for_run(
        &self,
        run_id: Uuid,
    ) -> Result<Vec<crate::neo4j::agent_execution::AgentExecutionNode>>;

    /// Create a USED_SKILL relationship from AgentExecution to Skill.
    async fn create_used_skill_relation(
        &self,
        agent_execution_id: Uuid,
        skill_id: Uuid,
        result: &str,
    ) -> Result<()>;
}
