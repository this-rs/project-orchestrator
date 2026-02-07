//! In-memory mock implementation of GraphStore for testing.
//!
//! Provides a complete mock of all graph operations using
//! `tokio::sync::RwLock<HashMap<K, V>>` collections.
//! Conditionally compiled with `#[cfg(test)]`.

use crate::neo4j::models::*;
use crate::neo4j::traits::GraphStore;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteFilters, NoteImportance, NoteStatus, PropagatedNote,
};
use crate::plan::models::{TaskDetails, UpdateTaskRequest};
use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// In-memory mock implementation of GraphStore for testing.
pub struct MockGraphStore {
    // Entity stores
    pub projects: RwLock<HashMap<Uuid, ProjectNode>>,
    pub workspaces: RwLock<HashMap<Uuid, WorkspaceNode>>,
    pub plans: RwLock<HashMap<Uuid, PlanNode>>,
    pub tasks: RwLock<HashMap<Uuid, TaskNode>>,
    pub steps: RwLock<HashMap<Uuid, StepNode>>,
    pub decisions: RwLock<HashMap<Uuid, DecisionNode>>,
    pub constraints: RwLock<HashMap<Uuid, ConstraintNode>>,
    pub commits: RwLock<HashMap<String, CommitNode>>,
    pub releases: RwLock<HashMap<Uuid, ReleaseNode>>,
    pub milestones: RwLock<HashMap<Uuid, MilestoneNode>>,
    pub workspace_milestones: RwLock<HashMap<Uuid, WorkspaceMilestoneNode>>,
    pub resources: RwLock<HashMap<Uuid, ResourceNode>>,
    pub components: RwLock<HashMap<Uuid, ComponentNode>>,
    pub files: RwLock<HashMap<String, FileNode>>,
    pub functions: RwLock<HashMap<String, FunctionNode>>,
    pub structs_map: RwLock<HashMap<String, StructNode>>,
    pub traits_map: RwLock<HashMap<String, TraitNode>>,
    pub enums_map: RwLock<HashMap<String, EnumNode>>,
    pub impls_map: RwLock<HashMap<String, ImplNode>>,
    pub imports: RwLock<HashMap<String, ImportNode>>,
    pub notes: RwLock<HashMap<Uuid, Note>>,
    pub chat_sessions: RwLock<HashMap<Uuid, ChatSessionNode>>,

    // Relationships (adjacency lists)
    pub plan_tasks: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub task_steps: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub task_decisions: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub task_dependencies: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub plan_constraints: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub project_plans: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub project_files: RwLock<HashMap<Uuid, Vec<String>>>,
    pub file_symbols: RwLock<HashMap<String, Vec<String>>>,
    pub task_files: RwLock<HashMap<Uuid, Vec<String>>>,
    pub task_commits: RwLock<HashMap<Uuid, Vec<String>>>,
    pub plan_commits: RwLock<HashMap<Uuid, Vec<String>>>,
    pub project_releases: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub project_milestones: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub release_tasks: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub release_commits: RwLock<HashMap<Uuid, Vec<String>>>,
    pub milestone_tasks: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub workspace_projects: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub workspace_ws_milestones: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub ws_milestone_tasks: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub workspace_resources: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub workspace_components: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub component_dependencies: RwLock<HashMap<Uuid, Vec<(Uuid, Option<String>, bool)>>>,
    pub component_projects: RwLock<HashMap<Uuid, Uuid>>,
    pub resource_implementers: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub resource_consumers: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    pub import_relationships: RwLock<HashMap<String, Vec<String>>>,
    pub call_relationships: RwLock<HashMap<String, Vec<String>>>,
    pub note_anchors: RwLock<HashMap<Uuid, Vec<NoteAnchor>>>,
    pub note_supersedes: RwLock<HashMap<Uuid, Uuid>>,
}

impl MockGraphStore {
    /// Create a new empty MockGraphStore.
    pub fn new() -> Self {
        Self {
            projects: RwLock::new(HashMap::new()),
            workspaces: RwLock::new(HashMap::new()),
            plans: RwLock::new(HashMap::new()),
            tasks: RwLock::new(HashMap::new()),
            steps: RwLock::new(HashMap::new()),
            decisions: RwLock::new(HashMap::new()),
            constraints: RwLock::new(HashMap::new()),
            commits: RwLock::new(HashMap::new()),
            releases: RwLock::new(HashMap::new()),
            milestones: RwLock::new(HashMap::new()),
            workspace_milestones: RwLock::new(HashMap::new()),
            resources: RwLock::new(HashMap::new()),
            components: RwLock::new(HashMap::new()),
            files: RwLock::new(HashMap::new()),
            functions: RwLock::new(HashMap::new()),
            structs_map: RwLock::new(HashMap::new()),
            traits_map: RwLock::new(HashMap::new()),
            enums_map: RwLock::new(HashMap::new()),
            impls_map: RwLock::new(HashMap::new()),
            imports: RwLock::new(HashMap::new()),
            notes: RwLock::new(HashMap::new()),
            chat_sessions: RwLock::new(HashMap::new()),
            plan_tasks: RwLock::new(HashMap::new()),
            task_steps: RwLock::new(HashMap::new()),
            task_decisions: RwLock::new(HashMap::new()),
            task_dependencies: RwLock::new(HashMap::new()),
            plan_constraints: RwLock::new(HashMap::new()),
            project_plans: RwLock::new(HashMap::new()),
            project_files: RwLock::new(HashMap::new()),
            file_symbols: RwLock::new(HashMap::new()),
            task_files: RwLock::new(HashMap::new()),
            task_commits: RwLock::new(HashMap::new()),
            plan_commits: RwLock::new(HashMap::new()),
            project_releases: RwLock::new(HashMap::new()),
            project_milestones: RwLock::new(HashMap::new()),
            release_tasks: RwLock::new(HashMap::new()),
            release_commits: RwLock::new(HashMap::new()),
            milestone_tasks: RwLock::new(HashMap::new()),
            workspace_projects: RwLock::new(HashMap::new()),
            workspace_ws_milestones: RwLock::new(HashMap::new()),
            ws_milestone_tasks: RwLock::new(HashMap::new()),
            workspace_resources: RwLock::new(HashMap::new()),
            workspace_components: RwLock::new(HashMap::new()),
            component_dependencies: RwLock::new(HashMap::new()),
            component_projects: RwLock::new(HashMap::new()),
            resource_implementers: RwLock::new(HashMap::new()),
            resource_consumers: RwLock::new(HashMap::new()),
            import_relationships: RwLock::new(HashMap::new()),
            call_relationships: RwLock::new(HashMap::new()),
            note_anchors: RwLock::new(HashMap::new()),
            note_supersedes: RwLock::new(HashMap::new()),
        }
    }

    // ========================================================================
    // Builder / seeding methods for tests
    // ========================================================================

    /// Seed a project into the store.
    pub async fn with_project(self, project: ProjectNode) -> Self {
        self.projects.write().await.insert(project.id, project);
        self
    }

    /// Seed a workspace into the store.
    pub async fn with_workspace(self, workspace: WorkspaceNode) -> Self {
        self.workspaces
            .write()
            .await
            .insert(workspace.id, workspace);
        self
    }

    /// Seed a plan into the store (optionally linking to project).
    pub async fn with_plan(self, plan: PlanNode) -> Self {
        let plan_id = plan.id;
        if let Some(project_id) = plan.project_id {
            self.project_plans
                .write()
                .await
                .entry(project_id)
                .or_default()
                .push(plan_id);
        }
        self.plans.write().await.insert(plan_id, plan);
        self
    }

    /// Seed a task linked to a plan.
    pub async fn with_task(self, plan_id: Uuid, task: TaskNode) -> Self {
        let task_id = task.id;
        self.plan_tasks
            .write()
            .await
            .entry(plan_id)
            .or_default()
            .push(task_id);
        self.tasks.write().await.insert(task_id, task);
        self
    }

    /// Seed a step linked to a task.
    pub async fn with_step(self, task_id: Uuid, step: StepNode) -> Self {
        let step_id = step.id;
        self.task_steps
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(step_id);
        self.steps.write().await.insert(step_id, step);
        self
    }

    /// Seed a decision linked to a task.
    pub async fn with_decision(self, task_id: Uuid, decision: DecisionNode) -> Self {
        let decision_id = decision.id;
        self.task_decisions
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(decision_id);
        self.decisions.write().await.insert(decision_id, decision);
        self
    }

    /// Seed a file node.
    pub async fn with_file(self, file: FileNode) -> Self {
        let path = file.path.clone();
        if let Some(pid) = file.project_id {
            self.project_files
                .write()
                .await
                .entry(pid)
                .or_default()
                .push(path.clone());
        }
        self.files.write().await.insert(path, file);
        self
    }

    /// Seed a note.
    pub async fn with_note(self, note: Note) -> Self {
        self.notes.write().await.insert(note.id, note);
        self
    }

    /// Seed a chat session.
    pub async fn with_chat_session(self, session: ChatSessionNode) -> Self {
        self.chat_sessions.write().await.insert(session.id, session);
        self
    }

    /// Seed a commit.
    pub async fn with_commit(self, commit: CommitNode) -> Self {
        self.commits
            .write()
            .await
            .insert(commit.hash.clone(), commit);
        self
    }

    /// Seed a release linked to a project.
    pub async fn with_release(self, release: ReleaseNode) -> Self {
        let release_id = release.id;
        let project_id = release.project_id;
        self.project_releases
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(release_id);
        self.releases.write().await.insert(release_id, release);
        self
    }

    /// Seed a milestone linked to a project.
    pub async fn with_milestone(self, milestone: MilestoneNode) -> Self {
        let ms_id = milestone.id;
        let project_id = milestone.project_id;
        self.project_milestones
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(ms_id);
        self.milestones.write().await.insert(ms_id, milestone);
        self
    }
}

// ============================================================================
// Helper: paginate a Vec
// ============================================================================
fn paginate<T: Clone>(items: &[T], limit: usize, offset: usize) -> Vec<T> {
    items.iter().skip(offset).take(limit).cloned().collect()
}

// ============================================================================
// GraphStore trait implementation
// ============================================================================

#[async_trait]
impl GraphStore for MockGraphStore {
    // ========================================================================
    // Project operations
    // ========================================================================

    async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        self.projects
            .write()
            .await
            .insert(project.id, project.clone());
        Ok(())
    }

    async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>> {
        Ok(self.projects.read().await.get(&id).cloned())
    }

    async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>> {
        Ok(self
            .projects
            .read()
            .await
            .values()
            .find(|p| p.slug == slug)
            .cloned())
    }

    async fn list_projects(&self) -> Result<Vec<ProjectNode>> {
        Ok(self.projects.read().await.values().cloned().collect())
    }

    async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()> {
        if let Some(p) = self.projects.write().await.get_mut(&id) {
            if let Some(n) = name {
                p.name = n;
            }
            if let Some(d) = description {
                p.description = d;
            }
            if let Some(r) = root_path {
                p.root_path = r;
            }
        }
        Ok(())
    }

    async fn update_project_synced(&self, id: Uuid) -> Result<()> {
        if let Some(p) = self.projects.write().await.get_mut(&id) {
            p.last_synced = Some(Utc::now());
        }
        Ok(())
    }

    async fn delete_project(&self, id: Uuid) -> Result<()> {
        self.projects.write().await.remove(&id);
        // Cascade: remove project files
        self.project_files.write().await.remove(&id);
        // Cascade: remove project plans and their children
        if let Some(plan_ids) = self.project_plans.write().await.remove(&id) {
            for plan_id in plan_ids {
                self.delete_plan(plan_id).await?;
            }
        }
        self.project_releases.write().await.remove(&id);
        self.project_milestones.write().await.remove(&id);
        Ok(())
    }

    // ========================================================================
    // Workspace operations
    // ========================================================================

    async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        self.workspaces
            .write()
            .await
            .insert(workspace.id, workspace.clone());
        Ok(())
    }

    async fn get_workspace(&self, id: Uuid) -> Result<Option<WorkspaceNode>> {
        Ok(self.workspaces.read().await.get(&id).cloned())
    }

    async fn get_workspace_by_slug(&self, slug: &str) -> Result<Option<WorkspaceNode>> {
        Ok(self
            .workspaces
            .read()
            .await
            .values()
            .find(|w| w.slug == slug)
            .cloned())
    }

    async fn list_workspaces(&self) -> Result<Vec<WorkspaceNode>> {
        Ok(self.workspaces.read().await.values().cloned().collect())
    }

    async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if let Some(w) = self.workspaces.write().await.get_mut(&id) {
            if let Some(n) = name {
                w.name = n;
            }
            if let Some(d) = description {
                w.description = Some(d);
            }
            if let Some(m) = metadata {
                w.metadata = m;
            }
            w.updated_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        self.workspaces.write().await.remove(&id);
        self.workspace_projects.write().await.remove(&id);
        self.workspace_ws_milestones.write().await.remove(&id);
        self.workspace_resources.write().await.remove(&id);
        self.workspace_components.write().await.remove(&id);
        Ok(())
    }

    async fn add_project_to_workspace(&self, workspace_id: Uuid, project_id: Uuid) -> Result<()> {
        self.workspace_projects
            .write()
            .await
            .entry(workspace_id)
            .or_default()
            .push(project_id);
        Ok(())
    }

    async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        if let Some(projects) = self.workspace_projects.write().await.get_mut(&workspace_id) {
            projects.retain(|p| *p != project_id);
        }
        Ok(())
    }

    async fn list_workspace_projects(&self, workspace_id: Uuid) -> Result<Vec<ProjectNode>> {
        let wp = self.workspace_projects.read().await;
        let projects = self.projects.read().await;
        let ids = wp.get(&workspace_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| projects.get(id).cloned())
            .collect())
    }

    async fn get_project_workspace(&self, project_id: Uuid) -> Result<Option<WorkspaceNode>> {
        let wp = self.workspace_projects.read().await;
        let workspaces = self.workspaces.read().await;
        for (ws_id, proj_ids) in wp.iter() {
            if proj_ids.contains(&project_id) {
                return Ok(workspaces.get(ws_id).cloned());
            }
        }
        Ok(None)
    }

    // ========================================================================
    // Workspace Milestone operations
    // ========================================================================

    async fn create_workspace_milestone(&self, milestone: &WorkspaceMilestoneNode) -> Result<()> {
        let ms_id = milestone.id;
        let ws_id = milestone.workspace_id;
        self.workspace_milestones
            .write()
            .await
            .insert(ms_id, milestone.clone());
        self.workspace_ws_milestones
            .write()
            .await
            .entry(ws_id)
            .or_default()
            .push(ms_id);
        Ok(())
    }

    async fn get_workspace_milestone(&self, id: Uuid) -> Result<Option<WorkspaceMilestoneNode>> {
        Ok(self.workspace_milestones.read().await.get(&id).cloned())
    }

    async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<WorkspaceMilestoneNode>> {
        let wm = self.workspace_ws_milestones.read().await;
        let milestones = self.workspace_milestones.read().await;
        let ids = wm.get(&workspace_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| milestones.get(id).cloned())
            .collect())
    }

    async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<WorkspaceMilestoneNode>, usize)> {
        let all = self.list_workspace_milestones(workspace_id).await?;
        let filtered: Vec<_> = all
            .into_iter()
            .filter(|m| {
                if let Some(s) = status {
                    let ms = serde_json::to_string(&m.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    ms == s
                } else {
                    true
                }
            })
            .collect();
        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn list_all_workspace_milestones_filtered(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(WorkspaceMilestoneNode, String, String, String)>> {
        let milestones = self.workspace_milestones.read().await;
        let workspaces = self.workspaces.read().await;
        let mut results: Vec<(WorkspaceMilestoneNode, String, String, String)> = Vec::new();
        for m in milestones.values() {
            if let Some(ws_id) = workspace_id {
                if m.workspace_id != ws_id {
                    continue;
                }
            }
            if let Some(s) = status {
                let ms = serde_json::to_string(&m.status)
                    .unwrap_or_default()
                    .trim_matches('"')
                    .to_string();
                if ms != s {
                    continue;
                }
            }
            let ws = workspaces.get(&m.workspace_id);
            let ws_id_str = m.workspace_id.to_string();
            let ws_name = ws.map(|w| w.name.clone()).unwrap_or_default();
            let ws_slug = ws.map(|w| w.slug.clone()).unwrap_or_default();
            results.push((m.clone(), ws_id_str, ws_name, ws_slug));
        }
        Ok(paginate(&results, limit, offset))
    }

    async fn count_all_workspace_milestones(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
    ) -> Result<usize> {
        let milestones = self.workspace_milestones.read().await;
        let count = milestones
            .values()
            .filter(|m| {
                if let Some(ws_id) = workspace_id {
                    if m.workspace_id != ws_id {
                        return false;
                    }
                }
                if let Some(s) = status {
                    let ms = serde_json::to_string(&m.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    if ms != s {
                        return false;
                    }
                }
                true
            })
            .count();
        Ok(count)
    }

    async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        if let Some(m) = self.workspace_milestones.write().await.get_mut(&id) {
            if let Some(t) = title {
                m.title = t;
            }
            if let Some(d) = description {
                m.description = Some(d);
            }
            if let Some(s) = status {
                m.status = s;
            }
            if let Some(td) = target_date {
                m.target_date = Some(td);
            }
        }
        Ok(())
    }

    async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        if let Some(m) = self.workspace_milestones.write().await.remove(&id) {
            if let Some(ids) = self
                .workspace_ws_milestones
                .write()
                .await
                .get_mut(&m.workspace_id)
            {
                ids.retain(|i| *i != id);
            }
        }
        self.ws_milestone_tasks.write().await.remove(&id);
        Ok(())
    }

    async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        self.ws_milestone_tasks
            .write()
            .await
            .entry(milestone_id)
            .or_default()
            .push(task_id);
        Ok(())
    }

    async fn remove_task_from_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        if let Some(tasks) = self.ws_milestone_tasks.write().await.get_mut(&milestone_id) {
            tasks.retain(|t| *t != task_id);
        }
        Ok(())
    }

    async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)> {
        let task_ids = self
            .ws_milestone_tasks
            .read()
            .await
            .get(&milestone_id)
            .cloned()
            .unwrap_or_default();
        let tasks = self.tasks.read().await;
        let mut total = 0u32;
        let mut completed = 0u32;
        let mut in_progress = 0u32;
        let mut pending = 0u32;
        for tid in &task_ids {
            if let Some(t) = tasks.get(tid) {
                total += 1;
                match t.status {
                    TaskStatus::Completed => completed += 1,
                    TaskStatus::InProgress => in_progress += 1,
                    TaskStatus::Pending => pending += 1,
                    _ => {}
                }
            }
        }
        Ok((total, completed, in_progress, pending))
    }

    async fn get_workspace_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>> {
        let task_ids = self
            .ws_milestone_tasks
            .read()
            .await
            .get(&milestone_id)
            .cloned()
            .unwrap_or_default();
        let tasks = self.tasks.read().await;
        Ok(task_ids
            .iter()
            .filter_map(|id| tasks.get(id).cloned())
            .collect())
    }

    // ========================================================================
    // Resource operations
    // ========================================================================

    async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        let rid = resource.id;
        if let Some(ws_id) = resource.workspace_id {
            self.workspace_resources
                .write()
                .await
                .entry(ws_id)
                .or_default()
                .push(rid);
        }
        self.resources.write().await.insert(rid, resource.clone());
        Ok(())
    }

    async fn get_resource(&self, id: Uuid) -> Result<Option<ResourceNode>> {
        Ok(self.resources.read().await.get(&id).cloned())
    }

    async fn list_workspace_resources(&self, workspace_id: Uuid) -> Result<Vec<ResourceNode>> {
        let wr = self.workspace_resources.read().await;
        let resources = self.resources.read().await;
        let ids = wr.get(&workspace_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| resources.get(id).cloned())
            .collect())
    }

    async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        if let Some(r) = self.resources.write().await.get_mut(&id) {
            if let Some(n) = name {
                r.name = n;
            }
            if let Some(fp) = file_path {
                r.file_path = fp;
            }
            if let Some(u) = url {
                r.url = Some(u);
            }
            if let Some(v) = version {
                r.version = Some(v);
            }
            if let Some(d) = description {
                r.description = Some(d);
            }
            r.updated_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn delete_resource(&self, id: Uuid) -> Result<()> {
        if let Some(r) = self.resources.write().await.remove(&id) {
            if let Some(ws_id) = r.workspace_id {
                if let Some(ids) = self.workspace_resources.write().await.get_mut(&ws_id) {
                    ids.retain(|i| *i != id);
                }
            }
        }
        self.resource_implementers.write().await.remove(&id);
        self.resource_consumers.write().await.remove(&id);
        Ok(())
    }

    async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        self.resource_implementers
            .write()
            .await
            .entry(resource_id)
            .or_default()
            .push(project_id);
        Ok(())
    }

    async fn link_project_uses_resource(&self, project_id: Uuid, resource_id: Uuid) -> Result<()> {
        self.resource_consumers
            .write()
            .await
            .entry(resource_id)
            .or_default()
            .push(project_id);
        Ok(())
    }

    async fn get_resource_implementers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let ri = self.resource_implementers.read().await;
        let projects = self.projects.read().await;
        let ids = ri.get(&resource_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| projects.get(id).cloned())
            .collect())
    }

    async fn get_resource_consumers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let rc = self.resource_consumers.read().await;
        let projects = self.projects.read().await;
        let ids = rc.get(&resource_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| projects.get(id).cloned())
            .collect())
    }

    // ========================================================================
    // Component operations (Topology)
    // ========================================================================

    async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        let cid = component.id;
        let ws_id = component.workspace_id;
        self.workspace_components
            .write()
            .await
            .entry(ws_id)
            .or_default()
            .push(cid);
        self.components.write().await.insert(cid, component.clone());
        Ok(())
    }

    async fn get_component(&self, id: Uuid) -> Result<Option<ComponentNode>> {
        Ok(self.components.read().await.get(&id).cloned())
    }

    async fn list_components(&self, workspace_id: Uuid) -> Result<Vec<ComponentNode>> {
        let wc = self.workspace_components.read().await;
        let components = self.components.read().await;
        let ids = wc.get(&workspace_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| components.get(id).cloned())
            .collect())
    }

    async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        if let Some(c) = self.components.write().await.get_mut(&id) {
            if let Some(n) = name {
                c.name = n;
            }
            if let Some(d) = description {
                c.description = Some(d);
            }
            if let Some(r) = runtime {
                c.runtime = Some(r);
            }
            if let Some(cfg) = config {
                c.config = cfg;
            }
            if let Some(t) = tags {
                c.tags = t;
            }
        }
        Ok(())
    }

    async fn delete_component(&self, id: Uuid) -> Result<()> {
        if let Some(c) = self.components.write().await.remove(&id) {
            if let Some(ids) = self
                .workspace_components
                .write()
                .await
                .get_mut(&c.workspace_id)
            {
                ids.retain(|i| *i != id);
            }
        }
        self.component_dependencies.write().await.remove(&id);
        self.component_projects.write().await.remove(&id);
        // Also remove this component from other components' dependency lists
        let mut cd = self.component_dependencies.write().await;
        for deps in cd.values_mut() {
            deps.retain(|(dep_id, _, _)| *dep_id != id);
        }
        Ok(())
    }

    async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        self.component_dependencies
            .write()
            .await
            .entry(component_id)
            .or_default()
            .push((depends_on_id, protocol, required));
        Ok(())
    }

    async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()> {
        if let Some(deps) = self
            .component_dependencies
            .write()
            .await
            .get_mut(&component_id)
        {
            deps.retain(|(id, _, _)| *id != depends_on_id);
        }
        Ok(())
    }

    async fn map_component_to_project(&self, component_id: Uuid, project_id: Uuid) -> Result<()> {
        self.component_projects
            .write()
            .await
            .insert(component_id, project_id);
        Ok(())
    }

    async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>> {
        let components = self.list_components(workspace_id).await?;
        let cd = self.component_dependencies.read().await;
        let cp = self.component_projects.read().await;
        let projects = self.projects.read().await;

        let mut result = Vec::new();
        for comp in components {
            let project_slug = cp
                .get(&comp.id)
                .and_then(|pid| projects.get(pid))
                .map(|p| p.slug.clone());
            let deps = cd
                .get(&comp.id)
                .map(|deps| {
                    deps.iter()
                        .map(|(to_id, protocol, required)| ComponentDependency {
                            from_id: comp.id,
                            to_id: *to_id,
                            protocol: protocol.clone(),
                            required: *required,
                        })
                        .collect()
                })
                .unwrap_or_default();
            result.push((comp, project_slug, deps));
        }
        Ok(result)
    }

    // ========================================================================
    // File operations
    // ========================================================================

    async fn get_project_file_paths(&self, project_id: Uuid) -> Result<Vec<String>> {
        Ok(self
            .project_files
            .read()
            .await
            .get(&project_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn delete_file(&self, path: &str) -> Result<()> {
        self.files.write().await.remove(path);
        // Remove from project_files
        let mut pf = self.project_files.write().await;
        for paths in pf.values_mut() {
            paths.retain(|p| p != path);
        }
        // Remove symbols linked to this file
        self.file_symbols.write().await.remove(path);
        self.import_relationships.write().await.remove(path);
        Ok(())
    }

    async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize)> {
        let current_paths = self
            .project_files
            .read()
            .await
            .get(&project_id)
            .cloned()
            .unwrap_or_default();
        let mut files_deleted = 0usize;
        let mut symbols_deleted = 0usize;
        for path in &current_paths {
            if !valid_paths.contains(path) {
                self.files.write().await.remove(path);
                if let Some(syms) = self.file_symbols.write().await.remove(path) {
                    symbols_deleted += syms.len();
                }
                files_deleted += 1;
            }
        }
        if let Some(paths) = self.project_files.write().await.get_mut(&project_id) {
            paths.retain(|p| valid_paths.contains(p));
        }
        Ok((files_deleted, symbols_deleted))
    }

    async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> Result<()> {
        self.project_files
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(file_path.to_string());
        Ok(())
    }

    async fn upsert_file(&self, file: &FileNode) -> Result<()> {
        self.files
            .write()
            .await
            .insert(file.path.clone(), file.clone());
        Ok(())
    }

    async fn get_file(&self, path: &str) -> Result<Option<FileNode>> {
        Ok(self.files.read().await.get(path).cloned())
    }

    async fn list_project_files(&self, project_id: Uuid) -> Result<Vec<FileNode>> {
        let pf = self.project_files.read().await;
        let files = self.files.read().await;
        let paths = pf.get(&project_id).cloned().unwrap_or_default();
        Ok(paths.iter().filter_map(|p| files.get(p).cloned()).collect())
    }

    // ========================================================================
    // Symbol operations
    // ========================================================================

    async fn upsert_function(&self, func: &FunctionNode) -> Result<()> {
        let id = format!("{}::{}", func.file_path, func.name);
        self.file_symbols
            .write()
            .await
            .entry(func.file_path.clone())
            .or_default()
            .push(id.clone());
        self.functions.write().await.insert(id, func.clone());
        Ok(())
    }

    async fn upsert_struct(&self, s: &StructNode) -> Result<()> {
        let id = format!("{}::{}", s.file_path, s.name);
        self.file_symbols
            .write()
            .await
            .entry(s.file_path.clone())
            .or_default()
            .push(id.clone());
        self.structs_map.write().await.insert(id, s.clone());
        Ok(())
    }

    async fn upsert_trait(&self, t: &TraitNode) -> Result<()> {
        let id = format!("{}::{}", t.file_path, t.name);
        self.file_symbols
            .write()
            .await
            .entry(t.file_path.clone())
            .or_default()
            .push(id.clone());
        self.traits_map.write().await.insert(id, t.clone());
        Ok(())
    }

    async fn find_trait_by_name(&self, name: &str) -> Result<Option<String>> {
        let traits = self.traits_map.read().await;
        for (id, t) in traits.iter() {
            if t.name == name {
                return Ok(Some(id.clone()));
            }
        }
        Ok(None)
    }

    async fn upsert_enum(&self, e: &EnumNode) -> Result<()> {
        let id = format!("{}::{}", e.file_path, e.name);
        self.file_symbols
            .write()
            .await
            .entry(e.file_path.clone())
            .or_default()
            .push(id.clone());
        self.enums_map.write().await.insert(id, e.clone());
        Ok(())
    }

    async fn upsert_impl(&self, impl_node: &ImplNode) -> Result<()> {
        let id = format!(
            "{}::impl_{}{}",
            impl_node.file_path,
            impl_node.for_type,
            impl_node
                .trait_name
                .as_ref()
                .map(|t| format!("_{}", t))
                .unwrap_or_default()
        );
        self.impls_map.write().await.insert(id, impl_node.clone());
        Ok(())
    }

    async fn create_import_relationship(
        &self,
        from_file: &str,
        to_file: &str,
        _import_path: &str,
    ) -> Result<()> {
        self.import_relationships
            .write()
            .await
            .entry(from_file.to_string())
            .or_default()
            .push(to_file.to_string());
        Ok(())
    }

    async fn upsert_import(&self, import: &ImportNode) -> Result<()> {
        let id = format!("{}::import_{}", import.file_path, import.path);
        self.imports.write().await.insert(id, import.clone());
        Ok(())
    }

    async fn create_call_relationship(&self, caller_id: &str, callee_name: &str) -> Result<()> {
        self.call_relationships
            .write()
            .await
            .entry(caller_id.to_string())
            .or_default()
            .push(callee_name.to_string());
        Ok(())
    }

    async fn get_callees(&self, function_id: &str, _depth: u32) -> Result<Vec<FunctionNode>> {
        let cr = self.call_relationships.read().await;
        let functions = self.functions.read().await;
        let callee_names = cr.get(function_id).cloned().unwrap_or_default();
        let mut result = Vec::new();
        for name in &callee_names {
            for (_, f) in functions.iter() {
                if f.name == *name {
                    result.push(f.clone());
                    break;
                }
            }
        }
        Ok(result)
    }

    async fn create_uses_type_relationship(
        &self,
        _function_id: &str,
        _type_name: &str,
    ) -> Result<()> {
        // Simplified: no-op for mock
        Ok(())
    }

    async fn find_trait_implementors(&self, trait_name: &str) -> Result<Vec<String>> {
        let impls = self.impls_map.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.trait_name.as_deref() == Some(trait_name) {
                result.push(imp.for_type.clone());
            }
        }
        Ok(result)
    }

    async fn get_type_traits(&self, type_name: &str) -> Result<Vec<String>> {
        let impls = self.impls_map.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.for_type == type_name {
                if let Some(t) = &imp.trait_name {
                    result.push(t.clone());
                }
            }
        }
        Ok(result)
    }

    async fn get_impl_blocks(&self, type_name: &str) -> Result<Vec<serde_json::Value>> {
        let impls = self.impls_map.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.for_type == type_name {
                result.push(serde_json::json!({
                    "for_type": imp.for_type,
                    "trait_name": imp.trait_name,
                    "file_path": imp.file_path,
                    "line_start": imp.line_start,
                    "line_end": imp.line_end,
                }));
            }
        }
        Ok(result)
    }

    // ========================================================================
    // Code exploration queries
    // ========================================================================

    async fn get_file_language(&self, path: &str) -> Result<Option<String>> {
        Ok(self
            .files
            .read()
            .await
            .get(path)
            .map(|f| f.language.clone()))
    }

    async fn get_file_functions_summary(&self, path: &str) -> Result<Vec<FunctionSummaryNode>> {
        let functions = self.functions.read().await;
        let mut result = Vec::new();
        for (id, f) in functions.iter() {
            if id.starts_with(&format!("{}::", path)) || f.file_path == path {
                let params_str = f
                    .params
                    .iter()
                    .map(|p| {
                        if let Some(t) = &p.type_name {
                            format!("{}: {}", p.name, t)
                        } else {
                            p.name.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let ret = f
                    .return_type
                    .as_ref()
                    .map(|r| format!(" -> {}", r))
                    .unwrap_or_default();
                let signature = format!("fn {}({}){}", f.name, params_str, ret);
                result.push(FunctionSummaryNode {
                    name: f.name.clone(),
                    signature,
                    line: f.line_start,
                    is_async: f.is_async,
                    is_public: f.visibility == Visibility::Public,
                    complexity: f.complexity,
                    docstring: f.docstring.clone(),
                });
            }
        }
        Ok(result)
    }

    async fn get_file_structs_summary(&self, path: &str) -> Result<Vec<StructSummaryNode>> {
        let structs = self.structs_map.read().await;
        let mut result = Vec::new();
        for s in structs.values() {
            if s.file_path == path {
                result.push(StructSummaryNode {
                    name: s.name.clone(),
                    line: s.line_start,
                    is_public: s.visibility == Visibility::Public,
                    docstring: s.docstring.clone(),
                });
            }
        }
        Ok(result)
    }

    async fn get_file_import_paths_list(&self, path: &str) -> Result<Vec<String>> {
        let imports = self.imports.read().await;
        let mut result = Vec::new();
        for imp in imports.values() {
            if imp.file_path == path {
                result.push(imp.path.clone());
            }
        }
        Ok(result)
    }

    async fn find_symbol_references(
        &self,
        _symbol: &str,
        _limit: usize,
    ) -> Result<Vec<SymbolReferenceNode>> {
        // Simplified: return empty in mock
        Ok(vec![])
    }

    async fn get_file_direct_imports(&self, path: &str) -> Result<Vec<FileImportNode>> {
        let ir = self.import_relationships.read().await;
        let files = self.files.read().await;
        let imported = ir.get(path).cloned().unwrap_or_default();
        Ok(imported
            .iter()
            .map(|p| {
                let lang = files
                    .get(p)
                    .map(|f| f.language.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                FileImportNode {
                    path: p.clone(),
                    language: lang,
                }
            })
            .collect())
    }

    async fn get_function_callers_by_name(
        &self,
        function_name: &str,
        _depth: u32,
    ) -> Result<Vec<String>> {
        let cr = self.call_relationships.read().await;
        let mut callers = Vec::new();
        for (caller, callees) in cr.iter() {
            if callees.contains(&function_name.to_string()) {
                callers.push(caller.clone());
            }
        }
        Ok(callers)
    }

    async fn get_function_callees_by_name(
        &self,
        function_name: &str,
        _depth: u32,
    ) -> Result<Vec<String>> {
        let cr = self.call_relationships.read().await;
        // Find the caller ID that matches the function name
        for (caller_id, callees) in cr.iter() {
            if caller_id.ends_with(&format!("::{}", function_name)) {
                return Ok(callees.clone());
            }
        }
        Ok(vec![])
    }

    async fn get_language_stats(&self) -> Result<Vec<LanguageStatsNode>> {
        let files = self.files.read().await;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for f in files.values() {
            *counts.entry(f.language.clone()).or_default() += 1;
        }
        Ok(counts
            .into_iter()
            .map(|(language, file_count)| LanguageStatsNode {
                language,
                file_count,
            })
            .collect())
    }

    async fn get_most_connected_files(&self, limit: usize) -> Result<Vec<String>> {
        let ir = self.import_relationships.read().await;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for targets in ir.values() {
            for target in targets {
                *counts.entry(target.clone()).or_default() += 1;
            }
        }
        let mut files: Vec<_> = counts.into_iter().collect();
        files.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(files.into_iter().take(limit).map(|(f, _)| f).collect())
    }

    async fn get_most_connected_files_detailed(
        &self,
        limit: usize,
    ) -> Result<Vec<ConnectedFileNode>> {
        let ir = self.import_relationships.read().await;
        let mut import_counts: HashMap<String, i64> = HashMap::new();
        let mut dependent_counts: HashMap<String, i64> = HashMap::new();

        for (from, tos) in ir.iter() {
            *import_counts.entry(from.clone()).or_default() += tos.len() as i64;
            for to in tos {
                *dependent_counts.entry(to.clone()).or_default() += 1;
            }
        }

        let all_files: std::collections::HashSet<_> = import_counts
            .keys()
            .chain(dependent_counts.keys())
            .cloned()
            .collect();

        let mut result: Vec<ConnectedFileNode> = all_files
            .into_iter()
            .map(|path| ConnectedFileNode {
                imports: *import_counts.get(&path).unwrap_or(&0),
                dependents: *dependent_counts.get(&path).unwrap_or(&0),
                path,
            })
            .collect();

        result.sort_by(|a, b| (b.imports + b.dependents).cmp(&(a.imports + a.dependents)));
        result.truncate(limit);
        Ok(result)
    }

    async fn get_file_symbol_names(&self, path: &str) -> Result<FileSymbolNamesNode> {
        let functions = self.functions.read().await;
        let structs = self.structs_map.read().await;
        let traits = self.traits_map.read().await;
        let enums = self.enums_map.read().await;

        let fn_names: Vec<String> = functions
            .values()
            .filter(|f| f.file_path == path)
            .map(|f| f.name.clone())
            .collect();
        let struct_names: Vec<String> = structs
            .values()
            .filter(|s| s.file_path == path)
            .map(|s| s.name.clone())
            .collect();
        let trait_names: Vec<String> = traits
            .values()
            .filter(|t| t.file_path == path)
            .map(|t| t.name.clone())
            .collect();
        let enum_names: Vec<String> = enums
            .values()
            .filter(|e| e.file_path == path)
            .map(|e| e.name.clone())
            .collect();

        Ok(FileSymbolNamesNode {
            functions: fn_names,
            structs: struct_names,
            traits: trait_names,
            enums: enum_names,
        })
    }

    async fn get_function_caller_count(&self, function_name: &str) -> Result<i64> {
        let cr = self.call_relationships.read().await;
        let mut count = 0i64;
        for callees in cr.values() {
            if callees.contains(&function_name.to_string()) {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn get_trait_info(&self, trait_name: &str) -> Result<Option<TraitInfoNode>> {
        let traits = self.traits_map.read().await;
        for t in traits.values() {
            if t.name == trait_name {
                return Ok(Some(TraitInfoNode {
                    is_external: t.is_external,
                    source: t.source.clone(),
                }));
            }
        }
        Ok(None)
    }

    async fn get_trait_implementors_detailed(
        &self,
        trait_name: &str,
    ) -> Result<Vec<TraitImplementorNode>> {
        let impls = self.impls_map.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.trait_name.as_deref() == Some(trait_name) {
                result.push(TraitImplementorNode {
                    type_name: imp.for_type.clone(),
                    file_path: imp.file_path.clone(),
                    line: imp.line_start,
                });
            }
        }
        Ok(result)
    }

    async fn get_type_trait_implementations(
        &self,
        type_name: &str,
    ) -> Result<Vec<TypeTraitInfoNode>> {
        let impls = self.impls_map.read().await;
        let traits = self.traits_map.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.for_type == type_name {
                if let Some(trait_name) = &imp.trait_name {
                    let trait_node = traits.values().find(|t| t.name == *trait_name);
                    result.push(TypeTraitInfoNode {
                        name: trait_name.clone(),
                        full_path: trait_node.map(|t| format!("{}::{}", t.file_path, t.name)),
                        file_path: imp.file_path.clone(),
                        is_external: trait_node.map(|t| t.is_external).unwrap_or(false),
                        source: trait_node.and_then(|t| t.source.clone()),
                    });
                }
            }
        }
        Ok(result)
    }

    async fn get_type_impl_blocks_detailed(
        &self,
        type_name: &str,
    ) -> Result<Vec<ImplBlockDetailNode>> {
        let impls = self.impls_map.read().await;
        let functions = self.functions.read().await;
        let mut result = Vec::new();
        for imp in impls.values() {
            if imp.for_type == type_name {
                // Find methods in this impl block by line range
                let methods: Vec<String> = functions
                    .values()
                    .filter(|f| {
                        f.file_path == imp.file_path
                            && f.line_start >= imp.line_start
                            && f.line_end <= imp.line_end
                    })
                    .map(|f| f.name.clone())
                    .collect();
                result.push(ImplBlockDetailNode {
                    file_path: imp.file_path.clone(),
                    line_start: imp.line_start,
                    line_end: imp.line_end,
                    trait_name: imp.trait_name.clone(),
                    methods,
                });
            }
        }
        Ok(result)
    }

    // ========================================================================
    // Plan operations
    // ========================================================================

    async fn create_plan(&self, plan: &PlanNode) -> Result<()> {
        let plan_id = plan.id;
        if let Some(project_id) = plan.project_id {
            self.project_plans
                .write()
                .await
                .entry(project_id)
                .or_default()
                .push(plan_id);
        }
        self.plans.write().await.insert(plan_id, plan.clone());
        Ok(())
    }

    async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>> {
        Ok(self.plans.read().await.get(&id).cloned())
    }

    async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        Ok(self
            .plans
            .read()
            .await
            .values()
            .filter(|p| {
                matches!(
                    p.status,
                    PlanStatus::Draft | PlanStatus::Approved | PlanStatus::InProgress
                )
            })
            .cloned()
            .collect())
    }

    async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        let pp = self.project_plans.read().await;
        let plans = self.plans.read().await;
        let ids = pp.get(&project_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| plans.get(id).cloned())
            .filter(|p| {
                matches!(
                    p.status,
                    PlanStatus::Draft | PlanStatus::Approved | PlanStatus::InProgress
                )
            })
            .collect())
    }

    async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)> {
        let pp = self.project_plans.read().await;
        let plans = self.plans.read().await;
        let ids = pp.get(&project_id).cloned().unwrap_or_default();
        let mut filtered: Vec<PlanNode> = ids
            .iter()
            .filter_map(|id| plans.get(id).cloned())
            .filter(|p| {
                if let Some(ref statuses) = status_filter {
                    let ps = serde_json::to_string(&p.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    statuses.contains(&ps)
                } else {
                    true
                }
            })
            .collect();
        filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()> {
        if let Some(p) = self.plans.write().await.get_mut(&id) {
            p.status = status;
        }
        Ok(())
    }

    async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        if let Some(p) = self.plans.write().await.get_mut(&plan_id) {
            // Remove from old project if any
            if let Some(old_pid) = p.project_id {
                if let Some(ids) = self.project_plans.write().await.get_mut(&old_pid) {
                    ids.retain(|id| *id != plan_id);
                }
            }
            p.project_id = Some(project_id);
        }
        self.project_plans
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(plan_id);
        Ok(())
    }

    async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        if let Some(p) = self.plans.write().await.get_mut(&plan_id) {
            if let Some(pid) = p.project_id.take() {
                if let Some(ids) = self.project_plans.write().await.get_mut(&pid) {
                    ids.retain(|id| *id != plan_id);
                }
            }
        }
        Ok(())
    }

    async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        // Cascade: delete tasks, steps, decisions, constraints
        if let Some(task_ids) = self.plan_tasks.write().await.remove(&plan_id) {
            for tid in task_ids {
                self.delete_task(tid).await?;
            }
        }
        if let Some(constraint_ids) = self.plan_constraints.write().await.remove(&plan_id) {
            let mut constraints = self.constraints.write().await;
            for cid in constraint_ids {
                constraints.remove(&cid);
            }
        }
        self.plan_commits.write().await.remove(&plan_id);
        if let Some(plan) = self.plans.write().await.remove(&plan_id) {
            if let Some(pid) = plan.project_id {
                if let Some(ids) = self.project_plans.write().await.get_mut(&pid) {
                    ids.retain(|id| *id != plan_id);
                }
            }
        }
        Ok(())
    }

    // ========================================================================
    // Task operations
    // ========================================================================

    async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> Result<()> {
        let task_id = task.id;
        self.plan_tasks
            .write()
            .await
            .entry(plan_id)
            .or_default()
            .push(task_id);
        self.tasks.write().await.insert(task_id, task.clone());
        Ok(())
    }

    async fn get_plan_tasks(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        let pt = self.plan_tasks.read().await;
        let tasks = self.tasks.read().await;
        let ids = pt.get(&plan_id).cloned().unwrap_or_default();
        Ok(ids.iter().filter_map(|id| tasks.get(id).cloned()).collect())
    }

    async fn get_task_with_full_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        let task = match self.tasks.read().await.get(&task_id).cloned() {
            Some(t) => t,
            None => return Ok(None),
        };
        let steps = self.get_task_steps(task_id).await?;
        let decisions = {
            let td = self.task_decisions.read().await;
            let decs = self.decisions.read().await;
            td.get(&task_id)
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(|id| decs.get(id).cloned())
                .collect()
        };
        let depends_on = self
            .task_dependencies
            .read()
            .await
            .get(&task_id)
            .cloned()
            .unwrap_or_default();
        let modifies_files = self
            .task_files
            .read()
            .await
            .get(&task_id)
            .cloned()
            .unwrap_or_default();
        Ok(Some(TaskDetails {
            task,
            steps,
            decisions,
            depends_on,
            modifies_files,
        }))
    }

    async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        let file_paths = self
            .task_files
            .read()
            .await
            .get(&task_id)
            .cloned()
            .unwrap_or_default();
        let ir = self.import_relationships.read().await;
        let mut impacted = Vec::new();
        for path in &file_paths {
            // Find files that import this file
            for (from, tos) in ir.iter() {
                if tos.contains(path) && !file_paths.contains(from) {
                    impacted.push(from.clone());
                }
            }
        }
        impacted.sort();
        impacted.dedup();
        Ok(impacted)
    }

    async fn find_blocked_tasks(&self, plan_id: Uuid) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        let task_ids = self
            .plan_tasks
            .read()
            .await
            .get(&plan_id)
            .cloned()
            .unwrap_or_default();
        let tasks = self.tasks.read().await;
        let deps = self.task_dependencies.read().await;

        let mut result = Vec::new();
        for tid in &task_ids {
            if let Some(task) = tasks.get(tid) {
                if task.status == TaskStatus::Pending {
                    if let Some(dep_ids) = deps.get(tid) {
                        let blockers: Vec<TaskNode> = dep_ids
                            .iter()
                            .filter_map(|did| {
                                tasks.get(did).and_then(|t| {
                                    if t.status != TaskStatus::Completed {
                                        Some(t.clone())
                                    } else {
                                        None
                                    }
                                })
                            })
                            .collect();
                        if !blockers.is_empty() {
                            result.push((task.clone(), blockers));
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        if let Some(t) = self.tasks.write().await.get_mut(&task_id) {
            t.status = status.clone();
            t.updated_at = Some(Utc::now());
            if status == TaskStatus::InProgress && t.started_at.is_none() {
                t.started_at = Some(Utc::now());
            }
            if status == TaskStatus::Completed {
                t.completed_at = Some(Utc::now());
            }
        }
        Ok(())
    }

    async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> Result<()> {
        if let Some(t) = self.tasks.write().await.get_mut(&task_id) {
            t.assigned_to = Some(agent_id.to_string());
            t.updated_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.task_dependencies
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(depends_on_id);
        Ok(())
    }

    async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        if let Some(deps) = self.task_dependencies.write().await.get_mut(&task_id) {
            deps.retain(|id| *id != depends_on_id);
        }
        Ok(())
    }

    async fn get_task_blockers(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let deps = self.task_dependencies.read().await;
        let tasks = self.tasks.read().await;
        let dep_ids = deps.get(&task_id).cloned().unwrap_or_default();
        Ok(dep_ids
            .iter()
            .filter_map(|did| {
                tasks.get(did).and_then(|t| {
                    if t.status != TaskStatus::Completed {
                        Some(t.clone())
                    } else {
                        None
                    }
                })
            })
            .collect())
    }

    async fn get_tasks_blocked_by(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let deps = self.task_dependencies.read().await;
        let tasks = self.tasks.read().await;
        let mut result = Vec::new();
        for (tid, dep_ids) in deps.iter() {
            if dep_ids.contains(&task_id) {
                if let Some(t) = tasks.get(tid) {
                    result.push(t.clone());
                }
            }
        }
        Ok(result)
    }

    async fn get_task_dependencies(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let deps = self.task_dependencies.read().await;
        let tasks = self.tasks.read().await;
        let dep_ids = deps.get(&task_id).cloned().unwrap_or_default();
        Ok(dep_ids
            .iter()
            .filter_map(|did| tasks.get(did).cloned())
            .collect())
    }

    async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        let task_ids = self
            .plan_tasks
            .read()
            .await
            .get(&plan_id)
            .cloned()
            .unwrap_or_default();
        let tasks_map = self.tasks.read().await;
        let deps = self.task_dependencies.read().await;

        let task_list: Vec<TaskNode> = task_ids
            .iter()
            .filter_map(|id| tasks_map.get(id).cloned())
            .collect();

        let mut edges = Vec::new();
        for tid in &task_ids {
            if let Some(dep_ids) = deps.get(tid) {
                for did in dep_ids {
                    edges.push((*tid, *did));
                }
            }
        }

        Ok((task_list, edges))
    }

    async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        let (tasks, edges) = self.get_plan_dependency_graph(plan_id).await?;
        if tasks.is_empty() {
            return Ok(vec![]);
        }

        // Simple longest-path: DFS from each node with no dependents
        let task_map: HashMap<Uuid, &TaskNode> = tasks.iter().map(|t| (t.id, t)).collect();
        // Build adjacency: task -> tasks it depends on
        let mut adj: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for (from, to) in &edges {
            adj.entry(*from).or_default().push(*to);
        }

        fn dfs(
            node: Uuid,
            adj: &HashMap<Uuid, Vec<Uuid>>,
            memo: &mut HashMap<Uuid, Vec<Uuid>>,
        ) -> Vec<Uuid> {
            if let Some(cached) = memo.get(&node) {
                return cached.clone();
            }
            let mut longest = vec![];
            if let Some(deps) = adj.get(&node) {
                for dep in deps {
                    let path = dfs(*dep, adj, memo);
                    if path.len() > longest.len() {
                        longest = path;
                    }
                }
            }
            longest.push(node);
            memo.insert(node, longest.clone());
            longest
        }

        let mut memo = HashMap::new();
        let mut best_path = vec![];
        for t in &tasks {
            let path = dfs(t.id, &adj, &mut memo);
            if path.len() > best_path.len() {
                best_path = path;
            }
        }

        // Reverse so it goes from root dependency to final task
        best_path.reverse();
        Ok(best_path
            .iter()
            .filter_map(|id| task_map.get(id).cloned().cloned())
            .collect())
    }

    async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        let task_ids = self
            .plan_tasks
            .read()
            .await
            .get(&plan_id)
            .cloned()
            .unwrap_or_default();
        let tasks = self.tasks.read().await;
        let deps = self.task_dependencies.read().await;

        for tid in &task_ids {
            if let Some(task) = tasks.get(tid) {
                if task.status != TaskStatus::Pending {
                    continue;
                }
                // Check all dependencies are completed
                let dep_ids = deps.get(tid).cloned().unwrap_or_default();
                let all_deps_completed = dep_ids.iter().all(|did| {
                    tasks
                        .get(did)
                        .map(|t| t.status == TaskStatus::Completed)
                        .unwrap_or(true)
                });
                if all_deps_completed {
                    return Ok(Some(task.clone()));
                }
            }
        }
        Ok(None)
    }

    async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>> {
        Ok(self.tasks.read().await.get(&task_id).cloned())
    }

    async fn update_task(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> Result<()> {
        if let Some(t) = self.tasks.write().await.get_mut(&task_id) {
            if let Some(ref title) = updates.title {
                t.title = Some(title.clone());
            }
            if let Some(ref desc) = updates.description {
                t.description = desc.clone();
            }
            if let Some(ref status) = updates.status {
                t.status = status.clone();
                if *status == TaskStatus::InProgress && t.started_at.is_none() {
                    t.started_at = Some(Utc::now());
                }
                if *status == TaskStatus::Completed {
                    t.completed_at = Some(Utc::now());
                }
            }
            if let Some(ref assigned) = updates.assigned_to {
                t.assigned_to = Some(assigned.clone());
            }
            if let Some(priority) = updates.priority {
                t.priority = Some(priority);
            }
            if let Some(ref tags) = updates.tags {
                t.tags = tags.clone();
            }
            if let Some(ref ac) = updates.acceptance_criteria {
                t.acceptance_criteria = ac.clone();
            }
            if let Some(ref af) = updates.affected_files {
                t.affected_files = af.clone();
            }
            if let Some(complexity) = updates.actual_complexity {
                t.actual_complexity = Some(complexity);
            }
            t.updated_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        self.tasks.write().await.remove(&task_id);
        // Cascade: steps
        if let Some(step_ids) = self.task_steps.write().await.remove(&task_id) {
            let mut steps = self.steps.write().await;
            for sid in step_ids {
                steps.remove(&sid);
            }
        }
        // Cascade: decisions
        if let Some(dec_ids) = self.task_decisions.write().await.remove(&task_id) {
            let mut decisions = self.decisions.write().await;
            for did in dec_ids {
                decisions.remove(&did);
            }
        }
        self.task_dependencies.write().await.remove(&task_id);
        self.task_files.write().await.remove(&task_id);
        self.task_commits.write().await.remove(&task_id);
        // Remove from other tasks' dependency lists
        let mut deps = self.task_dependencies.write().await;
        for dep_list in deps.values_mut() {
            dep_list.retain(|id| *id != task_id);
        }
        Ok(())
    }

    // ========================================================================
    // Step operations
    // ========================================================================

    async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        let step_id = step.id;
        self.task_steps
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(step_id);
        self.steps.write().await.insert(step_id, step.clone());
        Ok(())
    }

    async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>> {
        let ts = self.task_steps.read().await;
        let steps = self.steps.read().await;
        let ids = ts.get(&task_id).cloned().unwrap_or_default();
        let mut result: Vec<StepNode> =
            ids.iter().filter_map(|id| steps.get(id).cloned()).collect();
        result.sort_by_key(|s| s.order);
        Ok(result)
    }

    async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        if let Some(s) = self.steps.write().await.get_mut(&step_id) {
            s.status = status.clone();
            s.updated_at = Some(Utc::now());
            if status == StepStatus::Completed {
                s.completed_at = Some(Utc::now());
            }
        }
        Ok(())
    }

    async fn get_task_step_progress(&self, task_id: Uuid) -> Result<(u32, u32)> {
        let steps = self.get_task_steps(task_id).await?;
        let total = steps.len() as u32;
        let completed = steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count() as u32;
        Ok((completed, total))
    }

    async fn get_step(&self, step_id: Uuid) -> Result<Option<StepNode>> {
        Ok(self.steps.read().await.get(&step_id).cloned())
    }

    async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        self.steps.write().await.remove(&step_id);
        let mut ts = self.task_steps.write().await;
        for step_ids in ts.values_mut() {
            step_ids.retain(|id| *id != step_id);
        }
        Ok(())
    }

    // ========================================================================
    // Constraint operations
    // ========================================================================

    async fn create_constraint(&self, plan_id: Uuid, constraint: &ConstraintNode) -> Result<()> {
        let cid = constraint.id;
        self.plan_constraints
            .write()
            .await
            .entry(plan_id)
            .or_default()
            .push(cid);
        self.constraints
            .write()
            .await
            .insert(cid, constraint.clone());
        Ok(())
    }

    async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        let pc = self.plan_constraints.read().await;
        let constraints = self.constraints.read().await;
        let ids = pc.get(&plan_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| constraints.get(id).cloned())
            .collect())
    }

    async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>> {
        Ok(self.constraints.read().await.get(&constraint_id).cloned())
    }

    async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        if let Some(c) = self.constraints.write().await.get_mut(&constraint_id) {
            if let Some(d) = description {
                c.description = d;
            }
            if let Some(ct) = constraint_type {
                c.constraint_type = ct;
            }
            if let Some(eb) = enforced_by {
                c.enforced_by = Some(eb);
            }
        }
        Ok(())
    }

    async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        self.constraints.write().await.remove(&constraint_id);
        let mut pc = self.plan_constraints.write().await;
        for ids in pc.values_mut() {
            ids.retain(|id| *id != constraint_id);
        }
        Ok(())
    }

    // ========================================================================
    // Decision operations
    // ========================================================================

    async fn create_decision(&self, task_id: Uuid, decision: &DecisionNode) -> Result<()> {
        let did = decision.id;
        self.task_decisions
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(did);
        self.decisions.write().await.insert(did, decision.clone());
        Ok(())
    }

    async fn get_decision(&self, decision_id: Uuid) -> Result<Option<DecisionNode>> {
        Ok(self.decisions.read().await.get(&decision_id).cloned())
    }

    async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> Result<()> {
        if let Some(d) = self.decisions.write().await.get_mut(&decision_id) {
            if let Some(desc) = description {
                d.description = desc;
            }
            if let Some(rat) = rationale {
                d.rationale = rat;
            }
            if let Some(co) = chosen_option {
                d.chosen_option = Some(co);
            }
        }
        Ok(())
    }

    async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        self.decisions.write().await.remove(&decision_id);
        let mut td = self.task_decisions.write().await;
        for ids in td.values_mut() {
            ids.retain(|id| *id != decision_id);
        }
        Ok(())
    }

    // ========================================================================
    // Dependency analysis
    // ========================================================================

    async fn find_dependent_files(&self, file_path: &str, _depth: u32) -> Result<Vec<String>> {
        let ir = self.import_relationships.read().await;
        let mut dependents = Vec::new();
        for (from, tos) in ir.iter() {
            if tos.contains(&file_path.to_string()) {
                dependents.push(from.clone());
            }
        }
        Ok(dependents)
    }

    async fn find_callers(&self, function_id: &str) -> Result<Vec<FunctionNode>> {
        let cr = self.call_relationships.read().await;
        let functions = self.functions.read().await;
        // Extract function name from id
        let func_name = function_id.rsplit("::").next().unwrap_or(function_id);
        let mut callers = Vec::new();
        for (caller_id, callees) in cr.iter() {
            if callees.contains(&func_name.to_string()) {
                if let Some(f) = functions.get(caller_id) {
                    callers.push(f.clone());
                }
            }
        }
        Ok(callers)
    }

    // ========================================================================
    // Task-file linking
    // ========================================================================

    async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> Result<()> {
        self.task_files
            .write()
            .await
            .entry(task_id)
            .or_default()
            .extend(file_paths.iter().cloned());
        Ok(())
    }

    // ========================================================================
    // Commit operations
    // ========================================================================

    async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        self.commits
            .write()
            .await
            .insert(commit.hash.clone(), commit.clone());
        Ok(())
    }

    async fn get_commit(&self, hash: &str) -> Result<Option<CommitNode>> {
        Ok(self.commits.read().await.get(hash).cloned())
    }

    async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        self.task_commits
            .write()
            .await
            .entry(task_id)
            .or_default()
            .push(commit_hash.to_string());
        Ok(())
    }

    async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        self.plan_commits
            .write()
            .await
            .entry(plan_id)
            .or_default()
            .push(commit_hash.to_string());
        Ok(())
    }

    async fn get_task_commits(&self, task_id: Uuid) -> Result<Vec<CommitNode>> {
        let tc = self.task_commits.read().await;
        let commits = self.commits.read().await;
        let hashes = tc.get(&task_id).cloned().unwrap_or_default();
        Ok(hashes
            .iter()
            .filter_map(|h| commits.get(h).cloned())
            .collect())
    }

    async fn get_plan_commits(&self, plan_id: Uuid) -> Result<Vec<CommitNode>> {
        let pc = self.plan_commits.read().await;
        let commits = self.commits.read().await;
        let hashes = pc.get(&plan_id).cloned().unwrap_or_default();
        Ok(hashes
            .iter()
            .filter_map(|h| commits.get(h).cloned())
            .collect())
    }

    async fn delete_commit(&self, hash: &str) -> Result<()> {
        self.commits.write().await.remove(hash);
        let mut tc = self.task_commits.write().await;
        for hashes in tc.values_mut() {
            hashes.retain(|h| h != hash);
        }
        let mut pc = self.plan_commits.write().await;
        for hashes in pc.values_mut() {
            hashes.retain(|h| h != hash);
        }
        Ok(())
    }

    // ========================================================================
    // Release operations
    // ========================================================================

    async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        let rid = release.id;
        let pid = release.project_id;
        self.project_releases
            .write()
            .await
            .entry(pid)
            .or_default()
            .push(rid);
        self.releases.write().await.insert(rid, release.clone());
        Ok(())
    }

    async fn get_release(&self, id: Uuid) -> Result<Option<ReleaseNode>> {
        Ok(self.releases.read().await.get(&id).cloned())
    }

    async fn list_project_releases(&self, project_id: Uuid) -> Result<Vec<ReleaseNode>> {
        let pr = self.project_releases.read().await;
        let releases = self.releases.read().await;
        let ids = pr.get(&project_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| releases.get(id).cloned())
            .collect())
    }

    async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        if let Some(r) = self.releases.write().await.get_mut(&id) {
            if let Some(s) = status {
                r.status = s;
            }
            if let Some(td) = target_date {
                r.target_date = Some(td);
            }
            if let Some(ra) = released_at {
                r.released_at = Some(ra);
            }
            if let Some(t) = title {
                r.title = Some(t);
            }
            if let Some(d) = description {
                r.description = Some(d);
            }
        }
        Ok(())
    }

    async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()> {
        self.release_tasks
            .write()
            .await
            .entry(release_id)
            .or_default()
            .push(task_id);
        Ok(())
    }

    async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()> {
        self.release_commits
            .write()
            .await
            .entry(release_id)
            .or_default()
            .push(commit_hash.to_string());
        Ok(())
    }

    async fn get_release_details(
        &self,
        release_id: Uuid,
    ) -> Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>> {
        let release = match self.releases.read().await.get(&release_id).cloned() {
            Some(r) => r,
            None => return Ok(None),
        };
        let task_ids = self
            .release_tasks
            .read()
            .await
            .get(&release_id)
            .cloned()
            .unwrap_or_default();
        let tasks_map = self.tasks.read().await;
        let tasks: Vec<TaskNode> = task_ids
            .iter()
            .filter_map(|id| tasks_map.get(id).cloned())
            .collect();
        let commit_hashes = self
            .release_commits
            .read()
            .await
            .get(&release_id)
            .cloned()
            .unwrap_or_default();
        let commits_map = self.commits.read().await;
        let commits: Vec<CommitNode> = commit_hashes
            .iter()
            .filter_map(|h| commits_map.get(h).cloned())
            .collect();
        Ok(Some((release, tasks, commits)))
    }

    async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        if let Some(r) = self.releases.write().await.remove(&release_id) {
            if let Some(ids) = self.project_releases.write().await.get_mut(&r.project_id) {
                ids.retain(|id| *id != release_id);
            }
        }
        self.release_tasks.write().await.remove(&release_id);
        self.release_commits.write().await.remove(&release_id);
        Ok(())
    }

    // ========================================================================
    // Milestone operations
    // ========================================================================

    async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        let mid = milestone.id;
        let pid = milestone.project_id;
        self.project_milestones
            .write()
            .await
            .entry(pid)
            .or_default()
            .push(mid);
        self.milestones.write().await.insert(mid, milestone.clone());
        Ok(())
    }

    async fn get_milestone(&self, id: Uuid) -> Result<Option<MilestoneNode>> {
        Ok(self.milestones.read().await.get(&id).cloned())
    }

    async fn list_project_milestones(&self, project_id: Uuid) -> Result<Vec<MilestoneNode>> {
        let pm = self.project_milestones.read().await;
        let milestones = self.milestones.read().await;
        let ids = pm.get(&project_id).cloned().unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| milestones.get(id).cloned())
            .collect())
    }

    async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        if let Some(m) = self.milestones.write().await.get_mut(&id) {
            if let Some(s) = status {
                m.status = s;
            }
            if let Some(td) = target_date {
                m.target_date = Some(td);
            }
            if let Some(ca) = closed_at {
                m.closed_at = Some(ca);
            }
            if let Some(t) = title {
                m.title = t;
            }
            if let Some(d) = description {
                m.description = Some(d);
            }
        }
        Ok(())
    }

    async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        self.milestone_tasks
            .write()
            .await
            .entry(milestone_id)
            .or_default()
            .push(task_id);
        Ok(())
    }

    async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>> {
        let milestone = match self.milestones.read().await.get(&milestone_id).cloned() {
            Some(m) => m,
            None => return Ok(None),
        };
        let tasks = self.get_milestone_tasks(milestone_id).await?;
        Ok(Some((milestone, tasks)))
    }

    async fn get_milestone_progress(&self, milestone_id: Uuid) -> Result<(u32, u32)> {
        let tasks = self.get_milestone_tasks(milestone_id).await?;
        let total = tasks.len() as u32;
        let completed = tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count() as u32;
        Ok((completed, total))
    }

    async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        if let Some(m) = self.milestones.write().await.remove(&milestone_id) {
            if let Some(ids) = self.project_milestones.write().await.get_mut(&m.project_id) {
                ids.retain(|id| *id != milestone_id);
            }
        }
        self.milestone_tasks.write().await.remove(&milestone_id);
        Ok(())
    }

    async fn get_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>> {
        let mt = self.milestone_tasks.read().await;
        let tasks = self.tasks.read().await;
        let ids = mt.get(&milestone_id).cloned().unwrap_or_default();
        Ok(ids.iter().filter_map(|id| tasks.get(id).cloned()).collect())
    }

    async fn get_release_tasks(&self, release_id: Uuid) -> Result<Vec<TaskNode>> {
        let rt = self.release_tasks.read().await;
        let tasks = self.tasks.read().await;
        let ids = rt.get(&release_id).cloned().unwrap_or_default();
        Ok(ids.iter().filter_map(|id| tasks.get(id).cloned()).collect())
    }

    // ========================================================================
    // Project stats
    // ========================================================================

    async fn get_project_progress(&self, project_id: Uuid) -> Result<(u32, u32, u32, u32)> {
        let tasks = self.get_project_tasks(project_id).await?;
        let total = tasks.len() as u32;
        let completed = tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count() as u32;
        let in_progress = tasks
            .iter()
            .filter(|t| t.status == TaskStatus::InProgress)
            .count() as u32;
        let pending = tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Pending)
            .count() as u32;
        Ok((total, completed, in_progress, pending))
    }

    async fn get_project_task_dependencies(&self, project_id: Uuid) -> Result<Vec<(Uuid, Uuid)>> {
        let plan_ids = self
            .project_plans
            .read()
            .await
            .get(&project_id)
            .cloned()
            .unwrap_or_default();
        let pt = self.plan_tasks.read().await;
        let deps = self.task_dependencies.read().await;

        let mut result = Vec::new();
        for plan_id in &plan_ids {
            let task_ids = pt.get(plan_id).cloned().unwrap_or_default();
            for tid in &task_ids {
                if let Some(dep_ids) = deps.get(tid) {
                    for did in dep_ids {
                        result.push((*tid, *did));
                    }
                }
            }
        }
        Ok(result)
    }

    async fn get_project_tasks(&self, project_id: Uuid) -> Result<Vec<TaskNode>> {
        let plan_ids = self
            .project_plans
            .read()
            .await
            .get(&project_id)
            .cloned()
            .unwrap_or_default();
        let pt = self.plan_tasks.read().await;
        let tasks = self.tasks.read().await;

        let mut result = Vec::new();
        for plan_id in &plan_ids {
            let task_ids = pt.get(plan_id).cloned().unwrap_or_default();
            for tid in &task_ids {
                if let Some(t) = tasks.get(tid) {
                    result.push(t.clone());
                }
            }
        }
        Ok(result)
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
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)> {
        let plans = self.plans.read().await;
        let pp = self.project_plans.read().await;

        let plan_ids_for_project: Option<Vec<Uuid>> =
            project_id.map(|pid| pp.get(&pid).cloned().unwrap_or_default());

        let filtered: Vec<PlanNode> = plans
            .values()
            .filter(|p| {
                if let Some(ref ids) = plan_ids_for_project {
                    if !ids.contains(&p.id) {
                        return false;
                    }
                }
                if let Some(ref sts) = statuses {
                    let ps = serde_json::to_string(&p.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    if !sts.contains(&ps) {
                        return false;
                    }
                }
                if let Some(min) = priority_min {
                    if p.priority < min {
                        return false;
                    }
                }
                if let Some(max) = priority_max {
                    if p.priority > max {
                        return false;
                    }
                }
                if let Some(q) = search {
                    let q = q.to_lowercase();
                    if !p.title.to_lowercase().contains(&q)
                        && !p.description.to_lowercase().contains(&q)
                    {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
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
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<TaskWithPlan>, usize)> {
        let pt = self.plan_tasks.read().await;
        let tasks = self.tasks.read().await;
        let plans = self.plans.read().await;

        // Build task_id -> plan_id mapping
        let mut task_plan_map: HashMap<Uuid, Uuid> = HashMap::new();
        for (pid, tids) in pt.iter() {
            for tid in tids {
                task_plan_map.insert(*tid, *pid);
            }
        }

        let filtered: Vec<TaskWithPlan> = tasks
            .values()
            .filter(|t| {
                if let Some(pid) = plan_id {
                    if task_plan_map.get(&t.id) != Some(&pid) {
                        return false;
                    }
                }
                if let Some(ref sts) = statuses {
                    let ts = serde_json::to_string(&t.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    if !sts.contains(&ts) {
                        return false;
                    }
                }
                if let Some(min) = priority_min {
                    if t.priority.unwrap_or(0) < min {
                        return false;
                    }
                }
                if let Some(max) = priority_max {
                    if t.priority.unwrap_or(0) > max {
                        return false;
                    }
                }
                if let Some(ref tag_filter) = tags {
                    if !tag_filter.iter().any(|tg| t.tags.contains(tg)) {
                        return false;
                    }
                }
                if let Some(agent) = assigned_to {
                    if t.assigned_to.as_deref() != Some(agent) {
                        return false;
                    }
                }
                true
            })
            .filter_map(|t| {
                let pid = task_plan_map.get(&t.id)?;
                let plan = plans.get(pid)?;
                Some(TaskWithPlan {
                    task: t.clone(),
                    plan_id: *pid,
                    plan_title: plan.title.clone(),
                })
            })
            .collect();

        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<ReleaseNode>, usize)> {
        let all = self.list_project_releases(project_id).await?;
        let filtered: Vec<ReleaseNode> = all
            .into_iter()
            .filter(|r| {
                if let Some(ref sts) = statuses {
                    let rs = serde_json::to_string(&r.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    sts.contains(&rs)
                } else {
                    true
                }
            })
            .collect();
        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<MilestoneNode>, usize)> {
        let all = self.list_project_milestones(project_id).await?;
        let filtered: Vec<MilestoneNode> = all
            .into_iter()
            .filter(|m| {
                if let Some(ref sts) = statuses {
                    let ms = serde_json::to_string(&m.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    sts.contains(&ms)
                } else {
                    true
                }
            })
            .collect();
        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<ProjectNode>, usize)> {
        let projects = self.projects.read().await;
        let filtered: Vec<ProjectNode> = projects
            .values()
            .filter(|p| {
                if let Some(q) = search {
                    let q = q.to_lowercase();
                    p.name.to_lowercase().contains(&q)
                        || p.slug.to_lowercase().contains(&q)
                        || p.description
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(&q)
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        let total = filtered.len();
        Ok((paginate(&filtered, limit, offset), total))
    }

    // ========================================================================
    // Knowledge Note operations
    // ========================================================================

    async fn create_note(&self, note: &Note) -> Result<()> {
        self.notes.write().await.insert(note.id, note.clone());
        Ok(())
    }

    async fn get_note(&self, id: Uuid) -> Result<Option<Note>> {
        Ok(self.notes.read().await.get(&id).cloned())
    }

    async fn update_note(
        &self,
        id: Uuid,
        content: Option<String>,
        importance: Option<NoteImportance>,
        status: Option<NoteStatus>,
        tags: Option<Vec<String>>,
        staleness_score: Option<f64>,
    ) -> Result<Option<Note>> {
        let mut notes = self.notes.write().await;
        if let Some(n) = notes.get_mut(&id) {
            if let Some(c) = content {
                n.content = c;
            }
            if let Some(i) = importance {
                n.importance = i;
            }
            if let Some(s) = status {
                n.status = s;
            }
            if let Some(t) = tags {
                n.tags = t;
            }
            if let Some(ss) = staleness_score {
                n.staleness_score = ss;
            }
            Ok(Some(n.clone()))
        } else {
            Ok(None)
        }
    }

    async fn delete_note(&self, id: Uuid) -> Result<bool> {
        let removed = self.notes.write().await.remove(&id).is_some();
        self.note_anchors.write().await.remove(&id);
        Ok(removed)
    }

    async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        let notes = self.notes.read().await;
        let filtered: Vec<Note> = notes
            .values()
            .filter(|n| {
                if let Some(pid) = project_id {
                    if n.project_id != pid {
                        return false;
                    }
                }
                if let Some(ref statuses) = filters.status {
                    if !statuses.contains(&n.status) {
                        return false;
                    }
                }
                if let Some(ref note_types) = filters.note_type {
                    if !note_types.contains(&n.note_type) {
                        return false;
                    }
                }
                if let Some(ref importances) = filters.importance {
                    if !importances.contains(&n.importance) {
                        return false;
                    }
                }
                if let Some(ref tag_filter) = filters.tags {
                    if !tag_filter.iter().any(|tg| n.tags.contains(tg)) {
                        return false;
                    }
                }
                if let Some(ref q) = filters.search {
                    let q = q.to_lowercase();
                    if !n.content.to_lowercase().contains(&q) {
                        return false;
                    }
                }
                if let Some(min_s) = filters.min_staleness {
                    if n.staleness_score < min_s {
                        return false;
                    }
                }
                if let Some(max_s) = filters.max_staleness {
                    if n.staleness_score > max_s {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();
        let total = filtered.len();
        let limit = filters.limit.unwrap_or(50) as usize;
        let offset = filters.offset.unwrap_or(0) as usize;
        Ok((paginate(&filtered, limit, offset), total))
    }

    async fn link_note_to_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> Result<()> {
        let anchor = NoteAnchor {
            entity_type: entity_type.clone(),
            entity_id: entity_id.to_string(),
            signature_hash: signature_hash.map(|s| s.to_string()),
            body_hash: body_hash.map(|s| s.to_string()),
            last_verified: Utc::now(),
            is_valid: true,
        };
        self.note_anchors
            .write()
            .await
            .entry(note_id)
            .or_default()
            .push(anchor);
        Ok(())
    }

    async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        if let Some(anchors) = self.note_anchors.write().await.get_mut(&note_id) {
            anchors.retain(|a| !(&a.entity_type == entity_type && a.entity_id == entity_id));
        }
        Ok(())
    }

    async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<Note>> {
        let anchors = self.note_anchors.read().await;
        let notes = self.notes.read().await;
        let mut result = Vec::new();
        for (note_id, note_anchors) in anchors.iter() {
            for a in note_anchors {
                if &a.entity_type == entity_type && a.entity_id == entity_id {
                    if let Some(n) = notes.get(note_id) {
                        result.push(n.clone());
                    }
                    break;
                }
            }
        }
        Ok(result)
    }

    async fn get_propagated_notes(
        &self,
        _entity_type: &EntityType,
        _entity_id: &str,
        _max_depth: u32,
        _min_score: f64,
    ) -> Result<Vec<PropagatedNote>> {
        // Simplified: propagation requires graph traversal; return empty
        Ok(vec![])
    }

    async fn get_workspace_notes_for_project(
        &self,
        project_id: Uuid,
        propagation_factor: f64,
    ) -> Result<Vec<PropagatedNote>> {
        // Find workspace for this project
        let ws = self.get_project_workspace(project_id).await?;
        if let Some(workspace) = ws {
            let ws_notes = self
                .get_notes_for_entity(&EntityType::Workspace, &workspace.id.to_string())
                .await?;
            Ok(ws_notes
                .into_iter()
                .map(|n| PropagatedNote {
                    relevance_score: propagation_factor,
                    source_entity: format!("workspace:{}", workspace.slug),
                    propagation_path: vec![format!("workspace:{}", workspace.slug)],
                    distance: 1,
                    note: n,
                })
                .collect())
        } else {
            Ok(vec![])
        }
    }

    async fn supersede_note(&self, old_note_id: Uuid, new_note_id: Uuid) -> Result<()> {
        self.note_supersedes
            .write()
            .await
            .insert(old_note_id, new_note_id);
        // Update old note status
        if let Some(n) = self.notes.write().await.get_mut(&old_note_id) {
            n.status = NoteStatus::Obsolete;
            n.superseded_by = Some(new_note_id);
        }
        // Update new note supersedes field
        if let Some(n) = self.notes.write().await.get_mut(&new_note_id) {
            n.supersedes = Some(old_note_id);
        }
        Ok(())
    }

    async fn confirm_note(&self, note_id: Uuid, confirmed_by: &str) -> Result<Option<Note>> {
        let mut notes = self.notes.write().await;
        if let Some(n) = notes.get_mut(&note_id) {
            n.confirm(confirmed_by);
            Ok(Some(n.clone()))
        } else {
            Ok(None)
        }
    }

    async fn get_notes_needing_review(&self, project_id: Option<Uuid>) -> Result<Vec<Note>> {
        let notes = self.notes.read().await;
        Ok(notes
            .values()
            .filter(|n| {
                if let Some(pid) = project_id {
                    if n.project_id != pid {
                        return false;
                    }
                }
                matches!(n.status, NoteStatus::NeedsReview | NoteStatus::Stale)
            })
            .cloned()
            .collect())
    }

    async fn update_staleness_scores(&self) -> Result<usize> {
        let mut notes = self.notes.write().await;
        let mut count = 0;
        for n in notes.values_mut() {
            if n.status == NoteStatus::Active {
                // Simple staleness: time since last confirmed
                if let Some(confirmed_at) = n.last_confirmed_at {
                    let days = (Utc::now() - confirmed_at).num_days() as f64;
                    let base_days = n.base_decay_days();
                    let decay = n.importance.decay_factor();
                    n.staleness_score = (days * decay / base_days).min(1.0).max(0.0);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    async fn get_note_anchors(&self, note_id: Uuid) -> Result<Vec<NoteAnchor>> {
        Ok(self
            .note_anchors
            .read()
            .await
            .get(&note_id)
            .cloned()
            .unwrap_or_default())
    }

    // ========================================================================
    // Chat session operations
    // ========================================================================

    async fn create_chat_session(&self, session: &ChatSessionNode) -> Result<()> {
        self.chat_sessions
            .write()
            .await
            .insert(session.id, session.clone());
        Ok(())
    }

    async fn get_chat_session(&self, id: Uuid) -> Result<Option<ChatSessionNode>> {
        Ok(self.chat_sessions.read().await.get(&id).cloned())
    }

    async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ChatSessionNode>, usize)> {
        let sessions = self.chat_sessions.read().await;
        let mut filtered: Vec<_> = sessions
            .values()
            .filter(|s| {
                if let Some(slug) = project_slug {
                    s.project_slug.as_deref() == Some(slug)
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        // Sort by updated_at descending
        filtered.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        let total = filtered.len();
        let page = paginate(&filtered, limit, offset);
        Ok((page, total))
    }

    async fn update_chat_session(
        &self,
        id: Uuid,
        cli_session_id: Option<String>,
        title: Option<String>,
        message_count: Option<i64>,
        total_cost_usd: Option<f64>,
    ) -> Result<Option<ChatSessionNode>> {
        let mut sessions = self.chat_sessions.write().await;
        if let Some(session) = sessions.get_mut(&id) {
            session.updated_at = Utc::now();
            if let Some(v) = cli_session_id {
                session.cli_session_id = Some(v);
            }
            if let Some(v) = title {
                session.title = Some(v);
            }
            if let Some(v) = message_count {
                session.message_count = v;
            }
            if let Some(v) = total_cost_usd {
                session.total_cost_usd = Some(v);
            }
            Ok(Some(session.clone()))
        } else {
            Ok(None)
        }
    }

    async fn delete_chat_session(&self, id: Uuid) -> Result<bool> {
        Ok(self.chat_sessions.write().await.remove(&id).is_some())
    }
}
