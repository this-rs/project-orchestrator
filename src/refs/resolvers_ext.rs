//! The resolvers of the kinds added after the first five (see
//! [`super::resolvers`] for the rules every resolver follows): conversation,
//! project, milestone, release, workspace, commit, protocol, persona, skill,
//! file and link. The table of kinds points at the constructors below.

use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::access::RefMeta;
use super::label;
use super::resolvers::{scope_of, snake, Candidate, Candidates, KindResolver, Memo, SCAN_CAP};
use super::types::{RefId, RefKind};
use super::wire::ScopeLabel;
use crate::neo4j::models::{
    ChatSessionNode, CommitNode, MilestoneNode, PersonaNode, ProjectNode, ReleaseNode,
    WorkspaceNode,
};
use crate::neo4j::GraphStore;
use crate::protocol::Protocol;
use crate::skills::SkillNode;

/// The answer for a kind nobody registered: nothing is found, nothing is
/// suggested. `GraphRefSource` falls back on it so that a kind without a
/// resolver reads as `not_found` instead of panicking; a test pins that every
/// kind of the table has a real one.
pub struct Missing;

#[async_trait]
impl KindResolver for Missing {
    fn kind(&self) -> RefKind {
        RefKind::Link
    }
    async fn load(&self, _id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        Ok(None)
    }
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        Ok(vec![])
    }
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

/// The projects a search looks into: the named one, the projects of the named
/// workspace, or (no scope given) all of them, at most [`SCAN_CAP`].
async fn projects_in_scope(
    graph: &dyn GraphStore,
    c: &Candidates,
) -> anyhow::Result<Vec<ProjectNode>> {
    let mut projects = if let Some(id) = c.project_id {
        graph.get_project(id).await?.into_iter().collect()
    } else if let Some(slug) = c.workspace_slug.as_deref() {
        match graph.get_workspace_by_slug(slug).await? {
            Some(ws) => graph.list_workspace_projects(ws.id).await?,
            None => Vec::new(),
        }
    } else {
        graph.list_projects().await?
    };
    projects.truncate(SCAN_CAP);
    Ok(projects)
}

/// The name of an entity from its candidate texts, or `"<kind> <id prefix>"`.
fn named(kind: RefKind, id: &RefId, texts: &[&str]) -> String {
    texts
        .iter()
        .find_map(|t| label::first_line(t))
        .unwrap_or_else(|| {
            let short: String = id.as_str().chars().take(8).collect();
            format!("{kind} {short}")
        })
}

fn cand(meta: RefMeta, body: impl Into<String>, at: chrono::DateTime<chrono::Utc>) -> Candidate {
    Candidate::new(meta, body, Some(at))
}

macro_rules! uuid_or_none {
    ($id:expr) => {
        match $id.uuid() {
            Some(id) => id,
            None => return Ok(None),
        }
    };
}

// ----------------------------------------------------------------------------
// project
// ----------------------------------------------------------------------------

pub struct ProjectResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn project(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(ProjectResolver { graph })
}

impl ProjectResolver {
    async fn meta(&self, p: &ProjectNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        // `scope_of` names the project and its workspace; the project IS its own scope.
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(p.id)).await?;
        Ok(RefMeta {
            kind: RefKind::Project,
            id: p.id.into(),
            label: named(RefKind::Project, &p.id.into(), &[&p.name, &p.slug]),
            subtitle: Some(p.slug.clone()),
            project,
            workspace,
            entity_status: None,
        })
    }
}

#[async_trait]
impl KindResolver for ProjectResolver {
    fn kind(&self) -> RefKind {
        RefKind::Project
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_project(id).await? {
            Some(p) => Ok(Some(self.meta(&p, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for p in projects_in_scope(self.graph.as_ref(), c).await? {
            let body = format!("{} {}", p.slug, p.description.as_deref().unwrap_or(""));
            if c.matches(&p.name, &body) {
                out.push(cand(self.meta(&p, memo).await?, body, p.created_at));
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// workspace
// ----------------------------------------------------------------------------

pub struct WorkspaceResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn workspace(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(WorkspaceResolver { graph })
}

fn workspace_meta(w: &WorkspaceNode) -> RefMeta {
    RefMeta {
        kind: RefKind::Workspace,
        id: w.id.into(),
        label: named(RefKind::Workspace, &w.id.into(), &[&w.name, &w.slug]),
        subtitle: Some(w.slug.clone()),
        project: None,
        workspace: Some(ScopeLabel {
            id: w.id,
            slug: w.slug.clone(),
            name: w.name.clone(),
        }),
        entity_status: None,
    }
}

#[async_trait]
impl KindResolver for WorkspaceResolver {
    fn kind(&self) -> RefKind {
        RefKind::Workspace
    }

    async fn load(&self, id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        Ok(self
            .graph
            .get_workspace(id)
            .await?
            .map(|w| workspace_meta(&w)))
    }

    async fn candidates(&self, c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for w in self.graph.list_workspaces().await? {
            let body = format!("{} {}", w.slug, w.description.as_deref().unwrap_or(""));
            if c.matches(&w.name, &body) {
                out.push(cand(workspace_meta(&w), body, w.created_at));
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// milestone, release, protocol, skill, persona: owned by a project
// ----------------------------------------------------------------------------

pub struct MilestoneResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn milestone(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(MilestoneResolver { graph })
}

impl MilestoneResolver {
    async fn meta(&self, m: &MilestoneNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(m.project_id)).await?;
        Ok(RefMeta {
            kind: RefKind::Milestone,
            id: m.id.into(),
            label: named(RefKind::Milestone, &m.id.into(), &[&m.title]),
            subtitle: None,
            project,
            workspace,
            entity_status: snake(&m.status),
        })
    }
}

#[async_trait]
impl KindResolver for MilestoneResolver {
    fn kind(&self) -> RefKind {
        RefKind::Milestone
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_milestone(id).await? {
            Some(m) => Ok(Some(self.meta(&m, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for p in projects_in_scope(self.graph.as_ref(), c).await? {
            for m in self.graph.list_project_milestones(p.id).await? {
                let body = m.description.clone().unwrap_or_default();
                if c.matches(&m.title, &body) {
                    out.push(cand(self.meta(&m, memo).await?, body, m.created_at));
                }
            }
        }
        Ok(out)
    }
}

pub struct ReleaseResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn release(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(ReleaseResolver { graph })
}

impl ReleaseResolver {
    async fn meta(&self, r: &ReleaseNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(r.project_id)).await?;
        let title = r.title.as_deref().unwrap_or("");
        Ok(RefMeta {
            kind: RefKind::Release,
            id: r.id.into(),
            label: named(RefKind::Release, &r.id.into(), &[title, &r.version]),
            subtitle: (!title.is_empty()).then(|| r.version.clone()),
            project,
            workspace,
            entity_status: snake(&r.status),
        })
    }
}

#[async_trait]
impl KindResolver for ReleaseResolver {
    fn kind(&self) -> RefKind {
        RefKind::Release
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_release(id).await? {
            Some(r) => Ok(Some(self.meta(&r, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for p in projects_in_scope(self.graph.as_ref(), c).await? {
            for r in self.graph.list_project_releases(p.id).await? {
                let title = r.title.clone().unwrap_or_else(|| r.version.clone());
                let body = format!("{} {}", r.version, r.description.as_deref().unwrap_or(""));
                if c.matches(&title, &body) {
                    out.push(cand(self.meta(&r, memo).await?, body, r.created_at));
                }
            }
        }
        Ok(out)
    }
}

pub struct ProtocolResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn protocol(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(ProtocolResolver { graph })
}

impl ProtocolResolver {
    async fn meta(&self, p: &Protocol, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(p.project_id)).await?;
        Ok(RefMeta {
            kind: RefKind::Protocol,
            id: p.id.into(),
            label: named(RefKind::Protocol, &p.id.into(), &[&p.name]),
            subtitle: snake(&p.protocol_category),
            project,
            workspace,
            entity_status: None,
        })
    }
}

#[async_trait]
impl KindResolver for ProtocolResolver {
    fn kind(&self) -> RefKind {
        RefKind::Protocol
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_protocol(id).await? {
            Some(p) => Ok(Some(self.meta(&p, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for project in projects_in_scope(self.graph.as_ref(), c).await? {
            let (found, _) = self
                .graph
                .list_protocols(project.id, None, SCAN_CAP, 0)
                .await?;
            for p in found {
                if c.matches(&p.name, &p.description) {
                    out.push(cand(
                        self.meta(&p, memo).await?,
                        p.description.clone(),
                        p.created_at,
                    ));
                }
            }
        }
        Ok(out)
    }
}

pub struct SkillResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn skill(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(SkillResolver { graph })
}

impl SkillResolver {
    async fn meta(&self, s: &SkillNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(s.project_id)).await?;
        Ok(RefMeta {
            kind: RefKind::Skill,
            id: s.id.into(),
            label: named(RefKind::Skill, &s.id.into(), &[&s.name]),
            subtitle: label::first_line(&s.description),
            project,
            workspace,
            entity_status: snake(&s.status),
        })
    }
}

#[async_trait]
impl KindResolver for SkillResolver {
    fn kind(&self) -> RefKind {
        RefKind::Skill
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_skill(id).await? {
            Some(s) => Ok(Some(self.meta(&s, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for project in projects_in_scope(self.graph.as_ref(), c).await? {
            let (found, _) = self
                .graph
                .list_skills(project.id, None, SCAN_CAP, 0)
                .await?;
            for s in found {
                if c.matches(&s.name, &s.description) {
                    out.push(cand(
                        self.meta(&s, memo).await?,
                        s.description.clone(),
                        s.created_at,
                    ));
                }
            }
        }
        Ok(out)
    }
}

pub struct PersonaResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn persona(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(PersonaResolver { graph })
}

impl PersonaResolver {
    async fn meta(&self, p: &PersonaNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        // A persona may be global: then it has no project, and no scope.
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, p.project_id).await?;
        Ok(RefMeta {
            kind: RefKind::Persona,
            id: p.id.into(),
            label: named(RefKind::Persona, &p.id.into(), &[&p.name]),
            subtitle: label::first_line(&p.description),
            project,
            workspace,
            entity_status: snake(&p.status),
        })
    }
}

#[async_trait]
impl KindResolver for PersonaResolver {
    fn kind(&self) -> RefKind {
        RefKind::Persona
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_persona(id).await? {
            Some(p) => Ok(Some(self.meta(&p, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut found: Vec<PersonaNode> = Vec::new();
        for project in projects_in_scope(self.graph.as_ref(), c).await? {
            let (list, _) = self
                .graph
                .list_personas(project.id, None, SCAN_CAP, 0)
                .await?;
            found.extend(list);
        }
        // Global personas belong to no project: only a search that names no
        // scope looks at them (a scope in the query excludes what has none).
        if c.project_id.is_none() && c.workspace_slug.is_none() {
            found.extend(self.graph.list_global_personas().await?);
        }
        let mut out = Vec::new();
        for p in found {
            if c.matches(&p.name, &p.description) {
                out.push(cand(
                    self.meta(&p, memo).await?,
                    p.description.clone(),
                    p.created_at,
                ));
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// conversation
// ----------------------------------------------------------------------------

pub struct ConversationResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn conversation(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(ConversationResolver { graph })
}

impl ConversationResolver {
    /// `None` when the session names a project or a workspace that no longer
    /// exists: it cannot be placed, so it is not offered (fail closed).
    async fn meta(&self, s: &ChatSessionNode, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let (mut project, mut workspace) = (None, None);
        if let Some(slug) = s.project_slug.as_deref() {
            let Some(p) = self.graph.get_project_by_slug(slug).await? else {
                return Ok(None);
            };
            (project, workspace) = scope_of(self.graph.as_ref(), memo, Some(p.id)).await?;
        }
        if let Some(slug) = s.workspace_slug.as_deref() {
            let Some(w) = self.graph.get_workspace_by_slug(slug).await? else {
                return Ok(None);
            };
            if workspace.is_none() {
                workspace = Some(ScopeLabel {
                    id: w.id,
                    slug: w.slug,
                    name: w.name,
                });
            }
        }
        let title = s.title.as_deref().unwrap_or("");
        let preview = s.preview.as_deref().unwrap_or("");
        Ok(Some(RefMeta {
            kind: RefKind::Conversation,
            id: s.id.into(),
            label: named(RefKind::Conversation, &s.id.into(), &[title, preview]),
            subtitle: Some(match s.message_count {
                1 => "1 message".to_string(),
                n => format!("{n} messages"),
            }),
            project,
            workspace,
            entity_status: None,
        }))
    }
}

#[async_trait]
impl KindResolver for ConversationResolver {
    fn kind(&self) -> RefKind {
        RefKind::Conversation
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let id = uuid_or_none!(id);
        match self.graph.get_chat_session(id).await? {
            Some(s) => self.meta(&s, memo).await,
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let project_slug = match c.project_id {
            Some(id) => match self.graph.get_project(id).await? {
                Some(p) => Some(p.slug),
                None => return Ok(Vec::new()),
            },
            None => None,
        };
        let (sessions, _) = self
            .graph
            .list_chat_sessions(
                project_slug.as_deref(),
                c.workspace_slug.as_deref(),
                SCAN_CAP,
                0,
                false,
            )
            .await?;
        let mut out = Vec::new();
        for s in sessions {
            // The TITLE is what the user remembers; the preview only backs it up.
            let title = s.title.clone().unwrap_or_default();
            let preview = s.preview.clone().unwrap_or_default();
            if !c.matches(&title, &preview) {
                continue;
            }
            if let Some(meta) = self.meta(&s, memo).await? {
                out.push(cand(meta, preview, s.updated_at).titled(title));
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// commit
// ----------------------------------------------------------------------------

pub struct CommitResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn commit(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(CommitResolver { graph })
}

impl CommitResolver {
    /// The commits of a project: those linked to one of its plans or tasks
    /// (the graph has no other link from a commit to a project). Bounded by
    /// [`SCAN_CAP`] plans and tasks.
    async fn of_project(&self, project: Uuid) -> anyhow::Result<Vec<CommitNode>> {
        let (plans, _) = self
            .graph
            .list_plans_filtered(
                Some(project),
                None,
                None,
                None,
                None,
                None,
                SCAN_CAP,
                0,
                Some("created_at"),
                "desc",
            )
            .await?;
        let mut tasks = self.graph.get_project_tasks(project).await?;
        tasks.truncate(SCAN_CAP);
        let mut out: Vec<CommitNode> = Vec::new();
        let mut add = |list: Vec<CommitNode>| {
            for c in list {
                if !out.iter().any(|o| o.hash == c.hash) {
                    out.push(c);
                }
            }
        };
        for p in &plans {
            add(self.graph.get_plan_commits(p.id).await?);
        }
        for t in &tasks {
            add(self.graph.get_task_commits(t.id).await?);
        }
        Ok(out)
    }

    fn meta(
        &self,
        project: (Option<ScopeLabel>, Option<ScopeLabel>),
        id: RefId,
        c: &CommitNode,
    ) -> RefMeta {
        let short: String = c.hash.chars().take(7).collect();
        RefMeta {
            kind: RefKind::Commit,
            id,
            label: label::first_line(&c.message).unwrap_or_else(|| short.clone()),
            subtitle: Some(format!("{short} · {}", c.author)),
            project: project.0,
            workspace: project.1,
            entity_status: None,
        }
    }
}

#[async_trait]
impl KindResolver for CommitResolver {
    fn kind(&self) -> RefKind {
        RefKind::Commit
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let Some((project, hash)) = id.project_and_rest() else {
            return Ok(None);
        };
        if self.graph.get_project(project).await?.is_none() {
            return Ok(None);
        }
        let Some(commit) = self.graph.get_commit(hash).await? else {
            return Ok(None);
        };
        // The id carries the project the client claims; the graph decides.
        if !self
            .of_project(project)
            .await?
            .iter()
            .any(|c| c.hash == commit.hash)
        {
            return Ok(None);
        }
        let scope = scope_of(self.graph.as_ref(), memo, Some(project)).await?;
        Ok(Some(self.meta(scope, id.clone(), &commit)))
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for project in projects_in_scope(self.graph.as_ref(), c).await? {
            let scope = scope_of(self.graph.as_ref(), memo, Some(project.id)).await?;
            for commit in self.of_project(project.id).await? {
                let title = label::first_line(&commit.message).unwrap_or_default();
                if !c.matches(&title, &format!("{} {}", commit.message, commit.hash)) {
                    continue;
                }
                let Some(id) = super::types::canonical_id(
                    RefKind::Commit,
                    &format!("{}:{}", project.id, commit.hash),
                ) else {
                    continue;
                };
                out.push(cand(
                    self.meta(scope.clone(), id, &commit),
                    commit.message.clone(),
                    commit.timestamp,
                ));
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// file
// ----------------------------------------------------------------------------

pub struct FileResolver {
    graph: Arc<dyn GraphStore>,
}

pub fn file(graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(FileResolver { graph })
}

/// Files a reference never names, whatever the graph holds: anything hidden
/// (a segment starting with `.`: `.env`, `.git/`, `.ssh/`) and anything whose
/// name says it is a secret (keys, certificates, credentials). The picker
/// would otherwise print their path in a prompt.
pub fn is_private_path(rel: &str) -> bool {
    let lower = rel.to_lowercase();
    if lower.split('/').any(|seg| seg.starts_with('.')) {
        return true;
    }
    let name = lower.rsplit('/').next().unwrap_or("");
    const EXT: [&str; 7] = [".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".crt"];
    const PART: [&str; 7] = [
        "secret",
        "credential",
        "password",
        "passwd",
        "id_rsa",
        "id_ed25519",
        "token",
    ];
    EXT.iter().any(|e| name.ends_with(e)) || PART.iter().any(|p| name.contains(p))
}

impl FileResolver {
    fn rel_of<'a>(root: &str, abs: &'a str) -> Option<&'a str> {
        let root = root.trim_end_matches('/');
        abs.strip_prefix(root)?.strip_prefix('/')
    }

    fn meta(
        &self,
        scope: (Option<ScopeLabel>, Option<ScopeLabel>),
        project: Uuid,
        rel: &str,
        language: Option<&str>,
    ) -> Option<RefMeta> {
        let id = super::types::canonical_id(RefKind::File, &format!("{project}:{rel}"))?;
        Some(RefMeta {
            kind: RefKind::File,
            id,
            label: label::truncate(rel.rsplit('/').next().unwrap_or(rel)),
            subtitle: Some(label::truncate(rel)),
            project: scope.0,
            workspace: scope.1,
            entity_status: language.filter(|l| !l.is_empty()).map(str::to_string),
        })
    }
}

#[async_trait]
impl KindResolver for FileResolver {
    fn kind(&self) -> RefKind {
        RefKind::File
    }

    async fn load(&self, id: &RefId, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let Some((project, rel)) = id.project_and_rest() else {
            return Ok(None);
        };
        if is_private_path(rel) {
            return Ok(None);
        }
        let Some(p) = self.graph.get_project(project).await? else {
            return Ok(None);
        };
        let abs = format!("{}/{}", p.root_path.trim_end_matches('/'), rel);
        let Some(file) = self.graph.get_file(&abs).await? else {
            return Ok(None);
        };
        // The graph says which project owns the file; the id only claims it.
        if file.project_id != Some(project) {
            return Ok(None);
        }
        let scope = scope_of(self.graph.as_ref(), memo, Some(project)).await?;
        Ok(self.meta(scope, project, rel, Some(&file.language)))
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for project in projects_in_scope(self.graph.as_ref(), c).await? {
            let scope = scope_of(self.graph.as_ref(), memo, Some(project.id)).await?;
            let mut paths = self.graph.get_project_file_paths(project.id).await?;
            paths.truncate(SCAN_CAP * 10);
            for abs in &paths {
                let Some(rel) = Self::rel_of(&project.root_path, abs) else {
                    continue;
                };
                if is_private_path(rel) {
                    continue;
                }
                let name = rel.rsplit('/').next().unwrap_or(rel);
                if !c.matches(name, rel) {
                    continue;
                }
                if let Some(meta) = self.meta(scope.clone(), project.id, rel, None) {
                    out.push(Candidate::new(meta, rel, None));
                }
            }
        }
        Ok(out)
    }
}

/// The resolver of `link`. It holds no store and does no I/O: the answer is the
/// address itself, normalized by [`super::link`] (a pure parser, no lookup, no
/// request). The label is the host; reading the page is the agent's act and
/// goes through the consent by origin, not through here.
pub struct LinkResolver;

#[async_trait]
impl KindResolver for LinkResolver {
    fn kind(&self) -> RefKind {
        RefKind::Link
    }

    async fn load(&self, id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        // Only the canonical spelling is a link reference: anything else was
        // not built by the validation.
        let Ok(url) = super::link::normalize(id.as_str()) else {
            return Ok(None);
        };
        if url != id.as_str() {
            return Ok(None);
        }
        let Some(host) = super::link::host_label(&url) else {
            return Ok(None);
        };
        let shown = url
            .split_once("://")
            .map_or(url.as_str(), |(_, rest)| rest)
            .trim_end_matches('/');
        Ok(Some(RefMeta {
            kind: RefKind::Link,
            id: id.clone(),
            label: host,
            subtitle: Some(super::label::truncate(shown)),
            project: None,
            workspace: None,
            entity_status: None,
        }))
    }

    /// A link is pasted, not listed: nothing to suggest.
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        Ok(vec![])
    }
}

pub fn link(_graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(LinkResolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::GraphStore;
    use crate::refs::resolvers::GraphRefSource;
    use crate::refs::test_support::{user, world};
    use std::sync::Arc;

    #[tokio::test]
    async fn every_kind_has_a_resolver_of_its_own() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        for kind in RefKind::ALL {
            assert_eq!(source.resolver(kind).kind(), kind, "{kind}");
        }
    }
    // ---- link ---------------------------------------------------------

    use crate::refs::access::{AccessPolicy, Principal, Resolution, SessionScope};
    use crate::refs::types::EntityRef;

    async fn resolve(
        w: &crate::refs::test_support::World,
        p: &Principal,
        r: &EntityRef,
    ) -> Resolution {
        AccessPolicy::open_instance()
            .resolve_checked(p, &GraphRefSource::new(w.graph.clone()), r)
            .await
    }

    fn link_ref(url: &str) -> EntityRef {
        crate::refs::validate::validate_token(&format!("#link:{url}")).unwrap()
    }

    #[tokio::test]
    async fn a_link_resolves_to_its_host_without_touching_the_store() {
        let w = world().await;
        let r = link_ref("https://Docs.Example.com/guide?page=2#top");
        assert_eq!(r.id.as_str(), "https://docs.example.com/guide?page=2");
        let Resolution::Found(m) = resolve(&w, &user(), &r).await else {
            panic!("a valid link resolves");
        };
        assert_eq!(m.kind, RefKind::Link);
        assert_eq!(m.label, "docs.example.com");
        assert_eq!(m.subtitle.as_deref(), Some("docs.example.com/guide?page=2"));
        assert!(m.project.is_none() && m.workspace.is_none() && m.entity_status.is_none());
        // the resolver holds no store: it cannot read one.
        assert_eq!(std::mem::size_of::<LinkResolver>(), 0);
    }

    #[tokio::test]
    async fn a_link_is_shared_by_every_session_whatever_its_project() {
        let w = world().await;
        let r = link_ref("https://example.com/");
        for scope in [
            SessionScope::default(),
            SessionScope {
                project: Some(w.a),
                workspace: Some(w.ws.id),
            },
            SessionScope {
                project: Some(w.b),
                workspace: None,
            },
        ] {
            assert!(
                matches!(
                    resolve(&w, &Principal::Session(scope), &r).await,
                    Resolution::Found(_)
                ),
                "{scope:?}"
            );
        }
        assert_eq!(
            resolve(&w, &Principal::Unauthenticated, &r).await,
            Resolution::Forbidden
        );
    }

    #[tokio::test]
    async fn an_id_that_is_not_the_normalized_address_reads_not_found() {
        // Only validation builds a link reference; a stored block from elsewhere
        // may hold anything, and the resolver does not trust it.
        let w = world().await;
        let src = GraphRefSource::new(w.graph.clone());
        for bad in [
            "javascript:alert(1)",
            "https://user:pw@example.com/",
            "http://127.0.0.1/admin",
            "http://169.254.169.254/",
            "HTTPS://EXAMPLE.COM/",
            "https://example.com/#frag",
            "https://example.com/a b",
        ] {
            let id = serde_json::from_value::<RefId>(serde_json::json!(bad)).unwrap();
            let got = src
                .resolver(RefKind::Link)
                .load(&id, &mut Memo::default())
                .await
                .unwrap();
            assert!(got.is_none(), "{bad}");
        }
    }

    #[tokio::test]
    async fn a_link_is_not_searchable_and_the_search_does_not_fail() {
        let w = world().await;
        let q = crate::refs::search::RefSearchParams {
            q: Some("example".into()),
            kinds: Some("link,plan".into()),
            ..Default::default()
        }
        .into_query()
        .unwrap();
        let out = crate::refs::search::search(
            w.graph.clone(),
            &AccessPolicy::open_instance(),
            &user(),
            &q,
        )
        .await
        .unwrap();
        assert!(out.items.iter().all(|i| i.kind != RefKind::Link));
    }

    #[tokio::test]
    async fn a_turn_names_a_link_by_its_host_and_never_fetches_it() {
        let w = world().await;
        let r = link_ref("https://example.com/private-looking/path?token=abc");
        let g: Arc<dyn GraphStore> = w.graph.clone();
        let stored = crate::refs::block::encode("regarde", std::slice::from_ref(&r));
        let t = crate::refs::turn::expand_user_turn_if(&g, &stored, true).await;
        assert_eq!(t.resolved.len(), 1);
        assert_eq!(t.resolved[0].label.as_deref(), Some("example.com"));
        assert!(
            t.model_tail.contains("link \"example.com\""),
            "{}",
            t.model_tail
        );
    }
}
