//! Spawn envelope — what a session opened BY AN AGENT may be.
//!
//! A chat session can open another one: the MCP `chat(send_message)` action and
//! `plan(delegate_task)` both end in `ChatManager::create_session`. Until this
//! module, the child was whatever the calling model asked for — any permission
//! mode (delegation even forced `bypassPermissions`), any working directory, a
//! parent named in the request body, no depth limit. A session in Ask mode could
//! therefore open a full-shell child without a single prompt.
//!
//! The rule (decision A17): **a child is never more than its parent.**
//!
//! * the parent is read from the caller's SIGNED session token, never from the
//!   request body;
//! * the child's permission mode is the requested one clamped to the parent's;
//! * its working directory and extra directories stay inside the parent's;
//! * its project is the parent's;
//! * a child cannot itself spawn (depth 1), and a parent has at most
//!   [`MAX_LIVE_CHILDREN`] children alive.
//!
//! This is enforced on the server (REST handlers). The MCP tool handlers run in
//! the agent's own subprocess and are not a boundary.

use crate::auth::jwt::Claims;
use crate::chat::manager::ChatManager;
use crate::chat::types::{ChatRequest, SpawnedBy};
use crate::neo4j::traits::GraphStore;
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

/// How deep a spawned session may sit under a human-opened (or runner-opened)
/// one. 1 = a child may exist, a grandchild may not.
pub const MAX_DELEGATION_DEPTH: u32 = 1;

/// How many children of one parent may be alive at once.
pub const MAX_LIVE_CHILDREN: usize = 4;

/// How long a freshly opened child counts toward its parent's fan-out even before its first
/// turn is seen running.
const CHILD_STARTUP_GRACE: chrono::Duration = chrono::Duration::seconds(120);

/// Header the MCP client sets from `PO_SESSION_ID`. Only read when the server
/// runs WITHOUT authentication (no signing key, hence no signed token): in that
/// mode nothing is a boundary and the header is the best available hint.
pub const SESSION_ID_HEADER: &str = "x-session-id";

/// Why a spawn was refused. Every variant maps to HTTP 403 with a stable code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// An agent token that is not bound to a session cannot open one.
    UnboundAgentToken,
    /// The session named by the token is unknown.
    ParentNotFound(String),
    /// The caller is itself a spawned session.
    DepthExceeded { max: u32 },
    /// The parent already has the maximum number of live children.
    TooManyChildren { max: usize, live: usize },
    /// The requested working directory is outside the parent's.
    CwdOutsideParent { cwd: String },
    /// A requested extra directory is outside the parent's.
    AddDirOutsideParent { dir: String },
    /// The requested project is not the parent's.
    ProjectMismatch { requested: String, parent: String },
    /// The requested workspace is not the parent's.
    WorkspaceMismatch,
    /// An agent may only send to sessions it spawned.
    NotAChild { session_id: String },
}

impl EnvelopeError {
    /// Stable machine-readable code (documented in the error table).
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnboundAgentToken => "envelope_unbound_token",
            Self::ParentNotFound(_) => "envelope_parent_not_found",
            Self::DepthExceeded { .. } => "envelope_depth_exceeded",
            Self::TooManyChildren { .. } => "envelope_too_many_children",
            Self::CwdOutsideParent { .. } => "envelope_cwd_outside_parent",
            Self::AddDirOutsideParent { .. } => "envelope_add_dir_outside_parent",
            Self::ProjectMismatch { .. } => "envelope_project_mismatch",
            Self::WorkspaceMismatch => "envelope_workspace_mismatch",
            Self::NotAChild { .. } => "envelope_not_a_child",
        }
    }
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.code())?;
        match self {
            Self::UnboundAgentToken => {
                write!(
                    f,
                    "this agent token is not bound to a session and cannot open one"
                )
            }
            Self::ParentNotFound(id) => write!(f, "calling session {id} not found"),
            Self::DepthExceeded { max } => write!(
                f,
                "a spawned session cannot spawn another one (maximum depth {max})"
            ),
            Self::TooManyChildren { max, live } => write!(
                f,
                "{live} child sessions are already running (maximum {max}); wait for one to finish"
            ),
            Self::CwdOutsideParent { cwd } => write!(
                f,
                "working directory {cwd} is outside the calling session's directories"
            ),
            Self::AddDirOutsideParent { dir } => write!(
                f,
                "directory {dir} is outside the calling session's directories"
            ),
            Self::ProjectMismatch { requested, parent } => write!(
                f,
                "project {requested} is not the calling session's project ({parent})"
            ),
            Self::WorkspaceMismatch => {
                write!(f, "the workspace is not the calling session's workspace")
            }
            Self::NotAChild { session_id } => write!(
                f,
                "session {session_id} was not spawned by the calling session"
            ),
        }
    }
}

impl std::error::Error for EnvelopeError {}

impl From<EnvelopeError> for crate::api::handlers::AppError {
    fn from(e: EnvelopeError) -> Self {
        crate::api::handlers::AppError::Forbidden(e.to_string())
    }
}

// ============================================================================
// Permission modes: a total order, least to most permissive
// ============================================================================

/// Rank of a permission mode, least (0) to most permissive. Both the Claude
/// strings and the neutral ones (decision A43) are understood. An unknown mode
/// ranks as `default`, which is also what `PermissionConfig::to_nexus_mode`
/// turns it into.
/// The rank of trust (`trust`, `bypassPermissions`): [`mode_rank`] of the most permissive mode.
pub const TRUST_RANK: u8 = 4;

pub fn mode_rank(mode: &str) -> u8 {
    match mode {
        "plan" | "plan_only" => 0,
        "default" | "manual" | "dontAsk" | "ask" => 1,
        "acceptEdits" | "auto_edits" => 2,
        "auto" => 3,
        "bypassPermissions" | "trust" => 4,
        _ => 1,
    }
}

fn is_known_mode(mode: &str) -> bool {
    matches!(
        mode,
        "plan"
            | "plan_only"
            | "default"
            | "manual"
            | "dontAsk"
            | "ask"
            | "acceptEdits"
            | "auto_edits"
            | "auto"
            | "bypassPermissions"
            | "trust"
    )
}

/// The mode a child gets: the requested one if the parent allows it, the
/// parent's otherwise. Monotone: the result never ranks above `parent`.
pub fn restrict_permission_mode(parent: &str, requested: &str) -> String {
    let canonical = |m: &str| {
        if is_known_mode(m) {
            m.to_string()
        } else {
            "default".to_string()
        }
    };
    if mode_rank(requested) <= mode_rank(parent) {
        canonical(requested)
    } else {
        canonical(parent)
    }
}

// ============================================================================
// Paths
// ============================================================================

/// Resolve a path for comparison: `~` expanded, symlinks resolved when the path
/// exists, `.` / `..` folded lexically otherwise. Never fails.
fn resolve_path(raw: &str) -> PathBuf {
    let expanded = crate::expand_tilde(raw);
    let path = Path::new(&expanded);
    if let Ok(real) = std::fs::canonicalize(path) {
        return real;
    }
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Whether `candidate` is `root` or lies under it (whole path components).
pub fn is_within(root: &str, candidate: &str) -> bool {
    let root = resolve_path(root);
    let candidate = resolve_path(candidate);
    !root.as_os_str().is_empty() && root.is_absolute() && candidate.starts_with(&root)
}

// ============================================================================
// The envelope
// ============================================================================

/// What the calling session allows its children to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentEnvelope {
    pub parent_session_id: Uuid,
    /// The parent's effective permission mode — the child's ceiling.
    pub ceiling: String,
    /// Directories the parent may touch: its cwd, then its extra directories.
    pub roots: Vec<String>,
    pub project_slug: Option<String>,
    pub workspace_slug: Option<String>,
    /// ENV-WS: the projects of the parent's workspace, when the parent's project was
    /// DEDUCED from its cwd (opened in all-projects mode). A child may then name any of
    /// them. `None` (a chosen project, or no workspace): the child keeps the parent's project.
    /// Directories and the permission ceiling are checked as always.
    pub workspace_projects: Option<Vec<String>>,
}

/// The git directory of the repository that `dir` is a linked WORKTREE of, as git records it.
///
/// `dir` or one of its ancestors holds a `.git` FILE (`gitdir: <repo>/.git/worktrees/<name>`).
/// That admin directory must point back at this very `.git` file (its `gitdir` file, written by
/// `git worktree add`), so a `.git` file planted anywhere cannot borrow a repository. Its
/// `commondir` then names the repository's `.git`. `None`: not a linked worktree (no `.git`, a
/// `.git` DIRECTORY, a broken or forged link). Pure file reads: no `git` process.
fn worktree_repository(dir: &str) -> Option<(PathBuf, PathBuf)> {
    let start = resolve_path(dir);
    for candidate in start.ancestors() {
        let dot_git = candidate.join(".git");
        let Ok(meta) = std::fs::symlink_metadata(&dot_git) else {
            continue;
        };
        if !meta.is_file() {
            // An ordinary repository (or a link): not a worktree.
            return None;
        }
        let text = std::fs::read_to_string(&dot_git).ok()?;
        let admin = text
            .lines()
            .find_map(|line| line.strip_prefix("gitdir:"))?
            .trim();
        let admin = std::fs::canonicalize(candidate.join(admin)).ok()?;
        let back = std::fs::read_to_string(admin.join("gitdir")).ok()?;
        let back = std::fs::canonicalize(candidate.join(back.trim())).ok()?;
        if back != std::fs::canonicalize(&dot_git).ok()? {
            return None;
        }
        let common = std::fs::read_to_string(admin.join("commondir")).ok()?;
        let repository = std::fs::canonicalize(admin.join(common.trim())).ok()?;
        return Some((candidate.to_path_buf(), repository));
    }
    None
}

/// Whether `worktree` (a resolved worktree root) is too broad to be a child's working directory:
/// the filesystem root, the home directory or one of its ancestors, or an ancestor of a place no
/// file tool may reach. A parent in trust with a shell could otherwise fake a worktree there
/// (`~/.git` plus an admin directory in its own repository) to give its child `cwd = ~`.
fn too_broad_for_a_worktree(worktree: &Path, home: Option<&Path>, denied: &[PathBuf]) -> bool {
    let real = |path: &Path| std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    worktree.parent().is_none()
        || home.is_some_and(|home| real(home).starts_with(worktree) || home.starts_with(worktree))
        || denied
            .iter()
            .any(|place| real(place).starts_with(worktree) || place.starts_with(worktree))
}

impl ParentEnvelope {
    /// Whether `cwd` is a linked git worktree of a repository whose `.git` lies under one of the
    /// parent's directories ([`worktree_repository`]).
    fn is_worktree_of_a_root_repo(&self, cwd: &str) -> bool {
        self.worktree_allowed(
            cwd,
            dirs::home_dir().as_deref(),
            &crate::chat::provider::nexus_tools::host_deny_paths(None),
        )
    }

    /// [`Self::is_worktree_of_a_root_repo`] for a given home and deny list: a linked worktree of a
    /// repository under the parent's directories, whose root is not too broad
    /// ([`too_broad_for_a_worktree`]).
    fn worktree_allowed(&self, cwd: &str, home: Option<&Path>, denied: &[PathBuf]) -> bool {
        worktree_repository(cwd).is_some_and(|(worktree, repo)| {
            let repo = repo.to_string_lossy();
            !too_broad_for_a_worktree(&worktree, home, denied)
                && self.roots.iter().any(|root| is_within(root, &repo))
        })
    }

    /// Fit a new-session request inside the envelope: clamp what can be
    /// clamped (permission mode), fill what is inherited (project, workspace),
    /// refuse what would escape (directories, project, workspace).
    ///
    /// `default_mode` is the mode the child would get if the request names
    /// none (the global config) — it is clamped too.
    pub fn apply(
        &self,
        request: &mut ChatRequest,
        default_mode: &str,
    ) -> Result<(), EnvelopeError> {
        // Directories: the child works inside what the parent can already see. One exception,
        // for a parent in trust (`trust` or `bypassPermissions`, by rank): its child may work in
        // a git WORKTREE of a repository under the parent's directories (one another harness
        // created next to the repository). Never anywhere else, and extra directories never
        // leave the parent's: the child keeps its directories after the parent leaves trust.
        let parent_in_trust = mode_rank(&self.ceiling) >= TRUST_RANK;
        let cwd_allowed = self.roots.iter().any(|r| is_within(r, &request.cwd))
            || (parent_in_trust && self.is_worktree_of_a_root_repo(&request.cwd));
        if !cwd_allowed {
            return Err(EnvelopeError::CwdOutsideParent {
                cwd: request.cwd.clone(),
            });
        }
        for dir in request.add_dirs.iter().flatten() {
            if !self.roots.iter().any(|r| is_within(r, dir)) {
                return Err(EnvelopeError::AddDirOutsideParent { dir: dir.clone() });
            }
        }

        // Workspace: the parent's, exactly like the project just below. It is resolved
        // server-side into knowledge context and extra directories, so a request naming
        // ANOTHER workspace (or one when the parent has none) is refused — but a request
        // that names none INHERITS the parent's: a child opened without it (every
        // sub-agent of the `Agent` tool, every child of `chat(send_message)`) used to
        // lose the workspace's context and its extra directories (P18b, fix 1).
        match (&request.workspace_slug, &self.workspace_slug) {
            (Some(requested), Some(parent)) if requested != parent => {
                return Err(EnvelopeError::WorkspaceMismatch);
            }
            (Some(_), None) => return Err(EnvelopeError::WorkspaceMismatch),
            (None, Some(parent)) => request.workspace_slug = Some(parent.clone()),
            _ => {}
        }

        // Project: the parent's. A request may omit it; it may not differ — unless the
        // parent's project was deduced in a workspace (ENV-WS): then any project of that
        // workspace, and only those.
        let in_parent_workspace = |requested: &String| {
            self.workspace_projects
                .as_ref()
                .is_some_and(|projects| projects.iter().any(|p| p == requested))
        };
        match (&request.project_slug, &self.project_slug) {
            (Some(requested), Some(parent))
                if requested != parent && !in_parent_workspace(requested) =>
            {
                return Err(EnvelopeError::ProjectMismatch {
                    requested: requested.clone(),
                    parent: parent.clone(),
                });
            }
            (None, Some(parent)) => request.project_slug = Some(parent.clone()),
            _ => {}
        }

        // Policy: never more permissive than the parent.
        let requested = request.permission_mode.as_deref().unwrap_or(default_mode);
        request.permission_mode = Some(restrict_permission_mode(&self.ceiling, requested));
        Ok(())
    }
}

/// Who is asking to open a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnCaller {
    /// A person (or an internal caller with no session behind it): no envelope.
    Human,
    /// A chat session. `token_ceiling` is the ceiling signed into its token.
    Agent {
        session_id: String,
        token_ceiling: Option<String>,
    },
}

/// Identify the caller from what the server can trust.
///
/// * an `agent_session` token → the session signed into it (an unbound agent
///   token is refused: it cannot be placed in the tree);
/// * authentication disabled → the `X-Session-Id` header if present (nothing is
///   signed in that mode; the header is a hint, not a boundary);
/// * anything else → a human.
pub fn identify_caller(
    claims: Option<&Claims>,
    session_header: Option<&str>,
    auth_enabled: bool,
) -> Result<SpawnCaller, EnvelopeError> {
    if let Some(c) = claims.filter(|c| c.is_agent_session()) {
        let binding =
            crate::auth::jwt::agent_session_binding(c).ok_or(EnvelopeError::UnboundAgentToken)?;
        return Ok(SpawnCaller::Agent {
            session_id: binding.session_id,
            token_ceiling: binding.ceiling,
        });
    }
    if !auth_enabled {
        if let Some(sid) = session_header.map(str::trim).filter(|s| !s.is_empty()) {
            return Ok(SpawnCaller::Agent {
                session_id: sid.to_string(),
                token_ceiling: None,
            });
        }
    }
    Ok(SpawnCaller::Human)
}

/// The live facts the envelope needs from the chat manager. A trait so the
/// rule is testable without a running CLI.
#[async_trait::async_trait]
pub trait LiveSessions: Send + Sync {
    /// Whether the session currently has a running agent.
    async fn is_active(&self, session_id: &str) -> bool;
    /// Whether the session is DOING something: a turn running or a background task. A
    /// delegated child that answered stays active (in the live table) until the idle
    /// cleanup closes it, half an hour later, but it is no longer working: it must not
    /// hold one of its parent's slots (issue #876). Defaults to [`Self::is_active`].
    async fn is_working(&self, session_id: &str) -> bool {
        self.is_active(session_id).await
    }
    /// The session's current permission mode if it is live and has one.
    async fn live_permission_mode(&self, session_id: &str) -> Option<String>;
    /// The mode a session gets when none is specified (global config).
    async fn default_permission_mode(&self) -> String;
}

#[async_trait::async_trait]
impl LiveSessions for ChatManager {
    async fn is_active(&self, session_id: &str) -> bool {
        self.is_session_active(session_id).await
    }
    async fn is_working(&self, session_id: &str) -> bool {
        self.is_session_working(session_id).await
    }
    async fn live_permission_mode(&self, session_id: &str) -> Option<String> {
        self.live_session_permission_mode(session_id).await
    }
    async fn default_permission_mode(&self) -> String {
        ChatManager::default_permission_mode(self).await
    }
}

/// Build the envelope of the calling session, checking depth and fan-out.
///
/// `max_live_children` caps how many children one parent may have alive at once
/// (default [`MAX_LIVE_CHILDREN`]; set at runtime, see `delegation_limits`).
pub async fn resolve_envelope(
    graph: &dyn GraphStore,
    live: &dyn LiveSessions,
    parent_session_id: &str,
    token_ceiling: Option<&str>,
    max_live_children: usize,
) -> Result<ParentEnvelope, EnvelopeError> {
    let not_found = || EnvelopeError::ParentNotFound(parent_session_id.to_string());
    let parent_uuid = Uuid::parse_str(parent_session_id).map_err(|_| not_found())?;
    let parent = graph
        .get_chat_session(parent_uuid)
        .await
        .ok()
        .flatten()
        .ok_or_else(not_found)?;

    // Depth: a session that was itself spawned by a session cannot spawn.
    let parent_has_parent = parent
        .spawned_by
        .as_deref()
        .and_then(crate::chat::manager::parse_spawned_by)
        .is_some_and(|ctx| ctx.parent_session_id.is_some());
    if parent_has_parent {
        return Err(EnvelopeError::DepthExceeded {
            max: MAX_DELEGATION_DEPTH,
        });
    }

    // Fan-out: count the children that are still working (a turn running, a background
    // task). A child that has answered and sits idle until the cleanup closes it does not
    // count. A store error counts as "full": refusing a spawn is recoverable, a runaway
    // is not.
    let live_children = match graph.get_session_children(parent_uuid).await {
        Ok(children) => {
            let mut n = 0;
            let now = chrono::Utc::now();
            for child in &children {
                let id = child.id.to_string();
                // Just opened: its first turn may not have started yet (the flag is set by
                // the stream task), so it counts while it is young, or two delegations in a
                // row would both see room.
                let young = now.signed_duration_since(child.created_at) < CHILD_STARTUP_GRACE;
                if live.is_working(&id).await || (young && live.is_active(&id).await) {
                    n += 1;
                }
            }
            n
        }
        Err(_) => max_live_children,
    };
    if live_children >= max_live_children {
        return Err(EnvelopeError::TooManyChildren {
            max: max_live_children,
            live: live_children,
        });
    }

    // Ceiling: the parent's mode as the server knows it now (live, else
    // persisted, else the global default), further clamped by what was signed
    // into its token when it was spawned. Raising a session's mode later does
    // not raise what its already-issued token allows.
    let current = match live.live_permission_mode(parent_session_id).await {
        Some(m) => m,
        None => match parent.permission_mode.clone() {
            Some(m) => m,
            None => live.default_permission_mode().await,
        },
    };
    let ceiling = match token_ceiling {
        Some(signed) => restrict_permission_mode(signed, &current),
        None => restrict_permission_mode(&current, &current),
    };

    let mut roots = vec![parent.cwd.clone()];
    roots.extend(parent.add_dirs.clone().unwrap_or_default());

    // ENV-WS: a deduced project is context, not a boundary. Only for a parent that was
    // opened in a workspace; read from the graph, never from the request.
    let workspace_projects = match (parent.project_inferred, parent.workspace_slug.as_deref()) {
        (true, Some(ws)) => Some(workspace_project_slugs(graph, ws).await),
        _ => None,
    };

    Ok(ParentEnvelope {
        parent_session_id: parent_uuid,
        ceiling,
        roots,
        project_slug: parent.project_slug.clone(),
        workspace_slug: parent.workspace_slug.clone(),
        workspace_projects,
    })
}

/// The slugs of the projects of a workspace. A read error or an unknown workspace gives an
/// empty list: the child then keeps the parent's project (fail closed).
async fn workspace_project_slugs(graph: &dyn GraphStore, workspace_slug: &str) -> Vec<String> {
    let Ok(Some(ws)) = graph.get_workspace_by_slug(workspace_slug).await else {
        return Vec::new();
    };
    graph
        .list_workspace_projects(ws.id)
        .await
        .map(|projects| projects.into_iter().map(|p| p.slug).collect())
        .unwrap_or_default()
}

/// Identify the caller of a spawn request and, for an agent, resolve its
/// envelope. `None` = a human caller, no envelope.
///
/// `max_live_children`: see [`resolve_envelope`].
pub async fn envelope_for_caller(
    graph: &dyn GraphStore,
    live: &dyn LiveSessions,
    caller: &SpawnCaller,
    max_live_children: usize,
) -> Result<Option<ParentEnvelope>, EnvelopeError> {
    match caller {
        SpawnCaller::Human => Ok(None),
        SpawnCaller::Agent {
            session_id,
            token_ceiling,
        } => resolve_envelope(
            graph,
            live,
            session_id,
            token_ceiling.as_deref(),
            max_live_children,
        )
        .await
        .map(Some),
    }
}

/// The `X-Session-Id` header value, if any.
pub fn session_header(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers.get(SESSION_ID_HEADER).and_then(|v| v.to_str().ok())
}

/// An agent may only send a message to (or resume) a session it spawned.
pub async fn ensure_child_of(
    graph: &dyn GraphStore,
    parent_session_id: &str,
    target_session_id: &str,
) -> Result<(), EnvelopeError> {
    let refused = || EnvelopeError::NotAChild {
        session_id: target_session_id.to_string(),
    };
    let parent = Uuid::parse_str(parent_session_id).map_err(|_| refused())?;
    let children = graph
        .get_session_children(parent)
        .await
        .map_err(|_| refused())?;
    if children
        .iter()
        .any(|c| c.id.to_string() == target_session_id)
    {
        Ok(())
    } else {
        Err(refused())
    }
}

/// `spawned_by` payload of a sub-conversation opened by a session.
pub fn conversation_spawned_by(parent_session_id: Uuid) -> String {
    SpawnedBy::Conversation {
        parent_session_id,
        tool_use_id: None,
    }
    .to_json_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::test_chat_session;

    struct FakeLive {
        active: Vec<String>,
        /// Active sessions that are idle (answered, nothing running).
        idle: Vec<String>,
        live_mode: Option<String>,
        default_mode: String,
    }

    #[async_trait::async_trait]
    impl LiveSessions for FakeLive {
        async fn is_active(&self, session_id: &str) -> bool {
            self.active.iter().any(|s| s == session_id)
        }
        async fn is_working(&self, session_id: &str) -> bool {
            self.is_active(session_id).await && !self.idle.iter().any(|s| s == session_id)
        }
        async fn live_permission_mode(&self, _session_id: &str) -> Option<String> {
            self.live_mode.clone()
        }
        async fn default_permission_mode(&self) -> String {
            self.default_mode.clone()
        }
    }

    fn live(default_mode: &str) -> FakeLive {
        FakeLive {
            active: vec![],
            idle: vec![],
            live_mode: None,
            default_mode: default_mode.to_string(),
        }
    }

    fn request(cwd: &str, mode: Option<&str>) -> ChatRequest {
        ChatRequest {
            file_scope_lifted: false,
            access: None,
            routing_pool: None,
            routing_mode: None,
            attachments: Vec::new(),
            refs: Vec::new(),
            message: "go".into(),
            session_id: None,
            cwd: cwd.into(),
            project_slug: None,
            model: None,
            provider: None,
            task_alias: None,
            persona_alias: None,
            run_provider: None,
            run_model: None,
            max_tokens: None,
            task_class: None,
            permission_mode: mode.map(str::to_string),
            add_dirs: None,
            workspace_slug: None,
            user_claims: None,
            spawned_by: None,
            task_context: None,
            subagent_type: None,
            route_auto: false,
            scaffolding_override: None,
            runner_context: None,
            routing_decision_id: None,
        }
    }

    fn envelope(ceiling: &str) -> ParentEnvelope {
        ParentEnvelope {
            parent_session_id: Uuid::new_v4(),
            ceiling: ceiling.into(),
            roots: vec!["/work/repo".into(), "/work/shared".into()],
            project_slug: Some("demo".into()),
            workspace_slug: None,
            workspace_projects: None,
        }
    }

    // ── permission mode ────────────────────────────────────────────────────

    #[test]
    fn a_child_is_never_more_permissive_than_its_parent() {
        let modes = [
            "plan",
            "default",
            "acceptEdits",
            "auto",
            "bypassPermissions",
            "plan_only",
            "ask",
            "auto_edits",
            "trust",
            "manual",
            "dontAsk",
            "something-new",
        ];
        for parent in modes {
            for requested in modes {
                let got = restrict_permission_mode(parent, requested);
                assert!(
                    mode_rank(&got) <= mode_rank(parent),
                    "parent {parent} + requested {requested} gave {got}"
                );
                assert!(
                    mode_rank(&got) <= mode_rank(requested),
                    "restricting must not raise the request ({requested} -> {got})"
                );
            }
        }
    }

    #[test]
    fn parent_in_ask_mode_cannot_spawn_a_bypass_child() {
        let mut req = request("/work/repo", Some("bypassPermissions"));
        envelope("default").apply(&mut req, "default").unwrap();
        assert_eq!(req.permission_mode.as_deref(), Some("default"));
    }

    #[test]
    fn a_child_may_ask_for_less_than_its_parent() {
        let mut req = request("/work/repo", Some("plan"));
        envelope("bypassPermissions")
            .apply(&mut req, "default")
            .unwrap();
        assert_eq!(req.permission_mode.as_deref(), Some("plan"));
    }

    #[test]
    fn an_unspecified_mode_is_the_global_default_clamped_to_the_parent() {
        // Global config says bypass; the parent was lowered to plan.
        let mut req = request("/work/repo", None);
        envelope("plan")
            .apply(&mut req, "bypassPermissions")
            .unwrap();
        assert_eq!(req.permission_mode.as_deref(), Some("plan"));
    }

    #[test]
    fn an_unknown_mode_never_wins() {
        assert_eq!(restrict_permission_mode("default", "yolo"), "default");
        assert_eq!(
            restrict_permission_mode("yolo", "bypassPermissions"),
            "default"
        );
    }

    // ── directories, project, workspace ────────────────────────────────────

    #[test]
    fn cwd_outside_the_parent_is_refused() {
        for cwd in [
            "/",
            "/work",
            "/work/repo-other",
            "/work/repo/../secrets",
            "/etc",
        ] {
            let mut req = request(cwd, None);
            let err = envelope("default").apply(&mut req, "default").unwrap_err();
            assert_eq!(err.code(), "envelope_cwd_outside_parent", "cwd {cwd}");
        }
    }

    #[test]
    fn cwd_inside_the_parent_or_its_add_dirs_is_accepted() {
        for cwd in ["/work/repo", "/work/repo/crates/x", "/work/shared/lib"] {
            let mut req = request(cwd, None);
            envelope("default").apply(&mut req, "default").unwrap();
        }
    }

    #[test]
    fn add_dirs_outside_the_parent_are_refused() {
        let mut req = request("/work/repo", None);
        req.add_dirs = Some(vec!["/work/repo/a".into(), "/home/me/.ssh".into()]);
        let err = envelope("default").apply(&mut req, "default").unwrap_err();
        assert_eq!(
            err,
            EnvelopeError::AddDirOutsideParent {
                dir: "/home/me/.ssh".into()
            }
        );
    }

    /// A repository under `parent_root` and a linked worktree of it OUTSIDE, laid out as
    /// `git worktree add` writes them (no `git` process needed): `<wt>/.git` names the admin
    /// directory, whose `gitdir` points back and whose `commondir` names the repository.
    fn repo_with_outside_worktree() -> (tempfile::TempDir, tempfile::TempDir, String, String) {
        let parent = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let repo = parent.path().join("repo");
        let admin = repo.join(".git").join("worktrees").join("feat");
        std::fs::create_dir_all(&admin).unwrap();
        let wt = outside.path().join("feat");
        std::fs::create_dir_all(wt.join("src")).unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", admin.display())).unwrap();
        std::fs::write(
            admin.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        std::fs::write(admin.join("commondir"), "../..\n").unwrap();
        let root = parent.path().display().to_string();
        let wt = wt.display().to_string();
        (parent, outside, root, wt)
    }

    fn envelope_over(ceiling: &str, root: &str) -> ParentEnvelope {
        ParentEnvelope {
            roots: vec![root.to_owned()],
            ..envelope(ceiling)
        }
    }

    #[test]
    fn a_parent_in_trust_may_open_its_child_in_a_worktree_of_its_repository_whatever_the_spelling()
    {
        let (_p, _o, root, wt) = repo_with_outside_worktree();
        // `trust` and `bypassPermissions` are the same mode (by rank, not by the literal).
        for ceiling in ["trust", "bypassPermissions"] {
            let mut req = request(&wt, None);
            envelope_over(ceiling, &root)
                .apply(&mut req, "default")
                .unwrap_or_else(|e| panic!("{ceiling}: {e}"));
            // A directory inside the worktree is the worktree too.
            let mut req = request(&format!("{wt}/src"), None);
            envelope_over(ceiling, &root)
                .apply(&mut req, "default")
                .unwrap();
        }
    }

    #[test]
    fn a_parent_in_trust_cannot_open_its_child_anywhere_else() {
        let (_p, outside, root, _wt) = repo_with_outside_worktree();
        let plain = outside.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        for ceiling in ["trust", "bypassPermissions"] {
            for cwd in ["/", "/tmp", plain.to_str().unwrap()] {
                let mut req = request(cwd, None);
                assert!(
                    matches!(
                        envelope_over(ceiling, &root).apply(&mut req, "default"),
                        Err(EnvelopeError::CwdOutsideParent { .. })
                    ),
                    "{ceiling} {cwd}"
                );
            }
        }
    }

    #[test]
    fn extra_directories_never_leave_the_parents_even_in_trust() {
        let (_p, _o, root, wt) = repo_with_outside_worktree();
        for ceiling in ["trust", "bypassPermissions"] {
            for dir in ["/home/me/.ssh", "/", wt.as_str()] {
                let mut req = request(&root, None);
                req.add_dirs = Some(vec![dir.into()]);
                assert!(
                    matches!(
                        envelope_over(ceiling, &root).apply(&mut req, "default"),
                        Err(EnvelopeError::AddDirOutsideParent { .. })
                    ),
                    "{ceiling} {dir}"
                );
            }
        }
    }

    /// Follow-up review of #729 (point 4): a parent in trust with a shell can fake a worktree
    /// (`~/.git` plus an admin directory in its own repository). Its root may not be the
    /// filesystem root, the home directory or one of its ancestors, nor an ancestor of a place
    /// no file tool may reach.
    #[test]
    fn a_worktree_root_may_not_be_the_root_the_home_or_above_a_denied_place() {
        let base = tempfile::tempdir().unwrap();
        let base = std::fs::canonicalize(base.path()).unwrap();
        let home = base.join("home");
        let elsewhere = base.join("elsewhere");
        let secrets = elsewhere.join("vault");
        for dir in [&home, &secrets] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let denied = vec![secrets.join("vault.json")];
        for broad in [
            Path::new("/"),
            home.as_path(),
            base.as_path(),
            elsewhere.as_path(),
            secrets.as_path(),
        ] {
            assert!(
                too_broad_for_a_worktree(broad, Some(&home), &denied),
                "{broad:?}"
            );
        }
        for fine in [home.join("src/wt"), elsewhere.join("wt")] {
            assert!(
                !too_broad_for_a_worktree(&fine, Some(&home), &denied),
                "{fine:?}"
            );
        }
    }

    /// The same rule through the envelope: a worktree faked AT the home (a `.git` file pointing at
    /// a real admin directory of the parent's repository, back-link included) is refused; the
    /// same worktree elsewhere is accepted.
    #[test]
    fn a_worktree_faked_at_the_home_is_refused() {
        let (_p, outside, root, wt) = repo_with_outside_worktree();
        let envelope = envelope_over("trust", &root);
        let wt_path = std::fs::canonicalize(&wt).unwrap();
        let unrelated_home = outside.path().join("not-home");
        assert!(envelope.worktree_allowed(&wt, Some(&unrelated_home), &[]));
        // The worktree root is the home itself, or the home lies below it.
        assert!(!envelope.worktree_allowed(&wt, Some(&wt_path), &[]));
        assert!(!envelope.worktree_allowed(&wt, Some(&wt_path.join("me")), &[]));
        // A denied place below the worktree root.
        assert!(!envelope.worktree_allowed(&wt, Some(&unrelated_home), &[wt_path.join(".aws")]));
    }

    #[test]
    fn a_planted_git_file_does_not_borrow_the_parents_repository() {
        let (_p, outside, root, _wt) = repo_with_outside_worktree();
        // Points at the real admin directory, but the admin directory points at another `.git`.
        let fake = outside.path().join("fake");
        std::fs::create_dir_all(&fake).unwrap();
        let admin = std::path::Path::new(&root).join("repo/.git/worktrees/feat");
        std::fs::write(fake.join(".git"), format!("gitdir: {}\n", admin.display())).unwrap();
        let mut req = request(fake.to_str().unwrap(), None);
        assert!(matches!(
            envelope_over("trust", &root).apply(&mut req, "default"),
            Err(EnvelopeError::CwdOutsideParent { .. })
        ));
    }

    #[test]
    fn a_parent_out_of_trust_cannot_open_its_child_in_an_outside_worktree() {
        let (_p, _o, root, wt) = repo_with_outside_worktree();
        for ceiling in ["default", "acceptEdits", "auto", "plan"] {
            let mut req = request(&wt, None);
            assert!(
                matches!(
                    envelope_over(ceiling, &root).apply(&mut req, "default"),
                    Err(EnvelopeError::CwdOutsideParent { .. })
                ),
                "{ceiling}"
            );
        }
    }

    #[test]
    fn the_child_inherits_the_parents_project_and_cannot_name_another() {
        let mut req = request("/work/repo", None);
        envelope("default").apply(&mut req, "default").unwrap();
        assert_eq!(req.project_slug.as_deref(), Some("demo"));

        let mut req = request("/work/repo", None);
        req.project_slug = Some("permissive-project".into());
        let err = envelope("default").apply(&mut req, "default").unwrap_err();
        assert_eq!(err.code(), "envelope_project_mismatch");
    }

    #[test]
    fn a_foreign_workspace_is_refused() {
        let mut req = request("/work/repo", None);
        req.workspace_slug = Some("other-ws".into());
        let err = envelope("default").apply(&mut req, "default").unwrap_err();
        assert_eq!(err, EnvelopeError::WorkspaceMismatch);
    }

    /// P18b, fix 1: the workspace is inherited like the project. Before it, a child that
    /// named no workspace got NONE — a sub-agent of a native session lost the workspace's
    /// knowledge context and its extra directories.
    #[test]
    fn the_child_inherits_the_parents_workspace_and_cannot_name_another() {
        let in_workspace = || ParentEnvelope {
            workspace_slug: Some("ws".into()),
            ..envelope("default")
        };

        let mut req = request("/work/repo", None);
        in_workspace().apply(&mut req, "default").unwrap();
        assert_eq!(req.workspace_slug.as_deref(), Some("ws"), "inherited");

        let mut req = request("/work/repo", None);
        req.workspace_slug = Some("ws".into());
        in_workspace().apply(&mut req, "default").unwrap();
        assert_eq!(
            req.workspace_slug.as_deref(),
            Some("ws"),
            "the same one: kept"
        );

        let mut req = request("/work/repo", None);
        req.workspace_slug = Some("other-ws".into());
        assert_eq!(
            in_workspace().apply(&mut req, "default").unwrap_err(),
            EnvelopeError::WorkspaceMismatch,
            "another workspace is still refused"
        );

        // A parent outside any workspace gives its child none.
        let mut req = request("/work/repo", None);
        envelope("default").apply(&mut req, "default").unwrap();
        assert_eq!(req.workspace_slug, None);
    }

    // ── who is calling ─────────────────────────────────────────────────────

    fn agent_claims(session: Option<&str>, ceiling: Option<&str>) -> Claims {
        let human = Claims::service_account("someone");
        let binding = session.map(|s| crate::auth::jwt::AgentSessionBinding {
            session_id: s.to_string(),
            ceiling: ceiling.map(str::to_string),
            tool_profile: None,
            third_party: false,
        });
        let secret = "test-secret-key-minimum-32-chars!!";
        let (token, _) =
            crate::auth::jwt::generate_session_token(&human, binding.as_ref(), secret, 60).unwrap();
        crate::auth::jwt::decode_jwt(&token, secret).unwrap()
    }

    #[test]
    fn the_parent_comes_from_the_signed_token_not_from_a_header() {
        let claims = agent_claims(Some("sess-signed"), Some("default"));
        let caller = identify_caller(Some(&claims), Some("sess-forged"), true).unwrap();
        assert_eq!(
            caller,
            SpawnCaller::Agent {
                session_id: "sess-signed".into(),
                token_ceiling: Some("default".into()),
            }
        );
    }

    #[test]
    fn an_unbound_agent_token_cannot_spawn() {
        let claims = agent_claims(None, None);
        assert_eq!(
            identify_caller(Some(&claims), None, true),
            Err(EnvelopeError::UnboundAgentToken)
        );
    }

    #[test]
    fn a_human_is_not_enveloped_and_the_header_is_ignored_when_auth_is_on() {
        let human = Claims::service_account("alice");
        assert_eq!(
            identify_caller(Some(&human), Some("sess-x"), true),
            Ok(SpawnCaller::Human)
        );
    }

    #[test]
    fn without_auth_the_header_names_the_parent() {
        let anon = Claims::anonymous();
        assert_eq!(
            identify_caller(Some(&anon), Some("sess-x"), false),
            Ok(SpawnCaller::Agent {
                session_id: "sess-x".into(),
                token_ceiling: None,
            })
        );
        assert_eq!(
            identify_caller(Some(&anon), None, false),
            Ok(SpawnCaller::Human)
        );
    }

    // ── resolve: depth, fan-out, ceiling ───────────────────────────────────

    async fn store_with_parent(mode: Option<&str>) -> (MockGraphStore, Uuid) {
        let graph = MockGraphStore::new();
        let mut parent = test_chat_session(Some("demo"));
        parent.cwd = "/work/repo".into();
        parent.permission_mode = mode.map(str::to_string);
        let id = parent.id;
        graph.create_chat_session(&parent).await.unwrap();
        (graph, id)
    }

    async fn add_child(graph: &MockGraphStore, parent: Uuid) -> Uuid {
        let mut child = test_chat_session(Some("demo"));
        child.spawned_by = Some(conversation_spawned_by(parent));
        let id = child.id;
        graph.create_chat_session(&child).await.unwrap();
        graph
            .create_spawned_by_relation(
                &id.to_string(),
                &parent.to_string(),
                "conversation",
                None,
                None,
            )
            .await
            .unwrap();
        id
    }

    #[tokio::test]
    async fn the_ceiling_is_the_parents_persisted_mode() {
        let (graph, parent) = store_with_parent(Some("acceptEdits")).await;
        let env = resolve_envelope(
            &graph,
            &live("bypassPermissions"),
            &parent.to_string(),
            None,
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap();
        assert_eq!(env.ceiling, "acceptEdits");
        assert_eq!(env.roots, vec!["/work/repo".to_string()]);
        assert_eq!(env.project_slug.as_deref(), Some("demo"));
    }

    #[tokio::test]
    async fn the_ceiling_falls_back_to_the_global_default() {
        let (graph, parent) = store_with_parent(None).await;
        let env = resolve_envelope(
            &graph,
            &live("default"),
            &parent.to_string(),
            None,
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap();
        assert_eq!(env.ceiling, "default");
    }

    #[tokio::test]
    async fn the_live_mode_and_the_signed_ceiling_both_clamp() {
        let (graph, parent) = store_with_parent(Some("bypassPermissions")).await;
        // A human lowered the live session to plan.
        let mut l = live("default");
        l.live_mode = Some("plan".into());
        let env = resolve_envelope(
            &graph,
            &l,
            &parent.to_string(),
            Some("bypassPermissions"),
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap();
        assert_eq!(env.ceiling, "plan");

        // The token was minted in Ask mode; the session was raised since.
        let mut l = live("default");
        l.live_mode = Some("bypassPermissions".into());
        let env = resolve_envelope(
            &graph,
            &l,
            &parent.to_string(),
            Some("default"),
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap();
        assert_eq!(env.ceiling, "default");
    }

    #[tokio::test]
    async fn a_spawned_session_cannot_spawn() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let child = add_child(&graph, parent).await;
        let err = resolve_envelope(
            &graph,
            &live("default"),
            &child.to_string(),
            None,
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap_err();
        assert_eq!(
            err,
            EnvelopeError::DepthExceeded {
                max: MAX_DELEGATION_DEPTH
            }
        );
    }

    #[tokio::test]
    async fn a_fifth_live_child_is_refused_but_finished_children_do_not_count() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let mut children = Vec::new();
        for _ in 0..MAX_LIVE_CHILDREN {
            children.push(add_child(&graph, parent).await.to_string());
        }

        // All four alive → refused.
        let mut l = live("default");
        l.active = children.clone();
        let err = resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            EnvelopeError::TooManyChildren {
                max: MAX_LIVE_CHILDREN,
                live: MAX_LIVE_CHILDREN
            }
        );

        // One finished → accepted.
        l.active.pop();
        resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN)
            .await
            .unwrap();
    }

    /// A child opened `age_secs` ago.
    async fn add_aged_child(graph: &MockGraphStore, parent: Uuid, age_secs: i64) -> Uuid {
        let mut child = test_chat_session(Some("demo"));
        child.spawned_by = Some(conversation_spawned_by(parent));
        child.created_at = chrono::Utc::now() - chrono::Duration::seconds(age_secs);
        let id = child.id;
        graph.create_chat_session(&child).await.unwrap();
        graph
            .create_spawned_by_relation(
                &id.to_string(),
                &parent.to_string(),
                "conversation",
                None,
                None,
            )
            .await
            .unwrap();
        id
    }

    #[tokio::test]
    async fn a_child_just_opened_counts_before_its_first_turn_is_seen() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let mut children = Vec::new();
        for _ in 0..MAX_LIVE_CHILDREN {
            children.push(add_child(&graph, parent).await.to_string());
        }
        // Not streaming yet (idle), but opened seconds ago: they all count.
        let mut l = live("default");
        l.active = children.clone();
        l.idle = children.clone();
        assert!(matches!(
            resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN).await,
            Err(EnvelopeError::TooManyChildren { .. })
        ));
    }

    #[tokio::test]
    async fn a_child_that_answered_and_sits_idle_does_not_hold_a_slot() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let mut children = Vec::new();
        for _ in 0..MAX_LIVE_CHILDREN {
            children.push(add_aged_child(&graph, parent, 600).await.to_string());
        }
        // All four still in the live table, but three of them have answered: idle.
        let mut l = live("default");
        l.active = children.clone();
        l.idle = children[..3].to_vec();
        resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN)
            .await
            .expect("only one child is working: three slots are free");
        // All four working again: refused.
        l.idle.clear();
        assert!(matches!(
            resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN).await,
            Err(EnvelopeError::TooManyChildren { .. })
        ));
    }

    /// A live-session source that does NOT override `is_working`: the trait's default
    /// (working = active) is the shape every implementation but the chat manager keeps,
    /// and it must keep answering for them.
    struct PlainLive {
        active: Vec<String>,
    }

    #[async_trait::async_trait]
    impl LiveSessions for PlainLive {
        async fn is_active(&self, session_id: &str) -> bool {
            self.active.iter().any(|s| s == session_id)
        }
        async fn live_permission_mode(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn default_permission_mode(&self) -> String {
            "default".to_string()
        }
    }

    #[tokio::test]
    async fn a_source_that_does_not_say_who_works_counts_every_active_session() {
        let l = PlainLive {
            active: vec!["s-open".to_string()],
        };
        assert!(LiveSessions::is_working(&l, "s-open").await);
        assert!(!LiveSessions::is_working(&l, "s-gone").await);

        // Through the rule, with no child: the envelope is built from what this source
        // says (no live mode, so its global default), the fan-out being under the cap.
        let (graph, parent) = store_with_parent(None).await;
        let envelope = resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN)
            .await
            .expect("no live child");
        assert_eq!(envelope.ceiling, "default");

        // And a source that says nothing about work counts every active session as
        // working: four of them fill the parent's fan-out.
        let mut l = PlainLive { active: vec![] };
        for _ in 0..MAX_LIVE_CHILDREN {
            l.active
                .push(add_aged_child(&graph, parent, 600).await.to_string());
        }
        assert!(matches!(
            resolve_envelope(&graph, &l, &parent.to_string(), None, MAX_LIVE_CHILDREN).await,
            Err(EnvelopeError::TooManyChildren { .. })
        ));
    }

    // ── ENV-WS: a project deduced in a workspace is context, not a boundary ────

    /// A parent in workspace `ws` (projects `demo` and `backend`) whose project is `demo`,
    /// `inferred` saying whether it was deduced from its cwd or chosen.
    async fn workspace_parent(inferred: bool) -> (MockGraphStore, Uuid) {
        let graph = MockGraphStore::new();
        let mut ws = crate::test_helpers::test_workspace();
        ws.slug = "ws".into();
        graph.create_workspace(&ws).await.unwrap();
        for slug in ["demo", "backend"] {
            let mut p = crate::test_helpers::test_project();
            p.slug = slug.into();
            graph.create_project(&p).await.unwrap();
            graph.add_project_to_workspace(ws.id, p.id).await.unwrap();
        }
        let mut parent = test_chat_session(Some("demo"));
        parent.cwd = "/work/repo".into();
        parent.workspace_slug = Some("ws".into());
        parent.project_inferred = inferred;
        let id = parent.id;
        graph.create_chat_session(&parent).await.unwrap();
        (graph, id)
    }

    async fn open(graph: &MockGraphStore, parent: Uuid) -> ParentEnvelope {
        resolve_envelope(
            graph,
            &live("default"),
            &parent.to_string(),
            None,
            MAX_LIVE_CHILDREN,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn a_deduced_project_lets_a_child_reach_another_project_of_the_same_workspace() {
        let (graph, parent) = workspace_parent(true).await;
        let env = open(&graph, parent).await;

        let mut req = request("/work/repo", None);
        req.project_slug = Some("backend".into());
        env.apply(&mut req, "default").unwrap();
        assert_eq!(req.project_slug.as_deref(), Some("backend"), "kept");
        assert_eq!(req.workspace_slug.as_deref(), Some("ws"), "inherited");

        // Never a project outside the parent's workspace.
        let mut req = request("/work/repo", None);
        req.project_slug = Some("elsewhere".into());
        assert_eq!(
            env.apply(&mut req, "default").unwrap_err().code(),
            "envelope_project_mismatch"
        );
        // Nor another workspace.
        let mut req = request("/work/repo", None);
        req.workspace_slug = Some("other-ws".into());
        req.project_slug = Some("backend".into());
        assert_eq!(
            env.apply(&mut req, "default").unwrap_err(),
            EnvelopeError::WorkspaceMismatch
        );
    }

    /// The project bound is the only thing that moves: the directories and the permission
    /// ceiling stay the parent's.
    #[tokio::test]
    async fn a_deduced_project_widens_neither_the_directories_nor_the_mode() {
        let (graph, parent) = workspace_parent(true).await;
        let env = open(&graph, parent).await;

        let mut req = request("/somewhere/else", None);
        req.project_slug = Some("backend".into());
        assert!(matches!(
            env.apply(&mut req, "default"),
            Err(EnvelopeError::CwdOutsideParent { .. })
        ));

        let mut req = request("/work/repo", Some("bypassPermissions"));
        req.project_slug = Some("backend".into());
        env.apply(&mut req, "default").unwrap();
        assert_eq!(
            req.permission_mode.as_deref(),
            Some("default"),
            "capped at the parent's mode"
        );
    }

    /// A project the person CHOSE keeps the strict rule; so does every session stored
    /// before the mark existed (it reads back as chosen).
    #[tokio::test]
    async fn a_chosen_project_still_bounds_the_child() {
        let (graph, parent) = workspace_parent(false).await;
        let env = open(&graph, parent).await;
        assert_eq!(env.workspace_projects, None);

        let mut req = request("/work/repo", None);
        req.project_slug = Some("backend".into());
        assert_eq!(
            env.apply(&mut req, "default").unwrap_err().code(),
            "envelope_project_mismatch"
        );

        let stored_before: crate::neo4j::models::ChatSessionNode = serde_json::from_value({
            let mut v = serde_json::to_value(test_chat_session(Some("demo"))).unwrap();
            v.as_object_mut().unwrap().remove("project_inferred");
            v
        })
        .unwrap();
        assert!(!stored_before.project_inferred, "absent = chosen");
    }

    /// Only the server marks a project as deduced, and only from the ABSENCE of a project in
    /// the request: a caller that writes the mark into its request changes nothing.
    #[test]
    fn no_caller_can_mark_its_project_as_deduced() {
        use crate::chat::manager::project_was_inferred;
        assert!(
            project_was_inferred(None, Some("demo")),
            "none asked, one found"
        );
        assert!(!project_was_inferred(Some("demo"), Some("demo")), "chosen");
        assert!(!project_was_inferred(None, None), "no project at all");

        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "message": "hi", "cwd": "/work/repo",
            "project_inferred": true, "workspace_projects": ["backend"]
        }))
        .unwrap();
        let back = serde_json::to_value(&req).unwrap();
        assert!(back.get("project_inferred").is_none());
        assert!(back.get("workspace_projects").is_none());
    }

    /// A deduced project with no workspace gives nothing more: there is no wider bound to
    /// fall back to, so the child keeps the parent's project.
    #[tokio::test]
    async fn a_deduced_project_without_a_workspace_keeps_the_strict_rule() {
        let graph = MockGraphStore::new();
        let mut parent = test_chat_session(Some("demo"));
        parent.cwd = "/work/repo".into();
        parent.project_inferred = true;
        graph.create_chat_session(&parent).await.unwrap();
        let env = open(&graph, parent.id).await;
        assert_eq!(env.workspace_projects, None);
    }

    #[tokio::test]
    async fn an_unknown_parent_is_refused() {
        let graph = MockGraphStore::new();
        for id in [Uuid::new_v4().to_string(), "not-a-uuid".to_string()] {
            let err = resolve_envelope(&graph, &live("default"), &id, None, MAX_LIVE_CHILDREN)
                .await
                .unwrap_err();
            assert_eq!(err.code(), "envelope_parent_not_found");
        }
    }

    #[tokio::test]
    async fn an_agent_only_reaches_sessions_it_spawned() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let child = add_child(&graph, parent).await;
        let (_, stranger) = store_with_parent(Some("default")).await;
        ensure_child_of(&graph, &parent.to_string(), &child.to_string())
            .await
            .unwrap();
        let err = ensure_child_of(&graph, &parent.to_string(), &stranger.to_string())
            .await
            .unwrap_err();
        assert_eq!(err.code(), "envelope_not_a_child");
    }
}
