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
}

impl ParentEnvelope {
    /// Fit a new-session request inside the envelope: clamp what can be
    /// clamped (permission mode), fill what is inherited (project), refuse
    /// what would escape (directories, project, workspace).
    ///
    /// `default_mode` is the mode the child would get if the request names
    /// none (the global config) — it is clamped too.
    pub fn apply(
        &self,
        request: &mut ChatRequest,
        default_mode: &str,
    ) -> Result<(), EnvelopeError> {
        // Directories: the child works inside what the parent can already see.
        if !self.roots.iter().any(|r| is_within(r, &request.cwd)) {
            return Err(EnvelopeError::CwdOutsideParent {
                cwd: request.cwd.clone(),
            });
        }
        for dir in request.add_dirs.iter().flatten() {
            if !self.roots.iter().any(|r| is_within(r, dir)) {
                return Err(EnvelopeError::AddDirOutsideParent { dir: dir.clone() });
            }
        }

        // Workspace: resolved server-side into extra directories, so only the
        // parent's own workspace is acceptable.
        if request.workspace_slug.is_some() && request.workspace_slug != self.workspace_slug {
            return Err(EnvelopeError::WorkspaceMismatch);
        }

        // Project: the parent's. A request may omit it; it may not differ.
        match (&request.project_slug, &self.project_slug) {
            (Some(requested), Some(parent)) if requested != parent => {
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
    async fn live_permission_mode(&self, session_id: &str) -> Option<String> {
        self.live_session_permission_mode(session_id).await
    }
    async fn default_permission_mode(&self) -> String {
        ChatManager::default_permission_mode(self).await
    }
}

/// Build the envelope of the calling session, checking depth and fan-out.
pub async fn resolve_envelope(
    graph: &dyn GraphStore,
    live: &dyn LiveSessions,
    parent_session_id: &str,
    token_ceiling: Option<&str>,
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

    // Fan-out: count the children that still have a running agent. A store
    // error counts as "full": refusing a spawn is recoverable, a runaway is not.
    let live_children = match graph.get_session_children(parent_uuid).await {
        Ok(children) => {
            let mut n = 0;
            for child in &children {
                if live.is_active(&child.id.to_string()).await {
                    n += 1;
                }
            }
            n
        }
        Err(_) => MAX_LIVE_CHILDREN,
    };
    if live_children >= MAX_LIVE_CHILDREN {
        return Err(EnvelopeError::TooManyChildren {
            max: MAX_LIVE_CHILDREN,
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

    Ok(ParentEnvelope {
        parent_session_id: parent_uuid,
        ceiling,
        roots,
        project_slug: parent.project_slug.clone(),
        workspace_slug: parent.workspace_slug.clone(),
    })
}

/// Identify the caller of a spawn request and, for an agent, resolve its
/// envelope. `None` = a human caller, no envelope.
pub async fn envelope_for_caller(
    graph: &dyn GraphStore,
    live: &dyn LiveSessions,
    caller: &SpawnCaller,
) -> Result<Option<ParentEnvelope>, EnvelopeError> {
    match caller {
        SpawnCaller::Human => Ok(None),
        SpawnCaller::Agent {
            session_id,
            token_ceiling,
        } => resolve_envelope(graph, live, session_id, token_ceiling.as_deref())
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
        live_mode: Option<String>,
        default_mode: String,
    }

    #[async_trait::async_trait]
    impl LiveSessions for FakeLive {
        async fn is_active(&self, session_id: &str) -> bool {
            self.active.iter().any(|s| s == session_id)
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
            live_mode: None,
            default_mode: default_mode.to_string(),
        }
    }

    fn request(cwd: &str, mode: Option<&str>) -> ChatRequest {
        ChatRequest {
            attachments: Vec::new(),
            message: "go".into(),
            session_id: None,
            cwd: cwd.into(),
            project_slug: None,
            model: None,
            provider: None,
            permission_mode: mode.map(str::to_string),
            add_dirs: None,
            workspace_slug: None,
            user_claims: None,
            spawned_by: None,
            task_context: None,
            scaffolding_override: None,
            runner_context: None,
        }
    }

    fn envelope(ceiling: &str) -> ParentEnvelope {
        ParentEnvelope {
            parent_session_id: Uuid::new_v4(),
            ceiling: ceiling.into(),
            roots: vec!["/work/repo".into(), "/work/shared".into()],
            project_slug: Some("demo".into()),
            workspace_slug: None,
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

    // ── who is calling ─────────────────────────────────────────────────────

    fn agent_claims(session: Option<&str>, ceiling: Option<&str>) -> Claims {
        let human = Claims::service_account("someone");
        let binding = session.map(|s| crate::auth::jwt::AgentSessionBinding {
            session_id: s.to_string(),
            ceiling: ceiling.map(str::to_string),
            tool_profile: None,
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
        let env = resolve_envelope(&graph, &live("default"), &parent.to_string(), None)
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
        let env = resolve_envelope(&graph, &l, &parent.to_string(), Some("bypassPermissions"))
            .await
            .unwrap();
        assert_eq!(env.ceiling, "plan");

        // The token was minted in Ask mode; the session was raised since.
        let mut l = live("default");
        l.live_mode = Some("bypassPermissions".into());
        let env = resolve_envelope(&graph, &l, &parent.to_string(), Some("default"))
            .await
            .unwrap();
        assert_eq!(env.ceiling, "default");
    }

    #[tokio::test]
    async fn a_spawned_session_cannot_spawn() {
        let (graph, parent) = store_with_parent(Some("default")).await;
        let child = add_child(&graph, parent).await;
        let err = resolve_envelope(&graph, &live("default"), &child.to_string(), None)
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
        let err = resolve_envelope(&graph, &l, &parent.to_string(), None)
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
        resolve_envelope(&graph, &l, &parent.to_string(), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn an_unknown_parent_is_refused() {
        let graph = MockGraphStore::new();
        for id in [Uuid::new_v4().to_string(), "not-a-uuid".to_string()] {
            let err = resolve_envelope(&graph, &live("default"), &id, None)
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
