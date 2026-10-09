//! Who may see what a reference points at.
//!
//! The server is the only source of truth: a client names an entity, the server
//! decides whether it exists and whether *this* principal may see it. Three
//! rules shape the module:
//!
//! * **One door.** [`RefSource::load_unchecked`] is the raw read; nothing but
//!   [`AccessPolicy::resolve_checked`] is meant to call it, and everything that
//!   turns a reference into content must go through `resolve_checked`.
//! * **Fail closed.** An unauthenticated principal, a store error, a result
//!   that is not the entity asked for: all end as "not available", never as
//!   content.
//! * **No existence leak.** With [`Disclosure::Uniform`] (the default) a
//!   denied entity and a missing one produce the same answer.
//!
//! Today the platform has no per-entity read ACL: an authenticated user reads
//! every plan, task, note and decision of the instance (`sharing/` governs the
//! *publication* of knowledge, not reading). [`OpenInstanceRule`] says exactly
//! that, and [`ScopeRule`] is the seam where a project/workspace rule will go
//! without touching a caller.

use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::types::{EntityRef, RefId, RefKind};
use super::wire::{RefResolution, RefStatus, ScopeLabel};

/// Nil UUID: the subject the auth layer injects in no-auth mode.
const ANONYMOUS_SUBJECT: Uuid = Uuid::nil();

/// The project and workspace a chat session is attached to (their ids). Both
/// `None` is a session attached to nothing: it reads what the instance shares,
/// and nothing sensitive (see [`OpenInstanceRule`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SessionScope {
    pub project: Option<Uuid>,
    pub workspace: Option<Uuid>,
}

/// The caller, as far as access is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Principal {
    /// A signed-in user (or an agent acting for one).
    User(Uuid),
    /// No-auth mode: the instance is open by configuration.
    LocalOpen,
    /// The session that is answering a turn. The message was authenticated
    /// when the API took it in; after that (queue, drain, another instance)
    /// no identity travels with it, so the turn acts as the session itself,
    /// inside the project or workspace the session is attached to.
    Session(SessionScope),
    /// No usable identity. Never allowed.
    Unauthenticated,
}

impl Principal {
    /// Build the principal from a JWT `sub`. An unparsable subject is
    /// `Unauthenticated` — fail closed.
    pub fn from_subject(sub: &str) -> Self {
        match Uuid::parse_str(sub) {
            Ok(id) if id == ANONYMOUS_SUBJECT => Principal::LocalOpen,
            Ok(id) => Principal::User(id),
            Err(_) => Principal::Unauthenticated,
        }
    }
}

/// What the store knows about a referenced entity — enough to decide access
/// and to draw a chip, never its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefMeta {
    pub kind: RefKind,
    pub id: RefId,
    pub label: String,
    pub subtitle: Option<String>,
    pub project: Option<ScopeLabel>,
    pub workspace: Option<ScopeLabel>,
    pub entity_status: Option<String>,
}

/// The raw read. Implemented over `GraphStore` by the resolvers (PR 2).
///
/// Every implementation is *unchecked*: calling it directly skips the policy.
/// Only [`AccessPolicy::resolve_checked`] should.
#[async_trait]
pub trait RefSource: Send + Sync {
    async fn load_unchecked(&self, r: &EntityRef) -> anyhow::Result<Option<RefMeta>>;
}

/// The seam where a project/workspace rule plugs in.
pub trait ScopeRule: Send + Sync {
    fn allows(&self, principal: &Principal, meta: &RefMeta) -> bool;
}

/// The rule of today: the instance is open to its signed-in users, a chat
/// session is held to its own project (or workspace), and the sensitive kinds
/// are held tighter.
///
/// * **A signed-in user** (the `#` picker) reads the instance. The search
///   itself narrows by the project or workspace the caller asks for.
/// * **A session** reads what is in its project: an entity of another project
///   is not readable, and neither is an entity of another workspace. A session
///   attached only to a workspace reads that workspace's projects. An entity
///   that belongs to no project and no workspace (a global note, a link) is
///   shared by the instance. A session attached to nothing reads, as before,
///   every non-sensitive entity of the instance.
/// * **Sensitive kinds** ([`RefKind::is_sensitive`]: file,
///   commit) must belong to a project, and a session must be attached to a
///   project or workspace to read them at all: a session that cannot say where
///   it is cannot prove it may.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenInstanceRule;

impl OpenInstanceRule {
    fn within(scope: &SessionScope, meta: &RefMeta) -> bool {
        match (scope.project, scope.workspace) {
            (Some(p), w) => match (&meta.project, &meta.workspace) {
                (Some(mp), _) => mp.id == p,
                (None, Some(mw)) => w == Some(mw.id),
                (None, None) => true,
            },
            (None, Some(w)) => match (&meta.project, &meta.workspace) {
                (_, Some(mw)) => mw.id == w,
                (Some(_), None) => false,
                (None, None) => true,
            },
            (None, None) => true,
        }
    }
}

impl ScopeRule for OpenInstanceRule {
    fn allows(&self, principal: &Principal, meta: &RefMeta) -> bool {
        let scope = match principal {
            Principal::Unauthenticated => return false,
            Principal::Session(s) => Some(s),
            Principal::User(_) | Principal::LocalOpen => None,
        };
        if meta.kind.is_sensitive() {
            if meta.project.is_none() {
                return false;
            }
            if scope.is_some_and(|s| s.project.is_none() && s.workspace.is_none()) {
                return false;
            }
        }
        scope.is_none_or(|s| Self::within(s, meta))
    }
}

/// What a denial tells the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Disclosure {
    /// Denied and missing look the same (`not_found`). The default.
    #[default]
    Uniform,
    /// Denied is reported as `forbidden`. Only for a principal who may already
    /// know the entity exists.
    Distinct,
}

/// The outcome of a checked read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    Found(Box<RefMeta>),
    NotFound,
    Forbidden,
    /// The store failed. Reported to the user as `not_found` (fail closed),
    /// kept apart here so it can be logged as what it is.
    Failed,
}

impl Resolution {
    /// What the user is told. `Failed` and, under `Uniform`, `Forbidden` both
    /// read as `not_found`: the wire never separates them.
    pub fn status(&self, disclosure: Disclosure) -> RefStatus {
        match (self, disclosure) {
            (Resolution::Found(_), _) => RefStatus::Ok,
            (Resolution::Forbidden, Disclosure::Distinct) => RefStatus::Forbidden,
            (Resolution::Forbidden, Disclosure::Uniform)
            | (Resolution::NotFound, _)
            | (Resolution::Failed, _) => RefStatus::NotFound,
        }
    }

    /// The `refs_resolved` entry for `r`. Only a found entity carries a label.
    pub fn to_wire(&self, r: &EntityRef, disclosure: Disclosure) -> RefResolution {
        let status = self.status(disclosure);
        match self {
            Resolution::Found(m) => RefResolution {
                kind: r.kind,
                id: r.id.clone(),
                status,
                label: Some(m.label.clone()),
                subtitle: m.subtitle.clone(),
                project: m.project.clone(),
                workspace: m.workspace.clone(),
                entity_status: m.entity_status.clone(),
            },
            _ => RefResolution {
                kind: r.kind,
                id: r.id.clone(),
                status,
                label: None,
                subtitle: None,
                project: None,
                workspace: None,
                entity_status: None,
            },
        }
    }
}

/// The policy: a rule and a disclosure mode.
#[derive(Clone)]
pub struct AccessPolicy {
    rule: Arc<dyn ScopeRule>,
    disclosure: Disclosure,
}

impl AccessPolicy {
    pub fn new(rule: Arc<dyn ScopeRule>, disclosure: Disclosure) -> Self {
        Self { rule, disclosure }
    }

    /// Today's policy: open instance, uniform answers.
    pub fn open_instance() -> Self {
        Self::new(Arc::new(OpenInstanceRule), Disclosure::Uniform)
    }

    pub fn disclosure(&self) -> Disclosure {
        self.disclosure
    }

    /// Read a referenced entity for `principal`, or say why not.
    ///
    /// The identity check comes first and the store is not even asked for an
    /// unauthenticated caller.
    pub async fn resolve_checked(
        &self,
        principal: &Principal,
        source: &dyn RefSource,
        r: &EntityRef,
    ) -> Resolution {
        if *principal == Principal::Unauthenticated {
            return Resolution::Forbidden;
        }
        let meta = match source.load_unchecked(r).await {
            Ok(Some(meta)) => meta,
            Ok(None) => return Resolution::NotFound,
            Err(e) => {
                tracing::warn!(error = %e, kind = %r.kind, "reference lookup failed");
                return Resolution::Failed;
            }
        };
        // A source that answers for another entity is a bug, not a result.
        if meta.kind != r.kind || meta.id != r.id {
            return Resolution::NotFound;
        }
        if self.rule.allows(principal, &meta) {
            Resolution::Found(Box::new(meta))
        } else {
            Resolution::Forbidden
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const A: &str = "3adeffc9-c8b0-4e2f-a674-55bfcb293433";
    const B: &str = "57cf05c9-25b6-495d-ab07-de4b11d64736";

    fn rf(kind: RefKind, id: &str) -> EntityRef {
        EntityRef::new(kind, Uuid::parse_str(id).unwrap())
    }

    fn meta_for(r: &EntityRef) -> RefMeta {
        RefMeta {
            kind: r.kind,
            id: r.id.clone(),
            label: "Titre".into(),
            subtitle: Some("sous-titre".into()),
            project: Some(ScopeLabel {
                id: Uuid::parse_str(B).unwrap(),
                slug: "proj".into(),
                name: "Proj".into(),
            }),
            workspace: None,
            entity_status: Some("in_progress".into()),
        }
    }

    /// A store that answers what it is told and counts its reads.
    struct Fake {
        answer: Result<Option<RefMeta>, &'static str>,
        reads: AtomicUsize,
    }

    impl Fake {
        fn new(answer: Result<Option<RefMeta>, &'static str>) -> Self {
            Self {
                answer,
                reads: AtomicUsize::new(0),
            }
        }
        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl RefSource for Fake {
        async fn load_unchecked(&self, _r: &EntityRef) -> anyhow::Result<Option<RefMeta>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.answer.clone().map_err(|e| anyhow::anyhow!(e))
        }
    }

    /// A rule that records that it was asked, and answers a fixed verdict.
    /// A test that expects a verdict from the policy must see `asked() == 1`:
    /// otherwise the outcome came from somewhere else.
    struct Spy {
        verdict: bool,
        asked: AtomicUsize,
    }

    impl Spy {
        fn new(verdict: bool) -> Arc<Self> {
            Arc::new(Self {
                verdict,
                asked: AtomicUsize::new(0),
            })
        }
        fn asked(&self) -> usize {
            self.asked.load(Ordering::SeqCst)
        }
    }

    impl ScopeRule for Spy {
        fn allows(&self, _p: &Principal, _m: &RefMeta) -> bool {
            self.asked.fetch_add(1, Ordering::SeqCst);
            self.verdict
        }
    }

    fn user() -> Principal {
        Principal::User(Uuid::parse_str(B).unwrap())
    }

    #[tokio::test]
    async fn allowed_by_the_rule_is_found_and_the_rule_spoke() {
        let r = rf(RefKind::Plan, A);
        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let src = Fake::new(Ok(Some(meta_for(&r))));
        let out = policy.resolve_checked(&user(), &src, &r).await;
        assert_eq!(out, Resolution::Found(Box::new(meta_for(&r))));
        assert_eq!(spy.asked(), 1, "the verdict must come from the rule");
        assert_eq!(src.reads(), 1);
    }

    #[tokio::test]
    async fn denied_by_the_rule_is_forbidden_and_the_rule_spoke() {
        let r = rf(RefKind::Note, A);
        let spy = Spy::new(false);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let src = Fake::new(Ok(Some(meta_for(&r))));
        let out = policy.resolve_checked(&user(), &src, &r).await;
        assert_eq!(out, Resolution::Forbidden);
        assert_eq!(spy.asked(), 1, "the denial must come from the rule");
    }

    #[tokio::test]
    async fn a_missing_entity_never_reaches_the_rule() {
        let r = rf(RefKind::Task, A);
        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let src = Fake::new(Ok(None));
        assert_eq!(
            policy.resolve_checked(&user(), &src, &r).await,
            Resolution::NotFound
        );
        assert_eq!(spy.asked(), 0);
    }

    #[tokio::test]
    async fn an_unauthenticated_caller_is_refused_before_the_store_is_read() {
        let r = rf(RefKind::Plan, A);
        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let src = Fake::new(Ok(Some(meta_for(&r))));
        let out = policy
            .resolve_checked(&Principal::Unauthenticated, &src, &r)
            .await;
        assert_eq!(out, Resolution::Forbidden);
        assert_eq!(src.reads(), 0, "no read for an anonymous-less caller");
        assert_eq!(spy.asked(), 0);
    }

    #[tokio::test]
    async fn a_store_error_fails_closed() {
        let r = rf(RefKind::Decision, A);
        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let src = Fake::new(Err("neo4j down"));
        assert_eq!(
            policy.resolve_checked(&user(), &src, &r).await,
            Resolution::Failed
        );
        assert_eq!(spy.asked(), 0);
    }

    #[tokio::test]
    async fn a_source_answering_for_another_entity_is_not_found() {
        let asked = rf(RefKind::Plan, A);
        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);

        let other_id = rf(RefKind::Plan, B);
        let src = Fake::new(Ok(Some(meta_for(&other_id))));
        assert_eq!(
            policy.resolve_checked(&user(), &src, &asked).await,
            Resolution::NotFound
        );

        let other_kind = rf(RefKind::Task, A);
        let src = Fake::new(Ok(Some(meta_for(&other_kind))));
        assert_eq!(
            policy.resolve_checked(&user(), &src, &asked).await,
            Resolution::NotFound
        );
        assert_eq!(spy.asked(), 0, "a mismatched answer is never judged");
    }

    #[tokio::test]
    async fn the_open_instance_policy_admits_signed_in_callers_only() {
        let r = rf(RefKind::Rfc, A);
        let policy = AccessPolicy::open_instance();
        let src = Fake::new(Ok(Some(meta_for(&r))));
        assert!(matches!(
            policy.resolve_checked(&user(), &src, &r).await,
            Resolution::Found(_)
        ));
        assert!(matches!(
            policy
                .resolve_checked(&Principal::LocalOpen, &src, &r)
                .await,
            Resolution::Found(_)
        ));
        assert_eq!(
            policy
                .resolve_checked(&Principal::Unauthenticated, &src, &r)
                .await,
            Resolution::Forbidden
        );
        assert_eq!(policy.disclosure(), Disclosure::Uniform);
    }

    #[test]
    fn the_open_rule_itself_refuses_an_unauthenticated_caller() {
        let m = meta_for(&rf(RefKind::Plan, A));
        assert!(OpenInstanceRule.allows(&user(), &m));
        assert!(OpenInstanceRule.allows(&Principal::LocalOpen, &m));
        assert!(OpenInstanceRule.allows(&Principal::Session(Default::default()), &m));
        assert!(!OpenInstanceRule.allows(&Principal::Unauthenticated, &m));
    }

    #[test]
    fn subjects_map_to_principals_and_garbage_fails_closed() {
        assert_eq!(
            Principal::from_subject(A),
            Principal::User(Uuid::parse_str(A).unwrap())
        );
        assert_eq!(
            Principal::from_subject("00000000-0000-0000-0000-000000000000"),
            Principal::LocalOpen
        );
        assert_eq!(Principal::from_subject("admin"), Principal::Unauthenticated);
        assert_eq!(Principal::from_subject(""), Principal::Unauthenticated);
    }

    #[test]
    fn uniform_disclosure_makes_denied_and_missing_indistinguishable() {
        let r = rf(RefKind::Plan, A);
        let denied = Resolution::Forbidden.to_wire(&r, Disclosure::Uniform);
        let missing = Resolution::NotFound.to_wire(&r, Disclosure::Uniform);
        let failed = Resolution::Failed.to_wire(&r, Disclosure::Uniform);
        assert_eq!(denied, missing);
        assert_eq!(failed, missing);
        assert_eq!(denied.status, RefStatus::NotFound);
        assert!(denied.label.is_none() && denied.project.is_none());
    }

    #[test]
    fn distinct_disclosure_only_separates_a_denial() {
        let r = rf(RefKind::Plan, A);
        assert_eq!(
            Resolution::Forbidden.status(Disclosure::Distinct),
            RefStatus::Forbidden
        );
        assert_eq!(
            Resolution::NotFound.status(Disclosure::Distinct),
            RefStatus::NotFound
        );
        assert_eq!(
            Resolution::Failed.status(Disclosure::Distinct),
            RefStatus::NotFound
        );
        let denied = Resolution::Forbidden.to_wire(&r, Disclosure::Distinct);
        assert!(denied.label.is_none(), "a denial never carries a label");
    }

    #[test]
    fn a_found_entity_carries_its_label_and_scope() {
        let r = rf(RefKind::Plan, A);
        let wire = Resolution::Found(Box::new(meta_for(&r))).to_wire(&r, Disclosure::Uniform);
        assert_eq!(wire.status, RefStatus::Ok);
        assert_eq!(wire.label.as_deref(), Some("Titre"));
        assert_eq!(wire.subtitle.as_deref(), Some("sous-titre"));
        assert_eq!(wire.project.as_ref().map(|p| p.slug.as_str()), Some("proj"));
        assert_eq!(wire.entity_status.as_deref(), Some("in_progress"));
    }

    // ---- the scope of a session, and the sensitive kinds ----------------

    fn label(n: u128, slug: &str) -> ScopeLabel {
        ScopeLabel {
            id: Uuid::from_u128(n),
            slug: slug.into(),
            name: slug.into(),
        }
    }

    const P1: u128 = 1;
    const P2: u128 = 2;
    const W1: u128 = 11;
    const W2: u128 = 12;

    fn meta_in(kind: RefKind, project: Option<u128>, workspace: Option<u128>) -> RefMeta {
        RefMeta {
            kind,
            id: Uuid::from_u128(99).into(),
            label: "x".into(),
            subtitle: None,
            project: project.map(|p| label(p, "p")),
            workspace: workspace.map(|w| label(w, "w")),
            entity_status: None,
        }
    }

    fn session(project: Option<u128>, workspace: Option<u128>) -> Principal {
        Principal::Session(SessionScope {
            project: project.map(Uuid::from_u128),
            workspace: workspace.map(Uuid::from_u128),
        })
    }

    fn allowed(p: &Principal, m: &RefMeta) -> bool {
        OpenInstanceRule.allows(p, m)
    }

    #[test]
    fn a_project_session_reads_its_own_project_and_no_other() {
        let me = session(Some(P1), Some(W1));
        assert!(allowed(&me, &meta_in(RefKind::Plan, Some(P1), Some(W1))));
        assert!(!allowed(&me, &meta_in(RefKind::Plan, Some(P2), Some(W1))));
        assert!(!allowed(&me, &meta_in(RefKind::Plan, Some(P2), None)));
        assert!(!allowed(&me, &meta_in(RefKind::Task, Some(P2), Some(W2))));
    }

    #[test]
    fn a_project_session_reads_its_workspace_and_no_other() {
        let me = session(Some(P1), Some(W1));
        assert!(allowed(&me, &meta_in(RefKind::Workspace, None, Some(W1))));
        assert!(!allowed(&me, &meta_in(RefKind::Workspace, None, Some(W2))));
        let no_workspace = session(Some(P1), None);
        assert!(!allowed(
            &no_workspace,
            &meta_in(RefKind::Workspace, None, Some(W1))
        ));
    }

    #[test]
    fn what_belongs_to_no_project_and_no_workspace_is_shared() {
        for me in [
            session(Some(P1), Some(W1)),
            session(None, Some(W1)),
            session(None, None),
        ] {
            assert!(allowed(&me, &meta_in(RefKind::Note, None, None)), "{me:?}");
            assert!(allowed(&me, &meta_in(RefKind::Link, None, None)), "{me:?}");
        }
    }

    #[test]
    fn a_workspace_session_reads_the_projects_of_its_workspace_only() {
        let me = session(None, Some(W1));
        assert!(allowed(&me, &meta_in(RefKind::Plan, Some(P1), Some(W1))));
        assert!(allowed(&me, &meta_in(RefKind::Project, Some(P2), Some(W1))));
        assert!(!allowed(&me, &meta_in(RefKind::Plan, Some(P1), Some(W2))));
        assert!(
            !allowed(&me, &meta_in(RefKind::Plan, Some(P1), None)),
            "a project of no workspace"
        );
    }

    #[test]
    fn a_session_attached_to_nothing_reads_what_the_instance_shares() {
        let me = session(None, None);
        assert!(allowed(&me, &meta_in(RefKind::Plan, Some(P2), Some(W2))));
        assert!(allowed(&me, &meta_in(RefKind::Plan, Some(P1), None)));
    }

    #[test]
    fn a_sensitive_kind_needs_a_project_and_a_session_that_can_prove_its_own() {
        for kind in [RefKind::File, RefKind::Commit] {
            let own = meta_in(kind, Some(P1), Some(W1));
            assert!(allowed(&session(Some(P1), Some(W1)), &own), "{kind}");
            assert!(
                allowed(&session(None, Some(W1)), &own),
                "{kind}: workspace session"
            );
            assert!(
                !allowed(&session(Some(P2), Some(W1)), &own),
                "{kind}: another project"
            );
            assert!(
                !allowed(&session(None, Some(W2)), &own),
                "{kind}: another workspace"
            );
            assert!(
                !allowed(&session(None, None), &own),
                "{kind}: unscoped session"
            );
            assert!(
                !allowed(&session(Some(P1), Some(W1)), &meta_in(kind, None, None)),
                "{kind}: no project"
            );
            assert!(
                !allowed(&session(Some(P1), Some(W1)), &meta_in(kind, None, Some(W1))),
                "{kind}: a workspace is not a project"
            );
            // the picker: a signed-in user, whose search names the scope
            assert!(
                allowed(&Principal::User(Uuid::from_u128(5)), &own),
                "{kind}"
            );
            assert!(
                !allowed(&Principal::LocalOpen, &meta_in(kind, None, None)),
                "{kind}"
            );
            assert!(!allowed(&Principal::Unauthenticated, &own), "{kind}");
        }
    }

    /// persona and skill are ACTORS, not sensitive data: a persona may be
    /// global (`PersonaNode.project_id` is optional; a skill always has a
    /// project). They follow the ordinary session rule: a project session
    /// reads the actors of its project and the global ones, never another
    /// project's; a session attached to nothing reads the instance's actors.
    #[test]
    fn an_actor_follows_the_ordinary_scope_and_a_global_persona_is_shared() {
        for kind in [RefKind::Persona, RefKind::Skill] {
            assert!(!kind.is_sensitive(), "{kind}");
            let mine = session(Some(P1), Some(W1));
            assert!(allowed(&mine, &meta_in(kind, Some(P1), Some(W1))), "{kind}");
            assert!(
                !allowed(&mine, &meta_in(kind, Some(P2), Some(W1))),
                "{kind}: other project"
            );
            assert!(allowed(&mine, &meta_in(kind, None, None)), "{kind}: global");
            assert!(
                allowed(&session(None, None), &meta_in(kind, Some(P2), None)),
                "{kind}"
            );
            assert!(
                !allowed(&Principal::Unauthenticated, &meta_in(kind, None, None)),
                "{kind}"
            );
        }
    }
    #[test]
    fn a_scoped_denial_reads_not_found_never_forbidden() {
        let r = rf(RefKind::Plan, A);
        let mut m = meta_for(&r);
        m.project = Some(label(P2, "other"));
        let policy = AccessPolicy::open_instance();
        let outcome = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(policy.resolve_checked(
                &session(Some(P1), None),
                &Fake::new(Ok(Some(m))),
                &r,
            ));
        assert_eq!(outcome, Resolution::Forbidden);
        assert_eq!(outcome.status(policy.disclosure()), RefStatus::NotFound);
    }
}
