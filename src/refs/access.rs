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

use super::types::{EntityRef, RefKind};
use super::wire::{RefResolution, RefStatus, ScopeLabel};

/// Nil UUID: the subject the auth layer injects in no-auth mode.
const ANONYMOUS_SUBJECT: Uuid = Uuid::nil();

/// The caller, as far as access is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Principal {
    /// A signed-in user (or an agent acting for one).
    User(Uuid),
    /// No-auth mode: the instance is open by configuration.
    LocalOpen,
    /// The session that is answering a turn. The message was authenticated
    /// when the API took it in; after that (queue, drain, another instance)
    /// no identity travels with it, so the turn acts as the session itself.
    /// Today that reads what a signed-in user reads; the day a project or
    /// workspace rule exists it is the seam where the session's envelope goes.
    Session,
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
    pub id: Uuid,
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

/// Today's reality: signed-in users read the whole instance.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenInstanceRule;

impl ScopeRule for OpenInstanceRule {
    fn allows(&self, principal: &Principal, _meta: &RefMeta) -> bool {
        matches!(
            principal,
            Principal::User(_) | Principal::LocalOpen | Principal::Session
        )
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
                id: r.id,
                status,
                label: Some(m.label.clone()),
                subtitle: m.subtitle.clone(),
                project: m.project.clone(),
                workspace: m.workspace.clone(),
                entity_status: m.entity_status.clone(),
            },
            _ => RefResolution {
                kind: r.kind,
                id: r.id,
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
            id: r.id,
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
        assert!(OpenInstanceRule.allows(&Principal::Session, &m));
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
}
