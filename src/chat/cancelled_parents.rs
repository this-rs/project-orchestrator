//! The "parent cancelled" tombstones (B36-RACE).
//!
//! A delegated session's node (its `spawned_by` names the parent) is written well BEFORE
//! the session is live: its provider is opened in between (lasting scope, grants, the CLI
//! or the agent launched). A cascade close of the parent
//! ([`super::manager::ChatManager::close_session_with_cascade`]) that reads the parent's
//! children in that window sees this child INACTIVE and skips it; once live, nothing would
//! ever close it, and the delegation outlives the run it belonged to. Reading inactive
//! descendants again is no answer: a descendant closed long ago is inactive too, so every
//! cancel would have to wait.
//!
//! The fix is at the source: the cascade leaves a tombstone on each session it closes,
//! BEFORE it closes it (and so before its children are read), and every activation of a
//! session whose `spawned_by` names a parent checks the parent's tombstone INSIDE the
//! critical section that inserts it in the live map (the write lock of `active_sessions`
//! on the legacy engine, of the agent runtime's sessions on the agent engine). Either:
//!
//! * the child was inserted BEFORE the tombstone was set — then the cascade, which reads
//!   the children and their liveness only after setting it, sees the child LIVE and closes
//!   it; or
//! * it is inserted AFTER — then it reads the tombstone and is refused: never inserted, its
//!   provider closed, the delegation answered [`ParentCancelled`].
//!
//! The tombstone's own mutex is taken inside the insertion's critical section, so the two
//! orders are the only ones: a check that read "no tombstone" holds the live map's write
//! lock until the insert is done, and the cascade's later read of that map waits for it.
//! No cancel waits for anything: setting a tombstone is a map insert.
//!
//! Tombstones are bounded in time ([`TOMBSTONE_TTL`]): an in-flight delegation lands within
//! its provider's opening timeout, far below it. Expired ones are swept on every access, so
//! the map holds only the cascades of the last few minutes. A session activated again (a
//! person resumes the cancelled parent) loses its tombstone: its new delegations are not
//! refused — nor are the delegations of that parent still opening at that moment (at most
//! an opening's timeout, `PROVIDER_OPEN_TIMEOUT`): the person took the parent back, its
//! children may live again.
//!
//! WHO is refused (B36-RACE-RESUME). The tombstone stops a DELEGATION that lands late, not
//! a person: a child of a cancelled run that someone reopens by hand right after the cancel
//! must open. A session CREATED with a parent is always a delegation (an `Agent` call, a
//! `delegate_task`, a runner): it is always checked. A RESUME is a delegation only when the
//! delegation path says so — the warm pool resuming a closed child for a new `Agent` call
//! ([`as_delegation`]); every other resume (the chat, the API, the NATS RPC) is a person's,
//! and passes. The marker is a task-local, so it travels down the awaited resume to the
//! `admit` under the live map's lock without threading a flag through every reopen path.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::HashMap;
use std::future::Future;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a cascade's tombstone refuses a late child. Well above an opening's timeout (an
/// in-flight delegation lands well within it), short enough that nothing accumulates.
pub(crate) const TOMBSTONE_TTL: Duration = Duration::from_secs(5 * 60);

/// A delegated session refused at its activation: its parent was cancelled (closed with
/// cascade) while it was being opened. The delegation that opened it reads this error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("parent session {parent_session_id} was cancelled: the delegated session {session_id} was not started")]
pub struct ParentCancelled {
    pub parent_session_id: String,
    pub session_id: String,
}

/// The code a refused activation answers with (HTTP 409, `OpenFailure::code`).
pub const PARENT_CANCELLED: &str = "parent_cancelled";

impl ParentCancelled {
    /// The refusal anywhere in `err`'s chain: the openings wrap their errors (the provider
    /// they were raised on, `provider::errors::on_provider`).
    pub fn in_chain(err: &anyhow::Error) -> Option<&ParentCancelled> {
        err.chain()
            .find_map(|cause| cause.downcast_ref::<ParentCancelled>())
    }

    /// What the API answers: a conflict, not retryable (the run it belonged to is over).
    pub fn failure(&self) -> super::provider::errors::OpenFailure {
        super::provider::errors::OpenFailure {
            status: 409,
            code: PARENT_CANCELLED,
            message: self.to_string(),
            provider_id: None,
            action: None,
            retryable: false,
            retry_after_ms: None,
            resets_at_ms: None,
            fallbacks: None,
            consent_scope: None,
            candidates: None,
        }
    }
}

/// The sessions a cascade closed in the last [`TOMBSTONE_TTL`].
#[derive(Debug)]
pub(crate) struct CancelledParents {
    marks: Mutex<HashMap<String, Instant>>,
    ttl: Duration,
}

impl Default for CancelledParents {
    fn default() -> Self {
        Self::with_ttl(TOMBSTONE_TTL)
    }
}

impl CancelledParents {
    pub(crate) fn with_ttl(ttl: Duration) -> Self {
        Self {
            marks: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, HashMap<String, Instant>> {
        let mut marks = self
            .marks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ttl = self.ttl;
        let now = Instant::now();
        marks.retain(|_, at| now.duration_since(*at) < ttl);
        marks
    }

    /// `session_id` is being closed by a cascade: a child of it activated from now on is
    /// refused. Set BEFORE the session is closed and its children read.
    pub(crate) fn mark(&self, session_id: &str) {
        self.locked().insert(session_id.to_owned(), Instant::now());
    }

    /// Whether `parent_session_id` was closed by a cascade within the TTL.
    #[cfg(test)]
    pub(crate) fn is_marked(&self, parent_session_id: &str) -> bool {
        self.locked().contains_key(parent_session_id)
    }

    /// The activation of `session_id`, whose `spawned_by` names `parent` (if any): refused
    /// when the parent carries a tombstone. Admitted, the session itself loses its own (a
    /// cancelled parent that a person resumes delegates again). Called INSIDE the critical
    /// section that inserts the session in its live map (module documentation).
    pub(crate) fn admit(
        &self,
        session_id: &str,
        parent: Option<&str>,
    ) -> Result<(), ParentCancelled> {
        let mut marks = self.locked();
        if let Some(parent) = parent.filter(|p| marks.contains_key(*p)) {
            return Err(ParentCancelled {
                parent_session_id: parent.to_owned(),
                session_id: session_id.to_owned(),
            });
        }
        marks.remove(session_id);
        Ok(())
    }

    /// Number of live tombstones (expired ones swept).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.locked().len()
    }
}

tokio::task_local! {
    /// Set while a delegation (re)opens its child: see the module documentation.
    static DELEGATED_OPENING: ();
}

/// Runs `opening` as a DELEGATION's: a resume inside it is checked against the parent's
/// tombstone like a fresh child. Only the delegation path wraps its resume in this.
pub(crate) async fn as_delegation<F: Future>(opening: F) -> F::Output {
    DELEGATED_OPENING.scope((), opening).await
}

/// Whether the current task opens a session on behalf of a delegation ([`as_delegation`]).
pub(crate) fn opening_is_delegation() -> bool {
    DELEGATED_OPENING.try_with(|_| ()).is_ok()
}

/// The parent whose tombstone a RESUME of a session is checked against: its `spawned_by`
/// parent when the resume is a delegation's ([`as_delegation`]), none when a person resumes
/// it (B36-RACE-RESUME). A creation is always checked: it uses [`parent_of`] directly.
pub(crate) fn parent_checked_on_resume(spawned_by: Option<&str>) -> Option<String> {
    if opening_is_delegation() {
        parent_of(spawned_by)
    } else {
        None
    }
}

/// The parent a session's `spawned_by` names, if any.
pub(crate) fn parent_of(spawned_by: Option<&str>) -> Option<String> {
    spawned_by
        .and_then(super::manager::parse_spawned_by)
        .and_then(|ctx| ctx.parent_session_id)
        .filter(|p| !p.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    /// B36-RACE-RESUME: the resume's parent is checked only inside a delegation, and the
    /// marker reaches nested awaits (the reopen paths between the caller and `admit`).
    #[tokio::test]
    async fn a_resume_checks_the_parent_only_inside_a_delegation() {
        let spawned_by = r#"{"type":"delegation","parent_session_id":"p"}"#;
        assert!(!opening_is_delegation());
        assert_eq!(
            parent_checked_on_resume(Some(spawned_by)),
            None,
            "a person's resume"
        );

        let inside = as_delegation(async {
            // A nested await, as on the real reopen paths.
            tokio::task::yield_now().await;
            parent_checked_on_resume(Some(spawned_by))
        })
        .await;
        assert_eq!(inside.as_deref(), Some("p"), "a delegation's resume");
        assert!(
            !opening_is_delegation(),
            "the marker does not outlive its scope"
        );
    }

    #[test]
    fn a_child_of_a_cancelled_parent_is_refused_with_a_typed_error() {
        let tombs = CancelledParents::default();
        tombs.mark("p");
        let err = tombs.admit("c", Some("p")).unwrap_err();
        assert_eq!(
            err,
            ParentCancelled {
                parent_session_id: "p".into(),
                session_id: "c".into()
            }
        );
        assert!(err.to_string().contains("parent session p was cancelled"));
    }

    #[test]
    fn the_refusal_is_found_through_a_wrapped_error_and_answers_a_conflict() {
        let refused = ParentCancelled {
            parent_session_id: "p".into(),
            session_id: "c".into(),
        };
        let wrapped = super::super::provider::errors::on_provider(
            "local",
            anyhow::Error::new(refused.clone()),
        );
        assert_eq!(ParentCancelled::in_chain(&wrapped), Some(&refused));
        let failure = super::super::provider::errors::classify_open_error(&wrapped, None)
            .expect("classified");
        assert_eq!((failure.status, failure.code), (409, PARENT_CANCELLED));
        assert!(!failure.retryable);
    }

    #[test]
    fn a_session_without_parent_or_with_a_live_parent_is_admitted() {
        let tombs = CancelledParents::default();
        tombs.mark("p");
        assert!(tombs.admit("a", None).is_ok());
        assert!(tombs.admit("b", Some("other")).is_ok());
    }

    #[test]
    fn a_tombstone_expires() {
        let tombs = CancelledParents::with_ttl(Duration::from_millis(20));
        tombs.mark("p");
        assert!(tombs.is_marked("p"));
        std::thread::sleep(Duration::from_millis(40));
        assert!(tombs.admit("c", Some("p")).is_ok(), "expired: admitted");
        assert_eq!(tombs.len(), 0, "swept, nothing kept");
    }

    #[test]
    fn a_cancelled_parent_activated_again_loses_its_tombstone() {
        let tombs = CancelledParents::default();
        tombs.mark("p");
        assert!(tombs.admit("p", None).is_ok());
        assert!(!tombs.is_marked("p"));
        assert!(tombs.admit("c", Some("p")).is_ok());
    }

    #[test]
    fn the_parent_is_read_from_spawned_by() {
        let sb = serde_json::json!({"type": "delegation", "parent_session_id": "p-1"}).to_string();
        assert_eq!(parent_of(Some(&sb)).as_deref(), Some("p-1"));
        assert_eq!(parent_of(None), None);
        assert_eq!(parent_of(Some(r#"{"type":"runner"}"#)), None);
    }
}
