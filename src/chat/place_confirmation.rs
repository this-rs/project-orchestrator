//! « Exécute dans X » — the confirmation a PERSON gives before a delegated task writes in an
//! ephemeral worktree (T5).
//!
//! A delegation asked with `execution_place: "ephemeral_worktree"` opens nothing at once. PO
//! resolves the place X itself (`runner::ephemeral_worktree::plan`: the repository, a frozen base
//! SHA, a branch and the path, never taken from a model's text) and keeps a PENDING placement
//! here. Only a person confirms it, the same rule as the network consent of N23b:
//!
//! - **Who may ask.** A request is refused outright, and no placement is kept, when nobody would
//!   be there to answer ([`person_in_front`]): an agent session that is a child (delegated, an
//!   `Agent` sub-agent, a runner or a protocol session — `spawned_by` set), a session the server
//!   opened (no owner), the anonymous user of a server without authentication, an agent / MCP /
//!   vault token or the server's own account calling directly. The model can never be the one
//!   that answers its own request.
//! - **Who may confirm.** A person signed in with their own token
//!   (`Claims::may_keep_lasting_approvals`), and the OWNER of the placement: the person in front
//!   of the session that asked (its `owner`), or the person who asked directly. An agent token
//!   never confirms, whoever it acts for. Consent given in advance (EXEC-CONFIRM): when the
//!   asking SESSION runs in bypassPermissions — a mode only its person sets — the HARNESS answers
//!   for that owner ([`PlacementBook::confirm_for_owner`]), with the path it reads from the book;
//!   the audit writes the cause `auto:bypassPermissions`.
//! - **What is confirmed.** The confirmation must name X exactly as PO resolved it: a
//!   confirmation of another path is refused (`execution_place_mismatch`) and the placement stays
//!   pending.
//!
//! Every step is written to the append-only audit (`ExecutionPlaceEvent`) by the handlers.
//! Requests are idempotent (the same task, owner and repository give back the same pending
//! placement) and bounded ([`MAX_PENDING_PER_OWNER`], [`MAX_PENDING`], [`PENDING_TTL`]).

use std::collections::HashMap;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::auth::jwt::{Claims, ANONYMOUS_USER_ID};
use crate::neo4j::models::ChatSessionNode;
use crate::runner::ephemeral_worktree::PlannedWorktree;

/// How long a placement waits for its person.
pub const PENDING_TTL: chrono::Duration = chrono::Duration::minutes(15);
/// Pending placements one person may have at once.
pub const MAX_PENDING_PER_OWNER: usize = 4;
/// Pending placements on the whole server.
pub const MAX_PENDING: usize = 32;

/// Why a placement cannot be asked, confirmed or rejected. The message starts with a stable code.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("execution_place_no_person: {0}")]
    NoPersonInFront(&'static str),
    #[error("execution_place_not_a_person: only a person signed in with their own token confirms « exécute dans X »")]
    NotAPerson,
    #[error("execution_place_not_the_owner: only the person in front of the session that asked may answer")]
    NotTheOwner,
    #[error("execution_place_mismatch: the confirmation names {seen}, PO resolved {resolved}")]
    PlaceMismatch { seen: String, resolved: String },
    #[error("execution_place_not_found: no pending placement {0}")]
    NotFound(Uuid),
    #[error("execution_place_expired: placement {0} waited longer than its time to live")]
    Expired(Uuid),
    #[error("execution_place_too_many: {0}")]
    TooMany(&'static str),
}

impl PlacementError {
    /// The HTTP error for this refusal.
    pub fn into_app_error(self) -> crate::api::handlers::AppError {
        use crate::api::handlers::AppError;
        match self {
            Self::NoPersonInFront(_) | Self::NotAPerson | Self::NotTheOwner => {
                AppError::Forbidden(self.to_string())
            }
            Self::PlaceMismatch { .. } | Self::TooMany(_) => AppError::Conflict(self.to_string()),
            Self::NotFound(_) => AppError::NotFound(self.to_string()),
            Self::Expired(_) => AppError::Gone(self.to_string()),
        }
    }
}

/// Who asked for the placement, as the audit writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requester {
    /// A person, directly (REST with their own token).
    Person { sub: String },
    /// A chat session a person is in front of.
    Session { session_id: Uuid },
}

impl Requester {
    pub fn audit_label(&self) -> String {
        match self {
            Self::Person { sub } => format!("person:{sub}"),
            Self::Session { session_id } => format!("session:{session_id}"),
        }
    }
}

/// The person who must answer a request: the owner of the placement, or the refusal.
///
/// `parent` is the session that asked (`None`: the caller is not a session), `claims` the
/// caller's token. Read from the session's own shape and the signed token, never from anything
/// the model says.
pub fn person_in_front(
    claims: Option<&Claims>,
    parent: Option<&ChatSessionNode>,
) -> Result<(String, Requester), PlacementError> {
    match parent {
        Some(session) => {
            if session
                .spawned_by
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty())
            {
                return Err(PlacementError::NoPersonInFront(
                    "the asking session was spawned (a delegated task, a sub-agent, a runner or a protocol run): nobody is in front of it to answer",
                ));
            }
            let owner = session
                .owner
                .as_deref()
                .map(str::trim)
                .filter(|o| !o.is_empty() && *o != ANONYMOUS_USER_ID.to_string());
            match owner {
                Some(owner) => Ok((
                    owner.to_string(),
                    Requester::Session {
                        session_id: session.id,
                    },
                )),
                None => Err(PlacementError::NoPersonInFront(
                    "the asking session belongs to no signed-in person",
                )),
            }
        }
        None => match claims {
            Some(c) if c.may_keep_lasting_approvals() => {
                Ok((c.sub.clone(), Requester::Person { sub: c.sub.clone() }))
            }
            _ => Err(PlacementError::NoPersonInFront(
                "the caller is not a person signed in with their own token",
            )),
        },
    }
}

/// A placement waiting for its person. `T` is what the confirmation launches.
#[derive(Debug, Clone)]
pub struct Pending<T> {
    pub id: Uuid,
    pub task_id: Uuid,
    pub plan_id: Uuid,
    pub owner: String,
    pub requester: Requester,
    pub place: PlannedWorktree,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub launch: T,
}

/// What a request gave back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requested {
    /// A new placement.
    New(Uuid),
    /// The same task, owner and repository were already waiting: that one (idempotent).
    Existing(Uuid),
}

impl Requested {
    pub fn id(&self) -> Uuid {
        match self {
            Self::New(id) | Self::Existing(id) => *id,
        }
    }
}

/// The pending placements of the server.
pub struct PlacementBook<T> {
    pending: Mutex<HashMap<Uuid, Pending<T>>>,
}

impl<T: Clone> Default for PlacementBook<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> PlacementBook<T> {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Uuid, Pending<T>>> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Keep a placement for `owner` (see the module for the limits). `now` is passed so the
    /// time to live is testable.
    #[allow(clippy::too_many_arguments)]
    pub fn request(
        &self,
        task_id: Uuid,
        plan_id: Uuid,
        owner: String,
        requester: Requester,
        place: PlannedWorktree,
        launch: T,
        now: DateTime<Utc>,
    ) -> Result<Requested, PlacementError> {
        let mut pending = self.lock();
        pending.retain(|_, p| p.expires_at > now);
        if let Some(existing) = pending.values().find(|p| {
            p.task_id == task_id && p.owner == owner && p.place.repository == place.repository
        }) {
            return Ok(Requested::Existing(existing.id));
        }
        if pending.values().filter(|p| p.owner == owner).count() >= MAX_PENDING_PER_OWNER {
            return Err(PlacementError::TooMany(
                "this person already has the maximum of placements waiting for an answer",
            ));
        }
        if pending.len() >= MAX_PENDING {
            return Err(PlacementError::TooMany(
                "the server already has the maximum of placements waiting for an answer",
            ));
        }
        let id = Uuid::new_v4();
        pending.insert(
            id,
            Pending {
                id,
                task_id,
                plan_id,
                owner,
                requester,
                place,
                created_at: now,
                expires_at: now + PENDING_TTL,
                launch,
            },
        );
        Ok(Requested::New(id))
    }

    /// Drop a placement without an answer (its audit entry could not be written).
    pub fn discard(&self, id: Uuid) {
        self.lock().remove(&id);
    }

    /// A copy of a pending placement (expired ones are not shown).
    pub fn get(&self, id: Uuid, now: DateTime<Utc>) -> Option<Pending<T>> {
        self.lock().get(&id).filter(|p| p.expires_at > now).cloned()
    }

    /// Check who answers: a person, the owner. Shared by confirm and reject.
    fn check_answerer(p: &Pending<T>, answerer: Option<&Claims>) -> Result<(), PlacementError> {
        let answerer = answerer
            .filter(|c| c.may_keep_lasting_approvals())
            .ok_or(PlacementError::NotAPerson)?;
        if answerer.sub != p.owner {
            return Err(PlacementError::NotTheOwner);
        }
        Ok(())
    }

    fn live(
        pending: &mut HashMap<Uuid, Pending<T>>,
        id: Uuid,
        now: DateTime<Utc>,
    ) -> Result<(), PlacementError> {
        match pending.get(&id) {
            None => Err(PlacementError::NotFound(id)),
            Some(p) if p.expires_at <= now => {
                pending.remove(&id);
                Err(PlacementError::Expired(id))
            }
            Some(_) => Ok(()),
        }
    }

    /// Take the placement out for launching: `answerer` is the caller's token, `seen` the path the
    /// person was shown. A refusal leaves the placement pending (an agent's attempt does not
    /// consume the person's question). Taking it out makes a second confirmation fail
    /// (`not_found`): one confirmation, one launch.
    pub fn confirm(
        &self,
        id: Uuid,
        answerer: Option<&Claims>,
        seen: &str,
        now: DateTime<Utc>,
    ) -> Result<Pending<T>, PlacementError> {
        let mut pending = self.lock();
        Self::live(&mut pending, id, now)?;
        let p = pending.get(&id).expect("checked live");
        Self::check_answerer(p, answerer)?;
        let resolved = p.place.path.to_string_lossy().to_string();
        if seen != resolved {
            return Err(PlacementError::PlaceMismatch {
                seen: seen.to_string(),
                resolved,
            });
        }
        Ok(pending.remove(&id).expect("checked live"))
    }

    /// The HARNESS answers for the person who stands in front of the session that asked, because
    /// that person gave their consent in advance (the session runs in bypassPermissions). It is
    /// never reachable from a request: the caller is server code that read the mode itself.
    ///
    /// Same checks as [`Self::confirm`], minus the token: the placement must have been asked by a
    /// SESSION (a person in front of it, see [`person_in_front`]), `owner` must be its owner, and
    /// `seen` is the path the HARNESS read from the placement, so the object of the question is
    /// never the model's. Returns the owner as the actor.
    pub fn confirm_for_owner(
        &self,
        id: Uuid,
        owner: &str,
        seen: &str,
        now: DateTime<Utc>,
    ) -> Result<Pending<T>, PlacementError> {
        let mut pending = self.lock();
        Self::live(&mut pending, id, now)?;
        let p = pending.get(&id).expect("checked live");
        if !matches!(p.requester, Requester::Session { .. }) {
            return Err(PlacementError::NoPersonInFront(
                "only a session a person is in front of can have its consent given in advance",
            ));
        }
        if p.owner != owner {
            return Err(PlacementError::NotTheOwner);
        }
        let resolved = p.place.path.to_string_lossy().to_string();
        if seen != resolved {
            return Err(PlacementError::PlaceMismatch {
                seen: seen.to_string(),
                resolved,
            });
        }
        Ok(pending.remove(&id).expect("checked live"))
    }

    /// The person says no: the placement is dropped.
    pub fn reject(
        &self,
        id: Uuid,
        answerer: Option<&Claims>,
        now: DateTime<Utc>,
    ) -> Result<Pending<T>, PlacementError> {
        let mut pending = self.lock();
        Self::live(&mut pending, id, now)?;
        Self::check_answerer(pending.get(&id).expect("checked live"), answerer)?;
        Ok(pending.remove(&id).expect("checked live"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn place(path: &str) -> PlannedWorktree {
        PlannedWorktree {
            repository: PathBuf::from("/work/repo"),
            base_ref: "HEAD".into(),
            base_sha: "abc".into(),
            branch: "po/task-x".into(),
            path: PathBuf::from(path),
        }
    }

    fn person(sub: &str) -> Claims {
        Claims::person(sub, "p@example.org", "P")
    }

    fn agent_token(sub: &str) -> Claims {
        let mut c = person(sub);
        c.token_type = Some(crate::auth::jwt::TOKEN_TYPE_AGENT_SESSION.to_string());
        c
    }

    fn session(owner: Option<&str>, spawned_by: Option<&str>) -> ChatSessionNode {
        let mut s = crate::test_helpers::test_chat_session(Some("demo"));
        s.owner = owner.map(str::to_string);
        s.spawned_by = spawned_by.map(str::to_string);
        s
    }

    const SUB: &str = "11111111-1111-1111-1111-111111111111";

    fn book_with_one() -> (PlacementBook<()>, Uuid) {
        let book = PlacementBook::new();
        let id = book
            .request(
                Uuid::new_v4(),
                Uuid::new_v4(),
                SUB.into(),
                Requester::Person { sub: SUB.into() },
                place("/app/task-worktrees/x"),
                (),
                Utc::now(),
            )
            .unwrap()
            .id();
        (book, id)
    }

    #[test]
    fn a_session_with_nobody_in_front_is_refused_without_a_question() {
        let spawned = session(Some(SUB), Some(r#"{"type":"delegation"}"#));
        assert!(matches!(
            person_in_front(None, Some(&spawned)),
            Err(PlacementError::NoPersonInFront(_))
        ));
        let server_opened = session(None, None);
        assert!(matches!(
            person_in_front(None, Some(&server_opened)),
            Err(PlacementError::NoPersonInFront(_))
        ));
        let anonymous = session(Some(&ANONYMOUS_USER_ID.to_string()), None);
        assert!(matches!(
            person_in_front(None, Some(&anonymous)),
            Err(PlacementError::NoPersonInFront(_))
        ));
        // A direct caller that is not a person: an agent token, the server's account, nobody.
        for claims in [
            Some(agent_token(SUB)),
            Some(Claims::service_account("delegate-agent:x")),
            Some(Claims::anonymous()),
            None,
        ] {
            assert!(
                matches!(
                    person_in_front(claims.as_ref(), None),
                    Err(PlacementError::NoPersonInFront(_))
                ),
                "{claims:?}"
            );
        }
    }

    #[test]
    fn the_owner_of_a_human_session_or_the_direct_person_answers() {
        let s = session(Some(SUB), None);
        let (owner, requester) = person_in_front(None, Some(&s)).unwrap();
        assert_eq!(owner, SUB);
        assert_eq!(requester, Requester::Session { session_id: s.id });
        let (owner, _) = person_in_front(Some(&person("someone")), None).unwrap();
        assert_eq!(owner, "someone");
    }

    #[test]
    fn an_agent_token_never_confirms_even_for_its_own_person() {
        let (book, id) = book_with_one();
        let err = book
            .confirm(
                id,
                Some(&agent_token(SUB)),
                "/app/task-worktrees/x",
                Utc::now(),
            )
            .unwrap_err();
        assert_eq!(err, PlacementError::NotAPerson);
        for c in [
            Some(Claims::service_account("x")),
            Some(Claims::anonymous()),
            None,
        ] {
            assert_eq!(
                book.confirm(id, c.as_ref(), "/app/task-worktrees/x", Utc::now())
                    .unwrap_err(),
                PlacementError::NotAPerson
            );
        }
        // Still pending: an agent's attempt does not consume the person's question.
        assert!(book.get(id, Utc::now()).is_some());
        assert!(book
            .confirm(id, Some(&person(SUB)), "/app/task-worktrees/x", Utc::now())
            .is_ok());
    }

    #[test]
    fn another_person_cannot_confirm() {
        let (book, id) = book_with_one();
        assert_eq!(
            book.confirm(
                id,
                Some(&person("other")),
                "/app/task-worktrees/x",
                Utc::now()
            )
            .unwrap_err(),
            PlacementError::NotTheOwner
        );
        assert_eq!(
            book.reject(id, Some(&person("other")), Utc::now())
                .unwrap_err(),
            PlacementError::NotTheOwner
        );
    }

    #[test]
    fn the_confirmation_must_name_the_place_po_resolved() {
        let (book, id) = book_with_one();
        let err = book
            .confirm(id, Some(&person(SUB)), "/home/me/elsewhere", Utc::now())
            .unwrap_err();
        assert!(matches!(err, PlacementError::PlaceMismatch { .. }));
        assert!(book.get(id, Utc::now()).is_some());
    }

    #[test]
    fn one_confirmation_one_launch() {
        let (book, id) = book_with_one();
        book.confirm(id, Some(&person(SUB)), "/app/task-worktrees/x", Utc::now())
            .unwrap();
        assert_eq!(
            book.confirm(id, Some(&person(SUB)), "/app/task-worktrees/x", Utc::now())
                .unwrap_err(),
            PlacementError::NotFound(id)
        );
    }

    #[test]
    fn an_expired_placement_cannot_be_confirmed() {
        let (book, id) = book_with_one();
        let later = Utc::now() + PENDING_TTL + chrono::Duration::seconds(1);
        assert_eq!(
            book.confirm(id, Some(&person(SUB)), "/app/task-worktrees/x", later)
                .unwrap_err(),
            PlacementError::Expired(id)
        );
    }

    #[test]
    fn requests_are_idempotent_and_bounded() {
        let book: PlacementBook<()> = PlacementBook::new();
        let now = Utc::now();
        let task = Uuid::new_v4();
        let ask = |task: Uuid, path: &str| {
            book.request(
                task,
                Uuid::new_v4(),
                SUB.into(),
                Requester::Person { sub: SUB.into() },
                place(path),
                (),
                now,
            )
        };
        let first = ask(task, "/a").unwrap();
        assert!(matches!(first, Requested::New(_)));
        assert_eq!(ask(task, "/b").unwrap(), Requested::Existing(first.id()));
        for i in 1..MAX_PENDING_PER_OWNER {
            ask(Uuid::new_v4(), &format!("/c{i}")).unwrap();
        }
        assert!(matches!(
            ask(Uuid::new_v4(), "/d"),
            Err(PlacementError::TooMany(_))
        ));
    }

    #[test]
    fn the_harness_confirms_only_for_the_session_owner_and_the_exact_path() {
        let book: PlacementBook<()> = PlacementBook::new();
        let ask = |requester: Requester| {
            book.request(
                Uuid::new_v4(),
                Uuid::new_v4(),
                SUB.into(),
                requester,
                place("/work/x"),
                (),
                Utc::now(),
            )
            .unwrap()
            .id()
        };
        let id = ask(Requester::Session {
            session_id: Uuid::new_v4(),
        });
        assert!(matches!(
            book.confirm_for_owner(id, "mallory", "/work/x", Utc::now()),
            Err(PlacementError::NotTheOwner)
        ));
        assert!(matches!(
            book.confirm_for_owner(id, SUB, "/work/other", Utc::now()),
            Err(PlacementError::PlaceMismatch { .. })
        ));
        assert!(book
            .confirm_for_owner(id, SUB, "/work/x", Utc::now())
            .is_ok());
        assert!(matches!(
            book.confirm_for_owner(id, SUB, "/work/x", Utc::now()),
            Err(PlacementError::NotFound(_))
        ));
        // Asked directly by a person: nobody sits in front of a session, no advance consent.
        let id = ask(Requester::Person { sub: SUB.into() });
        assert!(matches!(
            book.confirm_for_owner(id, SUB, "/work/x", Utc::now()),
            Err(PlacementError::NoPersonInFront(_))
        ));
    }
}
