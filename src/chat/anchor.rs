//! Context anchors of a chat conversation: the model and its pure rules.
//!
//! A conversation is no longer defined by the directory it was started in; it
//! carries typed **anchors** towards entities of the graph (a file, a plan, a
//! task, a note, a project...). This module holds the vocabulary and every
//! rule, with no I/O: the Neo4j store and the in-memory mock both load the
//! anchors of a session, call [`apply_op`], and persist its result. Same
//! function, same rules, whichever store runs.
//!
//! # How the anchor relates to the existing links
//!
//! The three mechanisms answer different questions and none replaces another:
//!
//! - `DISCUSSED` (ChatSession -> entity): mentions **observed** in the
//!   conversation, written by the entity extractor. Unchanged.
//! - `ASSOCIATED_WITH` (ChatSession -> Plan/Task): the link between a session
//!   and the plan/task it works on. Unchanged.
//! - `ChatSession.project_slug`: the legacy project field. Unchanged.
//! - `Anchor` (this module): the **explicit** anchoring of a conversation, with
//!   roles, provenance, a state and an append-only journal.
//!
//! # Shape of the data
//!
//! One reified `Anchor` node per `(session_id, target_type, target_id)` (unique
//! key). There is no edge towards the target in v1 (targets carry
//! heterogeneous labels); the reverse query "conversations anchored on X" is
//! served by an index on `(target_type, target_id)`. Every change is journaled
//! as an `AnchorEvent` written in the same transaction as the anchor state.
//!
//! # Rules (all enforced by [`apply_op`])
//!
//! - at most [`MAX_FOCUS_ANCHORS`] anchors holding the `focus` role per session;
//! - at most one anchor holding the `origin` role, **immutable**: never removed,
//!   never demoted, never promoted onto, and `origin` can only be given when the
//!   anchor is created (its `state` may still follow its target, e.g. dangling);
//! - at most [`MAX_ANCHORS_PER_SESSION`] anchors per session;
//! - `by = agent` can only add a `mention`; it can neither receive nor promote
//!   `focus`, nor take `origin`, nor demote/remove an anchor holding `focus`;
//! - the target id is validated by type ([`validate_target_id`]);
//! - modifications carry the `version` the caller read (optimistic concurrency).

use crate::neo4j::models::ProjectNode;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use uuid::Uuid;

/// Maximum number of anchors of one session.
pub const MAX_ANCHORS_PER_SESSION: usize = 50;
/// Maximum number of anchors holding the `focus` role in one session.
pub const MAX_FOCUS_ANCHORS: usize = 3;
/// Maximum number of anchors holding the `origin` role in one session.
pub const MAX_ORIGIN_ANCHORS: usize = 1;
/// Longest accepted snapshot field / target id, in characters.
pub const MAX_FIELD_CHARS: usize = 1024;
/// Actor recorded on the anchors created by the project backfill.
pub const BACKFILL_ACTOR: &str = "migration:backfill_project_anchors";
/// Actor recorded when a target's deletion marks its anchors dangling.
pub const DANGLING_ACTOR: &str = "system:mark_anchors_dangling";

macro_rules! string_enum {
    ($(#[$m:meta])* $name:ident { $($variant:ident => $s:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];
            pub fn as_str(&self) -> &'static str {
                match self { $($name::$variant => $s),+ }
            }
            pub fn parse(s: &str) -> Option<Self> {
                match s { $($s => Some($name::$variant),)+ _ => None }
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

string_enum! {
    /// Entity types a conversation can be anchored on (v1).
    /// `reflection` and `diagnostic` are excluded: no node exists for them.
    AnchorTargetType {
        File => "file", Function => "function", Component => "component",
        Feature => "feature", Release => "release", Plan => "plan",
        Task => "task", Note => "note", Decision => "decision",
        Project => "project", Workspace => "workspace",
    }
}

string_enum! {
    /// What an anchor means for the conversation.
    AnchorRole { Focus => "focus", Mention => "mention", Origin => "origin" }
}

string_enum! {
    /// Lifecycle of an anchor, following its target.
    AnchorState {
        Live => "live", Moved => "moved", Archived => "archived",
        Dangling => "dangling", Unknown => "unknown",
    }
}

string_enum! {
    /// Who made the change.
    AnchorActor { User => "user", Agent => "agent", System => "system" }
}

string_enum! {
    /// Kind of journal entry.
    AnchorEventKind {
        Added => "added", Promoted => "promoted", Demoted => "demoted",
        Removed => "removed", StateChanged => "state_changed",
    }
}

/// A typed refusal. Never a panic: every rule violation is one of these.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AnchorError {
    #[error("invalid {target_type} target id: {reason}")]
    InvalidTargetId {
        target_type: AnchorTargetType,
        reason: String,
    },
    #[error("an anchor needs at least one role")]
    EmptyRoles,
    #[error("confidence must be a number between 0 and 1")]
    InvalidConfidence,
    #[error("field `{field}` is longer than {max} characters")]
    FieldTooLong { field: &'static str, max: usize },
    #[error("a session holds at most {max} focus anchors")]
    FocusCapReached { max: usize },
    #[error("a session holds at most {max} anchors")]
    SessionCapReached { max: usize },
    #[error("the session already has an origin anchor")]
    OriginAlreadyExists,
    #[error("the origin anchor is immutable")]
    OriginImmutable,
    #[error("an agent cannot receive or promote the focus role")]
    AgentCannotFocus,
    #[error("an agent may only add a mention (refused role: {0})")]
    AgentRoleForbidden(AnchorRole),
    #[error("the target is already anchored (anchor {anchor_id}); promote its role instead")]
    AlreadyAnchored { anchor_id: Uuid },
    #[error("anchor {0} not found in this session")]
    NotFound(Uuid),
    #[error("chat session {0} not found")]
    SessionNotFound(Uuid),
    #[error("version conflict: expected {expected}, found {actual}")]
    VersionConflict { expected: u64, actual: u64 },
    #[error("an anchor cannot be left without a role; remove it instead")]
    LastRole,
}

/// A stored anchor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub id: Uuid,
    pub session_id: Uuid,
    pub target_type: AnchorTargetType,
    pub target_id: String,
    pub roles: BTreeSet<AnchorRole>,
    pub state: AnchorState,
    pub by: AnchorActor,
    pub inferred: bool,
    pub confidence: f64,
    pub snapshot_name: Option<String>,
    pub snapshot_path: Option<String>,
    pub snapshot_type: Option<String>,
    /// Revision / sha of the target at anchoring time, when known.
    pub rev: Option<String>,
    /// Optimistic-concurrency counter, 1 at creation, +1 per change.
    pub version: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Anchor {
    pub fn has_role(&self, role: AnchorRole) -> bool {
        self.roles.contains(&role)
    }
}

/// One entry of the append-only journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorEvent {
    pub id: Uuid,
    pub session_id: Uuid,
    pub anchor_id: Uuid,
    pub kind: AnchorEventKind,
    pub by: AnchorActor,
    pub actor: String,
    pub at: DateTime<Utc>,
    pub before: Option<serde_json::Value>,
    pub after: Option<serde_json::Value>,
}

/// Request to create an anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct NewAnchor {
    pub target_type: AnchorTargetType,
    pub target_id: String,
    pub roles: BTreeSet<AnchorRole>,
    pub by: AnchorActor,
    /// Free-form identity of the actor (user id, agent session, migration name).
    pub actor: String,
    pub inferred: bool,
    pub confidence: f64,
    pub snapshot_name: Option<String>,
    pub snapshot_path: Option<String>,
    pub snapshot_type: Option<String>,
    pub rev: Option<String>,
}

impl NewAnchor {
    /// A plain explicit anchor with the given roles.
    pub fn new(
        target_type: AnchorTargetType,
        target_id: impl Into<String>,
        roles: impl IntoIterator<Item = AnchorRole>,
        by: AnchorActor,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            target_type,
            target_id: target_id.into(),
            roles: roles.into_iter().collect(),
            by,
            actor: actor.into(),
            inferred: false,
            confidence: 1.0,
            snapshot_name: None,
            snapshot_path: None,
            snapshot_type: None,
            rev: None,
        }
    }
}

/// An operation on the anchors of one session.
#[derive(Debug, Clone, PartialEq)]
pub enum AnchorOp {
    Add(NewAnchor),
    Promote {
        anchor_id: Uuid,
        role: AnchorRole,
        expected_version: u64,
        by: AnchorActor,
        actor: String,
    },
    Demote {
        anchor_id: Uuid,
        role: AnchorRole,
        expected_version: u64,
        by: AnchorActor,
        actor: String,
    },
    Remove {
        anchor_id: Uuid,
        expected_version: u64,
        by: AnchorActor,
        actor: String,
    },
    SetState {
        anchor_id: Uuid,
        state: AnchorState,
        /// `None` for system transitions that must not lose to a stale read.
        expected_version: Option<u64>,
        by: AnchorActor,
        actor: String,
    },
}

/// Result of [`apply_op`]: what to persist, in one transaction.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorChange {
    /// The anchor after the operation (the removed one for a removal).
    pub anchor: Anchor,
    /// `None` when the operation changed nothing (idempotent no-op).
    pub event: Option<AnchorEvent>,
    /// The anchor must be deleted (removal).
    pub deleted: bool,
    /// The anchor did not exist before.
    pub created: bool,
}

/// A page of the reverse query.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnchorPage {
    pub items: Vec<Anchor>,
    /// Pass back as `cursor` to continue; `None` on the last page.
    pub next_cursor: Option<String>,
}

/// Counters of a backfill run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackfillReport {
    pub created: usize,
    pub already_present: usize,
    /// Project slug resolving to no project, or to more than one.
    pub skipped_unresolved: usize,
    /// Refused by a rule (e.g. the session already holds 3 focus anchors).
    pub skipped_by_rule: usize,
}

/// Canonical id of a `function` target: the file path and the name, never a line.
pub fn function_target_id(file_path: &str, name: &str) -> String {
    format!("{file_path}::{name}")
}

fn bad(target_type: AnchorTargetType, reason: &str) -> AnchorError {
    AnchorError::InvalidTargetId {
        target_type,
        reason: reason.to_string(),
    }
}

/// True when `s` ends with a line locator: `:12`, `:L12`, `#L12`, `:12:3`.
fn ends_with_line_number(s: &str) -> bool {
    let digits = s.trim_end_matches(|c: char| c.is_ascii_digit());
    if digits.len() == s.len() {
        return false;
    }
    let trimmed = digits.trim_end_matches('L');
    trimmed.ends_with(':') || trimmed.ends_with('#') || trimmed.is_empty()
}

/// Validate a target id for its type and return its canonical form.
///
/// uuid types are normalised to the hyphenated lowercase form. `file` needs a
/// non-empty path. `function` is `<file_path>::<name>` (split at the first
/// `::`); a line number is never part of the identity.
pub fn validate_target_id(target_type: AnchorTargetType, id: &str) -> Result<String, AnchorError> {
    use AnchorTargetType as T;
    if id.chars().count() > MAX_FIELD_CHARS {
        return Err(AnchorError::FieldTooLong {
            field: "target_id",
            max: MAX_FIELD_CHARS,
        });
    }
    match target_type {
        T::Project
        | T::Workspace
        | T::Plan
        | T::Task
        | T::Note
        | T::Decision
        | T::Component
        | T::Feature
        | T::Release => Uuid::parse_str(id.trim())
            .map(|u| u.hyphenated().to_string())
            .map_err(|_| bad(target_type, "expected a uuid")),
        T::File => {
            if id.trim().is_empty() {
                Err(bad(target_type, "empty path"))
            } else if id.contains('\0') || id.contains('\n') {
                Err(bad(target_type, "path contains a control character"))
            } else {
                Ok(id.to_string())
            }
        }
        T::Function => {
            let Some((path, name)) = id.split_once("::") else {
                return Err(bad(target_type, "expected `<file_path>::<name>`"));
            };
            if path.trim().is_empty() || path.contains('\0') || path.contains('\n') {
                return Err(bad(target_type, "empty or invalid file path"));
            }
            if name.trim().is_empty() || name.chars().any(|c| c.is_whitespace() || c == '\0') {
                return Err(bad(target_type, "empty or invalid function name"));
            }
            if ends_with_line_number(name) || ends_with_line_number(path) {
                return Err(bad(
                    target_type,
                    "a line number is not part of a function identity",
                ));
            }
            Ok(id.to_string())
        }
    }
}

fn check_snapshot_field(field: &'static str, v: &Option<String>) -> Result<(), AnchorError> {
    match v {
        Some(s) if s.chars().count() > MAX_FIELD_CHARS => Err(AnchorError::FieldTooLong {
            field,
            max: MAX_FIELD_CHARS,
        }),
        _ => Ok(()),
    }
}

fn check_version(expected: u64, anchor: &Anchor) -> Result<(), AnchorError> {
    if anchor.version == expected {
        Ok(())
    } else {
        Err(AnchorError::VersionConflict {
            expected,
            actual: anchor.version,
        })
    }
}

fn find(existing: &[Anchor], id: Uuid) -> Result<&Anchor, AnchorError> {
    existing
        .iter()
        .find(|a| a.id == id)
        .ok_or(AnchorError::NotFound(id))
}

fn event(
    anchor: &Anchor,
    kind: AnchorEventKind,
    by: AnchorActor,
    actor: &str,
    at: DateTime<Utc>,
    before: Option<&Anchor>,
    after: Option<&Anchor>,
) -> AnchorEvent {
    AnchorEvent {
        id: Uuid::new_v4(),
        session_id: anchor.session_id,
        anchor_id: anchor.id,
        kind,
        by,
        actor: actor.to_string(),
        at,
        before: before.and_then(|a| serde_json::to_value(a).ok()),
        after: after.and_then(|a| serde_json::to_value(a).ok()),
    }
}

fn focus_count(existing: &[Anchor]) -> usize {
    existing
        .iter()
        .filter(|a| a.has_role(AnchorRole::Focus))
        .count()
}

/// Apply one operation to the anchors a session currently holds.
///
/// `existing` must be the complete set of the session's anchors. Pure: nothing
/// is stored; the caller persists the returned [`AnchorChange`] atomically.
pub fn apply_op(
    session_id: Uuid,
    existing: &[Anchor],
    op: AnchorOp,
    now: DateTime<Utc>,
) -> Result<AnchorChange, AnchorError> {
    match op {
        AnchorOp::Add(new) => plan_add(session_id, existing, new, now),
        AnchorOp::Promote {
            anchor_id,
            role,
            expected_version,
            by,
            actor,
        } => {
            let cur = find(existing, anchor_id)?;
            check_version(expected_version, cur)?;
            if role == AnchorRole::Origin || cur.has_role(AnchorRole::Origin) {
                return Err(AnchorError::OriginImmutable);
            }
            if by == AnchorActor::Agent && role == AnchorRole::Focus {
                return Err(AnchorError::AgentCannotFocus);
            }
            if cur.has_role(role) {
                return Ok(noop(cur));
            }
            if role == AnchorRole::Focus && focus_count(existing) >= MAX_FOCUS_ANCHORS {
                return Err(AnchorError::FocusCapReached {
                    max: MAX_FOCUS_ANCHORS,
                });
            }
            let mut next = cur.clone();
            next.roles.insert(role);
            bump(&mut next, now);
            let ev = event(
                &next,
                AnchorEventKind::Promoted,
                by,
                &actor,
                now,
                Some(cur),
                Some(&next),
            );
            Ok(changed(next, ev))
        }
        AnchorOp::Demote {
            anchor_id,
            role,
            expected_version,
            by,
            actor,
        } => {
            let cur = find(existing, anchor_id)?;
            check_version(expected_version, cur)?;
            if role == AnchorRole::Origin || cur.has_role(AnchorRole::Origin) {
                return Err(AnchorError::OriginImmutable);
            }
            if by == AnchorActor::Agent && cur.has_role(AnchorRole::Focus) {
                return Err(AnchorError::AgentRoleForbidden(AnchorRole::Focus));
            }
            if !cur.has_role(role) {
                return Ok(noop(cur));
            }
            if cur.roles.len() == 1 {
                return Err(AnchorError::LastRole);
            }
            let mut next = cur.clone();
            next.roles.remove(&role);
            bump(&mut next, now);
            let ev = event(
                &next,
                AnchorEventKind::Demoted,
                by,
                &actor,
                now,
                Some(cur),
                Some(&next),
            );
            Ok(changed(next, ev))
        }
        AnchorOp::Remove {
            anchor_id,
            expected_version,
            by,
            actor,
        } => {
            let cur = find(existing, anchor_id)?;
            check_version(expected_version, cur)?;
            if cur.has_role(AnchorRole::Origin) {
                return Err(AnchorError::OriginImmutable);
            }
            if by == AnchorActor::Agent && cur.has_role(AnchorRole::Focus) {
                return Err(AnchorError::AgentRoleForbidden(AnchorRole::Focus));
            }
            let ev = event(
                cur,
                AnchorEventKind::Removed,
                by,
                &actor,
                now,
                Some(cur),
                None,
            );
            Ok(AnchorChange {
                anchor: cur.clone(),
                event: Some(ev),
                deleted: true,
                created: false,
            })
        }
        AnchorOp::SetState {
            anchor_id,
            state,
            expected_version,
            by,
            actor,
        } => {
            let cur = find(existing, anchor_id)?;
            if let Some(v) = expected_version {
                check_version(v, cur)?;
            }
            if cur.state == state {
                return Ok(noop(cur));
            }
            let mut next = cur.clone();
            next.state = state;
            bump(&mut next, now);
            let ev = event(
                &next,
                AnchorEventKind::StateChanged,
                by,
                &actor,
                now,
                Some(cur),
                Some(&next),
            );
            Ok(changed(next, ev))
        }
    }
}

fn bump(a: &mut Anchor, now: DateTime<Utc>) {
    a.version += 1;
    a.updated_at = now;
}

fn noop(a: &Anchor) -> AnchorChange {
    AnchorChange {
        anchor: a.clone(),
        event: None,
        deleted: false,
        created: false,
    }
}

fn changed(anchor: Anchor, ev: AnchorEvent) -> AnchorChange {
    AnchorChange {
        anchor,
        event: Some(ev),
        deleted: false,
        created: false,
    }
}

fn plan_add(
    session_id: Uuid,
    existing: &[Anchor],
    new: NewAnchor,
    now: DateTime<Utc>,
) -> Result<AnchorChange, AnchorError> {
    let target_id = validate_target_id(new.target_type, &new.target_id)?;
    if new.roles.is_empty() {
        return Err(AnchorError::EmptyRoles);
    }
    if !new.confidence.is_finite() || !(0.0..=1.0).contains(&new.confidence) {
        return Err(AnchorError::InvalidConfidence);
    }
    check_snapshot_field("snapshot_name", &new.snapshot_name)?;
    check_snapshot_field("snapshot_path", &new.snapshot_path)?;
    check_snapshot_field("snapshot_type", &new.snapshot_type)?;
    check_snapshot_field("rev", &new.rev)?;
    if new.by == AnchorActor::Agent {
        if new.roles.contains(&AnchorRole::Focus) {
            return Err(AnchorError::AgentCannotFocus);
        }
        if let Some(r) = new.roles.iter().find(|r| **r != AnchorRole::Mention) {
            return Err(AnchorError::AgentRoleForbidden(*r));
        }
    }
    if let Some(cur) = existing
        .iter()
        .find(|a| a.target_type == new.target_type && a.target_id == target_id)
    {
        // Re-adding what is already there is idempotent; asking for more is a
        // promotion, with its own checks and its own journal entry.
        return if new.roles.is_subset(&cur.roles) {
            Ok(noop(cur))
        } else {
            Err(AnchorError::AlreadyAnchored { anchor_id: cur.id })
        };
    }
    if existing.len() >= MAX_ANCHORS_PER_SESSION {
        return Err(AnchorError::SessionCapReached {
            max: MAX_ANCHORS_PER_SESSION,
        });
    }
    if new.roles.contains(&AnchorRole::Origin)
        && existing
            .iter()
            .filter(|a| a.has_role(AnchorRole::Origin))
            .count()
            >= MAX_ORIGIN_ANCHORS
    {
        return Err(AnchorError::OriginAlreadyExists);
    }
    if new.roles.contains(&AnchorRole::Focus) && focus_count(existing) >= MAX_FOCUS_ANCHORS {
        return Err(AnchorError::FocusCapReached {
            max: MAX_FOCUS_ANCHORS,
        });
    }
    let anchor = Anchor {
        id: Uuid::new_v4(),
        session_id,
        target_type: new.target_type,
        target_id,
        roles: new.roles,
        state: AnchorState::Live,
        by: new.by,
        inferred: new.inferred,
        confidence: new.confidence,
        snapshot_name: new.snapshot_name,
        snapshot_path: new.snapshot_path,
        snapshot_type: new.snapshot_type,
        rev: new.rev,
        version: 1,
        created_at: now,
        updated_at: now,
    };
    let ev = event(
        &anchor,
        AnchorEventKind::Added,
        new.by,
        &new.actor,
        now,
        None,
        Some(&anchor),
    );
    Ok(AnchorChange {
        anchor,
        event: Some(ev),
        deleted: false,
        created: true,
    })
}

/// The anchor the backfill creates for a session whose `project_slug` resolved
/// to `project`: role `focus`, by system, inferred, confidence 1.0 (the slug is
/// explicit data, not a guess).
pub fn backfill_new_anchor(project: &ProjectNode) -> NewAnchor {
    let mut n = NewAnchor::new(
        AnchorTargetType::Project,
        project.id.to_string(),
        [AnchorRole::Focus],
        AnchorActor::System,
        BACKFILL_ACTOR,
    );
    n.inferred = true;
    n.snapshot_name = Some(project.name.clone());
    n.snapshot_path = project.root_path_opt().map(|p| p.to_string());
    n.snapshot_type = Some("project".to_string());
    n
}

/// True for the anchors `revert_inferred_anchors` may delete: created by the
/// backfill (inferred, by system, added by [`BACKFILL_ACTOR`]) and never
/// modified since (version 1).
pub fn is_revertible_backfill(anchor: &Anchor, added_by: Option<&str>) -> bool {
    anchor.inferred
        && anchor.by == AnchorActor::System
        && anchor.version == 1
        && added_by == Some(BACKFILL_ACTOR)
}

/// Timestamp format used for stored anchors: fixed width, so text order is time order.
pub fn format_ts(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Micros, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use AnchorActor::*;
    use AnchorRole::*;
    use AnchorTargetType as T;

    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    fn uuid_s() -> String {
        Uuid::new_v4().to_string()
    }

    fn add(
        sid: Uuid,
        set: &mut Vec<Anchor>,
        t: T,
        id: &str,
        roles: &[AnchorRole],
        by: AnchorActor,
    ) -> Result<Anchor, AnchorError> {
        let ch = apply_op(
            sid,
            set,
            AnchorOp::Add(NewAnchor::new(t, id, roles.iter().copied(), by, "tester")),
            now(),
        )?;
        if ch.created {
            set.push(ch.anchor.clone());
        }
        Ok(ch.anchor)
    }

    fn settle(set: &mut Vec<Anchor>, ch: &AnchorChange) {
        set.retain(|a| a.id != ch.anchor.id);
        if !ch.deleted {
            set.push(ch.anchor.clone());
        }
    }

    #[test]
    fn target_ids_are_validated_by_type() {
        for t in [
            T::Project,
            T::Workspace,
            T::Plan,
            T::Task,
            T::Note,
            T::Decision,
            T::Component,
            T::Feature,
            T::Release,
        ] {
            assert!(validate_target_id(t, &uuid_s()).is_ok(), "{t}");
            assert!(validate_target_id(t, "not-a-uuid").is_err(), "{t}");
            assert!(validate_target_id(t, "").is_err(), "{t}");
        }
        let upper = Uuid::new_v4().to_string().to_uppercase();
        assert_eq!(
            validate_target_id(T::Plan, &upper).unwrap(),
            upper.to_lowercase()
        );
        assert!(validate_target_id(T::File, "src/lib.rs").is_ok());
        assert!(validate_target_id(T::File, "").is_err());
        assert!(validate_target_id(T::File, "   ").is_err());
        assert!(validate_target_id(T::File, "a\0b").is_err());
        let long = "x".repeat(MAX_FIELD_CHARS + 1);
        assert!(matches!(
            validate_target_id(T::File, &long),
            Err(AnchorError::FieldTooLong { .. })
        ));
    }

    #[test]
    fn a_function_is_a_file_and_a_name_never_a_line() {
        let ok = function_target_id("src/a.rs", "Foo::bar");
        assert_eq!(validate_target_id(T::Function, &ok).unwrap(), ok);
        for bad in [
            "src/a.rs",
            "::name",
            "src/a.rs::",
            "src/a.rs::main:42",
            "src/a.rs::main#L42",
            "src/a.rs::main:L42",
            "src/a.rs::42",
            "src/a.rs:42::main",
            "src/a.rs::two words",
        ] {
            assert!(
                validate_target_id(T::Function, bad).is_err(),
                "{bad} should be refused"
            );
        }
        // digits inside a name are fine
        assert!(validate_target_id(T::Function, "src/a.rs::parse_v2").is_ok());
    }

    #[test]
    fn at_most_three_focus_anchors() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        for i in 0..3 {
            add(sid, &mut set, T::File, &format!("f{i}.rs"), &[Focus], User).unwrap();
        }
        assert_eq!(
            add(sid, &mut set, T::File, "f3.rs", &[Focus], User),
            Err(AnchorError::FocusCapReached { max: 3 })
        );
        // a mention is still fine, and promoting it is refused at the cap
        let m = add(sid, &mut set, T::File, "f3.rs", &[Mention], User).unwrap();
        let r = apply_op(
            sid,
            &set,
            AnchorOp::Promote {
                anchor_id: m.id,
                role: Focus,
                expected_version: 1,
                by: User,
                actor: "u".into(),
            },
            now(),
        );
        assert_eq!(r, Err(AnchorError::FocusCapReached { max: 3 }));
    }

    #[test]
    fn at_most_one_origin_and_it_is_immutable() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        let o = add(sid, &mut set, T::Plan, &uuid_s(), &[Origin], System).unwrap();
        assert_eq!(
            add(sid, &mut set, T::Plan, &uuid_s(), &[Origin], System),
            Err(AnchorError::OriginAlreadyExists)
        );
        let op = |op| apply_op(sid, &set, op, now());
        assert_eq!(
            op(AnchorOp::Remove {
                anchor_id: o.id,
                expected_version: 1,
                by: User,
                actor: "u".into()
            }),
            Err(AnchorError::OriginImmutable)
        );
        assert_eq!(
            op(AnchorOp::Promote {
                anchor_id: o.id,
                role: Focus,
                expected_version: 1,
                by: User,
                actor: "u".into()
            }),
            Err(AnchorError::OriginImmutable)
        );
        assert_eq!(
            op(AnchorOp::Demote {
                anchor_id: o.id,
                role: Origin,
                expected_version: 1,
                by: User,
                actor: "u".into()
            }),
            Err(AnchorError::OriginImmutable)
        );
        // origin cannot be given later to another anchor
        let m = add(sid, &mut set, T::Note, &uuid_s(), &[Mention], User).unwrap();
        assert_eq!(
            apply_op(
                sid,
                &set,
                AnchorOp::Promote {
                    anchor_id: m.id,
                    role: Origin,
                    expected_version: 1,
                    by: User,
                    actor: "u".into()
                },
                now()
            ),
            Err(AnchorError::OriginImmutable)
        );
        // but its state follows its target
        let ch = apply_op(
            sid,
            &set,
            AnchorOp::SetState {
                anchor_id: o.id,
                state: AnchorState::Dangling,
                expected_version: None,
                by: System,
                actor: "s".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(ch.anchor.state, AnchorState::Dangling);
    }

    #[test]
    fn an_agent_can_only_mention() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        assert_eq!(
            add(sid, &mut set, T::File, "a.rs", &[Focus], Agent),
            Err(AnchorError::AgentCannotFocus)
        );
        assert_eq!(
            add(sid, &mut set, T::File, "a.rs", &[Mention, Focus], Agent),
            Err(AnchorError::AgentCannotFocus)
        );
        assert_eq!(
            add(sid, &mut set, T::Plan, &uuid_s(), &[Origin], Agent),
            Err(AnchorError::AgentRoleForbidden(Origin))
        );
        let m = add(sid, &mut set, T::File, "a.rs", &[Mention], Agent).unwrap();
        assert_eq!(
            apply_op(
                sid,
                &set,
                AnchorOp::Promote {
                    anchor_id: m.id,
                    role: Focus,
                    expected_version: 1,
                    by: Agent,
                    actor: "a".into()
                },
                now()
            ),
            Err(AnchorError::AgentCannotFocus)
        );
        // a user promotes it, then the agent cannot take it away
        let ch = apply_op(
            sid,
            &set,
            AnchorOp::Promote {
                anchor_id: m.id,
                role: Focus,
                expected_version: 1,
                by: User,
                actor: "u".into(),
            },
            now(),
        )
        .unwrap();
        settle(&mut set, &ch);
        assert_eq!(
            apply_op(
                sid,
                &set,
                AnchorOp::Remove {
                    anchor_id: m.id,
                    expected_version: 2,
                    by: Agent,
                    actor: "a".into()
                },
                now()
            ),
            Err(AnchorError::AgentRoleForbidden(Focus))
        );
    }

    #[test]
    fn total_cap_per_session() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        for i in 0..MAX_ANCHORS_PER_SESSION {
            add(
                sid,
                &mut set,
                T::File,
                &format!("f{i}.rs"),
                &[Mention],
                User,
            )
            .unwrap();
        }
        assert_eq!(
            add(sid, &mut set, T::File, "one-more.rs", &[Mention], User),
            Err(AnchorError::SessionCapReached {
                max: MAX_ANCHORS_PER_SESSION
            })
        );
    }

    #[test]
    fn invalid_requests_are_typed_errors() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        assert_eq!(
            add(sid, &mut set, T::File, "a.rs", &[], User),
            Err(AnchorError::EmptyRoles)
        );
        let mut n = NewAnchor::new(T::File, "a.rs", [Mention], User, "u");
        n.confidence = 1.5;
        assert_eq!(
            apply_op(sid, &set, AnchorOp::Add(n.clone()), now()),
            Err(AnchorError::InvalidConfidence)
        );
        n.confidence = f64::NAN;
        assert_eq!(
            apply_op(sid, &set, AnchorOp::Add(n), now()),
            Err(AnchorError::InvalidConfidence)
        );
        assert!(matches!(
            apply_op(
                sid,
                &set,
                AnchorOp::Remove {
                    anchor_id: Uuid::new_v4(),
                    expected_version: 1,
                    by: User,
                    actor: "u".into()
                },
                now()
            ),
            Err(AnchorError::NotFound(_))
        ));
    }

    #[test]
    fn re_adding_is_idempotent_and_asking_for_more_is_refused() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        let a = add(sid, &mut set, T::File, "a.rs", &[Mention], User).unwrap();
        let ch = apply_op(
            sid,
            &set,
            AnchorOp::Add(NewAnchor::new(T::File, "a.rs", [Mention], Agent, "a")),
            now(),
        )
        .unwrap();
        assert!(ch.event.is_none() && !ch.created && ch.anchor.id == a.id);
        assert_eq!(
            add(sid, &mut set, T::File, "a.rs", &[Focus], User),
            Err(AnchorError::AlreadyAnchored { anchor_id: a.id })
        );
    }

    #[test]
    fn versions_and_events() {
        let sid = Uuid::new_v4();
        let mut set = vec![];
        let a = add(sid, &mut set, T::File, "a.rs", &[Mention], User).unwrap();
        assert_eq!(a.version, 1);
        let stale = apply_op(
            sid,
            &set,
            AnchorOp::Promote {
                anchor_id: a.id,
                role: Focus,
                expected_version: 7,
                by: User,
                actor: "u".into(),
            },
            now(),
        );
        assert_eq!(
            stale,
            Err(AnchorError::VersionConflict {
                expected: 7,
                actual: 1
            })
        );
        let ch = apply_op(
            sid,
            &set,
            AnchorOp::Promote {
                anchor_id: a.id,
                role: Focus,
                expected_version: 1,
                by: User,
                actor: "u".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(ch.anchor.version, 2);
        let ev = ch.event.clone().unwrap();
        assert_eq!(ev.kind, AnchorEventKind::Promoted);
        assert_eq!(ev.anchor_id, a.id);
        assert!(ev.before.is_some() && ev.after.is_some());
        settle(&mut set, &ch);
        // same role again: no-op, no event, no bump
        let again = apply_op(
            sid,
            &set,
            AnchorOp::Promote {
                anchor_id: a.id,
                role: Focus,
                expected_version: 2,
                by: User,
                actor: "u".into(),
            },
            now(),
        )
        .unwrap();
        assert!(again.event.is_none() && again.anchor.version == 2);
        // demote the focus: leaves mention; demoting the last role is refused
        let d = apply_op(
            sid,
            &set,
            AnchorOp::Demote {
                anchor_id: a.id,
                role: Focus,
                expected_version: 2,
                by: User,
                actor: "u".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(d.event.as_ref().unwrap().kind, AnchorEventKind::Demoted);
        settle(&mut set, &d);
        assert_eq!(
            apply_op(
                sid,
                &set,
                AnchorOp::Demote {
                    anchor_id: a.id,
                    role: Mention,
                    expected_version: 3,
                    by: User,
                    actor: "u".into()
                },
                now()
            ),
            Err(AnchorError::LastRole)
        );
        let rm = apply_op(
            sid,
            &set,
            AnchorOp::Remove {
                anchor_id: a.id,
                expected_version: 3,
                by: User,
                actor: "u".into(),
            },
            now(),
        )
        .unwrap();
        assert!(rm.deleted);
        assert_eq!(rm.event.unwrap().kind, AnchorEventKind::Removed);
    }

    #[test]
    fn string_enums_round_trip() {
        for t in AnchorTargetType::ALL {
            assert_eq!(AnchorTargetType::parse(t.as_str()), Some(*t));
        }
        for s in AnchorState::ALL {
            assert_eq!(AnchorState::parse(s.as_str()), Some(*s));
        }
        assert_eq!(AnchorTargetType::ALL.len(), 11);
        assert_eq!(AnchorTargetType::parse("reflection"), None);
    }
}
