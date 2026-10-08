//! Pure validation of what a client sends. No I/O, no clock, no graph: the
//! same input always gives the same answer, which is what lets the tests pin
//! every branch.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::registry::{lookup, Lookup};
use super::types::{split_token, EntityRef, RawRef};

/// Most references one message may carry.
pub const MAX_REFS_PER_MESSAGE: usize = 20;
/// Longest search text accepted by `GET /api/refs/search`.
pub const MAX_QUERY_CHARS: usize = 200;
/// Search page size: default and ceiling.
pub const DEFAULT_SEARCH_LIMIT: usize = 20;
pub const MAX_SEARCH_LIMIT: usize = 50;

/// Why a request was refused. Stable wire strings: the frontend switches on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidReason {
    /// More than [`MAX_REFS_PER_MESSAGE`] references.
    TooMany,
    /// The kind is not one the registry knows.
    UnknownKind,
    /// The kind is declared but switched off (tier B).
    KindDisabled,
    /// The id is not a UUID, or is the nil UUID.
    BadId,
    /// The `#kind:id` token is malformed.
    BadToken,
    /// The search text is longer than [`MAX_QUERY_CHARS`].
    QueryTooLong,
    /// The search limit is 0 or above [`MAX_SEARCH_LIMIT`].
    BadLimit,
}

impl InvalidReason {
    pub fn message(self) -> &'static str {
        match self {
            InvalidReason::TooMany => "too many references in one message",
            InvalidReason::UnknownKind => "unknown reference kind",
            InvalidReason::KindDisabled => "this reference kind is not available yet",
            InvalidReason::BadId => "reference id is not a valid UUID",
            InvalidReason::BadToken => "malformed #kind:id token",
            InvalidReason::QueryTooLong => "search text is too long",
            InvalidReason::BadLimit => "search limit is out of range",
        }
    }
}

/// A refusal: the reason and, for a list, the index of the offending element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefsInvalid {
    pub reason: InvalidReason,
    pub index: Option<usize>,
}

/// Validate one raw reference.
pub fn validate_one(raw: &RawRef) -> Result<EntityRef, InvalidReason> {
    let kind = match lookup(&raw.kind) {
        Lookup::Active(kind) => kind,
        Lookup::Reserved(_) => return Err(InvalidReason::KindDisabled),
        Lookup::Unknown => return Err(InvalidReason::UnknownKind),
    };
    match Uuid::parse_str(&raw.id) {
        Ok(id) if !id.is_nil() => Ok(EntityRef::new(kind, id)),
        _ => Err(InvalidReason::BadId),
    }
}

/// Validate the `refs[]` of a message: at most [`MAX_REFS_PER_MESSAGE`], every
/// one valid; duplicates are collapsed, first occurrence wins, order kept.
pub fn validate_refs(raw: &[RawRef]) -> Result<Vec<EntityRef>, RefsInvalid> {
    if raw.len() > MAX_REFS_PER_MESSAGE {
        return Err(RefsInvalid {
            reason: InvalidReason::TooMany,
            index: None,
        });
    }
    let mut out: Vec<EntityRef> = Vec::with_capacity(raw.len());
    for (index, r) in raw.iter().enumerate() {
        let valid = validate_one(r).map_err(|reason| RefsInvalid {
            reason,
            index: Some(index),
        })?;
        if !out.contains(&valid) {
            out.push(valid);
        }
    }
    Ok(out)
}

/// Validate a `#kind:id` token.
pub fn validate_token(token: &str) -> Result<EntityRef, InvalidReason> {
    let raw = split_token(token).ok_or(InvalidReason::BadToken)?;
    validate_one(&raw)
}

/// Validate the search text and limit; returns the effective limit.
pub fn validate_search(q: &str, limit: Option<usize>) -> Result<usize, InvalidReason> {
    if q.chars().count() > MAX_QUERY_CHARS {
        return Err(InvalidReason::QueryTooLong);
    }
    match limit {
        None => Ok(DEFAULT_SEARCH_LIMIT),
        Some(n) if (1..=MAX_SEARCH_LIMIT).contains(&n) => Ok(n),
        Some(_) => Err(InvalidReason::BadLimit),
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::RefKind;
    use super::*;

    const ID_A: &str = "3adeffc9-c8b0-4e2f-a674-55bfcb293433";
    const ID_B: &str = "57cf05c9-25b6-495d-ab07-de4b11d64736";

    fn raw(kind: &str, id: &str) -> RawRef {
        RawRef {
            kind: kind.into(),
            id: id.into(),
        }
    }

    #[test]
    fn every_active_kind_is_accepted() {
        for kind in RefKind::ALL {
            let r = validate_one(&raw(kind.as_str(), ID_A)).unwrap();
            assert_eq!(r.kind, kind);
            assert_eq!(r.id.to_string(), ID_A);
        }
    }

    #[test]
    fn excluded_kinds_are_refused_by_name() {
        for kind in [
            "workspace",
            "project",
            "step",
            "file",
            "milestone",
            "Plan",
            "",
        ] {
            assert_eq!(
                validate_one(&raw(kind, ID_A)),
                Err(InvalidReason::UnknownKind),
                "{kind}"
            );
        }
    }

    #[test]
    fn tier_b_kinds_are_known_but_disabled() {
        for kind in ["persona", "skill"] {
            assert_eq!(
                validate_one(&raw(kind, ID_A)),
                Err(InvalidReason::KindDisabled),
                "{kind}"
            );
        }
    }

    #[test]
    fn bad_ids_are_refused() {
        for id in [
            "",
            "not-a-uuid",
            "00000000-0000-0000-0000-000000000000",
            "3adeffc9",
        ] {
            assert_eq!(
                validate_one(&raw("plan", id)),
                Err(InvalidReason::BadId),
                "{id}"
            );
        }
    }

    #[test]
    fn the_kind_is_checked_before_the_id() {
        assert_eq!(
            validate_one(&raw("nope", "nope")),
            Err(InvalidReason::UnknownKind)
        );
    }

    #[test]
    fn a_list_keeps_order_and_collapses_duplicates() {
        let out = validate_refs(&[
            raw("task", ID_A),
            raw("plan", ID_B),
            raw("task", ID_A),
            raw("note", ID_A),
        ])
        .unwrap();
        let kinds: Vec<_> = out.iter().map(|r| r.kind).collect();
        assert_eq!(kinds, [RefKind::Task, RefKind::Plan, RefKind::Note]);
    }

    #[test]
    fn an_empty_list_is_valid() {
        assert_eq!(validate_refs(&[]).unwrap(), vec![]);
    }

    #[test]
    fn the_limit_is_inclusive_and_counts_before_dedup() {
        let at_limit: Vec<_> = (0..MAX_REFS_PER_MESSAGE)
            .map(|_| raw("plan", ID_A))
            .collect();
        assert_eq!(validate_refs(&at_limit).unwrap().len(), 1);
        let over: Vec<_> = (0..=MAX_REFS_PER_MESSAGE)
            .map(|_| raw("plan", ID_A))
            .collect();
        assert_eq!(
            validate_refs(&over),
            Err(RefsInvalid {
                reason: InvalidReason::TooMany,
                index: None
            })
        );
    }

    #[test]
    fn the_error_names_the_offending_index() {
        let err = validate_refs(&[raw("plan", ID_A), raw("plan", ID_B), raw("workspace", ID_A)])
            .unwrap_err();
        assert_eq!(err.reason, InvalidReason::UnknownKind);
        assert_eq!(err.index, Some(2));
    }

    #[test]
    fn tokens_are_validated_end_to_end() {
        assert_eq!(
            validate_token(&format!("#rfc:{ID_A}")).unwrap().kind,
            RefKind::Rfc
        );
        assert_eq!(validate_token("rfc:abc"), Err(InvalidReason::BadToken));
        assert_eq!(validate_token("#rfc"), Err(InvalidReason::BadToken));
        assert_eq!(
            validate_token(&format!("#persona:{ID_A}")),
            Err(InvalidReason::KindDisabled)
        );
        assert_eq!(validate_token("#plan:zzz"), Err(InvalidReason::BadId));
    }

    #[test]
    fn search_bounds() {
        assert_eq!(validate_search("", None), Ok(DEFAULT_SEARCH_LIMIT));
        assert_eq!(validate_search("abc", Some(1)), Ok(1));
        assert_eq!(
            validate_search("abc", Some(MAX_SEARCH_LIMIT)),
            Ok(MAX_SEARCH_LIMIT)
        );
        assert_eq!(
            validate_search("abc", Some(0)),
            Err(InvalidReason::BadLimit)
        );
        assert_eq!(
            validate_search("abc", Some(MAX_SEARCH_LIMIT + 1)),
            Err(InvalidReason::BadLimit)
        );
        let long = "é".repeat(MAX_QUERY_CHARS);
        assert_eq!(validate_search(&long, None), Ok(DEFAULT_SEARCH_LIMIT));
        let too_long = "é".repeat(MAX_QUERY_CHARS + 1);
        assert_eq!(
            validate_search(&too_long, None),
            Err(InvalidReason::QueryTooLong)
        );
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn arbitrary_raw_refs_never_panic_and_stay_bounded(
                items in proptest::collection::vec((any::<String>(), any::<String>()), 0..40)
            ) {
                let raw: Vec<RawRef> = items.into_iter().map(|(kind, id)| RawRef { kind, id }).collect();
                if let Ok(valid) = validate_refs(&raw) {
                    prop_assert!(valid.len() <= MAX_REFS_PER_MESSAGE);
                    for r in &valid {
                        prop_assert!(!r.id.is_nil());
                    }
                }
            }

            #[test]
            fn arbitrary_tokens_and_searches_never_panic(s in any::<String>(), n in any::<Option<usize>>()) {
                let _ = validate_token(&s);
                let _ = validate_search(&s, n);
            }

            #[test]
            fn a_valid_token_always_round_trips(kind in 0usize..5, bytes in any::<[u8; 16]>()) {
                let id = Uuid::from_bytes(bytes);
                prop_assume!(!id.is_nil());
                let r = EntityRef::new(RefKind::ALL[kind], id);
                prop_assert_eq!(validate_token(&r.token()), Ok(r));
            }
        }
    }

    #[test]
    fn every_reason_has_a_message() {
        for r in [
            InvalidReason::TooMany,
            InvalidReason::UnknownKind,
            InvalidReason::KindDisabled,
            InvalidReason::BadId,
            InvalidReason::BadToken,
            InvalidReason::QueryTooLong,
            InvalidReason::BadLimit,
        ] {
            assert!(!r.message().is_empty());
        }
    }
}
