//! Pure validation of what a client sends. No I/O, no clock, no graph: the
//! same input always gives the same answer, which is what lets the tests pin
//! every branch.

use serde::{Deserialize, Serialize};

use super::registry::{lookup, Lookup};
use super::types::{canonical_id, split_token, EntityRef, RawRef, RefKind};

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
    /// The id is not valid for its kind: not a UUID (or the nil one), not
    /// `<project>:<hash>`, not `<project>:<path>`.
    BadId,
    /// A `link` that is not allowed: not http/https, credentials in it, too
    /// long, or aimed at a private address. See `refs::link`.
    BadLink,
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
            InvalidReason::BadLink => {
                "this link is not allowed (http or https only, no credentials, no private address)"
            }
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
    match canonical_id(kind, &raw.id) {
        Some(id) => Ok(EntityRef::new(kind, id)),
        None if kind == RefKind::Link => Err(InvalidReason::BadLink),
        None => Err(InvalidReason::BadId),
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

/// Validate a `#kind:id` or `@kind:id` token. The sigil is carried by the token
/// and follows from the class of the kind: `#` for data, `@` for an actor, and
/// only that: the other one is `bad_token`.
pub fn validate_token(token: &str) -> Result<EntityRef, InvalidReason> {
    let (sigil, raw) = split_token(token).ok_or(InvalidReason::BadToken)?;
    if let Lookup::Active(kind) = lookup(&raw.kind) {
        if !kind.class().sigil().starts_with(sigil) {
            return Err(InvalidReason::BadToken);
        }
    }
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
    use super::super::types::{IdFormat, RefKind};
    use super::*;

    const ID_A: &str = "3adeffc9-c8b0-4e2f-a674-55bfcb293433";
    const ID_B: &str = "57cf05c9-25b6-495d-ab07-de4b11d64736";

    fn raw(kind: &str, id: &str) -> RawRef {
        RawRef {
            kind: kind.into(),
            id: id.into(),
        }
    }

    fn a_valid_id_for(kind: RefKind) -> String {
        match kind.id_format() {
            IdFormat::Uuid => ID_A.to_string(),
            IdFormat::ProjectCommit => format!("{ID_B}:{}", "ab12".repeat(10)),
            IdFormat::ProjectPath => format!("{ID_B}:src/refs/mod.rs"),
            IdFormat::Url => "https://example.com/a".to_string(),
        }
    }

    #[test]
    fn every_active_kind_is_accepted_with_an_id_of_its_format() {
        for kind in RefKind::ALL {
            let id = a_valid_id_for(kind);
            let r = validate_one(&raw(kind.as_str(), &id)).unwrap();
            assert_eq!(r.kind, kind);
            assert_eq!(r.id.as_str(), id);
        }
    }

    #[test]
    fn excluded_kinds_are_refused_by_name() {
        for kind in ["step", "constraint", "document", "Plan", "Workspace", ""] {
            assert_eq!(
                validate_one(&raw(kind, ID_A)),
                Err(InvalidReason::UnknownKind),
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
        let err =
            validate_refs(&[raw("plan", ID_A), raw("plan", ID_B), raw("step", ID_A)]).unwrap_err();
        assert_eq!(err.reason, InvalidReason::UnknownKind);
        assert_eq!(err.index, Some(2));
    }

    #[test]
    fn the_sigil_follows_the_class_of_the_kind_in_both_directions() {
        use super::super::kinds::KindClass;
        for kind in RefKind::ALL {
            let id = a_valid_id_for(kind);
            let (right, wrong) = match kind.class() {
                KindClass::Data => ('#', '@'),
                KindClass::Actor => ('@', '#'),
            };
            let ok = validate_token(&format!("{right}{kind}:{id}")).unwrap();
            assert_eq!(ok.kind, kind);
            assert_eq!(ok.token(), format!("{right}{kind}:{}", ok.id));
            assert_eq!(
                validate_token(&format!("{wrong}{kind}:{id}")),
                Err(InvalidReason::BadToken),
                "{wrong}{kind}"
            );
        }
        assert_eq!(
            validate_token(&format!("#persona:{ID_A}")),
            Err(InvalidReason::BadToken)
        );
        assert_eq!(
            validate_token(&format!("@plan:{ID_A}")),
            Err(InvalidReason::BadToken)
        );
        assert_eq!(
            validate_token(&format!("@nope:{ID_A}")),
            Err(InvalidReason::UnknownKind)
        );
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
            validate_token(&format!("#step:{ID_A}")),
            Err(InvalidReason::UnknownKind)
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
                        prop_assert!(r.id.uuid().is_some_and(|u| !u.is_nil()));
                    }
                }
            }

            #[test]
            fn arbitrary_tokens_and_searches_never_panic(s in any::<String>(), n in any::<Option<usize>>()) {
                let _ = validate_token(&s);
                let _ = validate_search(&s, n);
            }

            #[test]
            fn a_valid_token_always_round_trips(kind in 0usize..10, bytes in any::<[u8; 16]>()) {
                let id = uuid::Uuid::from_bytes(bytes);
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
            InvalidReason::BadLink,
            InvalidReason::BadToken,
            InvalidReason::QueryTooLong,
            InvalidReason::BadLimit,
        ] {
            assert!(!r.message().is_empty());
        }
    }

    #[test]
    fn only_the_hyphenated_uuid_is_an_id() {
        for bad in [
            "3adeffc9c8b04e2fa67455bfcb293433",
            "{3adeffc9-c8b0-4e2f-a674-55bfcb293433}",
            "urn:uuid:3adeffc9-c8b0-4e2f-a674-55bfcb293433",
            " 3adeffc9-c8b0-4e2f-a674-55bfcb293433",
            "3adeffc9-c8b0-4e2f-a674-55bfcb29343",
            "３adeffc9-c8b0-4e2f-a674-55bfcb293433",
        ] {
            assert_eq!(
                validate_one(&raw("plan", bad)),
                Err(InvalidReason::BadId),
                "{bad}"
            );
            assert_eq!(
                validate_token(&format!("#plan:{bad}")),
                Err(InvalidReason::BadId),
                "{bad}"
            );
        }
    }

    #[test]
    fn an_upper_case_id_is_accepted_and_lowered() {
        let r = validate_one(&raw("plan", &ID_A.to_uppercase())).unwrap();
        assert_eq!(r.id.to_string(), ID_A);
    }
}
