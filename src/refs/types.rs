//! The wire shapes of a reference.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The kinds of thing a `#` reference can designate. Active kinds only: a kind
/// that is reserved but not yet delivered lives in [`super::registry`], not
/// here, so that no code path can hold a value it must not resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefKind {
    Plan,
    Task,
    Note,
    Decision,
    /// A note of type `rfc`. There is no `EntityType::Rfc`: the kind is backed
    /// by `EntityType::Note` + `NoteType::Rfc` (see the registry).
    Rfc,
}

impl RefKind {
    /// Every active kind, in the order the registry and the fixtures use.
    pub const ALL: [RefKind; 5] = [
        RefKind::Plan,
        RefKind::Task,
        RefKind::Note,
        RefKind::Decision,
        RefKind::Rfc,
    ];

    /// The name used on the wire and in the `#kind:id` token.
    pub fn as_str(self) -> &'static str {
        match self {
            RefKind::Plan => "plan",
            RefKind::Task => "task",
            RefKind::Note => "note",
            RefKind::Decision => "decision",
            RefKind::Rfc => "rfc",
        }
    }
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A validated reference. It carries no label: the label is written by the
/// server when it resolves the reference, so a client cannot dress one entity
/// up as another. Unknown fields are refused for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityRef {
    pub kind: RefKind,
    pub id: Uuid,
}

impl EntityRef {
    pub fn new(kind: RefKind, id: Uuid) -> Self {
        Self { kind, id }
    }

    /// The composer token for this reference: `#kind:id`.
    pub fn token(&self) -> String {
        format!("#{}:{}", self.kind, self.id)
    }
}

/// A reference as it arrives from a client, before any check: both parts are
/// plain strings so that each failure can be named precisely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawRef {
    pub kind: String,
    pub id: String,
}

/// Split a `#kind:id` token into its two raw parts. `None` when the shape is
/// not that of a token (no leading `#`, no `:`); what the parts mean is for
/// [`super::validate`] to decide.
pub fn split_token(token: &str) -> Option<RawRef> {
    let body = token.strip_prefix('#')?;
    let (kind, id) = body.split_once(':')?;
    Some(RawRef {
        kind: kind.to_string(),
        id: id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> Uuid {
        Uuid::parse_str("3adeffc9-c8b0-4e2f-a674-55bfcb293433").unwrap()
    }

    #[test]
    fn kinds_round_trip_through_their_wire_name() {
        for kind in RefKind::ALL {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            assert_eq!(serde_json::from_str::<RefKind>(&json).unwrap(), kind);
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    #[test]
    fn token_has_the_documented_shape() {
        let r = EntityRef::new(RefKind::Rfc, id());
        assert_eq!(r.token(), format!("#rfc:{}", id()));
    }

    #[test]
    fn split_token_cuts_at_the_first_colon() {
        let raw = split_token("#plan:a:b").unwrap();
        assert_eq!((raw.kind.as_str(), raw.id.as_str()), ("plan", "a:b"));
    }

    #[test]
    fn split_token_refuses_what_is_not_a_token() {
        assert!(split_token("plan:abc").is_none());
        assert!(split_token("#plan").is_none());
        assert!(split_token("").is_none());
    }

    #[test]
    fn an_entity_ref_refuses_a_client_label() {
        let json = format!(r#"{{"kind":"plan","id":"{}","label":"Trusted"}}"#, id());
        assert!(serde_json::from_str::<EntityRef>(&json).is_err());
        assert!(serde_json::from_str::<RawRef>(&json).is_err());
    }
}
