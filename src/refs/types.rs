//! The wire shapes of a reference.

use std::fmt;
use std::marker::PhantomData;

use serde::de::value::MapAccessDeserializer;
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
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
///
/// Deserialization is as strict as [`super::validate::validate_one`]: an
/// object (never the positional `["plan","uuid"]`), a hyphenated UUID, not nil.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct EntityRef {
    pub kind: RefKind,
    pub id: Uuid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntityRefWire {
    kind: RefKind,
    id: String,
}

impl<'de> Deserialize<'de> for EntityRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let MapOnly(w) = MapOnly::<EntityRefWire>::deserialize(d)?;
        let id = parse_ref_id(&w.id).ok_or_else(|| D::Error::custom("not a valid reference id"))?;
        Ok(EntityRef { kind: w.kind, id })
    }
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
/// plain strings so that each failure can be named precisely. Only its SHAPE is
/// enforced on the way in: an object, with exactly these two fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RawRef {
    pub kind: String,
    pub id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRefWire {
    kind: String,
    id: String,
}

impl<'de> Deserialize<'de> for RawRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let MapOnly(w) = MapOnly::<RawRefWire>::deserialize(d)?;
        Ok(RawRef {
            kind: w.kind,
            id: w.id,
        })
    }
}

/// The id of a reference: a HYPHENATED UUID of 36 characters, never nil.
/// Upper case is accepted (the frontend's token regex accepts it); the result
/// is a `Uuid`, which prints in lower case. The simple (no hyphens), braced and
/// `urn:uuid:` spellings that `Uuid::parse_str` also takes are refused.
pub fn parse_ref_id(s: &str) -> Option<Uuid> {
    if s.len() != 36 {
        return None;
    }
    Uuid::parse_str(s).ok().filter(|id| !id.is_nil())
}

/// `T` read from a map only. A derived struct deserializer also accepts a
/// sequence (`["plan","uuid"]`); this refuses it.
pub(crate) struct MapOnly<T>(pub T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for MapOnly<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for V<T> {
            type Value = MapOnly<T>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                T::deserialize(MapAccessDeserializer::new(map)).map(MapOnly)
            }
        }
        d.deserialize_map(V(PhantomData))
    }
}

/// `deserialize_with` for a bare `Uuid` field that must obey [`parse_ref_id`].
pub(crate) fn de_ref_id<'de, D: Deserializer<'de>>(d: D) -> Result<Uuid, D::Error> {
    let s = String::deserialize(d)?;
    parse_ref_id(&s).ok_or_else(|| D::Error::custom("not a valid reference id"))
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

    #[test]
    fn the_wire_shape_is_an_object_with_a_real_hyphenated_id() {
        let nil = "00000000-0000-0000-0000-000000000000";
        for bad in [
            format!(r#"["plan","{}"]"#, id()),
            format!(r#"{{"kind":"plan","id":"{nil}"}}"#),
            format!(r#"{{"kind":"plan","id":"{}"}}"#, id().simple()),
            format!(r#"{{"kind":"plan","id":"{}"}}"#, id().braced()),
            format!(r#"{{"kind":"plan","id":"{}"}}"#, id().urn()),
        ] {
            assert!(serde_json::from_str::<EntityRef>(&bad).is_err(), "{bad}");
        }
        assert!(serde_json::from_str::<RawRef>(&format!(r#"["plan","{}"]"#, id())).is_err());
        // an upper-case id is accepted and comes out lower case.
        let up = format!(
            r#"{{"kind":"plan","id":"{}"}}"#,
            id().to_string().to_uppercase()
        );
        let r: EntityRef = serde_json::from_str(&up).unwrap();
        assert_eq!(r.id, id());
        // the serialized form is unchanged.
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            format!(r#"{{"kind":"plan","id":"{}"}}"#, id())
        );
        // a duplicate or unknown field is refused, and a RawRef still takes strings as they are.
        assert!(serde_json::from_str::<RawRef>(r#"{"kind":"a","kind":"b","id":"c"}"#).is_err());
        let raw: RawRef = serde_json::from_str(r#"{"kind":"zzz","id":"nope"}"#).unwrap();
        assert_eq!((raw.kind.as_str(), raw.id.as_str()), ("zzz", "nope"));
    }
}
