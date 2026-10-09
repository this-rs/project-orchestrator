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
    /// A chat session, found by a fragment of its TITLE.
    Conversation,
    Project,
    /// A project milestone (not a workspace milestone).
    Milestone,
    Release,
    Workspace,
    /// A git commit known to the graph, designated inside its project.
    Commit,
    Protocol,
    Persona,
    Skill,
    /// A source file of a project, designated by its project-relative path.
    File,
    /// An external web address. Inert: see [`super::link`].
    Link,
}

/// What the `id` of a kind looks like. Stable wire names: the frontend reads
/// them from `GET /api/refs/kinds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdFormat {
    /// A hyphenated UUID, never nil.
    Uuid,
    /// `<project uuid>:<git hash>`: 40 or 64 hexadecimal characters.
    ProjectCommit,
    /// `<project uuid>:<relative path>`.
    ProjectPath,
    /// A normalized `http`/`https` address.
    Url,
}

/// Longest project-relative path of a `file` reference, in bytes.
pub const MAX_FILE_PATH_BYTES: usize = 512;

impl RefKind {
    /// Every active kind, in the order the registry and the fixtures use.
    pub const ALL: [RefKind; 16] = super::kinds::all();

    /// The five kinds of the first release. A client that cannot learn the
    /// server's kinds (an older server has no `GET /api/refs/kinds`) offers
    /// these, and a search without `kinds` asks for these.
    pub const HISTORICAL: [RefKind; 5] = super::kinds::historical();

    /// The name used on the wire and in the `#kind:id` token.
    pub fn as_str(self) -> &'static str {
        self.descriptor().name
    }

    /// The table entry of this kind: everything the system knows about it.
    pub fn descriptor(self) -> &'static super::kinds::KindDescriptor {
        &super::kinds::KINDS[self as usize]
    }

    pub fn class(self) -> super::kinds::KindClass {
        self.descriptor().class
    }

    pub fn id_format(self) -> IdFormat {
        self.descriptor().id_format
    }

    /// A kind whose content can be sensitive (the source tree, the history): readable only inside a
    /// project the session is scoped to, and searchable only inside a scope.
    /// See `access` for the rule.
    pub fn is_sensitive(self) -> bool {
        self.descriptor().access == super::kinds::AccessClass::Sensitive
    }
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The id of a reference, in the canonical spelling of its kind: a lower-case
/// hyphenated UUID, `<project>:<hash>`, `<project>:<path>` or a normalized
/// address. Build one with [`canonical_id`] (validated) or from a `Uuid`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct RefId(String);

impl RefId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The UUID when this is the id of a UUID kind.
    pub fn uuid(&self) -> Option<Uuid> {
        parse_ref_id(&self.0)
    }

    /// `(project, rest)` of a `<project>:<rest>` id.
    pub fn project_and_rest(&self) -> Option<(Uuid, &str)> {
        let (head, rest) = self.0.split_at_checked(36)?;
        let rest = rest.strip_prefix(':')?;
        Some((parse_ref_id(head)?, rest))
    }
}

impl From<Uuid> for RefId {
    fn from(id: Uuid) -> Self {
        RefId(id.to_string())
    }
}

impl fmt::Display for RefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq<Uuid> for RefId {
    fn eq(&self, other: &Uuid) -> bool {
        self.uuid().as_ref() == Some(other)
    }
}

/// Any string at most this long: the shape of an id depends on its kind, so
/// it is checked where the kind is known ([`canonical_id`]).
const MAX_RAW_ID_BYTES: usize = 2048;

impl<'de> Deserialize<'de> for RefId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if s.is_empty() || s.len() > MAX_RAW_ID_BYTES {
            return Err(D::Error::custom("not a valid reference id"));
        }
        Ok(RefId(s))
    }
}

/// Is `rest` a git hash (SHA-1 or SHA-256), spelled in hexadecimal?
fn canonical_hash(rest: &str) -> Option<String> {
    let ok = matches!(rest.len(), 40 | 64) && rest.bytes().all(|b| b.is_ascii_hexdigit());
    ok.then(|| rest.to_ascii_lowercase())
}

/// A project-relative path: no root, no `.`/`..`/empty segment, no backslash,
/// no control character, bounded.
fn valid_rel_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= MAX_FILE_PATH_BYTES
        && !p.contains('\\')
        && !p.chars().any(char::is_control)
        && p.split('/').all(|s| !matches!(s, "" | "." | ".."))
}

/// The id of `kind` spelled canonically, or `None` when `s` is not a valid id
/// for that kind. Upper-case hex and UUIDs are accepted and lowered; an
/// address is normalized (so two spellings of one address are one reference).
pub fn canonical_id(kind: RefKind, s: &str) -> Option<RefId> {
    match kind.id_format() {
        IdFormat::Uuid => parse_ref_id(s).map(RefId::from),
        IdFormat::ProjectCommit => {
            let (project, rest) = RefId(s.to_string())
                .project_and_rest()
                .map(|(p, r)| (p, r.to_string()))?;
            Some(RefId(format!("{project}:{}", canonical_hash(&rest)?)))
        }
        IdFormat::ProjectPath => {
            let (project, rest) = RefId(s.to_string())
                .project_and_rest()
                .map(|(p, r)| (p, r.to_string()))?;
            valid_rel_path(&rest).then(|| RefId(format!("{project}:{rest}")))
        }
        IdFormat::Url => super::link::normalize(s).ok().map(RefId),
    }
}

/// A validated reference. It carries no label: the label is written by the
/// server when it resolves the reference, so a client cannot dress one entity
/// up as another. Unknown fields are refused for the same reason.
///
/// Deserialization is as strict as [`super::validate::validate_one`]: an
/// object (never the positional `["plan","uuid"]`), and an id that is valid
/// for the kind (a hyphenated UUID, not nil, for most).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct EntityRef {
    pub kind: RefKind,
    pub id: RefId,
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
        let id = canonical_id(w.kind, &w.id)
            .ok_or_else(|| D::Error::custom("not a valid reference id"))?;
        Ok(EntityRef { kind: w.kind, id })
    }
}

impl EntityRef {
    /// `id` is a `Uuid` (the common case) or an already canonical [`RefId`].
    pub fn new(kind: RefKind, id: impl Into<RefId>) -> Self {
        Self {
            kind,
            id: id.into(),
        }
    }

    /// The composer token for this reference: `#kind:id` for a data kind,
    /// `@kind:id` for an actor kind (the sigil follows from the class).
    pub fn token(&self) -> String {
        format!("{}{}:{}", self.kind.class().sigil(), self.kind, self.id)
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

/// Split a `<sigil>kind:id` token (`#` or `@`) into its sigil and its two raw
/// parts. `None` when the shape is not that of a token (no sigil, no `:`); what
/// the parts mean, and whether the sigil is the one of the kind's class, is
/// for [`super::validate`] to decide.
pub fn split_token(token: &str) -> Option<(char, RawRef)> {
    let sigil = token.chars().next().filter(|c| matches!(c, '#' | '@'))?;
    let body = &token[1..];
    let (kind, id) = body.split_once(':')?;
    Some((
        sigil,
        RawRef {
            kind: kind.to_string(),
            id: id.to_string(),
        },
    ))
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
        let (sigil, raw) = split_token("#plan:a:b").unwrap();
        assert_eq!(sigil, '#');
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

    const P: &str = "00333b5f-2d0a-4467-9c98-155e55d2b7e5";
    const HASH: &str = "fbb4a32c1d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a";

    #[test]
    fn every_kind_has_an_id_format_and_the_sensitive_ones_are_commit_and_file() {
        let sensitive: Vec<_> = RefKind::ALL.iter().filter(|k| k.is_sensitive()).collect();
        assert_eq!(sensitive, [&RefKind::Commit, &RefKind::File]);
        assert_eq!(RefKind::Commit.id_format(), IdFormat::ProjectCommit);
        assert_eq!(RefKind::File.id_format(), IdFormat::ProjectPath);
        assert_eq!(RefKind::Link.id_format(), IdFormat::Url);
        assert_eq!(RefKind::Conversation.id_format(), IdFormat::Uuid);
        assert_eq!(&RefKind::ALL[..5], &RefKind::HISTORICAL[..]);
    }

    #[test]
    fn a_commit_id_is_a_project_and_a_full_hash_lowered() {
        let ok = canonical_id(RefKind::Commit, &format!("{P}:{}", HASH.to_uppercase())).unwrap();
        assert_eq!(ok.as_str(), format!("{P}:{HASH}"));
        let (project, rest) = ok.project_and_rest().unwrap();
        assert_eq!((project.to_string().as_str(), rest), (P, HASH));
        let sha256 = "a".repeat(64);
        assert!(canonical_id(RefKind::Commit, &format!("{P}:{sha256}")).is_some());
        for bad in [
            HASH.to_string(),
            format!("{P}:{}", &HASH[..39]),
            format!("{P}:{}", &HASH[..7]),
            format!("{P}:{HASH}0"),
            format!("{P}:{}g", &HASH[..39]),
            format!("{P}:"),
            format!("{P}{HASH}"),
            format!("not-a-uuid:{HASH}"),
            format!("00000000-0000-0000-0000-000000000000:{HASH}"),
        ] {
            assert!(canonical_id(RefKind::Commit, &bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn a_file_id_is_a_project_and_a_relative_path_that_cannot_climb() {
        let ok = canonical_id(RefKind::File, &format!("{P}:src/refs/mod.rs")).unwrap();
        assert_eq!(ok.project_and_rest().unwrap().1, "src/refs/mod.rs");
        assert!(canonical_id(RefKind::File, &format!("{P}:a b/é.rs")).is_some());
        let too_long = format!("{P}:{}", "a".repeat(MAX_FILE_PATH_BYTES + 1));
        let at_limit = format!("{P}:{}", "a".repeat(MAX_FILE_PATH_BYTES));
        assert!(canonical_id(RefKind::File, &too_long).is_none());
        assert!(canonical_id(RefKind::File, &at_limit).is_some());
        for bad in [
            "",
            "..",
            "../x",
            "a/../b",
            "a/./b",
            ".",
            "/etc/passwd",
            "a//b",
            "a/",
            "a\\b",
            "a\nb",
            "a\u{0}b",
        ] {
            assert!(
                canonical_id(RefKind::File, &format!("{P}:{bad}")).is_none(),
                "{bad:?}"
            );
        }
        assert!(
            canonical_id(RefKind::File, "src/main.rs").is_none(),
            "no project"
        );
    }

    #[test]
    fn a_link_id_is_the_normalized_address_and_nothing_else() {
        let id = canonical_id(RefKind::Link, "HTTPS://Example.com:443/a#x").unwrap();
        assert_eq!(id.as_str(), "https://example.com/a");
        assert_eq!(canonical_id(RefKind::Link, id.as_str()), Some(id));
        assert!(canonical_id(RefKind::Link, "javascript:alert(1)").is_none());
        assert!(canonical_id(RefKind::Link, "http://127.0.0.1/").is_none());
    }

    #[test]
    fn the_wire_object_of_each_new_kind_is_strict_too() {
        for (kind, id) in [
            ("commit", format!("{P}:{HASH}")),
            ("file", format!("{P}:src/lib.rs")),
            ("link", "https://example.com/".to_string()),
            ("project", P.to_string()),
        ] {
            let json = format!(r#"{{"kind":"{kind}","id":"{id}"}}"#);
            let r: EntityRef = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&r).unwrap(), json);
            let extra = format!(r#"{{"kind":"{kind}","id":"{id}","label":"x"}}"#);
            assert!(serde_json::from_str::<EntityRef>(&extra).is_err(), "{kind}");
        }
        for bad in [
            format!(r#"{{"kind":"commit","id":"{HASH}"}}"#),
            format!(r#"{{"kind":"file","id":"{P}:../x"}}"#),
            r#"{"kind":"link","id":"javascript:alert(1)"}"#.to_string(),
            r#"{"kind":"link","id":"https://u:p@example.com/"}"#.to_string(),
            format!(r#"{{"kind":"project","id":"{P}:x"}}"#),
        ] {
            assert!(serde_json::from_str::<EntityRef>(&bad).is_err(), "{bad}");
        }
    }
}
