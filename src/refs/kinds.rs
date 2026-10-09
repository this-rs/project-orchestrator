//! THE table of reference kinds.
//!
//! One entry per kind says everything the rest of the system needs to know
//! about it, and everything else derives from this table: the wire name, the
//! id format, the registry backing, the access class, the search switch, the
//! `GET /api/refs/kinds` answer, the prompt section that teaches the agent
//! which kinds it may cite, and the resolver that reads it from the store.
//!
//! **Adding a kind is ONE entry here** (and the variant of [`RefKind`], which
//! the compiler asks for) **plus its resolver** (`KindResolver`, the function
//! the entry points at). No other list has to be touched; the tests at the
//! bottom, and the exhaustiveness tests of the registry, fail if one is
//! forgotten.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::registry::Backing;
use super::resolvers::{self, KindResolver};
use super::resolvers_ext;
use super::types::RefKind;
use crate::events::EntityType as E;
use crate::neo4j::GraphStore;
use crate::notes::models::{EntityType as N, NoteType};

/// What a reference to this kind is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KindClass {
    /// A thing to read or discuss: cited with `#`.
    Data,
    /// A thing that can be invoked (a persona, a skill): cited with `@`.
    /// This table only fixes the class; what invoking one does is not here.
    Actor,
}

impl KindClass {
    /// The character that starts the token of this class in the composer.
    pub fn sigil(self) -> &'static str {
        match self {
            KindClass::Data => "#",
            KindClass::Actor => "@",
        }
    }
}

/// What the label of a reference is made of. The server writes every label
/// from this and never from a client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// The entity's title.
    Title,
    /// Its title, or the first line of its text when it has none.
    TitleOrFirstLine,
    /// The first line of its text (it has no title).
    FirstLine,
    Name,
    /// Its title, or its version when it has none.
    TitleOrVersion,
    /// Its path.
    Path,
    /// The host of an address (never fetched).
    Host,
}

/// Who may read an entity of this kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessClass {
    /// Whoever may read the instance, narrowed to the project or workspace of
    /// the session that cites it.
    Shared,
    /// Sensitive: must belong to a project, is read by a session only inside a
    /// project or workspace it is attached to, and is searched only inside a
    /// scope. See `access::OpenInstanceRule`.
    Sensitive,
}

/// How a resolver is built from the store.
pub type Build = fn(Arc<dyn GraphStore>) -> Box<dyn KindResolver>;

/// Everything about one kind.
#[derive(Clone)]
pub struct KindDescriptor {
    pub kind: RefKind,
    /// The wire name, `#name:id`.
    pub name: &'static str,
    pub class: KindClass,
    pub id_format: super::types::IdFormat,
    pub label: LabelSource,
    pub access: AccessClass,
    /// Offered by `GET /api/refs/search`. A kind with no list to search (a
    /// link is pasted) says `false`.
    pub searchable: bool,
    /// Where the app opens it, relative to `/workspace/:slug/`, with `{id}`
    /// (the reference id) or `{slug}` (the slug of `project` in the result).
    /// `None`: the app has no page for it.
    pub open_route: Option<&'static str>,
    /// Opened outside the app (a link).
    pub opens_external: bool,
    /// A key the frontend maps to an icon.
    pub icon: &'static str,
    /// One of the five kinds of the first release: what a client falls back on
    /// when the server cannot say.
    pub historical: bool,
    /// What stores it, for the exhaustiveness proof of the registry.
    pub backing: Backing,
    /// Its resolver.
    pub resolver: Build,
}

use super::types::IdFormat as F;
use AccessClass::{Sensitive, Shared};
use KindClass::{Actor, Data};
use LabelSource as L;

const fn backing(e: Option<E>, n: Option<N>, t: Option<NoteType>) -> Backing {
    Backing {
        events_entity: e,
        notes_entity: n,
        note_type: t,
    }
}

/// The table. Its order is the display order and the discriminant order of
/// [`RefKind`] (pinned by a constant assertion below).
pub const TABLE: [KindDescriptor; 16] = [
    KindDescriptor {
        kind: RefKind::Plan,
        name: "plan",
        class: Data,
        id_format: F::Uuid,
        label: L::Title,
        access: Shared,
        searchable: true,
        open_route: Some("plans/{id}"),
        opens_external: false,
        icon: "clipboard-list",
        historical: true,
        backing: backing(Some(E::Plan), Some(N::Plan), None),
        resolver: resolvers::plan,
    },
    KindDescriptor {
        kind: RefKind::Task,
        name: "task",
        class: Data,
        id_format: F::Uuid,
        label: L::TitleOrFirstLine,
        access: Shared,
        searchable: true,
        open_route: Some("tasks/{id}"),
        opens_external: false,
        icon: "check-square",
        historical: true,
        backing: backing(Some(E::Task), Some(N::Task), None),
        resolver: resolvers::task,
    },
    KindDescriptor {
        kind: RefKind::Note,
        name: "note",
        class: Data,
        id_format: F::Uuid,
        label: L::FirstLine,
        access: Shared,
        searchable: true,
        open_route: Some("notes/{id}"),
        opens_external: false,
        icon: "sticky-note",
        historical: true,
        backing: backing(Some(E::Note), Some(N::Note), None),
        resolver: resolvers::note,
    },
    KindDescriptor {
        kind: RefKind::Decision,
        name: "decision",
        class: Data,
        id_format: F::Uuid,
        label: L::FirstLine,
        access: Shared,
        searchable: true,
        open_route: Some("decisions/{id}"),
        opens_external: false,
        icon: "scale",
        historical: true,
        backing: backing(Some(E::Decision), Some(N::Decision), None),
        resolver: resolvers::decision,
    },
    KindDescriptor {
        // An RFC is a Note of type `rfc`: there is no `EntityType::Rfc`.
        kind: RefKind::Rfc,
        name: "rfc",
        class: Data,
        id_format: F::Uuid,
        label: L::Title,
        access: Shared,
        searchable: true,
        open_route: Some("rfcs/{id}"),
        opens_external: false,
        icon: "file-text",
        historical: true,
        backing: backing(Some(E::Note), Some(N::Note), Some(NoteType::Rfc)),
        resolver: resolvers::rfc,
    },
    KindDescriptor {
        kind: RefKind::Conversation,
        name: "conversation",
        class: Data,
        id_format: F::Uuid,
        label: L::Title,
        access: Shared,
        searchable: true,
        open_route: Some("chat/{id}"),
        opens_external: false,
        icon: "message-square",
        historical: false,
        backing: backing(Some(E::ChatSession), Some(N::ChatSession), None),
        resolver: resolvers_ext::conversation,
    },
    KindDescriptor {
        kind: RefKind::Project,
        name: "project",
        class: Data,
        id_format: F::Uuid,
        label: L::Name,
        access: Shared,
        searchable: true,
        open_route: Some("projects/{slug}"),
        opens_external: false,
        icon: "folder",
        historical: false,
        backing: backing(Some(E::Project), Some(N::Project), None),
        resolver: resolvers_ext::project,
    },
    KindDescriptor {
        kind: RefKind::Milestone,
        name: "milestone",
        class: Data,
        id_format: F::Uuid,
        label: L::Title,
        access: Shared,
        searchable: true,
        open_route: Some("milestones/{id}"),
        opens_external: false,
        icon: "flag",
        historical: false,
        backing: backing(Some(E::Milestone), Some(N::Milestone), None),
        resolver: resolvers_ext::milestone,
    },
    KindDescriptor {
        kind: RefKind::Release,
        name: "release",
        class: Data,
        id_format: F::Uuid,
        label: L::TitleOrVersion,
        access: Shared,
        searchable: true,
        open_route: None,
        opens_external: false,
        icon: "tag",
        historical: false,
        backing: backing(Some(E::Release), Some(N::Release), None),
        resolver: resolvers_ext::release,
    },
    KindDescriptor {
        kind: RefKind::Workspace,
        name: "workspace",
        class: Data,
        id_format: F::Uuid,
        label: L::Name,
        access: Shared,
        searchable: true,
        open_route: Some("overview"),
        opens_external: false,
        icon: "layers",
        historical: false,
        backing: backing(Some(E::Workspace), Some(N::Workspace), None),
        resolver: resolvers_ext::workspace,
    },
    KindDescriptor {
        kind: RefKind::Commit,
        name: "commit",
        class: Data,
        id_format: F::ProjectCommit,
        label: L::FirstLine,
        access: Sensitive,
        searchable: true,
        open_route: None,
        opens_external: false,
        icon: "git-commit",
        historical: false,
        backing: backing(Some(E::Commit), Some(N::Commit), None),
        resolver: resolvers_ext::commit,
    },
    KindDescriptor {
        kind: RefKind::Protocol,
        name: "protocol",
        class: Data,
        id_format: F::Uuid,
        label: L::Name,
        access: Shared,
        searchable: true,
        open_route: Some("protocols/{id}"),
        opens_external: false,
        icon: "workflow",
        historical: false,
        backing: backing(Some(E::Protocol), Some(N::Protocol), None),
        resolver: resolvers_ext::protocol,
    },
    KindDescriptor {
        kind: RefKind::Persona,
        name: "persona",
        class: Actor,
        id_format: F::Uuid,
        label: L::Name,
        access: Shared,
        searchable: true,
        open_route: Some("personas/{id}"),
        opens_external: false,
        icon: "user-round",
        historical: false,
        backing: backing(Some(E::Persona), None, None),
        resolver: resolvers_ext::persona,
    },
    KindDescriptor {
        kind: RefKind::Skill,
        name: "skill",
        class: Actor,
        id_format: F::Uuid,
        label: L::Name,
        access: Shared,
        searchable: true,
        open_route: Some("skills/{id}"),
        opens_external: false,
        icon: "sparkles",
        historical: false,
        backing: backing(Some(E::Skill), Some(N::Skill), None),
        resolver: resolvers_ext::skill,
    },
    KindDescriptor {
        kind: RefKind::File,
        name: "file",
        class: Data,
        id_format: F::ProjectPath,
        label: L::Path,
        access: Sensitive,
        searchable: true,
        open_route: None,
        opens_external: false,
        icon: "file-code",
        historical: false,
        backing: backing(None, Some(N::File), None),
        resolver: resolvers_ext::file,
    },
    KindDescriptor {
        kind: RefKind::Link,
        name: "link",
        class: Data,
        id_format: F::Url,
        label: L::Host,
        access: Shared,
        searchable: false,
        open_route: None,
        opens_external: true,
        icon: "link",
        historical: false,
        backing: backing(None, None, None),
        resolver: resolvers_ext::link,
    },
];

/// The table, as a static the rest of the code borrows from.
pub static KINDS: [KindDescriptor; 16] = TABLE;

// The discriminant of a kind is its place in the table, so that
// `RefKind::descriptor` is an index and cannot miss.
const _: () = {
    let mut i = 0;
    while i < TABLE.len() {
        assert!(TABLE[i].kind as usize == i, "KINDS is out of order");
        i += 1;
    }
};

/// Every kind, in table order.
pub const fn all() -> [RefKind; 16] {
    let mut out = [RefKind::Plan; 16];
    let mut i = 0;
    while i < TABLE.len() {
        out[i] = TABLE[i].kind;
        i += 1;
    }
    out
}

/// The kinds of the first release.
pub const fn historical() -> [RefKind; 5] {
    let mut out = [RefKind::Plan; 5];
    let mut n = 0;
    let mut i = 0;
    while i < TABLE.len() {
        if TABLE[i].historical {
            out[n] = TABLE[i].kind;
            n += 1;
        }
        i += 1;
    }
    assert!(n == 5, "exactly five historical kinds");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Variant names of `pub enum RefKind` read from `types.rs`.
    fn variants() -> Vec<String> {
        let source = include_str!("types.rs");
        let header = "pub enum RefKind {";
        let start = source.find(header).unwrap() + header.len();
        let body = &source[start..];
        let end = body.find("\n}").unwrap();
        body[..end]
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && !l.starts_with('#'))
            .filter_map(|l| l.strip_suffix(','))
            .map(str::to_string)
            .collect()
    }

    fn snake(camel: &str) -> String {
        let mut out = String::new();
        for (i, c) in camel.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        }
        out
    }

    #[test]
    fn every_variant_has_one_entry_and_every_entry_a_variant() {
        let names: Vec<String> = variants().iter().map(|v| snake(v)).collect();
        let table: Vec<String> = TABLE.iter().map(|d| d.name.to_string()).collect();
        assert_eq!(names, table, "RefKind and the table list the same kinds");
    }

    #[test]
    fn the_name_of_an_entry_is_its_serde_name() {
        for d in &TABLE {
            assert_eq!(serde_json::to_value(d.kind).unwrap(), d.name, "{}", d.name);
        }
    }

    #[test]
    fn names_are_unique_lowercase_words() {
        let mut seen = std::collections::HashSet::new();
        for d in &TABLE {
            assert!(seen.insert(d.name), "{}", d.name);
            assert!(d.name.chars().all(|c| c.is_ascii_lowercase()), "{}", d.name);
            assert!(!d.icon.is_empty());
        }
    }

    #[test]
    fn actors_are_exactly_persona_and_skill() {
        let actors: Vec<_> = TABLE
            .iter()
            .filter(|d| d.class == KindClass::Actor)
            .map(|d| d.name)
            .collect();
        assert_eq!(actors, ["persona", "skill"]);
        assert_eq!(KindClass::Data.sigil(), "#");
        assert_eq!(KindClass::Actor.sigil(), "@");
    }

    #[test]
    fn a_kind_opens_somewhere_or_nowhere_never_both() {
        for d in &TABLE {
            assert!(!(d.opens_external && d.open_route.is_some()), "{}", d.name);
            if let Some(route) = d.open_route {
                assert!(
                    !route.starts_with('/'),
                    "{}: relative to the workspace",
                    d.name
                );
                assert!(route.contains("{id}") || route.contains("{slug}") || route == "overview");
            }
        }
    }

    #[test]
    fn the_first_five_are_the_historical_ones_in_order() {
        assert_eq!(
            historical(),
            [
                RefKind::Plan,
                RefKind::Task,
                RefKind::Note,
                RefKind::Decision,
                RefKind::Rfc
            ]
        );
        assert_eq!(&all()[..5], &historical()[..]);
    }
}
