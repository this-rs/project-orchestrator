//! The registry of reference kinds.
//!
//! One place says which kinds exist, what backs each in the graph, and which
//! are switched on. The exhaustiveness tests at the bottom tie it to the two
//! `EntityType` enums of the code base: a new variant there makes this file
//! stop compiling (the `classify_*` matches have no wildcard) *and* fail a test
//! that names the variant, so nobody can add an entity type without deciding
//! whether it can be referenced.

use crate::events::EntityType as EventsEntity;
use crate::notes::models::{EntityType as NotesEntity, NoteType};

use super::types::RefKind;

/// How far a kind has been delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Delivered by the MVP (`#`).
    A,
    /// Declared and reserved, not delivered (`@`, phase 2).
    B,
}

/// What stores an active kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backing {
    pub events_entity: EventsEntity,
    pub notes_entity: NotesEntity,
    /// `Some` when the kind is a sub-type of the entity (an RFC is a `Note`).
    pub note_type: Option<NoteType>,
}

/// An active kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindSpec {
    pub kind: RefKind,
    pub tier: Tier,
    pub enabled: bool,
    pub backing: Backing,
}

/// A kind that is declared but must not be resolved yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedSpec {
    pub name: &'static str,
    pub tier: Tier,
    pub enabled: bool,
}

/// Tier-B kinds: the names are taken, the switch is off.
pub const RESERVED: [ReservedSpec; 2] = [
    ReservedSpec {
        name: "persona",
        tier: Tier::B,
        enabled: false,
    },
    ReservedSpec {
        name: "skill",
        tier: Tier::B,
        enabled: false,
    },
];

/// The registry entry of an active kind.
pub fn spec(kind: RefKind) -> KindSpec {
    let (events_entity, notes_entity, note_type) = match kind {
        RefKind::Plan => (EventsEntity::Plan, NotesEntity::Plan, None),
        RefKind::Task => (EventsEntity::Task, NotesEntity::Task, None),
        RefKind::Note => (EventsEntity::Note, NotesEntity::Note, None),
        RefKind::Decision => (EventsEntity::Decision, NotesEntity::Decision, None),
        RefKind::Rfc => (EventsEntity::Note, NotesEntity::Note, Some(NoteType::Rfc)),
    };
    KindSpec {
        kind,
        tier: Tier::A,
        enabled: true,
        backing: Backing {
            events_entity,
            notes_entity,
            note_type,
        },
    }
}

/// What a name sent by a client means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    Active(RefKind),
    /// Declared, switched off.
    Reserved(ReservedSpec),
    Unknown,
}

pub fn lookup(name: &str) -> Lookup {
    if let Some(kind) = RefKind::ALL
        .into_iter()
        .find(|k| k.as_str() == name && spec(*k).enabled)
    {
        return Lookup::Active(kind);
    }
    match RESERVED.into_iter().find(|r| r.name == name) {
        Some(r) => Lookup::Reserved(r),
        None => Lookup::Unknown,
    }
}

/// The decision taken for one `EntityType` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// Backs this active kind.
    Kind(RefKind),
    /// Backs a reserved (tier B) kind.
    Reserved(&'static str),
    /// Deliberately not referenceable, with the reason.
    Excluded(&'static str),
}

const NOT_A_PAGE: &str = "no page of its own to point at (owned by a parent entity)";
const CODE_INTEL: &str = "code-intelligence node, not a unit of work";
const INFRA: &str = "system or runtime record, not a unit of work";
const LATER: &str = "work entity, not scheduled for references yet";

/// Exhaustive on purpose: no `_` arm.
pub fn classify_events(e: &EventsEntity) -> Classification {
    use Classification::{Excluded, Kind, Reserved};
    match e {
        EventsEntity::Plan => Kind(RefKind::Plan),
        EventsEntity::Task => Kind(RefKind::Task),
        EventsEntity::Decision => Kind(RefKind::Decision),
        // An RFC is a Note of type `rfc`: the same entity backs two kinds, the
        // plain one is the default reading.
        EventsEntity::Note => Kind(RefKind::Note),
        EventsEntity::Persona => Reserved("persona"),
        EventsEntity::Skill => Reserved("skill"),
        EventsEntity::Step => Excluded(NOT_A_PAGE),
        EventsEntity::Project
        | EventsEntity::Constraint
        | EventsEntity::Commit
        | EventsEntity::Release
        | EventsEntity::Milestone
        | EventsEntity::Environment
        | EventsEntity::Deployment
        | EventsEntity::Workspace
        | EventsEntity::WorkspaceMilestone
        | EventsEntity::Resource
        | EventsEntity::Component
        | EventsEntity::ChatSession
        | EventsEntity::ProtocolRun
        | EventsEntity::Protocol
        | EventsEntity::Episode => Excluded(LATER),
        EventsEntity::FeatureGraph | EventsEntity::TopologyRule => Excluded(CODE_INTEL),
        EventsEntity::Runner
        | EventsEntity::Alert
        | EventsEntity::AnalysisProfile
        | EventsEntity::Trigger
        | EventsEntity::LifecycleHook
        | EventsEntity::Learning
        | EventsEntity::AttentionChanged => Excluded(INFRA),
    }
}

/// Exhaustive on purpose: no `_` arm.
pub fn classify_notes(e: &NotesEntity) -> Classification {
    use Classification::{Excluded, Kind, Reserved};
    match e {
        NotesEntity::Plan => Kind(RefKind::Plan),
        NotesEntity::Task => Kind(RefKind::Task),
        NotesEntity::Decision => Kind(RefKind::Decision),
        NotesEntity::Note => Kind(RefKind::Note),
        NotesEntity::Skill => Reserved("skill"),
        NotesEntity::Step => Excluded(NOT_A_PAGE),
        NotesEntity::File
        | NotesEntity::Module
        | NotesEntity::Function
        | NotesEntity::Struct
        | NotesEntity::Trait
        | NotesEntity::Enum
        | NotesEntity::Impl => Excluded(CODE_INTEL),
        NotesEntity::Process => Excluded(INFRA),
        NotesEntity::Project
        | NotesEntity::Commit
        | NotesEntity::Constraint
        | NotesEntity::Milestone
        | NotesEntity::Release
        | NotesEntity::Workspace
        | NotesEntity::WorkspaceMilestone
        | NotesEntity::Resource
        | NotesEntity::Component
        | NotesEntity::FeatureGraph
        | NotesEntity::Protocol
        | NotesEntity::ProtocolState
        | NotesEntity::ProtocolRun
        | NotesEntity::PlanRun
        | NotesEntity::ChatSession
        | NotesEntity::Document => Excluded(LATER),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Variant names of `pub enum <name>` read from the source text, so that
    /// the lists below cannot silently fall behind the enums.
    fn variants_in(source: &str, enum_name: &str) -> Vec<String> {
        let header = format!("pub enum {enum_name} {{");
        let start = source.find(&header).expect("enum not found") + header.len();
        let body = &source[start..];
        let end = body.find("\n}").expect("enum not closed");
        body[..end]
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && !l.starts_with('#'))
            .filter_map(|l| l.strip_suffix(','))
            .filter(|n| n.chars().all(|c| c.is_ascii_alphanumeric()) && !n.is_empty())
            .map(str::to_string)
            .collect()
    }

    const EVENTS_ALL: [EventsEntity; 31] = [
        EventsEntity::Project,
        EventsEntity::Plan,
        EventsEntity::Task,
        EventsEntity::Step,
        EventsEntity::Decision,
        EventsEntity::Constraint,
        EventsEntity::Commit,
        EventsEntity::Release,
        EventsEntity::Milestone,
        EventsEntity::Environment,
        EventsEntity::Deployment,
        EventsEntity::Workspace,
        EventsEntity::WorkspaceMilestone,
        EventsEntity::Resource,
        EventsEntity::Component,
        EventsEntity::Note,
        EventsEntity::ChatSession,
        EventsEntity::ProtocolRun,
        EventsEntity::Runner,
        EventsEntity::Alert,
        EventsEntity::Persona,
        EventsEntity::Skill,
        EventsEntity::Protocol,
        EventsEntity::FeatureGraph,
        EventsEntity::Episode,
        EventsEntity::AnalysisProfile,
        EventsEntity::Trigger,
        EventsEntity::TopologyRule,
        EventsEntity::LifecycleHook,
        EventsEntity::Learning,
        EventsEntity::AttentionChanged,
    ];

    const NOTES_ALL: [NotesEntity; 30] = [
        NotesEntity::Project,
        NotesEntity::File,
        NotesEntity::Module,
        NotesEntity::Function,
        NotesEntity::Struct,
        NotesEntity::Trait,
        NotesEntity::Enum,
        NotesEntity::Impl,
        NotesEntity::Task,
        NotesEntity::Plan,
        NotesEntity::Step,
        NotesEntity::Commit,
        NotesEntity::Decision,
        NotesEntity::Constraint,
        NotesEntity::Milestone,
        NotesEntity::Release,
        NotesEntity::Workspace,
        NotesEntity::WorkspaceMilestone,
        NotesEntity::Resource,
        NotesEntity::Component,
        NotesEntity::FeatureGraph,
        NotesEntity::Protocol,
        NotesEntity::ProtocolState,
        NotesEntity::ProtocolRun,
        NotesEntity::PlanRun,
        NotesEntity::Skill,
        NotesEntity::Note,
        NotesEntity::ChatSession,
        NotesEntity::Process,
        NotesEntity::Document,
    ];

    #[test]
    fn events_entity_type_is_fully_classified() {
        let source = include_str!("../events/types.rs");
        let in_source = variants_in(source, "EntityType");
        let listed: Vec<String> = EVENTS_ALL.iter().map(|e| format!("{e:?}")).collect();
        for name in &in_source {
            assert!(
                listed.contains(name),
                "events::EntityType::{name} is not in EVENTS_ALL: decide in registry.rs \
                 whether it can be referenced (classify_events), then list it"
            );
        }
        assert_eq!(
            listed.len(),
            in_source.len(),
            "EVENTS_ALL lists a removed variant"
        );
    }

    #[test]
    fn notes_entity_type_is_fully_classified() {
        let source = include_str!("../notes/models.rs");
        let in_source = variants_in(source, "EntityType");
        let listed: Vec<String> = NOTES_ALL.iter().map(|e| format!("{e:?}")).collect();
        for name in &in_source {
            assert!(
                listed.contains(name),
                "notes::EntityType::{name} is not in NOTES_ALL: decide in registry.rs \
                 whether it can be referenced (classify_notes), then list it"
            );
        }
        assert_eq!(
            listed.len(),
            in_source.len(),
            "NOTES_ALL lists a removed variant"
        );
    }

    #[test]
    fn every_classified_kind_is_backed_by_the_registry() {
        for e in &EVENTS_ALL {
            if let Classification::Kind(k) = classify_events(e) {
                assert_eq!(&spec(k).backing.events_entity, e, "events::{e:?}");
            }
        }
        for e in &NOTES_ALL {
            if let Classification::Kind(k) = classify_notes(e) {
                assert_eq!(&spec(k).backing.notes_entity, e, "notes::{e:?}");
            }
        }
    }

    #[test]
    fn every_active_kind_is_reachable_from_both_enums() {
        for kind in RefKind::ALL {
            let s = spec(kind);
            // The plain reading of an entity must be classified to a kind
            // whose backing is that entity; an RFC shares its entity with Note.
            assert!(
                matches!(
                    classify_events(&s.backing.events_entity),
                    Classification::Kind(_)
                ),
                "{kind}: events side"
            );
            assert!(
                matches!(
                    classify_notes(&s.backing.notes_entity),
                    Classification::Kind(_)
                ),
                "{kind}: notes side"
            );
        }
    }

    #[test]
    fn rfc_is_a_note_of_type_rfc_and_has_no_entity_type_of_its_own() {
        let s = spec(RefKind::Rfc);
        assert_eq!(s.backing.events_entity, EventsEntity::Note);
        assert_eq!(s.backing.notes_entity, NotesEntity::Note);
        assert_eq!(s.backing.note_type, Some(NoteType::Rfc));
        assert!(
            !variants_in(include_str!("../events/types.rs"), "EntityType")
                .iter()
                .any(|v| v == "Rfc")
        );
        assert!(
            !variants_in(include_str!("../notes/models.rs"), "EntityType")
                .iter()
                .any(|v| v == "Rfc")
        );
        for kind in [
            RefKind::Plan,
            RefKind::Task,
            RefKind::Note,
            RefKind::Decision,
        ] {
            assert_eq!(spec(kind).backing.note_type, None);
        }
    }

    #[test]
    fn active_kinds_are_tier_a_enabled_and_reserved_ones_are_off() {
        for kind in RefKind::ALL {
            let s = spec(kind);
            assert_eq!((s.tier, s.enabled), (Tier::A, true), "{kind}");
        }
        for r in RESERVED {
            assert_eq!((r.tier, r.enabled), (Tier::B, false), "{}", r.name);
        }
    }

    #[test]
    fn reserved_names_are_classified_and_never_collide_with_an_active_kind() {
        for r in RESERVED {
            assert!(RefKind::ALL.iter().all(|k| k.as_str() != r.name));
            let seen = EVENTS_ALL
                .iter()
                .any(|e| classify_events(e) == Classification::Reserved(r.name));
            assert!(seen, "{} is reserved but no EntityType maps to it", r.name);
        }
    }

    #[test]
    fn lookup_tells_active_reserved_and_unknown_apart() {
        assert_eq!(lookup("plan"), Lookup::Active(RefKind::Plan));
        assert_eq!(lookup("rfc"), Lookup::Active(RefKind::Rfc));
        assert_eq!(lookup("persona"), Lookup::Reserved(RESERVED[0]));
        assert_eq!(lookup("skill"), Lookup::Reserved(RESERVED[1]));
        assert_eq!(lookup("workspace"), Lookup::Unknown);
        assert_eq!(lookup("Plan"), Lookup::Unknown);
        assert_eq!(lookup(""), Lookup::Unknown);
    }

    #[test]
    fn exclusions_carry_a_reason() {
        for e in &EVENTS_ALL {
            if let Classification::Excluded(why) = classify_events(e) {
                assert!(!why.is_empty(), "events::{e:?}");
            }
        }
        for e in &NOTES_ALL {
            if let Classification::Excluded(why) = classify_notes(e) {
                assert!(!why.is_empty(), "notes::{e:?}");
            }
        }
    }
}
