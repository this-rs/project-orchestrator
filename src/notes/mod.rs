//! Knowledge Notes module
//!
//! Provides a system for capturing and managing contextual knowledge from conversations:
//! guidelines, gotchas, patterns, tips, and verifiable assertions.
//!
//! Notes can be linked to code entities and automatically surfaced to agents
//! based on relevance and graph propagation.

pub mod hashing;
pub mod lifecycle;
pub mod manager;
pub mod models;
pub mod witness;

/// Time constant (days) of note energy decay — the single value used by the
/// heartbeat, skill maintenance and data migrations (it used to be 14 days
/// in one place and 90 in others).
pub const ENERGY_HALF_LIFE_DAYS: f64 = 90.0;

/// Days without human activity (code sync, chat) after which a project is
/// dormant: its notes' energy and synapses stop decaying until it is active
/// again, so knowledge survives months without work on the project.
pub const PROJECT_DORMANT_AFTER_DAYS: i64 = 14;

/// Whether a note is knowledge the agent may still be given: active or
/// awaiting review, and not replaced by a newer note. Archived, obsolete
/// (invalidated), stale and superseded notes must never reach its context —
/// newer knowledge wins.
///
/// Audit-trail notes written by skill evolution (`created_by =
/// "skill-evolution"`, e.g. "Skill X orphaned and archived") are a log, not
/// knowledge: never given to the agent either.
pub fn is_current_knowledge(note: &Note) -> bool {
    matches!(note.status, NoteStatus::Active | NoteStatus::NeedsReview)
        && note.superseded_by.is_none()
        && note.created_by != SKILL_EVOLUTION_AUTHOR
}

/// `created_by` of the audit-trail notes skill evolution writes.
pub const SKILL_EVOLUTION_AUTHOR: &str = "skill-evolution";

pub use hashing::*;
pub use lifecycle::*;
pub use manager::{BackfillProgress, NoteManager, SynapseBackfillProgress, SynapseConfig};
pub use models::*;
pub use witness::{CodeRef, ExternalRef, Witness, WitnessKind, WitnessValidationError};

#[cfg(test)]
mod current_knowledge_tests {
    use super::*;

    #[test]
    fn test_only_current_real_knowledge_reaches_the_agent() {
        let mk = || Note::new(None, NoteType::Gotcha, "x".into(), "user".into());
        assert!(is_current_knowledge(&mk()));
        let mut archived = mk();
        archived.status = NoteStatus::Archived;
        assert!(!is_current_knowledge(&archived));
        let mut replaced = mk();
        replaced.superseded_by = Some(uuid::Uuid::new_v4());
        assert!(!is_current_knowledge(&replaced));
        let trace = Note::new(
            None,
            NoteType::Observation,
            "Skill X orphaned".into(),
            SKILL_EVOLUTION_AUTHOR.into(),
        );
        assert!(
            !is_current_knowledge(&trace),
            "evolution audit trail is not knowledge"
        );
    }
}
