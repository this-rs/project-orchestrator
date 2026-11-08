//! The single gate on the read side: what a note is ALLOWED to say.
//!
//! Every surface that puts knowledge into a prompt used to decide by itself —
//! and none of them consulted who wrote the note. `Recent Knowledge` took the
//! five newest `active` notes; `Relevant Notes` ranked on lexical/vector
//! similarity and printed `importance` as a label. A sentence a regex picked out
//! of a conversation therefore arrived with the same authority as a gotcha a
//! person wrote, and the reader had no way to tell — the information was not in
//! the graph.
//!
//! This module is that information, and the one rule that turns it into a label:
//! **a note whose author is a heuristic, and that no person ever confirmed, is an
//! `observation` — never the `gotcha|critical` its own writer claimed.** It is
//! also capped per block ([`MAX_UNVOUCHED_PER_BLOCK`]), because a block of
//! observations is not a block of knowledge however well-formed each one is.
//!
//! The gate reads [`Provenance`], which both carriers can produce: a `Note` from
//! the graph, or a `NoteDocument` from the search index.

use crate::notes::models::Note;

/// Most unvouched notes a single injected block may carry.
///
/// Two is enough for the model to know the graph noticed something, and few
/// enough that observations cannot crowd out knowledge.
pub const MAX_UNVOUCHED_PER_BLOCK: usize = 2;

/// What a note may be presented as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Visible {
    /// The type to display — `observation` when the author is a heuristic.
    pub note_type: String,
    /// The importance to display — `low` in the same case.
    pub importance: String,
    /// A person stands behind it (wrote it, or confirmed it).
    pub vouched: bool,
    /// Why, in one line: logged, and asserted by tests.
    pub reason: &'static str,
}

/// Everything the gate reads about a note, whatever carries it.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// `human` | `agent` | `system`. `None` on a document indexed before the
    /// provenance field existed.
    pub author_kind: Option<String>,
    /// The note's tags. `auto-observed` is the fallback signal, because every
    /// detector note has carried it since the detector existed.
    pub tags: Vec<String>,
    pub note_type: String,
    pub importance: String,
    /// `ephemeral` | `operational` | `consolidated`.
    pub memory_horizon: Option<String>,
    /// Whether a person has confirmed it (`Note::is_human_confirmed`).
    pub human_confirmed: bool,
}

impl Provenance {
    /// Read a full note from the graph.
    pub fn of_note(note: &Note) -> Self {
        Self {
            author_kind: Some(note.author_kind.as_str().to_string()),
            tags: note.tags.clone(),
            note_type: note.note_type.to_string(),
            importance: note.importance.to_string(),
            memory_horizon: Some(note.memory_horizon.to_string()),
            human_confirmed: note.is_human_confirmed(),
        }
    }

    /// Whether the author is a heuristic rather than a deliberate writer.
    ///
    /// The tag is checked even when the author kind is known: a note written by
    /// a buggy or older path would otherwise pass as deliberate.
    pub fn is_heuristic(&self) -> bool {
        match self.author_kind.as_deref() {
            Some("system") => true,
            _ => self.tags.iter().any(|t| t == "auto-observed"),
        }
    }

    /// A person stands behind this note.
    pub fn is_vouched(&self) -> bool {
        self.human_confirmed || self.author_kind.as_deref() == Some("human")
    }
}

/// The gate itself.
pub fn visibility(provenance: &Provenance) -> Visible {
    if provenance.is_heuristic() && !provenance.is_vouched() {
        return Visible {
            note_type: "observation".to_string(),
            importance: "low".to_string(),
            vouched: false,
            reason: "heuristic author, never confirmed by a person",
        };
    }
    let vouched = provenance.is_vouched();
    Visible {
        note_type: provenance.note_type.clone(),
        importance: provenance.importance.clone(),
        vouched,
        reason: if vouched {
            "vouched by a person"
        } else {
            "deliberate author, no human signature"
        },
    }
}

/// Whether a note may be presented to a model as foundation knowledge.
///
/// Requires both: a person behind it, and a life long enough to have survived
/// being read. Consolidation alone proves nothing — attention is what used to
/// promote notes here, and every note is born at energy 1.0.
pub fn foundational_eligible(provenance: &Provenance, visible: &Visible) -> bool {
    visible.vouched && provenance.memory_horizon.as_deref() == Some("consolidated")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::models::{NoteAuthorKind, NoteImportance, NoteType};

    fn detector_note() -> Note {
        let mut note = Note::new(
            None,
            NoteType::Observation,
            "**Gotcha**: Le contrôle des prix les a tous rattrapés.".to_string(),
            "system/observation-detector".to_string(),
        )
        .with_author(NoteAuthorKind::System, "system/observation-detector");
        note.importance = NoteImportance::Low;
        note.tags = vec!["auto-observed".to_string()];
        note
    }

    #[test]
    fn a_heuristic_author_is_an_observation_not_a_gotcha() {
        let mut note = detector_note();
        // The detector used to declare `gotcha|high`; the gate must not believe it.
        note.note_type = NoteType::Gotcha;
        note.importance = NoteImportance::High;

        let visible = visibility(&Provenance::of_note(&note));
        assert_eq!(visible.note_type, "observation");
        assert_eq!(visible.importance, "low");
        assert!(!visible.vouched);
    }

    #[test]
    fn an_old_document_with_no_author_kind_falls_back_to_the_tag() {
        let provenance = Provenance {
            author_kind: None,
            tags: vec!["auto-observed".to_string(), "gotcha-candidate".to_string()],
            note_type: "gotcha".to_string(),
            importance: "high".to_string(),
            memory_horizon: Some("consolidated".to_string()),
            human_confirmed: false,
        };
        let visible = visibility(&provenance);
        assert_eq!(visible.note_type, "observation");
        assert!(!visible.vouched);
        assert!(
            !foundational_eligible(&provenance, &visible),
            "a consolidated observation is not foundation knowledge"
        );
    }

    #[test]
    fn a_human_confirmed_note_keeps_its_own_word() {
        let mut note = detector_note();
        note.note_type = NoteType::Gotcha;
        note.importance = NoteImportance::High;
        note.confirm_by_human("julien", chrono::Utc::now());

        let visible = visibility(&Provenance::of_note(&note));
        assert_eq!(visible.note_type, "gotcha");
        assert_eq!(visible.importance, "high");
        assert!(visible.vouched);
    }

    #[test]
    fn a_person_writing_a_note_vouches_for_it_by_construction() {
        let note = Note::new(
            None,
            NoteType::Guideline,
            "Always use Arc<dyn GraphStore> in async code.".to_string(),
            "julien".to_string(),
        )
        .with_author(NoteAuthorKind::Human, "julien");

        let visible = visibility(&Provenance::of_note(&note));
        assert_eq!(visible.note_type, "guideline");
        assert!(visible.vouched);
        assert_eq!(visible.reason, "vouched by a person");
    }

    #[test]
    fn an_agent_note_is_deliberate_but_not_vouched() {
        let note = Note::new(
            None,
            NoteType::Pattern,
            "The enrichment stages all follow the same pipeline.".to_string(),
            "agent:204ce865-d253-4f59-816c-5f9d8f5ff1ac".to_string(),
        );

        let provenance = Provenance::of_note(&note);
        let visible = visibility(&provenance);

        assert!(!provenance.is_heuristic(), "an agent is not a heuristic");
        assert!(!visible.vouched);
        assert_eq!(
            visible.note_type, "pattern",
            "its own word is kept — it is not a machine fragment"
        );
        assert_eq!(visible.reason, "deliberate author, no human signature");
        assert!(
            !foundational_eligible(&provenance, &visible),
            "and it is not foundation knowledge until a person says so"
        );
    }
}
