//! Project-safe Hebbian reinforcement of a set of co-activated notes.
//!
//! The note sets handed to `reinforce_synapses` by the chat, the bridge
//! subgraph and the isomorphic-group paths come from `get_notes_for_entity`,
//! which is not scoped to a project: one file path can anchor notes of several
//! projects. Synapses never cross projects, so the set is split per project
//! first; each group is reinforced on its own and the returned count is the
//! number of synapses really written.

use std::collections::BTreeMap;

use anyhow::Result;
use uuid::Uuid;

use super::models::Note;
use crate::neo4j::traits::GraphStore;

/// Note ids grouped by owning project. Duplicates are merged; notes without a
/// `project_id` belong to no group (they cannot be reinforced).
pub fn group_by_project(notes: &[Note]) -> BTreeMap<Uuid, Vec<Uuid>> {
    let mut groups: BTreeMap<Uuid, Vec<Uuid>> = BTreeMap::new();
    for note in notes {
        if let Some(project) = note.project_id {
            let ids = groups.entry(project).or_default();
            if !ids.contains(&note.id) {
                ids.push(note.id);
            }
        }
    }
    groups
}

/// Reinforce the synapses between the notes of each project separately.
///
/// Returns the sum of the synapses written. Groups of fewer than two notes are
/// skipped. Every group is attempted; the first error (if any) is returned
/// after the others have been tried.
pub async fn reinforce_per_project(
    graph: &dyn GraphStore,
    notes: &[Note],
    boost: f64,
) -> Result<usize> {
    let mut total = 0usize;
    let mut first_err: Option<anyhow::Error> = None;
    for ids in group_by_project(notes).values() {
        if ids.len() < 2 {
            continue;
        }
        match graph.reinforce_synapses(ids, boost).await {
            Ok(n) => total += n,
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(total),
    }
}

/// Project of a chat session (its `project_slug` resolved), if any.
pub async fn session_project_id(graph: &dyn GraphStore, session_id: Uuid) -> Option<Uuid> {
    let session = graph.get_chat_session(session_id).await.ok().flatten()?;
    let slug = session.project_slug?;
    graph
        .get_project_by_slug(&slug)
        .await
        .ok()
        .flatten()
        .map(|p| p.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::notes::models::NoteType;

    fn note(project: Option<Uuid>) -> Note {
        Note::new(project, NoteType::Guideline, "n".into(), "t".into())
    }

    #[tokio::test]
    async fn counts_only_the_synapses_written_inside_each_project() {
        let mock = MockGraphStore::new();
        let (p, q) = (Uuid::new_v4(), Uuid::new_v4());
        let notes = vec![
            note(Some(p)),
            note(Some(p)),
            note(Some(p)),
            note(Some(q)),
            note(None),
        ];
        for n in &notes {
            mock.create_note(n).await.unwrap();
        }
        // Project p: 3 notes = 3 pairs = 6 synapses; q has 1 note, the
        // project-less note has none: the 9 cross-group pairs add nothing.
        let n = reinforce_per_project(&mock, &notes, 0.1).await.unwrap();
        assert_eq!(n, 6);
        assert!(mock.get_synapses(notes[3].id).await.unwrap().is_empty());
        assert!(mock.get_synapses(notes[4].id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn reinforces_each_project_on_its_own() {
        let mock = MockGraphStore::new();
        let (p, q) = (Uuid::new_v4(), Uuid::new_v4());
        let notes = vec![note(Some(p)), note(Some(q)), note(Some(p)), note(Some(q))];
        for n in &notes {
            mock.create_note(n).await.unwrap();
        }
        let n = reinforce_per_project(&mock, &notes, 0.1).await.unwrap();
        assert_eq!(n, 4);
        let syn = mock.get_synapses(notes[0].id).await.unwrap();
        assert_eq!(syn.len(), 1);
        assert_eq!(syn[0].0, notes[2].id);
    }

    #[tokio::test]
    async fn duplicates_and_lone_notes_are_not_an_error() {
        let mock = MockGraphStore::new();
        let p = Uuid::new_v4();
        let a = note(Some(p));
        mock.create_note(&a).await.unwrap();
        let n = reinforce_per_project(&mock, &[a.clone(), a.clone()], 0.1)
            .await
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn group_by_project_drops_unknown_project() {
        let p = Uuid::new_v4();
        let (a, b) = (note(Some(p)), note(None));
        let g = group_by_project(&[a.clone(), b]);
        assert_eq!(g.len(), 1);
        assert_eq!(g[&p], vec![a.id]);
    }
}
