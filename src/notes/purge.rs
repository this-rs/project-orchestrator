//! What the heuristic authors wrote, and its repair.
//!
//! The observation detector wrote notes the graph had no way to recognise as
//! machine prose (see [`super::visibility`]). This module answers two questions,
//! in one place so that the answer cannot drift between the HTTP route and the
//! MCP action that both expose it:
//!
//! - *how much of the knowledge base is a regex's prose?* — the shape report;
//! - *can we put it aside?* — archiving the ones nobody confirmed.
//!
//! Nothing is ever deleted: an archived note keeps its history and can come
//! back. And `dry_run` is the default, everywhere: a question must not be
//! answered with a write.

use anyhow::Result;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

/// Report the population a heuristic wrote, and — unless `dry_run` — archive
/// the unconfirmed part of it.
///
/// Returns the before/after shapes, so a caller can see what changed rather
/// than trusting a count.
pub async fn purge_auto_observed(
    graph: &dyn GraphStore,
    project_id: Option<Uuid>,
    dry_run: bool,
) -> Result<Value> {
    let before = graph.auto_observed_report(project_id).await?;

    if dry_run {
        return Ok(json!({
            "dry_run": true,
            "project_id": project_id,
            "before": before,
            "archived": 0,
            "hint": "POST the same body with dry_run=false to archive the unconfirmed ones",
        }));
    }

    let archived = graph.archive_unconfirmed_auto_observed(project_id).await?;
    let after = graph.auto_observed_report(project_id).await?;

    Ok(json!({
        "dry_run": false,
        "project_id": project_id,
        "before": before,
        "archived": archived,
        "after": after,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use crate::notes::models::{Note, NoteAuthorKind, NoteImportance, NoteType};
    use crate::notes::NoteStatus;

    /// One detector note, one note a person wrote, in the same project.
    async fn two_notes() -> (MockGraphStore, Uuid, Uuid) {
        let store = MockGraphStore::new();
        let project = Uuid::new_v4();

        let mut detector = Note::new(
            Some(project),
            NoteType::Observation,
            "**Gotcha**: Le contrôle des prix les a tous rattrapés.".to_string(),
            "system/observation-detector".to_string(),
        )
        .with_author(NoteAuthorKind::System, "system/observation-detector");
        detector.importance = NoteImportance::Low;
        detector.tags = vec!["auto-observed".to_string()];
        let detector_id = detector.id;
        store.create_note(&detector).await.unwrap();

        let human = Note::new(
            Some(project),
            NoteType::Gotcha,
            "Le contrôle des prix se fait en fin de batch.".to_string(),
            "julien".to_string(),
        )
        .with_author(NoteAuthorKind::Human, "julien");
        let human_id = human.id;
        store.create_note(&human).await.unwrap();

        (store, detector_id, human_id)
    }

    #[tokio::test]
    async fn a_dry_run_reports_and_writes_nothing() {
        let (store, detector_id, _) = two_notes().await;

        let report = purge_auto_observed(&store, None, true).await.unwrap();

        assert_eq!(report["dry_run"], true);
        assert_eq!(report["archived"], 0);
        let before = report["before"].as_array().unwrap();
        assert_eq!(before.len(), 1, "one shape: the detector's single note");
        assert_eq!(before[0]["notes"], 1);
        assert!(before[0]["avg_size"].is_number());

        // Nothing moved: the note is still active, the human one untouched.
        let notes = store.notes.read().await;
        assert_eq!(notes[&detector_id].status, NoteStatus::Active);
    }

    #[tokio::test]
    async fn archiving_puts_the_unconfirmed_aside_and_keeps_the_persons_note() {
        let (store, detector_id, human_id) = two_notes().await;

        let report = purge_auto_observed(&store, None, false).await.unwrap();
        assert_eq!(report["dry_run"], false);
        assert_eq!(report["archived"], 1, "exactly one note is a heuristic's");

        let notes = store.notes.read().await;
        assert_eq!(
            notes[&detector_id].status,
            NoteStatus::Archived,
            "archived, never deleted"
        );
        assert_eq!(
            notes[&human_id].status,
            NoteStatus::Active,
            "a person's note is not the janitor's business"
        );
    }

    #[tokio::test]
    async fn the_report_is_scoped_to_one_project() {
        let (store, _, _) = two_notes().await;
        let elsewhere = purge_auto_observed(&store, Some(Uuid::new_v4()), true)
            .await
            .unwrap();
        assert_eq!(
            elsewhere["before"].as_array().unwrap().len(),
            0,
            "another project's population is not reported"
        );
    }
}
