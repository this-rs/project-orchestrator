//! Episode Collector — assembles complete Episodes from the knowledge graph.
//!
//! Given a completed ProtocolRun, the collector queries Neo4j to build:
//! - **Stimulus**: from run.triggered_by + run.started_at
//! - **Process**: from run.states_visited + PersistedReasoningTree (via REASONING_FOR)
//! - **Outcome**: from PRODUCED_DURING relations (notes + decisions)
//! - **Validation**: from reasoning feedback (if available)
//!
//! The resulting Episode can then be converted to a PortableEpisode for
//! cross-instance exchange via SkillPackage v3.

use anyhow::{Context, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::episodes::models::*;
use crate::neo4j::GraphStore;

/// Collect a complete Episode from a completed ProtocolRun.
///
/// Returns `None` if the run doesn't exist.
/// Returns an error only on Neo4j failures.
pub async fn collect_episode(
    neo4j: &dyn GraphStore,
    run_id: Uuid,
    project_id: Uuid,
) -> Result<Option<Episode>> {
    // 1. Get the protocol run
    let run = match neo4j.get_protocol_run(run_id).await? {
        Some(r) => r,
        None => return Ok(None),
    };

    // 2. Build Stimulus
    let stimulus = Stimulus {
        request: run.triggered_by.clone(),
        trigger: match run.triggered_by.as_str() {
            "manual" => StimulusTrigger::Manual,
            s if s.starts_with("event:") => StimulusTrigger::SystemEvent,
            s if s.starts_with("schedule:") => StimulusTrigger::SystemEvent,
            _ => StimulusTrigger::ProtocolTransition,
        },
        timestamp: run.started_at,
        context_hash: None,
    };

    // 3. Build Process — states visited + optional reasoning tree
    let states_visited: Vec<String> = run
        .states_visited
        .iter()
        .map(|sv| sv.state_name.clone())
        .collect();

    // Build enriched StateVisitRecords from the run's StateVisits
    let state_visits: Vec<StateVisitRecord> = run
        .states_visited
        .iter()
        .map(|sv| StateVisitRecord {
            state_name: sv.state_name.clone(),
            entered_at: sv.entered_at,
            exited_at: sv.exited_at,
            duration_ms: sv.duration_ms.or_else(|| {
                // Compute duration from timestamps if not already set
                sv.exited_at
                    .map(|exit| (exit - sv.entered_at).num_milliseconds())
            }),
            trigger: sv.trigger.clone(),
        })
        .collect();

    let duration_ms = run
        .completed_at
        .map(|end| (end - run.started_at).num_milliseconds());

    // Check for a persisted reasoning tree linked to this run
    let reasoning_tree_id: Option<Uuid> = neo4j
        .get_run_reasoning_tree_id(run_id)
        .await
        .unwrap_or_default();

    let process = Process {
        reasoning_tree_id,
        states_visited,
        state_visits,
        duration_ms,
    };

    // 4. Build Outcome — notes + decisions produced during this run
    let artefacts = neo4j.get_run_outcomes(run_id).await?;

    let mut note_ids = Vec::new();
    let mut decision_ids = Vec::new();
    for art in &artefacts {
        match art.entity_type.as_str() {
            "note" => note_ids.push(art.entity_id),
            "decision" => decision_ids.push(art.entity_id),
            _ => {}
        }
    }

    let outcome = Outcome {
        note_ids,
        decision_ids,
        commit_shas: Vec::new(), // TODO: link commits to runs in the future
        files_modified: 0,       // TODO: count from commit files_changed
    };

    // 5. Build Validation — check for reasoning feedback
    // For now, use implicit signals: if the run completed successfully
    // and produced artefacts, that's implicit positive feedback.
    let validation =
        if !artefacts.is_empty() && run.status == crate::protocol::models::RunStatus::Completed {
            Validation {
                feedback_type: FeedbackType::ImplicitPositive,
                score: None,
                evidence_count: artefacts.len(),
            }
        } else {
            Validation {
                feedback_type: FeedbackType::None,
                score: None,
                evidence_count: 0,
            }
        };

    // 6. Assemble the Episode
    let episode = Episode {
        id: Uuid::new_v4(),
        project_id,
        stimulus,
        process,
        outcome,
        validation,
        lesson: None, // Lessons are extracted later (optionally by LLM)
        collected_at: Utc::now(),
        source_run_id: Some(run_id),
        embedding: None, // Computed later if embedding pipeline is available
    };

    Ok(Some(episode))
}

/// List all collectable episodes for a project.
///
/// Finds completed ProtocolRuns linked to the project and collects each one.
/// Skips runs that fail to collect (logs a warning).
pub async fn list_episodes(
    neo4j: &dyn GraphStore,
    project_id: Uuid,
    limit: usize,
) -> Result<Vec<Episode>> {
    // Find completed runs for this project's protocols
    let runs = neo4j
        .list_completed_runs_for_project(project_id, limit)
        .await
        .context("Failed to list completed runs for project")?;

    let mut episodes = Vec::with_capacity(runs.len());
    for run in &runs {
        match collect_episode(neo4j, run.id, project_id).await {
            Ok(Some(ep)) => episodes.push(ep),
            Ok(None) => {} // run disappeared between list and collect
            Err(e) => {
                tracing::warn!(
                    run_id = %run.id,
                    error = %e,
                    "Failed to collect episode from run, skipping"
                );
            }
        }
    }

    Ok(episodes)
}

// ============================================================================
// Distillation collection
// ============================================================================

/// A bundle of episode data ready for the distillation pipeline.
///
/// Collects the raw materials that `abstract_lesson()` and `anonymize()` need.
#[derive(Debug, Clone)]
pub struct DistillationBundle {
    /// The source episode.
    pub episode: Episode,
    /// Note titles extracted from the episode's outcome.
    pub note_titles: Vec<String>,
    /// Tags aggregated from all notes.
    pub tags: Vec<String>,
    /// Concatenated note content.
    pub content: String,
}

/// Prepare an episode for the distillation pipeline.
///
/// Assembles a [`DistillationBundle`] from an episode and its associated note
/// metadata. In a full implementation this would query Neo4j for note details;
/// here we accept pre-resolved note data to keep the function testable without
/// a live graph.
pub fn collect_for_distillation(
    episode: Episode,
    note_titles: Vec<String>,
    note_tags: Vec<String>,
    note_content: String,
) -> DistillationBundle {
    DistillationBundle {
        episode,
        note_titles,
        tags: note_tags,
        content: note_content,
    }
}

impl DistillationBundle {
    /// Convert into the [`EpisodeContent`] expected by `abstract_lesson()`.
    pub fn to_episode_content(&self) -> crate::episodes::distill::EpisodeContent {
        crate::episodes::distill::EpisodeContent {
            note_titles: self.note_titles.clone(),
            tags: self.tags.clone(),
            content: self.content.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_episode() -> Episode {
        Episode {
            id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            stimulus: Stimulus {
                request: "test request".to_string(),
                trigger: StimulusTrigger::UserRequest,
                timestamp: Utc::now(),
                context_hash: None,
            },
            process: Process {
                reasoning_tree_id: None,
                states_visited: vec![],
                state_visits: vec![],
                duration_ms: None,
            },
            outcome: Outcome {
                note_ids: vec![Uuid::new_v4()],
                decision_ids: vec![],
                commit_shas: vec![],
                files_modified: 0,
            },
            validation: Validation {
                feedback_type: FeedbackType::None,
                score: None,
                evidence_count: 0,
            },
            lesson: None,
            collected_at: Utc::now(),
            source_run_id: None,
            embedding: None,
        }
    }

    #[test]
    fn test_collect_for_distillation() {
        let episode = sample_episode();
        let ep_id = episode.id;
        let bundle = collect_for_distillation(
            episode,
            vec!["Note A".to_string(), "Note B".to_string()],
            vec!["rust".to_string(), "testing".to_string()],
            "Combined note content here.".to_string(),
        );
        assert_eq!(bundle.episode.id, ep_id);
        assert_eq!(bundle.note_titles.len(), 2);
        assert_eq!(bundle.tags.len(), 2);
        assert!(!bundle.content.is_empty());
    }

    #[test]
    fn test_bundle_to_episode_content() {
        let episode = sample_episode();
        let bundle = collect_for_distillation(
            episode,
            vec!["Title".to_string()],
            vec!["tag".to_string()],
            "content".to_string(),
        );
        let ec = bundle.to_episode_content();
        assert_eq!(ec.note_titles.len(), 1);
        assert_eq!(ec.tags, vec!["tag".to_string()]);
        assert_eq!(ec.content, "content");
    }

    #[tokio::test]
    async fn test_collect_episode_from_completed_run() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create a completed run with enriched state visits
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        let state2 = Uuid::new_v4();
        run.visit_state(state2, "Processing", "begin");
        run.complete();
        let run_id = run.id;

        // Persist in mock store
        store.protocol_runs.write().await.insert(run_id, run);

        // Collect episode
        let episode = collect_episode(&store, run_id, project_id)
            .await
            .unwrap()
            .expect("Should collect an episode");

        // Verify stimulus
        assert_eq!(episode.stimulus.request, "manual");
        assert_eq!(episode.stimulus.trigger, StimulusTrigger::Manual);

        // Verify process — states visited names
        assert_eq!(episode.process.states_visited, vec!["Start", "Processing"]);

        // Verify enriched state_visits
        assert_eq!(episode.process.state_visits.len(), 2);
        assert_eq!(episode.process.state_visits[0].state_name, "Start");
        assert!(
            episode.process.state_visits[0].exited_at.is_some(),
            "First state should have exited_at"
        );
        assert!(
            episode.process.state_visits[0].duration_ms.is_some(),
            "First state should have duration_ms"
        );
        assert_eq!(
            episode.process.state_visits[1].trigger,
            Some("begin".to_string())
        );

        // Verify duration is computed from timestamps
        assert!(episode.process.duration_ms.is_some());

        // Verify source run linkage
        assert_eq!(episode.source_run_id, Some(run_id));

        // Verify validation — no artefacts → FeedbackType::None
        assert_eq!(episode.validation.feedback_type, FeedbackType::None);
    }

    #[tokio::test]
    async fn test_collect_episode_run_not_found() {
        use crate::neo4j::mock::MockGraphStore;

        let store = MockGraphStore::new();
        let result = collect_episode(&store, Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        assert!(result.is_none(), "Non-existent run should return None");
    }

    #[tokio::test]
    async fn test_collect_episode_event_triggered_stimulus() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.triggered_by = "event:post_sync".to_string();
        run.complete();
        let run_id = run.id;
        store.protocol_runs.write().await.insert(run_id, run);

        let episode = collect_episode(&store, run_id, project_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(episode.stimulus.trigger, StimulusTrigger::SystemEvent);
        assert_eq!(episode.stimulus.request, "event:post_sync");
    }

    #[tokio::test]
    async fn test_collect_episode_schedule_triggered_stimulus() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.triggered_by = "schedule:daily".to_string();
        run.complete();
        let run_id = run.id;
        store.protocol_runs.write().await.insert(run_id, run);

        let episode = collect_episode(&store, run_id, project_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(episode.stimulus.trigger, StimulusTrigger::SystemEvent);
    }

    #[tokio::test]
    async fn test_collect_episode_state_visit_duration_fallback() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create a run with a manually crafted state visit that has exited_at
        // but no duration_ms — the collector should compute it from timestamps
        let now = Utc::now();
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "OnlyState");
        run.states_visited[0].exited_at = Some(now + chrono::Duration::seconds(5));
        run.states_visited[0].duration_ms = None; // Not pre-computed
        run.status = crate::protocol::models::RunStatus::Completed;
        run.completed_at = Some(now + chrono::Duration::seconds(5));
        let run_id = run.id;
        store.protocol_runs.write().await.insert(run_id, run);

        let episode = collect_episode(&store, run_id, project_id)
            .await
            .unwrap()
            .unwrap();

        // The collector should have computed duration from exited_at - entered_at
        let sv = &episode.process.state_visits[0];
        assert!(
            sv.duration_ms.is_some(),
            "Duration should be computed from timestamps"
        );
        assert!(
            sv.duration_ms.unwrap() >= 4000,
            "Duration should be ~5000ms"
        );
    }
}
