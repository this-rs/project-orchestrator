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
