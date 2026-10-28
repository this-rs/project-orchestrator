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
        session: None,
        embedding: None, // Computed later if embedding pipeline is available
    };

    Ok(Some(episode))
}

/// Namespace of [`session_episode_id`]. Fixed: the id of a session's episode must
/// not depend on the instance, only on the session.
const SESSION_EPISODE_NAMESPACE: Uuid = Uuid::from_u128(0x5e55_1015_e915_0de0_a11c_e0f5_e551_0000);

/// The id of the episode of session `session_id`: the same at every close.
pub fn session_episode_id(session_id: Uuid) -> Uuid {
    Uuid::new_v5(&SESSION_EPISODE_NAMESPACE, session_id.as_bytes())
}

/// Whether the session works inside a run that has an episode of its own, which a
/// session episode would count twice: a protocol state agent (`protocol_runner`: its
/// run IS a protocol run), or a session of a plan run (runner, pipeline, gate, trigger)
/// whose plan run carries a lifecycle protocol run (the run episode is collected from
/// it). A plan run without one leaves no run episode: its sessions keep theirs.
async fn run_has_its_own_episode(neo4j: &dyn GraphStore, spawned_by: Option<&str>) -> Result<bool> {
    let Some(v) = spawned_by.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok()) else {
        return Ok(false);
    };
    let Some(run_id) = v
        .get("run_id")
        .and_then(|r| r.as_str())
        .and_then(|r| Uuid::parse_str(r).ok())
    else {
        return Ok(false);
    };
    if v.get("type").and_then(|t| t.as_str()) == Some("protocol_runner") {
        return Ok(true);
    }
    Ok(neo4j
        .get_plan_run(run_id)
        .await?
        .is_some_and(|run| run.lifecycle_run_id.is_some()))
}

/// Take the episode a chat session leaves at its close, `close_reason` being the
/// reason announced on its `session_closed`.
///
/// Read from the session node as it stands at the close:
/// - **Stimulus**: its first user message (the node's `preview`), at its creation.
///   `UserRequest`, or `Manual` for a session another agent or a runner spawned.
/// - **Process**: no FSM state; its active span (`created_at` → last update).
/// - **Outcome**: no artefact list — nothing links what a session wrote back to it yet,
///   and an episode does not invent it. How it ended is `session.close_reason`.
/// - **Validation**: none.
/// - **Session**: id, messages exchanged, close reason, cost, agent.
///
/// Its id is derived from the session ([`session_episode_id`]): a later close of the
/// same session (after a resume) replaces the episode under the SAME id.
///
/// `Ok(None)` when there is nothing to keep: no such session, a session that never
/// received a message, one with no project (an episode belongs to a project), or one
/// that works inside a run with an episode of its own ([`run_has_its_own_episode`]:
/// counting the work twice would skew what is learned from it).
///
/// Known limits: `preview` is the first message cut to ~200 characters; a session
/// whose process died without a close (removed from the live table, never closed)
/// leaves no episode until it is resumed and closed.
pub async fn collect_session_episode(
    neo4j: &dyn GraphStore,
    session_id: Uuid,
    close_reason: &str,
) -> Result<Option<Episode>> {
    let Some(session) = neo4j.get_chat_session(session_id).await? else {
        return Ok(None);
    };
    let request = session
        .preview
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty());
    let Some(request) = request else {
        return Ok(None);
    };
    let Some(slug) = session.project_slug.as_deref() else {
        return Ok(None);
    };
    if run_has_its_own_episode(neo4j, session.spawned_by.as_deref()).await? {
        return Ok(None);
    }
    let Some(project) = neo4j.get_project_by_slug(slug).await? else {
        return Ok(None);
    };

    // The agent that ran it. Not naming it never costs the episode.
    let agent_id = match neo4j.agent_identity_namespace().await {
        Ok(ns) => crate::notes::provenance::agent_id(ns, &session.model),
        Err(e) => {
            tracing::warn!(session_id = %session_id, error = %e, "the agent of a closed session could not be named");
            None
        }
    };

    let spawned = session
        .spawned_by
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());
    let stimulus = Stimulus {
        request: request.to_string(),
        trigger: if spawned {
            StimulusTrigger::Manual
        } else {
            StimulusTrigger::UserRequest
        },
        timestamp: session.created_at,
        context_hash: None,
    };
    let process = Process {
        reasoning_tree_id: None,
        states_visited: Vec::new(),
        state_visits: Vec::new(),
        duration_ms: Some(
            (session.updated_at - session.created_at)
                .num_milliseconds()
                .max(0),
        ),
    };

    Ok(Some(Episode {
        id: session_episode_id(session_id),
        project_id: project.id,
        stimulus,
        process,
        outcome: Outcome {
            note_ids: Vec::new(),
            decision_ids: Vec::new(),
            commit_shas: Vec::new(),
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
        session: Some(SessionTrace {
            session_id,
            message_count: session.message_count,
            close_reason: close_reason.to_string(),
            total_cost_usd: session.total_cost_usd,
            agent_id,
        }),
        embedding: None,
    }))
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
            session: None,
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

    // ── session episodes ─────────────────────────────────────────────────

    /// A store holding project `p` and a session of it that received `preview`.
    async fn store_with_session(
        project: Option<&str>,
        preview: Option<&str>,
    ) -> (crate::neo4j::mock::MockGraphStore, Uuid, Option<Uuid>) {
        use crate::neo4j::GraphStore;
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = match project {
            Some(slug) => {
                let p = crate::test_helpers::test_project_named(slug);
                store.create_project(&p).await.unwrap();
                Some(p.id)
            }
            None => None,
        };
        let mut session = crate::test_helpers::test_chat_session(project);
        session.preview = preview.map(String::from);
        session.message_count = 4;
        session.total_cost_usd = Some(0.25);
        session.created_at = Utc::now() - chrono::Duration::seconds(90);
        session.updated_at = session.created_at + chrono::Duration::seconds(60);
        let session_id = session.id;
        store.create_chat_session(&session).await.unwrap();
        (store, session_id, project_id)
    }

    #[tokio::test]
    async fn a_closed_session_gives_an_episode_of_its_first_message_and_its_end() {
        use crate::neo4j::GraphStore;
        let (store, sid, project_id) =
            store_with_session(Some("alpha"), Some("  Fix the login bug  ")).await;
        let episode = collect_session_episode(&store, sid, "process_died")
            .await
            .unwrap()
            .expect("a session with a message and a project leaves an episode");

        assert_eq!(Some(episode.project_id), project_id);
        assert_eq!(episode.stimulus.request, "Fix the login bug");
        assert_eq!(episode.stimulus.trigger, StimulusTrigger::UserRequest);
        assert_eq!(episode.process.duration_ms, Some(60_000));
        assert_eq!(episode.source_run_id, None);
        assert_eq!(
            episode.session,
            Some(SessionTrace {
                session_id: sid,
                message_count: 4,
                close_reason: "process_died".into(),
                total_cost_usd: Some(0.25),
                agent_id: crate::notes::provenance::agent_id(
                    store.agent_identity_namespace().await.unwrap(),
                    &crate::test_helpers::test_chat_session(None).model,
                ),
            })
        );
        assert!(
            episode.session.as_ref().unwrap().agent_id.is_some(),
            "the agent that ran it is named"
        );
        // Nothing links what the session wrote back to it: nothing is invented.
        assert!(episode.outcome.note_ids.is_empty() && episode.outcome.decision_ids.is_empty());
        assert_eq!(episode.validation.feedback_type, FeedbackType::None);
    }

    #[tokio::test]
    async fn a_spawned_session_is_not_a_user_request() {
        use crate::neo4j::GraphStore;
        let (store, sid, _) = store_with_session(Some("alpha"), Some("delegated work")).await;
        let mut node = store.get_chat_session(sid).await.unwrap().unwrap();
        node.spawned_by = Some(r#"{"type":"runner"}"#.into());
        store.chat_sessions.write().await.insert(sid, node);
        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(episode.stimulus.trigger, StimulusTrigger::Manual);
    }

    #[tokio::test]
    async fn a_session_with_nothing_to_keep_leaves_no_episode() {
        // Never received a message.
        let (store, sid, _) = store_with_session(Some("alpha"), None).await;
        assert!(collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .is_none());
        let (store, sid, _) = store_with_session(Some("alpha"), Some("   ")).await;
        assert!(collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .is_none());
        // No project: an episode belongs to one.
        let (store, sid, _) = store_with_session(None, Some("hello")).await;
        assert!(collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .is_none());
        // A project slug that names no project.
        let (store, _, _) = store_with_session(Some("alpha"), Some("hello")).await;
        let mut orphan = crate::test_helpers::test_chat_session(Some("gone"));
        orphan.preview = Some("hello".into());
        let orphan_id = orphan.id;
        crate::neo4j::GraphStore::create_chat_session(&store, &orphan)
            .await
            .unwrap();
        assert!(collect_session_episode(&store, orphan_id, "closed")
            .await
            .unwrap()
            .is_none());
        // No such session.
        assert!(collect_session_episode(&store, Uuid::new_v4(), "closed")
            .await
            .unwrap()
            .is_none());
    }

    /// A session inside a run whose episode exists already leaves none (no double
    /// count); a session of a plan run that leaves no run episode keeps its own.
    #[tokio::test]
    async fn a_session_is_counted_once_with_its_run() {
        use crate::neo4j::GraphStore;
        let with_lifecycle = Uuid::new_v4();
        let without_lifecycle = Uuid::new_v4();
        let cases = [
            (
                format!(
                    r#"{{"type":"protocol_runner","run_id":"{}","state_name":"s"}}"#,
                    Uuid::new_v4()
                ),
                false,
            ),
            (
                format!(
                    r#"{{"type":"runner","run_id":"{with_lifecycle}","task_id":"{}"}}"#,
                    Uuid::new_v4()
                ),
                false,
            ),
            (
                format!(
                    r#"{{"type":"runner","run_id":"{without_lifecycle}","task_id":"{}"}}"#,
                    Uuid::new_v4()
                ),
                true,
            ),
            (
                format!(
                    r#"{{"type":"runner","run_id":"{}","task_id":"{}"}}"#,
                    Uuid::new_v4(),
                    Uuid::new_v4()
                ),
                true,
            ),
        ];
        for (spawned, kept) in cases {
            let (store, sid, _) = store_with_session(Some("alpha"), Some("run work")).await;
            for (run_id, lifecycle) in [(with_lifecycle, true), (without_lifecycle, false)] {
                let mut run =
                    crate::runner::RunnerState::new(run_id, Uuid::new_v4(), 1, Default::default());
                run.lifecycle_run_id = lifecycle.then(Uuid::new_v4);
                store.create_plan_run(&run).await.unwrap();
            }
            let mut node = store.get_chat_session(sid).await.unwrap().unwrap();
            node.spawned_by = Some(spawned.clone());
            store.chat_sessions.write().await.insert(sid, node);
            let episode = collect_session_episode(&store, sid, "closed")
                .await
                .unwrap();
            assert_eq!(episode.is_some(), kept, "{spawned}");
        }
    }

    #[tokio::test]
    async fn a_session_with_nothing_more_to_keep_leaves_no_episode() {
        let (store, _, _) = store_with_session(Some("alpha"), Some("hello")).await;
        // No such session.
        assert!(collect_session_episode(&store, Uuid::new_v4(), "closed")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn a_session_episode_is_stored_once_per_session_and_goes_with_the_session() {
        use crate::neo4j::GraphStore;
        let (store, sid, project_id) = store_with_session(Some("alpha"), Some("first")).await;
        let project_id = project_id.unwrap();
        let first = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .unwrap();
        assert!(store.save_session_episode(&first).await.unwrap());
        // A second close (after a resume) replaces the first: one episode per session.
        let second = collect_session_episode(&store, sid, "process_died")
            .await
            .unwrap()
            .unwrap();
        assert!(store.save_session_episode(&second).await.unwrap());
        let listed = store.list_session_episodes(project_id, 10).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, second.id);
        assert_eq!(
            first.id, second.id,
            "the same id at every close of a session"
        );
        assert_eq!(first.id, session_episode_id(sid));
        assert_eq!(
            listed[0].session.as_ref().unwrap().close_reason,
            "process_died"
        );

        // Deleted with its session; and never written for a session that is gone.
        assert!(store.delete_chat_session(sid).await.unwrap());
        assert!(store
            .list_session_episodes(project_id, 10)
            .await
            .unwrap()
            .is_empty());
        assert!(!store.save_session_episode(&second).await.unwrap());
        assert!(store
            .list_session_episodes(project_id, 10)
            .await
            .unwrap()
            .is_empty());
    }

    /// The recall of a closed session: by the vector of its first message, within its
    /// project, never an episode without a vector, the closest first, `limit` kept.
    #[tokio::test]
    async fn a_closed_session_is_recalled_by_the_vector_of_its_first_message() {
        use crate::neo4j::GraphStore;
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        store.create_project(&project).await.unwrap();
        let other = crate::test_helpers::test_project_named("beta");
        store.create_project(&other).await.unwrap();
        let mut saved = Vec::new();
        for (slug, preview, vector) in [
            ("alpha", "login page 500", Some(vec![1.0, 0.0, 0.0])),
            ("alpha", "login timeout", Some(vec![0.9, 0.1, 0.0])),
            ("alpha", "rotate api keys", Some(vec![0.0, 0.0, 1.0])),
            ("alpha", "no vector at all", None),
            ("beta", "login page 500", Some(vec![1.0, 0.0, 0.0])),
        ] {
            let mut session = crate::test_helpers::test_chat_session(Some(slug));
            session.preview = Some(preview.into());
            store.create_chat_session(&session).await.unwrap();
            let mut episode = collect_session_episode(&store, session.id, "closed")
                .await
                .unwrap()
                .unwrap();
            episode.embedding = vector;
            assert!(store.save_session_episode(&episode).await.unwrap());
            saved.push(episode.id);
        }
        let hits = store
            .search_episodes_by_embedding(project.id, &[1.0, 0.0, 0.0], 2)
            .await
            .unwrap();
        assert_eq!(
            hits.iter().map(|(e, _)| e.id).collect::<Vec<_>>(),
            vec![saved[0], saved[1]],
            "the closest first, the other project and the vectorless episode absent, limit kept"
        );
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
        assert!(hits[1].1 < 1.0 && hits[1].1 > 0.9);
    }

    #[test]
    fn an_episode_serialized_before_sessions_existed_still_reads() {
        let mut value = serde_json::to_value(sample_episode()).unwrap();
        assert!(
            value.get("session").is_none(),
            "absent, not null, on a run episode"
        );
        value.as_object_mut().unwrap().remove("session");
        let back: Episode = serde_json::from_value(value).unwrap();
        assert!(back.session.is_none());
    }
}

/// Episode builders for tests of other modules.
#[cfg(test)]
pub mod tests_support {
    use super::*;
    use chrono::{DateTime, Utc};

    /// An episode that started at `start`. With `closed_at`, a session episode closed
    /// then; otherwise a run episode lasting `duration_ms`.
    pub fn episode_at(
        start: DateTime<Utc>,
        duration_ms: Option<i64>,
        closed_at: Option<DateTime<Utc>>,
    ) -> Episode {
        Episode {
            id: Uuid::new_v4(),
            project_id: Uuid::nil(),
            stimulus: Stimulus {
                request: "r".into(),
                trigger: StimulusTrigger::Manual,
                timestamp: start,
                context_hash: None,
            },
            process: Process {
                reasoning_tree_id: None,
                states_visited: Vec::new(),
                state_visits: Vec::new(),
                duration_ms: closed_at
                    .map(|c| (c - start).num_milliseconds())
                    .or(duration_ms),
            },
            outcome: Outcome {
                note_ids: Vec::new(),
                decision_ids: Vec::new(),
                commit_shas: Vec::new(),
                files_modified: 0,
            },
            validation: Validation {
                feedback_type: FeedbackType::None,
                score: None,
                evidence_count: 0,
            },
            lesson: None,
            collected_at: closed_at.unwrap_or_else(Utc::now),
            source_run_id: closed_at.is_none().then(Uuid::new_v4),
            session: closed_at.map(|_| SessionTrace {
                session_id: Uuid::new_v4(),
                message_count: 1,
                close_reason: "closed".into(),
                total_cost_usd: None,
                agent_id: None,
            }),
            embedding: None,
        }
    }
}
