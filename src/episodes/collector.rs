//! Episode Collector — assembles complete Episodes from the knowledge graph.
//!
//! Given a completed ProtocolRun, the collector queries Neo4j to build:
//! - **Stimulus**: from run.triggered_by + run.started_at
//! - **Process**: from run.states_visited + PersistedReasoningTree (via REASONING_FOR)
//! - **Outcome**: the commits of the run's window (read from git) + the artefacts it
//!   produced (notes and decisions, via PRODUCED_DURING)
//! - **Validation**: from the EVIDENCE — a commit is proof, an artefact is not
//!
//! The resulting Episode can then be converted to a PortableEpisode for
//! cross-instance exchange via SkillPackage v3.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use anyhow::{Context, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::episodes::models::*;
#[allow(unused_imports)]
use crate::neo4j::GraphStore;

/// The validation a finished run leaves: what was PROVEN, not what was written.
///
/// `commit_shas` is the evidence — a commit is a fact the agent did not author alone.
/// Artefacts (notes, decisions) are deliberately absent from this decision: counting them
/// made "I documented my failure" a positive signal, and gave a reward for producing text
/// (RM-2, RM-5, decision `c23f494b`).
///
/// An empty evidence list is not a failure — it is `None`, "not verified", which is the
/// truth about a run nothing corroborates.
fn run_validation(commit_shas: &[String], completed: bool) -> Validation {
    let evidence = commit_shas.len();
    Validation {
        feedback_type: if evidence > 0 && completed {
            FeedbackType::ImplicitPositive
        } else {
            FeedbackType::None
        },
        score: None,
        evidence_count: evidence,
    }
}

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

    // 4. Build Outcome — the commits of the run's window, and the artefacts it produced.
    //
    // The commits are read from git in the run's OWN window, in the working tree of the
    // run's project — the same reader the session path uses (`commits_since`), so the two
    // cannot disagree about what a commit of this work is (RM-4, decision `c23f494b`).
    let commits = match run.project_id {
        Some(pid) => match neo4j.get_project(pid).await {
            Ok(Some(p)) => match p.root_path_opt() {
                Some(root) => crate::runner::enricher::commits_since(root, run.started_at)
                    .await
                    .unwrap_or_default(),
                None => Vec::new(),
            },
            _ => Vec::new(),
        },
        None => Vec::new(),
    };
    let committed_files: std::collections::HashSet<&str> = commits
        .iter()
        .flat_map(|c| c.files.iter().map(|f| f.path.as_str()))
        .collect();

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
        commit_shas: commits.iter().map(|c| c.hash.clone()).collect(),
        files_modified: committed_files.len(),
    };

    // 5. Build Validation — EVIDENCE, not output.
    //
    // This used to read: a finished run that produced artefacts is `ImplicitPositive`
    // with `evidence_count = artefacts.len()`. Writing a note was therefore counted as
    // proving a result, and an agent that documented its own failure collected a reward
    // for it. A note is not evidence, and a run that finished is not a run that delivered
    // (RM-2, RM-5, decision `c23f494b`).
    let validation = run_validation(
        &outcome.commit_shas,
        run.status == crate::protocol::models::RunStatus::Completed,
    );

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

/// The tools whose call changes a file. `SessionWorkLog` counts the first, second and fourth;
/// `MultiEdit` is added because it edits several places of one file in a single call.
const EDITING_TOOLS: [&str; 4] = ["Write", "Edit", "MultiEdit", "NotebookEdit"];

/// What a session's persisted events say it did, read once at the close.
#[derive(Debug, Default, PartialEq, Eq)]
struct SessionJournal {
    /// Distinct files an editing tool was called on (a file edited twice counts once).
    files_modified: usize,
    /// `user_message` events: the messages the session really received.
    user_messages: i64,
}

/// Reads the journal of a session from its events. Best effort: a graph that cannot answer
/// is logged and yields the empty journal, because the episode matters more than its detail.
async fn read_session_journal(neo4j: &dyn GraphStore, session_id: Uuid) -> SessionJournal {
    match neo4j.get_chat_events(session_id, -1, 100_000).await {
        Ok(events) => journal_of(
            events
                .iter()
                .map(|e| (e.event_type.as_str(), e.data.as_str())),
        ),
        Err(e) => {
            tracing::warn!(session_id = %session_id, error = %e, "the events of a closed session could not be read: its episode keeps no file count");
            SessionJournal::default()
        }
    }
}

/// The header lines an `apply_patch` body uses to name the file it touches.
const APPLY_PATCH_TAGS: [&str; 4] = [
    "*** Update File:",
    "*** Add File:",
    "*** Delete File:",
    "*** Move to:",
];

/// Record `raw` as a written path, unless it names nothing a count can trust.
///
/// Refused: a file descriptor (`&1`), a device (`/dev/null`), a variable (`$OUT`) and a
/// bare flag. Trailing punctuation from the command line (`;`, `\`, quotes) is stripped.
fn keep_path(out: &mut Vec<String>, raw: &str) {
    let path = raw
        .trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == '(' || c == ')')
        .trim_end_matches(';')
        .trim_end_matches('\\');
    if path.is_empty()
        || path.starts_with('&')
        || path.starts_with('$')
        || path.starts_with('-')
        || path.starts_with("/dev/")
    {
        return;
    }
    if !out.iter().any(|p| p == path) {
        out.push(path.to_string());
    }
}

/// The paths a shell command writes, read prudently.
///
/// A count that is too low is better than a count that invents files, so only the forms
/// that name their target unambiguously are read: a redirection (`> f`, `>> f`, `>f`), a
/// `tee` (skipping its flags), the last word of a `sed -i` (BSD and GNU put the file in
/// the same place, only the backup suffix differs), and the `*** Update File:` style
/// headers of an `apply_patch`. Everything else — a pipe into `xargs`, a glob, a path held
/// in a variable — names nothing and counts for nothing.
///
/// What it counts is "files a shell command named as output", not "source files changed":
/// a log redirected inside the repository is counted too. The authoritative answer is the
/// git diff at the close; this is the fallback for a session that left none (RM-3).
fn shell_write_targets(command: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    // `apply_patch` carries its file names inside the patch body, one header line each.
    for line in command.lines() {
        let line = line.trim_start();
        for tag in APPLY_PATCH_TAGS {
            if let Some(rest) = line.strip_prefix(tag) {
                keep_path(&mut out, rest.trim());
            }
        }
    }

    let tokens: Vec<&str> = command.split_whitespace().collect();
    let is_sed = tokens.contains(&"sed");
    for (i, token) in tokens.iter().enumerate() {
        if let Some(rest) = token.strip_prefix(">>").or_else(|| token.strip_prefix('>')) {
            // `> f` names the next word, `>f` names the rest of this one.
            if rest.is_empty() {
                if let Some(next) = tokens.get(i + 1) {
                    keep_path(&mut out, next);
                }
            } else {
                keep_path(&mut out, rest);
            }
        } else if *token == "tee" {
            let mut j = i + 1;
            while tokens.get(j).is_some_and(|t| t.starts_with('-')) {
                j += 1;
            }
            if let Some(target) = tokens.get(j) {
                keep_path(&mut out, target);
            }
        } else if is_sed && token.starts_with("-i") {
            // `sed -i '' 's/a/b/' file` (BSD) and `sed -i.bak 's/a/b/' file` (GNU) both
            // end on the file, whatever the suffix form.
            if let Some(last) = tokens.last() {
                keep_path(&mut out, last);
            }
        }
    }

    out
}

/// Pure counting over `(event_type, json)` pairs.
fn journal_of<'a>(events: impl Iterator<Item = (&'a str, &'a str)>) -> SessionJournal {
    let mut files = std::collections::HashSet::new();
    let mut user_messages = 0_i64;
    for (event_type, data) in events {
        match event_type {
            "user_message" => user_messages += 1,
            "tool_use" => {
                let Ok(crate::chat::types::ChatEvent::ToolUse { tool, input, .. }) =
                    serde_json::from_str(data)
                else {
                    continue;
                };
                if EDITING_TOOLS.contains(&tool.as_str()) {
                    // `NotebookEdit` names its target `notebook_path`; the others `file_path`.
                    if let Some(path) = ["file_path", "notebook_path"]
                        .iter()
                        .find_map(|k| input.get(*k).and_then(|v| v.as_str()))
                    {
                        files.insert(path.to_string());
                    }
                } else if tool == "Bash" {
                    // The native harness writes through the shell at least as much as
                    // through an editing tool: a heredoc, a `sed -i`, a `git apply`. Before
                    // this branch such a session counted ZERO files, whatever it wrote, and
                    // every episode of this graph carried `files_modified: 0` (RM-3).
                    if let Some(command) = input.get("command").and_then(|v| v.as_str()) {
                        files.extend(shell_write_targets(command));
                    }
                }
            }
            _ => {}
        }
    }
    SessionJournal {
        files_modified: files.len(),
        user_messages,
    }
}

/// Take the episode a chat session leaves at its close, `close_reason` being the
/// reason announced on its `session_closed`.
///
/// Read from the session node as it stands at the close:
/// - **Stimulus**: its first user message (the node's `preview`), at its creation.
///   `UserRequest`, or `Manual` for a session another agent or a runner spawned.
/// - **Process**: no FSM state; its active span (`created_at` → last update).
/// - **Outcome**: the commits the session made, read from git in the window
///   `session.created_at` → close, in `session.cwd` (`commit_shas`), and the number of
///   distinct files they touch. No note or decision is listed: nothing links what a
///   session wrote back to it yet, and an episode does not invent it. `files_modified` is
///   the larger of that file count and the distinct files an editing tool was called on
///   (see `journal_of`): one is a fact, the other an intent, and neither is complete
///   alone. How it ended is `session.close_reason`.
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
/// leaves no episode until it is resumed and closed. Two sessions running in the SAME
/// repository at the same time each see the other's commits: the window cannot tell them
/// apart, so the same sha may end up on two episodes. De-duplicating by hash when the
/// evidence is *counted* absorbs this (the reward counts distinct shas); the graph itself
/// is not corrected (RM-4, decision `c23f494b`).
pub async fn collect_session_episode(
    neo4j: &dyn GraphStore,
    session_id: Uuid,
    close_reason: &str,
) -> Result<Option<Episode>> {
    session_episode(neo4j, session_id, close_reason, AgentNode::Ensure).await
}

/// The episode of a session as it stands now, WITHOUT writing anything: the
/// on-demand snapshot (`POST /api/episodes/collect` with a `session_id`). Same
/// content as [`collect_session_episode`], but the `:Agent` node of the session's
/// model is only named, never upserted — a read is a read. `close_reason` is the
/// snapshot's (`open` / `unknown`), since the session is not closing.
pub async fn snapshot_session_episode(
    neo4j: &dyn GraphStore,
    session_id: Uuid,
    close_reason: &str,
) -> Result<Option<Episode>> {
    session_episode(neo4j, session_id, close_reason, AgentNode::ReadOnly).await
}

/// Whether the collection may write the `:Agent` node of the session's model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentNode {
    /// The close: the agent that lived the episode becomes a node (upsert).
    Ensure,
    /// A snapshot: the agent is named in the episode, the graph is not touched.
    ReadOnly,
}

async fn session_episode(
    neo4j: &dyn GraphStore,
    session_id: Uuid,
    close_reason: &str,
    agent_node: AgentNode,
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

    // The agent that lived it is a node, and the model it ran on a dated fact beside it
    // — written at the close only, never by a snapshot.
    if let (Some(id), AgentNode::Ensure) = (agent_id, agent_node) {
        if let Err(e) = neo4j.ensure_agent(id, Some(&session.model)).await {
            tracing::warn!(session_id = %session_id, error = %e, "the node of the agent of a closed session could not be made");
        }
    }

    let journal = read_session_journal(neo4j, session_id).await;

    // The commits this session produced, read from git in the SAME window the enricher
    // uses (`commits_since`, `session.created_at` in `session.cwd`) — one reader, one
    // window, so an episode cannot describe a graph it no longer matches.
    //
    // A count taken from the journal is an INTENT (a tool call named a file); a commit is
    // a FACT. Neither is complete alone — a commit does not see what was left uncommitted,
    // the journal does not see what git did. `files_modified` therefore takes the larger
    // of the two instead of pretending to add them: it is a floor on both sides.
    let commits = crate::runner::enricher::commits_since(&session.cwd, session.created_at)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(
                session_id = %session_id,
                error = %e,
                "the commits of a closed session could not be read: its episode keeps no sha"
            );
            Vec::new()
        });
    let committed_files: std::collections::HashSet<&str> = commits
        .iter()
        .flat_map(|c| c.files.iter().map(|f| f.path.as_str()))
        .collect();

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
            commit_shas: commits.iter().map(|c| c.hash.clone()).collect(),
            files_modified: journal.files_modified.max(committed_files.len()),
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
            // Raised by the persisted events, never lowered: the node's counter can lag the
            // last messages when it is read at the close.
            message_count: session.message_count.max(journal.user_messages),
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
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

    /// RM-5 — un run terminé sans aucune preuve n'est pas « vérifié », il est « non vérifié ».
    ///
    /// L'ancienne règle donnait `ImplicitPositive` avec `evidence_count = artefacts.len()`
    /// dès qu'une note existait : écrire une note était compté comme prouver un résultat.
    #[test]
    fn a_finished_run_without_a_commit_is_not_verified() {
        let v = run_validation(&[], true);
        assert_eq!(
            v.feedback_type,
            FeedbackType::None,
            "rien ne corrobore le run : None, pas un retour positif"
        );
        assert_eq!(v.evidence_count, 0);
    }

    #[test]
    fn a_finished_run_with_a_commit_is_proven() {
        let v = run_validation(&["abc1234".to_string()], true);
        assert_eq!(v.feedback_type, FeedbackType::ImplicitPositive);
        assert_eq!(v.evidence_count, 1, "un commit, une preuve");
    }

    #[test]
    fn a_run_that_did_not_finish_is_never_positive_even_with_a_commit() {
        let v = run_validation(&["abc1234".to_string()], false);
        assert_eq!(
            v.feedback_type,
            FeedbackType::None,
            "un run interrompu garde sa preuve mais ne se déclare pas réussi"
        );
        assert_eq!(v.evidence_count, 1);
    }

    /// RM-4 — un run nomme les commits de SA fenêtre, lus dans l'arbre de SON projet.
    ///
    /// Avant, `collect_run_episode` écrivait `commit_shas: Vec::new()` et
    /// `files_modified: 0` en dur : un run qui livrait du code déclarait zéro preuve, et
    /// `run_validation` n'avait donc jamais rien à compter.
    #[tokio::test]
    async fn a_run_names_the_commits_of_its_window() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let (_repo, cwd) = crate::test_helpers::git_repo_with_one_commit();

        let store = MockGraphStore::new();
        let mut project = crate::test_helpers::test_project_named("alpha");
        project.root_path = cwd;
        store.create_project(&project).await.unwrap();

        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.project_id = Some(project.id);
        // The window opens before the (backdated) commit, so the commit is inside it.
        run.started_at = Utc::now() - chrono::Duration::hours(3);
        run.complete();
        let run_id = run.id;
        store.protocol_runs.write().await.insert(run_id, run);

        let episode = collect_episode(&store, run_id, project.id)
            .await
            .unwrap()
            .expect("a completed run leaves an episode");

        assert_eq!(
            episode.outcome.commit_shas.len(),
            1,
            "the commit of the window is named: {:?}",
            episode.outcome.commit_shas
        );
        assert!(
            episode.outcome.files_modified >= 1,
            "a committed file counts as modified"
        );
        assert_eq!(
            episode.validation.evidence_count, 1,
            "the commit IS the evidence"
        );
        assert_eq!(
            episode.validation.feedback_type,
            FeedbackType::ImplicitPositive,
            "a run that delivered is a run that is proven"
        );
    }

    /// Un projet sans arbre de travail (projet `work`, ou pas encore lié à un dépôt) ne
    /// fait pas échouer la collecte : il ne nomme simplement aucun commit.
    #[tokio::test]
    async fn a_run_whose_project_has_no_working_tree_keeps_no_sha() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::models::ProtocolRun;

        let store = MockGraphStore::new();
        let mut project = crate::test_helpers::test_project_named("no-root");
        project.root_path = String::new();
        store.create_project(&project).await.unwrap();

        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "Start");
        run.project_id = Some(project.id);
        run.complete();
        let run_id = run.id;
        store.protocol_runs.write().await.insert(run_id, run);

        let episode = collect_episode(&store, run_id, project.id)
            .await
            .unwrap()
            .expect("a completed run leaves an episode");

        assert!(episode.outcome.commit_shas.is_empty());
        assert_eq!(episode.validation.evidence_count, 0);
        assert_eq!(episode.validation.feedback_type, FeedbackType::None);
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

    /// RM-4 — l'épisode nomme les commits que la session a réellement produits.
    ///
    /// Avant, `commit_shas` était écrit vide en dur et `files_modified` ne venait que des
    /// événements : une session qui livre par git déclarait zéro livraison. Ici le
    /// dépôt est réel, la fenêtre s'ouvre avant le commit, et l'épisode doit le nommer.
    #[tokio::test]
    async fn a_session_that_committed_names_its_shas_and_its_files() {
        use std::process::Command;

        let repo = tempfile::tempdir().unwrap();
        let dir = repo.path();
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git must be installed for this test");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.t"]);
        git(&["config", "user.name", "t"]);
        git(&["config", "commit.gpgsign", "false"]);
        std::fs::write(dir.join("a.rs"), "fn a() {}\n").unwrap();
        git(&["add", "."]);
        git(&["commit", "-q", "-m", "feat: a"]);

        let store = crate::neo4j::mock::MockGraphStore::new();
        let p = crate::test_helpers::test_project_named("alpha");
        store.create_project(&p).await.unwrap();
        let mut session = crate::test_helpers::test_chat_session(Some("alpha"));
        session.preview = Some("write a".to_string());
        session.cwd = dir.to_string_lossy().to_string();
        // The window opens before the commit, so the commit is inside it.
        session.created_at = Utc::now() - chrono::Duration::seconds(60);
        session.updated_at = Utc::now();
        let sid = session.id;
        store.create_chat_session(&session).await.unwrap();

        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .expect("a session with a message and a project leaves an episode");

        assert_eq!(
            episode.outcome.commit_shas.len(),
            1,
            "the commit of the window is named: {:?}",
            episode.outcome.commit_shas
        );
        assert!(
            !episode.outcome.commit_shas[0].is_empty(),
            "the sha is named, not blank"
        );
        assert!(
            episode.outcome.files_modified >= 1,
            "a committed file counts as modified, even with no editing tool call"
        );
    }

    /// Une session dont le `cwd` n'est pas un dépôt n'échoue pas et ne nomme aucun sha :
    /// un compte manquant vaut mieux qu'un close qui tombe.
    #[tokio::test]
    async fn a_session_outside_any_repository_keeps_no_sha_and_still_leaves_an_episode() {
        let (store, sid, _) = store_with_session(Some("alpha"), Some("nothing here")).await;
        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .expect("a session with a message and a project leaves an episode");
        assert!(
            episode.outcome.commit_shas.is_empty(),
            "no repository, no commit to name"
        );
    }

    #[tokio::test]
    async fn a_closed_session_gives_an_episode_of_its_first_message_and_its_end() {
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

    fn tool_use(tool: &str, input: serde_json::Value) -> String {
        serde_json::to_string(&crate::chat::types::ChatEvent::ToolUse {
            id: Uuid::new_v4().to_string(),
            tool: tool.into(),
            input,
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        })
        .unwrap()
    }

    #[test]
    fn the_journal_counts_distinct_edited_files_and_user_messages() {
        let edit_a = tool_use("Edit", serde_json::json!({"file_path": "/r/a.rs"}));
        let edit_a_again = tool_use("Edit", serde_json::json!({"file_path": "/r/a.rs"}));
        let write_b = tool_use("Write", serde_json::json!({"file_path": "/r/b.rs"}));
        let multi_c = tool_use("MultiEdit", serde_json::json!({"file_path": "/r/c.rs"}));
        let notebook = tool_use(
            "NotebookEdit",
            serde_json::json!({"notebook_path": "/r/n.ipynb"}),
        );
        let read_d = tool_use("Read", serde_json::json!({"file_path": "/r/d.rs"}));
        let no_path = tool_use("Edit", serde_json::json!({}));
        let events = [
            ("user_message", "{}"),
            ("tool_use", edit_a.as_str()),
            ("tool_use", edit_a_again.as_str()),
            ("tool_use", write_b.as_str()),
            ("tool_use", multi_c.as_str()),
            ("tool_use", notebook.as_str()),
            ("tool_use", read_d.as_str()),
            ("tool_use", no_path.as_str()),
            ("tool_use", "not json"),
            ("user_message", "{}"),
            ("assistant_text", "{}"),
        ];
        assert_eq!(
            journal_of(events.into_iter()),
            SessionJournal {
                files_modified: 4,
                user_messages: 2
            },
            "a, b, c and the notebook count once each; a read, a call without a path and a broken payload count for nothing"
        );
        assert_eq!(journal_of(std::iter::empty()), SessionJournal::default());
    }

    /// RM-3 — une session qui n'écrit QUE par le shell comptait zéro fichier.
    ///
    /// Le harnais natif écrit par le shell (heredoc, `sed -i`, `apply_patch`) au moins
    /// autant que par un outil d'édition. Avant cette branche `journal_of` ne voyait que
    /// `EDITING_TOOLS`, donc chaque épisode de ce graphe portait `files_modified: 0`.
    #[test]
    fn a_session_that_writes_only_through_the_shell_counts_its_files() {
        let heredoc = tool_use(
            "Bash",
            serde_json::json!({"command": "cat > src/new_module.rs <<'EOF'\nfn f() {}\nEOF"}),
        );
        let sed = tool_use(
            "Bash",
            serde_json::json!({"command": "sed -i '' 's/a/b/' src/collector.rs"}),
        );
        let patch = tool_use(
            "Bash",
            serde_json::json!({"command": "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: src/models.rs\n*** End Patch\nEOF"}),
        );
        let events = [
            ("user_message", "{}"),
            ("tool_use", heredoc.as_str()),
            ("tool_use", sed.as_str()),
            ("tool_use", patch.as_str()),
        ];
        assert_eq!(
            journal_of(events.into_iter()),
            SessionJournal {
                files_modified: 3,
                user_messages: 1
            },
            "three distinct files written by the shell, none by an editing tool"
        );
    }

    /// Le bruit du shell ne doit pas devenir des fichiers : un `> /dev/null`, un
    /// descripteur, une variable et un `tee` sans cible ne nomment rien.
    #[test]
    fn shell_noise_is_not_a_written_file() {
        let noisy = tool_use(
            "Bash",
            serde_json::json!({"command": "cargo test > /dev/null 2>&1 && echo ok > $OUT | tee -a"}),
        );
        let events = [("tool_use", noisy.as_str())];
        assert_eq!(
            journal_of(events.into_iter()),
            SessionJournal::default(),
            "devices, descriptors, variables and a flagless tee name no file"
        );
    }

    /// Un même fichier écrit deux fois — une fois par un outil d'édition, une fois par le
    /// shell — compte une fois.
    #[test]
    fn a_file_written_twice_counts_once_whatever_the_tool() {
        let edit = tool_use("Edit", serde_json::json!({"file_path": "src/a.rs"}));
        let shell = tool_use("Bash", serde_json::json!({"command": "echo x >> src/a.rs"}));
        let events = [("tool_use", edit.as_str()), ("tool_use", shell.as_str())];
        assert_eq!(
            journal_of(events.into_iter()),
            SessionJournal {
                files_modified: 1,
                user_messages: 0
            },
            "one file, two writes"
        );
    }

    #[tokio::test]
    async fn a_session_that_edited_three_files_leaves_an_episode_that_says_so() {
        use crate::neo4j::models::ChatEventRecord;
        let (store, sid, _) = store_with_session(Some("alpha"), Some("edit things")).await;
        // The node lags: it says 4 messages, the events hold 6 (the 9-versus-13 gap).
        let mut records = Vec::new();
        let mut push = |seq: i64, event_type: &str, data: String| {
            records.push(ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: sid,
                seq,
                event_type: event_type.into(),
                data,
                created_at: Utc::now(),
            });
        };
        for i in 0..6 {
            push(
                i * 2,
                "user_message",
                r#"{"type":"user_message","content":"m"}"#.into(),
            );
        }
        for (i, f) in ["/r/a.rs", "/r/b.rs", "/r/c.rs", "/r/a.rs"]
            .iter()
            .enumerate()
        {
            push(
                100 + i as i64,
                "tool_use",
                tool_use("Edit", serde_json::json!({ "file_path": f })),
            );
        }
        push(
            200,
            "tool_use",
            tool_use("Read", serde_json::json!({"file_path": "/r/z.rs"})),
        );
        store.store_chat_events(sid, records).await.unwrap();

        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(episode.outcome.files_modified, 3);
        assert_eq!(episode.session.unwrap().message_count, 6);
    }

    #[tokio::test]
    async fn a_session_without_edits_says_zero_and_keeps_its_own_count() {
        let (store, sid, _) = store_with_session(Some("alpha"), Some("just talking")).await;
        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(episode.outcome.files_modified, 0);
        // No events: the node's count stands, it is never lowered.
        assert_eq!(episode.session.unwrap().message_count, 4);
    }

    #[tokio::test]
    async fn the_agent_of_a_closed_session_is_a_node_and_the_backfill_rebuilds_the_rest() {
        use crate::neo4j::models::AgentBackfill;
        let (store, sid, project_id) = store_with_session(Some("alpha"), Some("work")).await;

        // Taking the episode names the agent AND makes its node, with the model as a fact.
        let episode = collect_session_episode(&store, sid, "closed")
            .await
            .unwrap()
            .unwrap();
        let named = episode.session.as_ref().unwrap().agent_id.unwrap();
        assert_eq!(
            store.agents.lock().unwrap().get(&named),
            Some(&Some(crate::test_helpers::test_chat_session(None).model))
        );

        // Another agent wrote a note and never became a node (history before this change).
        let other = Uuid::new_v4();
        let mut note = crate::test_helpers::test_note(
            project_id.unwrap(),
            crate::notes::NoteType::Tip,
            "written before agents were nodes",
        );
        note.created_by = crate::notes::provenance::actor(other);
        store.notes.write().await.insert(note.id, note);
        store.session_episodes.write().await.insert(sid, episode);

        let first = store.backfill_agents().await.unwrap();
        assert_eq!(
            first,
            AgentBackfill {
                before: 1,
                after: 2,
                from_notes: 1,
                from_episodes: 1,
                // The edges the same run drew: one note given to its author, one episode
                // given to the agent that lived it.
                authored: 1,
                lived: 1
            },
            "the older agent appears; the one already a node is not made twice"
        );
        let second = store.backfill_agents().await.unwrap();
        assert_eq!(second.before, second.after, "a second run creates nothing");
        assert_eq!(second.after, 2);
    }

    #[tokio::test]
    async fn a_spawned_session_is_not_a_user_request() {
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
        crate::neo4j::ChatStore::create_chat_session(&store, &orphan)
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
