//! Runner Feedback Collector — collects user signals post-run to detect patterns.
//!
//! Signals collected:
//! 1. **Episode collection**: On PlanCompleted, collect an Episode for the ProtocolRun
//! 2. **Manual actions**: Detect commits/PRs made after a run (gaps in automation)
//! 3. **Chat dissatisfaction**: Messages post-run with correction keywords
//! 4. **User overrides**: State skips or custom transitions during a run
//!
//! All observations are stored as Notes of type `observation` with tag `runner-feedback`,
//! linked to the ProtocolRun via PRODUCED_DURING.

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::episodes::collector as episode_collector;
use crate::neo4j::traits::GraphStore;
use crate::notes::models::{Note, NoteImportance, NoteType};

// ============================================================================
// Constants
// ============================================================================

/// Keywords that signal user dissatisfaction in post-run chat messages.
/// Both French and English to match the bilingual userbase.
const DISSATISFACTION_KEYWORDS: &[&str] = &[
    "manque",
    "oublié",
    "oublie",
    "il faut",
    "il fallait",
    "should have",
    "missing",
    "forgot",
    "forgotten",
    "missed",
    "didn't",
    "failed to",
    "wrong",
    "incorrect",
    "broken",
    "cassé",
    "pas bon",
    "devrait",
    "aurait dû",
];

/// Time window (in minutes) after PlanCompleted to scan for feedback messages.
const POST_RUN_FEEDBACK_WINDOW_MINUTES: i64 = 10;

// ============================================================================
// RunnerFeedbackCollector
// ============================================================================

/// Collects user signals after plan runs to detect emerging execution patterns.
///
/// The collector is stateless — it operates on the graph store and produces
/// observation notes linked to the relevant ProtocolRun.
pub struct RunnerFeedbackCollector {
    graph: Arc<dyn GraphStore>,
}

impl RunnerFeedbackCollector {
    /// Create a new feedback collector.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    // ========================================================================
    // Step 1: Episode collection on PlanCompleted
    // ========================================================================

    /// Collect an episode for a completed ProtocolRun.
    ///
    /// Called at PlanCompleted time. Assembles the full Episode (stimulus,
    /// process, outcome, validation) from the run's data in the graph.
    ///
    /// Returns the episode ID if successful, None if the run doesn't exist.
    pub async fn collect_run_episode(
        &self,
        run_id: Uuid,
        project_id: Uuid,
    ) -> Result<Option<Uuid>> {
        let episode = episode_collector::collect_episode(self.graph.as_ref(), run_id, project_id)
            .await?;

        match episode {
            Some(ep) => {
                let ep_id = ep.id;
                let outcome_str = if ep.validation.feedback_type
                    == crate::episodes::models::FeedbackType::ImplicitPositive
                {
                    "success"
                } else {
                    "neutral"
                };

                info!(
                    run_id = %run_id,
                    episode_id = %ep_id,
                    outcome = outcome_str,
                    states_visited = ep.process.states_visited.len(),
                    "Collected episode for completed run"
                );

                // Create an observation note summarizing the episode
                let content = format!(
                    "## Run Episode Collected\n\n\
                     **Run ID**: {}\n\
                     **Episode ID**: {}\n\
                     **Outcome**: {}\n\
                     **States visited**: {}\n\
                     **Notes produced**: {}\n\
                     **Decisions made**: {}\n\
                     **Duration**: {}ms\n",
                    run_id,
                    ep_id,
                    outcome_str,
                    ep.process.states_visited.join(" → "),
                    ep.outcome.note_ids.len(),
                    ep.outcome.decision_ids.len(),
                    ep.process.duration_ms.unwrap_or(0),
                );

                self.create_feedback_note(
                    project_id,
                    Some(run_id),
                    "runner-episode",
                    &content,
                    NoteImportance::Medium,
                )
                .await?;

                Ok(Some(ep_id))
            }
            None => {
                debug!(run_id = %run_id, "No run found for episode collection");
                Ok(None)
            }
        }
    }

    // ========================================================================
    // Step 2: Manual action detection post-run
    // ========================================================================

    /// Detect manual actions performed after a run completed.
    ///
    /// Scans for:
    /// - Git commits made after the run's completion that are NOT linked to the plan
    /// - PRs opened manually (detected by commit authorship heuristic)
    ///
    /// Each detected gap creates an observation note with tag `runner-feedback`.
    pub async fn detect_manual_actions(
        &self,
        run_id: Uuid,
        plan_id: Uuid,
        project_id: Uuid,
        run_completed_at: DateTime<Utc>,
        post_run_commits: Vec<ManualCommitInfo>,
    ) -> Result<Vec<Uuid>> {
        let mut created_note_ids = Vec::new();

        if post_run_commits.is_empty() {
            debug!(
                run_id = %run_id,
                "No manual commits detected after run"
            );
            return Ok(created_note_ids);
        }

        for commit in &post_run_commits {
            let content = format!(
                "## Manual Action Detected Post-Run\n\n\
                 **Run ID**: {}\n\
                 **Plan ID**: {}\n\
                 **Run completed at**: {}\n\n\
                 ### Commit Details\n\
                 - **SHA**: {}\n\
                 - **Message**: {}\n\
                 - **Author**: {}\n\
                 - **Committed at**: {}\n\
                 - **Files changed**: {}\n\n\
                 ### Gap Analysis\n\
                 This commit was made manually after the runner completed. \
                 This suggests the runner missed something that the user had to fix manually. \
                 Consider updating the protocol or adding a new step.\n",
                run_id,
                plan_id,
                run_completed_at,
                commit.sha,
                commit.message,
                commit.author,
                commit.committed_at,
                commit.files_changed.join(", "),
            );

            let note_id = self
                .create_feedback_note(
                    project_id,
                    Some(run_id),
                    "runner-manual-action",
                    &content,
                    NoteImportance::High,
                )
                .await?;

            created_note_ids.push(note_id);
        }

        info!(
            run_id = %run_id,
            manual_commits = post_run_commits.len(),
            notes_created = created_note_ids.len(),
            "Detected manual post-run actions"
        );

        Ok(created_note_ids)
    }

    // ========================================================================
    // Step 3: Chat dissatisfaction detection
    // ========================================================================

    /// Analyze post-run chat messages for dissatisfaction signals.
    ///
    /// Uses a keyword heuristic: messages within 10 minutes of PlanCompleted
    /// that contain correction/complaint keywords create observation notes.
    ///
    /// Returns the IDs of created feedback notes.
    pub async fn analyze_chat_feedback(
        &self,
        run_id: Uuid,
        project_id: Uuid,
        run_completed_at: DateTime<Utc>,
        messages: &[PostRunMessage],
    ) -> Result<Vec<Uuid>> {
        let mut created_note_ids = Vec::new();
        let window_end =
            run_completed_at + chrono::Duration::minutes(POST_RUN_FEEDBACK_WINDOW_MINUTES);

        for msg in messages {
            // Only consider messages within the feedback window
            if msg.timestamp < run_completed_at || msg.timestamp > window_end {
                continue;
            }

            // Check for dissatisfaction keywords (case-insensitive)
            let lower = msg.content.to_lowercase();
            let matched_keywords: Vec<&str> = DISSATISFACTION_KEYWORDS
                .iter()
                .filter(|kw| lower.contains(*kw))
                .copied()
                .collect();

            if matched_keywords.is_empty() {
                continue;
            }

            let content = format!(
                "## Chat Feedback Post-Run\n\n\
                 **Run ID**: {}\n\
                 **Message timestamp**: {}\n\
                 **Time after run**: {}s\n\
                 **Matched keywords**: {}\n\n\
                 ### User Message\n\
                 > {}\n\n\
                 ### Interpretation\n\
                 The user expressed dissatisfaction or requested corrections \
                 within {} minutes of run completion. Keywords matched: [{}]. \
                 This may indicate a missing step or incorrect behavior in the protocol.\n",
                run_id,
                msg.timestamp,
                (msg.timestamp - run_completed_at).num_seconds(),
                matched_keywords.join(", "),
                msg.content,
                POST_RUN_FEEDBACK_WINDOW_MINUTES,
                matched_keywords.join(", "),
            );

            let note_id = self
                .create_feedback_note(
                    project_id,
                    Some(run_id),
                    "runner-chat-feedback",
                    &content,
                    NoteImportance::High,
                )
                .await?;

            created_note_ids.push(note_id);
        }

        if !created_note_ids.is_empty() {
            info!(
                run_id = %run_id,
                feedback_notes = created_note_ids.len(),
                "Detected post-run chat dissatisfaction"
            );
        }

        Ok(created_note_ids)
    }

    // ========================================================================
    // Step 4: User override tracking
    // ========================================================================

    /// Record a user override during a run.
    ///
    /// Called when:
    /// - User manually skips a state in the protocol FSM
    /// - User fires a non-standard transition
    /// - User adds a custom step mid-run
    ///
    /// Creates an observation note with tag `runner-override`.
    pub async fn record_override(
        &self,
        run_id: Uuid,
        project_id: Uuid,
        override_info: UserOverride,
    ) -> Result<Uuid> {
        let content = match &override_info.override_type {
            OverrideType::StateSkip {
                skipped_state,
                from_state,
                to_state,
            } => {
                format!(
                    "## User Override: State Skip\n\n\
                     **Run ID**: {}\n\
                     **Override at**: {}\n\n\
                     ### Details\n\
                     - **Skipped state**: {}\n\
                     - **From state**: {}\n\
                     - **To state**: {}\n\
                     - **Reason**: {}\n\n\
                     ### Impact\n\
                     The user manually skipped state '{}' during the run. \
                     This suggests this state may be unnecessary or incorrectly placed \
                     in the protocol FSM.\n",
                    run_id,
                    override_info.timestamp,
                    skipped_state,
                    from_state,
                    to_state,
                    override_info.reason.as_deref().unwrap_or("(no reason given)"),
                    skipped_state,
                )
            }
            OverrideType::CustomTransition {
                trigger,
                from_state,
                to_state,
            } => {
                format!(
                    "## User Override: Custom Transition\n\n\
                     **Run ID**: {}\n\
                     **Override at**: {}\n\n\
                     ### Details\n\
                     - **Custom trigger**: {}\n\
                     - **From state**: {}\n\
                     - **To state**: {}\n\
                     - **Reason**: {}\n\n\
                     ### Impact\n\
                     The user fired a non-standard transition '{}'. \
                     This may indicate a missing transition in the protocol definition.\n",
                    run_id,
                    override_info.timestamp,
                    trigger,
                    from_state,
                    to_state,
                    override_info.reason.as_deref().unwrap_or("(no reason given)"),
                    trigger,
                )
            }
            OverrideType::CustomStep {
                step_description,
                inserted_after_state,
            } => {
                format!(
                    "## User Override: Custom Step Added\n\n\
                     **Run ID**: {}\n\
                     **Override at**: {}\n\n\
                     ### Details\n\
                     - **Step description**: {}\n\
                     - **Inserted after state**: {}\n\
                     - **Reason**: {}\n\n\
                     ### Impact\n\
                     The user added a custom step mid-run. \
                     This suggests the protocol is missing a step that should be formalized.\n",
                    run_id,
                    override_info.timestamp,
                    step_description,
                    inserted_after_state,
                    override_info.reason.as_deref().unwrap_or("(no reason given)"),
                )
            }
        };

        let tag = match &override_info.override_type {
            OverrideType::StateSkip { .. } => "runner-override-skip",
            OverrideType::CustomTransition { .. } => "runner-override-transition",
            OverrideType::CustomStep { .. } => "runner-override-step",
        };

        let note_id = self
            .create_feedback_note(project_id, Some(run_id), tag, &content, NoteImportance::High)
            .await?;

        info!(
            run_id = %run_id,
            override_type = ?override_info.override_type,
            note_id = %note_id,
            "Recorded user override"
        );

        Ok(note_id)
    }

    // ========================================================================
    // Internal: create observation note with runner-feedback tag
    // ========================================================================

    /// Create an observation note with runner-feedback tags and link to the run.
    async fn create_feedback_note(
        &self,
        project_id: Uuid,
        run_id: Option<Uuid>,
        specific_tag: &str,
        content: &str,
        importance: NoteImportance,
    ) -> Result<Uuid> {
        let mut note = Note::new(
            Some(project_id),
            NoteType::Observation,
            content.to_string(),
            "runner-feedback".to_string(),
        );
        note.importance = importance;
        note.tags = vec![
            "runner-feedback".to_string(),
            specific_tag.to_string(),
            "auto-generated".to_string(),
        ];

        let note_id = note.id;

        self.graph.create_note(&note).await?;

        // Link to the ProtocolRun via PRODUCED_DURING if we have a run_id
        if let Some(rid) = run_id {
            if let Err(e) = self
                .graph
                .create_produced_during("Note", note_id, rid)
                .await
            {
                warn!(
                    note_id = %note_id,
                    run_id = %rid,
                    error = %e,
                    "Failed to link feedback note to run (non-fatal)"
                );
            }
        }

        debug!(
            note_id = %note_id,
            tag = specific_tag,
            "Created runner feedback observation note"
        );

        Ok(note_id)
    }
}

// ============================================================================
// Supporting types
// ============================================================================

/// Information about a commit made manually after a run.
#[derive(Debug, Clone)]
pub struct ManualCommitInfo {
    /// Git commit SHA
    pub sha: String,
    /// Commit message
    pub message: String,
    /// Author name or email
    pub author: String,
    /// When the commit was made
    pub committed_at: DateTime<Utc>,
    /// Files changed in this commit
    pub files_changed: Vec<String>,
}

/// A chat message sent after a run completed.
#[derive(Debug, Clone)]
pub struct PostRunMessage {
    /// Message content
    pub content: String,
    /// When the message was sent
    pub timestamp: DateTime<Utc>,
    /// Session ID (if available)
    pub session_id: Option<Uuid>,
}

/// A user override action during a run.
#[derive(Debug, Clone)]
pub struct UserOverride {
    /// What type of override was performed
    pub override_type: OverrideType,
    /// When the override happened
    pub timestamp: DateTime<Utc>,
    /// Optional reason provided by the user
    pub reason: Option<String>,
}

/// Types of user overrides during a protocol run.
#[derive(Debug, Clone)]
pub enum OverrideType {
    /// User manually skipped a state
    StateSkip {
        /// The state that was skipped
        skipped_state: String,
        /// State the FSM was in before the skip
        from_state: String,
        /// State the FSM moved to after the skip
        to_state: String,
    },
    /// User fired a non-standard transition
    CustomTransition {
        /// The custom trigger name
        trigger: String,
        /// From state
        from_state: String,
        /// To state
        to_state: String,
    },
    /// User added a custom step mid-run
    CustomStep {
        /// Description of the custom step
        step_description: String,
        /// Which state it was inserted after
        inserted_after_state: String,
    },
}

/// Check if a message contains dissatisfaction keywords.
///
/// Public for use by external callers (e.g., chat pipeline).
pub fn contains_dissatisfaction(text: &str) -> Vec<&'static str> {
    let lower = text.to_lowercase();
    DISSATISFACTION_KEYWORDS
        .iter()
        .filter(|kw| lower.contains(**kw))
        .copied()
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::protocol::models::ProtocolRun;

    /// Helper: create a mock graph with a completed protocol run.
    async fn setup_run(mock: &MockGraphStore) -> (Uuid, Uuid) {
        let project_id = Uuid::new_v4();
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "init");
        run.visit_state(Uuid::new_v4(), "executing", "run_started");
        run.complete();
        let run_id = run.id;
        mock.protocol_runs.write().await.insert(run_id, run);
        (run_id, project_id)
    }

    // ========================================================================
    // Step 1: Episode collection tests
    // ========================================================================

    #[tokio::test]
    async fn test_collect_run_episode_success() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let result = collector
            .collect_run_episode(run_id, project_id)
            .await
            .unwrap();

        assert!(result.is_some(), "Should collect an episode");

        // Verify a feedback note was created
        let notes = mock.notes.read().await;
        let feedback_notes: Vec<_> = notes
            .values()
            .filter(|n| n.tags.contains(&"runner-episode".to_string()))
            .collect();
        assert_eq!(feedback_notes.len(), 1, "Should create one episode note");
        assert_eq!(feedback_notes[0].note_type, NoteType::Observation);
        assert!(feedback_notes[0]
            .tags
            .contains(&"runner-feedback".to_string()));
    }

    #[tokio::test]
    async fn test_collect_run_episode_not_found() {
        let mock = Arc::new(MockGraphStore::new());
        let collector = RunnerFeedbackCollector::new(mock as Arc<dyn GraphStore>);
        let result = collector
            .collect_run_episode(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    // ========================================================================
    // Step 2: Manual action detection tests
    // ========================================================================

    #[tokio::test]
    async fn test_detect_manual_actions() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;
        let plan_id = Uuid::new_v4();
        let now = Utc::now();

        let commits = vec![ManualCommitInfo {
            sha: "abc1234".to_string(),
            message: "fix: add missing push step".to_string(),
            author: "user@example.com".to_string(),
            committed_at: now + chrono::Duration::minutes(5),
            files_changed: vec!["src/main.rs".to_string()],
        }];

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_ids = collector
            .detect_manual_actions(run_id, plan_id, project_id, now, commits)
            .await
            .unwrap();

        assert_eq!(note_ids.len(), 1, "Should create one note per manual commit");

        let notes = mock.notes.read().await;
        let note = notes.get(&note_ids[0]).unwrap();
        assert!(note.content.contains("Manual Action Detected"));
        assert!(note.content.contains("abc1234"));
        assert!(note.tags.contains(&"runner-manual-action".to_string()));
        assert!(note.tags.contains(&"runner-feedback".to_string()));
    }

    #[tokio::test]
    async fn test_detect_no_manual_actions() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;

        let collector = RunnerFeedbackCollector::new(mock as Arc<dyn GraphStore>);
        let note_ids = collector
            .detect_manual_actions(run_id, Uuid::new_v4(), project_id, Utc::now(), vec![])
            .await
            .unwrap();

        assert!(note_ids.is_empty());
    }

    // ========================================================================
    // Step 3: Chat feedback tests
    // ========================================================================

    #[tokio::test]
    async fn test_chat_feedback_dissatisfaction_detected() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;
        let now = Utc::now();

        let messages = vec![
            PostRunMessage {
                content: "il manque le push".to_string(),
                timestamp: now + chrono::Duration::minutes(2),
                session_id: None,
            },
            PostRunMessage {
                content: "looks good!".to_string(),
                timestamp: now + chrono::Duration::minutes(3),
                session_id: None,
            },
        ];

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_ids = collector
            .analyze_chat_feedback(run_id, project_id, now, &messages)
            .await
            .unwrap();

        assert_eq!(
            note_ids.len(),
            1,
            "Should detect only the dissatisfied message"
        );

        let notes = mock.notes.read().await;
        let note = notes.get(&note_ids[0]).unwrap();
        assert!(note.content.contains("il manque le push"));
        assert!(note.content.contains("manque"));
        assert!(note.tags.contains(&"runner-chat-feedback".to_string()));
    }

    #[tokio::test]
    async fn test_chat_feedback_outside_window_ignored() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;
        let now = Utc::now();

        let messages = vec![PostRunMessage {
            content: "il manque quelque chose".to_string(),
            // 15 minutes after — outside the 10-minute window
            timestamp: now + chrono::Duration::minutes(15),
            session_id: None,
        }];

        let collector = RunnerFeedbackCollector::new(mock as Arc<dyn GraphStore>);
        let note_ids = collector
            .analyze_chat_feedback(run_id, project_id, now, &messages)
            .await
            .unwrap();

        assert!(
            note_ids.is_empty(),
            "Messages outside window should be ignored"
        );
    }

    #[tokio::test]
    async fn test_chat_feedback_english_keywords() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;
        let now = Utc::now();

        let messages = vec![PostRunMessage {
            content: "It should have included the tests".to_string(),
            timestamp: now + chrono::Duration::minutes(1),
            session_id: None,
        }];

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_ids = collector
            .analyze_chat_feedback(run_id, project_id, now, &messages)
            .await
            .unwrap();

        assert_eq!(note_ids.len(), 1, "Should detect English keywords");
    }

    // ========================================================================
    // Step 4: User override tracking tests
    // ========================================================================

    #[tokio::test]
    async fn test_record_state_skip_override() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;

        let override_info = UserOverride {
            override_type: OverrideType::StateSkip {
                skipped_state: "review".to_string(),
                from_state: "implementing".to_string(),
                to_state: "done".to_string(),
            },
            timestamp: Utc::now(),
            reason: Some("No review needed for this small change".to_string()),
        };

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_id = collector
            .record_override(run_id, project_id, override_info)
            .await
            .unwrap();

        let notes = mock.notes.read().await;
        let note = notes.get(&note_id).unwrap();
        assert!(note.content.contains("State Skip"));
        assert!(note.content.contains("review"));
        assert!(note
            .tags
            .contains(&"runner-override-skip".to_string()));
        assert!(note.tags.contains(&"runner-feedback".to_string()));
    }

    #[tokio::test]
    async fn test_record_custom_transition_override() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;

        let override_info = UserOverride {
            override_type: OverrideType::CustomTransition {
                trigger: "force_merge".to_string(),
                from_state: "reviewing".to_string(),
                to_state: "merged".to_string(),
            },
            timestamp: Utc::now(),
            reason: None,
        };

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_id = collector
            .record_override(run_id, project_id, override_info)
            .await
            .unwrap();

        let notes = mock.notes.read().await;
        let note = notes.get(&note_id).unwrap();
        assert!(note.content.contains("Custom Transition"));
        assert!(note.content.contains("force_merge"));
        assert!(note
            .tags
            .contains(&"runner-override-transition".to_string()));
    }

    #[tokio::test]
    async fn test_record_custom_step_override() {
        let mock = Arc::new(MockGraphStore::new());
        let (run_id, project_id) = setup_run(&mock).await;

        let override_info = UserOverride {
            override_type: OverrideType::CustomStep {
                step_description: "Run integration tests".to_string(),
                inserted_after_state: "implementing".to_string(),
            },
            timestamp: Utc::now(),
            reason: Some("Need integration tests before merge".to_string()),
        };

        let collector = RunnerFeedbackCollector::new(mock.clone() as Arc<dyn GraphStore>);
        let note_id = collector
            .record_override(run_id, project_id, override_info)
            .await
            .unwrap();

        let notes = mock.notes.read().await;
        let note = notes.get(&note_id).unwrap();
        assert!(note.content.contains("Custom Step Added"));
        assert!(note.content.contains("integration tests"));
        assert!(note.tags.contains(&"runner-override-step".to_string()));
    }

    // ========================================================================
    // Utility function tests
    // ========================================================================

    #[test]
    fn test_contains_dissatisfaction_french() {
        let matches = contains_dissatisfaction("il manque le push");
        assert!(matches.contains(&"manque"));
    }

    #[test]
    fn test_contains_dissatisfaction_english() {
        let matches = contains_dissatisfaction("It should have included tests");
        assert!(matches.contains(&"should have"));
    }

    #[test]
    fn test_contains_dissatisfaction_none() {
        let matches = contains_dissatisfaction("Looks great, thanks!");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_contains_dissatisfaction_case_insensitive() {
        let matches = contains_dissatisfaction("MISSING the deployment step");
        assert!(matches.contains(&"missing"));
    }

    #[test]
    fn test_contains_dissatisfaction_multiple_keywords() {
        let matches = contains_dissatisfaction("il manque des tests et c'est cassé");
        assert!(matches.contains(&"manque"));
        assert!(matches.contains(&"cassé"));
        assert_eq!(matches.len(), 2);
    }
}
