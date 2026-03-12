//! Episodic Memory data model
//!
//! Defines the core types for episodic knowledge capture:
//!
//! - [`Episode`]: A complete cognitive episode assembled from a ProtocolRun
//! - [`PortableEpisode`]: Anonymized, portable version for cross-instance exchange
//! - Sub-structs: [`Stimulus`], [`Process`], [`Outcome`], [`Validation`], [`Lesson`]
//!
//! An Episode captures the full lifecycle of a knowledge-producing event:
//! 1. **Stimulus**: What triggered the episode (request, event, context)
//! 2. **Process**: How it was handled (reasoning tree, FSM states visited)
//! 3. **Outcome**: What it produced (notes, decisions, commits)
//! 4. **Validation**: How the outcome was evaluated (feedback, evidence)
//! 5. **Lesson**: The abstract, portable pattern extracted (optional)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Episode — local, fully-resolved version
// ============================================================================

/// A complete cognitive episode assembled from a ProtocolRun and its outcomes.
///
/// Episodes are the bridge between ephemeral execution (protocol runs,
/// reasoning trees) and persistent knowledge (notes, decisions). They answer
/// "what happened, why, and what did we learn?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: Uuid,

    /// The project this episode belongs to
    pub project_id: Uuid,

    /// What triggered this episode
    pub stimulus: Stimulus,

    /// How the episode was processed (reasoning + FSM trace)
    pub process: Process,

    /// What the episode produced (notes, decisions, commits)
    pub outcome: Outcome,

    /// How the outcome was validated (feedback, evidence)
    pub validation: Validation,

    /// The abstract lesson extracted (optional — may be added later)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lesson: Option<Lesson>,

    /// When this episode was assembled
    pub collected_at: DateTime<Utc>,

    /// The ProtocolRun that generated this episode (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_run_id: Option<Uuid>,
}

/// What triggered the episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stimulus {
    /// The original request or event that started the episode
    pub request: String,

    /// How the episode was triggered
    pub trigger: StimulusTrigger,

    /// When the stimulus occurred
    pub timestamp: DateTime<Utc>,

    /// Hash of the context at trigger time (for dedup and correlation)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_hash: Option<String>,
}

/// How an episode was triggered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StimulusTrigger {
    /// Triggered by a user request via chat/MCP
    UserRequest,
    /// Triggered by a protocol FSM transition
    ProtocolTransition,
    /// Triggered by a reasoning tree feedback (outcome=success)
    ReasoningFeedback,
    /// Triggered by a system event (webhook, schedule)
    SystemEvent,
    /// Triggered manually by the agent
    Manual,
}

/// How the episode was processed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Process {
    /// ID of the PersistedReasoningTree used (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tree_id: Option<Uuid>,

    /// Ordered list of FSM states visited during the protocol run
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub states_visited: Vec<String>,

    /// Duration of the processing phase
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
}

/// What the episode produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// IDs of notes created during this episode (via PRODUCED_DURING)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub note_ids: Vec<Uuid>,

    /// IDs of decisions made during this episode (via PRODUCED_DURING)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decision_ids: Vec<Uuid>,

    /// SHAs of commits made during this episode
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commit_shas: Vec<String>,

    /// Number of files modified
    #[serde(default)]
    pub files_modified: usize,
}

/// How the outcome was validated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validation {
    /// Type of feedback received
    pub feedback_type: FeedbackType,

    /// Validation score (0.0 - 1.0), if available
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,

    /// Number of evidence items supporting the validation
    #[serde(default)]
    pub evidence_count: usize,
}

/// Type of feedback for an episode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackType {
    /// Explicit positive feedback (reason_feedback outcome=success)
    ExplicitPositive,
    /// Explicit negative feedback
    ExplicitNegative,
    /// Implicit positive (note created, task completed)
    ImplicitPositive,
    /// No feedback received
    None,
}

impl Default for FeedbackType {
    fn default() -> Self {
        Self::None
    }
}

/// The abstract lesson extracted from an episode.
///
/// Lessons are the portable knowledge — they capture the "why" without
/// the "what" (no UUIDs, no absolute paths, no project-specific context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    /// The abstract pattern (human-readable, context-independent)
    pub abstract_pattern: String,

    /// Domain tags for routing (e.g., ["rust", "neo4j", "api-design"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain_tags: Vec<String>,

    /// Portability layer: how universal is this lesson?
    /// 1 = project-specific, 2 = language-specific, 3 = universal
    #[serde(default = "default_portability")]
    pub portability_layer: u8,

    /// Confidence in the lesson extraction (0.0 - 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

fn default_portability() -> u8 {
    2
}

// ============================================================================
// PortableEpisode — anonymized, cross-instance version
// ============================================================================

/// Anonymized, portable version of an Episode for cross-instance exchange.
///
/// All UUIDs are replaced with symbolic references (note_0, decision_0),
/// absolute paths are replaced with patterns, and timestamps are converted
/// to relative durations. A reader can understand the lesson without access
/// to the source graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableEpisode {
    /// Schema version for backward compatibility
    #[serde(default = "default_episode_schema_version")]
    pub schema_version: u32,

    /// Anonymized stimulus
    pub stimulus: PortableStimulus,

    /// Anonymized process trace
    pub process: PortableProcess,

    /// Anonymized outcomes
    pub outcome: PortableOutcome,

    /// Anonymized validation
    pub validation: PortableValidation,

    /// The lesson (already portable by nature)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lesson: Option<PortableLesson>,
}

fn default_episode_schema_version() -> u32 {
    1
}

/// Anonymized stimulus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableStimulus {
    /// The original request (may be sanitized)
    pub request: String,

    /// How it was triggered
    pub trigger: StimulusTrigger,
}

/// Anonymized process trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableProcess {
    /// Whether a reasoning tree was used
    pub had_reasoning_tree: bool,

    /// Ordered FSM states visited (names, not IDs)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub states_visited: Vec<String>,

    /// Relative duration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
}

/// Anonymized outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableOutcome {
    /// Number of notes produced (not their IDs)
    pub notes_produced: usize,

    /// Number of decisions made
    pub decisions_made: usize,

    /// Number of commits
    pub commits_made: usize,

    /// Number of files modified
    pub files_modified: usize,

    /// Symbolic references: "note_0" → summary of the note content
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub note_summaries: Vec<String>,

    /// Symbolic references: "decision_0" → summary of the decision
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decision_summaries: Vec<String>,
}

/// Anonymized validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableValidation {
    pub feedback_type: FeedbackType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default)]
    pub evidence_count: usize,
}

/// Portable lesson (same as Lesson — already context-independent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableLesson {
    pub abstract_pattern: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain_tags: Vec<String>,
    #[serde(default = "default_portability")]
    pub portability_layer: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

// ============================================================================
// Conversion: Episode → PortableEpisode
// ============================================================================

impl Episode {
    /// Convert to a portable, anonymized episode.
    ///
    /// Strips all UUIDs, replaces absolute paths with patterns,
    /// and converts timestamps to relative durations.
    pub fn to_portable(&self) -> PortableEpisode {
        PortableEpisode {
            schema_version: 1,
            stimulus: PortableStimulus {
                request: self.stimulus.request.clone(),
                trigger: self.stimulus.trigger.clone(),
            },
            process: PortableProcess {
                had_reasoning_tree: self.process.reasoning_tree_id.is_some(),
                states_visited: self.process.states_visited.clone(),
                duration_ms: self.process.duration_ms,
            },
            outcome: PortableOutcome {
                notes_produced: self.outcome.note_ids.len(),
                decisions_made: self.outcome.decision_ids.len(),
                commits_made: self.outcome.commit_shas.len(),
                files_modified: self.outcome.files_modified,
                // Summaries are populated later by the collector (requires graph access)
                note_summaries: Vec::new(),
                decision_summaries: Vec::new(),
            },
            validation: PortableValidation {
                feedback_type: self.validation.feedback_type.clone(),
                score: self.validation.score,
                evidence_count: self.validation.evidence_count,
            },
            lesson: self.lesson.as_ref().map(|l| PortableLesson {
                abstract_pattern: l.abstract_pattern.clone(),
                domain_tags: l.domain_tags.clone(),
                portability_layer: l.portability_layer,
                confidence: l.confidence,
            }),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_episode() -> Episode {
        Episode {
            id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            stimulus: Stimulus {
                request: "How to implement PRODUCED_DURING relation?".to_string(),
                trigger: StimulusTrigger::UserRequest,
                timestamp: Utc::now(),
                context_hash: Some("abc123".to_string()),
            },
            process: Process {
                reasoning_tree_id: Some(Uuid::new_v4()),
                states_visited: vec![
                    "analyze".to_string(),
                    "implement".to_string(),
                    "done".to_string(),
                ],
                duration_ms: Some(15000),
            },
            outcome: Outcome {
                note_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
                decision_ids: vec![Uuid::new_v4()],
                commit_shas: vec!["abc1234".to_string()],
                files_modified: 3,
            },
            validation: Validation {
                feedback_type: FeedbackType::ExplicitPositive,
                score: Some(0.85),
                evidence_count: 2,
            },
            lesson: Some(Lesson {
                abstract_pattern: "When adding a new graph relation, always create an index and a backfill script for existing data.".to_string(),
                domain_tags: vec!["neo4j".to_string(), "schema-migration".to_string()],
                portability_layer: 2,
                confidence: Some(0.9),
            }),
            collected_at: Utc::now(),
            source_run_id: Some(Uuid::new_v4()),
        }
    }

    #[test]
    fn test_episode_to_portable() {
        let episode = sample_episode();
        let portable = episode.to_portable();

        assert_eq!(portable.schema_version, 1);
        assert_eq!(portable.stimulus.request, episode.stimulus.request);
        assert_eq!(portable.stimulus.trigger, StimulusTrigger::UserRequest);
        assert!(portable.process.had_reasoning_tree);
        assert_eq!(portable.process.states_visited.len(), 3);
        assert_eq!(portable.outcome.notes_produced, 2);
        assert_eq!(portable.outcome.decisions_made, 1);
        assert_eq!(portable.outcome.commits_made, 1);
        assert_eq!(portable.outcome.files_modified, 3);
        assert_eq!(
            portable.validation.feedback_type,
            FeedbackType::ExplicitPositive
        );
        assert!(portable.lesson.is_some());
        let lesson = portable.lesson.unwrap();
        assert_eq!(lesson.portability_layer, 2);
        assert!(lesson.abstract_pattern.contains("graph relation"));
    }

    #[test]
    fn test_episode_without_lesson() {
        let mut episode = sample_episode();
        episode.lesson = None;
        let portable = episode.to_portable();
        assert!(portable.lesson.is_none());
    }

    #[test]
    fn test_portable_no_uuids() {
        let episode = sample_episode();
        let portable = episode.to_portable();
        let json = serde_json::to_string(&portable).unwrap();

        // Portable should not contain any UUID pattern (8-4-4-4-12 hex)
        let uuid_regex =
            regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
                .unwrap();
        assert!(
            !uuid_regex.is_match(&json),
            "PortableEpisode JSON should not contain UUIDs: {}",
            json
        );
    }

    #[test]
    fn test_serde_round_trip() {
        let episode = sample_episode();
        let json = serde_json::to_string_pretty(&episode).unwrap();
        let restored: Episode = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, episode.id);
        assert_eq!(restored.stimulus.request, episode.stimulus.request);
        assert_eq!(restored.outcome.note_ids.len(), 2);
    }

    #[test]
    fn test_portable_serde_round_trip() {
        let episode = sample_episode();
        let portable = episode.to_portable();
        let json = serde_json::to_string_pretty(&portable).unwrap();
        let restored: PortableEpisode = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.outcome.notes_produced, 2);
        assert_eq!(restored.process.states_visited.len(), 3);
    }

    #[test]
    fn test_feedback_type_default() {
        let ft: FeedbackType = Default::default();
        assert_eq!(ft, FeedbackType::None);
    }

    #[test]
    fn test_portability_layer_default() {
        let lesson = PortableLesson {
            abstract_pattern: "test".to_string(),
            domain_tags: vec![],
            portability_layer: default_portability(),
            confidence: None,
        };
        assert_eq!(lesson.portability_layer, 2);
    }
}
