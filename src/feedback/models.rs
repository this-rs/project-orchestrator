//! Feedback models — ExplicitFeedback, FeedbackTarget, ImplicitSignal.
//!
//! These models define the data structures for the closed-loop learning system.
//! ExplicitFeedback captures user ratings; ImplicitSignal captures system-detected events.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Feedback Target
// ============================================================================

/// What entity the feedback targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackTarget {
    /// A knowledge note (gotcha, guideline, pattern, etc.)
    Note,
    /// An architectural decision
    Decision,
    /// A plan
    Plan,
    /// A task within a plan
    Task,
    /// A commit
    Commit,
    /// A reasoning tree path
    ReasoningPath,
}

impl std::fmt::Display for FeedbackTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Note => write!(f, "note"),
            Self::Decision => write!(f, "decision"),
            Self::Plan => write!(f, "plan"),
            Self::Task => write!(f, "task"),
            Self::Commit => write!(f, "commit"),
            Self::ReasoningPath => write!(f, "reasoning_path"),
        }
    }
}

impl std::str::FromStr for FeedbackTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "note" => Ok(Self::Note),
            "decision" => Ok(Self::Decision),
            "plan" => Ok(Self::Plan),
            "task" => Ok(Self::Task),
            "commit" => Ok(Self::Commit),
            "reasoning_path" => Ok(Self::ReasoningPath),
            _ => Err(format!("Unknown feedback target: {}", s)),
        }
    }
}

// ============================================================================
// Explicit Feedback
// ============================================================================

/// Explicit feedback submitted by a user or agent via the API.
///
/// Score is normalized to [-1.0, +1.0]:
/// - +1.0 = strongly positive (this was helpful / correct)
/// -  0.0 = neutral
/// - -1.0 = strongly negative (this was wrong / harmful)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicitFeedback {
    /// Unique identifier for this feedback entry
    pub id: Uuid,
    /// What type of entity is being rated
    pub target_type: FeedbackTarget,
    /// UUID of the target entity
    pub target_id: Uuid,
    /// Score in [-1.0, +1.0]
    pub score: f64,
    /// Optional comment explaining the rating
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Who submitted this feedback (agent ID, user ID, or "system")
    pub submitted_by: String,
    /// Optional project context
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    /// Optional session context (which chat session triggered this)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    /// When this feedback was created
    pub created_at: DateTime<Utc>,
}

impl ExplicitFeedback {
    /// Create new explicit feedback with score validation.
    ///
    /// Returns None if score is outside [-1.0, +1.0].
    pub fn new(
        target_type: FeedbackTarget,
        target_id: Uuid,
        score: f64,
        submitted_by: String,
    ) -> Option<Self> {
        if !(-1.0..=1.0).contains(&score) {
            return None;
        }
        Some(Self {
            id: Uuid::new_v4(),
            target_type,
            target_id,
            score,
            comment: None,
            submitted_by,
            project_id: None,
            session_id: None,
            created_at: Utc::now(),
        })
    }

    /// Builder: set comment
    pub fn with_comment(mut self, comment: String) -> Self {
        self.comment = Some(comment);
        self
    }

    /// Builder: set project_id
    pub fn with_project(mut self, project_id: Uuid) -> Self {
        self.project_id = Some(project_id);
        self
    }

    /// Builder: set session_id
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }
}

// ============================================================================
// Feedback Stats
// ============================================================================

/// Aggregated feedback statistics for a target entity.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeedbackStats {
    /// Target entity ID
    pub target_id: Uuid,
    /// Target type
    pub target_type: String,
    /// Total number of feedback entries
    pub count: usize,
    /// Average score (EMA-weighted if enough data)
    pub avg_score: f64,
    /// Minimum score received
    pub min_score: f64,
    /// Maximum score received
    pub max_score: f64,
    /// Most recent feedback timestamp
    pub last_feedback_at: Option<DateTime<Utc>>,
}

// ============================================================================
// Implicit Signals
// ============================================================================

/// Implicit signals detected from system events.
///
/// These are auto-detected without user input and feed into the ScorePropagator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "signal_type")]
pub enum ImplicitSignal {
    /// A commit was reverted (negative signal for related notes/decisions).
    /// Detected via HeartbeatEngine or git history analysis.
    CommitReverted {
        commit_id: Uuid,
        /// The commit that performed the revert (if known)
        revert_commit_id: Option<Uuid>,
    },
    /// A test suite failed after changes (negative signal).
    TestFailed {
        /// Related task or commit ID
        related_id: Uuid,
        /// Number of failures
        failure_count: usize,
    },
    /// A task was restarted (potential negative signal — previous attempt failed).
    TaskRestarted {
        task_id: Uuid,
        /// How many times the task has been restarted
        restart_count: usize,
    },
    /// A plan completed cleanly (positive signal for all involved notes/decisions).
    PlanCompletedClean {
        plan_id: Uuid,
        /// Number of tasks in the plan
        task_count: usize,
    },
    /// A note was accessed many times in a short period (high activation).
    /// Signals the note is valuable and should be promoted.
    NoteHighActivation {
        note_id: Uuid,
        /// Number of accesses in the window
        access_count: usize,
        /// Time window in seconds
        window_secs: u64,
    },
}

impl ImplicitSignal {
    /// Get a human-readable label for this signal type.
    pub fn label(&self) -> &'static str {
        match self {
            Self::CommitReverted { .. } => "commit_reverted",
            Self::TestFailed { .. } => "test_failed",
            Self::TaskRestarted { .. } => "task_restarted",
            Self::PlanCompletedClean { .. } => "plan_completed_clean",
            Self::NoteHighActivation { .. } => "note_high_activation",
        }
    }

    /// Whether this signal is positive (true) or negative (false).
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            Self::PlanCompletedClean { .. } | Self::NoteHighActivation { .. }
        )
    }
}

// ============================================================================
// API Request/Response DTOs
// ============================================================================

/// Request body for POST /api/feedback
#[derive(Debug, Deserialize)]
pub struct CreateFeedbackRequest {
    /// Target entity type
    pub target_type: FeedbackTarget,
    /// Target entity UUID
    pub target_id: Uuid,
    /// Score in [-1.0, +1.0]
    pub score: f64,
    /// Optional comment
    #[serde(default)]
    pub comment: Option<String>,
    /// Optional project context
    #[serde(default)]
    pub project_id: Option<Uuid>,
    /// Optional session context
    #[serde(default)]
    pub session_id: Option<Uuid>,
}

/// Query params for GET /api/feedback/stats
#[derive(Debug, Deserialize)]
pub struct FeedbackStatsQuery {
    /// Filter by target type
    #[serde(default)]
    pub target_type: Option<String>,
    /// Filter by target ID
    #[serde(default)]
    pub target_id: Option<Uuid>,
    /// Filter by project
    #[serde(default)]
    pub project_id: Option<Uuid>,
}

/// Query params for GET /api/feedback
#[derive(Debug, Deserialize)]
pub struct ListFeedbackQuery {
    /// Filter by target type
    #[serde(default)]
    pub target_type: Option<String>,
    /// Filter by target ID
    #[serde(default)]
    pub target_id: Option<Uuid>,
    /// Filter by project
    #[serde(default)]
    pub project_id: Option<Uuid>,
    /// Max results (default 50)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

/// Response for feedback stats endpoint
#[derive(Debug, Serialize)]
pub struct FeedbackStatsResponse {
    pub stats: Vec<FeedbackStats>,
    /// Global summary
    pub total_feedback_count: usize,
    pub avg_score: f64,
}

/// Response for learning stats (admin MCP action)
#[derive(Debug, Serialize)]
pub struct LearningStats {
    /// Total explicit feedback entries
    pub total_explicit_feedback: usize,
    /// Total implicit signals processed
    pub total_implicit_signals: usize,
    /// Average explicit feedback score
    pub avg_explicit_score: f64,
    /// Breakdown by signal type
    pub signal_counts: std::collections::HashMap<String, usize>,
    /// Number of positive propagations
    pub positive_propagations: usize,
    /// Number of negative propagations
    pub negative_propagations: usize,
}

impl Default for LearningStats {
    fn default() -> Self {
        Self {
            total_explicit_feedback: 0,
            total_implicit_signals: 0,
            avg_explicit_score: 0.0,
            signal_counts: std::collections::HashMap::new(),
            positive_propagations: 0,
            negative_propagations: 0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_score_validation() {
        // Valid scores
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 1.0, "test".into())
                .is_some()
        );
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), -1.0, "test".into())
                .is_some()
        );
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 0.0, "test".into())
                .is_some()
        );
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 0.5, "test".into())
                .is_some()
        );

        // Invalid scores
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 1.1, "test".into())
                .is_none()
        );
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), -1.1, "test".into())
                .is_none()
        );
        assert!(
            ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 2.0, "test".into())
                .is_none()
        );
    }

    #[test]
    fn test_feedback_target_display() {
        assert_eq!(FeedbackTarget::Note.to_string(), "note");
        assert_eq!(FeedbackTarget::Decision.to_string(), "decision");
        assert_eq!(FeedbackTarget::ReasoningPath.to_string(), "reasoning_path");
    }

    #[test]
    fn test_feedback_target_parse() {
        assert_eq!(
            "note".parse::<FeedbackTarget>().unwrap(),
            FeedbackTarget::Note
        );
        assert_eq!(
            "decision".parse::<FeedbackTarget>().unwrap(),
            FeedbackTarget::Decision
        );
        assert!("unknown".parse::<FeedbackTarget>().is_err());
    }

    #[test]
    fn test_implicit_signal_polarity() {
        let positive = ImplicitSignal::PlanCompletedClean {
            plan_id: Uuid::new_v4(),
            task_count: 5,
        };
        assert!(positive.is_positive());

        let negative = ImplicitSignal::CommitReverted {
            commit_id: Uuid::new_v4(),
            revert_commit_id: None,
        };
        assert!(!negative.is_positive());

        let high_act = ImplicitSignal::NoteHighActivation {
            note_id: Uuid::new_v4(),
            access_count: 10,
            window_secs: 3600,
        };
        assert!(high_act.is_positive());
    }

    #[test]
    fn test_implicit_signal_label() {
        let sig = ImplicitSignal::TaskRestarted {
            task_id: Uuid::new_v4(),
            restart_count: 2,
        };
        assert_eq!(sig.label(), "task_restarted");
    }

    #[test]
    fn test_feedback_builder() {
        let project_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();
        let fb = ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 0.8, "agent-1".into())
            .unwrap()
            .with_comment("Great note!".into())
            .with_project(project_id)
            .with_session(session_id);

        assert_eq!(fb.score, 0.8);
        assert_eq!(fb.comment.as_deref(), Some("Great note!"));
        assert_eq!(fb.project_id, Some(project_id));
        assert_eq!(fb.session_id, Some(session_id));
    }
}
