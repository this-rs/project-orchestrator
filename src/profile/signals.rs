//! Behavior Signals — implicit observations used to update the UserProfile.
//!
//! Each signal variant represents an observable user behavior that can be
//! translated into a profile dimension update. Signals are sent via an
//! unbounded mpsc channel for non-blocking collection (< 1ms per signal).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// BehaviorSignal
// ============================================================================

/// An implicit behavioral signal detected during a chat session.
///
/// Signals carry just enough context to update the appropriate profile
/// dimension. They NEVER contain message content or code — only metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorSignal {
    /// The LLM response was truncated (too long for the user's preference).
    /// Suggests the user prefers lower verbosity.
    ResponseTruncated {
        /// Session where truncation was detected
        session_id: Uuid,
    },

    /// A suggested note was confirmed/accepted by the user.
    /// Signals engagement and validates the detection quality.
    NoteConfirmed {
        /// Session where the note was confirmed
        session_id: Uuid,
        /// The type of note confirmed (gotcha, pattern, guideline, etc.)
        note_type: String,
    },

    /// A suggested note was rejected by the user.
    /// Signals the suggestion was not helpful.
    NoteRejected {
        /// Session where the note was rejected
        session_id: Uuid,
        /// The type of note rejected
        note_type: String,
    },

    /// The user asked a follow-up question (detected heuristically).
    /// Suggests the previous response was not detailed enough → increase verbosity.
    FollowUpQuestion {
        /// Session where the follow-up was detected
        session_id: Uuid,
    },

    /// The user switched to a different project.
    /// Used to track WORKS_ON frequency and last_active.
    ProjectSwitch {
        /// Session ID
        session_id: Uuid,
        /// The project the user switched to
        project_id: Uuid,
    },

    /// A message was sent in a specific language.
    /// Used to track the user's preferred interaction language.
    LanguageDetected {
        /// Session ID
        session_id: Uuid,
        /// ISO 639-1 language code (e.g., "en", "fr")
        language: String,
    },

    /// The user used advanced features (complex queries, direct Neo4j, etc.).
    /// Suggests higher expertise level.
    ExpertBehavior {
        /// Session ID
        session_id: Uuid,
    },

    /// A generic interaction tick — increments the interaction counter.
    /// Sent once per user message.
    Interaction {
        /// Session ID
        session_id: Uuid,
    },
}

impl BehaviorSignal {
    /// Extract the session_id from any signal variant.
    pub fn session_id(&self) -> Uuid {
        match self {
            Self::ResponseTruncated { session_id } => *session_id,
            Self::NoteConfirmed { session_id, .. } => *session_id,
            Self::NoteRejected { session_id, .. } => *session_id,
            Self::FollowUpQuestion { session_id } => *session_id,
            Self::ProjectSwitch { session_id, .. } => *session_id,
            Self::LanguageDetected { session_id, .. } => *session_id,
            Self::ExpertBehavior { session_id } => *session_id,
            Self::Interaction { session_id } => *session_id,
        }
    }

    /// Get a short label for logging/metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ResponseTruncated { .. } => "response_truncated",
            Self::NoteConfirmed { .. } => "note_confirmed",
            Self::NoteRejected { .. } => "note_rejected",
            Self::FollowUpQuestion { .. } => "follow_up_question",
            Self::ProjectSwitch { .. } => "project_switch",
            Self::LanguageDetected { .. } => "language_detected",
            Self::ExpertBehavior { .. } => "expert_behavior",
            Self::Interaction { .. } => "interaction",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_session_id() {
        let sid = Uuid::new_v4();
        let signal = BehaviorSignal::ResponseTruncated { session_id: sid };
        assert_eq!(signal.session_id(), sid);
    }

    #[test]
    fn test_signal_label() {
        let sid = Uuid::new_v4();
        assert_eq!(
            BehaviorSignal::NoteConfirmed {
                session_id: sid,
                note_type: "gotcha".into(),
            }
            .label(),
            "note_confirmed"
        );
        assert_eq!(
            BehaviorSignal::FollowUpQuestion { session_id: sid }.label(),
            "follow_up_question"
        );
    }
}
