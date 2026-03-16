//! Signal Wiring — integration points for emitting BehaviorSignals.
//!
//! These functions are called from existing processing pipelines
//! (feedback.rs, observation_detector.rs, ChatManager) to emit signals
//! to the SignalCollector without modifying existing function signatures.
//!
//! Each function is non-blocking (< 1ms) — it just sends on the channel.

use super::collector::{is_follow_up_question, is_response_truncated, SignalCollector};
use super::signals::BehaviorSignal;
use tracing::debug;
use uuid::Uuid;

// ============================================================================
// Post-response signal emission
// ============================================================================

/// Emit signals from a post-response analysis.
///
/// Called after each LLM response to detect:
/// - ResponseTruncated: response was cut off
///
/// Non-blocking, safe to call from async contexts.
pub fn emit_post_response_signals(
    collector: &SignalCollector,
    session_id: Uuid,
    response_text: &str,
) {
    // Check for truncation
    if is_response_truncated(response_text) {
        debug!("[signal_wiring] Response truncated detected, emitting signal");
        collector.send(BehaviorSignal::ResponseTruncated { session_id });
    }

    // Always emit an interaction tick
    collector.send(BehaviorSignal::Interaction { session_id });
}

// ============================================================================
// Observation feedback signals
// ============================================================================

/// Emit a signal when a user confirms a suggested note.
pub fn emit_note_confirmed(collector: &SignalCollector, session_id: Uuid, note_type: &str) {
    debug!(
        "[signal_wiring] Note confirmed: type={}, emitting signal",
        note_type
    );
    collector.send(BehaviorSignal::NoteConfirmed {
        session_id,
        note_type: note_type.to_string(),
    });
}

/// Emit a signal when a user rejects a suggested note.
pub fn emit_note_rejected(collector: &SignalCollector, session_id: Uuid, note_type: &str) {
    debug!(
        "[signal_wiring] Note rejected: type={}, emitting signal",
        note_type
    );
    collector.send(BehaviorSignal::NoteRejected {
        session_id,
        note_type: note_type.to_string(),
    });
}

// ============================================================================
// User message signal emission
// ============================================================================

/// Emit signals from user message analysis.
///
/// Called before processing a user message to detect:
/// - FollowUpQuestion: user is asking for clarification
///
/// Non-blocking, safe to call from async contexts.
pub fn emit_user_message_signals(collector: &SignalCollector, session_id: Uuid, message: &str) {
    if is_follow_up_question(message) {
        debug!("[signal_wiring] Follow-up question detected, emitting signal");
        collector.send(BehaviorSignal::FollowUpQuestion { session_id });
    }
}

// ============================================================================
// Project tracking signals
// ============================================================================

/// Emit a project switch signal when the user starts working on a project.
pub fn emit_project_switch(collector: &SignalCollector, session_id: Uuid, project_id: Uuid) {
    collector.send(BehaviorSignal::ProjectSwitch {
        session_id,
        project_id,
    });
}

/// Emit a language detection signal.
pub fn emit_language_detected(collector: &SignalCollector, session_id: Uuid, language: &str) {
    collector.send(BehaviorSignal::LanguageDetected {
        session_id,
        language: language.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::collector::create_signal_channel;

    #[tokio::test]
    async fn test_post_response_signals_truncated() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        // Long response without proper ending → truncated
        let response = format!("{}abc", "x".repeat(600));
        emit_post_response_signals(&collector, sid, &response);

        let signals = receiver.drain();
        // Should have: ResponseTruncated + Interaction
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].label(), "response_truncated");
        assert_eq!(signals[1].label(), "interaction");
    }

    #[tokio::test]
    async fn test_post_response_signals_normal() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        emit_post_response_signals(&collector, sid, "This is a normal response.");

        let signals = receiver.drain();
        // Should have only: Interaction (no truncation)
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].label(), "interaction");
    }

    #[tokio::test]
    async fn test_user_message_follow_up() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        emit_user_message_signals(&collector, sid, "Why is this happening?");

        let signals = receiver.drain();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].label(), "follow_up_question");
    }

    #[tokio::test]
    async fn test_user_message_not_question() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        emit_user_message_signals(&collector, sid, "Please update the config file.");

        let signals = receiver.drain();
        assert_eq!(signals.len(), 0);
    }

    #[tokio::test]
    async fn test_note_confirmed_rejected() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        emit_note_confirmed(&collector, sid, "gotcha");
        emit_note_rejected(&collector, sid, "pattern");

        let signals = receiver.drain();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].label(), "note_confirmed");
        assert_eq!(signals[1].label(), "note_rejected");
    }

    #[tokio::test]
    async fn test_project_switch() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();
        let pid = Uuid::new_v4();

        emit_project_switch(&collector, sid, pid);

        let signals = receiver.drain();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].label(), "project_switch");
    }
}
