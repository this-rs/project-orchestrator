//! Signal Collector — non-blocking channel for behavioral signals.
//!
//! Uses an unbounded mpsc channel to collect `BehaviorSignal`s from various
//! parts of the system (feedback.rs, observation_detector.rs, ChatManager).
//! Collection is < 1ms per signal (just a channel send).
//!
//! The receiver side is consumed by the `ProfileAggregator` (T2.3) which
//! batches updates and persists to Neo4j every 30 seconds.

use super::signals::BehaviorSignal;
use tokio::sync::mpsc;
use tracing::debug;

// ============================================================================
// SignalCollector
// ============================================================================

/// A cloneable handle for sending behavioral signals.
///
/// Wraps an unbounded mpsc sender. Sending is non-blocking (< 1ms).
/// Clone this freely — all clones share the same channel.
#[derive(Debug, Clone)]
pub struct SignalCollector {
    tx: mpsc::UnboundedSender<BehaviorSignal>,
}

impl SignalCollector {
    /// Send a behavioral signal to the collector.
    ///
    /// This is non-blocking and never fails in practice (unbounded channel).
    /// If the receiver is dropped, the signal is silently discarded.
    pub fn send(&self, signal: BehaviorSignal) {
        let label = signal.label();
        if self.tx.send(signal).is_err() {
            debug!(
                "[signal_collector] Channel closed, discarding {} signal",
                label
            );
        }
    }
}

/// Create a new signal collector and its receiver.
///
/// Returns `(SignalCollector, SignalReceiver)`:
/// - `SignalCollector` is cloneable and used by producers (feedback, observation, etc.)
/// - `SignalReceiver` is consumed by the `ProfileAggregator`
pub fn create_signal_channel() -> (SignalCollector, SignalReceiver) {
    let (tx, rx) = mpsc::unbounded_channel();
    (SignalCollector { tx }, SignalReceiver { rx })
}

/// The receiving end of the signal channel.
///
/// Consumed by the `ProfileAggregator` to batch and process signals.
pub struct SignalReceiver {
    rx: mpsc::UnboundedReceiver<BehaviorSignal>,
}

impl SignalReceiver {
    /// Receive the next signal, blocking until one is available.
    ///
    /// Returns `None` when all senders have been dropped.
    pub async fn recv(&mut self) -> Option<BehaviorSignal> {
        self.rx.recv().await
    }

    /// Try to receive a signal without blocking.
    pub fn try_recv(&mut self) -> Option<BehaviorSignal> {
        self.rx.try_recv().ok()
    }

    /// Drain all available signals into a Vec (non-blocking).
    ///
    /// Useful for batching: call this periodically to collect all
    /// accumulated signals since the last drain.
    pub fn drain(&mut self) -> Vec<BehaviorSignal> {
        let mut signals = Vec::new();
        while let Ok(signal) = self.rx.try_recv() {
            signals.push(signal);
        }
        signals
    }
}

// ============================================================================
// Detection heuristics
// ============================================================================

/// Detect if a user message is a follow-up question.
///
/// Heuristic: short messages that start with a question word or contain "?"
/// after a non-question previous message. This is intentionally conservative
/// to avoid false positives.
pub fn is_follow_up_question(message: &str) -> bool {
    let trimmed = message.trim();

    // Must be relatively short (follow-ups are typically brief)
    if trimmed.len() > 200 {
        return false;
    }

    // Check for question marks
    if trimmed.contains('?') {
        return true;
    }

    // Check for question words at the start (EN + FR)
    let lower = trimmed.to_lowercase();
    let question_starters = [
        "why ",
        "how ",
        "what ",
        "where ",
        "when ",
        "which ",
        "who ",
        "can you ",
        "could you ",
        "do you ",
        "is there ",
        "are there ",
        "pourquoi ",
        "comment ",
        "quoi ",
        "où ",
        "quand ",
        "quel ",
        "est-ce que ",
        "peux-tu ",
        "pouvez-vous ",
    ];

    for starter in &question_starters {
        if lower.starts_with(starter) {
            return true;
        }
    }

    false
}

/// Detect if an LLM response was truncated.
///
/// Heuristic: response ends abruptly (no sentence-ending punctuation)
/// and is above a minimum length (short responses are intentionally terse).
pub fn is_response_truncated(response_text: &str) -> bool {
    let trimmed = response_text.trim();

    // Must be long enough to be a truncation (not just a short answer)
    if trimmed.len() < 500 {
        return false;
    }

    // Check if it ends without proper sentence termination
    let last_char = trimmed.chars().last().unwrap_or('.');
    let ends_properly = matches!(
        last_char,
        '.' | '!' | '?' | ')' | ']' | '`' | '"' | '\'' | ':'
    );

    // Also check for code blocks that might be truncated
    let open_blocks = trimmed.matches("```").count();
    let unclosed_code_block = !open_blocks.is_multiple_of(2);

    !ends_properly || unclosed_code_block
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_signal_channel_basic() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        collector.send(BehaviorSignal::Interaction { session_id: sid });
        collector.send(BehaviorSignal::FollowUpQuestion { session_id: sid });

        let s1 = receiver.recv().await.unwrap();
        assert_eq!(s1.label(), "interaction");

        let s2 = receiver.recv().await.unwrap();
        assert_eq!(s2.label(), "follow_up_question");
    }

    #[tokio::test]
    async fn test_signal_channel_drain() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        collector.send(BehaviorSignal::Interaction { session_id: sid });
        collector.send(BehaviorSignal::Interaction { session_id: sid });
        collector.send(BehaviorSignal::ExpertBehavior { session_id: sid });

        // Small delay to ensure sends complete
        tokio::task::yield_now().await;

        let drained = receiver.drain();
        assert_eq!(drained.len(), 3);
    }

    #[tokio::test]
    async fn test_collector_clone() {
        let (collector, mut receiver) = create_signal_channel();
        let sid = Uuid::new_v4();

        let c1 = collector.clone();
        let c2 = collector.clone();

        c1.send(BehaviorSignal::Interaction { session_id: sid });
        c2.send(BehaviorSignal::ExpertBehavior { session_id: sid });

        let s1 = receiver.recv().await.unwrap();
        let s2 = receiver.recv().await.unwrap();
        assert_eq!(s1.label(), "interaction");
        assert_eq!(s2.label(), "expert_behavior");
    }

    #[test]
    fn test_is_follow_up_question() {
        assert!(is_follow_up_question("Why did that happen?"));
        assert!(is_follow_up_question("How do I fix this?"));
        assert!(is_follow_up_question("Can you explain more?"));
        assert!(is_follow_up_question("Pourquoi cette erreur ?"));
        assert!(is_follow_up_question("Comment ça marche ?"));
        assert!(is_follow_up_question("what does this mean"));

        // Not follow-up questions
        assert!(!is_follow_up_question(
            "Please update the file src/main.rs with the new implementation."
        ));
        assert!(!is_follow_up_question("ok"));
        assert!(!is_follow_up_question(&"a".repeat(300))); // too long
    }

    #[test]
    fn test_is_response_truncated() {
        // Not truncated (ends properly)
        assert!(!is_response_truncated("Short answer."));
        assert!(!is_response_truncated(&format!("{}.", "x".repeat(600))));

        // Truncated (long, no ending punctuation)
        assert!(is_response_truncated(&format!("{}abc", "x".repeat(600))));

        // Truncated (unclosed code block)
        let with_unclosed = format!("{}```rust\nfn main() {{}}", "x".repeat(600));
        assert!(is_response_truncated(&with_unclosed));

        // Not truncated (properly closed code block)
        let with_closed = format!("{}```rust\nfn main() {{}}\n```.", "x".repeat(600));
        assert!(!is_response_truncated(&with_closed));
    }

    #[tokio::test]
    async fn test_send_after_receiver_dropped() {
        let (collector, receiver) = create_signal_channel();
        drop(receiver);

        // Should not panic — just silently discards
        collector.send(BehaviorSignal::Interaction {
            session_id: Uuid::new_v4(),
        });
    }
}
