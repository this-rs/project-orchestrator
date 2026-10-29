//! S2 slice: building one [`Observation`] per turn from a task signature.
//!
//! The segmenter (S1) only decides, from fingerprints, when a conversation
//! changes direction. This slice extracts what a turn looks like on each
//! [`Dimension`] from the offline [`TaskSignature`] and the topic labels of the
//! message, following a pinned vocabulary (v1):
//!
//! * **Harness** — the provider id serving the turn.
//! * **Topic** — the message's topic labels (empty means the axis is absent).
//! * **Problem** — the task class key, plus `"retry"` on any attempt after the
//!   first.
//! * **Resolution** — the task class key (how the problem is being solved).
//!
//! **Understanding is intentionally not fed.** The first cut pinned it to
//! `"understood"` on the first attempt and `"uncertain"` after: a binary latch
//! that adds nothing over the attempt count (already visible as `Problem`'s
//! `"retry"` label) and, in the chat path, was constant (`from_chat_request`
//! always sets `attempt = 1`, so it could never fire). There is no reliable,
//! offline signal of "how well the situation is understood" yet, so rather than
//! emit a degenerate axis that never changes, this slice leaves
//! `Dimension::Understanding` out of the live observation. The variant stays in
//! `segment.rs` (and the default policy keeps a threshold for it) for the day a
//! real signal exists.
//!
//! Nothing here reads a message body, calls a model or touches the network: the
//! labels come straight from the signature and the topic slice. Keeping this
//! extraction separate from the segment decision is what makes the vocabulary
//! testable and the segmentation replayable.
//!
//! # Arbitration with the existing per-class latch
//!
//! `agent_hooks.rs` already keeps a per-class anti-oscillation latch
//! (`held_by_class`, `last_change_class`) and `moved_in`/`AFTER_PROVIDER_MOVE`
//! bound provider moves. These segments do NOT replace them: the segmenter adds a
//! conversation-scale, multi-axis signal on top, and only RELAXES the move guard
//! on a confirmed boundary. `Topic` comes from [`super::topic::topic_labels`] on
//! the turn's message text; `Problem`/`Resolution` come from `TaskSignature`'s
//! class key. (Decision recorded 2026-10, S2: keep both hystereses; do not wire a
//! second independent latch.)
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::BTreeMap;

use super::segment::{Dimension, Fingerprint, Observation};
use super::signature::TaskSignature;

/// Turns one turn of a conversation into an [`Observation`], applying the
/// pinned vocabulary. Only non-empty fingerprints are recorded: an axis with
/// nothing to say is absent rather than present-but-empty.
pub fn observation(
    turn: u32,
    provider_id: &str,
    signature: &TaskSignature,
    topic: &[String],
) -> Observation {
    let class_key = signature.class.key();
    let mut prints = BTreeMap::new();

    let harness = Fingerprint::new([provider_id]);
    if !harness.is_empty() {
        prints.insert(Dimension::Harness, harness);
    }

    let topic_print = Fingerprint::new(topic.iter().map(String::as_str));
    if !topic_print.is_empty() {
        prints.insert(Dimension::Topic, topic_print);
    }

    let mut problem_labels = vec![class_key.clone()];
    if signature.attempt > 1 {
        problem_labels.push("retry".to_owned());
    }
    let problem = Fingerprint::new(problem_labels);
    if !problem.is_empty() {
        prints.insert(Dimension::Problem, problem);
    }

    let resolution = Fingerprint::new([class_key.as_str()]);
    if !resolution.is_empty() {
        prints.insert(Dimension::Resolution, resolution);
    }

    Observation { turn, prints }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::resolver::Role;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn signature(attempt: u32) -> TaskSignature {
        TaskSignature {
            role: Role::Executor,
            class: crate::chat::provider::cognitive::signature::TaskClass::Chat(
                crate::chat::provider::cognitive::signature::ChatIntent::Debug,
            ),
            complexity: 1,
            context_need_tokens: 8_000,
            needs_tools: true,
            needs_images: false,
            budget_remaining_usd: None,
            attempt,
            project_slug: None,
            persona: None,
        }
    }

    fn labels(print: &Fingerprint) -> Vec<&str> {
        print.labels().collect()
    }

    #[test]
    fn the_live_axes_are_populated_for_a_chat_turn() {
        let observation = observation(
            7,
            "deepseek",
            &signature(1),
            &["Relay".to_owned(), "provider".to_owned()],
        );
        assert_eq!(observation.turn, 7);
        assert_eq!(observation.prints.len(), 4);
        assert_eq!(
            labels(observation.prints.get(&Dimension::Harness).unwrap()),
            ["deepseek"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Topic).unwrap()),
            ["provider", "relay"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Problem).unwrap()),
            ["chat.debug"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Resolution).unwrap()),
            ["chat.debug"]
        );
    }

    #[test]
    fn a_retry_adds_the_retry_label() {
        let observation = observation(3, "deepseek", &signature(2), &[]);
        assert_eq!(
            labels(observation.prints.get(&Dimension::Problem).unwrap()),
            ["chat.debug", "retry"]
        );
        // The resolution keeps only the class key: a retry does not change how it
        // is solved, only what problem it is.
        assert_eq!(
            labels(observation.prints.get(&Dimension::Resolution).unwrap()),
            ["chat.debug"]
        );
    }

    #[test]
    fn understanding_is_not_fed_into_the_live_observation() {
        // The axis is deliberately absent: the only signal we have (the attempt
        // count) is a binary latch already carried by Problem's "retry" label and
        // constant in the chat path. Feeding a degenerate axis would only add noise.
        for attempt in [1, 2, 3] {
            let observation = observation(0, "a", &signature(attempt), &[]);
            assert!(
                !observation.prints.contains_key(&Dimension::Understanding),
                "Understanding must stay out of the live vocabulary (attempt={attempt})"
            );
        }
    }

    #[test]
    fn an_empty_topic_omits_the_topic_axis() {
        let observation = observation(0, "a", &signature(1), &[]);
        assert!(!observation.prints.contains_key(&Dimension::Topic));
    }

    #[test]
    fn labels_are_normalized_and_stable() {
        let observation = observation(
            0,
            " DeepSeek ",
            &signature(1),
            &["  Relay ".to_owned(), "RELAY".to_owned(), "".to_owned()],
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Harness).unwrap()),
            ["deepseek"]
        );
        // Empty and duplicate labels are dropped; the rest are trimmed and
        // lower-cased, and the fingerprint keeps them in sorted order.
        assert_eq!(
            labels(observation.prints.get(&Dimension::Topic).unwrap()),
            ["relay"]
        );
    }
}
