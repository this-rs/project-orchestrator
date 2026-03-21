//! # Episode → EpisodeData Adapter
//!
//! Converts the rich [`Episode`] model (from the Neo4j graph) into the simplified
//! [`EpisodeData`] format expected by [`EpisodeAnalyzer`](super::feedback::EpisodeAnalyzer).
//!
//! This adapter bridges the COLLECT phase (T1) and the ANALYZE phase (T2)
//! of the autonomous learning loop.
//!
//! ## Mapping strategy
//!
//! | EpisodeData field  | Source                                                        |
//! |--------------------|---------------------------------------------------------------|
//! | `protocol_name`    | Resolved from ProtocolRun → Protocol name (fallback: trigger) |
//! | `tech_stack`       | From episode lesson domain_tags (if available)                |
//! | `outcome`          | Mapped from validation.feedback_type + run status             |
//! | `gate_results`     | Each state visit → gate (passed = has exited_at without error)|
//! | `duration_ms`      | From process.duration_ms                                      |
//! | `retry_count`      | Count of duplicate state names in states_visited              |

use std::collections::HashMap;

use uuid::Uuid;

use crate::episodes::models::{Episode, FeedbackType};
use crate::pipeline::feedback::{EpisodeData, EpisodeOutcome, GateOutcome};

/// Convert an [`Episode`] into an [`EpisodeData`] suitable for analysis.
///
/// `protocol_name` is provided externally because resolving it requires
/// an async Neo4j lookup (ProtocolRun → Protocol). The caller should
/// resolve it before calling this function.
pub fn episode_to_data(episode: &Episode, protocol_name: &str) -> EpisodeData {
    // Map validation feedback to outcome
    let outcome = map_outcome(&episode.validation.feedback_type);

    // Extract tech_stack from lesson domain_tags if available
    let tech_stack = episode
        .lesson
        .as_ref()
        .map(|l| l.domain_tags.clone())
        .unwrap_or_default();

    // Convert state visits to gate outcomes
    // Each state in the FSM is treated as a "gate" that the run must pass through
    let gate_results = map_state_visits_to_gates(episode);

    // Count retries: duplicate state names indicate the FSM revisited a state
    let retry_count = count_retries(&episode.process.states_visited);

    // Duration
    let duration_ms = episode.process.duration_ms.unwrap_or(0) as u64;

    EpisodeData {
        episode_id: episode.id,
        protocol_name: protocol_name.to_string(),
        tech_stack,
        outcome,
        gate_results,
        duration_ms,
        retry_count,
    }
}

/// Batch-convert episodes with their resolved protocol names.
///
/// Takes `(Episode, protocol_name)` pairs and returns `EpisodeData` vec.
pub fn episodes_to_data(episodes: &[(Episode, String)]) -> Vec<EpisodeData> {
    episodes
        .iter()
        .map(|(ep, name)| episode_to_data(ep, name))
        .collect()
}

/// Map [`FeedbackType`] to [`EpisodeOutcome`].
fn map_outcome(feedback: &FeedbackType) -> EpisodeOutcome {
    match feedback {
        FeedbackType::ExplicitPositive | FeedbackType::ImplicitPositive => EpisodeOutcome::Success,
        FeedbackType::ExplicitNegative => EpisodeOutcome::Failure,
        FeedbackType::None => EpisodeOutcome::Partial,
    }
}

/// Convert state visits into gate outcomes.
///
/// Each state visit is treated as a "gate":
/// - **Passed**: the state has an `exited_at` timestamp (FSM moved on)
/// - **Failed**: the state has no `exited_at` AND it's not the last state
///   (indicates the run got stuck or failed at this state)
///
/// The last state may have no `exited_at` if it's a terminal state (success).
fn map_state_visits_to_gates(episode: &Episode) -> Vec<GateOutcome> {
    let visits = &episode.process.state_visits;
    let total = visits.len();

    visits
        .iter()
        .enumerate()
        .map(|(i, sv)| {
            let is_last = i == total.saturating_sub(1);
            let passed = sv.exited_at.is_some() || is_last;

            GateOutcome {
                gate_name: sv.state_name.clone(),
                passed,
                failure_message: if !passed {
                    Some(format!(
                        "Run stuck at state '{}' — no exit recorded",
                        sv.state_name
                    ))
                } else {
                    None
                },
            }
        })
        .collect()
}

/// Count retries by detecting duplicate state names in the visit history.
///
/// If the FSM visits state "build" 3 times, that's 2 retries.
fn count_retries(states_visited: &[String]) -> usize {
    if states_visited.is_empty() {
        return 0;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in states_visited {
        *counts.entry(s.as_str()).or_insert(0) += 1;
    }
    // Total retries = sum of (count - 1) for each state that appeared more than once
    counts.values().filter(|&&c| c > 1).map(|c| c - 1).sum()
}

/// Resolve protocol names for a batch of episodes.
///
/// Given episodes with `source_run_id`, looks up each ProtocolRun → Protocol
/// to get the protocol name. Falls back to `stimulus.request` if resolution fails.
pub async fn resolve_protocol_names(
    neo4j: &dyn crate::neo4j::GraphStore,
    episodes: &[Episode],
) -> Vec<String> {
    let mut names = Vec::with_capacity(episodes.len());

    for ep in episodes {
        let name = if let Some(run_id) = ep.source_run_id {
            resolve_single_protocol_name(neo4j, run_id).await
        } else {
            None
        };
        names.push(name.unwrap_or_else(|| ep.stimulus.request.clone()));
    }

    names
}

/// Resolve a single protocol name from a ProtocolRun UUID.
async fn resolve_single_protocol_name(
    neo4j: &dyn crate::neo4j::GraphStore,
    run_id: Uuid,
) -> Option<String> {
    let run = neo4j.get_protocol_run(run_id).await.ok()??;
    let protocol = neo4j.get_protocol(run.protocol_id).await.ok()??;
    Some(protocol.name)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::models::*;
    use chrono::Utc;

    fn sample_episode() -> Episode {
        let now = Utc::now();
        Episode {
            id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            stimulus: Stimulus {
                request: "manual".to_string(),
                trigger: StimulusTrigger::Manual,
                timestamp: now,
                context_hash: None,
            },
            process: Process {
                reasoning_tree_id: None,
                states_visited: vec![
                    "analyze".to_string(),
                    "implement".to_string(),
                    "review".to_string(),
                ],
                state_visits: vec![
                    StateVisitRecord {
                        state_name: "analyze".to_string(),
                        entered_at: now,
                        exited_at: Some(now + chrono::Duration::seconds(10)),
                        duration_ms: Some(10_000),
                        trigger: Some("start".to_string()),
                    },
                    StateVisitRecord {
                        state_name: "implement".to_string(),
                        entered_at: now + chrono::Duration::seconds(10),
                        exited_at: Some(now + chrono::Duration::seconds(60)),
                        duration_ms: Some(50_000),
                        trigger: Some("analysis_done".to_string()),
                    },
                    StateVisitRecord {
                        state_name: "review".to_string(),
                        entered_at: now + chrono::Duration::seconds(60),
                        exited_at: None, // terminal state
                        duration_ms: None,
                        trigger: Some("impl_done".to_string()),
                    },
                ],
                duration_ms: Some(65_000),
            },
            outcome: Outcome {
                note_ids: vec![Uuid::new_v4()],
                decision_ids: vec![],
                commit_shas: vec!["abc123".to_string()],
                files_modified: 3,
            },
            validation: Validation {
                feedback_type: FeedbackType::ImplicitPositive,
                score: None,
                evidence_count: 1,
            },
            lesson: Some(Lesson {
                abstract_pattern: "test pattern".to_string(),
                domain_tags: vec!["rust".to_string(), "api".to_string()],
                portability_layer: 1,
                confidence: Some(0.8),
            }),
            collected_at: now,
            source_run_id: Some(Uuid::new_v4()),
            embedding: None,
        }
    }

    #[test]
    fn converts_episode_to_data() {
        let ep = sample_episode();
        let data = episode_to_data(&ep, "code-review");

        assert_eq!(data.episode_id, ep.id);
        assert_eq!(data.protocol_name, "code-review");
        assert_eq!(data.tech_stack, vec!["rust", "api"]);
        assert_eq!(data.outcome, EpisodeOutcome::Success);
        assert_eq!(data.duration_ms, 65_000);
        assert_eq!(data.retry_count, 0); // no duplicates
        assert_eq!(data.gate_results.len(), 3);
    }

    #[test]
    fn maps_feedback_to_outcome() {
        assert_eq!(
            map_outcome(&FeedbackType::ExplicitPositive),
            EpisodeOutcome::Success
        );
        assert_eq!(
            map_outcome(&FeedbackType::ImplicitPositive),
            EpisodeOutcome::Success
        );
        assert_eq!(
            map_outcome(&FeedbackType::ExplicitNegative),
            EpisodeOutcome::Failure
        );
        assert_eq!(
            map_outcome(&FeedbackType::None),
            EpisodeOutcome::Partial
        );
    }

    #[test]
    fn gate_results_from_state_visits() {
        let ep = sample_episode();
        let gates = map_state_visits_to_gates(&ep);

        // First two states have exited_at → passed
        assert!(gates[0].passed);
        assert!(gates[1].passed);
        // Last state has no exited_at but is terminal → still passed
        assert!(gates[2].passed);
        // No failure messages
        assert!(gates.iter().all(|g| g.failure_message.is_none()));
    }

    #[test]
    fn gate_results_detects_stuck_state() {
        let now = Utc::now();
        let ep = Episode {
            process: Process {
                reasoning_tree_id: None,
                states_visited: vec!["a".into(), "b".into(), "c".into()],
                state_visits: vec![
                    StateVisitRecord {
                        state_name: "a".into(),
                        entered_at: now,
                        exited_at: Some(now + chrono::Duration::seconds(1)),
                        duration_ms: Some(1000),
                        trigger: None,
                    },
                    StateVisitRecord {
                        state_name: "b".into(),
                        entered_at: now + chrono::Duration::seconds(1),
                        exited_at: None, // stuck here — NOT the last state
                        duration_ms: None,
                        trigger: None,
                    },
                    StateVisitRecord {
                        state_name: "c".into(),
                        entered_at: now + chrono::Duration::seconds(5),
                        exited_at: None, // terminal
                        duration_ms: None,
                        trigger: None,
                    },
                ],
                duration_ms: Some(5000),
            },
            ..sample_episode()
        };

        let gates = map_state_visits_to_gates(&ep);
        assert!(gates[0].passed);
        assert!(!gates[1].passed); // stuck
        assert!(gates[1].failure_message.is_some());
        assert!(gates[2].passed); // terminal
    }

    #[test]
    fn counts_retries_correctly() {
        // No retries
        assert_eq!(
            count_retries(&["a".into(), "b".into(), "c".into()]),
            0
        );

        // One state retried once
        assert_eq!(
            count_retries(&["a".into(), "b".into(), "a".into(), "c".into()]),
            1
        );

        // Multiple retries
        assert_eq!(
            count_retries(&[
                "build".into(),
                "test".into(),
                "build".into(),
                "test".into(),
                "build".into(),
                "deploy".into()
            ]),
            3 // build: 2 retries, test: 1 retry
        );
    }

    #[test]
    fn empty_episode_no_panic() {
        let ep = Episode {
            process: Process {
                reasoning_tree_id: None,
                states_visited: vec![],
                state_visits: vec![],
                duration_ms: None,
            },
            validation: Validation {
                feedback_type: FeedbackType::None,
                score: None,
                evidence_count: 0,
            },
            lesson: None,
            ..sample_episode()
        };

        let data = episode_to_data(&ep, "empty-proto");
        assert_eq!(data.protocol_name, "empty-proto");
        assert_eq!(data.outcome, EpisodeOutcome::Partial);
        assert!(data.tech_stack.is_empty());
        assert!(data.gate_results.is_empty());
        assert_eq!(data.retry_count, 0);
        assert_eq!(data.duration_ms, 0);
    }

    #[test]
    fn batch_conversion() {
        let ep1 = sample_episode();
        let ep2 = sample_episode();

        let pairs = vec![
            (ep1, "proto-a".to_string()),
            (ep2, "proto-b".to_string()),
        ];

        let data = episodes_to_data(&pairs);
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].protocol_name, "proto-a");
        assert_eq!(data[1].protocol_name, "proto-b");
    }
}
