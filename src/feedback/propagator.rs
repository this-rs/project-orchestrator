//! ScorePropagator — propagates feedback signals to the knowledge graph.
//!
//! Uses existing GraphStore trait methods (boost_energy, reinforce_synapses,
//! decay_synapses, apply_scars) to update the neural knowledge layer.
//!
//! All propagation goes through GraphStore — no raw Cypher here.

use std::sync::Arc;

use anyhow::Result;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::notes::models::{Note, NoteImportance, NoteType};

use super::models::{ExplicitFeedback, FeedbackTarget, ImplicitSignal};

// ============================================================================
// Constants
// ============================================================================

/// Energy boost for positive feedback
const POSITIVE_ENERGY_BOOST: f64 = 0.15;
/// Synapse reinforcement for positive feedback
const POSITIVE_SYNAPSE_BOOST: f64 = 0.05;
/// Scar increment for negative feedback
const NEGATIVE_SCAR_INCREMENT: f64 = 0.15;
/// Synapse decay amount for negative feedback
const NEGATIVE_SYNAPSE_DECAY: f64 = 0.02;
/// Synapse prune threshold
const SYNAPSE_PRUNE_THRESHOLD: f64 = 0.05;
/// Scar increment for commit revert
const REVERT_SCAR_INCREMENT: f64 = 0.3;
/// Energy boost for high-activation promotion
const HIGH_ACTIVATION_ENERGY_BOOST: f64 = 0.25;
/// Synapse boost for high-activation co-firing
const HIGH_ACTIVATION_SYNAPSE_BOOST: f64 = 0.1;

// ============================================================================
// Explicit Feedback Propagation
// ============================================================================

/// Propagate an explicit feedback entry to the knowledge graph.
///
/// Positive scores → boost energy + reinforce synapses
/// Negative scores → apply scars + decay synapses
pub async fn propagate_feedback(
    graph: Arc<dyn GraphStore>,
    feedback: &ExplicitFeedback,
) -> Result<()> {
    if feedback.score > 0.0 {
        propagate_positive(
            graph,
            feedback.target_id,
            feedback.score,
            &feedback.target_type,
        )
        .await
    } else if feedback.score < 0.0 {
        propagate_negative(
            graph,
            feedback.target_id,
            feedback.score,
            &feedback.target_type,
        )
        .await
    } else {
        // Neutral feedback (0.0) — no propagation needed
        debug!(
            "[propagator] Neutral feedback for {:?}/{} — no propagation",
            feedback.target_type, feedback.target_id
        );
        Ok(())
    }
}

/// Propagate positive feedback: boost energy + reinforce synapses.
///
/// Energy boost is proportional to the score: boost = base × score.
/// Synapse reinforcement connects the target with related entities.
async fn propagate_positive(
    graph: Arc<dyn GraphStore>,
    target_id: Uuid,
    score: f64,
    target_type: &FeedbackTarget,
) -> Result<()> {
    let energy_boost = POSITIVE_ENERGY_BOOST * score;

    // Boost energy (only for Note and Decision targets — they have energy fields)
    match target_type {
        FeedbackTarget::Note | FeedbackTarget::Decision => {
            if let Err(e) = graph.boost_energy(target_id, energy_boost).await {
                debug!(
                    "[propagator] Failed to boost energy for {}: {} (entity may not exist)",
                    target_id, e
                );
            } else {
                debug!(
                    "[propagator] Boosted energy for {:?}/{} by {:.3}",
                    target_type, target_id, energy_boost
                );
            }

            // Decisions are not reinforced: reinforce_synapses only links Note nodes.
            if matches!(target_type, FeedbackTarget::Decision) {
                debug!(
                    "[propagator] Decision {} — synapse reinforcement not supported (notes only)",
                    target_id
                );
                return Ok(());
            }

            // Reinforce synapses between this note and its neighbors
            let synapse_boost = POSITIVE_SYNAPSE_BOOST * score;
            match reinforce_known_neighbors(&graph, target_id, synapse_boost).await {
                Ok(count) => {
                    debug!(
                        "[propagator] Reinforced {} synapses for {:?}/{}",
                        count, target_type, target_id
                    );
                }
                Err(e) => {
                    debug!(
                        "[propagator] Failed to reinforce synapses for {}: {}",
                        target_id, e
                    );
                }
            }
        }
        _ => {
            debug!(
                "[propagator] Positive feedback for {:?}/{} — no energy/synapse target",
                target_type, target_id
            );
        }
    }

    Ok(())
}

/// Maximum number of known neighbors reinforced around a target.
const MAX_REINFORCED_NEIGHBORS: usize = 10;

static NO_COACTIVATION_SET_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Reinforce the synapses between `note_id` and its already-known neighbors.
///
/// The co-activation set is `{note_id, neighbor}` for each neighbor returned by
/// `get_synapses` (strongest first, capped). Pairs are reinforced one by one so
/// that neighbors are not linked to each other. `reinforce_synapses` itself
/// refuses cross-project pairs. Returns the number of synapses reinforced; when
/// the target has no known neighbor nothing is reinforced and a single `warn`
/// is logged per process.
async fn reinforce_known_neighbors(
    graph: &Arc<dyn GraphStore>,
    note_id: Uuid,
    boost: f64,
) -> Result<usize> {
    let neighbors = graph.get_synapses(note_id).await?;
    if neighbors.is_empty() {
        if !NO_COACTIVATION_SET_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            warn!(
                "[propagator] no co-activation set for {} (no known synapse neighbors): \
                 synapses are not reinforced (this warning is emitted once)",
                note_id
            );
        }
        return Ok(0);
    }
    let mut total = 0usize;
    for (neighbor, _) in neighbors.into_iter().take(MAX_REINFORCED_NEIGHBORS) {
        if neighbor == note_id {
            continue;
        }
        total += graph
            .reinforce_synapses(&[note_id, neighbor], boost)
            .await?;
    }
    Ok(total)
}
/// Propagate negative feedback: apply scars + decay nearby synapses.
///
/// Scar intensity is proportional to the absolute score.
async fn propagate_negative(
    graph: Arc<dyn GraphStore>,
    target_id: Uuid,
    score: f64,
    target_type: &FeedbackTarget,
) -> Result<()> {
    let scar_increment = NEGATIVE_SCAR_INCREMENT * score.abs();

    match target_type {
        FeedbackTarget::Note | FeedbackTarget::Decision => {
            // Apply scar
            match graph.apply_scars(&[target_id], scar_increment).await {
                Ok(count) => {
                    debug!(
                        "[propagator] Applied scar ({:.3}) to {} nodes for {:?}/{}",
                        scar_increment, count, target_type, target_id
                    );
                }
                Err(e) => {
                    debug!("[propagator] Failed to apply scar for {}: {}", target_id, e);
                }
            }

            // Weaken the TARGET's synapses only (light touch — proportional
            // to the negative score). This used to decay every synapse of the
            // graph on each negative feedback about a single node.
            let decay_amount = NEGATIVE_SYNAPSE_DECAY * score.abs();
            match graph
                .weaken_node_synapses(target_id, decay_amount, SYNAPSE_PRUNE_THRESHOLD)
                .await
            {
                Ok(weakened) => {
                    debug!(
                        "[propagator] Weakened {} synapses of {} for negative feedback",
                        weakened, target_id
                    );
                }
                Err(e) => {
                    debug!(
                        "[propagator] Failed to decay synapses for {}: {}",
                        target_id, e
                    );
                }
            }
        }
        _ => {
            debug!(
                "[propagator] Negative feedback for {:?}/{} — no scar target",
                target_type, target_id
            );
        }
    }

    Ok(())
}

// ============================================================================
// Implicit Signal Propagation
// ============================================================================

/// Propagate an implicit signal to the knowledge graph.
pub async fn propagate_signal(graph: Arc<dyn GraphStore>, signal: &ImplicitSignal) -> Result<()> {
    match signal {
        ImplicitSignal::CommitReverted {
            commit_id,
            revert_commit_id,
        } => handle_revert(graph, *commit_id, *revert_commit_id).await,
        ImplicitSignal::TestFailed {
            related_id,
            failure_count,
        } => {
            // Apply light scar proportional to failure count
            let scar = (NEGATIVE_SCAR_INCREMENT * (*failure_count as f64) * 0.5).min(0.5);
            match graph.apply_scars(&[*related_id], scar).await {
                Ok(_) => debug!(
                    "[propagator] Applied test-failure scar ({:.3}) to {}",
                    scar, related_id
                ),
                Err(e) => debug!(
                    "[propagator] Failed to apply test-failure scar to {}: {}",
                    related_id, e
                ),
            }
            Ok(())
        }
        ImplicitSignal::TaskRestarted {
            task_id,
            restart_count,
        } => {
            // Each restart increases scar slightly
            let scar = (NEGATIVE_SCAR_INCREMENT * 0.5 * (*restart_count as f64)).min(0.5);
            match graph.apply_scars(&[*task_id], scar).await {
                Ok(_) => debug!(
                    "[propagator] Applied restart scar ({:.3}) to task {}",
                    scar, task_id
                ),
                Err(e) => debug!(
                    "[propagator] Failed to apply restart scar to task {}: {}",
                    task_id, e
                ),
            }
            Ok(())
        }
        ImplicitSignal::PlanCompletedClean {
            plan_id,
            task_count,
        } => {
            // Positive signal — boost energy for the plan entity
            let boost = POSITIVE_ENERGY_BOOST * (1.0 + (*task_count as f64 * 0.01)).min(1.5);
            if let Err(e) = graph.boost_energy(*plan_id, boost).await {
                debug!(
                    "[propagator] Failed to boost plan {} energy: {} (plan may not be a note)",
                    plan_id, e
                );
            }
            debug!(
                "[propagator] Plan {} completed clean ({} tasks) — boosted {:.3}",
                plan_id, task_count, boost
            );
            Ok(())
        }
        ImplicitSignal::NoteHighActivation {
            note_id,
            access_count,
            ..
        } => handle_high_activation(graph, *note_id, *access_count).await,
    }
}

// ============================================================================
// Specific Handlers
// ============================================================================

/// Handle a commit revert: apply scar (0.3) + auto-create gotcha note.
///
/// The scar penalizes the commit's related notes/decisions in future search.
/// A gotcha note is auto-created to capture the revert as institutional knowledge.
async fn handle_revert(
    graph: Arc<dyn GraphStore>,
    commit_id: Uuid,
    revert_commit_id: Option<Uuid>,
) -> Result<()> {
    // Apply scar to the reverted commit's related entities
    match graph.apply_scars(&[commit_id], REVERT_SCAR_INCREMENT).await {
        Ok(count) => {
            debug!(
                "[propagator] Applied revert scar ({:.3}) to {} entities for commit {}",
                REVERT_SCAR_INCREMENT, count, commit_id
            );
        }
        Err(e) => {
            warn!(
                "[propagator] Failed to apply revert scar for commit {}: {}",
                commit_id, e
            );
        }
    }

    // Auto-create a gotcha note about the revert
    let revert_info = match revert_commit_id {
        Some(rid) => format!("Reverted by commit {}", rid),
        None => "Revert detected (reverting commit unknown)".to_string(),
    };

    let content = format!(
        "## Commit Reverted\n\n\
         Commit `{}` was reverted. {}\n\n\
         **Action**: Investigate why this commit needed reverting. \
         Check if the approach was fundamentally flawed or if it was \
         a minor issue that can be fixed.\n\n\
         *Auto-generated by OutcomeTracker*",
        commit_id, revert_info
    );

    let mut note = Note::new(
        None, // No project context from commit alone
        NoteType::Gotcha,
        content,
        "system/outcome-tracker".to_string(),
    );
    note.importance = NoteImportance::High;
    note.tags = vec![
        "auto-generated".to_string(),
        "commit-revert".to_string(),
        "outcome-tracker".to_string(),
    ];

    match graph.create_note(&note).await {
        Ok(()) => {
            debug!(
                "[propagator] Created gotcha note {} for reverted commit {}",
                note.id, commit_id
            );
        }
        Err(e) => {
            warn!(
                "[propagator] Failed to create gotcha note for reverted commit {}: {}",
                commit_id, e
            );
        }
    }

    Ok(())
}

/// Handle high activation: promote ephemeral→consolidated + boost synapses.
///
/// When a note is accessed frequently, it's clearly valuable and should be
/// promoted in the memory hierarchy. Also boost synapses to strengthen
/// connections to frequently co-activated neighbors.
async fn handle_high_activation(
    graph: Arc<dyn GraphStore>,
    note_id: Uuid,
    access_count: usize,
) -> Result<()> {
    // Boost energy significantly
    if let Err(e) = graph
        .boost_energy(note_id, HIGH_ACTIVATION_ENERGY_BOOST)
        .await
    {
        debug!(
            "[propagator] Failed to boost energy for high-activation note {}: {}",
            note_id, e
        );
    } else {
        debug!(
            "[propagator] Boosted energy for high-activation note {} by {:.3} ({} accesses)",
            note_id, HIGH_ACTIVATION_ENERGY_BOOST, access_count
        );
    }

    // Reinforce synapses (the note is clearly valuable — strengthen its connections)
    match reinforce_known_neighbors(&graph, note_id, HIGH_ACTIVATION_SYNAPSE_BOOST).await {
        Ok(count) => {
            debug!(
                "[propagator] Reinforced {} synapses for high-activation note {}",
                count, note_id
            );
        }
        Err(e) => {
            debug!(
                "[propagator] Failed to reinforce synapses for high-activation note {}: {}",
                note_id, e
            );
        }
    }

    // Attempt to promote memory horizon (ephemeral → operational → consolidated)
    // We do this by reading the note, checking its horizon, and updating if needed.
    match graph.get_note(note_id).await {
        Ok(Some(note)) => {
            use crate::notes::models::MemoryHorizon;
            let new_horizon = match note.memory_horizon {
                MemoryHorizon::Ephemeral => Some(MemoryHorizon::Operational),
                MemoryHorizon::Operational => Some(MemoryHorizon::Consolidated),
                MemoryHorizon::Consolidated => None, // Already at max
            };

            if let Some(horizon) = new_horizon {
                debug!(
                    "[propagator] Promoting note {} from {:?} to {:?} (high activation)",
                    note_id, note.memory_horizon, horizon
                );
                // Use consolidate_memory for the promotion — it handles the transition
                // We boost energy so the consolidation logic picks it up
                if let Err(e) = graph.boost_energy(note_id, 0.1).await {
                    debug!(
                        "[propagator] Additional boost for promotion failed for {}: {}",
                        note_id, e
                    );
                }
            }
        }
        Ok(None) => {
            debug!(
                "[propagator] Note {} not found for promotion check",
                note_id
            );
        }
        Err(e) => {
            debug!(
                "[propagator] Failed to get note {} for promotion: {}",
                note_id, e
            );
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_propagate_positive_feedback() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        // Create a note to target
        let note_id = Uuid::new_v4();
        let mut note = Note::new(None, NoteType::Guideline, "Test note".into(), "test".into());
        note.id = note_id;
        note.energy = 0.5;
        mock.create_note(&note).await.unwrap();

        let feedback =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, 0.8, "test-agent".into()).unwrap();

        propagate_feedback(mock.clone(), &feedback).await.unwrap();

        // Verify energy was boosted
        let notes = mock.notes.read().await;
        let updated = notes.get(&note_id).unwrap();
        assert!(
            updated.energy > 0.5,
            "Energy should be boosted from 0.5, got {}",
            updated.energy
        );
    }

    #[tokio::test]
    async fn test_propagate_negative_feedback() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let note_id = Uuid::new_v4();
        let mut note = Note::new(None, NoteType::Gotcha, "Bad note".into(), "test".into());
        note.id = note_id;
        note.scar_intensity = 0.0;
        mock.create_note(&note).await.unwrap();

        let feedback =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, -0.8, "test-agent".into())
                .unwrap();

        propagate_feedback(mock.clone(), &feedback).await.unwrap();

        // Verify scar was applied
        let notes = mock.notes.read().await;
        let updated = notes.get(&note_id).unwrap();
        assert!(
            updated.scar_intensity > 0.0,
            "Scar should be applied, got {}",
            updated.scar_intensity
        );
    }

    #[tokio::test]
    async fn test_propagate_neutral_feedback() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let feedback = ExplicitFeedback::new(
            FeedbackTarget::Note,
            Uuid::new_v4(),
            0.0,
            "test-agent".into(),
        )
        .unwrap();

        // Should not error, just skip
        propagate_feedback(mock, &feedback).await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_revert_creates_gotcha() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let commit_id = Uuid::new_v4();
        let revert_id = Uuid::new_v4();

        handle_revert(mock.clone(), commit_id, Some(revert_id))
            .await
            .unwrap();

        // Verify gotcha note was created
        let notes = mock.notes.read().await;
        assert_eq!(notes.len(), 1, "Should have created 1 gotcha note");
        let note = notes.values().next().unwrap();
        assert_eq!(note.note_type, NoteType::Gotcha);
        assert_eq!(note.importance, NoteImportance::High);
        assert!(note.content.contains(&commit_id.to_string()));
        assert!(note.tags.contains(&"commit-revert".to_string()));
    }

    #[tokio::test]
    async fn test_handle_high_activation() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let note_id = Uuid::new_v4();
        let mut note = Note::new(
            None,
            NoteType::Pattern,
            "Frequently accessed pattern".into(),
            "test".into(),
        );
        note.id = note_id;
        note.energy = 0.3;
        mock.create_note(&note).await.unwrap();

        handle_high_activation(mock.clone(), note_id, 10)
            .await
            .unwrap();

        // Verify energy was boosted
        let notes = mock.notes.read().await;
        let updated = notes.get(&note_id).unwrap();
        assert!(
            updated.energy > 0.3,
            "Energy should be boosted from 0.3, got {}",
            updated.energy
        );
    }

    #[tokio::test]
    async fn test_propagate_signal_commit_reverted() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let commit_id = Uuid::new_v4();

        let signal = ImplicitSignal::CommitReverted {
            commit_id,
            revert_commit_id: None,
        };

        propagate_signal(mock.clone(), &signal).await.unwrap();

        // Should create a gotcha note
        let notes = mock.notes.read().await;
        assert_eq!(notes.len(), 1);
    }

    #[tokio::test]
    async fn test_propagate_signal_plan_completed() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let signal = ImplicitSignal::PlanCompletedClean {
            plan_id: Uuid::new_v4(),
            task_count: 5,
        };

        // Should not error even if plan doesn't exist as a note
        propagate_signal(mock, &signal).await.unwrap();
    }

    fn note_in(project: Option<Uuid>, title: &str) -> Note {
        let mut n = Note::new(project, NoteType::Guideline, title.into(), "test".into());
        n.energy = 0.5;
        n
    }

    #[tokio::test]
    async fn test_positive_feedback_reinforces_existing_synapse() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = Uuid::new_v4();
        let a = note_in(Some(project), "a");
        let b = note_in(Some(project), "b");
        mock.create_note(&a).await.unwrap();
        mock.create_note(&b).await.unwrap();
        mock.create_synapses(a.id, &[(b.id, 0.5)]).await.unwrap();

        let fb = ExplicitFeedback::new(FeedbackTarget::Note, a.id, 1.0, "t".into()).unwrap();
        propagate_feedback(mock.clone(), &fb).await.unwrap();

        let w = mock.get_synapses(a.id).await.unwrap();
        let w = w.iter().find(|(id, _)| *id == b.id).unwrap().1;
        assert!(w > 0.5, "synapse a-b must be reinforced, got {w}");
    }

    #[tokio::test]
    async fn test_positive_feedback_without_neighbors_reinforces_nothing() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let a = note_in(Some(Uuid::new_v4()), "a");
        mock.create_note(&a).await.unwrap();
        let fb = ExplicitFeedback::new(FeedbackTarget::Note, a.id, 1.0, "t".into()).unwrap();
        propagate_feedback(mock.clone(), &fb).await.unwrap();
        assert!(mock.get_synapses(a.id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_reinforce_synapses_requires_two_ids() {
        let mock = crate::neo4j::mock::MockGraphStore::new();
        assert!(mock
            .reinforce_synapses(&[Uuid::new_v4()], 0.1)
            .await
            .is_err());
        assert!(mock.reinforce_synapses(&[], 0.1).await.is_err());
    }

    #[tokio::test]
    async fn test_reinforce_synapses_refuses_cross_project_and_unknown_project() {
        let mock = crate::neo4j::mock::MockGraphStore::new();
        let a = note_in(Some(Uuid::new_v4()), "a");
        let b = note_in(Some(Uuid::new_v4()), "b");
        let c = note_in(None, "c");
        let ghost = Uuid::new_v4();
        for n in [&a, &b, &c] {
            mock.create_note(n).await.unwrap();
        }
        assert_eq!(
            mock.reinforce_synapses(&[a.id, b.id], 0.1).await.unwrap(),
            0
        );
        assert_eq!(
            mock.reinforce_synapses(&[a.id, c.id], 0.1).await.unwrap(),
            0
        );
        assert_eq!(
            mock.reinforce_synapses(&[a.id, ghost], 0.1).await.unwrap(),
            0
        );
        assert!(mock.get_synapses(a.id).await.unwrap().is_empty());
    }
}
