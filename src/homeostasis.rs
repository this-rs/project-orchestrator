//! HomeostasisController — auto-corrective thermostat for the knowledge graph.
//!
//! Pure arithmetic controller that reads health metrics and determines
//! corrective actions. No LLM calls — just threshold-based decisions.
//!
//! Rules:
//! - `synapse_health > 3.0` -> decay synapses (amount: 0.02, prune_threshold: 0.15)
//! - `synapse_health < 0.2` -> backfill synapses
//! - `dead_notes_ratio > 50%` -> archive dead notes (energy < 0.05, age > 90d)
//! - `note_density > 2.5/file` -> new notes start at energy 0.5 instead of 1.0
//!
//! Max 3 corrective actions per cycle to prevent runaway corrections.

use std::fmt;

/// Maximum number of corrective actions per evaluation cycle.
const MAX_ACTIONS_PER_CYCLE: usize = 3;

// ============================================================================
// Metrics input
// ============================================================================

/// Health metrics consumed by the HomeostasisController.
///
/// These map to ratios from [`crate::neo4j::models::HomeostasisReport`]:
/// - `synapse_health`: SYNAPSE count per note (target: 0.2 - 3.0)
/// - `dead_notes_ratio`: fraction of notes with energy < 0.05 and age > 90d
/// - `note_density`: notes per file (target: 0.3 - 2.5)
/// - `pain_score`: aggregated pain from the homeostasis report (0.0 - 1.0)
#[derive(Debug, Clone)]
pub struct HomeostasisMetrics {
    /// Synapse-to-note ratio. High = too dense, low = too sparse.
    pub synapse_health: f64,
    /// Fraction of dead notes (energy < 0.05, age > 90d). Range 0.0 - 1.0.
    pub dead_notes_ratio: f64,
    /// Notes per file. High = over-documented.
    pub note_density: f64,
    /// Aggregated pain score from the homeostasis report (0.0 - 1.0).
    pub pain_score: f64,
}

// ============================================================================
// Corrective actions
// ============================================================================

/// A corrective action determined by homeostasis evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum HomeostasisAction {
    /// Decay all synapse weights and prune weak ones.
    /// Triggered when synapse_health > 3.0 (network too dense).
    DecaySynapses {
        /// Amount to subtract from each synapse weight.
        amount: f64,
        /// Synapses below this weight are deleted.
        prune_threshold: f64,
    },
    /// Backfill missing SYNAPSE relations via embedding similarity.
    /// Triggered when synapse_health < 0.2 (network too sparse).
    BackfillSynapses,
    /// Archive dead notes (energy < 0.05, age > 90 days).
    /// Triggered when dead_notes_ratio > 50%.
    ArchiveDeadNotes,
    /// Reduce initial energy for new notes (instead of default 1.0).
    /// Triggered when note_density > 2.5 per file.
    ReduceInitialEnergy(f64),
}

impl fmt::Display for HomeostasisAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HomeostasisAction::DecaySynapses {
                amount,
                prune_threshold,
            } => write!(
                f,
                "DecaySynapses(amount={}, prune_threshold={})",
                amount, prune_threshold
            ),
            HomeostasisAction::BackfillSynapses => write!(f, "BackfillSynapses"),
            HomeostasisAction::ArchiveDeadNotes => write!(f, "ArchiveDeadNotes"),
            HomeostasisAction::ReduceInitialEnergy(e) => {
                write!(f, "ReduceInitialEnergy({})", e)
            }
        }
    }
}

// ============================================================================
// Controller
// ============================================================================

/// Controller that reads health metrics and determines corrective actions.
///
/// This is a pure-logic module: `evaluate()` takes metrics and returns actions.
/// Execution of those actions is handled separately by the caller.
pub struct HomeostasisController;

impl HomeostasisController {
    /// Evaluate current metrics and return corrective actions (max 3).
    ///
    /// Rules are applied in priority order:
    /// 1. Synapse health (decay if too dense, backfill if too sparse)
    /// 2. Dead notes ratio (archive if > 50%)
    /// 3. Note density (reduce initial energy if > 2.5/file)
    pub fn evaluate(metrics: &HomeostasisMetrics) -> Vec<HomeostasisAction> {
        let mut actions = Vec::new();

        // Rule 1: Synapse health (mutually exclusive: decay OR backfill)
        if metrics.synapse_health > 3.0 {
            actions.push(HomeostasisAction::DecaySynapses {
                amount: 0.02,
                prune_threshold: 0.15,
            });
        } else if metrics.synapse_health < 0.2 {
            actions.push(HomeostasisAction::BackfillSynapses);
        }

        // Rule 2: Dead notes cleanup
        if metrics.dead_notes_ratio > 0.5 {
            actions.push(HomeostasisAction::ArchiveDeadNotes);
        }

        // Rule 3: Note density throttle
        if metrics.note_density > 2.5 {
            actions.push(HomeostasisAction::ReduceInitialEnergy(0.5));
        }

        // Safety: cap at MAX_ACTIONS_PER_CYCLE to prevent runaway corrections
        actions.truncate(MAX_ACTIONS_PER_CYCLE);
        actions
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build metrics with defaults.
    fn healthy_metrics() -> HomeostasisMetrics {
        HomeostasisMetrics {
            synapse_health: 1.0,
            dead_notes_ratio: 0.1,
            note_density: 1.0,
            pain_score: 0.1,
        }
    }

    #[test]
    fn test_all_healthy_returns_no_actions() {
        let actions = HomeostasisController::evaluate(&healthy_metrics());
        assert!(actions.is_empty(), "healthy metrics should produce no actions");
    }

    #[test]
    fn test_high_synapse_health_triggers_decay() {
        let metrics = HomeostasisMetrics {
            synapse_health: 14.2,
            ..healthy_metrics()
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            HomeostasisAction::DecaySynapses {
                amount: 0.02,
                prune_threshold: 0.15,
            }
        );
    }

    #[test]
    fn test_low_synapse_health_triggers_backfill() {
        let metrics = HomeostasisMetrics {
            synapse_health: 0.1,
            ..healthy_metrics()
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], HomeostasisAction::BackfillSynapses);
    }

    #[test]
    fn test_synapse_at_boundary_no_action() {
        // Exactly at boundary values — should NOT trigger
        let metrics_low = HomeostasisMetrics {
            synapse_health: 0.2,
            ..healthy_metrics()
        };
        assert!(HomeostasisController::evaluate(&metrics_low).is_empty());

        let metrics_high = HomeostasisMetrics {
            synapse_health: 3.0,
            ..healthy_metrics()
        };
        assert!(HomeostasisController::evaluate(&metrics_high).is_empty());
    }

    #[test]
    fn test_dead_notes_ratio_triggers_archive() {
        let metrics = HomeostasisMetrics {
            dead_notes_ratio: 0.55,
            ..healthy_metrics()
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], HomeostasisAction::ArchiveDeadNotes);
    }

    #[test]
    fn test_dead_notes_at_boundary_no_action() {
        let metrics = HomeostasisMetrics {
            dead_notes_ratio: 0.5,
            ..healthy_metrics()
        };
        assert!(HomeostasisController::evaluate(&metrics).is_empty());
    }

    #[test]
    fn test_high_note_density_triggers_reduced_energy() {
        let metrics = HomeostasisMetrics {
            note_density: 3.0,
            ..healthy_metrics()
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], HomeostasisAction::ReduceInitialEnergy(0.5));
    }

    #[test]
    fn test_note_density_at_boundary_no_action() {
        let metrics = HomeostasisMetrics {
            note_density: 2.5,
            ..healthy_metrics()
        };
        assert!(HomeostasisController::evaluate(&metrics).is_empty());
    }

    #[test]
    fn test_multiple_conditions_all_triggered() {
        let metrics = HomeostasisMetrics {
            synapse_health: 5.0,
            dead_notes_ratio: 0.7,
            note_density: 4.0,
            pain_score: 0.8,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        // All 3 rules fire: DecaySynapses, ArchiveDeadNotes, ReduceInitialEnergy
        assert_eq!(actions.len(), 3);
        assert_eq!(
            actions[0],
            HomeostasisAction::DecaySynapses {
                amount: 0.02,
                prune_threshold: 0.15,
            }
        );
        assert_eq!(actions[1], HomeostasisAction::ArchiveDeadNotes);
        assert_eq!(actions[2], HomeostasisAction::ReduceInitialEnergy(0.5));
    }

    #[test]
    fn test_max_three_actions_enforced() {
        // Even though only 3 rules can fire (synapse is mutually exclusive),
        // verify truncation logic works if rules were ever expanded.
        let metrics = HomeostasisMetrics {
            synapse_health: 5.0,
            dead_notes_ratio: 0.7,
            note_density: 4.0,
            pain_score: 0.9,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert!(
            actions.len() <= MAX_ACTIONS_PER_CYCLE,
            "should never exceed {} actions, got {}",
            MAX_ACTIONS_PER_CYCLE,
            actions.len()
        );
    }

    #[test]
    fn test_backfill_and_archive_combined() {
        let metrics = HomeostasisMetrics {
            synapse_health: 0.05,
            dead_notes_ratio: 0.8,
            note_density: 1.0,
            pain_score: 0.6,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], HomeostasisAction::BackfillSynapses);
        assert_eq!(actions[1], HomeostasisAction::ArchiveDeadNotes);
    }

    #[test]
    fn test_decay_and_density_combined() {
        let metrics = HomeostasisMetrics {
            synapse_health: 4.0,
            dead_notes_ratio: 0.1,
            note_density: 3.5,
            pain_score: 0.3,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            HomeostasisAction::DecaySynapses {
                amount: 0.02,
                prune_threshold: 0.15,
            }
        );
        assert_eq!(actions[1], HomeostasisAction::ReduceInitialEnergy(0.5));
    }

    #[test]
    fn test_action_display() {
        let decay = HomeostasisAction::DecaySynapses {
            amount: 0.02,
            prune_threshold: 0.15,
        };
        assert_eq!(
            decay.to_string(),
            "DecaySynapses(amount=0.02, prune_threshold=0.15)"
        );

        assert_eq!(
            HomeostasisAction::BackfillSynapses.to_string(),
            "BackfillSynapses"
        );
        assert_eq!(
            HomeostasisAction::ArchiveDeadNotes.to_string(),
            "ArchiveDeadNotes"
        );
        assert_eq!(
            HomeostasisAction::ReduceInitialEnergy(0.5).to_string(),
            "ReduceInitialEnergy(0.5)"
        );
    }

    #[test]
    fn test_zero_metrics() {
        let metrics = HomeostasisMetrics {
            synapse_health: 0.0,
            dead_notes_ratio: 0.0,
            note_density: 0.0,
            pain_score: 0.0,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        // synapse_health 0.0 < 0.2 -> BackfillSynapses
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], HomeostasisAction::BackfillSynapses);
    }

    #[test]
    fn test_extreme_metrics() {
        let metrics = HomeostasisMetrics {
            synapse_health: 100.0,
            dead_notes_ratio: 1.0,
            note_density: 50.0,
            pain_score: 1.0,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 3);
    }
}
