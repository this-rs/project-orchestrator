//! HomeostasisController — auto-corrective thermostat for the knowledge graph.
//!
//! Pure arithmetic controller that reads health metrics and determines
//! corrective actions. No LLM calls — just threshold-based decisions.
//!
//! Rules:
//! - `synapse_health > 3.0` -> decay synapses (amount: 0.02, prune_threshold: 0.15)
//! - `synapse_health < 0.2` -> backfill synapses
//! - `note_density > 2.5/file` -> new notes start at energy 0.5 instead of 1.0
//!
//! Note: dead notes archival is handled exclusively by `ConsolidationCheck`
//! (via `consolidate_memory`). The homeostasis controller focuses on synapse
//! health and note density only, avoiding duplicate consolidation calls.
//!
//! Max 2 corrective actions per cycle to prevent runaway corrections.

use std::fmt;
use std::sync::Arc;

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::meilisearch::SearchStore;
use crate::neo4j::traits::GraphStore;
use crate::notes::manager::NoteManager;

/// Maximum number of corrective actions per evaluation cycle.
const MAX_ACTIONS_PER_CYCLE: usize = 2;

/// Default batch size for paginated backfill (notes per cycle).
const DEFAULT_BACKFILL_BATCH_SIZE: usize = 20;

// ============================================================================
// Metrics input
// ============================================================================

/// Health metrics consumed by the HomeostasisController.
///
/// These map to ratios from [`crate::neo4j::models::HomeostasisReport`]:
/// - `synapse_health`: SYNAPSE count per note (target: 0.2 - 3.0)
/// - `note_density`: notes per file (target: 0.3 - 2.5)
/// - `pain_score`: aggregated pain from the homeostasis report (0.0 - 1.0)
///
/// Note: `dead_notes_ratio` was removed — dead notes archival is handled
/// exclusively by `ConsolidationCheck` to avoid duplicate `consolidate_memory` calls.
#[derive(Debug, Clone)]
pub struct HomeostasisMetrics {
    /// Synapse-to-note ratio. High = too dense, low = too sparse.
    pub synapse_health: f64,
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
    /// Evaluate current metrics and return corrective actions (max 2).
    ///
    /// Rules are applied in priority order:
    /// 1. Synapse health (decay if too dense, backfill if too sparse)
    /// 2. Note density (reduce initial energy if > 2.5/file)
    ///
    /// Dead notes archival is handled by `ConsolidationCheck` (separate heartbeat check).
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

        // Rule 2: Note density throttle
        if metrics.note_density > 2.5 {
            actions.push(HomeostasisAction::ReduceInitialEnergy(0.5));
        }

        // Safety: cap at MAX_ACTIONS_PER_CYCLE to prevent runaway corrections
        actions.truncate(MAX_ACTIONS_PER_CYCLE);
        actions
    }
}

// ============================================================================
// Backfill cursor (inter-cycle pagination)
// ============================================================================

/// Cursor for paginated backfill across heartbeat cycles.
///
/// Each cycle processes at most one batch of notes. The cursor tracks
/// where to resume on the next cycle, avoiding the old pattern of
/// attempting a full backfill within a single 5s timeout.
#[derive(Debug, Clone, Default)]
pub struct BackfillCursor {
    /// Offset into the list of notes needing synapses.
    pub offset: usize,
    /// Whether the backfill has completed (all notes processed).
    pub completed: bool,
}

impl BackfillCursor {
    /// Create a fresh cursor starting at offset 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the cursor to start from the beginning.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.completed = false;
    }
}

/// Result of a paginated `execute_actions` call.
#[derive(Debug)]
pub struct ExecuteResult {
    /// Number of actions successfully executed.
    pub executed: usize,
    /// Updated backfill cursor (None if no BackfillSynapses action was run).
    pub backfill_cursor: Option<BackfillCursor>,
}

// ============================================================================
// Action executor
// ============================================================================

/// Execute a list of corrective actions against the graph store.
///
/// Each action maps to a specific GraphStore method:
/// - `DecaySynapses` → `graph.decay_synapses(amount, prune_threshold)`
/// - `BackfillSynapses` → `NoteManager::backfill_synapses` (paginated via cursor)
/// - `ReduceInitialEnergy` → logged only (informational, caller adjusts defaults)
///
/// When `search` is `Some`, BackfillSynapses processes one batch of notes
/// starting from `cursor.offset`. The returned `ExecuteResult` contains
/// the updated cursor for the next cycle.
///
/// When `None`, BackfillSynapses is logged as a recommendation only.
///
/// Returns an `ExecuteResult` with the count of executed actions and
/// the updated backfill cursor.
pub async fn execute_actions(
    graph: &Arc<dyn GraphStore>,
    search: Option<&Arc<dyn SearchStore>>,
    actions: &[HomeostasisAction],
    cursor: Option<&BackfillCursor>,
) -> Result<ExecuteResult> {
    let mut executed = 0;
    let mut new_cursor: Option<BackfillCursor> = None;

    for action in actions {
        match action {
            HomeostasisAction::DecaySynapses {
                amount,
                prune_threshold,
            } => {
                info!(
                    amount,
                    prune_threshold, "homeostasis: executing DecaySynapses"
                );
                match graph.decay_synapses(*amount, *prune_threshold).await {
                    Ok((decayed, pruned)) => {
                        debug!(decayed, pruned, "homeostasis: DecaySynapses completed");
                        executed += 1;
                    }
                    Err(e) => {
                        warn!("homeostasis: DecaySynapses failed: {}", e);
                    }
                }
            }
            HomeostasisAction::BackfillSynapses => {
                if let Some(search) = search {
                    let offset = cursor.map(|c| c.offset).unwrap_or(0);
                    info!(
                        offset,
                        batch_size = DEFAULT_BACKFILL_BATCH_SIZE,
                        "homeostasis: executing BackfillSynapses (paginated)"
                    );
                    let note_manager = NoteManager::new(Arc::clone(graph), Arc::clone(search));
                    // Use conservative parameters for heartbeat context:
                    // - batch_size from constant (paginated per cycle)
                    // - min_similarity=0.0 (auto-calibrate from existing weights)
                    // - max_neighbors=3 (top-K=3, recommended density target)
                    match note_manager
                        .backfill_synapses(DEFAULT_BACKFILL_BATCH_SIZE, 0.0, 3, None)
                        .await
                    {
                        Ok(progress) => {
                            let all_done = progress.processed + progress.errors + progress.skipped
                                >= progress.total
                                || progress.total == 0;
                            let next_offset = if all_done {
                                0
                            } else {
                                offset + progress.processed + progress.errors
                            };

                            debug!(
                                created = progress.synapses_created,
                                processed = progress.processed,
                                total = progress.total,
                                next_offset,
                                all_done,
                                "homeostasis: BackfillSynapses batch completed"
                            );

                            new_cursor = Some(BackfillCursor {
                                offset: next_offset,
                                completed: all_done,
                            });
                            executed += 1;
                        }
                        Err(e) => {
                            warn!("homeostasis: BackfillSynapses failed: {}", e);
                            // Preserve cursor position on failure — retry same offset next cycle
                            new_cursor = Some(BackfillCursor {
                                offset: cursor.map(|c| c.offset).unwrap_or(0),
                                completed: false,
                            });
                        }
                    }
                } else {
                    info!(
                        "homeostasis: BackfillSynapses recommended — \
                         no SearchStore available, skipping"
                    );
                }
            }
            HomeostasisAction::ReduceInitialEnergy(energy) => {
                info!(
                    energy,
                    "homeostasis: ReduceInitialEnergy recommended — \
                     new notes should start at this energy level"
                );
                // Informational only: the caller (e.g., NoteManager) should
                // read this and adjust its default energy for new notes.
            }
        }
    }

    Ok(ExecuteResult {
        executed,
        backfill_cursor: new_cursor,
    })
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
            note_density: 1.0,
            pain_score: 0.1,
        }
    }

    #[test]
    fn test_all_healthy_returns_no_actions() {
        let actions = HomeostasisController::evaluate(&healthy_metrics());
        assert!(
            actions.is_empty(),
            "healthy metrics should produce no actions"
        );
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
            note_density: 4.0,
            pain_score: 0.8,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        // Both rules fire: DecaySynapses + ReduceInitialEnergy
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
    fn test_max_two_actions_enforced() {
        // Verify truncation logic works if rules were ever expanded.
        let metrics = HomeostasisMetrics {
            synapse_health: 5.0,
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
    fn test_backfill_and_density_combined() {
        let metrics = HomeostasisMetrics {
            synapse_health: 0.05,
            note_density: 3.0,
            pain_score: 0.6,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], HomeostasisAction::BackfillSynapses);
        assert_eq!(actions[1], HomeostasisAction::ReduceInitialEnergy(0.5));
    }

    #[test]
    fn test_decay_and_density_combined() {
        let metrics = HomeostasisMetrics {
            synapse_health: 4.0,
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
            HomeostasisAction::ReduceInitialEnergy(0.5).to_string(),
            "ReduceInitialEnergy(0.5)"
        );
    }

    #[test]
    fn test_zero_metrics() {
        let metrics = HomeostasisMetrics {
            synapse_health: 0.0,
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
            note_density: 50.0,
            pain_score: 1.0,
        };
        let actions = HomeostasisController::evaluate(&metrics);
        // DecaySynapses + ReduceInitialEnergy = 2
        assert_eq!(actions.len(), 2);
    }

    // ========================================================================
    // BackfillCursor tests
    // ========================================================================

    #[test]
    fn test_backfill_cursor_new() {
        let cursor = BackfillCursor::new();
        assert_eq!(cursor.offset, 0);
        assert!(!cursor.completed);
    }

    #[test]
    fn test_backfill_cursor_default() {
        let cursor = BackfillCursor::default();
        assert_eq!(cursor.offset, 0);
        assert!(!cursor.completed);
    }

    #[test]
    fn test_backfill_cursor_reset() {
        let mut cursor = BackfillCursor {
            offset: 42,
            completed: true,
        };
        cursor.reset();
        assert_eq!(cursor.offset, 0);
        assert!(!cursor.completed);
    }

    // ========================================================================
    // execute_actions pagination tests (with MockGraphStore)
    // ========================================================================

    use crate::meilisearch::mock::MockSearchStore;
    use crate::neo4j::mock::MockGraphStore;

    #[tokio::test]
    async fn test_execute_decay_synapses() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let actions = vec![HomeostasisAction::DecaySynapses {
            amount: 0.02,
            prune_threshold: 0.15,
        }];
        let result = execute_actions(&graph, None, &actions, None).await.unwrap();
        assert_eq!(result.executed, 1);
        assert!(result.backfill_cursor.is_none(), "no backfill → no cursor");
    }

    #[tokio::test]
    async fn test_execute_reduce_initial_energy_is_informational() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let actions = vec![HomeostasisAction::ReduceInitialEnergy(0.5)];
        let result = execute_actions(&graph, None, &actions, None).await.unwrap();
        // ReduceInitialEnergy is informational-only, doesn't count as executed
        assert_eq!(result.executed, 0);
        assert!(result.backfill_cursor.is_none());
    }

    #[tokio::test]
    async fn test_execute_backfill_without_search_skips() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let actions = vec![HomeostasisAction::BackfillSynapses];
        // No SearchStore → backfill is skipped (logged as recommendation)
        let result = execute_actions(&graph, None, &actions, None).await.unwrap();
        assert_eq!(result.executed, 0);
        assert!(
            result.backfill_cursor.is_none(),
            "no search → no backfill → no cursor"
        );
    }

    #[tokio::test]
    async fn test_execute_backfill_with_search_returns_cursor() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let search: Arc<dyn crate::meilisearch::SearchStore> = Arc::new(MockSearchStore::new());
        let actions = vec![HomeostasisAction::BackfillSynapses];

        // MockGraphStore has no notes → backfill completes immediately with total=0
        let result = execute_actions(&graph, Some(&search), &actions, None)
            .await
            .unwrap();
        assert_eq!(result.executed, 1);
        let cursor = result.backfill_cursor.expect("backfill should return cursor");
        assert!(cursor.completed, "empty graph → backfill done immediately");
        assert_eq!(cursor.offset, 0);
    }

    #[tokio::test]
    async fn test_execute_backfill_preserves_cursor_on_no_work() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let search: Arc<dyn crate::meilisearch::SearchStore> = Arc::new(MockSearchStore::new());
        let actions = vec![HomeostasisAction::BackfillSynapses];

        // Start with a cursor at offset 10
        let cursor = BackfillCursor {
            offset: 10,
            completed: false,
        };
        let result = execute_actions(&graph, Some(&search), &actions, Some(&cursor))
            .await
            .unwrap();
        // MockGraphStore has no notes → completes
        let new_cursor = result.backfill_cursor.expect("should have cursor");
        assert!(new_cursor.completed);
    }

    #[tokio::test]
    async fn test_execute_mixed_actions_with_cursor() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let search: Arc<dyn crate::meilisearch::SearchStore> = Arc::new(MockSearchStore::new());
        let actions = vec![
            HomeostasisAction::DecaySynapses {
                amount: 0.02,
                prune_threshold: 0.15,
            },
            HomeostasisAction::BackfillSynapses,
        ];

        let result = execute_actions(&graph, Some(&search), &actions, None)
            .await
            .unwrap();
        // Decay succeeds + Backfill succeeds (empty graph)
        assert_eq!(result.executed, 2);
        assert!(result.backfill_cursor.is_some());
    }

    #[test]
    fn test_default_backfill_batch_size() {
        assert_eq!(DEFAULT_BACKFILL_BATCH_SIZE, 20);
    }
}
