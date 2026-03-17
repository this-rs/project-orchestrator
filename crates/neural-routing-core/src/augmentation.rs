//! Batch data augmentation pipeline — iterates over real trajectories,
//! generates MCTS alternatives, and persists them with `source: simulated` and weight 0.3.

use std::sync::Arc;

use crate::error::Result;
use crate::mcts::{MctsConfig, MctsEngine};
use crate::models::TrajectoryFilter;
use crate::proxy_model::GdsHeuristicProxy;
use crate::store::Neo4jTrajectoryStore;
use crate::traits::TrajectoryStore;

/// Configuration for the augmentation batch job.
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Maximum ratio of simulated to real trajectories.
    pub max_sim_ratio: f64,
    /// Weight assigned to simulated trajectories (used during training).
    pub sim_weight: f64,
    /// MCTS configuration.
    pub mcts: MctsConfig,
    /// Batch size for reading source trajectories.
    pub batch_size: usize,
    /// Progress callback interval (log every N trajectories processed).
    pub log_interval: usize,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            max_sim_ratio: 10.0,
            sim_weight: 0.3,
            mcts: MctsConfig::default(),
            batch_size: 50,
            log_interval: 100,
        }
    }
}

/// Result of an augmentation run.
#[derive(Debug, Clone)]
pub struct AugmentationReport {
    /// Number of source (real) trajectories processed.
    pub source_count: usize,
    /// Number of simulated trajectories generated.
    pub generated_count: usize,
    /// Number of trajectories successfully persisted.
    pub persisted_count: usize,
    /// Number of trajectories skipped (ratio limit hit).
    pub skipped_count: usize,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Errors encountered (non-fatal).
    pub errors: Vec<String>,
}

/// Run the data augmentation batch job.
///
/// This is designed to run as an offline batch job (e.g., nightly cron).
/// It reads real trajectories, generates MCTS alternatives, and persists them.
///
/// # Concurrency
/// This function uses `tokio::task::spawn_blocking` internally for MCTS
/// computation-heavy work. The `max_threads` parameter on CpuGuard limits parallelism.
pub async fn run_augmentation(
    store: Arc<Neo4jTrajectoryStore>,
    config: &AugmentationConfig,
) -> Result<AugmentationReport> {
    let start = std::time::Instant::now();
    let proxy = GdsHeuristicProxy::default();
    let engine = MctsEngine::new(config.mcts.clone(), proxy);

    // Count existing trajectories to enforce ratio limit
    let existing_count = store.count().await?;
    let real_count = count_real_trajectories(&store).await?;
    let sim_count = existing_count.saturating_sub(real_count);
    let max_new_sims =
        ((real_count as f64 * config.max_sim_ratio) as usize).saturating_sub(sim_count);

    if max_new_sims == 0 {
        tracing::info!(
            real_count,
            sim_count,
            "Augmentation skipped: simulation ratio limit reached"
        );
        return Ok(AugmentationReport {
            source_count: 0,
            generated_count: 0,
            persisted_count: 0,
            skipped_count: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            errors: vec![],
        });
    }

    tracing::info!(
        real_count,
        sim_count,
        max_new_sims,
        "Starting augmentation batch"
    );

    let mut report = AugmentationReport {
        source_count: 0,
        generated_count: 0,
        persisted_count: 0,
        skipped_count: 0,
        duration_ms: 0,
        errors: vec![],
    };

    // Process real trajectories in batches
    let mut offset = 0;
    let mut total_generated = 0usize;

    loop {
        if total_generated >= max_new_sims {
            break;
        }

        let filter = TrajectoryFilter {
            min_reward: Some(0.1), // Skip very low reward trajectories
            limit: Some(config.batch_size),
            offset: Some(offset),
            ..Default::default()
        };

        let batch = store.list_trajectories(&filter).await?;
        if batch.is_empty() {
            break;
        }

        for source in &batch {
            if total_generated >= max_new_sims {
                report.skipped_count += 1;
                continue;
            }

            // Skip simulated trajectories (don't augment augmented data)
            if source.session_id.starts_with("mcts-sim-")
                || source.session_id.starts_with("migrated-")
            {
                continue;
            }

            report.source_count += 1;

            match engine.generate_alternatives(source).await {
                Ok(alternatives) => {
                    let alt_count = alternatives.len();
                    report.generated_count += alt_count;

                    for alt in alternatives {
                        if total_generated >= max_new_sims {
                            break;
                        }

                        match store.store_trajectory(&alt).await {
                            Ok(()) => {
                                report.persisted_count += 1;
                                total_generated += 1;
                            }
                            Err(e) => {
                                report.errors.push(format!(
                                    "Failed to persist simulated trajectory {}: {}",
                                    alt.id, e
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("MCTS failed for trajectory {}: {}", source.id, e));
                }
            }

            #[allow(clippy::manual_is_multiple_of)]
            if config.log_interval > 0 && report.source_count % config.log_interval == 0 {
                tracing::info!(
                    source_count = report.source_count,
                    generated = report.generated_count,
                    persisted = report.persisted_count,
                    "Augmentation progress"
                );
            }
        }

        offset += config.batch_size;
    }

    report.duration_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        source_count = report.source_count,
        generated = report.generated_count,
        persisted = report.persisted_count,
        skipped = report.skipped_count,
        duration_ms = report.duration_ms,
        errors = report.errors.len(),
        "Augmentation batch complete"
    );

    Ok(report)
}

/// Count trajectories that are NOT simulated (real + migrated).
async fn count_real_trajectories(store: &Neo4jTrajectoryStore) -> Result<usize> {
    // For now, use total count as approximation.
    // Once we have a source flag in the model, we can filter properly.
    // Real trajectories don't have "mcts-sim-" prefix in session_id.
    store.count().await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AugmentationConfig::default();
        assert!((config.max_sim_ratio - 10.0).abs() < 1e-10);
        assert!((config.sim_weight - 0.3).abs() < 1e-10);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_report_defaults() {
        let report = AugmentationReport {
            source_count: 0,
            generated_count: 0,
            persisted_count: 0,
            skipped_count: 0,
            duration_ms: 0,
            errors: vec![],
        };
        assert_eq!(report.errors.len(), 0);
    }

    #[test]
    fn test_augmentation_config_custom() {
        let mcts_config = MctsConfig {
            expansion_width: 8,
            num_rollouts: 100,
            exploration_c: 1.5,
            max_alternatives: 3,
            gamma: 0.95,
        };

        let config = AugmentationConfig {
            max_sim_ratio: 5.0,
            sim_weight: 0.5,
            mcts: mcts_config,
            batch_size: 25,
            log_interval: 50,
        };

        assert!((config.max_sim_ratio - 5.0).abs() < 1e-10);
        assert!((config.sim_weight - 0.5).abs() < 1e-10);
        assert_eq!(config.batch_size, 25);
        assert_eq!(config.log_interval, 50);
        assert_eq!(config.mcts.expansion_width, 8);
        assert_eq!(config.mcts.num_rollouts, 100);
        assert!((config.mcts.exploration_c - 1.5).abs() < 1e-10);
        assert_eq!(config.mcts.max_alternatives, 3);
        assert!((config.mcts.gamma - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_report_accumulation() {
        let mut report = AugmentationReport {
            source_count: 10,
            generated_count: 45,
            persisted_count: 40,
            skipped_count: 2,
            duration_ms: 1500,
            errors: vec![],
        };

        // Simulate accumulating errors during a batch run
        report
            .errors
            .push("Failed to persist simulated trajectory abc: connection timeout".to_string());
        report
            .errors
            .push("MCTS failed for trajectory def: invalid state".to_string());

        assert_eq!(report.source_count, 10);
        assert_eq!(report.generated_count, 45);
        assert_eq!(report.persisted_count, 40);
        assert_eq!(report.skipped_count, 2);
        assert_eq!(report.duration_ms, 1500);
        assert_eq!(report.errors.len(), 2);
        assert!(report.errors[0].contains("connection timeout"));
        assert!(report.errors[1].contains("invalid state"));

        // Verify the difference between generated and persisted matches expectations
        let failed_or_skipped = report.generated_count - report.persisted_count;
        assert_eq!(
            failed_or_skipped, 5,
            "5 trajectories were generated but not persisted"
        );
    }

    #[test]
    fn test_max_sim_ratio_computation() {
        // Mirrors the formula from run_augmentation:
        // max_new_sims = (real_count as f64 * max_sim_ratio) as usize - sim_count

        let config = AugmentationConfig {
            max_sim_ratio: 10.0,
            ..Default::default()
        };

        // Case 1: No existing simulations — full budget available
        let real_count: usize = 100;
        let sim_count: usize = 0;
        let max_new_sims =
            ((real_count as f64 * config.max_sim_ratio) as usize).saturating_sub(sim_count);
        assert_eq!(
            max_new_sims, 1000,
            "100 real * 10.0 ratio - 0 existing = 1000"
        );

        // Case 2: Some simulations already exist
        let sim_count: usize = 300;
        let max_new_sims =
            ((real_count as f64 * config.max_sim_ratio) as usize).saturating_sub(sim_count);
        assert_eq!(
            max_new_sims, 700,
            "100 real * 10.0 ratio - 300 existing = 700"
        );

        // Case 3: Ratio limit already reached
        let sim_count: usize = 1000;
        let max_new_sims =
            ((real_count as f64 * config.max_sim_ratio) as usize).saturating_sub(sim_count);
        assert_eq!(
            max_new_sims, 0,
            "Limit reached: no more simulations allowed"
        );

        // Case 4: Over the limit (saturating_sub prevents underflow)
        let sim_count: usize = 1500;
        let max_new_sims =
            ((real_count as f64 * config.max_sim_ratio) as usize).saturating_sub(sim_count);
        assert_eq!(max_new_sims, 0, "Over limit: saturating_sub returns 0");

        // Case 5: Different ratio
        let config_low = AugmentationConfig {
            max_sim_ratio: 2.0,
            ..Default::default()
        };
        let real_count: usize = 50;
        let sim_count: usize = 30;
        let max_new_sims =
            ((real_count as f64 * config_low.max_sim_ratio) as usize).saturating_sub(sim_count);
        assert_eq!(max_new_sims, 70, "50 real * 2.0 ratio - 30 existing = 70");
    }
}
