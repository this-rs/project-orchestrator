//! SynapseDecayCheck — decays synapse weights and updates energy scores.
//!
//! Runs `decay_synapses` to weaken unused neural links,
//! then `update_energy_scores` to decay note energy based on
//! time since last activation (exponential half-life model).

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::debug;

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Default synapse decay amount per cycle.
const DEFAULT_DECAY_AMOUNT: f64 = 0.02;
/// Default prune threshold — synapses below this weight are deleted.
const DEFAULT_PRUNE_THRESHOLD: f64 = 0.05;
/// Default energy half-life in days.
const DEFAULT_HALF_LIFE_DAYS: f64 = 14.0;

/// Decay synapses and update energy scores (every 6 hours).
pub struct SynapseDecayCheck;

#[async_trait]
impl HeartbeatCheck for SynapseDecayCheck {
    fn name(&self) -> &str {
        "synapse_decay"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(6 * 60 * 60) // 6 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        // 1. Decay synapses
        let (decayed, pruned) = ctx
            .graph
            .decay_synapses(DEFAULT_DECAY_AMOUNT, DEFAULT_PRUNE_THRESHOLD)
            .await?;
        debug!(
            "SynapseDecayCheck: decayed {} synapses, pruned {}",
            decayed, pruned
        );

        // 2. Update energy scores
        let updated = ctx
            .graph
            .update_energy_scores(DEFAULT_HALF_LIFE_DAYS)
            .await?;
        debug!("SynapseDecayCheck: updated energy for {} notes", updated);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_decay_check_name() {
        let check = SynapseDecayCheck;
        assert_eq!(check.name(), "synapse_decay");
    }

    #[test]
    fn test_synapse_decay_check_interval() {
        let check = SynapseDecayCheck;
        assert_eq!(check.interval(), Duration::from_secs(6 * 3600));
    }

    #[test]
    fn test_default_constants() {
        assert!((DEFAULT_DECAY_AMOUNT - 0.02).abs() < f64::EPSILON);
        assert!((DEFAULT_PRUNE_THRESHOLD - 0.05).abs() < f64::EPSILON);
        assert!((DEFAULT_HALF_LIFE_DAYS - 14.0).abs() < f64::EPSILON);
    }
}
