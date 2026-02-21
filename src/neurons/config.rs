//! Configuration for the Spreading Activation engine.
//!
//! All parameters have sensible defaults and can be overridden per-query.

use serde::{Deserialize, Serialize};

/// Configuration for the spreading activation algorithm.
///
/// Controls the 3-phase retrieval process:
/// 1. Initial activation (vector search → top-K seed notes)
/// 2. Spreading (propagation through SYNAPSE edges, hop by hop)
/// 3. Ranking (merge, deduplicate, sort by score)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingActivationConfig {
    /// Number of seed notes from the initial vector search (Phase 1).
    /// Higher values cast a wider net but cost more in spreading.
    pub initial_k: usize,

    /// Maximum number of hops for spreading through synapses (Phase 2).
    /// 1 = direct neighbors only, 2 = neighbors-of-neighbors.
    /// Values > 3 are not recommended (exponential fan-out).
    pub max_hops: usize,

    /// Minimum activation score to keep a note in the result set.
    /// Notes below this threshold are discarded.
    pub min_activation: f64,

    /// Decay factor applied at each hop during spreading.
    /// 0.5 means each hop halves the signal strength.
    /// Formula: spread_score = parent_activation × synapse_weight × neighbor_energy × decay_per_hop
    pub decay_per_hop: f64,

    /// Minimum energy threshold — notes with energy below this are "dead neurons"
    /// and won't participate in spreading (neither as source nor target).
    pub min_energy: f64,

    /// Maximum number of results to return (Phase 3).
    pub max_results: usize,
}

impl Default for SpreadingActivationConfig {
    fn default() -> Self {
        Self {
            initial_k: 20,
            max_hops: 2,
            min_activation: 0.1,
            decay_per_hop: 0.5,
            min_energy: 0.05,
            max_results: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpreadingActivationConfig::default();
        assert_eq!(config.initial_k, 20);
        assert_eq!(config.max_hops, 2);
        assert!((config.min_activation - 0.1).abs() < f64::EPSILON);
        assert!((config.decay_per_hop - 0.5).abs() < f64::EPSILON);
        assert!((config.min_energy - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.max_results, 10);
    }
}
