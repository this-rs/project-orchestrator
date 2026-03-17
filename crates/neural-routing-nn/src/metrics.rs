//! Metrics for the Nearest Neighbor Router.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe metrics for monitoring NN Router performance.
#[derive(Debug, Clone)]
pub struct NNMetrics {
    inner: Arc<NNMetricsInner>,
}

#[derive(Debug)]
struct NNMetricsInner {
    /// Total number of route queries.
    total_queries: AtomicU64,
    /// Queries that found a neighbor above min_similarity threshold.
    hits: AtomicU64,
    /// Queries that used a cached result.
    cache_hits: AtomicU64,
    /// Sum of similarity scores for all hits (for averaging).
    similarity_sum: AtomicU64, // stored as f64 bits
    /// Sum of source trajectory rewards for all hits.
    reward_sum: AtomicU64, // stored as f64 bits
}

impl NNMetrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(NNMetricsInner {
                total_queries: AtomicU64::new(0),
                hits: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                similarity_sum: AtomicU64::new(0.0f64.to_bits()),
                reward_sum: AtomicU64::new(0.0f64.to_bits()),
            }),
        }
    }

    pub fn record_query(&self) {
        self.inner.total_queries.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_hit(&self, similarity: f64, reward: f64) {
        self.inner.hits.fetch_add(1, Ordering::Relaxed);
        atomic_add_f64(&self.inner.similarity_sum, similarity);
        atomic_add_f64(&self.inner.reward_sum, reward);
    }

    pub fn record_cache_hit(&self) {
        self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> NNMetricsSnapshot {
        let total = self.inner.total_queries.load(Ordering::Relaxed);
        let hits = self.inner.hits.load(Ordering::Relaxed);
        let cache_hits = self.inner.cache_hits.load(Ordering::Relaxed);
        let sim_sum = f64::from_bits(self.inner.similarity_sum.load(Ordering::Relaxed));
        let rew_sum = f64::from_bits(self.inner.reward_sum.load(Ordering::Relaxed));

        NNMetricsSnapshot {
            total_queries: total,
            hits,
            cache_hits,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
            cache_hit_rate: if total > 0 { cache_hits as f64 / total as f64 } else { 0.0 },
            avg_similarity: if hits > 0 { sim_sum / hits as f64 } else { 0.0 },
            avg_reward: if hits > 0 { rew_sum / hits as f64 } else { 0.0 },
        }
    }
}

impl Default for NNMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of NN Router metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNMetricsSnapshot {
    pub total_queries: u64,
    pub hits: u64,
    pub cache_hits: u64,
    pub hit_rate: f64,
    pub cache_hit_rate: f64,
    pub avg_similarity: f64,
    pub avg_reward: f64,
}

/// Atomically add a f64 to an AtomicU64 (stored as bits).
fn atomic_add_f64(atomic: &AtomicU64, value: f64) {
    let mut current = atomic.load(Ordering::Relaxed);
    loop {
        let current_f64 = f64::from_bits(current);
        let new_f64 = current_f64 + value;
        match atomic.compare_exchange_weak(
            current,
            new_f64.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = NNMetrics::new();

        metrics.record_query();
        metrics.record_query();
        metrics.record_hit(0.9, 0.8);
        metrics.record_cache_hit();

        let snap = metrics.snapshot();
        assert_eq!(snap.total_queries, 2);
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.cache_hits, 1);
        assert!((snap.hit_rate - 0.5).abs() < 1e-5);
        assert!((snap.avg_similarity - 0.9).abs() < 1e-5);
        assert!((snap.avg_reward - 0.8).abs() < 1e-5);
    }
}
