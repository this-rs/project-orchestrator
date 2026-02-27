//! Negative Feedback via Utility Signal
//!
//! Tracks the utility of skill activations to identify skills that inject
//! irrelevant context. When an activation is followed by another tool call
//! on the same pattern within 60 seconds, it's considered a "miss" (the
//! injected context didn't help).
//!
//! The hit_rate = (activations - misses) / activations drives energy penalties:
//! - hit_rate < 0.3 after 10+ activations → energy penalty (-0.05)

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the activation feedback system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Maximum number of activation events to retain per project (default: 100)
    pub max_buffer_size: usize,
    /// Time window in seconds for detecting misses (default: 60)
    pub miss_window_secs: u64,
    /// Minimum activations before applying hit_rate penalty (default: 10)
    pub min_activations_for_penalty: usize,
    /// Hit rate threshold below which energy is penalized (default: 0.3)
    pub penalty_hit_rate_threshold: f64,
    /// Energy penalty amount for low hit_rate skills (default: 0.05)
    pub energy_penalty: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 100,
            miss_window_secs: 60,
            min_activations_for_penalty: 10,
            penalty_hit_rate_threshold: 0.3,
            energy_penalty: 0.05,
        }
    }
}

// ============================================================================
// Activation Event
// ============================================================================

/// A recorded skill activation event.
#[derive(Debug, Clone)]
pub struct ActivationEvent {
    /// The skill that was activated
    pub skill_id: Uuid,
    /// The extracted pattern that triggered the activation
    pub pattern: String,
    /// Optional file context (for file-specific activations)
    pub file_context: Option<String>,
    /// When the activation occurred
    pub timestamp: Instant,
    /// Whether this activation was detected as a miss
    pub is_miss: bool,
}

// ============================================================================
// Activation Buffer
// ============================================================================

/// Per-project circular buffer of recent activation events.
///
/// Thread-safe via `RwLock<HashMap>` for concurrent access.
/// Bounded to `max_buffer_size` entries per project.
#[derive(Debug, Clone)]
pub struct ActivationBuffer {
    /// Per-project event buffers
    buffers: Arc<RwLock<HashMap<Uuid, VecDeque<ActivationEvent>>>>,
    /// Configuration
    config: FeedbackConfig,
}

impl ActivationBuffer {
    /// Create a new activation buffer with the given config.
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Record an activation event and detect if it's a miss.
    ///
    /// A miss is detected when a previous activation with the same pattern
    /// (and optionally same file context) occurred within the miss window.
    /// This suggests the previous activation's injected context wasn't helpful.
    ///
    /// Returns `true` if a miss was detected (the PREVIOUS activation was a miss).
    pub fn record_activation(
        &self,
        project_id: Uuid,
        skill_id: Uuid,
        pattern: &str,
        file_context: Option<&str>,
    ) -> bool {
        let now = Instant::now();
        let miss_window = Duration::from_secs(self.config.miss_window_secs);

        let mut is_miss = false;

        let mut buffers = self.buffers.write().unwrap_or_else(|e| e.into_inner());
        let buffer = buffers.entry(project_id).or_default();

        // Check if there's a recent activation with the same pattern
        for event in buffer.iter_mut().rev() {
            if now.duration_since(event.timestamp) > miss_window {
                break; // Events are chronological, no need to check older ones
            }
            if event.pattern == pattern && event.file_context.as_deref() == file_context {
                // The previous activation with the same pattern was a miss
                event.is_miss = true;
                is_miss = true;
                debug!(
                    skill_id = %event.skill_id,
                    pattern = pattern,
                    "Detected miss: repeated pattern within {}s window",
                    self.config.miss_window_secs
                );
                break;
            }
        }

        // Record the new activation
        buffer.push_back(ActivationEvent {
            skill_id,
            pattern: pattern.to_string(),
            file_context: file_context.map(|s| s.to_string()),
            timestamp: now,
            is_miss: false,
        });

        // Trim buffer to max size
        while buffer.len() > self.config.max_buffer_size {
            buffer.pop_front();
        }

        is_miss
    }

    /// Calculate the hit rate for a specific skill across all projects.
    ///
    /// Returns `(hit_rate, total_activations)` or `None` if no activations found.
    pub fn get_skill_hit_rate(&self, skill_id: Uuid) -> Option<(f64, usize)> {
        let mut total = 0usize;
        let mut misses = 0usize;

        let buffers = self.buffers.read().unwrap_or_else(|e| e.into_inner());
        for buffer in buffers.values() {
            for event in buffer.iter() {
                if event.skill_id == skill_id {
                    total += 1;
                    if event.is_miss {
                        misses += 1;
                    }
                }
            }
        }

        if total == 0 {
            return None;
        }

        let hit_rate = (total - misses) as f64 / total as f64;
        Some((hit_rate, total))
    }

    /// Get all skills that should receive an energy penalty.
    ///
    /// Returns skills with hit_rate < threshold after min_activations.
    pub fn get_penalty_candidates(&self) -> Vec<(Uuid, f64, usize)> {
        // Collect all skill IDs from all buffers
        let mut skill_stats: HashMap<Uuid, (usize, usize)> = HashMap::new();

        let buffers = self.buffers.read().unwrap_or_else(|e| e.into_inner());
        for buffer in buffers.values() {
            for event in buffer.iter() {
                let stat = skill_stats.entry(event.skill_id).or_insert((0, 0));
                stat.0 += 1; // total
                if event.is_miss {
                    stat.1 += 1; // misses
                }
            }
        }

        skill_stats
            .into_iter()
            .filter_map(|(skill_id, (total, misses))| {
                if total < self.config.min_activations_for_penalty {
                    return None;
                }
                let hit_rate = (total - misses) as f64 / total as f64;
                if hit_rate < self.config.penalty_hit_rate_threshold {
                    Some((skill_id, hit_rate, total))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Apply energy penalties to skills with low hit rates.
    ///
    /// Returns the number of skills penalized.
    pub async fn apply_penalties(
        &self,
        graph_store: &dyn crate::neo4j::traits::GraphStore,
    ) -> anyhow::Result<usize> {
        let candidates = self.get_penalty_candidates();
        let mut penalized = 0;

        for (skill_id, hit_rate, total) in &candidates {
            if let Ok(Some(mut skill)) = graph_store.get_skill(*skill_id).await {
                let old_energy = skill.energy;
                skill.energy = (skill.energy - self.config.energy_penalty).max(0.0);
                skill.hit_rate = *hit_rate;
                skill.updated_at = chrono::Utc::now();

                if let Err(e) = graph_store.update_skill(&skill).await {
                    tracing::warn!(
                        skill_id = %skill_id,
                        error = %e,
                        "Failed to apply energy penalty"
                    );
                } else {
                    info!(
                        skill_id = %skill_id,
                        hit_rate = hit_rate,
                        activations = total,
                        old_energy = old_energy,
                        new_energy = skill.energy,
                        "Applied energy penalty for low hit rate"
                    );
                    penalized += 1;
                }
            }
        }

        Ok(penalized)
    }

    /// Clear the buffer for a specific project.
    pub fn clear_project(&self, project_id: Uuid) {
        let mut buffers = self.buffers.write().unwrap_or_else(|e| e.into_inner());
        buffers.remove(&project_id);
    }

    /// Get the total number of events across all projects.
    pub fn total_events(&self) -> usize {
        let buffers = self.buffers.read().unwrap_or_else(|e| e.into_inner());
        buffers.values().map(|b| b.len()).sum()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buffer() -> ActivationBuffer {
        ActivationBuffer::new(FeedbackConfig {
            max_buffer_size: 10,
            miss_window_secs: 60,
            min_activations_for_penalty: 3,
            penalty_hit_rate_threshold: 0.3,
            energy_penalty: 0.05,
        })
    }

    #[test]
    fn test_config_defaults() {
        let config = FeedbackConfig::default();
        assert_eq!(config.max_buffer_size, 100);
        assert_eq!(config.miss_window_secs, 60);
        assert_eq!(config.min_activations_for_penalty, 10);
        assert!((config.penalty_hit_rate_threshold - 0.3).abs() < f64::EPSILON);
        assert!((config.energy_penalty - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_first_activation_no_miss() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        let is_miss = buffer.record_activation(project_id, skill_id, "neo4j", None);
        assert!(!is_miss);
        assert_eq!(buffer.total_events(), 1);
    }

    #[test]
    fn test_record_same_pattern_is_miss() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        buffer.record_activation(project_id, skill_id, "neo4j", None);
        let is_miss = buffer.record_activation(project_id, skill_id, "neo4j", None);
        assert!(is_miss, "Same pattern within window should be a miss");
    }

    #[test]
    fn test_different_pattern_no_miss() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        buffer.record_activation(project_id, skill_id, "neo4j", None);
        let is_miss = buffer.record_activation(project_id, skill_id, "api", None);
        assert!(!is_miss, "Different pattern should not be a miss");
    }

    #[test]
    fn test_same_pattern_different_file_no_miss() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        buffer.record_activation(project_id, skill_id, "neo4j", Some("src/a.rs"));
        let is_miss = buffer.record_activation(project_id, skill_id, "neo4j", Some("src/b.rs"));
        assert!(
            !is_miss,
            "Same pattern but different file should not be a miss"
        );
    }

    #[test]
    fn test_buffer_trim_to_max_size() {
        let buffer = make_buffer(); // max_buffer_size = 10
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        for i in 0..15 {
            buffer.record_activation(project_id, skill_id, &format!("pattern-{}", i), None);
        }

        assert_eq!(buffer.total_events(), 10);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        // 5 activations, 2 will be misses (repeated pattern)
        buffer.record_activation(project_id, skill_id, "a", None);
        buffer.record_activation(project_id, skill_id, "a", None); // miss on first "a"
        buffer.record_activation(project_id, skill_id, "b", None);
        buffer.record_activation(project_id, skill_id, "b", None); // miss on first "b"
        buffer.record_activation(project_id, skill_id, "c", None);

        let (hit_rate, total) = buffer.get_skill_hit_rate(skill_id).unwrap();
        assert_eq!(total, 5);
        // 2 events marked as miss out of 5 → hit_rate = 3/5 = 0.6
        assert!(
            (hit_rate - 0.6).abs() < 0.01,
            "Expected hit_rate ~0.6, got {}",
            hit_rate
        );
    }

    #[test]
    fn test_hit_rate_no_activations() {
        let buffer = make_buffer();
        assert!(buffer.get_skill_hit_rate(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_penalty_candidates_below_min_activations() {
        let buffer = make_buffer(); // min_activations_for_penalty = 3
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        // Only 2 activations — below threshold
        buffer.record_activation(project_id, skill_id, "a", None);
        buffer.record_activation(project_id, skill_id, "a", None); // miss

        let candidates = buffer.get_penalty_candidates();
        assert!(
            candidates.is_empty(),
            "Should not penalize with fewer than min_activations"
        );
    }

    #[test]
    fn test_penalty_candidates_low_hit_rate() {
        let buffer = make_buffer(); // min=3, threshold=0.3
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        // 4 activations, 3 misses → hit_rate = 1/4 = 0.25 < 0.3
        buffer.record_activation(project_id, skill_id, "a", None);
        buffer.record_activation(project_id, skill_id, "a", None); // miss #1
        buffer.record_activation(project_id, skill_id, "a", None); // miss #2
        buffer.record_activation(project_id, skill_id, "a", None); // miss #3

        let candidates = buffer.get_penalty_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, skill_id);
        assert!(candidates[0].1 < 0.3);
    }

    #[test]
    fn test_penalty_candidates_good_hit_rate() {
        let buffer = make_buffer(); // min=3, threshold=0.3
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        // 4 unique patterns — no misses → hit_rate = 1.0
        buffer.record_activation(project_id, skill_id, "a", None);
        buffer.record_activation(project_id, skill_id, "b", None);
        buffer.record_activation(project_id, skill_id, "c", None);
        buffer.record_activation(project_id, skill_id, "d", None);

        let candidates = buffer.get_penalty_candidates();
        assert!(
            candidates.is_empty(),
            "Good hit rate should not be penalized"
        );
    }

    #[test]
    fn test_clear_project() {
        let buffer = make_buffer();
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        buffer.record_activation(project_id, skill_id, "a", None);
        assert_eq!(buffer.total_events(), 1);

        buffer.clear_project(project_id);
        assert_eq!(buffer.total_events(), 0);
    }

    #[test]
    fn test_cross_project_isolation() {
        let buffer = make_buffer();
        let project_a = Uuid::new_v4();
        let project_b = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        buffer.record_activation(project_a, skill_id, "neo4j", None);
        // Same pattern but different project — should NOT be a miss
        let is_miss = buffer.record_activation(project_b, skill_id, "neo4j", None);
        assert!(!is_miss, "Different projects should be isolated");
    }

    #[test]
    fn test_five_activations_four_misses_triggers_penalty() {
        // Spec scenario: 5 activations where 4 are followed by re-search on
        // the same pattern → hit_rate = 1/5 = 0.2 → penalty candidate.
        // Using min_activations_for_penalty = 5 to match this scenario.
        let buffer = ActivationBuffer::new(FeedbackConfig {
            max_buffer_size: 100,
            miss_window_secs: 60,
            min_activations_for_penalty: 5,
            penalty_hit_rate_threshold: 0.3,
            energy_penalty: 0.05,
        });
        let project_id = Uuid::new_v4();
        let skill_id = Uuid::new_v4();

        // Activation 1: "neo4j" → no miss (first time)
        buffer.record_activation(project_id, skill_id, "neo4j", None);
        // Activation 2: "neo4j" again → miss #1 (re-search = context wasn't helpful)
        buffer.record_activation(project_id, skill_id, "neo4j", None);
        // Activation 3: "neo4j" again → miss #2
        buffer.record_activation(project_id, skill_id, "neo4j", None);
        // Activation 4: "neo4j" again → miss #3
        buffer.record_activation(project_id, skill_id, "neo4j", None);
        // Activation 5: "neo4j" again → miss #4
        buffer.record_activation(project_id, skill_id, "neo4j", None);

        // Verify hit rate: 4 misses out of 5 → hit_rate = 1/5 = 0.2
        let (hit_rate, total) = buffer.get_skill_hit_rate(skill_id).unwrap();
        assert_eq!(total, 5);
        assert!(
            (hit_rate - 0.2).abs() < 0.01,
            "Expected hit_rate ~0.2, got {}",
            hit_rate
        );

        // Verify penalty candidate is produced (hit_rate 0.2 < 0.3 threshold)
        let candidates = buffer.get_penalty_candidates();
        assert_eq!(
            candidates.len(),
            1,
            "Should produce exactly 1 penalty candidate"
        );
        assert_eq!(candidates[0].0, skill_id);
        assert!(
            candidates[0].1 < 0.3,
            "Hit rate {} should be below 0.3 threshold",
            candidates[0].1
        );
    }
}
