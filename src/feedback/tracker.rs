//! OutcomeTracker — central coordinator for the closed-loop learning system.
//!
//! Maintains in-memory stores for feedback and signal state, provides
//! aggregation queries, and coordinates the signal detector + propagator.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

use super::models::*;
use super::signals::ImplicitSignalDetector;

// ============================================================================
// Global singleton
// ============================================================================

static GLOBAL_TRACKER: LazyLock<OutcomeTracker> = LazyLock::new(OutcomeTracker::new);

// ============================================================================
// OutcomeTracker
// ============================================================================

/// Central coordinator for the closed-loop learning system.
///
/// The OutcomeTracker:
/// 1. Stores explicit feedback entries (in-memory, with future Neo4j persistence)
/// 2. Manages the ImplicitSignalDetector
/// 3. Tracks signal processing statistics
/// 4. Provides aggregation queries for stats endpoints
///
/// Designed to work standalone — explicit feedback API works even without
/// HeartbeatEngine, NeuralFeedback, or other subsystems.
#[derive(Debug)]
pub struct OutcomeTracker {
    /// Explicit feedback store (in-memory)
    feedback_store: Arc<RwLock<Vec<ExplicitFeedback>>>,
    /// Implicit signal detector
    pub signal_detector: Arc<ImplicitSignalDetector>,
    /// Processing counters
    counters: Arc<RwLock<TrackerCounters>>,
}

#[derive(Debug, Default)]
struct TrackerCounters {
    /// Count of implicit signals processed, by type
    signal_counts: HashMap<String, usize>,
    /// Number of positive propagations
    positive_propagations: usize,
    /// Number of negative propagations
    negative_propagations: usize,
}

impl OutcomeTracker {
    /// Create a new OutcomeTracker.
    pub fn new() -> Self {
        Self {
            feedback_store: Arc::new(RwLock::new(Vec::new())),
            signal_detector: Arc::new(ImplicitSignalDetector::new()),
            counters: Arc::new(RwLock::new(TrackerCounters::default())),
        }
    }

    /// Get the global OutcomeTracker instance.
    pub fn global() -> &'static OutcomeTracker {
        &GLOBAL_TRACKER
    }

    // ========================================================================
    // Explicit Feedback
    // ========================================================================

    /// Record an explicit feedback entry.
    pub async fn record_explicit_feedback(&self, feedback: ExplicitFeedback) {
        debug!(
            "[tracker] Recording feedback {} for {:?}/{}",
            feedback.id, feedback.target_type, feedback.target_id
        );
        let mut store = self.feedback_store.write().await;
        store.push(feedback);
    }

    /// List feedback entries with optional filters.
    pub async fn list_feedback(
        &self,
        target_type: Option<&str>,
        target_id: Option<Uuid>,
        project_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> Vec<ExplicitFeedback> {
        let store = self.feedback_store.read().await;
        store
            .iter()
            .filter(|fb| {
                if let Some(tt) = target_type {
                    if fb.target_type.to_string() != tt {
                        return false;
                    }
                }
                if let Some(tid) = target_id {
                    if fb.target_id != tid {
                        return false;
                    }
                }
                if let Some(pid) = project_id {
                    if fb.project_id != Some(pid) {
                        return false;
                    }
                }
                true
            })
            .skip(offset)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Compute aggregated feedback statistics.
    pub async fn compute_stats(
        &self,
        target_type: Option<&str>,
        target_id: Option<Uuid>,
        project_id: Option<Uuid>,
    ) -> FeedbackStatsResponse {
        let store = self.feedback_store.read().await;

        // Filter entries
        let filtered: Vec<&ExplicitFeedback> = store
            .iter()
            .filter(|fb| {
                if let Some(tt) = target_type {
                    if fb.target_type.to_string() != tt {
                        return false;
                    }
                }
                if let Some(tid) = target_id {
                    if fb.target_id != tid {
                        return false;
                    }
                }
                if let Some(pid) = project_id {
                    if fb.project_id != Some(pid) {
                        return false;
                    }
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            return FeedbackStatsResponse {
                stats: Vec::new(),
                total_feedback_count: 0,
                avg_score: 0.0,
            };
        }

        // Group by (target_type, target_id)
        let mut groups: HashMap<(String, Uuid), Vec<&ExplicitFeedback>> = HashMap::new();
        for fb in &filtered {
            groups
                .entry((fb.target_type.to_string(), fb.target_id))
                .or_default()
                .push(fb);
        }

        let mut stats = Vec::new();
        for ((tt, tid), entries) in &groups {
            let count = entries.len();
            let scores: Vec<f64> = entries.iter().map(|e| e.score).collect();
            let avg = scores.iter().sum::<f64>() / count as f64;
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let last = entries.iter().map(|e| e.created_at).max();

            stats.push(FeedbackStats {
                target_id: *tid,
                target_type: tt.clone(),
                count,
                avg_score: avg,
                min_score: min,
                max_score: max,
                last_feedback_at: last,
            });
        }

        let total = filtered.len();
        let global_avg = filtered.iter().map(|f| f.score).sum::<f64>() / total as f64;

        FeedbackStatsResponse {
            stats,
            total_feedback_count: total,
            avg_score: global_avg,
        }
    }

    // ========================================================================
    // Implicit Signal Tracking
    // ========================================================================

    /// Record that an implicit signal was processed.
    pub async fn record_signal_processed(&self, signal: &ImplicitSignal) {
        let mut counters = self.counters.write().await;
        *counters
            .signal_counts
            .entry(signal.label().to_string())
            .or_insert(0) += 1;

        if signal.is_positive() {
            counters.positive_propagations += 1;
        } else {
            counters.negative_propagations += 1;
        }
    }

    // ========================================================================
    // Learning Stats (for admin MCP action)
    // ========================================================================

    /// Get comprehensive learning statistics.
    pub async fn get_learning_stats(&self) -> LearningStats {
        let store = self.feedback_store.read().await;
        let counters = self.counters.read().await;

        let total_explicit = store.len();
        let avg_score = if total_explicit > 0 {
            store.iter().map(|f| f.score).sum::<f64>() / total_explicit as f64
        } else {
            0.0
        };

        let total_implicit: usize = counters.signal_counts.values().sum();

        LearningStats {
            total_explicit_feedback: total_explicit,
            total_implicit_signals: total_implicit,
            avg_explicit_score: avg_score,
            signal_counts: counters.signal_counts.clone(),
            positive_propagations: counters.positive_propagations,
            negative_propagations: counters.negative_propagations,
        }
    }
}

impl Default for OutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_and_list_feedback() {
        let tracker = OutcomeTracker::new();
        let note_id = Uuid::new_v4();

        let fb1 =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, 0.8, "agent-1".into()).unwrap();
        let fb2 =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, -0.3, "agent-2".into()).unwrap();

        tracker.record_explicit_feedback(fb1).await;
        tracker.record_explicit_feedback(fb2).await;

        let all = tracker.list_feedback(None, None, None, 50, 0).await;
        assert_eq!(all.len(), 2);

        // Filter by target_id
        let filtered = tracker
            .list_feedback(None, Some(note_id), None, 50, 0)
            .await;
        assert_eq!(filtered.len(), 2);

        // Filter by non-existent target
        let empty = tracker
            .list_feedback(None, Some(Uuid::new_v4()), None, 50, 0)
            .await;
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn test_compute_stats() {
        let tracker = OutcomeTracker::new();
        let note_id = Uuid::new_v4();

        let fb1 =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, 0.8, "agent".into()).unwrap();
        let fb2 =
            ExplicitFeedback::new(FeedbackTarget::Note, note_id, 0.4, "agent".into()).unwrap();

        tracker.record_explicit_feedback(fb1).await;
        tracker.record_explicit_feedback(fb2).await;

        let stats = tracker.compute_stats(None, None, None).await;
        assert_eq!(stats.total_feedback_count, 2);
        assert!((stats.avg_score - 0.6).abs() < 0.01);
        assert_eq!(stats.stats.len(), 1);
        assert_eq!(stats.stats[0].count, 2);
        assert!((stats.stats[0].avg_score - 0.6).abs() < 0.01);
        assert!((stats.stats[0].min_score - 0.4).abs() < 0.01);
        assert!((stats.stats[0].max_score - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_empty_stats() {
        let tracker = OutcomeTracker::new();
        let stats = tracker.compute_stats(None, None, None).await;
        assert_eq!(stats.total_feedback_count, 0);
        assert_eq!(stats.avg_score, 0.0);
        assert!(stats.stats.is_empty());
    }

    #[tokio::test]
    async fn test_learning_stats() {
        let tracker = OutcomeTracker::new();

        // Add some feedback
        let fb = ExplicitFeedback::new(FeedbackTarget::Note, Uuid::new_v4(), 0.5, "agent".into())
            .unwrap();
        tracker.record_explicit_feedback(fb).await;

        // Record some signals
        let sig1 = ImplicitSignal::CommitReverted {
            commit_id: Uuid::new_v4(),
            revert_commit_id: None,
        };
        let sig2 = ImplicitSignal::PlanCompletedClean {
            plan_id: Uuid::new_v4(),
            task_count: 3,
        };
        tracker.record_signal_processed(&sig1).await;
        tracker.record_signal_processed(&sig2).await;

        let stats = tracker.get_learning_stats().await;
        assert_eq!(stats.total_explicit_feedback, 1);
        assert_eq!(stats.total_implicit_signals, 2);
        assert!((stats.avg_explicit_score - 0.5).abs() < 0.01);
        assert_eq!(stats.positive_propagations, 1);
        assert_eq!(stats.negative_propagations, 1);
        assert_eq!(stats.signal_counts.get("commit_reverted"), Some(&1));
        assert_eq!(stats.signal_counts.get("plan_completed_clean"), Some(&1));
    }

    #[tokio::test]
    async fn test_pagination() {
        let tracker = OutcomeTracker::new();

        // Add 10 feedback entries
        for i in 0..10 {
            let fb = ExplicitFeedback::new(
                FeedbackTarget::Note,
                Uuid::new_v4(),
                (i as f64) * 0.1,
                "agent".into(),
            )
            .unwrap();
            tracker.record_explicit_feedback(fb).await;
        }

        // Page 1: first 3
        let page1 = tracker.list_feedback(None, None, None, 3, 0).await;
        assert_eq!(page1.len(), 3);

        // Page 2: next 3
        let page2 = tracker.list_feedback(None, None, None, 3, 3).await;
        assert_eq!(page2.len(), 3);

        // Verify different entries
        assert_ne!(page1[0].id, page2[0].id);
    }
}
