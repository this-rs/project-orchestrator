//! # Progress Oracle — Objective progress measurement
//!
//! Measures pipeline execution progress using metrics from quality gates.
//! Calculates a normalized score \[0,1\] and detects stagnation.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Aggregated metrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique sequence number.
    pub seq: u64,
    /// Total tests discovered.
    pub tests_total: usize,
    /// Tests currently passing.
    pub tests_passing: usize,
    /// Tests currently failing.
    pub tests_failing: usize,
    /// Build error count.
    pub build_errors_count: usize,
    /// Code coverage percentage.
    pub coverage_pct: f64,
    /// Steps completed in the plan.
    pub steps_completed: usize,
    /// Total steps in the plan.
    pub steps_total: usize,
    /// Warning count.
    pub warnings_count: usize,
    /// When this checkpoint was taken.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Score with delta tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressScore {
    /// Normalized score \[0, 1\].
    pub score: f64,
    /// Delta from previous checkpoint (`None` if first).
    pub delta: Option<f64>,
    /// Individual dimension scores.
    pub dimensions: ScoreDimensions,
}

/// Per-dimension breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDimensions {
    /// 0 or 1 — whether the project compiles.
    pub build: f64,
    /// Ratio of passing tests (0..1).
    pub tests: f64,
    /// `coverage_pct / 100` (0..1).
    pub coverage: f64,
    /// Ratio of completed steps (0..1).
    pub steps: f64,
}

/// Weight configuration for score computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub build: f64,
    pub tests: f64,
    pub coverage: f64,
    pub steps: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            build: 0.30,
            tests: 0.35,
            coverage: 0.15,
            steps: 0.20,
        }
    }
}

/// Summary of current progress state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressSummary {
    pub current_score: f64,
    pub trend: ProgressTrend,
    pub total_checkpoints: usize,
    pub best_score: f64,
    pub worst_score: f64,
}

/// Trend direction derived from recent deltas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressTrend {
    Improving,
    Stable,
    Regressing,
    Stagnant,
    /// Not enough data to determine trend.
    Unknown,
}

// ---------------------------------------------------------------------------
// Progress Oracle
// ---------------------------------------------------------------------------

/// The Progress Oracle — tracks checkpoints and scores over time.
pub struct ProgressOracle {
    checkpoints: VecDeque<Checkpoint>,
    scores: VecDeque<ProgressScore>,
    max_checkpoints: usize,
    weights: ScoreWeights,
    seq_counter: u64,
}

impl ProgressOracle {
    /// Create a new oracle with explicit capacity and weights.
    pub fn new(max_checkpoints: usize, weights: ScoreWeights) -> Self {
        Self {
            checkpoints: VecDeque::with_capacity(max_checkpoints),
            scores: VecDeque::with_capacity(max_checkpoints),
            max_checkpoints,
            weights,
            seq_counter: 0,
        }
    }

    /// Create a new oracle with default weights and a capacity of 100.
    pub fn with_defaults() -> Self {
        Self::new(100, ScoreWeights::default())
    }

    /// Record a checkpoint, compute and store its score, return the score.
    ///
    /// The checkpoint's `seq` field is overwritten with the next internal
    /// sequence number so ordering is always consistent.
    pub fn record_checkpoint(&mut self, mut checkpoint: Checkpoint) -> ProgressScore {
        // Assign monotonic seq
        self.seq_counter += 1;
        checkpoint.seq = self.seq_counter;

        let dimensions = Self::compute_score(&checkpoint, &self.weights);

        let weighted_score = dimensions.build * self.weights.build
            + dimensions.tests * self.weights.tests
            + dimensions.coverage * self.weights.coverage
            + dimensions.steps * self.weights.steps;

        let delta = self.scores.back().map(|prev| weighted_score - prev.score);

        let progress_score = ProgressScore {
            score: weighted_score,
            delta,
            dimensions,
        };

        // Evict oldest if at capacity
        if self.checkpoints.len() == self.max_checkpoints {
            self.checkpoints.pop_front();
            self.scores.pop_front();
        }

        self.checkpoints.push_back(checkpoint);
        self.scores.push_back(progress_score.clone());

        progress_score
    }

    /// Compute per-dimension scores from a checkpoint.
    pub fn compute_score(checkpoint: &Checkpoint, _weights: &ScoreWeights) -> ScoreDimensions {
        let build = if checkpoint.build_errors_count == 0 {
            1.0
        } else {
            0.0
        };

        let tests = if checkpoint.tests_total == 0 {
            // No tests discovered yet — treat as neutral (0).
            0.0
        } else {
            checkpoint.tests_passing as f64 / checkpoint.tests_total as f64
        };

        let coverage = (checkpoint.coverage_pct / 100.0).clamp(0.0, 1.0);

        let steps = if checkpoint.steps_total == 0 {
            0.0
        } else {
            checkpoint.steps_completed as f64 / checkpoint.steps_total as f64
        };

        ScoreDimensions {
            build,
            tests,
            coverage,
            steps,
        }
    }

    /// Return the most recent score, if any.
    pub fn latest_score(&self) -> Option<&ProgressScore> {
        self.scores.back()
    }

    /// Returns `true` if the last `k` scores all have `delta <= 0`.
    ///
    /// Returns `false` if fewer than `k` scores exist or `k == 0`.
    pub fn is_stagnant(&self, k: usize) -> bool {
        if k == 0 || self.scores.len() < k {
            return false;
        }

        self.scores
            .iter()
            .rev()
            .take(k)
            .all(|s| matches!(s.delta, Some(d) if d <= 0.0))
    }

    /// Build a summary of the current progress state.
    pub fn progress_summary(&self) -> ProgressSummary {
        let current_score = self
            .scores
            .back()
            .map(|s| s.score)
            .unwrap_or(0.0);

        let best_score = self
            .scores
            .iter()
            .map(|s| s.score)
            .fold(f64::NEG_INFINITY, f64::max);

        let worst_score = self
            .scores
            .iter()
            .map(|s| s.score)
            .fold(f64::INFINITY, f64::min);

        let trend = self.detect_trend();

        ProgressSummary {
            current_score,
            trend,
            total_checkpoints: self.checkpoints.len(),
            best_score: if self.scores.is_empty() { 0.0 } else { best_score },
            worst_score: if self.scores.is_empty() { 0.0 } else { worst_score },
        }
    }

    /// Read access to stored checkpoints.
    pub fn checkpoints(&self) -> &VecDeque<Checkpoint> {
        &self.checkpoints
    }

    /// Read access to stored scores.
    pub fn scores(&self) -> &VecDeque<ProgressScore> {
        &self.scores
    }

    // -- private helpers ----------------------------------------------------

    /// Detect trend from recent score deltas.
    ///
    /// Rules (looking at the last 3 scores' deltas):
    /// - All positive → `Improving`
    /// - All zero → `Stable`
    /// - All negative → `Regressing`
    /// - Mix of zero/negative for 5+ scores → `Stagnant`
    /// - Otherwise → `Unknown`
    fn detect_trend(&self) -> ProgressTrend {
        // Need at least 3 scores with deltas (i.e. 4 total checkpoints since
        // the first score has `delta = None`).
        let recent_deltas: Vec<f64> = self
            .scores
            .iter()
            .rev()
            .filter_map(|s| s.delta)
            .take(3)
            .collect();

        if recent_deltas.len() < 3 {
            return ProgressTrend::Unknown;
        }

        let all_positive = recent_deltas.iter().all(|&d| d > 0.0);
        let all_zero = recent_deltas.iter().all(|&d| d == 0.0);
        let all_negative = recent_deltas.iter().all(|&d| d < 0.0);

        if all_positive {
            return ProgressTrend::Improving;
        }
        if all_zero {
            return ProgressTrend::Stable;
        }
        if all_negative {
            return ProgressTrend::Regressing;
        }

        // Check for stagnation: mix of zero/negative over last 5+ deltas
        let stagnation_deltas: Vec<f64> = self
            .scores
            .iter()
            .rev()
            .filter_map(|s| s.delta)
            .take(5)
            .collect();

        if stagnation_deltas.len() >= 5
            && stagnation_deltas.iter().all(|&d| d <= 0.0)
        {
            return ProgressTrend::Stagnant;
        }

        ProgressTrend::Unknown
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    /// Helper: build a checkpoint with the given parameters.
    fn make_checkpoint(
        tests_total: usize,
        tests_passing: usize,
        build_errors: usize,
        coverage_pct: f64,
        steps_completed: usize,
        steps_total: usize,
    ) -> Checkpoint {
        Checkpoint {
            seq: 0,
            tests_total,
            tests_passing,
            tests_failing: tests_total.saturating_sub(tests_passing),
            build_errors_count: build_errors,
            coverage_pct,
            steps_completed,
            steps_total,
            warnings_count: 0,
            timestamp: Utc::now(),
        }
    }

    // -- Score computation --------------------------------------------------

    #[test]
    fn test_perfect_score() {
        let cp = make_checkpoint(10, 10, 0, 100.0, 5, 5);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());

        assert_eq!(dims.build, 1.0);
        assert_eq!(dims.tests, 1.0);
        assert_eq!(dims.coverage, 1.0);
        assert_eq!(dims.steps, 1.0);
    }

    #[test]
    fn test_zero_score() {
        let cp = make_checkpoint(10, 0, 5, 0.0, 0, 10);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());

        assert_eq!(dims.build, 0.0);
        assert_eq!(dims.tests, 0.0);
        assert_eq!(dims.coverage, 0.0);
        assert_eq!(dims.steps, 0.0);
    }

    #[test]
    fn test_partial_scores() {
        let cp = make_checkpoint(10, 7, 0, 80.0, 3, 10);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());

        assert_eq!(dims.build, 1.0);
        assert!((dims.tests - 0.7).abs() < 1e-9);
        assert!((dims.coverage - 0.8).abs() < 1e-9);
        assert!((dims.steps - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_no_tests_discovered() {
        let cp = make_checkpoint(0, 0, 0, 50.0, 1, 2);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());
        assert_eq!(dims.tests, 0.0);
    }

    #[test]
    fn test_no_steps() {
        let cp = make_checkpoint(5, 5, 0, 50.0, 0, 0);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());
        assert_eq!(dims.steps, 0.0);
    }

    #[test]
    fn test_coverage_clamped() {
        let cp = make_checkpoint(1, 1, 0, 150.0, 1, 1);
        let dims = ProgressOracle::compute_score(&cp, &ScoreWeights::default());
        assert_eq!(dims.coverage, 1.0);
    }

    // -- Delta computation --------------------------------------------------

    #[test]
    fn test_first_checkpoint_has_no_delta() {
        let mut oracle = ProgressOracle::with_defaults();
        let cp = make_checkpoint(10, 5, 0, 50.0, 1, 2);
        let score = oracle.record_checkpoint(cp);
        assert!(score.delta.is_none());
    }

    #[test]
    fn test_second_checkpoint_has_delta() {
        let mut oracle = ProgressOracle::with_defaults();

        let cp1 = make_checkpoint(10, 5, 0, 50.0, 1, 4);
        oracle.record_checkpoint(cp1);

        let cp2 = make_checkpoint(10, 8, 0, 70.0, 2, 4);
        let score2 = oracle.record_checkpoint(cp2);

        assert!(score2.delta.is_some());
        assert!(score2.delta.unwrap() > 0.0);
    }

    // -- Stagnation detection -----------------------------------------------

    #[test]
    fn test_not_stagnant_with_too_few_scores() {
        let mut oracle = ProgressOracle::with_defaults();
        let cp = make_checkpoint(10, 5, 0, 50.0, 1, 2);
        oracle.record_checkpoint(cp);
        assert!(!oracle.is_stagnant(3));
    }

    #[test]
    fn test_stagnant_when_no_improvement() {
        let mut oracle = ProgressOracle::with_defaults();

        // Record several identical checkpoints — all deltas after the first
        // will be 0.
        for _ in 0..5 {
            let cp = make_checkpoint(10, 5, 0, 50.0, 1, 2);
            oracle.record_checkpoint(cp);
        }

        // The first score has delta = None, so is_stagnant requires deltas <=0.
        // Scores 2..5 have delta = 0, so k=4 should be stagnant.
        assert!(oracle.is_stagnant(4));
    }

    #[test]
    fn test_not_stagnant_when_improving() {
        let mut oracle = ProgressOracle::with_defaults();

        for i in 0..5 {
            let cp = make_checkpoint(10, i * 2, 0, (i as f64) * 20.0, i, 10);
            oracle.record_checkpoint(cp);
        }

        assert!(!oracle.is_stagnant(3));
    }

    #[test]
    fn test_is_stagnant_k_zero() {
        let oracle = ProgressOracle::with_defaults();
        assert!(!oracle.is_stagnant(0));
    }

    // -- Trend detection ----------------------------------------------------

    #[test]
    fn test_trend_unknown_with_few_data() {
        let mut oracle = ProgressOracle::with_defaults();
        let cp = make_checkpoint(10, 5, 0, 50.0, 1, 2);
        oracle.record_checkpoint(cp);

        let summary = oracle.progress_summary();
        assert_eq!(summary.trend, ProgressTrend::Unknown);
    }

    #[test]
    fn test_trend_improving() {
        let mut oracle = ProgressOracle::with_defaults();

        // Need 4 checkpoints so that 3 scores have deltas.
        for i in 0..4 {
            let passing = 2 + i * 2; // 2, 4, 6, 8
            let cp = make_checkpoint(10, passing, 0, (passing as f64) * 10.0, i + 1, 10);
            oracle.record_checkpoint(cp);
        }

        let summary = oracle.progress_summary();
        assert_eq!(summary.trend, ProgressTrend::Improving);
    }

    #[test]
    fn test_trend_regressing() {
        let mut oracle = ProgressOracle::with_defaults();

        for i in 0..4 {
            let passing = 10 - i * 2; // 10, 8, 6, 4
            let cov = (passing as f64) * 10.0;
            let steps = 10 - i;
            let cp = make_checkpoint(10, passing, 0, cov, steps, 10);
            oracle.record_checkpoint(cp);
        }

        let summary = oracle.progress_summary();
        assert_eq!(summary.trend, ProgressTrend::Regressing);
    }

    #[test]
    fn test_trend_stable() {
        let mut oracle = ProgressOracle::with_defaults();

        for _ in 0..4 {
            let cp = make_checkpoint(10, 5, 0, 50.0, 3, 10);
            oracle.record_checkpoint(cp);
        }

        let summary = oracle.progress_summary();
        assert_eq!(summary.trend, ProgressTrend::Stable);
    }

    #[test]
    fn test_trend_stagnant() {
        let mut oracle = ProgressOracle::with_defaults();

        // Start with a good score, then regress/stagnate for 6+ checkpoints
        // so that the last 5 deltas are all <= 0.
        oracle.record_checkpoint(make_checkpoint(10, 8, 0, 80.0, 5, 10));

        for _ in 0..6 {
            // same or slightly worse
            oracle.record_checkpoint(make_checkpoint(10, 7, 0, 70.0, 4, 10));
        }

        let summary = oracle.progress_summary();
        assert_eq!(summary.trend, ProgressTrend::Stagnant);
    }

    // -- Max checkpoints / eviction -----------------------------------------

    #[test]
    fn test_max_checkpoints_eviction() {
        let mut oracle = ProgressOracle::new(3, ScoreWeights::default());

        for i in 0..5 {
            let cp = make_checkpoint(10, i * 2, 0, 50.0, 1, 2);
            oracle.record_checkpoint(cp);
        }

        assert_eq!(oracle.checkpoints().len(), 3);
        assert_eq!(oracle.scores().len(), 3);

        // The oldest retained checkpoint should have seq = 3
        assert_eq!(oracle.checkpoints().front().unwrap().seq, 3);
    }

    // -- with_defaults ------------------------------------------------------

    #[test]
    fn test_with_defaults() {
        let oracle = ProgressOracle::with_defaults();
        assert_eq!(oracle.checkpoints().len(), 0);
        assert_eq!(oracle.scores().len(), 0);
        assert!(oracle.latest_score().is_none());
    }

    // -- Summary best/worst -------------------------------------------------

    #[test]
    fn test_summary_best_worst() {
        let mut oracle = ProgressOracle::with_defaults();

        oracle.record_checkpoint(make_checkpoint(10, 2, 0, 20.0, 1, 10));
        oracle.record_checkpoint(make_checkpoint(10, 10, 0, 90.0, 8, 10));
        oracle.record_checkpoint(make_checkpoint(10, 5, 0, 50.0, 4, 10));

        let summary = oracle.progress_summary();
        assert!(summary.best_score >= summary.worst_score);
        assert_eq!(summary.total_checkpoints, 3);
    }

    #[test]
    fn test_empty_summary() {
        let oracle = ProgressOracle::with_defaults();
        let summary = oracle.progress_summary();
        assert_eq!(summary.current_score, 0.0);
        assert_eq!(summary.best_score, 0.0);
        assert_eq!(summary.worst_score, 0.0);
        assert_eq!(summary.trend, ProgressTrend::Unknown);
        assert_eq!(summary.total_checkpoints, 0);
    }
}
