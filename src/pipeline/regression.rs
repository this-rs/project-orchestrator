//! # Regression Detector — Loop detection and convergence guards
//!
//! Analyzes errors during pipeline execution to detect:
//! - Loops (same error seen N+ times)
//! - Regressions (a test that was passing now fails)
//! - Stagnation (no progress over K iterations)

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

// ─── Error signature ────────────────────────────────────────────────────────

/// Signature of an error for dedup detection.
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct ErrorSignature {
    /// Hash of the error message (normalized).
    pub message_hash: u64,
    /// File where the error occurred (if known).
    pub file: Option<String>,
    /// Line number (if known).
    pub line: Option<u32>,
    /// Error category (compile, test, lint, etc.).
    pub category: ErrorCategory,
}

impl PartialEq for ErrorSignature {
    fn eq(&self, other: &Self) -> bool {
        self.message_hash == other.message_hash
            && self.file == other.file
            && self.line == other.line
            && self.category == other.category
    }
}

impl Hash for ErrorSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.message_hash.hash(state);
        self.file.hash(state);
        self.line.hash(state);
        self.category.hash(state);
    }
}

impl ErrorSignature {
    /// Build a signature from a raw error message.
    ///
    /// The message is normalized (trimmed, lowercased) then hashed via FxHash-style
    /// computation so that minor whitespace differences collapse into the same sig.
    pub fn from_message(
        message: &str,
        file: Option<&str>,
        line: Option<u32>,
        category: ErrorCategory,
    ) -> Self {
        let normalized = message.trim().to_lowercase();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        normalized.hash(&mut hasher);
        let message_hash = hasher.finish();

        Self {
            message_hash,
            file: file.map(|s| s.to_owned()),
            line,
            category,
        }
    }
}

// ─── Error category ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    Compile,
    Test,
    Lint,
    Runtime,
    Coverage,
    Other,
}

// ─── Stop reason ────────────────────────────────────────────────────────────

/// Why the pipeline should stop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    /// Continue execution (no stop needed).
    Continue,
    /// Same error seen too many times — we are in a loop.
    Loop {
        error: ErrorSignature,
        seen_count: usize,
    },
    /// A previously passing test is now failing.
    Regression { test_name: String },
    /// No progress over K consecutive checkpoints.
    Stagnation { iterations_without_progress: usize },
}

// ─── Progress checkpoint ────────────────────────────────────────────────────

/// Metrics checkpoint at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressCheckpoint {
    pub tests_passing: usize,
    pub tests_failing: usize,
    pub build_errors: usize,
    pub coverage_pct: f64,
    pub warnings: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Names of tests that are currently failing (for regression detection).
    #[serde(default)]
    pub failing_test_names: Vec<String>,
}

impl ProgressCheckpoint {
    /// Calculate a simple progress score in `[0, 1]`.
    pub fn score(&self) -> f64 {
        let total_tests = self.tests_passing + self.tests_failing;
        if total_tests == 0 && self.build_errors == 0 {
            return 0.5; // no data
        }

        let test_score = if total_tests > 0 {
            self.tests_passing as f64 / total_tests as f64
        } else {
            0.0
        };

        let build_score = if self.build_errors == 0 { 1.0 } else { 0.0 };
        let coverage_score = self.coverage_pct / 100.0;

        // Weighted: build 40%, tests 40%, coverage 20%
        build_score * 0.4 + test_score * 0.4 + coverage_score * 0.2
    }
}

// ─── Regression Detector ────────────────────────────────────────────────────

/// The Regression Detector.
///
/// Tracks error signatures, progress checkpoints and test pass/fail status to
/// decide whether the pipeline should stop due to loops, regressions or stagnation.
pub struct RegressionDetector {
    /// Errors seen so far: signature -> count.
    error_history: HashMap<ErrorSignature, usize>,
    /// Insertion-order queue for evicting oldest errors when `max_errors` is reached.
    error_order: VecDeque<ErrorSignature>,
    /// Max errors to track (ring buffer behavior beyond this).
    max_errors: usize,
    /// Threshold: how many times the same error before we call it a loop.
    loop_threshold: usize,
    /// Previously passing tests (for regression detection).
    previously_passing: HashSet<String>,
    /// Progress checkpoints.
    checkpoints: Vec<ProgressCheckpoint>,
    /// Max checkpoints to keep.
    max_checkpoints: usize,
    /// How many stagnant checkpoints before we stop.
    stagnation_threshold: usize,
}

impl RegressionDetector {
    /// Create a new detector with the given limits.
    ///
    /// Defaults: `max_errors = 1000`, `max_checkpoints = 100`,
    /// `loop_threshold = 2`, `stagnation_threshold = 5`.
    pub fn new(
        max_errors: usize,
        max_checkpoints: usize,
        loop_threshold: usize,
        stagnation_threshold: usize,
    ) -> Self {
        Self {
            error_history: HashMap::new(),
            error_order: VecDeque::new(),
            max_errors,
            loop_threshold,
            previously_passing: HashSet::new(),
            checkpoints: Vec::new(),
            max_checkpoints,
            stagnation_threshold,
        }
    }

    /// Create a detector with default settings.
    pub fn default() -> Self {
        Self::new(1000, 100, 2, 5)
    }

    /// Record an error occurrence.
    ///
    /// If the error is new, it is added to history. If we exceed `max_errors`,
    /// the oldest error is evicted.
    pub fn record_error(&mut self, sig: ErrorSignature) {
        // If already tracked, just bump count.
        if let Some(count) = self.error_history.get_mut(&sig) {
            *count += 1;
            return;
        }

        // Evict oldest if at capacity.
        if self.error_order.len() >= self.max_errors {
            if let Some(oldest) = self.error_order.pop_front() {
                self.error_history.remove(&oldest);
            }
        }

        self.error_order.push_back(sig.clone());
        self.error_history.insert(sig, 1);
    }

    /// Record a progress checkpoint.
    ///
    /// Evicts the oldest checkpoint if we exceed `max_checkpoints`.
    pub fn record_checkpoint(&mut self, checkpoint: ProgressCheckpoint) {
        if self.checkpoints.len() >= self.max_checkpoints {
            self.checkpoints.remove(0);
        }
        self.checkpoints.push(checkpoint);
    }

    /// Record tests that are currently passing.
    ///
    /// These names are stored so that future checkpoints can detect regressions
    /// (a test that was passing is now failing).
    pub fn record_passing_tests(&mut self, tests: &[String]) {
        for t in tests {
            self.previously_passing.insert(t.clone());
        }
    }

    /// Evaluate whether the pipeline should stop.
    ///
    /// Checks three conditions in order:
    /// 1. **Loop**: any error seen >= `loop_threshold` times.
    /// 2. **Regression**: a test in `previously_passing` appears in the latest
    ///    checkpoint's `failing_test_names`.
    /// 3. **Stagnation**: last K checkpoints show no score improvement (delta <= 0).
    pub fn should_stop(&self) -> StopReason {
        // 1. Loop detection
        for (sig, &count) in &self.error_history {
            if count >= self.loop_threshold {
                return StopReason::Loop {
                    error: sig.clone(),
                    seen_count: count,
                };
            }
        }

        // 2. Regression detection
        if let Some(latest) = self.checkpoints.last() {
            for failing in &latest.failing_test_names {
                if self.previously_passing.contains(failing) {
                    return StopReason::Regression {
                        test_name: failing.clone(),
                    };
                }
            }
        }

        // 3. Stagnation detection
        if self.checkpoints.len() >= self.stagnation_threshold {
            let tail = &self.checkpoints[self.checkpoints.len() - self.stagnation_threshold..];
            let mut stagnant = true;
            for window in tail.windows(2) {
                if window[1].score() > window[0].score() {
                    stagnant = false;
                    break;
                }
            }
            if stagnant {
                return StopReason::Stagnation {
                    iterations_without_progress: self.stagnation_threshold,
                };
            }
        }

        StopReason::Continue
    }

    /// Clear all tracked state.
    pub fn reset(&mut self) {
        self.error_history.clear();
        self.error_order.clear();
        self.previously_passing.clear();
        self.checkpoints.clear();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_checkpoint(
        passing: usize,
        failing: usize,
        build_errors: usize,
        coverage: f64,
    ) -> ProgressCheckpoint {
        ProgressCheckpoint {
            tests_passing: passing,
            tests_failing: failing,
            build_errors,
            coverage_pct: coverage,
            warnings: 0,
            timestamp: Utc::now(),
            failing_test_names: Vec::new(),
        }
    }

    fn make_checkpoint_with_failures(
        passing: usize,
        failing: usize,
        failing_names: Vec<String>,
    ) -> ProgressCheckpoint {
        ProgressCheckpoint {
            tests_passing: passing,
            tests_failing: failing,
            build_errors: 0,
            coverage_pct: 80.0,
            warnings: 0,
            timestamp: Utc::now(),
            failing_test_names: failing_names,
        }
    }

    // ── Loop detection ──────────────────────────────────────────────────

    #[test]
    fn loop_detection_triggers_after_threshold() {
        let mut det = RegressionDetector::new(1000, 100, 3, 5);
        let sig = ErrorSignature::from_message(
            "cannot find value `x`",
            Some("main.rs"),
            Some(10),
            ErrorCategory::Compile,
        );

        det.record_error(sig.clone());
        assert_eq!(det.should_stop(), StopReason::Continue);

        det.record_error(sig.clone());
        assert_eq!(det.should_stop(), StopReason::Continue);

        det.record_error(sig.clone());
        match det.should_stop() {
            StopReason::Loop { seen_count, .. } => assert_eq!(seen_count, 3),
            other => panic!("expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn loop_detection_default_threshold_is_two() {
        let mut det = RegressionDetector::default();
        let sig = ErrorSignature::from_message("type mismatch", None, None, ErrorCategory::Compile);

        det.record_error(sig.clone());
        assert_eq!(det.should_stop(), StopReason::Continue);

        det.record_error(sig.clone());
        match det.should_stop() {
            StopReason::Loop { seen_count, .. } => assert_eq!(seen_count, 2),
            other => panic!("expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn different_errors_do_not_trigger_loop() {
        let mut det = RegressionDetector::new(1000, 100, 3, 5);

        det.record_error(ErrorSignature::from_message(
            "error A",
            None,
            None,
            ErrorCategory::Compile,
        ));
        det.record_error(ErrorSignature::from_message(
            "error B",
            None,
            None,
            ErrorCategory::Compile,
        ));
        det.record_error(ErrorSignature::from_message(
            "error C",
            None,
            None,
            ErrorCategory::Compile,
        ));

        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    // ── Regression detection ────────────────────────────────────────────

    #[test]
    fn regression_detected_when_passing_test_fails() {
        let mut det = RegressionDetector::default();

        det.record_passing_tests(&["test_login".to_string(), "test_signup".to_string()]);

        let cp = make_checkpoint_with_failures(1, 1, vec!["test_login".to_string()]);
        det.record_checkpoint(cp);

        match det.should_stop() {
            StopReason::Regression { test_name } => assert_eq!(test_name, "test_login"),
            other => panic!("expected Regression, got {:?}", other),
        }
    }

    #[test]
    fn no_regression_when_failing_test_was_never_passing() {
        let mut det = RegressionDetector::default();

        // Never recorded "test_new" as passing
        let cp = make_checkpoint_with_failures(5, 1, vec!["test_new".to_string()]);
        det.record_checkpoint(cp);

        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    // ── Stagnation detection ────────────────────────────────────────────

    #[test]
    fn stagnation_detected_after_flat_checkpoints() {
        let mut det = RegressionDetector::new(1000, 100, 10, 5);

        // 5 identical checkpoints → score never improves
        for _ in 0..5 {
            det.record_checkpoint(make_checkpoint(5, 5, 0, 50.0));
        }

        match det.should_stop() {
            StopReason::Stagnation {
                iterations_without_progress,
            } => {
                assert_eq!(iterations_without_progress, 5);
            }
            other => panic!("expected Stagnation, got {:?}", other),
        }
    }

    #[test]
    fn no_stagnation_when_progress_is_made() {
        let mut det = RegressionDetector::new(1000, 100, 10, 5);

        det.record_checkpoint(make_checkpoint(3, 7, 0, 30.0));
        det.record_checkpoint(make_checkpoint(4, 6, 0, 35.0));
        det.record_checkpoint(make_checkpoint(5, 5, 0, 40.0));
        det.record_checkpoint(make_checkpoint(6, 4, 0, 50.0));
        det.record_checkpoint(make_checkpoint(7, 3, 0, 60.0));

        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    #[test]
    fn stagnation_not_triggered_with_few_checkpoints() {
        let mut det = RegressionDetector::new(1000, 100, 10, 5);

        det.record_checkpoint(make_checkpoint(5, 5, 0, 50.0));
        det.record_checkpoint(make_checkpoint(5, 5, 0, 50.0));

        // Only 2 checkpoints, threshold is 5
        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    // ── Continue when everything is fine ─────────────────────────────────

    #[test]
    fn continue_when_no_issues() {
        let mut det = RegressionDetector::default();

        det.record_error(ErrorSignature::from_message(
            "some warning",
            None,
            None,
            ErrorCategory::Lint,
        ));
        det.record_checkpoint(make_checkpoint(10, 0, 0, 90.0));

        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    // ── Eviction ────────────────────────────────────────────────────────

    #[test]
    fn max_errors_eviction() {
        let mut det = RegressionDetector::new(3, 100, 2, 5);

        let sig_a = ErrorSignature::from_message("error A", None, None, ErrorCategory::Compile);
        let sig_b = ErrorSignature::from_message("error B", None, None, ErrorCategory::Compile);
        let sig_c = ErrorSignature::from_message("error C", None, None, ErrorCategory::Compile);
        let sig_d = ErrorSignature::from_message("error D", None, None, ErrorCategory::Compile);

        det.record_error(sig_a.clone());
        det.record_error(sig_b.clone());
        det.record_error(sig_c.clone());

        assert_eq!(det.error_history.len(), 3);

        // Adding a 4th should evict sig_a
        det.record_error(sig_d.clone());
        assert_eq!(det.error_history.len(), 3);
        assert!(!det.error_history.contains_key(&sig_a));
        assert!(det.error_history.contains_key(&sig_d));
    }

    #[test]
    fn max_checkpoints_eviction() {
        let mut det = RegressionDetector::new(1000, 3, 10, 5);

        det.record_checkpoint(make_checkpoint(1, 0, 0, 10.0));
        det.record_checkpoint(make_checkpoint(2, 0, 0, 20.0));
        det.record_checkpoint(make_checkpoint(3, 0, 0, 30.0));
        det.record_checkpoint(make_checkpoint(4, 0, 0, 40.0));

        assert_eq!(det.checkpoints.len(), 3);
        // The oldest (10% coverage) should have been evicted
        assert!((det.checkpoints[0].coverage_pct - 20.0).abs() < f64::EPSILON);
    }

    // ── Checkpoint scoring ──────────────────────────────────────────────

    #[test]
    fn score_perfect() {
        let cp = make_checkpoint(10, 0, 0, 100.0);
        assert!((cp.score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_no_data() {
        let cp = make_checkpoint(0, 0, 0, 0.0);
        assert!((cp.score() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn score_half_tests_no_coverage() {
        let cp = make_checkpoint(5, 5, 0, 0.0);
        // build=1.0*0.4 + test=0.5*0.4 + cov=0.0*0.2 = 0.6
        assert!((cp.score() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn score_build_errors_tank_score() {
        let cp = make_checkpoint(10, 0, 3, 100.0);
        // build=0.0*0.4 + test=1.0*0.4 + cov=1.0*0.2 = 0.6
        assert!((cp.score() - 0.6).abs() < f64::EPSILON);
    }

    // ── Reset ───────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_all_state() {
        let mut det = RegressionDetector::default();

        det.record_error(ErrorSignature::from_message(
            "err",
            None,
            None,
            ErrorCategory::Compile,
        ));
        det.record_error(ErrorSignature::from_message(
            "err",
            None,
            None,
            ErrorCategory::Compile,
        ));
        det.record_passing_tests(&["test_a".to_string()]);
        det.record_checkpoint(make_checkpoint(5, 5, 0, 50.0));

        det.reset();

        assert!(det.error_history.is_empty());
        assert!(det.error_order.is_empty());
        assert!(det.previously_passing.is_empty());
        assert!(det.checkpoints.is_empty());
        assert_eq!(det.should_stop(), StopReason::Continue);
    }

    // ── ErrorSignature::from_message ────────────────────────────────────

    #[test]
    fn from_message_normalizes_whitespace_and_case() {
        let a = ErrorSignature::from_message(
            "  Error FOO  ",
            Some("a.rs"),
            None,
            ErrorCategory::Compile,
        );
        let b =
            ErrorSignature::from_message("error foo", Some("a.rs"), None, ErrorCategory::Compile);
        assert_eq!(a, b);
    }

    #[test]
    fn from_message_different_category_not_equal() {
        let a = ErrorSignature::from_message("error foo", None, None, ErrorCategory::Compile);
        let b = ErrorSignature::from_message("error foo", None, None, ErrorCategory::Test);
        assert_ne!(a, b);
    }
}
