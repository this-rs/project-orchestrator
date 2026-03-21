//! System-level prediction confidence scoring.
//!
//! Provides confidence scores for PO system predictions (impact analysis,
//! missing link prediction) — distinct from runner task quality scores.
//! Confidence is derived from graph density, signal convergence, and
//! calibration feedback.
//!
//! # References
//! - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

// ============================================================================
// Core types
// ============================================================================

/// Basis on which a confidence score was computed.
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceBasis {
    /// Confidence derived from local graph density (edges / nodes in k-hop neighborhood)
    GraphDensity,
    /// Confidence derived from convergence of multiple independent signals
    SignalConvergence,
    /// Confidence derived from variance across sample predictions
    SampleVariance,
    /// Weighted combination of multiple bases
    Composite,
}

/// A system-level prediction confidence score.
///
/// Represents how confident the PO is about a specific prediction result
/// (e.g., impact analysis, missing link prediction). This is NOT about
/// runner task quality — it is about the system's own prediction reliability.
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Confidence value in [0.0, 1.0]
    pub score: f64,
    /// What the confidence is based on
    pub basis: ConfidenceBasis,
    /// Number of data points used to compute this score
    pub sample_size: usize,
}

impl ConfidenceScore {
    /// Create a new confidence score, clamping to [0.0, 1.0].
    pub fn new(score: f64, basis: ConfidenceBasis, sample_size: usize) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            basis,
            sample_size,
        }
    }
}

// ============================================================================
// Confidence computations
// ============================================================================

/// Compute confidence from local graph density in a k=2 neighborhood.
///
/// Dense neighborhoods (many edges relative to nodes) yield higher confidence
/// because there is more structural evidence for the prediction.
///
/// - Dense graph (>10 edges) -> confidence > 0.8
/// - Sparse graph (<=2 edges) -> confidence < 0.4
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
pub fn confidence_from_graph_density(edge_count: usize, node_count: usize) -> ConfidenceScore {
    let sample_size = node_count + edge_count;
    if node_count == 0 {
        return ConfidenceScore::new(0.0, ConfidenceBasis::GraphDensity, 0);
    }

    // Density-based score: sigmoid-like mapping of edge count
    // For >10 edges: score > 0.8; for <=2 edges: score < 0.4
    let score = if edge_count > 10 {
        // High density: 0.8 + up to 0.2 based on how far above 10
        let extra = ((edge_count as f64 - 10.0) / 20.0).min(1.0);
        0.8 + 0.2 * extra
    } else if edge_count <= 2 {
        // Sparse: scale from 0.0 to 0.35
        edge_count as f64 * 0.175
    } else {
        // Mid range (3..=10): linear interpolation from 0.4 to 0.8
        let t = (edge_count as f64 - 2.0) / 8.0;
        0.4 + t * 0.4
    };

    ConfidenceScore::new(score, ConfidenceBasis::GraphDensity, sample_size)
}

/// Compute confidence from signal convergence in link prediction.
///
/// When multiple independent signals (co-change, structural, community/DNA,
/// Jaccard, Adamic-Adar) agree, confidence is higher.
///
/// # Arguments
/// * `signal_scores` — individual signal scores (each in [0, 1])
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
pub fn confidence_from_signal_convergence(signal_scores: &[(String, f64)]) -> ConfidenceScore {
    let n = signal_scores.len();
    if n == 0 {
        return ConfidenceScore::new(0.0, ConfidenceBasis::SignalConvergence, 0);
    }

    // Count how many signals are "active" (> 0.1 threshold)
    let active_count = signal_scores.iter().filter(|(_, s)| *s > 0.1).count();

    // Compute mean of active signals
    let active_signals: Vec<f64> = signal_scores
        .iter()
        .filter(|(_, s)| *s > 0.1)
        .map(|(_, s)| *s)
        .collect();

    if active_count == 0 {
        return ConfidenceScore::new(0.1, ConfidenceBasis::SignalConvergence, n);
    }

    let mean = active_signals.iter().sum::<f64>() / active_count as f64;

    // Variance of active signals (lower variance = higher convergence)
    let variance = if active_count > 1 {
        active_signals
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / active_count as f64
    } else {
        0.0
    };

    // Convergence factor: low variance boosts confidence
    let convergence_bonus = (1.0 - variance.sqrt()).max(0.0) * 0.3;

    // Coverage factor: more active signals = more confidence
    let coverage = active_count as f64 / n as f64;

    let score = (mean * 0.4 + coverage * 0.3 + convergence_bonus).min(1.0);

    ConfidenceScore::new(score, ConfidenceBasis::SignalConvergence, n)
}

// ============================================================================
// Aggregated system confidence
// ============================================================================

/// Aggregated system confidence across recent predictions.
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfidence {
    /// Weighted average confidence score
    pub average_score: f64,
    /// Number of predictions in the window
    pub prediction_count: usize,
    /// Calibration bias (positive = overconfident, negative = underconfident)
    pub calibration_bias: f64,
    /// Breakdown by basis type
    pub breakdown: Vec<BasisBreakdown>,
}

/// Per-basis confidence breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisBreakdown {
    pub basis: ConfidenceBasis,
    pub average_score: f64,
    pub count: usize,
}

/// Feedback on a prediction outcome (confirmed or refuted).
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFeedback {
    /// Identifier of the prediction being evaluated
    pub prediction_id: String,
    /// Whether the prediction was confirmed or refuted
    pub actual_outcome: PredictionOutcome,
}

/// Outcome of a prediction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PredictionOutcome {
    Confirmed,
    Refuted,
}

/// In-memory store for confidence tracking and calibration.
///
/// Maintains a rolling window of recent predictions and feedback
/// for computing aggregated system confidence with bias correction.
///
/// # References
/// - ELL (2025) — "Experience-driven Lifelong Learning" — 4th pillar: self-evaluation
pub struct ConfidenceTracker {
    /// Rolling window of recent confidence scores
    recent_scores: Mutex<VecDeque<ConfidenceScore>>,
    /// Rolling window of feedback outcomes (predicted_score, was_correct)
    feedback_log: Mutex<VecDeque<(f64, bool)>>,
    /// Maximum window size
    max_window: usize,
}

impl ConfidenceTracker {
    /// Create a new tracker with the given window size.
    pub fn new(max_window: usize) -> Self {
        Self {
            recent_scores: Mutex::new(VecDeque::with_capacity(max_window)),
            feedback_log: Mutex::new(VecDeque::with_capacity(max_window)),
            max_window,
        }
    }

    /// Record a new prediction confidence score.
    pub fn record(&self, score: ConfidenceScore) {
        let mut scores = self.recent_scores.lock().unwrap();
        if scores.len() >= self.max_window {
            scores.pop_front();
        }
        scores.push_back(score);
    }

    /// Record feedback on a prediction.
    /// `predicted_score` is the confidence that was assigned; `confirmed` indicates
    /// whether the prediction turned out to be correct.
    pub fn record_feedback(&self, predicted_score: f64, confirmed: bool) {
        let mut log = self.feedback_log.lock().unwrap();
        if log.len() >= self.max_window {
            log.pop_front();
        }
        log.push_back((predicted_score, confirmed));
    }

    /// Compute aggregated system confidence from recent predictions.
    pub fn aggregate(&self) -> SystemConfidence {
        let scores = self.recent_scores.lock().unwrap();
        let feedback = self.feedback_log.lock().unwrap();

        if scores.is_empty() {
            return SystemConfidence {
                average_score: 0.0,
                prediction_count: 0,
                calibration_bias: 0.0,
                breakdown: Vec::new(),
            };
        }

        // Weighted average (more recent predictions weighted higher)
        let n = scores.len();
        let mut weight_sum = 0.0;
        let mut weighted_score = 0.0;
        for (i, s) in scores.iter().enumerate() {
            let w = 1.0 + (i as f64 / n as f64); // 1.0 to 2.0
            weighted_score += s.score * w;
            weight_sum += w;
        }
        let average_score = if weight_sum > 0.0 {
            weighted_score / weight_sum
        } else {
            0.0
        };

        // Calibration bias: mean(predicted - actual_binary)
        // Positive = overconfident, negative = underconfident
        let calibration_bias = if feedback.is_empty() {
            0.0
        } else {
            let sum: f64 = feedback
                .iter()
                .map(|(pred, ok)| pred - if *ok { 1.0 } else { 0.0 })
                .sum();
            sum / feedback.len() as f64
        };

        // Breakdown by basis
        let mut density_scores = Vec::new();
        let mut convergence_scores = Vec::new();
        let mut variance_scores = Vec::new();
        let mut composite_scores = Vec::new();

        for s in scores.iter() {
            match s.basis {
                ConfidenceBasis::GraphDensity => density_scores.push(s.score),
                ConfidenceBasis::SignalConvergence => convergence_scores.push(s.score),
                ConfidenceBasis::SampleVariance => variance_scores.push(s.score),
                ConfidenceBasis::Composite => composite_scores.push(s.score),
            }
        }

        let mut breakdown = Vec::new();
        for (basis, vals) in [
            (ConfidenceBasis::GraphDensity, &density_scores),
            (ConfidenceBasis::SignalConvergence, &convergence_scores),
            (ConfidenceBasis::SampleVariance, &variance_scores),
            (ConfidenceBasis::Composite, &composite_scores),
        ] {
            if !vals.is_empty() {
                let avg = vals.iter().sum::<f64>() / vals.len() as f64;
                breakdown.push(BasisBreakdown {
                    basis,
                    average_score: avg,
                    count: vals.len(),
                });
            }
        }

        SystemConfidence {
            average_score,
            prediction_count: n,
            calibration_bias,
            breakdown,
        }
    }
}

impl Default for ConfidenceTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_from_graph_density_sparse() {
        let c = confidence_from_graph_density(2, 3);
        assert!(c.score < 0.4, "sparse graph should have confidence < 0.4, got {}", c.score);
        assert_eq!(c.basis, ConfidenceBasis::GraphDensity);
    }

    #[test]
    fn test_confidence_from_graph_density_dense() {
        let c = confidence_from_graph_density(15, 5);
        assert!(c.score > 0.8, "dense graph should have confidence > 0.8, got {}", c.score);
    }

    #[test]
    fn test_confidence_from_graph_density_empty() {
        let c = confidence_from_graph_density(0, 0);
        assert_eq!(c.score, 0.0);
    }

    #[test]
    fn test_signal_convergence_all_active() {
        let signals = vec![
            ("jaccard".to_string(), 0.7),
            ("co_change".to_string(), 0.8),
            ("structural".to_string(), 0.6),
        ];
        let c = confidence_from_signal_convergence(&signals);
        assert!(c.score > 0.5, "convergent signals should yield decent confidence, got {}", c.score);
        assert_eq!(c.basis, ConfidenceBasis::SignalConvergence);
    }

    #[test]
    fn test_signal_convergence_empty() {
        let c = confidence_from_signal_convergence(&[]);
        assert_eq!(c.score, 0.0);
    }

    #[test]
    fn test_tracker_aggregate_empty() {
        let tracker = ConfidenceTracker::new(10);
        let agg = tracker.aggregate();
        assert_eq!(agg.prediction_count, 0);
        assert_eq!(agg.average_score, 0.0);
    }

    #[test]
    fn test_tracker_record_and_aggregate() {
        let tracker = ConfidenceTracker::new(10);
        tracker.record(ConfidenceScore::new(0.8, ConfidenceBasis::GraphDensity, 5));
        tracker.record(ConfidenceScore::new(0.6, ConfidenceBasis::SignalConvergence, 3));

        let agg = tracker.aggregate();
        assert_eq!(agg.prediction_count, 2);
        assert!(agg.average_score > 0.0);
        assert_eq!(agg.breakdown.len(), 2);
    }

    #[test]
    fn test_tracker_calibration_bias() {
        let tracker = ConfidenceTracker::new(10);
        // Overconfident: predicted 0.9 but refuted
        tracker.record_feedback(0.9, false);
        // Well-calibrated: predicted 0.8 and confirmed
        tracker.record_feedback(0.8, true);
        tracker.record(ConfidenceScore::new(0.85, ConfidenceBasis::Composite, 2));

        let agg = tracker.aggregate();
        // bias = mean( (0.9 - 0.0) + (0.8 - 1.0) ) / 2 = (0.9 + -0.2) / 2 = 0.35
        assert!((agg.calibration_bias - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_tracker_window_eviction() {
        let tracker = ConfidenceTracker::new(3);
        for i in 0..5 {
            tracker.record(ConfidenceScore::new(i as f64 * 0.2, ConfidenceBasis::GraphDensity, 1));
        }
        let agg = tracker.aggregate();
        assert_eq!(agg.prediction_count, 3, "should evict oldest entries");
    }
}
