//! Benchmark metrics: Recall@K, MRR, F1, AUC-ROC — all in Rust, no external deps.

use serde::{Deserialize, Serialize};

/// Set of evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricSet {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_5: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_10: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auc_roc: Option<f64>,
}

// ---------------------------------------------------------------------------
// Recall@K
// ---------------------------------------------------------------------------

/// Compute Recall@K: fraction of queries where the correct answer is in top-K.
///
/// * `ranked_results` — for each query, ordered list of (item_id, is_relevant)
/// * `k` — cutoff
pub fn recall_at_k(ranked_results: &[Vec<(usize, bool)>], k: usize) -> f64 {
    if ranked_results.is_empty() || k == 0 {
        return 0.0;
    }

    let hits: usize = ranked_results
        .iter()
        .filter(|results| results.iter().take(k).any(|(_, rel)| *rel))
        .count();

    hits as f64 / ranked_results.len() as f64
}

// ---------------------------------------------------------------------------
// MRR (Mean Reciprocal Rank)
// ---------------------------------------------------------------------------

/// Compute Mean Reciprocal Rank.
///
/// For each query, the reciprocal rank of the first relevant result.
pub fn mrr(ranked_results: &[Vec<(usize, bool)>]) -> f64 {
    if ranked_results.is_empty() {
        return 0.0;
    }

    let sum: f64 = ranked_results
        .iter()
        .map(|results| {
            results
                .iter()
                .position(|(_, rel)| *rel)
                .map(|pos| 1.0 / (pos + 1) as f64)
                .unwrap_or(0.0)
        })
        .sum();

    sum / ranked_results.len() as f64
}

// ---------------------------------------------------------------------------
// F1 score (binary classification)
// ---------------------------------------------------------------------------

/// Compute precision, recall, and F1 from binary predictions.
///
/// * `predictions` — list of (predicted_positive, actual_positive)
pub fn f1_score(predictions: &[(bool, bool)]) -> (f64, f64, f64) {
    if predictions.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;

    for &(pred, actual) in predictions {
        match (pred, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

// ---------------------------------------------------------------------------
// AUC-ROC (sort + trapezoid)
// ---------------------------------------------------------------------------

/// Compute AUC-ROC via sort + trapezoid rule.
///
/// * `scores` — list of (score, is_positive)
pub fn auc_roc(scores: &[(f64, bool)]) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }

    let total_pos = scores.iter().filter(|(_, p)| *p).count() as f64;
    let total_neg = scores.iter().filter(|(_, p)| !*p).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5;
    }

    let mut sorted: Vec<(f64, bool)> = scores.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;
    let mut auc = 0.0;

    for &(_, is_pos) in &sorted {
        if is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    auc
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&va, &vb) in a.iter().zip(b.iter()) {
        dot += va as f64 * vb as f64;
        norm_a += va as f64 * va as f64;
        norm_b += vb as f64 * vb as f64;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    dot / denom
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let ranked = vec![
            vec![(0, false), (1, true), (2, false)],  // hit at rank 2
            vec![(0, true), (1, false), (2, false)],   // hit at rank 1
            vec![(0, false), (1, false), (2, false)],  // no hit
        ];

        assert!((recall_at_k(&ranked, 1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((recall_at_k(&ranked, 2) - 2.0 / 3.0).abs() < 1e-10);
        assert!((recall_at_k(&ranked, 3) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mrr() {
        let ranked = vec![
            vec![(0, false), (1, true)],  // first relevant at rank 2 → 1/2
            vec![(0, true)],               // first relevant at rank 1 → 1/1
            vec![(0, false), (1, false)],  // no relevant → 0
        ];

        let result = mrr(&ranked);
        let expected = (0.5 + 1.0 + 0.0) / 3.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let predictions = vec![
            (true, true),   // TP
            (true, true),   // TP
            (true, false),  // FP
            (false, true),  // FN
            (false, false), // TN
        ];

        let (precision, recall, f1) = f1_score(&predictions);
        // precision = 2/3, recall = 2/3, f1 = 2/3
        assert!((precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((recall - 2.0 / 3.0).abs() < 1e-10);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_roc_perfect() {
        let scores = vec![(0.9, true), (0.8, true), (0.2, false), (0.1, false)];
        let auc = auc_roc(&scores);
        assert!((auc - 1.0).abs() < 1e-10, "Perfect separation → AUC=1.0, got {}", auc);
    }

    #[test]
    fn test_auc_roc_random() {
        // Interleaved scores where positives and negatives overlap
        let scores = vec![
            (0.9, true),
            (0.8, false),
            (0.7, true),
            (0.6, false),
            (0.5, true),
            (0.4, false),
            (0.3, true),
            (0.2, false),
        ];
        let auc = auc_roc(&scores);
        // With perfectly interleaved scores, AUC should be close to 0.5
        assert!((auc - 0.5).abs() < 0.15, "Interleaved scores → AUC≈0.5, got {}", auc);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }
}
