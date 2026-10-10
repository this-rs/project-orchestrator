//! Confidence and margin of a distribution over candidate scores.
//!
//! One implementation for the whole backend: routing sections, cognitive decisions,
//! skill activation and trigger ranking all call these pure functions. Nothing here
//! depends on any other crate.
//!
//! Both functions treat the scores as unnormalised weights: negative and non-finite
//! scores count as zero, and the rest is normalised to a probability distribution.

/// Probabilities of the scores, after clamping negatives and non-finite values to 0.
///
/// `None` when there is no positive mass (empty input, or every score is zero), i.e.
/// no distribution at all.
fn probabilities(scores: &[f64]) -> Option<Vec<f64>> {
    let clamped: Vec<f64> = scores
        .iter()
        .map(|s| if s.is_finite() { s.max(0.0) } else { 0.0 })
        .collect();
    let total: f64 = clamped.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    Some(clamped.into_iter().map(|s| s / total).collect())
}

/// Normalised Shannon entropy confidence `1 - H(p) / ln(k)`.
///
/// - Fewer than two scores (`k < 2`): `1.0`. `ln(1) = 0` makes the normalisation
///   undefined, and with a single outcome there is no ambiguity to measure.
/// - No positive mass (all zero, or all negative): `0.0`. There is no distribution.
/// - Equal scores: `0.0`, the minimum. Peaked distributions tend to `1.0`.
///
/// Measures how concentrated the scores are, not whether the choice is correct.
pub fn normalized_entropy_confidence(scores: &[f64]) -> f64 {
    let k = scores.len();
    if k < 2 {
        return 1.0;
    }
    let Some(probs) = probabilities(scores) else {
        return 0.0;
    };
    let entropy: f64 = probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - entropy / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Gap between the two largest normalised scores, `p(top1) - p(top2)`, in `[0, 1]`.
///
/// `0.0` when the top two are tied, `1.0` when one candidate holds all the mass.
/// A single candidate carrying mass has a gap of its full mass, `1.0`.
/// `0.0` when there is no distribution (no candidate, or no positive mass).
pub fn top_margin(scores: &[f64]) -> f64 {
    let Some(mut probs) = probabilities(scores) else {
        return 0.0;
    };
    probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = probs[0];
    let top2 = probs.get(1).copied().unwrap_or(0.0);
    (top1 - top2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn fewer_than_two_scores_is_full_confidence() {
        assert!(close(normalized_entropy_confidence(&[]), 1.0));
        assert!(close(normalized_entropy_confidence(&[0.7]), 1.0));
        assert!(close(normalized_entropy_confidence(&[0.0]), 1.0));
    }

    #[test]
    fn equal_scores_are_zero_confidence() {
        assert!(close(
            normalized_entropy_confidence(&[1.0, 1.0, 1.0, 1.0]),
            0.0
        ));
        assert!(close(normalized_entropy_confidence(&[0.3, 0.3]), 0.0));
    }

    #[test]
    fn no_positive_mass_is_zero_confidence() {
        assert!(close(normalized_entropy_confidence(&[0.0, 0.0, 0.0]), 0.0));
        assert!(close(normalized_entropy_confidence(&[-1.0, -2.0]), 0.0));
        assert!(close(normalized_entropy_confidence(&[f64::NAN, 0.0]), 0.0));
    }

    #[test]
    fn dominant_candidate_is_full_confidence() {
        assert!(close(normalized_entropy_confidence(&[1.0, 0.0, 0.0]), 1.0));
    }

    #[test]
    fn matches_closed_form() {
        // p = (0.75, 0.25): H = -(0.75 ln 0.75 + 0.25 ln 0.25), conf = 1 - H / ln 2.
        let h = -(0.75_f64 * 0.75_f64.ln() + 0.25_f64 * 0.25_f64.ln());
        let expected = 1.0 - h / 2.0_f64.ln();
        assert!(close(
            normalized_entropy_confidence(&[0.75, 0.25]),
            expected
        ));
        // Scale invariance: raw weights and their normalised form agree.
        assert!(close(
            normalized_entropy_confidence(&[300.0, 100.0]),
            normalized_entropy_confidence(&[0.75, 0.25])
        ));
    }

    #[test]
    fn negative_scores_are_clamped_to_zero_mass() {
        assert!(close(
            normalized_entropy_confidence(&[-5.0, 1.0, 1.0]),
            normalized_entropy_confidence(&[0.0, 1.0, 1.0])
        ));
    }

    #[test]
    fn margin_is_zero_when_top_two_are_tied() {
        assert!(close(top_margin(&[0.4, 0.4, 0.2]), 0.0));
        assert!(close(top_margin(&[1.0, 1.0, 0.0, 0.0]), 0.0));
        assert!(close(top_margin(&[2.0, 2.0]), 0.0));
    }

    #[test]
    fn margin_is_one_for_a_dominant_candidate() {
        assert!(close(top_margin(&[1.0, 0.0, 0.0]), 1.0));
        assert!(close(top_margin(&[5.0]), 1.0));
    }

    #[test]
    fn margin_is_the_gap_between_the_two_largest_probabilities() {
        // p = (0.5, 0.3, 0.2): gap 0.2.
        assert!(close(top_margin(&[5.0, 3.0, 2.0]), 0.2));
        // Order of the input does not matter.
        assert!(close(top_margin(&[2.0, 5.0, 3.0]), 0.2));
    }

    #[test]
    fn margin_without_distribution_is_zero() {
        assert!(close(top_margin(&[]), 0.0));
        assert!(close(top_margin(&[0.0, 0.0]), 0.0));
        assert!(close(top_margin(&[-1.0]), 0.0));
    }
}
