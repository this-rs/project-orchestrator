//! Shared utility helpers.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::HashMap;

pub mod file_path_extractor;
pub mod paths;

/// Stable equivalent of the nightly-only `str::floor_char_boundary`.
/// Returns the largest byte index `<= index` that is a valid UTF-8 char boundary.
///
/// Useful for safely truncating a `&str` at a maximum byte length without
/// panicking on multi-byte characters.
pub(crate) fn floor_char_boundary(s: &str, index: usize) -> usize {
    let index = index.min(s.len());
    (0..=index)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

/// Smoothing constant of Reciprocal Rank Fusion (`1 / (RRF_K + rank)`).
///
/// Cormack, Clarke & Buttcher (2009) use 60 and report that the result is
/// insensitive to it: it only damps how much the very first ranks dominate.
pub(crate) const RRF_K: f64 = 60.0;

/// Fuse ranked candidate lists by Reciprocal Rank Fusion.
///
/// Each channel contributes `1 / (RRF_K + rank)` (rank starting at 1) for every
/// candidate it lists, and a candidate's score is the sum over the channels that
/// list it. Summing ranks rather than scores is what makes this work: BM25
/// scores are unbounded and differ per query, cosine lives in [-1, 1], and no
/// affine map between them is meaningful - ranks are comparable by construction.
/// A candidate both channels rank highly therefore beats one only a single
/// channel likes, which is the case worth promoting.
///
/// The result is rescaled so the best fused candidate scores `anchor`, keeping
/// the head of the list on the scale the caller already reads - consensus
/// changes the *order*, not the currency.
///
/// Returns `(id, score)` sorted by descending score, ties broken by id so the
/// order is deterministic (a `HashMap` iteration is not).
pub(crate) fn reciprocal_rank_fusion(channels: &[&[String]], anchor: f64) -> Vec<(String, f64)> {
    let mut fused: HashMap<&str, f64> = HashMap::new();
    for channel in channels {
        for (index, id) in channel.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + (index as f64) + 1.0);
            *fused.entry(id.as_str()).or_insert(0.0) += contribution;
        }
    }
    let peak = fused.values().copied().fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let mut out: Vec<(String, f64)> = fused
        .into_iter()
        .map(|(id, score)| (id.to_string(), score * anchor / peak))
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    // -- Reciprocal Rank Fusion --------------------------------------------

    /// A flat dense channel must yield to a discriminative lexical one: this is
    /// the whole point of fusing, because closely clustered cosine scores barely
    /// separate anything while a lexical first place is a strong signal.
    #[test]
    fn flat_dense_yields_to_the_lexical_channel() {
        let dense = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let lexical = vec!["b".to_string()];
        let fused = reciprocal_rank_fusion(&[&dense, &lexical], 0.92);
        assert_eq!(fused[0].0, "b", "lexical consensus promotes b: {fused:?}");
        assert!(
            (fused[0].1 - 0.92).abs() < 1e-9,
            "head keeps the anchor scale, got {}",
            fused[0].1
        );
    }

    /// A note both channels like outranks one only a single channel leads.
    #[test]
    fn consensus_beats_a_single_channel_lead() {
        let dense = vec!["d".to_string(), "e".to_string()];
        let lexical = vec!["f".to_string(), "e".to_string()];
        let fused = reciprocal_rank_fusion(&[&dense, &lexical], 1.0);
        assert_eq!(fused[0].0, "e", "e is in both channels: {fused:?}");
    }

    #[test]
    fn empty_channels_give_empty() {
        assert!(reciprocal_rank_fusion(&[], 1.0).is_empty());
        let empty: Vec<String> = vec![];
        assert!(reciprocal_rank_fusion(&[&empty, &empty], 1.0).is_empty());
    }

    #[test]
    fn test_floor_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(floor_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_floor_char_boundary_multibyte() {
        // "é" is 2 bytes (0xC3 0xA9); boundary at byte 1 is invalid
        let s = "héllo";
        assert_eq!(floor_char_boundary(s, 2), 1); // step back to valid boundary
        assert_eq!(floor_char_boundary(s, 3), 3);
    }

    #[test]
    fn test_floor_char_boundary_beyond_len() {
        let s = "hi";
        assert_eq!(floor_char_boundary(s, 100), 2);
    }
}
