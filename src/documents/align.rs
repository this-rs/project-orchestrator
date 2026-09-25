//! Locating a quoted passage back in its source, with a quality verdict.
//!
//! Borrowed from LangExtract (Google, Apache-2.0), where every extraction
//! carries both its character offsets **and** an alignment status — and one that
//! cannot be located at all is reported as unlocated rather than as a result.
//! That is how they catch a model quoting its own few-shot examples instead of
//! the document.
//!
//! ## Why PO needs this
//!
//! Notes and decisions assert things about code and documents — "`manager.rs`
//! does X", quoting a line. Nothing ever re-checks the assertion. PO has
//! `staleness_score` and `last_confirmed_at`, but those measure **age**, not
//! validity: a note can be minutes old and already wrong, or a year old and
//! still exact.
//!
//! Demonstrated the hard way while writing this: a citation of
//! `routing.rs:367` written in the morning pointed at unrelated code four hours
//! later, moved by a merge in the same session. Nothing flagged it.
//!
//! [`locate`] turns "is this citation still true?" into a measured property with
//! three degrees, so a re-check can distinguish *moved* from *rewritten* from
//! *gone*.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ByteInterval;

/// How well a passage matched its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentStatus {
    /// Byte-identical. The citation is still literally true.
    Exact,
    /// Identical once whitespace runs are collapsed — reflowed, reindented, or
    /// rewrapped. The claim holds; only the formatting moved.
    Normalized,
    /// Enough tokens matched, densely enough, to be the same passage rewritten.
    /// Worth a human look: the claim may or may not still hold.
    Fuzzy,
}

/// Where a passage was found, and how well.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    pub interval: ByteInterval,
    pub status: AlignmentStatus,
    /// Fraction of the passage's tokens that were found (1.0 when exact).
    pub coverage: f32,
    /// Matched tokens divided by the tokens the match spans.
    ///
    /// LangExtract's `fuzzy_alignment_min_density`, and the guard that matters
    /// most: without it a short passage "matches" by scattering three common
    /// words across two pages of unrelated text.
    pub density: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignConfig {
    /// Fall back to token matching when exact and normalized both fail.
    pub enable_fuzzy: bool,
    /// Minimum fraction of the passage's tokens that must be found.
    pub min_coverage: f32,
    /// Minimum matched-to-spanned ratio. Rejects scattered matches.
    pub min_density: f32,
    /// How far a fuzzy match may stretch, as a multiple of the passage's token
    /// count. Bounds the search and encodes "a rewrite, not a whole section".
    pub max_span_ratio: usize,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy: true,
            // Two thirds of the tokens: tolerant of an edited sentence, not of
            // a different one.
            min_coverage: 0.66,
            // Half the spanned tokens must be matches. Below that it is not the
            // same passage, it is coincidence.
            min_density: 0.5,
            max_span_ratio: 3,
        }
    }
}

/// Find `needle` in `source`, cheapest strategy first.
///
/// Returns `None` when the passage cannot be located at all — which is the
/// answer that matters most: a citation that no longer resolves.
pub fn locate(source: &str, needle: &str, config: &AlignConfig) -> Option<Alignment> {
    if needle.trim().is_empty() || source.is_empty() {
        return None;
    }

    // 1 — exact.
    if let Some(start) = source.find(needle) {
        return Some(Alignment {
            interval: ByteInterval::new(start, start + needle.len()),
            status: AlignmentStatus::Exact,
            coverage: 1.0,
            density: 1.0,
        });
    }

    // 2 — whitespace-insensitive.
    if let Some(interval) = locate_normalized(source, needle) {
        return Some(Alignment {
            interval,
            status: AlignmentStatus::Normalized,
            coverage: 1.0,
            density: 1.0,
        });
    }

    // 3 — token overlap, guarded by coverage and density.
    if config.enable_fuzzy {
        return locate_fuzzy(source, needle, config);
    }
    None
}

/// Collapse whitespace runs to single spaces, keeping a byte-offset map back to
/// the original.
///
/// `map[i]` is the original byte offset of the character occupying normalized
/// byte `i`; `map[normalized.len()]` is the original length, so a normalized
/// range `[s, e)` maps to the original range `[map[s], map[e])`.
fn normalize_with_map(s: &str) -> (String, Vec<usize>) {
    let mut out = String::with_capacity(s.len());
    let mut map: Vec<usize> = Vec::with_capacity(s.len() + 1);
    let mut prev_ws = false;

    for (idx, ch) in s.char_indices() {
        if ch.is_whitespace() {
            if prev_ws {
                continue;
            }
            prev_ws = true;
            out.push(' ');
            map.push(idx);
        } else {
            prev_ws = false;
            let before = out.len();
            out.push(ch);
            for _ in before..out.len() {
                map.push(idx);
            }
        }
    }
    map.push(s.len());
    (out, map)
}

fn locate_normalized(source: &str, needle: &str) -> Option<ByteInterval> {
    let (norm_source, map) = normalize_with_map(source);
    let (norm_needle_raw, _) = normalize_with_map(needle);
    let norm_needle = norm_needle_raw.trim();
    if norm_needle.is_empty() {
        return None;
    }

    let at = norm_source.find(norm_needle)?;
    let end = at + norm_needle.len();
    // `map` is indexed by normalized byte position, and both ends are within it
    // by construction.
    Some(ByteInterval::new(map[at], map[end]))
}

/// A word with its byte span in the source.
struct Token<'a> {
    lower: String,
    interval: ByteInterval,
    _raw: &'a str,
}

fn tokenize(s: &str) -> Vec<Token<'_>> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in s.char_indices() {
        let is_word = ch.is_alphanumeric() || ch == '_';
        match (is_word, start) {
            (true, None) => start = Some(idx),
            (false, Some(st)) => {
                let raw = &s[st..idx];
                out.push(Token {
                    lower: raw.to_lowercase(),
                    interval: ByteInterval::new(st, idx),
                    _raw: raw,
                });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(st) = start {
        let raw = &s[st..];
        out.push(Token {
            lower: raw.to_lowercase(),
            interval: ByteInterval::new(st, s.len()),
            _raw: raw,
        });
    }
    out
}

fn locate_fuzzy(source: &str, needle: &str, config: &AlignConfig) -> Option<Alignment> {
    let needle_tokens = tokenize(needle);
    let source_tokens = tokenize(source);
    if needle_tokens.is_empty() || source_tokens.is_empty() {
        return None;
    }

    let mut wanted: HashMap<&str, usize> = HashMap::new();
    for t in &needle_tokens {
        *wanted.entry(t.lower.as_str()).or_insert(0) += 1;
    }
    let needle_len = needle_tokens.len();
    let max_span = needle_len
        .saturating_mul(config.max_span_ratio)
        .max(needle_len);

    let mut best: Option<Alignment> = None;

    for start in 0..source_tokens.len() {
        let mut remaining = wanted.clone();
        let mut matched = 0usize;

        let limit = (start + max_span).min(source_tokens.len());
        for end in start..limit {
            let tok = source_tokens[end].lower.as_str();
            if let Some(count) = remaining.get_mut(tok) {
                if *count > 0 {
                    *count -= 1;
                    matched += 1;
                }
            }

            let span_len = end - start + 1;
            let coverage = matched as f32 / needle_len as f32;
            let density = matched as f32 / span_len as f32;

            if coverage < config.min_coverage || density < config.min_density {
                continue;
            }
            let candidate = Alignment {
                interval: ByteInterval::new(
                    source_tokens[start].interval.start,
                    source_tokens[end].interval.end,
                ),
                status: AlignmentStatus::Fuzzy,
                coverage,
                density,
            };
            // Prefer coverage, then density, then the shorter span — the
            // tightest passage that still explains the quote.
            let better = match &best {
                None => true,
                Some(b) => {
                    (
                        candidate.coverage,
                        candidate.density,
                        usize::MAX - candidate.interval.len(),
                    ) > (b.coverage, b.density, usize::MAX - b.interval.len())
                }
            };
            if better {
                best = Some(candidate);
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "The quick brown fox jumps over the lazy dog.\n\
                       Pack my box with five dozen liquor jugs.\n\
                       How vexingly quick daft zebras jump!";

    #[test]
    fn exact_match_wins_and_resolves_to_the_right_slice() {
        let a = locate(SRC, "five dozen liquor jugs", &AlignConfig::default()).unwrap();
        assert_eq!(a.status, AlignmentStatus::Exact);
        assert_eq!(a.coverage, 1.0);
        assert_eq!(a.interval.slice(SRC).unwrap(), "five dozen liquor jugs");
    }

    #[test]
    fn reflowed_whitespace_still_resolves_as_normalized() {
        // The citation was written before the file was re-wrapped.
        let a = locate(
            SRC,
            "five    dozen\n  liquor   jugs",
            &AlignConfig::default(),
        )
        .unwrap();
        assert_eq!(a.status, AlignmentStatus::Normalized);
        assert_eq!(a.interval.slice(SRC).unwrap(), "five dozen liquor jugs");
    }

    #[test]
    fn an_edited_sentence_resolves_as_fuzzy_and_points_at_the_right_region() {
        // One word changed: not exact, not a whitespace issue, but clearly the
        // same passage.
        let a = locate(SRC, "quick brown cat jumps over", &AlignConfig::default()).unwrap();
        assert_eq!(a.status, AlignmentStatus::Fuzzy);
        assert!(a.coverage >= 0.66, "coverage was {}", a.coverage);
        let slice = a.interval.slice(SRC).unwrap();
        assert!(
            slice.contains("quick brown fox jumps over"),
            "got {slice:?}"
        );
    }

    #[test]
    fn a_passage_that_is_gone_does_not_resolve() {
        // The answer that matters most: the citation no longer holds.
        assert!(locate(
            SRC,
            "entirely unrelated wording about turtles",
            &AlignConfig::default()
        )
        .is_none());
    }

    #[test]
    fn the_density_guard_rejects_a_scattered_match() {
        // Every one of these words appears in the source, far apart. Without a
        // density floor this "matches" and reports a span covering everything.
        let scattered = "the my how";
        let permissive = AlignConfig {
            min_density: 0.0,
            min_coverage: 0.66,
            ..AlignConfig::default()
        };
        let guarded = AlignConfig::default();

        let loose = locate(SRC, scattered, &permissive);
        assert!(loose.is_some(), "without the guard this matches");
        assert!(
            loose.unwrap().density < 0.5,
            "and it matches thinly — that is the failure mode"
        );
        assert!(
            locate(SRC, scattered, &guarded).is_none(),
            "with the guard it must be rejected"
        );
    }

    #[test]
    fn fuzzy_can_be_switched_off_entirely() {
        let strict = AlignConfig {
            enable_fuzzy: false,
            ..AlignConfig::default()
        };
        assert!(locate(SRC, "quick brown cat jumps over", &strict).is_none());
        // Exact and normalized still work.
        assert!(locate(SRC, "lazy dog", &strict).is_some());
    }

    #[test]
    fn empty_inputs_never_resolve() {
        let cfg = AlignConfig::default();
        assert!(locate(SRC, "", &cfg).is_none());
        assert!(locate(SRC, "   \n ", &cfg).is_none());
        assert!(locate("", "anything", &cfg).is_none());
    }

    #[test]
    fn multibyte_sources_produce_valid_intervals() {
        let src = "Voilà une phrase accentuée. 日本語のテキスト. Et la fin.";
        let cfg = AlignConfig::default();
        for needle in ["phrase accentuée", "日本語のテキスト", "Et la fin"] {
            let a = locate(src, needle, &cfg).unwrap_or_else(|| panic!("should locate {needle:?}"));
            assert!(
                a.interval.slice(src).is_some(),
                "interval must fall on char boundaries for {needle:?}"
            );
        }
    }

    #[test]
    fn normalization_maps_back_to_original_offsets_exactly() {
        let src = "alpha   beta\n\n\tgamma";
        let (norm, map) = normalize_with_map(src);
        assert_eq!(norm, "alpha beta gamma");
        assert_eq!(map.len(), norm.len() + 1);
        assert_eq!(map[map.len() - 1], src.len());
        // Every normalized position maps to a valid char boundary.
        for &orig in &map {
            assert!(orig <= src.len() && src.is_char_boundary(orig));
        }
    }
}
