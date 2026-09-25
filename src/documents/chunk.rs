//! Sentence-aware chunking that tiles the source exactly.
//!
//! Ported in spirit from LangExtract's `ChunkIterator` (Google, Apache-2.0),
//! whose docstring lays out three cases with worked examples:
//!
//! - **C** — several whole sentences fit in the budget, so they are packed together.
//! - **A** — a sentence exceeds the budget, so it is split while respecting
//!   newlines and word boundaries.
//! - **B** — a single indivisible token exceeds the budget, so it *becomes* the
//!   chunk, overflowing rather than being cut.
//!
//! Case B is the one worth internalising: **the budget is a target, not a
//! constraint.** Cutting through an indivisible unit would destroy the very
//! thing the chunk exists to preserve — the ability to point back at the source.
//! An oversized chunk is a problem for the next stage; a mangled one is a
//! problem forever.
//!
//! ## The invariant
//!
//! Chunks **tile** the source: they are contiguous, non-overlapping, and their
//! concatenation reproduces the input byte for byte. Equivalently,
//! `chunk.text == &source[chunk.interval]` for every chunk. Everything
//! downstream — grounding a quote, highlighting a passage, mapping an
//! extraction back to a line — rests on that property, so it is asserted
//! directly in the tests rather than assumed.

use serde::{Deserialize, Serialize};

use super::ByteInterval;

/// Default chunk budget in bytes.
///
/// Sized for an embedding model's window rather than an LLM's: chunks here feed
/// retrieval, and `multilingual-e5-base` (PO's default embedding provider) has a
/// 512-token window, which is roughly 1.5–2 KB of prose in a Latin script and
/// less in others. 1500 leaves headroom without shredding paragraphs.
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 1500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Target maximum size of a chunk, in bytes.
    ///
    /// A target: a single indivisible token longer than this is emitted whole
    /// (case B). Use [`TextChunk::is_oversized`] to detect those.
    pub max_bytes: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_CHUNK_BYTES,
        }
    }
}

/// One chunk of a source document, carrying where it came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextChunk {
    /// Exactly `&source[interval]`. Held alongside the interval so consumers
    /// need not keep the whole document alive to read a chunk.
    pub text: String,
    pub interval: ByteInterval,
}

impl TextChunk {
    /// Whether this chunk had to exceed the budget (case B).
    pub fn is_oversized(&self, config: &ChunkConfig) -> bool {
        self.interval.len() > config.max_bytes
    }
}

/// Split `source` into chunks that tile it exactly.
///
/// An empty or whitespace-only source yields no chunks: there is nothing to
/// ground, and an empty chunk would be an entry that can never match anything.
pub fn chunk_text(source: &str, config: &ChunkConfig) -> Vec<TextChunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }
    // A zero budget would loop forever looking for a fit; treat it as "one
    // chunk per indivisible unit" rather than erroring, since the budget is
    // advisory by design.
    let max = config.max_bytes.max(1);

    let sentences = sentence_spans(source);
    let mut out: Vec<TextChunk> = Vec::new();
    let mut pending: Option<ByteInterval> = None;

    for sentence in sentences {
        // Case A/B: this sentence cannot share a chunk with anything.
        if sentence.len() > max {
            if let Some(p) = pending.take() {
                out.push(materialize(source, p));
            }
            for piece in split_oversized(source, sentence, max) {
                out.push(materialize(source, piece));
            }
            continue;
        }

        match pending {
            // Case C: extend the run while the whole thing still fits.
            Some(p) if sentence.end - p.start <= max => {
                pending = Some(ByteInterval::new(p.start, sentence.end));
            }
            Some(p) => {
                out.push(materialize(source, p));
                pending = Some(sentence);
            }
            None => pending = Some(sentence),
        }
    }

    if let Some(p) = pending {
        out.push(materialize(source, p));
    }
    out
}

fn materialize(source: &str, interval: ByteInterval) -> TextChunk {
    TextChunk {
        text: interval
            .slice(source)
            .expect("intervals produced here always fall on char boundaries")
            .to_string(),
        interval,
    }
}

/// Byte spans of the source's sentences, tiling it completely.
///
/// Sentence detection is deliberately naive — `.`, `!` or `?` followed by
/// whitespace, plus blank lines as hard breaks. No abbreviation dictionary, no
/// language model. A wrong boundary costs a slightly odd split; it never costs
/// correctness, because chunks tile regardless of where the boundaries land.
///
/// Trailing whitespace stays attached to the sentence it follows, which is what
/// keeps the spans contiguous.
fn sentence_spans(source: &str) -> Vec<ByteInterval> {
    let bytes = source.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        let is_terminator = matches!(b, b'.' | b'!' | b'?');
        let is_break = b == b'\n'
            && bytes
                .get(i + 1..)
                .map(|rest| rest.starts_with(b"\n"))
                .unwrap_or(false);

        if is_terminator || is_break {
            // Consume the terminator and any whitespace that follows, so the
            // next span starts on real content.
            let mut end = i + 1;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            // A terminator not followed by whitespace is mid-token ("3.14",
            // "file.rs") — not a sentence end.
            if is_break || end > i + 1 || end == bytes.len() {
                spans.push(ByteInterval::new(start, end));
                start = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }

    if start < bytes.len() {
        spans.push(ByteInterval::new(start, bytes.len()));
    }
    spans
}

/// Tile an over-budget span, preferring newline breaks, then word breaks.
///
/// A single token longer than the budget is emitted alone and oversized
/// (case B) rather than cut: the whole point of a chunk is to be quotable back
/// to its source.
fn split_oversized(source: &str, span: ByteInterval, max: usize) -> Vec<ByteInterval> {
    let mut out = Vec::new();
    let mut cursor = span.start;

    while cursor < span.end {
        let remaining = span.end - cursor;
        if remaining <= max {
            out.push(ByteInterval::new(cursor, span.end));
            break;
        }

        let window_end = cursor + max;
        // Prefer a newline inside the window, then any whitespace. Both are
        // ASCII, so a break at these positions is always a char boundary.
        let window = &source.as_bytes()[cursor..window_end];
        let cut = window
            .iter()
            .rposition(|&b| b == b'\n')
            .or_else(|| window.iter().rposition(|&b| b.is_ascii_whitespace()))
            .map(|rel| cursor + rel + 1);

        match cut {
            // `> cursor` guards against a break at position 0 producing no progress.
            Some(end) if end > cursor => {
                out.push(ByteInterval::new(cursor, end));
                cursor = end;
            }
            // Case B: no break available — one token owns the whole window and
            // then some. Emit it whole, overflowing the budget.
            _ => {
                let end = next_break_after(source, cursor, span.end);
                out.push(ByteInterval::new(cursor, end));
                cursor = end;
            }
        }
    }

    out
}

/// End of the indivisible token starting at `from`: the next whitespace, or the
/// end of the span.
fn next_break_after(source: &str, from: usize, limit: usize) -> usize {
    let bytes = source.as_bytes();
    let mut i = from;
    while i < limit && !bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    // Absorb the whitespace so spans stay contiguous.
    while i < limit && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i > from {
        i
    } else {
        limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property everything else rests on.
    fn assert_tiles(source: &str, chunks: &[TextChunk]) {
        let mut expected_start = 0usize;
        for c in chunks {
            assert_eq!(
                c.interval.start, expected_start,
                "chunks must be contiguous — gap or overlap before {:?}",
                c.text
            );
            assert_eq!(
                c.text,
                c.interval.slice(source).unwrap(),
                "chunk text must equal its slice of the source"
            );
            expected_start = c.interval.end;
        }
        if !chunks.is_empty() {
            assert_eq!(expected_start, source.len(), "chunks must cover the source");
            let rebuilt: String = chunks.iter().map(|c| c.text.as_str()).collect();
            assert_eq!(rebuilt, source, "concatenation must reproduce the source");
        }
    }

    #[test]
    fn empty_and_blank_sources_yield_nothing() {
        let cfg = ChunkConfig::default();
        assert!(chunk_text("", &cfg).is_empty());
        assert!(chunk_text("   \n\t  ", &cfg).is_empty());
    }

    #[test]
    fn case_c_packs_whole_sentences() {
        // LangExtract's worked example, byte-for-byte.
        let src = "Roses are red. Violets are blue. Flowers are nice. And so are you.";
        let chunks = chunk_text(src, &ChunkConfig { max_bytes: 60 });
        assert_tiles(src, &chunks);
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].text.trim(),
            "Roses are red. Violets are blue. Flowers are nice."
        );
        assert_eq!(chunks[1].text.trim(), "And so are you.");
    }

    #[test]
    fn case_a_splits_a_long_sentence_on_newlines_first() {
        let src = "No man is an island,\nEntire of itself,\nEvery man is a piece of the continent,\nA part of the main.";
        let chunks = chunk_text(src, &ChunkConfig { max_bytes: 40 });
        assert_tiles(src, &chunks);
        assert!(chunks.len() >= 3, "should split, got {}", chunks.len());
        for c in &chunks {
            // Every break lands after a newline, never mid-line.
            assert!(
                c.text.ends_with('\n') || c.interval.end == src.len(),
                "chunk should end at a line break: {:?}",
                c.text
            );
        }
    }

    #[test]
    fn case_b_lets_an_indivisible_token_overflow_rather_than_cutting_it() {
        let src = "This is antidisestablishmentarianism.";
        let cfg = ChunkConfig { max_bytes: 20 };
        let chunks = chunk_text(src, &cfg);
        assert_tiles(src, &chunks);

        let long = chunks
            .iter()
            .find(|c| c.text.contains("antidisestablishmentarianism"))
            .expect("the long token must survive intact in some chunk");
        assert!(
            long.is_oversized(&cfg),
            "the chunk holding it must be reported as oversized"
        );
        // The point of case B: the word is never split across chunks.
        assert_eq!(
            chunks
                .iter()
                .filter(|c| c.text.contains("antidisestablish"))
                .count(),
            1
        );
    }

    #[test]
    fn a_decimal_or_filename_is_not_a_sentence_end() {
        let src = "Edit src/chat/manager.rs at line 3.14 then rerun. Done.";
        let chunks = chunk_text(src, &ChunkConfig { max_bytes: 1000 });
        assert_tiles(src, &chunks);
        assert_eq!(chunks.len(), 1, "a single chunk under budget");
    }

    #[test]
    fn blank_lines_break_sentences() {
        let src = "First paragraph with no terminator\n\nSecond paragraph";
        let chunks = chunk_text(src, &ChunkConfig { max_bytes: 40 });
        assert_tiles(src, &chunks);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn multibyte_text_never_splits_a_character() {
        // Accents, CJK and an emoji: every interval must stay on a boundary, or
        // `slice` returns None and `assert_tiles` fails.
        let src =
            "Voilà une phrase accentuée. 日本語のテキストもあります。Et un emoji 🎯 ici. Fin.";
        for max in [8usize, 16, 32, 64, 1000] {
            let chunks = chunk_text(src, &ChunkConfig { max_bytes: max });
            assert_tiles(src, &chunks);
        }
    }

    #[test]
    fn a_degenerate_budget_still_terminates_and_tiles() {
        // max_bytes = 0 would loop forever looking for a fit if the budget were
        // treated as a hard constraint.
        let src = "Short one. And another one here.";
        let chunks = chunk_text(src, &ChunkConfig { max_bytes: 0 });
        assert_tiles(src, &chunks);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn chunks_tile_arbitrary_prose_at_every_budget() {
        let src = "# Title\n\nA paragraph of prose that runs on for a while. \
                   It has several sentences! Some are short. Others ramble on \
                   considerably longer than the others do, for no good reason.\n\n\
                   - a list item\n- another item\n\nClosing line.";
        for max in [1usize, 2, 5, 13, 40, 100, 10_000] {
            let chunks = chunk_text(src, &ChunkConfig { max_bytes: max });
            assert_tiles(src, &chunks);
        }
    }
}
