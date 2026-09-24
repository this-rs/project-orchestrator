//! Document ingestion core — chunking with preserved provenance.
//!
//! This is the format-agnostic foundation for attachments: it turns a blob of
//! text into chunks that can still point back at exactly where they came from,
//! and it can tell whether a claim about a document still holds.
//!
//! Deliberately **not** in this module: upload handling, format decoding, and
//! how a chunk reaches an agent's context. Those depend on product decisions
//! (which formats, injected or searched or exposed as a tool) that this layer
//! does not need to know about. [`store`] is the one exception that has since
//! been settled: the original bytes are kept on local disk, addressed by their
//! SHA-256.
//!
//! ## Why provenance is the hard part
//!
//! Borrowed from Google's LangExtract (Apache-2.0), whose largest source file
//! is not the chunker but the *aligner*: their hard problem is not splitting
//! text, it is proving where an extraction came from. An extraction that cannot
//! be located in the source is reported as unlocated rather than as a result —
//! which is how they detect a model quoting its own few-shot examples.
//!
//! PO has the same problem one level up, and already suffers from it: notes and
//! decisions assert things about files (`manager.rs:1933 does X`) and nothing
//! ever re-checks the assertion. `staleness_score` measures *age*, not validity.
//! [`align`] is the mechanism that turns that into a measured property.

pub mod align;
pub mod chunk;
pub mod extract;
pub mod store;

pub use align::{locate, Alignment, AlignmentStatus};
pub use chunk::{chunk_text, ChunkConfig, TextChunk};
pub use extract::{
    DocumentFormat, ExtractError, ExtractedText, ExtractorRegistry, FormatProbe, TextExtractor,
};
pub use store::{DocumentStore, StoreError, MAX_BLOB_BYTES};

use serde::{Deserialize, Serialize};

/// A half-open byte range `[start, end)` into a source document.
///
/// Byte offsets, not character offsets: they are what `&source[..]` takes, so
/// `&source[interval.start..interval.end]` is always valid and always cheap.
/// Every interval this module produces falls on a UTF-8 character boundary by
/// construction — the chunker only ever cuts between characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteInterval {
    pub start: usize,
    pub end: usize,
}

impl ByteInterval {
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "interval start must not exceed end");
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Slice the source this interval refers to.
    ///
    /// Returns `None` rather than panicking when the interval does not fit the
    /// text or lands mid-character — an interval stored months ago may be read
    /// back against a file that has since changed, and that must degrade to
    /// "cannot resolve" instead of bringing down the caller.
    pub fn slice<'a>(&self, source: &'a str) -> Option<&'a str> {
        if self.end > source.len()
            || !source.is_char_boundary(self.start)
            || !source.is_char_boundary(self.end)
        {
            return None;
        }
        source.get(self.start..self.end)
    }
}
