//! Turning bytes into text, with provenance intact.
//!
//! One trait, one registry, one extractor per family. The chunker and aligner
//! downstream ([`super::chunk`], [`super::align`]) work on the *text* this
//! produces, so everything here exists to hand them something faithful to the
//! original.
//!
//! ## Detection is by content, not by extension
//!
//! A file called `report.txt` that starts with `%PDF-` is a PDF. A file called
//! `notes.md` uploaded by someone else may be anything at all. Extensions are a
//! hint from an untrusted source; magic bytes are evidence. The extension is
//! used only to disambiguate between families that share a container — `.docx`
//! and `.jar` are both ZIPs.
//!
//! ## Pages are part of provenance
//!
//! [`ExtractedText::pages`] records where each page or section starts and ends
//! in the extracted text, so an alignment landing at byte 12 000 can be reported
//! as "page 7" rather than as an offset nobody can act on. Formats without
//! pagination leave it empty; that is information, not a gap.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod text;

#[cfg(feature = "docx")]
pub mod docx;
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pptx")]
pub mod pptx;
#[cfg(feature = "xlsx")]
pub mod xlsx;
#[cfg(any(feature = "docx", feature = "pptx"))]
mod xml;

use serde::{Deserialize, Serialize};

use super::ByteInterval;

/// Families of document this module knows how to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    /// Plain text, Markdown, source code, JSON, CSV, YAML, TOML — anything whose
    /// bytes are already the text.
    PlainText,
    /// Office Open XML word processing document (`.docx`).
    Docx,
    /// Portable Document Format.
    Pdf,
    /// Office Open XML spreadsheet (`.xlsx`).
    Xlsx,
    /// Office Open XML presentation (`.pptx`).
    Pptx,
    /// Anything else, stored as an opaque attachment: no extractor read it, so
    /// there is no text and no chunks. Never produced by an extractor — only by
    /// the upload pipeline when nothing recognised the bytes.
    Binary,
}

impl DocumentFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PlainText => "plain_text",
            Self::Docx => "docx",
            Self::Pdf => "pdf",
            Self::Xlsx => "xlsx",
            Self::Pptx => "pptx",
            Self::Binary => "binary",
        }
    }
}

/// What extraction produced, plus whatever went wrong along the way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedText {
    pub text: String,
    pub format: DocumentFormat,
    /// Byte spans of pages (or top-level sections) within `text`, in order.
    ///
    /// Empty for formats that have no pagination. When present, the spans tile
    /// `text` the same way chunks tile a source, so a byte offset maps to a page
    /// by a single scan.
    pub pages: Vec<ByteInterval>,
    /// Non-fatal problems: a page that could not be decoded, an unsupported
    /// embedded object, text recovered with low confidence.
    ///
    /// Surfaced rather than swallowed. An extraction that silently dropped half
    /// a document would poison every chunk and alignment built on it, and
    /// nothing downstream could tell.
    pub warnings: Vec<String>,
}

impl ExtractedText {
    /// 1-based page number containing `offset`, if the format is paginated.
    pub fn page_of(&self, offset: usize) -> Option<usize> {
        self.pages
            .iter()
            .position(|p| offset >= p.start && offset < p.end)
            .map(|i| i + 1)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("no extractor for this content (sniffed: {sniffed})")]
    UnsupportedFormat { sniffed: String },
    #[error("{format} support is not compiled in — rebuild with the `{feature}` feature")]
    FeatureDisabled {
        format: &'static str,
        feature: &'static str,
    },
    #[error("input is empty")]
    Empty,
    #[error("malformed {format} document: {detail}")]
    Malformed {
        format: &'static str,
        detail: String,
    },
}

/// Evidence about what a blob of bytes is.
#[derive(Debug, Clone)]
pub struct FormatProbe<'a> {
    pub bytes: &'a [u8],
    /// Filename as supplied by the uploader. A hint, never a decision on its
    /// own — see the module docs.
    pub filename: Option<&'a str>,
}

impl<'a> FormatProbe<'a> {
    pub fn new(bytes: &'a [u8], filename: Option<&'a str>) -> Self {
        Self { bytes, filename }
    }

    /// Lowercased extension, if the filename carries one.
    pub fn extension(&self) -> Option<String> {
        self.filename
            .and_then(|f| f.rsplit_once('.'))
            .map(|(_, ext)| ext.to_ascii_lowercase())
    }

    pub fn starts_with(&self, magic: &[u8]) -> bool {
        self.bytes.starts_with(magic)
    }

    /// A short, safe description of what was sniffed, for error messages.
    pub fn describe(&self) -> String {
        let head: String = self
            .bytes
            .iter()
            .take(8)
            .map(|b| {
                if b.is_ascii_graphic() {
                    (*b as char).to_string()
                } else {
                    format!("\\x{b:02x}")
                }
            })
            .collect();
        match self.extension() {
            Some(ext) => format!("{head} (.{ext})"),
            None => head,
        }
    }
}

/// An extractor for one document family.
pub trait TextExtractor: Send + Sync {
    fn format(&self) -> DocumentFormat;

    /// Whether this extractor recognises the content.
    ///
    /// Implementations must decide on `probe.bytes` first and use the extension
    /// only to disambiguate between formats sharing a container.
    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool;

    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError>;
}

/// Ordered set of extractors; first match wins.
///
/// Order matters: specific container formats must be probed before the
/// catch-all text extractor, which accepts anything that decodes as UTF-8 and
/// would otherwise swallow a DOCX as mojibake.
pub struct ExtractorRegistry {
    extractors: Vec<Box<dyn TextExtractor>>,
}

impl ExtractorRegistry {
    /// Empty registry — prefer [`ExtractorRegistry::with_builtins`].
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
        }
    }

    /// Every extractor compiled into this build, in probe order.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();

        // Container formats first — see the ordering note on the struct.
        #[cfg(feature = "pdf")]
        reg.register(Box::new(pdf::PdfExtractor));
        #[cfg(feature = "docx")]
        reg.register(Box::new(docx::DocxExtractor));
        #[cfg(feature = "xlsx")]
        reg.register(Box::new(xlsx::XlsxExtractor));
        #[cfg(feature = "pptx")]
        reg.register(Box::new(pptx::PptxExtractor));

        // Catch-all last.
        reg.register(Box::new(text::PlainTextExtractor));
        reg
    }

    pub fn register(&mut self, extractor: Box<dyn TextExtractor>) {
        self.extractors.push(extractor);
    }

    /// Which formats this build can actually read.
    ///
    /// Reported rather than assumed: PDF and DOCX are behind cargo features, so
    /// "unsupported" and "not compiled in" are different answers and callers
    /// deserve to tell them apart.
    pub fn supported_formats(&self) -> Vec<DocumentFormat> {
        self.extractors.iter().map(|e| e.format()).collect()
    }

    pub fn extract(
        &self,
        bytes: &[u8],
        filename: Option<&str>,
    ) -> Result<ExtractedText, ExtractError> {
        if bytes.is_empty() {
            return Err(ExtractError::Empty);
        }
        let probe = FormatProbe::new(bytes, filename);

        for extractor in &self.extractors {
            if extractor.can_handle(&probe) {
                return extractor.extract(bytes);
            }
        }

        // Tell the caller the capability is missing rather than the file is
        // broken, when that is what actually happened.
        if probe.starts_with(b"%PDF-") {
            return Err(ExtractError::FeatureDisabled {
                format: "PDF",
                feature: "pdf",
            });
        }
        if probe.starts_with(b"PK\x03\x04") && probe.extension().as_deref() == Some("docx") {
            return Err(ExtractError::FeatureDisabled {
                format: "DOCX",
                feature: "docx",
            });
        }

        Err(ExtractError::UnsupportedFormat {
            sniffed: probe.describe(),
        })
    }
}

impl Default for ExtractorRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn empty_input_is_rejected_before_any_extractor_runs() {
        let reg = ExtractorRegistry::with_builtins();
        assert!(matches!(reg.extract(b"", None), Err(ExtractError::Empty)));
    }

    #[test]
    fn plain_text_round_trips() {
        let reg = ExtractorRegistry::with_builtins();
        let out = reg
            .extract(b"# Title\n\nSome prose.", Some("notes.md"))
            .unwrap();
        assert_eq!(out.format, DocumentFormat::PlainText);
        assert_eq!(out.text, "# Title\n\nSome prose.");
    }

    #[test]
    fn content_beats_extension() {
        // A PDF named .txt must not be handed to the text extractor as mojibake.
        let reg = ExtractorRegistry::with_builtins();
        let pdf_bytes = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\nbinary garbage";
        let result = reg.extract(pdf_bytes, Some("report.txt"));
        // What this asserts is that the *text* extractor did not claim the file.
        // With the `pdf` feature in the default set (guarded by
        // `pdf_bytes_extract_by_default`), the PDF extractor claims the magic
        // bytes; the stub is then either parsed or reported malformed — never
        // "feature disabled".
        match result {
            // Claimed and parsed.
            Ok(out) => assert_eq!(out.format, DocumentFormat::Pdf),
            // Claimed, then found to be the truncated stub it is. Still the PDF
            // extractor's verdict, which is the point.
            Err(ExtractError::Malformed { format, .. }) => assert_eq!(format, "PDF"),
            // "Feature disabled" is the regression this suite now guards against:
            // it used to be accepted here as a success, which is exactly why a
            // shipped binary could silently lack a PDF extractor.
            Err(ExtractError::FeatureDisabled { format, .. }) => {
                panic!("PDF extractor disabled in a default build (format: {format})")
            }
            // This is the failure the test exists to catch: nobody recognised
            // the magic bytes, or worse, the text extractor swallowed them.
            Err(other) => panic!("content did not beat extension: {other}"),
        }
    }

    /// A minimal but valid single-page PDF with a text layer, hand-written the
    /// way the PDF extractor's own fixtures are (see `pdf::tests::build_pdf`) so
    /// this test states exactly what it depends on: one Helvetica page, one
    /// `Tj` showing "Hello", a proper xref table and trailer.
    #[cfg(feature = "pdf")]
    fn minimal_pdf_with_text() -> Vec<u8> {
        let stream = b"BT /F1 24 Tf 72 700 Td (Hello) Tj ET\n";
        let mut content = format!("<< /Length {} >>\nstream\n", stream.len()).into_bytes();
        content.extend_from_slice(stream);
        content.extend_from_slice(b"endstream");
        let objects: Vec<(usize, Vec<u8>)> = vec![
            (1, b"<< /Type /Catalog /Pages 2 0 R >>".to_vec()),
            (2, b"<< /Type /Pages /Kids [ 4 0 R ] /Count 1 >>".to_vec()),
            (
                3,
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            ),
            (
                4,
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                     /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"
                    .to_vec(),
            ),
            (5, content),
        ];

        let mut out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let max_id = 5usize;
        let mut offsets = vec![0usize; max_id + 1];
        for (id, body) in &objects {
            offsets[*id] = out.len();
            out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
            out.extend_from_slice(body);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref_offset = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", max_id + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
                max_id + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn pdf_bytes_extract_by_default() {
        // PDF support lives behind a cargo feature. If `pdf` ever falls out of
        // the default set, every shipped binary answers 501 FeatureDisabled to a
        // PDF attachment and nothing signals it. This test is the tripwire: it
        // must fail unless the feature is compiled in on the default feature set.
        #[cfg(feature = "pdf")]
        {
            let reg = ExtractorRegistry::with_builtins();
            let out = reg
                .extract(&minimal_pdf_with_text(), Some("report.pdf"))
                .unwrap_or_else(|e| {
                    panic!("a default-feature build must extract a PDF attachment, got: {e}")
                });
            assert_eq!(
                out.format,
                DocumentFormat::Pdf,
                "the PDF extractor must claim the bytes as Pdf"
            );
            assert!(
                !out.text.is_empty(),
                "the extracted PDF must carry its text layer"
            );
        }

        #[cfg(not(feature = "pdf"))]
        {
            panic!(
                "the `pdf` feature is missing from the default features — a shipped \
                 binary would answer FeatureDisabled to every PDF attachment"
            );
        }
    }

    #[test]
    fn an_unreadable_blob_reports_what_was_sniffed() {
        let reg = ExtractorRegistry::with_builtins();
        // Invalid UTF-8 with no known magic.
        let err = reg.extract(&[0xff, 0xfe, 0x00, 0x01, 0x02], Some("mystery.bin"));
        match err {
            Err(ExtractError::UnsupportedFormat { sniffed }) => {
                assert!(sniffed.contains("\\xff"), "sniffed was {sniffed:?}");
                assert!(sniffed.contains(".bin"));
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn page_lookup_is_one_based_and_bounded() {
        let extracted = ExtractedText {
            text: "aaaabbbbcccc".to_string(),
            format: DocumentFormat::Pdf,
            pages: vec![
                ByteInterval::new(0, 4),
                ByteInterval::new(4, 8),
                ByteInterval::new(8, 12),
            ],
            warnings: vec![],
        };
        assert_eq!(extracted.page_of(0), Some(1));
        assert_eq!(extracted.page_of(7), Some(2));
        assert_eq!(extracted.page_of(11), Some(3));
        assert_eq!(extracted.page_of(12), None, "past the end is not a page");
    }

    #[test]
    fn unpaginated_formats_report_no_page() {
        let extracted = ExtractedText {
            text: "hello".to_string(),
            format: DocumentFormat::PlainText,
            pages: vec![],
            warnings: vec![],
        };
        assert_eq!(extracted.page_of(0), None);
    }
}
