//! PDF, read page by page so a quote can be reported as "page 7".
//!
//! Behind the `pdf` cargo feature: `pdf_oxide` is a full PDF toolkit and pulls
//! in a large dependency subtree, which is not something every build of this
//! binary should pay for.
//!
//! ## Why this module extracts per page rather than whole-document
//!
//! `pdf_oxide` offers `extract_all_text()`, which is one call instead of a
//! loop. It is not used here: it returns a single string with no record of
//! where each page ended, and that record is the entire reason this module
//! exists. [`ExtractedText::pages`] has to tile the output so that
//! [`ExtractedText::page_of`] can turn a byte offset from the aligner back into
//! a page number a human can open. Extracting page by page and measuring the
//! text as it is appended is the only way to build those spans honestly —
//! anything else would be reverse-engineering page boundaries out of a string
//! that no longer has them.
//!
//! ## Pages are never dropped
//!
//! A page whose text cannot be extracted still occupies an interval — an empty
//! one, plus a warning. Skipping it would renumber every page after it, and a
//! citation pointing at the wrong page is worse than one pointing at a page
//! known to be unreadable.
//!
//! ## Scanned documents are an error, not an empty success
//!
//! See [`PdfExtractor::extract`]. A PDF with no text layer anywhere yields
//! [`ExtractError::Malformed`] rather than an empty [`ExtractedText`], because
//! downstream nothing can tell an empty extraction from a document that is
//! genuinely blank: it chunks into nothing and aligns to nothing, silently.

use std::panic::{catch_unwind, AssertUnwindSafe};

use pdf_oxide::PdfDocument;

use super::{DocumentFormat, ExtractError, ExtractedText, FormatProbe, TextExtractor};
use crate::documents::ByteInterval;

/// Inserted between pages so that page N's text does not run into page N+1's.
///
/// A blank line rather than a form feed: the chunker downstream treats a blank
/// line as a paragraph boundary, so a page break becomes a natural place to
/// split. A `\x0c` would be an invisible character inside a chunk instead.
const PAGE_SEPARATOR: &str = "\n\n";

/// Cap on library-level diagnostics copied into `warnings`.
///
/// `pdf_oxide` emits one diagnostic per malformed font, broken stream or
/// recovered object; a badly generated 500-page file can produce thousands.
/// The first few identify the problem, the rest only make the warning list
/// unreadable, and this struct is serialised and stored.
const MAX_LIBRARY_WARNINGS: usize = 16;

pub struct PdfExtractor;

impl TextExtractor for PdfExtractor {
    fn format(&self) -> DocumentFormat {
        DocumentFormat::Pdf
    }

    /// Magic bytes only.
    ///
    /// `%PDF-` is the header the spec requires (ISO 32000-1 §7.5.2). The
    /// extension is deliberately not consulted: `.pdf` on a ZIP is a lie we
    /// would be repeating, and a PDF named `report.txt` is still a PDF.
    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool {
        probe.starts_with(b"%PDF-")
    }

    /// Extract text and page boundaries.
    ///
    /// # Errors
    ///
    /// [`ExtractError::Malformed`] when the file cannot be parsed, when it is
    /// password-protected, or when it carries no text layer at all. The last
    /// one is a deliberate choice, not an oversight:
    ///
    /// - A scan with **no** extractable text is returned as an error. The
    ///   detail says how many pages lack a text layer, so a caller that has an
    ///   OCR path can route the document there instead of guessing. Returning
    ///   `Ok` with an empty string would put an empty document into the chunker
    ///   and every downstream check would pass on nothing.
    /// - A document where **some** pages are image-only is returned as `Ok`,
    ///   with one warning naming each such page. The text that does exist is
    ///   worth keeping, the page spans stay correct, and the warning tells the
    ///   caller exactly which pages it is missing.
    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
        // A PDF parser is the most hostile surface in this module: it runs on
        // uploaded bytes, and `pdf_oxide` is a large amount of index
        // arithmetic over attacker-controlled offsets. It returns `Result`
        // everywhere and is well tested, but a panic here would take down the
        // task handling the upload rather than failing one document. Catching
        // it converts the worst case into the same `Malformed` every other
        // parse failure produces.
        catch_unwind(AssertUnwindSafe(|| extract_inner(bytes))).unwrap_or_else(|payload| {
            Err(ExtractError::Malformed {
                format: "PDF",
                detail: format!("PDF parser panicked: {}", panic_message(&payload)),
            })
        })
    }
}

fn extract_inner(bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
    let doc = PdfDocument::from_bytes(bytes.to_vec()).map_err(|e| malformed(&e))?;

    // Checked before anything is read. `pdf_oxide` already tried the empty user
    // password while opening the file, so reaching here unauthenticated means a
    // real password is required — or that the file uses RC4/MD5 encryption,
    // which `legacy-crypto` would provide and we deliberately do not enable.
    // Either way, extraction would quietly return blank pages; say so instead.
    if doc.is_encrypted() && !doc.is_authenticated() {
        return Err(ExtractError::Malformed {
            format: "PDF",
            detail: "document is encrypted and the empty user password does not open it \
                     (a password is required, or it uses legacy RC4 encryption, which this \
                     build does not support)"
                .to_string(),
        });
    }

    let page_count = doc.page_count().map_err(|e| malformed(&e))?;
    if page_count == 0 {
        return Err(ExtractError::Malformed {
            format: "PDF",
            detail: "document declares no pages".to_string(),
        });
    }

    let mut text = String::new();
    let mut pages = Vec::with_capacity(page_count);
    let mut warnings = Vec::new();
    let mut pages_without_text = 0usize;

    for index in 0..page_count {
        let start = text.len();

        // `has_text_layer` is conservative — it answers `true` whenever it
        // cannot inspect the page — so this distinguishes "image-only scan"
        // from "extraction returned nothing", which are different problems for
        // the caller even though both produce an empty page.
        let image_only = doc.has_text_layer(index).map(|has| !has).unwrap_or(false);

        let page_text = match doc.extract_text(index) {
            Ok(t) => t,
            Err(e) => {
                // The page keeps its interval; only its content is lost. See
                // the module docs on renumbering.
                warnings.push(format!("page {}: text extraction failed: {e}", index + 1));
                String::new()
            }
        };

        if page_text.trim().is_empty() {
            pages_without_text += 1;
            if image_only {
                warnings.push(format!(
                    "page {}: no text layer (image-only page — OCR would be required)",
                    index + 1
                ));
            }
        }

        text.push_str(&page_text);
        // The separator belongs to the page that precedes it, so the intervals
        // tile `text` end to end with no offset falling outside every page.
        if index + 1 < page_count {
            text.push_str(PAGE_SEPARATOR);
        }
        pages.push(ByteInterval::new(start, text.len()));
    }

    if pages_without_text == page_count {
        return Err(ExtractError::Malformed {
            format: "PDF",
            detail: format!(
                "no extractable text on any of the {page_count} page(s) — the document is \
                 probably a scan and needs OCR, which this extractor does not perform"
            ),
        });
    }

    // Diagnostics the parser recorded while recovering: broken fonts, damaged
    // streams, objects rebuilt from a reconstructed xref. They mean the text
    // above is a best effort, which is exactly what `warnings` is for. The
    // structured surface is used rather than `take_warnings()` because it
    // carries the page number, and a diagnostic without a page cannot be acted
    // on any more than an offset without a page can.
    let library_warnings = doc.take_structured_warnings();
    let dropped = library_warnings.len().saturating_sub(MAX_LIBRARY_WARNINGS);
    warnings.extend(library_warnings.into_iter().take(MAX_LIBRARY_WARNINGS).map(
        |w| match w.page {
            Some(p) => format!("page {}: {}: {}", p + 1, w.category.as_str(), w.message),
            None => format!("document: {}: {}", w.category.as_str(), w.message),
        },
    ));
    if dropped > 0 {
        warnings.push(format!("... and {dropped} further parser diagnostics"));
    }

    Ok(ExtractedText {
        text,
        format: DocumentFormat::Pdf,
        pages,
        warnings,
    })
}

fn malformed(error: &pdf_oxide::Error) -> ExtractError {
    ExtractError::Malformed {
        format: "PDF",
        // `pdf_oxide`'s Display carries the byte offset and the reason, which
        // is what makes a corrupt-upload report actionable.
        detail: error.to_string(),
    }
}

/// Best-effort text of a panic payload, which is `&str` or `String` in practice.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a syntactically valid PDF in memory.
    ///
    /// Written out by hand rather than shipped as a fixture so the tests below
    /// state what they depend on: `Some(text)` is a page with a font and a
    /// `Tj`, `None` is a page with no `/Resources` at all — which is what an
    /// image-only scan looks like to a text extractor.
    fn build_pdf(pages: &[Option<&str>]) -> Vec<u8> {
        let n = pages.len();
        // Object ids: 1 catalog, 2 page tree, 3 font, then page i at 4 + 2i and
        // its content stream at 5 + 2i. Contiguous, which the xref table needs.
        let kids: String = (0..n).map(|i| format!("{} 0 R ", 4 + 2 * i)).collect();
        let mut objects: Vec<(usize, Vec<u8>)> = vec![
            (1, b"<< /Type /Catalog /Pages 2 0 R >>".to_vec()),
            (
                2,
                format!("<< /Type /Pages /Kids [ {kids}] /Count {n} >>").into_bytes(),
            ),
            (
                3,
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            ),
        ];

        for (i, page) in pages.iter().enumerate() {
            let page_id = 4 + 2 * i;
            let content_id = page_id + 1;
            let (resources, stream) = match page {
                Some(body) => (
                    "/Resources << /Font << /F1 3 0 R >> >>".to_string(),
                    format!("BT /F1 24 Tf 72 700 Td ({body}) Tj ET\n"),
                ),
                None => (String::new(), "0 0 0 rg 72 72 200 200 re f\n".to_string()),
            };
            objects.push((
                page_id,
                format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] {resources} \
                     /Contents {content_id} 0 R >>"
                )
                .into_bytes(),
            ));
            let mut content = format!("<< /Length {} >>\nstream\n", stream.len()).into_bytes();
            content.extend_from_slice(stream.as_bytes());
            content.extend_from_slice(b"endstream");
            objects.push((content_id, content));
        }

        // The binary comment on line 2 is what the spec recommends so that
        // transfer agents treat the file as binary; it also makes the fixture
        // look like a real producer's output.
        let mut out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let max_id = objects.iter().map(|(id, _)| *id).max().unwrap_or(0);
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
        // Entry 0 is the free-list head written above; objects 1..=max_id follow
        // in id order, which is what makes the ids contiguous a requirement.
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
    fn claims_pdfs_by_magic_and_ignores_the_extension_entirely() {
        let pdf = build_pdf(&[Some("hello")]);
        assert!(PdfExtractor.can_handle(&FormatProbe::new(&pdf, Some("report.txt"))));
        assert!(PdfExtractor.can_handle(&FormatProbe::new(&pdf, None)));
        // A `.pdf` name on something that is not a PDF must not be claimed:
        // the text extractor is a better home for it than a parse error.
        assert!(!PdfExtractor.can_handle(&FormatProbe::new(b"Not a PDF", Some("invoice.pdf"))));
        assert!(!PdfExtractor.can_handle(&FormatProbe::new(b"PK\x03\x04", Some("a.pdf"))));
    }

    #[test]
    fn extracts_text_from_a_single_page() {
        let pdf = build_pdf(&[Some("Hello from page one")]);
        let out = PdfExtractor.extract(&pdf).unwrap();
        assert_eq!(out.format, DocumentFormat::Pdf);
        assert!(
            out.text.contains("Hello from page one"),
            "text was {:?}",
            out.text
        );
        assert_eq!(out.pages.len(), 1);
    }

    #[test]
    fn page_spans_tile_the_text_and_resolve_to_real_page_numbers() {
        // The point of the whole module: an offset must come back as a page.
        let pdf = build_pdf(&[Some("Alpha one"), Some("Beta two"), Some("Gamma three")]);
        let out = PdfExtractor.extract(&pdf).unwrap();

        assert_eq!(out.pages.len(), 3, "one interval per page");

        // Tiling: no gaps, no overlaps, first starts at 0, last ends at the end.
        assert_eq!(out.pages[0].start, 0);
        assert_eq!(out.pages.last().unwrap().end, out.text.len());
        for pair in out.pages.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "pages must be contiguous");
        }

        // And every offset in the document maps to some page — which is what
        // `page_of` returning `None` would break for the aligner.
        for offset in 0..out.text.len() {
            assert!(
                out.page_of(offset).is_some(),
                "offset {offset} fell outside every page"
            );
        }

        // Each page's marker must be found inside that page's own interval.
        for (i, marker) in ["Alpha", "Beta", "Gamma"].iter().enumerate() {
            let at = out
                .text
                .find(marker)
                .unwrap_or_else(|| panic!("{marker} missing from {:?}", out.text));
            assert_eq!(
                out.page_of(at),
                Some(i + 1),
                "{marker} should be reported as page {}",
                i + 1
            );
            let slice = out.pages[i].slice(&out.text).expect("interval must slice");
            assert!(slice.contains(marker), "page {} was {slice:?}", i + 1);
        }
    }

    #[test]
    fn a_page_without_a_text_layer_is_reported_but_does_not_renumber_the_others() {
        // Page 2 is image-only. Page 3 must still be page 3.
        let pdf = build_pdf(&[Some("First page"), None, Some("Third page")]);
        let out = PdfExtractor.extract(&pdf).unwrap();

        assert_eq!(out.pages.len(), 3);
        let third = out
            .text
            .find("Third page")
            .expect("third page text missing");
        assert_eq!(out.page_of(third), Some(3), "the blank page must not shift");

        assert!(
            out.warnings.iter().any(|w| w.starts_with("page 2:")),
            "the image-only page must be named in the warnings: {:?}",
            out.warnings
        );
    }

    #[test]
    fn a_fully_scanned_document_errors_rather_than_returning_an_empty_success() {
        // Documented choice: an empty `ExtractedText` is indistinguishable
        // downstream from a genuinely blank document, so it must not be the
        // return value for a scan.
        let pdf = build_pdf(&[None, None]);
        match PdfExtractor.extract(&pdf) {
            Err(ExtractError::Malformed { format, detail }) => {
                assert_eq!(format, "PDF");
                assert!(
                    detail.contains("scan") && detail.contains("OCR"),
                    "detail must point at OCR, was {detail:?}"
                );
            }
            other => panic!("expected Malformed for a text-free scan, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_pdf_is_malformed_with_a_usable_detail_and_never_panics() {
        let pdf = build_pdf(&[Some("Some text")]);
        // Cut the xref table and trailer off: a plausible truncated upload.
        let truncated = &pdf[..pdf.len() / 2];
        match PdfExtractor.extract(truncated) {
            Err(ExtractError::Malformed { format, detail }) => {
                assert_eq!(format, "PDF");
                assert!(!detail.is_empty(), "detail must say something usable");
            }
            // Recovery from a truncated file is legitimate — pdf_oxide
            // reconstructs the xref — but only if a page came back with it.
            Ok(out) => assert!(!out.pages.is_empty()),
            Err(other) => panic!("expected Malformed or recovery, got {other}"),
        }
    }

    #[test]
    fn garbage_behind_a_pdf_header_does_not_panic() {
        // Bytes that pass `can_handle` and then make no sense at all. The point
        // is the absence of a panic; either outcome of the parse is acceptable.
        let cases: Vec<Vec<u8>> = vec![
            b"%PDF-1.7\n".to_vec(),
            b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nbinary garbage\n".to_vec(),
            {
                let mut v = b"%PDF-1.5\n".to_vec();
                v.extend((0u8..=255).cycle().take(4096));
                v
            },
            b"%PDF-1.4\nxref\n0 99999999\ntrailer\n<< /Size 99999999 /Root 1 0 R >>\nstartxref\n999999999\n%%EOF".to_vec(),
        ];
        for bytes in cases {
            assert!(PdfExtractor.can_handle(&FormatProbe::new(&bytes, None)));
            match PdfExtractor.extract(&bytes) {
                Err(ExtractError::Malformed { format, .. }) => assert_eq!(format, "PDF"),
                Err(other) => panic!("expected Malformed, got {other}"),
                Ok(out) => assert!(
                    !out.text.trim().is_empty(),
                    "a successful parse must carry text"
                ),
            }
        }
    }

    #[test]
    fn an_encrypted_pdf_is_refused_instead_of_yielding_blank_pages() {
        // A standard security handler the empty user password cannot open. The
        // failure mode being guarded against is not an error but a *success*:
        // without the check, extraction returns blank pages and the caller
        // stores an empty document.
        let mut pdf = build_pdf(&[Some("Secret text")]);
        let marker = b"/Root 1 0 R";
        let at = pdf
            .windows(marker.len())
            .position(|w| w == marker)
            .expect("trailer marker");
        let encrypt = b" /Encrypt << /Filter /Standard /V 5 /R 6 /Length 256 \
                        /O <0011223344556677889900112233445566778899001122334455667788990011> \
                        /U <9988776655443322110099887766554433221100998877665544332211009988> \
                        /P -3904 >> /ID [<01020304050607080910111213141516> \
                        <01020304050607080910111213141516>]";
        pdf.splice(
            at + marker.len()..at + marker.len(),
            encrypt.iter().copied(),
        );

        match PdfExtractor.extract(&pdf) {
            Err(ExtractError::Malformed { format, detail }) => {
                assert_eq!(format, "PDF");
                assert!(!detail.is_empty(), "detail was empty");
            }
            other => panic!("expected Malformed for an encrypted PDF, got {other:?}"),
        }
    }

    #[test]
    fn the_registry_routes_a_pdf_here_even_when_the_filename_lies() {
        use super::super::ExtractorRegistry;
        let pdf = build_pdf(&[Some("Routed correctly")]);
        let out = ExtractorRegistry::with_builtins()
            .extract(&pdf, Some("notes.txt"))
            .expect("registry must hand a %PDF- blob to this extractor");
        assert_eq!(out.format, DocumentFormat::Pdf);
    }
}
