//! Office Open XML word processing documents (`.docx`).
//!
//! A `.docx` is a ZIP archive of XML parts. The body text lives in one of them,
//! `word/document.xml`, as a tree of paragraphs (`<w:p>`) containing runs
//! (`<w:r>`) containing text nodes (`<w:t>`). Everything else in the archive —
//! styles, fonts, numbering, relationships, embedded images — is presentation
//! or binary, and none of it is text a reader would quote.
//!
//! ## Why the container has to be opened before claiming the file
//!
//! `PK\x03\x04` is the magic of the whole ZIP family: `.docx`, `.xlsx`, `.pptx`,
//! `.odt`, `.jar`, `.apk`, `.epub`, and any plain archive someone renamed. The
//! extension does not settle it either — an uploader controls the filename, and
//! [`super`] is explicit that it is a hint from an untrusted source. The only
//! evidence that a blob is a Word document is that its archive actually
//! contains `word/document.xml`, so [`DocxExtractor::can_handle`] opens the
//! central directory and looks. That costs one seek to the end of the buffer
//! and a directory parse; it does not decompress anything.
//!
//! ## Why the separators matter as much as the text
//!
//! WordprocessingML carries no whitespace between paragraphs: two `<w:p>`
//! elements are adjacent in the XML and their `<w:t>` contents would
//! concatenate into one run-on string. Extracting only the text nodes therefore
//! produces a document that is a *single line*, which reads plausibly and is
//! useless: [`super::super::chunk`] splits on sentence and paragraph
//! boundaries, and a document with none of either collapses into one
//! unsplittable chunk. So `</w:p>` emits a newline, `<w:br/>` a newline and
//! `<w:tab/>` a tab — the layout elements are content, not decoration.

use std::io::{Cursor, Read};

use quick_xml::events::Event;
use quick_xml::Reader;

use super::{DocumentFormat, ExtractError, ExtractedText, FormatProbe, TextExtractor};

/// The archive member holding the document body. Fixed by ECMA-376; a Word
/// document that lacks it is not one.
const DOCUMENT_PART: &str = "word/document.xml";

/// Ceiling on the *decompressed* size of `word/document.xml`.
///
/// A ZIP stores the uncompressed size in its headers, but that field is written
/// by whoever built the archive and a bomb simply lies about it. The only
/// honest defence is to stop reading, which is what this bounds.
///
/// Sized against the largest plausible real document rather than a round
/// number. A 1 000-page report holds on the order of 400 words per page, i.e.
/// ~400 000 words ≈ 2.5 MB of prose. WordprocessingML wraps every styled span
/// in its own `<w:r><w:rPr>…</w:rPr><w:t>` scaffolding, which in practice costs
/// 10–20x the text it carries, so such a document lands around 25–50 MB of XML.
/// 64 MiB leaves headroom above that worst case while a classic bomb — a few
/// hundred KB of archive expanding to gigabytes of a single repeated byte — is
/// cut off after reading 64 MiB, having allocated exactly that and no more.
const MAX_DOCUMENT_XML_BYTES: u64 = 64 * 1024 * 1024;

pub struct DocxExtractor;

impl TextExtractor for DocxExtractor {
    fn format(&self) -> DocumentFormat {
        DocumentFormat::Docx
    }

    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool {
        // Cheap gate first: everything below parses a ZIP directory, and the
        // registry probes every extractor against every upload.
        if !probe.starts_with(b"PK\x03\x04") {
            return false;
        }
        // Then the only evidence that distinguishes a .docx from every other
        // ZIP: the body part is really in there. A .xlsx (`xl/workbook.xml`), a
        // .jar (`META-INF/MANIFEST.MF`) and a renamed archive all fail here,
        // whatever their extension claims.
        match zip::ZipArchive::new(Cursor::new(probe.bytes)) {
            Ok(archive) => archive.index_for_name(DOCUMENT_PART).is_some(),
            // A truncated or corrupt archive is not this extractor's to claim.
            // Declining sends it back to the registry, which reports what was
            // sniffed instead of a misleading "malformed DOCX".
            Err(_) => false,
        }
    }

    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
        let mut archive =
            zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| ExtractError::Malformed {
                format: "DOCX",
                detail: format!("not a readable ZIP archive: {e}"),
            })?;

        let mut warnings = Vec::new();
        let xml = read_document_part(&mut archive)?;
        let text = extract_body_text(&xml, &mut warnings)?;

        if text.is_empty() {
            // Not an error: a document really can be empty, or hold nothing but
            // images. Saying so beats handing the caller an empty string that
            // looks like a successful extraction of nothing.
            warnings.push(format!("{DOCUMENT_PART} contained no text"));
        }

        Ok(ExtractedText {
            text,
            format: DocumentFormat::Docx,
            // A .docx has no stable pagination: page breaks are computed by the
            // renderer from the current font metrics, paper size and printer,
            // so "page 7" is a property of a viewing session and not of the
            // file. Recording invented page spans here would make every
            // provenance report downstream confidently wrong, so `pages` stays
            // empty — see the module docs on `pages` being information, not a
            // gap. (`<w:lastRenderedPageBreak>` hints exist but only record
            // where *some* renderer once broke, and are absent as often as not.)
            pages: Vec::new(),
            warnings,
        })
    }
}

/// Read `word/document.xml` out of the archive, refusing to decompress more
/// than [`MAX_DOCUMENT_XML_BYTES`].
fn read_document_part<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<String, ExtractError> {
    let mut part = archive
        .by_name(DOCUMENT_PART)
        .map_err(|e| ExtractError::Malformed {
            format: "DOCX",
            detail: format!("missing {DOCUMENT_PART}: {e}"),
        })?;

    // The declared size is a claim, not a fact, so it is only used to fail fast
    // on an archive that is honest about being enormous. The real guard is the
    // bounded read below, which is what stops an archive that under-reports.
    let declared = part.size();
    if declared > MAX_DOCUMENT_XML_BYTES {
        return Err(ExtractError::Malformed {
            format: "DOCX",
            detail: format!(
                "{DOCUMENT_PART} declares {declared} bytes, over the {MAX_DOCUMENT_XML_BYTES}-byte limit"
            ),
        });
    }

    // `take` is what bounds the *memory*: whatever the headers claimed, this
    // buffer cannot grow past the limit plus one byte. The extra byte is what
    // makes the overflow observable — if it arrives, more was coming.
    let mut raw = Vec::new();
    part.by_ref()
        .take(MAX_DOCUMENT_XML_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|e| ExtractError::Malformed {
            format: "DOCX",
            detail: format!("could not decompress {DOCUMENT_PART}: {e}"),
        })?;

    if raw.len() as u64 > MAX_DOCUMENT_XML_BYTES {
        return Err(ExtractError::Malformed {
            format: "DOCX",
            detail: format!(
                "{DOCUMENT_PART} decompresses past the {MAX_DOCUMENT_XML_BYTES}-byte limit \
                 (declared {declared} bytes) — refusing to expand it further"
            ),
        });
    }

    String::from_utf8(raw).map_err(|e| ExtractError::Malformed {
        format: "DOCX",
        detail: format!(
            "{DOCUMENT_PART} is not valid UTF-8 at byte {}",
            e.utf8_error().valid_up_to()
        ),
    })
}

/// Walk the body XML, emitting text and the separators that give it structure.
///
/// A streaming pass rather than a DOM: the tree is deep (body -> paragraph ->
/// run -> text) and nothing here needs an ancestor, only "am I inside a
/// `<w:t>`".
fn extract_body_text(xml: &str, warnings: &mut Vec<String>) -> Result<String, ExtractError> {
    let mut reader = Reader::from_str(xml);
    // `<w:t xml:space="preserve">` marks runs whose leading or trailing spaces
    // are real text — trimming here would silently weld words together across
    // run boundaries, which is exactly the kind of corruption an aligner cannot
    // detect afterwards.
    reader.config_mut().trim_text(false);

    let mut text = String::new();
    // `<w:t>` content arrives as its own event, so "inside a text node" has to
    // be tracked across events rather than read off the current one.
    let mut in_text_node = false;
    // `<w:tab/>` is ambiguous outside a run: inside `<w:pPr><w:tabs>` the same
    // element declares a tab *stop* at a position, which is layout metadata and
    // not a character. Per ECMA-376 the separators are children of `<w:r>`, so
    // that is the gate — without it every paragraph with custom tab stops
    // gains a stray tab per stop.
    let mut in_run = false;
    // quick-xml reports EOF happily on a truncated file, so unbalanced depth is
    // the only signal that the part was cut short rather than ended.
    let mut depth = 0usize;
    // A ZIP member named `word/document.xml` that is not WordprocessingML is
    // possible (anyone can build such an archive) and must not extract as an
    // empty success.
    let mut saw_document_root = false;
    // Field codes (`<w:instrText>`, e.g. `PAGE \* MERGEFORMAT`) and deleted
    // runs (`<w:delText>`, tracked changes) are text nodes too, and neither is
    // anything a reader sees. Counted so the skip is reported rather than
    // silent — a document that lost content should say so.
    let mut skipped_hidden_runs = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                depth += 1;
                match e.local_name().as_ref() {
                    b"document" => saw_document_root = true,
                    b"r" => in_run = true,
                    b"t" => in_text_node = true,
                    b"instrText" | b"delText" => skipped_hidden_runs += 1,
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                depth = depth.saturating_sub(1);
                match e.local_name().as_ref() {
                    b"r" => in_run = false,
                    b"t" => in_text_node = false,
                    // The paragraph break. Without it the whole document is one
                    // line and the sentence chunker downstream has nothing to
                    // cut on.
                    b"p" => text.push('\n'),
                    _ => {}
                }
            }
            // An empty `<w:document/>` is a real, if pointless, document.
            Ok(Event::Empty(e)) if e.local_name().as_ref() == b"document" => {
                saw_document_root = true;
            }
            // The separators are empty elements, so they only ever arrive here.
            Ok(Event::Empty(e)) if in_run => match e.local_name().as_ref() {
                b"tab" => text.push('\t'),
                // `<w:br/>` is a soft line break inside a paragraph;
                // `<w:cr/>` is the legacy carriage return with the same effect.
                b"br" | b"cr" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Text(e)) if in_text_node => match e.decode() {
                Ok(decoded) => text.push_str(&decoded),
                Err(err) => {
                    // One unreadable run is not a reason to lose the document,
                    // but losing it silently would poison every chunk built on
                    // the result with no way for anyone to notice.
                    warnings.push(format!("dropped an unreadable text run: {err}"));
                }
            },
            // quick-xml splits text around entity references rather than
            // resolving them, so `R&amp;D` arrives as three events. Ignoring
            // this one would quietly delete the `&` from every company name and
            // every `&lt;` from every quoted snippet.
            Ok(Event::GeneralRef(e)) if in_text_node => match resolve_entity(&e) {
                Some(resolved) => text.push_str(&resolved),
                None => warnings.push(format!(
                    "dropped an entity reference this parser cannot resolve: &{};",
                    String::from_utf8_lossy(e.as_ref())
                )),
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(err) => {
                // Malformed XML partway through: keep what was recovered and
                // say where it stopped. Returning an error would throw away a
                // document that is 99% readable; returning the prefix silently
                // would pass a truncated document off as whole.
                if text.is_empty() {
                    return Err(ExtractError::Malformed {
                        format: "DOCX",
                        detail: format!("{DOCUMENT_PART} is not well-formed XML: {err}"),
                    });
                }
                warnings.push(format!(
                    "{DOCUMENT_PART} is malformed at byte {}; kept the {} bytes read before it: {err}",
                    reader.buffer_position(),
                    text.len()
                ));
                depth = 0;
                break;
            }
        }
    }

    if !saw_document_root {
        return Err(ExtractError::Malformed {
            format: "DOCX",
            detail: format!("{DOCUMENT_PART} has no <w:document> root element"),
        });
    }
    if depth > 0 {
        // Reached the end of the part with elements still open. quick-xml
        // returns a clean `Eof` here, so without this check a document
        // truncated mid-body extracts as a complete one.
        warnings.push(format!(
            "{DOCUMENT_PART} ends with {depth} element(s) unclosed — the part is truncated \
             and the text below stops where it was cut"
        ));
    }

    if skipped_hidden_runs > 0 {
        warnings.push(format!(
            "skipped {skipped_hidden_runs} field-code or deleted-text run(s), which are not visible document text"
        ));
    }

    // Every paragraph ends with a newline, so a well-formed document always
    // ends with one, plus one more per trailing empty paragraph. Those carry no
    // content and would surface downstream as empty chunks.
    text.truncate(text.trim_end_matches('\n').len());
    Ok(text)
}

/// Resolve one entity reference to the text it stands for.
///
/// Delegates to quick-xml rather than matching the five predefined names by
/// hand, because Word also emits numeric references (`&#8217;` for a curly
/// apostrophe) and those have to resolve the same way.
fn resolve_entity(reference: &quick_xml::events::BytesRef<'_>) -> Option<String> {
    let name = reference.decode().ok()?;
    quick_xml::escape::unescape(&format!("&{name};"))
        .ok()
        .map(|s| s.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    /// Build a ZIP in memory from `(name, content)` pairs.
    ///
    /// Tests build their own archives rather than carrying binary fixtures: a
    /// checked-in `.docx` hides what it contains, and half of what is under
    /// test here is the handling of archives no word processor would produce.
    fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in entries {
            writer
                .start_file(*name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(content).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    /// Rewrite the uncompressed-size fields of a single-entry ZIP so the
    /// archive under-reports what it contains.
    ///
    /// The bytes are patched by hand because no ZIP writer will produce this:
    /// that is the point. The size lives in two places — the local file header
    /// at offset 22 and the central directory entry at offset 24 — and a reader
    /// that believed either one would be fooled.
    fn with_falsified_sizes(mut bytes: Vec<u8>, claimed: u32) -> Vec<u8> {
        let claimed = claimed.to_le_bytes();
        assert_eq!(&bytes[0..4], b"PK\x03\x04", "expected a local file header");
        bytes[22..26].copy_from_slice(&claimed);
        let cd = bytes
            .windows(4)
            .position(|w| w == b"PK\x01\x02")
            .expect("central directory header");
        bytes[cd + 24..cd + 28].copy_from_slice(&claimed);
        bytes
    }

    /// A minimal but structurally real `.docx` whose body is `document_xml`.
    fn docx_with_body(document_xml: &str) -> Vec<u8> {
        zip_of(&[
            (
                "[Content_Types].xml",
                br#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>"#.as_slice(),
            ),
            (DOCUMENT_PART, document_xml.as_bytes()),
        ])
    }

    /// Wrap plain paragraph texts in the WordprocessingML a word processor emits.
    fn docx_with_paragraphs(paragraphs: &[&str]) -> Vec<u8> {
        let body: String = paragraphs
            .iter()
            .map(|p| format!("<w:p><w:r><w:t>{p}</w:t></w:r></w:p>"))
            .collect();
        docx_with_body(&format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}</w:body></w:document>"#
        ))
    }

    fn probe<'a>(bytes: &'a [u8], name: Option<&'a str>) -> FormatProbe<'a> {
        FormatProbe::new(bytes, name)
    }

    #[test]
    fn extracts_paragraph_text() {
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&["First.", "Second."]))
            .unwrap();
        assert_eq!(out.text, "First.\nSecond.");
        assert_eq!(out.format, DocumentFormat::Docx);
        assert!(out.warnings.is_empty(), "warnings: {:?}", out.warnings);
    }

    #[test]
    fn paragraphs_do_not_run_together() {
        // The failure this guards against is not a crash: drop the `</w:p>`
        // handling and this document extracts as "One sentence.Another
        // sentence.Third." — one line, no sentence boundary the chunker can
        // find, one unsplittable chunk for the whole document.
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&[
                "One sentence.",
                "Another sentence.",
                "Third.",
            ]))
            .unwrap();
        assert_eq!(out.text.lines().count(), 3, "got {:?}", out.text);
        assert!(!out.text.contains("sentence.Another"));
    }

    #[test]
    fn tabs_and_line_breaks_survive_as_separators() {
        // `<w:tab/>` is how a table of contents separates an entry from its
        // page number, and `<w:br/>` how an address block separates its lines.
        // Dropped, both become a single word glued to the next.
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Introduction</w:t><w:tab/><w:t>3</w:t></w:r></w:p>
<w:p><w:r><w:t>Line one</w:t><w:br/><w:t>Line two</w:t></w:r></w:p>
<w:p><w:r><w:t>Old style</w:t><w:cr/><w:t>break</w:t></w:r></w:p>
</w:body></w:document>"#,
            ))
            .unwrap();
        assert_eq!(
            out.text,
            "Introduction\t3\nLine one\nLine two\nOld style\nbreak"
        );
    }

    #[test]
    fn tab_stop_declarations_are_not_tab_characters() {
        // `<w:tab/>` means two different things depending on where it sits: a
        // tab character inside a run, a tab *stop* position inside paragraph
        // properties. Treating both as text prepends a stray tab to every
        // paragraph that defines stops — which is most of them in a document
        // with a table of contents.
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:tabs><w:tab w:val="right" w:pos="9026"/><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr><w:r><w:t>Chapter one</w:t><w:tab/><w:t>7</w:t></w:r></w:p>
</w:body></w:document>"#,
            ))
            .unwrap();
        assert_eq!(out.text, "Chapter one\t7");
    }

    #[test]
    fn empty_paragraphs_become_blank_lines_not_nothing() {
        // A blank line is the paragraph separator every prose chunker keys on;
        // collapsing it would merge two sections into one block of text.
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&["Heading", "", "Body text."]))
            .unwrap();
        assert_eq!(out.text, "Heading\n\nBody text.");
    }

    #[test]
    fn preserves_significant_whitespace_inside_runs() {
        // Word splits a styled phrase across runs and marks the spaces it must
        // keep. Trimming run text welds the words: "Boldtextand more".
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t xml:space="preserve">Bold </w:t></w:r><w:r><w:t xml:space="preserve">text </w:t></w:r><w:r><w:t>and more</w:t></w:r></w:p>
</w:body></w:document>"#,
            ))
            .unwrap();
        assert_eq!(out.text, "Bold text and more");
    }

    #[test]
    fn preserves_accented_and_cjk_text() {
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&[
                "Une phrase accentuée, à relire.",
                "日本語のテキストも保持される。",
            ]))
            .unwrap();
        assert_eq!(
            out.text,
            "Une phrase accentuée, à relire.\n日本語のテキストも保持される。"
        );
    }

    #[test]
    fn decodes_xml_entities_rather_than_leaking_markup() {
        // `&amp;` reaching a chunk verbatim would be quoted back to a user as
        // markup they never wrote.
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&["R&amp;D &lt;draft&gt;"]))
            .unwrap();
        assert_eq!(out.text, "R&D <draft>");
    }

    #[test]
    fn ignores_markup_that_is_not_visible_text_and_says_so() {
        // Field codes and tracked deletions are text nodes in the XML but not
        // in the document. Emitting them would inject "PAGE \* MERGEFORMAT" and
        // retracted sentences into the extracted prose.
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Visible.</w:t></w:r><w:r><w:instrText> PAGE \* MERGEFORMAT </w:instrText></w:r></w:p>
<w:p><w:del><w:r><w:delText>Retracted sentence.</w:delText></w:r></w:del><w:r><w:t>Kept.</w:t></w:r></w:p>
</w:body></w:document>"#,
            ))
            .unwrap();
        assert_eq!(out.text, "Visible.\nKept.");
        assert!(
            out.warnings.iter().any(|w| w.contains("skipped 2")),
            "dropping content must be reported, got {:?}",
            out.warnings
        );
    }

    #[test]
    fn a_docx_has_no_pages_and_does_not_invent_any() {
        // Pagination in a .docx is the renderer's, not the file's.
        let out = DocxExtractor
            .extract(&docx_with_paragraphs(&["Anything."]))
            .unwrap();
        assert!(out.pages.is_empty());
        assert_eq!(out.page_of(0), None);
    }

    #[test]
    fn empty_body_is_reported_rather_than_returned_as_success() {
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>"#,
            ))
            .unwrap();
        assert_eq!(out.text, "");
        assert!(
            out.warnings.iter().any(|w| w.contains("no text")),
            "warnings: {:?}",
            out.warnings
        );
    }

    #[test]
    fn claims_a_real_docx() {
        let bytes = docx_with_paragraphs(&["Hello."]);
        assert!(DocxExtractor.can_handle(&probe(&bytes, Some("report.docx"))));
        // And still claims it when the filename says nothing useful: the
        // decision is the archive's contents, not the name.
        assert!(DocxExtractor.can_handle(&probe(&bytes, None)));
        assert!(DocxExtractor.can_handle(&probe(&bytes, Some("attachment"))));
    }

    #[test]
    fn refuses_other_zips_however_they_are_named() {
        // The whole point of looking inside: these share `.docx`'s magic bytes
        // exactly, and a `.docx` extension is one rename away.
        for (label, entries) in [
            ("xlsx", vec![("xl/workbook.xml", b"<workbook/>".as_slice())]),
            (
                "jar",
                vec![(
                    "META-INF/MANIFEST.MF",
                    b"Manifest-Version: 1.0\n".as_slice(),
                )],
            ),
            ("plain archive", vec![("notes.txt", b"hello".as_slice())]),
            // Close enough to fool a prefix check, still not the body part.
            (
                "near miss",
                vec![("word/settings.xml", b"<settings/>".as_slice())],
            ),
        ] {
            let bytes = zip_of(&entries);
            assert!(
                !DocxExtractor.can_handle(&probe(&bytes, Some("renamed.docx"))),
                "{label} must be refused despite the .docx extension"
            );
        }
    }

    #[test]
    fn refuses_a_corrupt_archive_instead_of_claiming_it() {
        // Truncated upload: the ZIP magic is there, the central directory is
        // not. Claiming it would turn "your upload is incomplete" into
        // "malformed DOCX", which points at the wrong thing.
        let mut bytes = docx_with_paragraphs(&["Hello."]);
        bytes.truncate(bytes.len() / 2);
        assert!(!DocxExtractor.can_handle(&probe(&bytes, Some("report.docx"))));
        assert!(matches!(
            DocxExtractor.extract(&bytes),
            Err(ExtractError::Malformed { format: "DOCX", .. })
        ));
    }

    #[test]
    fn refuses_a_zip_whose_body_part_is_missing_at_extract_time() {
        // `extract` is public and reachable without `can_handle` having run.
        let bytes = zip_of(&[("xl/workbook.xml", b"<workbook/>".as_slice())]);
        match DocxExtractor.extract(&bytes) {
            Err(ExtractError::Malformed { detail, .. }) => {
                assert!(detail.contains(DOCUMENT_PART), "detail was {detail:?}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn refuses_an_archive_that_declares_an_oversized_body_part() {
        // A tiny archive whose body part inflates without bound, with honest
        // headers. Caught before a single byte is decompressed.
        let payload = vec![b'A'; (MAX_DOCUMENT_XML_BYTES + 4096) as usize];
        let bytes = zip_of(&[(DOCUMENT_PART, &payload)]);
        assert!(
            (bytes.len() as u64) < MAX_DOCUMENT_XML_BYTES / 100,
            "the archive itself must stay small, was {} bytes",
            bytes.len()
        );
        match DocxExtractor.extract(&bytes) {
            Err(ExtractError::Malformed { detail, .. }) => {
                assert!(detail.contains("declares"), "detail was {detail:?}");
            }
            other => panic!("expected the bomb to be refused, got {other:?}"),
        }
    }

    #[test]
    fn refuses_a_zip_bomb_that_under_reports_its_size() {
        // The case the declared-size check cannot see. An attacker writes the
        // archive, so the size in its headers is whatever they want it to be:
        // here it claims 1 KB and delivers 64 MiB + 4 KiB. Only the bounded
        // read stops this, and only because it never trusts that field.
        let payload = vec![b'A'; (MAX_DOCUMENT_XML_BYTES + 4096) as usize];
        let bytes = with_falsified_sizes(zip_of(&[(DOCUMENT_PART, &payload)]), 1024);

        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes[..])).unwrap();
        assert_eq!(
            archive.by_name(DOCUMENT_PART).unwrap().size(),
            1024,
            "the archive must be lying for this test to mean anything"
        );

        match DocxExtractor.extract(&bytes) {
            Err(ExtractError::Malformed { detail, .. }) => {
                assert!(
                    detail.contains("decompresses past"),
                    "detail was {detail:?}"
                );
            }
            other => panic!("expected the bomb to be refused, got {other:?}"),
        }
    }

    #[test]
    fn keeps_what_it_read_when_the_xml_breaks_partway() {
        // A truncated body part still holds real prose. Erroring out would lose
        // it; keeping it without a warning would pass a half document off as
        // whole.
        let out = DocxExtractor
            .extract(&docx_with_body(
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Recovered text.</w:t></w:r></w:p>
<w:p><w:r><w:t>Cut off here"#,
            ))
            .unwrap();
        assert!(out.text.starts_with("Recovered text."), "{:?}", out.text);
        // quick-xml reports a clean `Eof` on a truncated part, so only the
        // unclosed-element count separates this from a whole document.
        assert!(
            out.warnings.iter().any(|w| w.contains("truncated")),
            "a truncated document must not look complete, warnings: {:?}",
            out.warnings
        );
    }

    #[test]
    fn rejects_a_body_part_that_is_not_xml_at_all() {
        // Nothing recoverable, so this is an error rather than a warning.
        let bytes = docx_with_body("this is not XML");
        assert!(matches!(
            DocxExtractor.extract(&bytes),
            Err(ExtractError::Malformed { format: "DOCX", .. })
        ));
    }

    #[test]
    fn rejects_a_body_part_that_is_not_utf8() {
        let bytes = zip_of(&[(DOCUMENT_PART, &[0xff, 0xfe, 0x00, 0x41])]);
        match DocxExtractor.extract(&bytes) {
            Err(ExtractError::Malformed { detail, .. }) => {
                assert!(detail.contains("UTF-8"), "detail was {detail:?}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn the_registry_routes_a_docx_here_and_not_to_plain_text() {
        // The catch-all text extractor refuses ZIP magic, but this checks the
        // whole path: a .docx must reach this extractor through the registry,
        // with the extension carrying none of the decision.
        let reg = super::super::ExtractorRegistry::with_builtins();
        let bytes = docx_with_paragraphs(&["Routed correctly."]);
        let out = reg.extract(&bytes, Some("anything.bin")).unwrap();
        assert_eq!(out.format, DocumentFormat::Docx);
        assert_eq!(out.text, "Routed correctly.");
    }
}
