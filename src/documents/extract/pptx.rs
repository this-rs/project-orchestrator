//! Office Open XML presentations (`.pptx`).
//!
//! A `.pptx` is a ZIP of XML parts, like a `.docx`. What is worth reading is
//! the text on each slide (`ppt/slides/slideN.xml`) and the speaker notes
//! attached to it (`ppt/notesSlides/notesSlideM.xml`). Text lives in
//! DrawingML runs: `<a:p>` paragraphs containing `<a:r>` runs containing
//! `<a:t>` text nodes — the same "no whitespace between paragraphs" trap as
//! WordprocessingML, so `</a:p>` emits a newline (see `docx.rs` for why).
//!
//! ## Slide order is the presentation's, not the filesystem's
//!
//! `slide2.xml` is not necessarily the second slide: reordering slides in
//! PowerPoint rewrites `<p:sldIdLst>` in `ppt/presentation.xml` and leaves the
//! part names alone. Order is therefore taken from that list, resolved through
//! `ppt/_rels/presentation.xml.rels`; part-name order is only the fallback for
//! a presentation whose list cannot be read. Likewise a slide's notes are found
//! through the slide's own relationships, not by matching numbers.
//!
//! ## What is left out
//!
//! Slide numbers, footers and date placeholders repeat on every slide and say
//! nothing — dropped. In a notes page only the `body` placeholder is the
//! speaker's text; the thumbnail and slide-number placeholders are not.
//! Images, charts and SmartArt carry no text runs here and are skipped.
//!
//! ## Output shape
//!
//! ```text
//! ## Slide 1
//! Quarterly review
//!
//! ### Notes
//! Mention the churn number first.
//!
//! ## Slide 2
//! ...
//! ```
//!
//! Each slide is one entry in [`ExtractedText::pages`], so a hit is "slide 4".

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use super::{DocumentFormat, ExtractError, ExtractedText, FormatProbe, TextExtractor};
use crate::documents::ByteInterval;

const PRESENTATION_PART: &str = "ppt/presentation.xml";
const PRESENTATION_RELS_PART: &str = "ppt/_rels/presentation.xml.rels";

/// Ceiling on the *decompressed* size of any single XML part. The bound is
/// enforced by a limited read, not by the size the archive declares — see the
/// note on `MAX_DOCUMENT_XML_BYTES` in `docx.rs`. A slide is a few KB to a few
/// MB (embedded tables), so this is generous.
const MAX_PART_BYTES: u64 = 32 * 1024 * 1024;

/// Ceiling on the extracted text across all slides.
pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;

/// Ceiling on slides read, so a hand-built archive with a million slide entries
/// terminates in bounded time.
const MAX_SLIDES: usize = 5_000;

pub struct PptxExtractor;

impl TextExtractor for PptxExtractor {
    fn format(&self) -> DocumentFormat {
        DocumentFormat::Pptx
    }

    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool {
        if !probe.starts_with(b"PK\x03\x04") {
            return false;
        }
        match zip::ZipArchive::new(Cursor::new(probe.bytes)) {
            Ok(archive) => archive.index_for_name(PRESENTATION_PART).is_some(),
            Err(_) => false,
        }
    }

    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
        let mut archive =
            zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| ExtractError::Malformed {
                format: "PPTX",
                detail: format!("not a readable ZIP archive: {e}"),
            })?;

        let mut warnings = Vec::new();
        let slide_parts = ordered_slide_parts(&mut archive, &mut warnings)?;
        if slide_parts.is_empty() {
            warnings.push("the presentation contains no slides".to_string());
        }

        let mut text = String::new();
        let mut pages = Vec::new();
        let mut any_text = false;

        for (i, slide_part) in slide_parts.iter().enumerate() {
            if i >= MAX_SLIDES || text.len() > MAX_TEXT_BYTES {
                warnings.push(format!(
                    "stopped at slide {} of {}: size limit reached",
                    i + 1,
                    slide_parts.len()
                ));
                break;
            }
            let start = text.len();
            text.push_str(&format!("## Slide {}\n", i + 1));

            match read_part(&mut archive, slide_part) {
                Ok(Some(xml)) => {
                    let body = shape_text(&xml, Mode::Slide, slide_part, &mut warnings)?;
                    if !body.is_empty() {
                        any_text = true;
                        text.push_str(&body);
                        text.push('\n');
                    }
                }
                Ok(None) => warnings.push(format!("{slide_part} is listed but missing")),
                Err(detail) => {
                    warnings.push(format!("slide {} skipped: {detail}", i + 1));
                }
            }

            if let Some(notes_part) = notes_part_for(&mut archive, slide_part) {
                match read_part(&mut archive, &notes_part) {
                    Ok(Some(xml)) => {
                        let notes = shape_text(&xml, Mode::Notes, &notes_part, &mut warnings)?;
                        if !notes.is_empty() {
                            any_text = true;
                            text.push_str("\n### Notes\n");
                            text.push_str(&notes);
                            text.push('\n');
                        }
                    }
                    Ok(None) => {}
                    Err(detail) => {
                        warnings.push(format!("notes of slide {} skipped: {detail}", i + 1));
                    }
                }
            }

            text.push('\n');
            pages.push(ByteInterval::new(start, text.len()));
        }

        // The last slide's trailing blank line separates nothing.
        if text.ends_with("\n\n") {
            text.pop();
            if let Some(last) = pages.last_mut() {
                *last = ByteInterval::new(last.start, text.len());
            }
        }

        if !any_text && !slide_parts.is_empty() {
            warnings.push("the presentation contained no text".to_string());
        }

        Ok(ExtractedText {
            text,
            format: DocumentFormat::Pptx,
            pages,
            warnings,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Slide,
    Notes,
}

/// Read one part as UTF-8, refusing to decompress past [`MAX_PART_BYTES`].
///
/// `Ok(None)` when the part is absent; `Err` carries a human-readable reason.
fn read_part<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Option<String>, String> {
    let Some(index) = archive.index_for_name(name) else {
        return Ok(None);
    };
    let part = archive
        .by_index(index)
        .map_err(|e| format!("{name}: {e}"))?;
    let mut raw = Vec::new();
    part.take(MAX_PART_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|e| format!("could not decompress {name}: {e}"))?;
    if raw.len() as u64 > MAX_PART_BYTES {
        return Err(format!(
            "{name} decompresses past the {MAX_PART_BYTES}-byte limit"
        ));
    }
    String::from_utf8(raw).map(Some).map_err(|e| {
        format!(
            "{name} is not valid UTF-8 at byte {}",
            e.utf8_error().valid_up_to()
        )
    })
}

/// A `<Relationship>` from a `.rels` part.
struct Relationship {
    id: String,
    kind: String,
    target: String,
}

fn attr(e: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.local_name().as_ref() == name)
        .and_then(|a| a.unescape_value().ok().map(|v| v.into_owned()))
}

fn parse_relationships(xml: &str) -> Vec<Relationship> {
    let mut reader = Reader::from_str(xml);
    let mut out = Vec::new();
    while let Ok(event) = reader.read_event() {
        match event {
            Event::Start(e) | Event::Empty(e) if e.local_name().as_ref() == b"Relationship" => {
                if let (Some(id), Some(target)) = (attr(&e, b"Id"), attr(&e, b"Target")) {
                    out.push(Relationship {
                        id,
                        kind: attr(&e, b"Type").unwrap_or_default(),
                        target,
                    });
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    out
}

/// Resolve a relationship `target` against the directory of the part that owns
/// the relationship, collapsing `..` — the result is an archive member name.
fn resolve_target(base_dir: &str, target: &str) -> String {
    if let Some(absolute) = target.strip_prefix('/') {
        return absolute.to_string();
    }
    let mut parts: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Slide part names in presentation order.
fn ordered_slide_parts<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    warnings: &mut Vec<String>,
) -> Result<Vec<String>, ExtractError> {
    let presentation = read_part(archive, PRESENTATION_PART)
        .map_err(|detail| ExtractError::Malformed {
            format: "PPTX",
            detail,
        })?
        .ok_or_else(|| ExtractError::Malformed {
            format: "PPTX",
            detail: format!("missing {PRESENTATION_PART}"),
        })?;

    // rIds from <p:sldIdLst>, in list order.
    let mut rids = Vec::new();
    let mut reader = Reader::from_str(&presentation);
    let mut saw_root = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                b"presentation" => saw_root = true,
                b"sldId" => {
                    // `id` (numeric) and `r:id` (relationship reference) share
                    // a local name; the reference is the prefixed one.
                    let reference = e.attributes().flatten().find_map(|a| {
                        a.key
                            .as_ref()
                            .ends_with(b":id")
                            .then(|| a.unescape_value().ok().map(|v| v.into_owned()))
                            .flatten()
                    });
                    if let Some(rid) = reference {
                        rids.push(rid);
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(err) => {
                warnings.push(format!(
                    "{PRESENTATION_PART} is malformed ({err}); slide order falls back to part names"
                ));
                rids.clear();
                break;
            }
        }
    }
    if !saw_root && rids.is_empty() {
        return Err(ExtractError::Malformed {
            format: "PPTX",
            detail: format!("{PRESENTATION_PART} has no <p:presentation> root element"),
        });
    }

    let rels: HashMap<String, String> = read_part(archive, PRESENTATION_RELS_PART)
        .ok()
        .flatten()
        .map(|xml| {
            parse_relationships(&xml)
                .into_iter()
                .map(|r| (r.id, resolve_target("ppt", &r.target)))
                .collect()
        })
        .unwrap_or_default();

    let ordered: Vec<String> = rids
        .iter()
        .filter_map(|rid| rels.get(rid).cloned())
        .filter(|name| archive.index_for_name(name).is_some())
        .collect();
    if !ordered.is_empty() {
        return Ok(ordered);
    }

    // Fallback: every `ppt/slides/slideN.xml`, in numeric order.
    let mut numbered: Vec<(u64, String)> = archive
        .file_names()
        .filter_map(|n| {
            let num = n
                .strip_prefix("ppt/slides/slide")?
                .strip_suffix(".xml")?
                .parse::<u64>()
                .ok()?;
            Some((num, n.to_string()))
        })
        .collect();
    numbered.sort();
    if !numbered.is_empty() && !rids.is_empty() {
        warnings
            .push("could not resolve the slide list; slides are in part-name order".to_string());
    }
    Ok(numbered.into_iter().map(|(_, n)| n).collect())
}

/// The notes part attached to `slide_part`, via the slide's own relationships.
fn notes_part_for<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    slide_part: &str,
) -> Option<String> {
    let (dir, file) = slide_part.rsplit_once('/')?;
    let rels_name = format!("{dir}/_rels/{file}.rels");
    let xml = read_part(archive, &rels_name).ok().flatten()?;
    parse_relationships(&xml)
        .into_iter()
        .find(|r| r.kind.ends_with("/notesSlide"))
        .map(|r| resolve_target(dir, &r.target))
}

/// Text of a slide or notes page, paragraph per line.
///
/// `part` is only used in warnings.
fn shape_text(
    xml: &str,
    mode: Mode,
    part: &str,
    warnings: &mut Vec<String>,
) -> Result<String, ExtractError> {
    let mut reader = Reader::from_str(xml);
    // Significant spaces sit at run edges (`xml:space="preserve"` or not);
    // trimming would weld words across runs.
    reader.config_mut().trim_text(false);

    let mut text = String::new();
    let mut in_text_node = false;
    // Text inside a <a:fld type="slidenum"> is the placeholder glyph "‹#›".
    let mut skip_field = false;
    // Shape being read: where it starts in `text` and which placeholder it is.
    let mut shape_start: Option<usize> = None;
    let mut placeholder: Option<String> = None;
    let mut depth = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                depth += 1;
                match e.local_name().as_ref() {
                    b"sp" if shape_start.is_none() => {
                        shape_start = Some(text.len());
                        placeholder = None;
                    }
                    b"ph" => placeholder = Some(attr(&e, b"type").unwrap_or_else(|| "obj".into())),
                    b"fld" => {
                        skip_field =
                            attr(&e, b"type").is_some_and(|t| t.eq_ignore_ascii_case("slidenum"));
                    }
                    b"t" if !skip_field => in_text_node = true,
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                b"ph" => placeholder = Some(attr(&e, b"type").unwrap_or_else(|| "obj".into())),
                b"br" => text.push('\n'),
                _ => {}
            },
            Ok(Event::End(e)) => {
                depth = depth.saturating_sub(1);
                match e.local_name().as_ref() {
                    b"t" => in_text_node = false,
                    b"fld" => skip_field = false,
                    b"p" => text.push('\n'),
                    // Table cell: tab-separated; row: newline. The paragraph
                    // newline the cell already emitted is replaced.
                    b"tc" => {
                        trim_trailing(&mut text, '\n');
                        text.push('\t');
                    }
                    b"tr" => {
                        trim_trailing(&mut text, '\t');
                        text.push('\n');
                    }
                    b"sp" => {
                        if let Some(start) = shape_start.take() {
                            let keep = match (mode, placeholder.as_deref()) {
                                (Mode::Slide, Some("sldNum" | "ftr" | "dt")) => false,
                                (Mode::Slide, _) => true,
                                (Mode::Notes, Some("body")) => true,
                                (Mode::Notes, _) => false,
                            };
                            if !keep {
                                text.truncate(start);
                            }
                        }
                        placeholder = None;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) if in_text_node => match e.decode() {
                Ok(decoded) => text.push_str(&decoded),
                Err(err) => {
                    warnings.push(format!("dropped an unreadable text run in {part}: {err}"))
                }
            },
            // quick-xml does not resolve entity references; see docx.rs.
            Ok(Event::GeneralRef(e)) if in_text_node => match super::xml::resolve_entity(&e) {
                Some(resolved) => text.push_str(&resolved),
                None => warnings.push(format!(
                    "dropped an entity reference this parser cannot resolve in {part}: &{};",
                    String::from_utf8_lossy(e.as_ref())
                )),
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(err) => {
                if text.is_empty() {
                    return Err(ExtractError::Malformed {
                        format: "PPTX",
                        detail: format!("{part} is not well-formed XML: {err}"),
                    });
                }
                warnings.push(format!(
                    "{part} is malformed at byte {}; kept the {} bytes read before it: {err}",
                    reader.buffer_position(),
                    text.len()
                ));
                depth = 0;
                break;
            }
        }
    }
    if depth > 0 {
        warnings.push(format!(
            "{part} ends with {depth} element(s) unclosed — the part is truncated"
        ));
    }

    text.truncate(text.trim_end_matches(['\n', '\t']).len());
    Ok(text)
}

fn trim_trailing(text: &mut String, ch: char) {
    let len = text.trim_end_matches(ch).len();
    text.truncate(len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn zip_of(entries: &[(String, String)]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in entries {
            writer
                .start_file(name.as_str(), SimpleFileOptions::default())
                .unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    const NS: &str = r#"xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#;
    const REL_NS: &str = "http://schemas.openxmlformats.org/package/2006/relationships";

    fn paragraphs(texts: &[&str]) -> String {
        texts
            .iter()
            .map(|t| format!("<a:p><a:r><a:t>{t}</a:t></a:r></a:p>"))
            .collect()
    }

    /// A shape holding `body`; `ph` is an optional placeholder type.
    fn shape(ph: Option<&str>, body: &str) -> String {
        let ph = ph.map_or(String::new(), |t| {
            format!(r#"<p:nvPr><p:ph type="{t}"/></p:nvPr>"#)
        });
        format!(
            r#"<p:sp><p:nvSpPr><p:cNvPr id="1" name="s"/><p:cNvSpPr/>{ph}</p:nvSpPr><p:txBody>{body}</p:txBody></p:sp>"#
        )
    }

    fn slide_xml(shapes: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><p:sld {NS}><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"#
        )
    }

    fn notes_xml(shapes: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><p:notes {NS}><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:notes>"#
        )
    }

    /// `slides` are `(part number, slide xml, optional notes xml)`, listed in the
    /// order they should appear in the deck.
    fn deck(slides: &[(u32, String, Option<String>)]) -> Vec<u8> {
        let mut entries: Vec<(String, String)> = Vec::new();
        let mut ids = String::new();
        let mut pres_rels = String::new();
        for (i, (n, slide, notes)) in slides.iter().enumerate() {
            ids.push_str(&format!(
                r#"<p:sldId id="{}" r:id="rId{}"/>"#,
                256 + i,
                100 + i
            ));
            pres_rels.push_str(&format!(
                r#"<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{n}.xml"/>"#,
                100 + i
            ));
            entries.push((format!("ppt/slides/slide{n}.xml"), slide.clone()));
            if let Some(notes) = notes {
                // Notes numbered differently from the slide on purpose: the
                // link must come from the relationships, not from matching digits.
                let m = n + 40;
                entries.push((format!("ppt/notesSlides/notesSlide{m}.xml"), notes.clone()));
                entries.push((
                    format!("ppt/slides/_rels/slide{n}.xml.rels"),
                    format!(
                        r#"<Relationships xmlns="{REL_NS}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{m}.xml"/></Relationships>"#
                    ),
                ));
            }
        }
        entries.push((
            PRESENTATION_PART.into(),
            format!(
                r#"<?xml version="1.0"?><p:presentation {NS}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>"#
            ),
        ));
        entries.push((
            PRESENTATION_RELS_PART.into(),
            format!(r#"<Relationships xmlns="{REL_NS}">{pres_rels}</Relationships>"#),
        ));
        zip_of(&entries)
    }

    fn probe<'a>(bytes: &'a [u8], name: Option<&'a str>) -> FormatProbe<'a> {
        FormatProbe::new(bytes, name)
    }

    #[test]
    fn slides_become_numbered_sections_of_paragraphs() {
        let bytes = deck(&[
            (
                1,
                slide_xml(&shape(Some("title"), &paragraphs(&["Quarterly review"]))),
                None,
            ),
            (
                2,
                slide_xml(&shape(None, &paragraphs(&["Revenue up", "Churn down"]))),
                None,
            ),
        ]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.format, DocumentFormat::Pptx);
        assert_eq!(
            out.text,
            "## Slide 1\nQuarterly review\n\n## Slide 2\nRevenue up\nChurn down\n"
        );
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    }

    #[test]
    fn each_slide_is_a_page_that_tiles_the_text() {
        let bytes = deck(&[
            (1, slide_xml(&shape(None, &paragraphs(&["first"]))), None),
            (2, slide_xml(&shape(None, &paragraphs(&["second"]))), None),
        ]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.pages.len(), 2);
        assert_eq!(out.pages[0].start, 0);
        assert_eq!(out.pages[0].end, out.pages[1].start);
        assert_eq!(out.pages[1].end, out.text.len());
        assert_eq!(out.page_of(out.text.find("second").unwrap()), Some(2));
    }

    #[test]
    fn slide_order_follows_the_presentation_not_the_part_names() {
        // slide2.xml is listed first in <p:sldIdLst>.
        let bytes = deck(&[
            (
                2,
                slide_xml(&shape(None, &paragraphs(&["shown first"]))),
                None,
            ),
            (
                1,
                slide_xml(&shape(None, &paragraphs(&["shown second"]))),
                None,
            ),
        ]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert!(
            out.text.starts_with("## Slide 1\nshown first"),
            "{}",
            out.text
        );
        assert!(out.text.contains("## Slide 2\nshown second"));
    }

    #[test]
    fn speaker_notes_follow_their_slide_and_skip_the_page_furniture() {
        let notes = notes_xml(&format!(
            "{}{}{}",
            shape(Some("sldImg"), ""),
            shape(Some("body"), &paragraphs(&["Mention churn first"])),
            shape(Some("sldNum"), &paragraphs(&["7"])),
        ));
        let bytes = deck(&[(
            1,
            slide_xml(&shape(None, &paragraphs(&["Headline"]))),
            Some(notes),
        )]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(
            out.text,
            "## Slide 1\nHeadline\n\n### Notes\nMention churn first\n"
        );
    }

    #[test]
    fn footers_dates_and_slide_numbers_are_dropped_from_slides() {
        let shapes = format!(
            "{}{}{}{}",
            shape(None, &paragraphs(&["Real content"])),
            shape(Some("ftr"), &paragraphs(&["Confidential"])),
            shape(Some("dt"), &paragraphs(&["2026-01-01"])),
            shape(Some("sldNum"), &paragraphs(&["3"])),
        );
        let bytes = deck(&[(1, slide_xml(&shapes), None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "## Slide 1\nReal content\n");
    }

    #[test]
    fn slide_number_fields_inside_body_text_are_dropped() {
        let body = r#"<a:p><a:r><a:t>Page </a:t></a:r><a:fld id="{1}" type="slidenum"><a:t>‹#›</a:t></a:fld></a:p>"#;
        let bytes = deck(&[(1, slide_xml(&shape(None, body)), None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "## Slide 1\nPage \n");
    }

    #[test]
    fn table_cells_are_tab_separated_and_rows_are_lines() {
        let cell = |t: &str| format!("<a:tc><a:txBody>{}</a:txBody></a:tc>", paragraphs(&[t]));
        let row = |a: &str, b: &str| format!("<a:tr>{}{}</a:tr>", cell(a), cell(b));
        let table = format!(
            "<p:graphicFrame><a:graphic><a:graphicData><a:tbl>{}{}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>",
            row("Item", "Cost"),
            row("Licences", "1200"),
        );
        let bytes = deck(&[(1, slide_xml(&table), None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "## Slide 1\nItem\tCost\nLicences\t1200\n");
    }

    #[test]
    fn line_breaks_entities_and_unicode_survive() {
        let body =
            "<a:p><a:r><a:t>R&amp;D</a:t></a:r><a:br/><a:r><a:t>café 日本語</a:t></a:r></a:p>";
        let bytes = deck(&[(1, slide_xml(&shape(None, body)), None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "## Slide 1\nR&D\ncafé 日本語\n");
    }

    #[test]
    fn a_deck_with_no_text_says_so() {
        let bytes = deck(&[(1, slide_xml(""), None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert!(
            out.warnings.iter().any(|w| w.contains("no text")),
            "{:?}",
            out.warnings
        );
    }

    #[test]
    fn claims_a_real_deck_by_content_whatever_it_is_called() {
        let bytes = deck(&[(1, slide_xml(""), None)]);
        assert!(PptxExtractor.can_handle(&probe(&bytes, Some("deck.bin"))));
        assert!(PptxExtractor.can_handle(&probe(&bytes, None)));
    }

    #[test]
    fn refuses_other_zips_however_they_are_named() {
        let jar = zip_of(&[(
            "META-INF/MANIFEST.MF".into(),
            "Manifest-Version: 1.0".into(),
        )]);
        assert!(!PptxExtractor.can_handle(&probe(&jar, Some("deck.pptx"))));
        assert!(!PptxExtractor.can_handle(&probe(b"PK\x03\x04garbage", Some("deck.pptx"))));
        assert!(!PptxExtractor.can_handle(&probe(b"plain", Some("deck.pptx"))));
    }

    #[test]
    fn a_presentation_part_that_is_not_xml_is_malformed() {
        let bytes = zip_of(&[(PRESENTATION_PART.into(), "not xml".into())]);
        let err = PptxExtractor.extract(&bytes).unwrap_err();
        assert!(
            matches!(err, ExtractError::Malformed { format: "PPTX", .. }),
            "{err:?}"
        );
    }

    #[test]
    fn unreadable_slide_order_falls_back_to_part_names() {
        let mut entries = vec![
            (
                PRESENTATION_PART.to_string(),
                format!(r#"<p:presentation {NS}/>"#),
            ),
            (
                "ppt/slides/slide10.xml".into(),
                slide_xml(&shape(None, &paragraphs(&["ten"]))),
            ),
            (
                "ppt/slides/slide2.xml".into(),
                slide_xml(&shape(None, &paragraphs(&["two"]))),
            ),
        ];
        entries.push(("docProps/app.xml".into(), "<x/>".into()));
        let out = PptxExtractor.extract(&zip_of(&entries)).unwrap();
        // Numeric, not lexicographic: 2 before 10.
        assert_eq!(out.text, "## Slide 1\ntwo\n\n## Slide 2\nten\n");
    }

    #[test]
    fn a_slide_part_that_breaks_partway_keeps_what_was_read() {
        let broken = format!(
            r#"<p:sld {NS}><p:cSld><p:spTree>{}<p:sp><a:p><a:r><a:t>tail</a:t><<<"#,
            shape(None, &paragraphs(&["kept"]))
        );
        let bytes = deck(&[(1, broken, None)]);
        let out = PptxExtractor.extract(&bytes).unwrap();
        assert!(out.text.contains("kept"));
        assert!(
            out.warnings.iter().any(|w| w.contains("malformed")),
            "{:?}",
            out.warnings
        );
    }

    #[test]
    fn resolves_relationship_targets_against_the_owning_directory() {
        assert_eq!(
            resolve_target("ppt/slides", "../notesSlides/n.xml"),
            "ppt/notesSlides/n.xml"
        );
        assert_eq!(resolve_target("ppt", "slides/s.xml"), "ppt/slides/s.xml");
        assert_eq!(
            resolve_target("ppt", "/ppt/slides/s.xml"),
            "ppt/slides/s.xml"
        );
        assert_eq!(resolve_target("ppt", "../../../x.xml"), "x.xml");
    }

    #[test]
    fn the_registry_routes_a_deck_here_and_not_to_plain_text() {
        let reg = super::super::ExtractorRegistry::with_builtins();
        let bytes = deck(&[(1, slide_xml(&shape(None, &paragraphs(&["routed"]))), None)]);
        let out = reg.extract(&bytes, Some("anything.bin")).unwrap();
        assert_eq!(out.format, DocumentFormat::Pptx);
        assert!(out.text.contains("routed"));
    }
}
