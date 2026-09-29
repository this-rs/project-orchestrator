//! Office Open XML spreadsheets (`.xlsx`).
//!
//! Unlike `.docx` and `.pptx`, a spreadsheet is not worth parsing by hand:
//! cell values are split between the sheet XML and a shared-string table, and
//! numbers, booleans, dates and errors are all typed. [`calamine`] handles that;
//! this module only decides what to *emit*.
//!
//! ## Output shape
//!
//! One section per sheet, `# Sheet: <name>` followed by the used rows as
//! tab-separated text, one line per row:
//!
//! ```text
//! # Sheet: Budget
//! Item\tQ1\tQ2
//! Licences\t1200\t1300
//! ```
//!
//! Each sheet is one entry in [`ExtractedText::pages`], so a chunk found in a
//! spreadsheet can be reported as "sheet 2" the same way a PDF chunk is "page 2".
//! Sheet order is workbook order, which a user can see, unlike XML part order.
//!
//! ## Bounded on purpose
//!
//! The sheet is read as a *stream* of cells ([`calamine::Xlsx::worksheet_cells_reader`])
//! rather than through `worksheet_range`, which materialises a dense grid sized
//! to the bounding box of the used cells: one stray value in `XFD1048576` would
//! allocate tens of billions of cells. Streaming costs nothing per cell
//! we don't emit. Output is capped at [`MAX_TEXT_BYTES`], each sheet at
//! [`MAX_COLUMNS`] columns, and the declared decompressed size of the whole
//! archive at [`MAX_DECLARED_UNCOMPRESSED_BYTES`]. Whatever is cut is reported
//! in `warnings`, never silently dropped.

use std::io::Cursor;

use calamine::{Data, Reader, Xlsx};

use super::{DocumentFormat, ExtractError, ExtractedText, FormatProbe, TextExtractor};
use crate::documents::ByteInterval;

/// The workbook part. Fixed by ECMA-376; an `.xlsx` that lacks it is not one.
const WORKBOOK_PART: &str = "xl/workbook.xml";

/// Ceiling on the extracted text, across all sheets.
///
/// Extracted text is chunked and embedded downstream, so an unbounded dump of a
/// million-row export is not a feature. 4 MiB is roughly 100 000 rows of a
/// dozen short columns — a very large working spreadsheet.
pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;

/// Widest row emitted. Column `XFD` is 16 384; anything past this is far more
/// likely a formatting artefact than data, and each empty column costs a tab.
const MAX_COLUMNS: u32 = 1024;

/// Ceiling on cells inspected across the workbook, so a sheet of a million
/// blank-but-styled cells terminates in bounded time.
const MAX_CELLS: u64 = 20_000_000;

/// Ceiling on the sum of the *declared* uncompressed sizes of all entries.
///
/// The declaration is a claim (see the note in `docx.rs`), so this only rejects
/// archives honest about being enormous. calamine offers no hook to bound the
/// actual decompression, which is why the cell and text caps exist as well.
const MAX_DECLARED_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;

pub struct XlsxExtractor;

impl TextExtractor for XlsxExtractor {
    fn format(&self) -> DocumentFormat {
        DocumentFormat::Xlsx
    }

    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool {
        if !probe.starts_with(b"PK\x03\x04") {
            return false;
        }
        // Same reasoning as the DOCX extractor: the only evidence that a ZIP is
        // a workbook is that it contains one, whatever its extension claims.
        match zip::ZipArchive::new(Cursor::new(probe.bytes)) {
            Ok(archive) => archive.index_for_name(WORKBOOK_PART).is_some(),
            Err(_) => false,
        }
    }

    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
        let malformed = |detail: String| ExtractError::Malformed {
            format: "XLSX",
            detail,
        };

        // Fail fast on an archive that is honest about being enormous.
        let archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| malformed(format!("not a readable ZIP archive: {e}")))?;
        // `None` means the sizes cannot be known up front (data descriptors);
        // the cell and text caps below are then the only guard.
        let declared: u128 = archive.decompressed_size().unwrap_or(0);
        if declared > u128::from(MAX_DECLARED_UNCOMPRESSED_BYTES) {
            return Err(malformed(format!(
                "archive declares {declared} uncompressed bytes, over the \
                 {MAX_DECLARED_UNCOMPRESSED_BYTES}-byte limit"
            )));
        }
        drop(archive);

        let mut workbook: Xlsx<_> = Xlsx::new(Cursor::new(bytes))
            .map_err(|e| malformed(format!("not a readable workbook: {e}")))?;

        let mut warnings = Vec::new();
        let mut text = String::new();
        let mut pages = Vec::new();
        let mut cells_seen: u64 = 0;
        let mut truncated = false;

        for name in workbook.sheet_names() {
            if truncated {
                warnings.push(format!("sheet {name:?} skipped: text limit reached"));
                continue;
            }
            let section_start = text.len();
            text.push_str("# Sheet: ");
            text.push_str(&name);
            text.push('\n');

            let mut reader = match workbook.worksheet_cells_reader(&name) {
                Ok(r) => r,
                Err(e) => {
                    warnings.push(format!("sheet {name:?} could not be read: {e}"));
                    pages.push(ByteInterval::new(section_start, text.len()));
                    continue;
                }
            };

            let mut current_row: Option<u32> = None;
            let mut last_col: Option<u32> = None;
            let mut columns_clipped = false;

            loop {
                let cell = match reader.next_cell() {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => {
                        warnings.push(format!(
                            "sheet {name:?} stopped early, keeping the rows read before it: {e}"
                        ));
                        break;
                    }
                };
                cells_seen += 1;
                if cells_seen > MAX_CELLS {
                    warnings.push(format!("stopped after inspecting {MAX_CELLS} cells"));
                    truncated = true;
                    break;
                }

                let (row, col) = cell.get_position();
                let value = Data::from(cell.get_value().clone());
                if matches!(value, Data::Empty) {
                    continue;
                }
                if col >= MAX_COLUMNS {
                    columns_clipped = true;
                    continue;
                }

                if current_row != Some(row) {
                    if current_row.is_some() {
                        text.push('\n');
                    }
                    current_row = Some(row);
                    last_col = None;
                }
                // Pad skipped columns so a value stays under its header.
                let tabs = match last_col {
                    None => col,
                    Some(prev) => col.saturating_sub(prev).max(1),
                };
                for _ in 0..tabs {
                    text.push('\t');
                }
                push_cell(&mut text, &value);
                last_col = Some(col);

                if text.len() > MAX_TEXT_BYTES {
                    warnings.push(format!(
                        "text limit of {MAX_TEXT_BYTES} bytes reached in sheet {name:?}; \
                         the rest of the workbook is not included"
                    ));
                    truncated = true;
                    break;
                }
            }

            if columns_clipped {
                warnings.push(format!(
                    "sheet {name:?}: cells beyond column {MAX_COLUMNS} were ignored"
                ));
            }
            if current_row.is_some() {
                text.push('\n');
            }
            pages.push(ByteInterval::new(section_start, text.len()));
        }

        if truncated {
            // Cut mid-row: end on a whole line.
            if !text.ends_with('\n') {
                text.push('\n');
            }
            if let Some(last) = pages.last_mut() {
                *last = ByteInterval::new(last.start, text.len());
            }
        }

        if !text
            .lines()
            .any(|l| !l.starts_with("# Sheet: ") && !l.is_empty())
        {
            warnings.push("the workbook contained no cell values".to_string());
        }

        Ok(ExtractedText {
            text,
            format: DocumentFormat::Xlsx,
            pages,
            warnings,
        })
    }
}

/// Append one cell, flattening anything that would break the row/column grid.
///
/// A tab or newline inside a cell would shift every following column or split
/// the row, so each run of them becomes one space. `Display` is otherwise left
/// alone so dates and errors read the way the workbook shows them.
fn push_cell(out: &mut String, value: &Data) {
    let mut in_break = false;
    for c in value.to_string().chars() {
        if matches!(c, '\t' | '\n' | '\r') {
            if !in_break {
                out.push(' ');
            }
            in_break = true;
        } else {
            in_break = false;
            out.push(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn zip_of(entries: &[(&str, String)]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in entries {
            writer
                .start_file(*name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    /// One cell: `(reference, value)`; numbers as `n:` prefixed, else inline string.
    fn cell_xml(reference: &str, value: &str) -> String {
        match value.strip_prefix("n:") {
            Some(n) => format!(r#"<c r="{reference}"><v>{n}</v></c>"#),
            None => format!(
                r#"<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{value}</t></is></c>"#
            ),
        }
    }

    /// A row: its number and `(column letter, value)` cells.
    type Row<'a> = (u32, Vec<(&'a str, &'a str)>);

    /// A structurally real workbook: `sheets` is `(name, rows)` where a row is
    /// `(row number, [(column letter, value)])`.
    fn workbook(sheets: &[(&str, Vec<Row<'_>>)]) -> Vec<u8> {
        let mut entries: Vec<(String, String)> = vec![
            (
                "[Content_Types].xml".into(),
                r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/></Types>"#.into(),
            ),
            (
                "_rels/.rels".into(),
                r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#.into(),
            ),
        ];
        let mut sheet_tags = String::new();
        let mut rels = String::new();
        for (i, (name, rows)) in sheets.iter().enumerate() {
            let n = i + 1;
            sheet_tags.push_str(&format!(
                r#"<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>"#
            ));
            rels.push_str(&format!(
                r#"<Relationship Id="rId{n}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{n}.xml"/>"#
            ));
            let mut data = String::new();
            for (row_no, cells) in rows {
                data.push_str(&format!(r#"<row r="{row_no}">"#));
                for (col, value) in cells {
                    data.push_str(&cell_xml(&format!("{col}{row_no}"), value));
                }
                data.push_str("</row>");
            }
            entries.push((
                format!("xl/worksheets/sheet{n}.xml"),
                format!(
                    r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{data}</sheetData></worksheet>"#
                ),
            ));
        }
        entries.push((
            "xl/workbook.xml".into(),
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheet_tags}</sheets></workbook>"#
            ),
        ));
        entries.push((
            "xl/_rels/workbook.xml.rels".into(),
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>"#
            ),
        ));
        let refs: Vec<(&str, String)> = entries
            .iter()
            .map(|(n, c)| (n.as_str(), c.clone()))
            .collect();
        zip_of(&refs)
    }

    fn probe<'a>(bytes: &'a [u8], name: Option<&'a str>) -> FormatProbe<'a> {
        FormatProbe::new(bytes, name)
    }

    #[test]
    fn every_sheet_is_emitted_as_tab_separated_rows_under_a_heading() {
        let bytes = workbook(&[
            (
                "Budget",
                vec![
                    (1, vec![("A", "Item"), ("B", "Q1"), ("C", "Q2")]),
                    (2, vec![("A", "Licences"), ("B", "n:1200"), ("C", "n:1300")]),
                ],
            ),
            ("Notes", vec![(1, vec![("A", "Approved by finance")])]),
        ]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.format, DocumentFormat::Xlsx);
        assert_eq!(
            out.text,
            "# Sheet: Budget\nItem\tQ1\tQ2\nLicences\t1200\t1300\n# Sheet: Notes\nApproved by finance\n"
        );
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    }

    #[test]
    fn each_sheet_is_a_page_that_tiles_the_text() {
        let bytes = workbook(&[
            ("One", vec![(1, vec![("A", "a")])]),
            ("Two", vec![(1, vec![("A", "b")])]),
        ]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.pages.len(), 2);
        assert_eq!(out.pages[0].start, 0);
        assert_eq!(out.pages[0].end, out.pages[1].start);
        assert_eq!(out.pages[1].end, out.text.len());
        let second = out.text.find("b").unwrap();
        assert_eq!(out.page_of(second), Some(2));
    }

    #[test]
    fn skipped_columns_keep_values_under_their_header() {
        let bytes = workbook(&[(
            "S",
            vec![(1, vec![("A", "x"), ("D", "y")]), (3, vec![("B", "z")])],
        )]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "# Sheet: S\nx\t\t\ty\n\tz\n");
    }

    #[test]
    fn tabs_and_newlines_inside_a_cell_do_not_break_the_grid() {
        let bytes = workbook(&[(
            "S",
            vec![(1, vec![("A", "line1&#10;line2&#9;tabbed"), ("B", "next")])],
        )]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "# Sheet: S\nline1 line2 tabbed\tnext\n");
    }

    #[test]
    fn preserves_accented_and_cjk_text() {
        let bytes = workbook(&[("Été", vec![(1, vec![("A", "café"), ("B", "日本語")])])]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "# Sheet: Été\ncafé\t日本語\n");
    }

    #[test]
    fn a_stray_far_cell_does_not_allocate_a_dense_grid() {
        // Would be ~17 billion cells through `worksheet_range`.
        let bytes = workbook(&[(
            "S",
            vec![(1, vec![("A", "top")]), (1048576, vec![("A", "bottom")])],
        )]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert_eq!(out.text, "# Sheet: S\ntop\nbottom\n");
    }

    #[test]
    fn an_empty_workbook_is_reported_not_returned_as_silent_success() {
        let bytes = workbook(&[("Blank", vec![])]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert!(
            out.warnings.iter().any(|w| w.contains("no cell values")),
            "{:?}",
            out.warnings
        );
    }

    #[test]
    fn output_is_capped_and_says_so() {
        let big = "x".repeat(1000);
        let rows: Vec<Row<'_>> = (1..=5000).map(|r| (r, vec![("A", big.as_str())])).collect();
        let bytes = workbook(&[("Big", rows), ("After", vec![(1, vec![("A", "later")])])]);
        let out = XlsxExtractor.extract(&bytes).unwrap();
        assert!(out.text.len() <= MAX_TEXT_BYTES + 2048);
        assert!(out.text.ends_with('\n'));
        assert!(!out.text.contains("later"));
        assert!(out.warnings.iter().any(|w| w.contains("text limit")));
    }

    #[test]
    fn claims_a_real_workbook_by_content_whatever_it_is_called() {
        let bytes = workbook(&[("S", vec![(1, vec![("A", "a")])])]);
        assert!(XlsxExtractor.can_handle(&probe(&bytes, Some("data.bin"))));
        assert!(XlsxExtractor.can_handle(&probe(&bytes, None)));
    }

    #[test]
    fn refuses_other_zips_however_they_are_named() {
        let jar = zip_of(&[("META-INF/MANIFEST.MF", "Manifest-Version: 1.0".into())]);
        assert!(!XlsxExtractor.can_handle(&probe(&jar, Some("book.xlsx"))));
        assert!(!XlsxExtractor.can_handle(&probe(b"hello", Some("book.xlsx"))));
        assert!(!XlsxExtractor.can_handle(&probe(b"PK\x03\x04garbage", Some("book.xlsx"))));
    }

    #[test]
    fn a_corrupt_workbook_part_is_malformed_not_a_panic() {
        let bytes = zip_of(&[(WORKBOOK_PART, "not xml at all".into())]);
        let err = XlsxExtractor.extract(&bytes).unwrap_err();
        assert!(
            matches!(err, ExtractError::Malformed { format: "XLSX", .. }),
            "{err:?}"
        );
    }

    #[test]
    fn the_registry_routes_a_workbook_here_and_not_to_plain_text() {
        let reg = super::super::ExtractorRegistry::with_builtins();
        let bytes = workbook(&[("S", vec![(1, vec![("A", "routed")])])]);
        let out = reg.extract(&bytes, Some("anything.bin")).unwrap();
        assert_eq!(out.format, DocumentFormat::Xlsx);
        assert!(out.text.contains("routed"));
    }
}
