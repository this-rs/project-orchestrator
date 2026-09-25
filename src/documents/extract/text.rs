//! Formats whose bytes are already the text: Markdown, source code, JSON, CSV,
//! YAML, TOML, logs, plain prose.
//!
//! The catch-all of the registry, and therefore the one that must be **strict
//! about what it accepts**: it is probed last, so anything it claims is
//! something no specific extractor recognised. Accepting invalid UTF-8 here
//! would turn a corrupt upload into a document full of replacement characters
//! that chunks and aligns perfectly well and means nothing.

use super::{DocumentFormat, ExtractError, ExtractedText, FormatProbe, TextExtractor};

/// Magic bytes of binary containers this extractor must never claim, even when
/// the extension says otherwise.
const BINARY_MAGICS: &[&[u8]] = &[
    b"%PDF-",            // PDF
    b"PK\x03\x04",       // ZIP family: docx, xlsx, jar, odt
    b"\x7fELF",          // ELF executable
    b"\x89PNG",          // PNG
    b"GIF8",             // GIF
    b"\xff\xd8\xff",     // JPEG
    b"\x00\x61\x73\x6d", // WebAssembly
    b"\x1f\x8b",         // gzip
    b"BZh",              // bzip2
    b"\xfd7zXZ",         // xz
];

pub struct PlainTextExtractor;

impl TextExtractor for PlainTextExtractor {
    fn format(&self) -> DocumentFormat {
        DocumentFormat::PlainText
    }

    fn can_handle(&self, probe: &FormatProbe<'_>) -> bool {
        if BINARY_MAGICS.iter().any(|m| probe.starts_with(m)) {
            return false;
        }
        // UTF-16 BOMs: decodable in principle, but we do not decode them yet,
        // and claiming them would produce garbage rather than an honest error.
        if probe.starts_with(&[0xff, 0xfe]) || probe.starts_with(&[0xfe, 0xff]) {
            return false;
        }
        std::str::from_utf8(probe.bytes).is_ok()
    }

    fn extract(&self, bytes: &[u8]) -> Result<ExtractedText, ExtractError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| ExtractError::Malformed {
                format: "plain text",
                detail: format!("not valid UTF-8 at byte {}", e.valid_up_to()),
            })?
            .to_string();

        // A UTF-8 BOM is metadata, not content: left in place it becomes an
        // invisible first character that breaks exact alignment of any quote
        // starting at byte 0.
        let text = text
            .strip_prefix('\u{feff}')
            .map(str::to_string)
            .unwrap_or(text);

        Ok(ExtractedText {
            text,
            format: DocumentFormat::PlainText,
            pages: Vec::new(),
            warnings: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe<'a>(bytes: &'a [u8], name: Option<&'a str>) -> FormatProbe<'a> {
        FormatProbe::new(bytes, name)
    }

    #[test]
    fn accepts_utf8_text_and_preserves_it_byte_for_byte() {
        let src = "# Titre\n\nUne phrase accentuée, du 日本語, et un emoji 🎯.\n";
        let out = PlainTextExtractor.extract(src.as_bytes()).unwrap();
        assert_eq!(out.text, src);
        assert!(out.pages.is_empty(), "plain text has no pagination");
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn strips_a_utf8_bom_so_offset_zero_is_real_content() {
        let out = PlainTextExtractor
            .extract("\u{feff}hello".as_bytes())
            .unwrap();
        assert_eq!(out.text, "hello");
    }

    #[test]
    fn refuses_binary_containers_even_when_the_extension_lies() {
        // Probed last, so anything it claims was rejected by every specific
        // extractor. Claiming a PDF here would produce silent garbage.
        for magic in BINARY_MAGICS {
            let mut bytes = magic.to_vec();
            bytes.extend_from_slice(b"trailing content");
            assert!(
                !PlainTextExtractor.can_handle(&probe(&bytes, Some("notes.txt"))),
                "must refuse magic {magic:?} despite a .txt extension"
            );
        }
    }

    #[test]
    fn refuses_invalid_utf8_rather_than_mangling_it() {
        let bytes = [0x48, 0x65, 0xff, 0xfe, 0x6c];
        assert!(!PlainTextExtractor.can_handle(&probe(&bytes, None)));
        assert!(matches!(
            PlainTextExtractor.extract(&bytes),
            Err(ExtractError::Malformed { .. })
        ));
    }

    #[test]
    fn refuses_utf16_rather_than_pretending() {
        // Decodable in principle, not decoded yet — an honest error beats
        // silent mojibake.
        assert!(!PlainTextExtractor.can_handle(&probe(&[0xff, 0xfe, 0x68, 0x00], Some("a.txt"))));
        assert!(!PlainTextExtractor.can_handle(&probe(&[0xfe, 0xff, 0x00, 0x68], Some("a.txt"))));
    }

    #[test]
    fn accepts_the_whole_text_family_regardless_of_extension() {
        for (name, body) in [
            ("a.md", "# h\n\ntext"),
            ("a.txt", "plain"),
            ("a.json", r#"{"k": 1}"#),
            ("a.csv", "a,b\n1,2"),
            ("a.yaml", "k: v"),
            ("a.toml", "[t]\nk = 1"),
            ("a.rs", "fn main() {}"),
            ("no_extension_at_all", "still text"),
        ] {
            assert!(
                PlainTextExtractor.can_handle(&probe(body.as_bytes(), Some(name))),
                "should accept {name}"
            );
        }
    }
}
