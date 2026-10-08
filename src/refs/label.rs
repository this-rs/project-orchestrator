//! Labels are written by the server, from what the store holds, and never read
//! from a client. Three kinds have no title of their own (a note has only its
//! content, a decision only its description, a task may have no title): the
//! label is derived here, in one place, with one length limit.

/// Longest label, in characters, ellipsis included.
pub const MAX_LABEL_CHARS: usize = 80;

const ELLIPSIS: char = '…';

/// Cut `s` to [`MAX_LABEL_CHARS`] characters; a cut ends with `…`.
pub fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_LABEL_CHARS {
        return s.to_string();
    }
    let mut out: String = s.chars().take(MAX_LABEL_CHARS - 1).collect();
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    out.push(ELLIPSIS);
    out
}

/// The first non-empty line of `text`, a markdown heading marker (`# `) removed,
/// truncated. `None` when the text has no printable line.
pub fn first_line(text: &str) -> Option<String> {
    text.lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .map(truncate)
}

/// A neutral label for an entity whose text is empty: `kind` and the first
/// block of the id. Never empty, never invented prose.
pub fn fallback(kind: &str, id: &uuid::Uuid) -> String {
    let id = id.to_string();
    format!("{kind} {}", &id[..8])
}

/// The label of an entity from its candidate texts, first usable one wins.
pub fn derive(kind: &str, id: &uuid::Uuid, texts: &[&str]) -> String {
    texts
        .iter()
        .find_map(|t| first_line(t))
        .unwrap_or_else(|| fallback(kind, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn short_text_is_untouched_and_long_text_is_cut_to_the_limit() {
        assert_eq!(truncate("court"), "court");
        let exact = "a".repeat(MAX_LABEL_CHARS);
        assert_eq!(truncate(&exact), exact);
        let cut = truncate(&"a".repeat(MAX_LABEL_CHARS + 1));
        assert_eq!(cut.chars().count(), MAX_LABEL_CHARS);
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn a_cut_never_leaves_a_space_before_the_ellipsis_and_counts_characters() {
        let s = format!("{} {}", "é".repeat(MAX_LABEL_CHARS - 2), "z".repeat(10));
        let cut = truncate(&s);
        assert!(cut.chars().count() <= MAX_LABEL_CHARS);
        assert!(!cut.ends_with(" …"));
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn the_first_line_skips_blank_lines_and_heading_markers() {
        assert_eq!(first_line("\n\n  ## Titre  \nsuite"), Some("Titre".into()));
        assert_eq!(first_line("ligne\nautre"), Some("ligne".into()));
        assert_eq!(first_line("###\nvrai"), Some("vrai".into()));
        assert_eq!(first_line(""), None);
        assert_eq!(first_line(" \n\t\n"), None);
        assert_eq!(first_line("#"), None);
    }

    #[test]
    fn a_long_first_line_is_truncated() {
        let l = first_line(&"x".repeat(500)).unwrap();
        assert_eq!(l.chars().count(), MAX_LABEL_CHARS);
    }

    #[test]
    fn derive_takes_the_first_usable_text_then_falls_back_to_the_id() {
        let id = Uuid::parse_str("3adeffc9-c8b0-4e2f-a674-55bfcb293433").unwrap();
        assert_eq!(derive("task", &id, &["  ", "Desc\nx"]), "Desc");
        assert_eq!(derive("task", &id, &["Titre", "Desc"]), "Titre");
        assert_eq!(derive("task", &id, &["", " "]), "task 3adeffc9");
        assert_eq!(derive("note", &id, &[]), "note 3adeffc9");
    }
}
