//! One relevance score for every kind of the `#` / `@` picker.
//!
//! Deterministic and shared, so that a plan, a note and a file are ordered by
//! the same rule: how well the typed text matches the TITLE of the entity,
//! then its body; ties go to the most recent. Case and accents do not count
//! (`ete` finds `Été`). A candidate that does not match the text at all scores
//! `None` and is not suggested.
//!
//! | tier | the text is...                                   | score |
//! |------|--------------------------------------------------|-------|
//! | 1    | the whole title                                  | 100   |
//! | 2    | the start of the title                           | 80    |
//! | 3    | the start of a word of the title                 | 60    |
//! | 4    | anywhere in the title                            | 40    |
//! | 5    | every word of the text is somewhere in the title | 30    |
//! | 6    | anywhere in the body                             | 20    |
//! | 7    | every word of the text is in the title or body   | 10    |
//!
//! An empty text scores `0` for everything (recency decides).

use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

/// Lower case, accents removed, surrounding blanks and runs of blanks collapsed.
pub fn fold(s: &str) -> String {
    let plain: String = s
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect();
    plain.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Is `needle` at the start of a word of `hay` (both folded)?
fn word_start(hay: &str, needle: &str) -> bool {
    hay.match_indices(needle).any(|(i, _)| {
        i == 0
            || hay[..i]
                .chars()
                .next_back()
                .is_some_and(|c| !c.is_alphanumeric())
    })
}

/// The relevance of an entity (`title`, `body`) for the text `needle`.
pub fn score(needle: &str, title: &str, body: &str) -> Option<u32> {
    let needle = fold(needle);
    if needle.is_empty() {
        return Some(0);
    }
    let title = fold(title);
    let body = fold(body);
    let words: Vec<&str> = needle.split(' ').collect();
    let all_in = |hay: &str| words.iter().all(|w| hay.contains(w));
    if title == needle {
        Some(100)
    } else if title.starts_with(&needle) {
        Some(80)
    } else if word_start(&title, &needle) {
        Some(60)
    } else if title.contains(&needle) {
        Some(40)
    } else if all_in(&title) {
        Some(30)
    } else if body.contains(&needle) {
        Some(20)
    } else if words.iter().all(|w| title.contains(w) || body.contains(w)) {
        Some(10)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_ignores_case_accents_and_blanks() {
        assert_eq!(fold("  Été   ÉCOLE\t"), "ete ecole");
        assert_eq!(fold("Çà et là"), "ca et la");
    }

    #[test]
    fn the_tiers_are_ordered_exact_prefix_word_substring_body() {
        let body = "";
        let exact = score("fond", "Fond", body).unwrap();
        let prefix = score("fond", "Fondations refs", body).unwrap();
        let word = score("fond", "Les fondations", body).unwrap();
        let inside = score("fond", "Profondeur", body).unwrap();
        let in_body = score("fond", "Autre chose", "au milieu fond du texte").unwrap();
        assert!(exact > prefix, "{exact} {prefix}");
        assert!(prefix > word, "{prefix} {word}");
        assert!(word > inside, "{word} {inside}");
        assert!(inside > in_body, "{inside} {in_body}");
        assert!(in_body > 0);
    }

    #[test]
    fn case_and_accents_are_normalized() {
        assert_eq!(score("ETE", "Été 2026", ""), score("été", "ete 2026", ""));
        assert!(score("ete", "Été", "").unwrap() >= 100);
    }

    #[test]
    fn a_word_start_needs_a_boundary() {
        let w = score("ref", "plan de refs", "").unwrap();
        let m = score("ef", "plan de refs", "").unwrap();
        assert!(w > m, "word start {w} beats the middle of a word {m}");
        assert_eq!(
            score("de ref", "plan de refs", ""),
            score("de ref", "plan de refs", ""),
        );
    }

    #[test]
    fn no_match_is_none_and_an_empty_text_matches_everything_equally() {
        assert_eq!(score("zzz", "Plan", "corps"), None);
        assert_eq!(score("", "Plan", "corps"), Some(0));
        assert_eq!(score("   ", "Plan", ""), Some(0));
    }

    #[test]
    fn a_text_of_several_words_may_be_spread_over_the_title_then_the_body() {
        let in_title = score("billing plan", "Plan de billing", "").unwrap();
        let spread = score("billing plan", "Plan", "le billing").unwrap();
        assert!(in_title > spread && spread > 0, "{in_title} {spread}");
        assert_eq!(score("billing zzz", "Plan de billing", ""), None);
    }

    #[test]
    fn a_title_phrase_beats_the_same_words_spread() {
        let phrase = score("plan alpha", "Plan alpha refs", "").unwrap();
        let apart = score("plan alpha", "Plan refs alpha", "").unwrap();
        assert!(phrase > apart);
    }
}
