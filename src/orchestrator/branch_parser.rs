//! Branch → Task external_id parser.
//!
//! Extracts a task identifier (e.g. `T210`, `T246.7`) from a conventional
//! branch name like `feat/t210-mamba2-plan-a` or `perf/T246.7-lookahead`.
//!
//! Used by `project.resume_context` to auto-resolve the task an agent is
//! currently working on, based solely on the active git branch.
//!
//! ## Recognized patterns
//!
//! The parser is intentionally tolerant. It scans the branch name for the
//! first token shaped like `T<digits>(.<digits>)*` (case-insensitive) and
//! returns it normalized in UPPERCASE.
//!
//! | Input branch                              | Returns       |
//! |-------------------------------------------|---------------|
//! | `feat/t210-mamba2-plan-a`                 | `Some("T210")` |
//! | `perf/T246.7-lookahead-decoding`          | `Some("T246.7")` |
//! | `fix/T-300-foo`                           | `Some("T300")` |
//! | `refactor/t246.12.b-something`            | `Some("T246.12.B")` |
//! | `chore/cleanup-T9000`                     | `Some("T9000")` |
//! | `release/v0.2.0`                          | `None` |
//! | `main`                                    | `None` |
//! | `feat/some-feature-without-id`            | `None` |

/// Parse a git branch name and return the embedded task `external_id` if any.
///
/// Returns `None` when no recognizable identifier is found.
///
/// The returned id is normalized:
/// - Stripped of an optional `-`/`_` separator between `T` and the digits
/// - Trailing identifier segments (e.g. `.B`) are uppercased so lookups can
///   use exact-match indexing without ambiguity.
pub fn parse_branch_to_external_id(branch: &str) -> Option<String> {
    let bytes = branch.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find a 'T' or 't'
        if bytes[i] != b'T' && bytes[i] != b't' {
            i += 1;
            continue;
        }
        // Reject if the previous char is alphanumeric (avoid matching mid-word "test", "TIMEOUT", etc.)
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() {
                i += 1;
                continue;
            }
        }
        // Try to parse from here
        if let Some((parsed, advance)) = try_parse_at(&bytes[i..]) {
            // Reject if the next char is alphanumeric continuation that we didn't consume
            // (try_parse_at already stopped on a non-id char; nothing more to do)
            let _ = advance;
            return Some(parsed);
        }
        i += 1;
    }
    None
}

/// Attempt to parse a `T<digits>(.<digits>)*` token starting at `bytes[0]`.
///
/// Optionally tolerates a single `-` or `_` between `T` and the first digit
/// (so `T-300` and `T_300` both work).
///
/// Returns `(normalized_id, bytes_consumed)` on success.
fn try_parse_at(bytes: &[u8]) -> Option<(String, usize)> {
    // bytes[0] is 'T' or 't' — caller guaranteed.
    let mut idx = 1;

    // Optional separator
    if idx < bytes.len() && (bytes[idx] == b'-' || bytes[idx] == b'_') {
        idx += 1;
    }

    // Need at least one digit
    let digits_start = idx;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    if idx == digits_start {
        return None;
    }

    // Optional `.<digit-or-letter>+` segments
    loop {
        if idx >= bytes.len() || bytes[idx] != b'.' {
            break;
        }
        let after_dot = idx + 1;
        let mut end = after_dot;
        while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
            end += 1;
        }
        // Require at least one char after the dot, else stop without consuming the dot
        if end == after_dot {
            break;
        }
        idx = end;
    }

    // Build normalized id: uppercase 'T' + digits + segments
    let mut out = String::with_capacity(idx - digits_start + 1);
    out.push('T');
    for &b in &bytes[digits_start..idx] {
        out.push(b.to_ascii_uppercase() as char);
    }
    Some((out, idx))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feat_prefix_lowercase() {
        assert_eq!(
            parse_branch_to_external_id("feat/t210-mamba2-plan-a"),
            Some("T210".to_string())
        );
    }

    #[test]
    fn parses_perf_prefix_dotted_id() {
        assert_eq!(
            parse_branch_to_external_id("perf/T246.7-lookahead-decoding"),
            Some("T246.7".to_string())
        );
    }

    #[test]
    fn parses_fix_prefix_with_dash_separator() {
        assert_eq!(
            parse_branch_to_external_id("fix/T-300-foo"),
            Some("T300".to_string())
        );
    }

    #[test]
    fn parses_with_underscore_separator() {
        assert_eq!(
            parse_branch_to_external_id("fix/t_300_foo"),
            Some("T300".to_string())
        );
    }

    #[test]
    fn parses_multi_segment_dotted_id() {
        assert_eq!(
            parse_branch_to_external_id("refactor/t246.12.b-something"),
            Some("T246.12.B".to_string())
        );
    }

    #[test]
    fn parses_chore_with_trailing_id() {
        assert_eq!(
            parse_branch_to_external_id("chore/cleanup-T9000"),
            Some("T9000".to_string())
        );
    }

    #[test]
    fn parses_bare_id_branch() {
        assert_eq!(
            parse_branch_to_external_id("t42"),
            Some("T42".to_string())
        );
    }

    #[test]
    fn rejects_release_tag() {
        assert!(parse_branch_to_external_id("release/v0.2.0").is_none());
    }

    #[test]
    fn rejects_main() {
        assert!(parse_branch_to_external_id("main").is_none());
    }

    #[test]
    fn rejects_no_digits_after_t() {
        assert!(parse_branch_to_external_id("feat/typo-fix").is_none());
        assert!(parse_branch_to_external_id("test-suite").is_none());
    }

    #[test]
    fn rejects_t_inside_word() {
        // `iteration`, `attribute`, `timeout` etc. contain T<digits>-like spans
        // but only when 'T' follows alphanumerics — these must NOT match.
        assert!(parse_branch_to_external_id("iter4tion").is_none());
        assert!(parse_branch_to_external_id("attribute7-fix").is_none());
        assert!(parse_branch_to_external_id("HTTP2-upgrade").is_none());
    }

    #[test]
    fn picks_first_id_when_multiple() {
        // First well-formed id wins (deterministic).
        assert_eq!(
            parse_branch_to_external_id("feat/t210-then-T999-too"),
            Some("T210".to_string())
        );
    }

    #[test]
    fn handles_empty_branch() {
        assert_eq!(parse_branch_to_external_id(""), None);
    }
}
