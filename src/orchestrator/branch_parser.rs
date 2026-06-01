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
// High-level lookup: branch → Task
// ============================================================================

use crate::neo4j::models::TaskNode;
use crate::neo4j::traits::GraphStore;
use std::sync::Arc;
use uuid::Uuid;

/// A task located via its external_id (parsed from a branch).
#[derive(Debug, Clone)]
pub struct ResolvedTask {
    pub external_id: String,
    /// Present when a task matching `external_id` was found in the project's plans.
    /// `None` means the id was extracted from the branch but no Task carries it
    /// (e.g. branch was created proactively, plan/task not yet authored).
    pub task: Option<TaskNode>,
}

/// Boundary chars that may follow the external_id inside a task title.
/// Keeps `T210` distinct from `T2100` and `T210.5`.
const ID_BOUNDARY_CHARS: &[char] = &[' ', '\t', ':', '—', '-', '|', '/', '(', '[', ',', '.'];

/// Resolve a Task by parsing a git branch name and matching the extracted
/// external_id against task titles within a project.
///
/// **Lookup strategy** (no dedicated `external_id` column yet — operates on
/// the existing graph): scan in-progress + recent plans for the project and
/// pick the first task whose title starts with the parsed id followed by a
/// non-id boundary character. This is O(plans × tasks) but bounded in
/// practice and short-circuits on the first hit.
///
/// Returns `None` if:
/// - The branch doesn't contain a recognizable id.
///
/// Returns `Some` with `task: None` when the id was parsed but no matching
/// task was found — useful for downstream UIs to show "Working on T210
/// (task not registered)".
pub async fn resolve_task_from_branch(
    graph: &Arc<dyn GraphStore>,
    project_id: Uuid,
    branch: &str,
) -> Option<ResolvedTask> {
    let external_id = parse_branch_to_external_id(branch)?;
    let task = find_task_by_external_id(graph, project_id, &external_id).await;
    Some(ResolvedTask { external_id, task })
}

/// Check whether a title carries the given external_id as a prefix-token.
///
/// Visible for testing.
pub fn title_matches_external_id(title: &str, external_id: &str) -> bool {
    // Match case-insensitively but anchor at the very start.
    let lower_title = title.trim_start().to_ascii_lowercase();
    let lower_id = external_id.to_ascii_lowercase();
    if !lower_title.starts_with(&lower_id) {
        return false;
    }
    // Boundary check: the char immediately after the id must be a separator,
    // or the title must end exactly at the id.
    match lower_title[lower_id.len()..].chars().next() {
        None => true,
        Some(c) if ID_BOUNDARY_CHARS.contains(&c) => {
            // Reject `T210.5` when looking for `T210` — `.` is a boundary in
            // ID_BOUNDARY_CHARS but introduces a sub-id continuation. The id
            // continues only if a digit follows the dot.
            if c == '.' {
                let after = lower_title[lower_id.len() + 1..].chars().next();
                !matches!(after, Some(d) if d.is_ascii_digit())
            } else {
                true
            }
        }
        Some(_) => false,
    }
}

async fn find_task_by_external_id(
    graph: &Arc<dyn GraphStore>,
    project_id: Uuid,
    external_id: &str,
) -> Option<TaskNode> {
    // Active first, then everything else — recent work usually matches the branch.
    let active_filter = Some(vec![
        "in_progress".to_string(),
        "approved".to_string(),
        "draft".to_string(),
    ]);
    let (active_plans, _) = graph
        .list_plans_for_project(project_id, active_filter, 50, 0)
        .await
        .ok()?;

    for plan in &active_plans {
        if let Ok(tasks) = graph.get_plan_tasks(plan.id).await {
            for t in tasks {
                if let Some(ref title) = t.title {
                    if title_matches_external_id(title, external_id) {
                        return Some(t);
                    }
                }
            }
        }
    }

    // Fallback sweep: any plan status (completed plans often carry the task too)
    let (all_plans, _) = graph
        .list_plans_for_project(project_id, None, 100, 0)
        .await
        .ok()?;
    for plan in &all_plans {
        // Skip plans we already scanned above
        if active_plans.iter().any(|p| p.id == plan.id) {
            continue;
        }
        if let Ok(tasks) = graph.get_plan_tasks(plan.id).await {
            for t in tasks {
                if let Some(ref title) = t.title {
                    if title_matches_external_id(title, external_id) {
                        return Some(t);
                    }
                }
            }
        }
    }

    None
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
        assert_eq!(parse_branch_to_external_id("t42"), Some("T42".to_string()));
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

    // ====================================================================
    // title_matches_external_id
    // ====================================================================

    #[test]
    fn title_matches_with_em_dash() {
        assert!(title_matches_external_id("T210 — Mamba2 plan A", "T210"));
    }

    #[test]
    fn title_matches_with_colon() {
        assert!(title_matches_external_id(
            "T246.7: Lookahead decoding",
            "T246.7"
        ));
    }

    #[test]
    fn title_matches_with_space() {
        assert!(title_matches_external_id("T300 fix the regression", "T300"));
    }

    #[test]
    fn title_matches_exact_equals_id() {
        assert!(title_matches_external_id("T42", "T42"));
    }

    #[test]
    fn title_matches_is_case_insensitive() {
        assert!(title_matches_external_id("t210 — Mamba2", "T210"));
        assert!(title_matches_external_id("T210 — Mamba2", "t210"));
    }

    #[test]
    fn title_matches_rejects_longer_id() {
        // Looking for T210 should NOT match T2100
        assert!(!title_matches_external_id("T2100 — Other task", "T210"));
    }

    #[test]
    fn title_matches_rejects_sub_id() {
        // Looking for T246 should NOT match T246.5 (sub-id continuation)
        assert!(!title_matches_external_id("T246.5 — Decode push", "T246"));
    }

    #[test]
    fn title_matches_rejects_mid_word() {
        assert!(!title_matches_external_id("Fix HTTP T210 backport", "T210"));
    }
}
