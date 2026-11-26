//! Deterministic parsers for the three tool-output formats this first brick
//! understands: `cargo test`, `cargo check`/`cargo build` diagnostics, and
//! `git` (status / diff / log, in their short forms).
//!
//! Every parser is a pure function over text — no I/O, no LLM, no guessing. A
//! format that does not match is simply not parsed, and the caller falls back
//! to keeping the artifact in full under [`super::Kind::Unknown`].

use std::collections::BTreeMap;

use super::{split_lines, Kind};

/// How many summary lines a digest keeps before it declares itself truncated.
/// The artifact always holds the whole output; this bound applies only to the
/// digest's own summary.
pub(crate) const MAX_SUMMARY_LINES: usize = 40;

/// The result of parsing one known format.
pub(crate) struct Parsed {
    pub kind: Kind,
    pub summary: Vec<String>,
    pub counts: BTreeMap<String, usize>,
    pub truncated: bool,
}

/// Try each known format in order. `None` means "not one of the known formats",
/// so the caller keeps the artifact in full under `Kind::Unknown`.
pub(crate) fn parse(text: &str) -> Option<Parsed> {
    if looks_like_cargo_test(text) {
        return Some(parse_cargo_test(text));
    }
    if looks_like_cargo_diagnostics(text) {
        return Some(parse_cargo_diagnostics(text));
    }
    if looks_like_git(text) {
        return Some(parse_git(text));
    }
    None
}

/// Cap a summary to [`MAX_SUMMARY_LINES`], reporting whether lines were dropped.
fn cap(summary: Vec<String>) -> (Vec<String>, bool) {
    if summary.len() <= MAX_SUMMARY_LINES {
        (summary, false)
    } else {
        (summary.into_iter().take(MAX_SUMMARY_LINES).collect(), true)
    }
}

// ---------------------------------------------------------------------------
// cargo test
// ---------------------------------------------------------------------------

fn looks_like_cargo_test(text: &str) -> bool {
    split_lines(text).iter().any(|l| {
        let t = l.trim_start();
        t.starts_with("test result:")
            || (t.starts_with("running ") && t.ends_with(" tests"))
            || t == "Doc-tests"
    })
}

/// Parse one `test result: ok. 42 passed; 0 failed; ...` line into the
/// `label -> count` pairs it carries. A label cargo does not emit on this line
/// is simply absent from the returned map (the caller seeds the canonical
/// keys with 0 first).
fn parse_test_result_line(line: &str) -> BTreeMap<String, usize> {
    let rest = line.trim_start_matches("test result:").trim();
    // rest = "ok. 42 passed; 0 failed; 0 ignored; ..." or "FAILED. 1 passed; ..."
    let (_status, body) = rest.split_once(' ').unwrap_or((rest, ""));
    let mut nums = BTreeMap::new();
    for seg in body.split(';') {
        let seg = seg.trim();
        if let Some((n, label)) = seg.split_once(' ') {
            if let Ok(n) = n.parse::<usize>() {
                let key = match label.trim() {
                    "passed" => "passed",
                    "failed" => "failed",
                    "ignored" => "ignored",
                    "measured" => "measured",
                    "filtered out" => "filtered",
                    _ => continue,
                };
                nums.insert(key.to_string(), n);
            }
        }
    }
    nums
}

fn parse_cargo_test(text: &str) -> Parsed {
    let mut summary = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    // Canonical keys, always present even when the run had none of them.
    for key in ["passed", "failed", "ignored", "measured", "filtered"] {
        counts.insert(key.to_string(), 0);
    }
    let mut binaries = 0usize;
    for raw in split_lines(text) {
        let line = raw.trim();
        if line.starts_with("test result:") {
            binaries += 1;
            for (k, v) in parse_test_result_line(line) {
                *counts.entry(k).or_insert(0) += v;
            }
            summary.push(line.to_string());
        } else if line.starts_with("test ") && line.ends_with(" FAILED") {
            // A failing test's own line: `test tests::foo ... FAILED`.
            summary.push(line.to_string());
        }
    }
    counts.insert("binaries".to_string(), binaries);
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::CargoTest,
        summary,
        counts,
        truncated,
    }
}

// ---------------------------------------------------------------------------
// cargo check / cargo build diagnostics
// ---------------------------------------------------------------------------

fn looks_like_cargo_diagnostics(text: &str) -> bool {
    split_lines(text).iter().any(|l| {
        let t = l.trim_start();
        t.starts_with("error[")
            || t.starts_with("error:")
            || t.starts_with("warning:")
            || t.starts_with("Compiling ")
            || t.starts_with("Checking ")
            || t.starts_with("Finished ")
    })
}

fn parse_cargo_diagnostics(text: &str) -> Parsed {
    let lines = split_lines(text);
    let mut summary = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    counts.insert("errors".to_string(), 0);
    counts.insert("warnings".to_string(), 0);
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        let is_error = t.starts_with("error[") || t.starts_with("error:");
        let is_warning = t.starts_with("warning:");
        if is_error || is_warning {
            // Build-level summaries (`error: could not compile …`,
            // `warning: … generated N warning`) are not diagnostics: they stay
            // in the summary but do not inflate the `errors` / `warnings` counts.
            let is_summary = t.starts_with("error: could not compile")
                || t.starts_with("error: aborting due to")
                || (is_warning && t.contains("generated "));
            if !is_summary {
                let key = if is_error { "errors" } else { "warnings" };
                *counts.entry(key.to_string()).or_insert(0) += 1;
            }
            let mut entry = t.to_string();
            // Attach the `--> file:line:col` location that follows the header.
            if let Some(next) = lines.get(i + 1) {
                let nt = next.trim_start();
                if nt.starts_with("-->") {
                    entry.push(' ');
                    entry.push_str(nt);
                    i += 1;
                }
            }
            summary.push(entry);
        }
        i += 1;
    }
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::CargoCheck,
        summary,
        counts,
        truncated,
    }
}

// ---------------------------------------------------------------------------
// git (status / diff / log, short forms)
// ---------------------------------------------------------------------------

fn looks_like_git(text: &str) -> bool {
    let lines = split_lines(text);
    looks_like_git_diff(&lines)
        || looks_like_git_status_long(&lines)
        || looks_like_git_status_short(&lines)
        || looks_like_git_log(text)
}

fn parse_git(text: &str) -> Parsed {
    let lines = split_lines(text);
    if looks_like_git_diff(&lines) {
        return parse_git_diff(text);
    }
    if looks_like_git_status_long(&lines) {
        return parse_git_status_long(text);
    }
    if looks_like_git_status_short(&lines) {
        return parse_git_status_short(text);
    }
    parse_git_log(text)
}

/// A `git diff` / `git diff --stat` output: `diff --git` headers, a diffstat
/// `… files changed, … insertions(+), … deletions(-)` line, or the `--- a/` /
/// `+++ b/` markers of a hunk.
fn looks_like_git_diff(lines: &[&str]) -> bool {
    lines.iter().any(|l| {
        let t = l.trim_start();
        t.starts_with("diff --git ")
            || t.contains("files changed")
            || t.starts_with("--- a/")
            || t.starts_with("+++ b/")
    })
}

/// `N` from a `N word` segment of a diffstat line (`2 files changed`,
/// `3 insertions(+)`, `1 deletion(-)`).
fn extract_stat(line: &str, word: &str) -> usize {
    for part in line.split(',') {
        let part = part.trim();
        if part.contains(word) {
            if let Some((n, _)) = part.split_once(' ') {
                if let Ok(n) = n.parse::<usize>() {
                    return n;
                }
            }
        }
    }
    0
}

fn parse_git_diff(text: &str) -> Parsed {
    let mut summary = Vec::new();
    let mut diff_headers = 0usize;
    let mut stat_files: Option<usize> = None;
    let mut insertions = 0usize;
    let mut deletions = 0usize;
    for raw in split_lines(text) {
        let line = raw.trim();
        if line.starts_with("diff --git ") {
            diff_headers += 1;
            summary.push(line.to_string());
        } else if line.contains("files changed") {
            // `2 files changed, 3 insertions(+), 1 deletion(-)`
            summary.push(line.to_string());
            stat_files = Some(extract_stat(line, "files changed"));
            insertions = extract_stat(line, "insertion");
            deletions = extract_stat(line, "deletion");
        }
    }
    let mut counts = BTreeMap::new();
    // `git diff --stat` gives the count in its summary line; a plain `git diff`
    // does not, so fall back to the number of `diff --git` headers.
    counts.insert("files".to_string(), stat_files.unwrap_or(diff_headers));
    counts.insert("insertions".to_string(), insertions);
    counts.insert("deletions".to_string(), deletions);
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::Git,
        summary,
        counts,
        truncated,
    }
}

fn looks_like_git_status_long(lines: &[&str]) -> bool {
    lines.iter().any(|l| {
        let t = l.trim_start();
        t.starts_with("On branch ")
            || t.starts_with("Changes not staged")
            || t.starts_with("Changes to be committed")
            || t.starts_with("Untracked files")
    })
}

fn parse_git_status_long(text: &str) -> Parsed {
    let mut summary = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for raw in split_lines(text) {
        let l = raw.trim_start();
        if l.is_empty() || l.starts_with("(use ") {
            continue;
        }
        let key = if l.starts_with("modified:") {
            Some("modified")
        } else if l.starts_with("deleted:") {
            Some("deleted")
        } else if l.starts_with("new file:") {
            Some("added")
        } else if l.starts_with("renamed:") {
            Some("renamed")
        } else {
            None
        };
        if let Some(key) = key {
            *counts.entry(key.to_string()).or_insert(0) += 1;
            summary.push(l.to_string());
            continue;
        }
        if l.starts_with("On branch ")
            || l.starts_with("Your branch")
            || l.starts_with("Changes not staged")
            || l.starts_with("Changes to be committed")
            || l.starts_with("Untracked files")
            || l.starts_with("no changes added")
            || l.starts_with("nothing to commit")
            || l.starts_with("nothing added")
        {
            summary.push(l.to_string());
            continue;
        }
        // Anything left is an untracked file name (bare, under "Untracked files:").
        *counts.entry("untracked".to_string()).or_insert(0) += 1;
        summary.push(l.to_string());
    }
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::Git,
        summary,
        counts,
        truncated,
    }
}

fn is_porcelain_char(c: u8) -> bool {
    matches!(
        c,
        b'M' | b'A' | b'D' | b'R' | b'C' | b'U' | b'?' | b'!' | b' '
    )
}

/// A porcelain v1 `git status --short` line: two status chars, a space, a path.
fn is_porcelain_status_line(l: &str) -> bool {
    let b = l.as_bytes();
    b.len() >= 4 && is_porcelain_char(b[0]) && is_porcelain_char(b[1]) && b[2] == b' '
}

fn looks_like_git_status_short(lines: &[&str]) -> bool {
    let non_empty: Vec<&str> = lines
        .iter()
        .map(|l| l.trim_end())
        .filter(|l| !l.trim().is_empty())
        .collect();
    !non_empty.is_empty() && non_empty.iter().all(|l| is_porcelain_status_line(l))
}

fn parse_git_status_short(text: &str) -> Parsed {
    let mut summary = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for raw in split_lines(text) {
        // Porcelain v1 lines are `XY PATH`: the leading space (an unchanged
        // index column) is significant and must not be trimmed.
        let l = raw.trim_end();
        if l.is_empty() || !is_porcelain_status_line(l) {
            continue;
        }
        let x = l.as_bytes()[0] as char;
        let y = l.as_bytes()[1] as char;
        let key = match (x, y) {
            ('?', '?') => "untracked",
            (_, _) if x == 'D' || y == 'D' => "deleted",
            (_, _) if x == 'R' || y == 'R' => "renamed",
            (_, _) if x == 'A' || y == 'A' => "added",
            _ => "modified",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
        summary.push(l.to_string());
    }
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::Git,
        summary,
        counts,
        truncated,
    }
}

/// A `git log --oneline` line: 7..=40 hex chars, then a space, then the subject.
fn is_oneline_commit(line: &str) -> bool {
    line.len() > 8
        && line.as_bytes()[..7].iter().all(|c| c.is_ascii_hexdigit())
        && line.as_bytes()[7] == b' '
}

fn looks_like_git_log(text: &str) -> bool {
    let lines: Vec<&str> = split_lines(text)
        .into_iter()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    !lines.is_empty() && lines.iter().all(|l| is_oneline_commit(l))
}

fn parse_git_log(text: &str) -> Parsed {
    let mut summary = Vec::new();
    let mut commits = 0usize;
    for raw in split_lines(text) {
        let l = raw.trim();
        if l.is_empty() {
            continue;
        }
        commits += 1;
        summary.push(l.to_string());
    }
    let mut counts = BTreeMap::new();
    counts.insert("commits".to_string(), commits);
    let (summary, truncated) = cap(summary);
    Parsed {
        kind: Kind::Git,
        summary,
        counts,
        truncated,
    }
}
