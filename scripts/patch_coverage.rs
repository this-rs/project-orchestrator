//! Patch coverage of a git diff, from an lcov report — the same measure as
//! Codecov's `patch` status, but local, and it NAMES the uncovered lines.
//!
//! Why this exists: the repository gates `codecov/patch` at 95 % with a 0 %
//! threshold (`codecov.yml`, the single place to ratchet). Codecov runs only
//! after the fact, reports one percentage and no line numbers, and reads the
//! config of the DEFAULT branch — so a PR can go red on a bar nobody can see
//! locally. This tool answers the two questions a red patch status never
//! answers: WHICH lines of my diff are uncovered, and do I reach the bar?
//!
//! It is a LOWER BOUND when fed a partial lcov (`cargo llvm-cov --lib` only):
//! the CI merges unit + integration + api suites, so anything this tool calls
//! covered is covered there too. Measured below the bar locally, it is worth
//! running the full suite before concluding; measured above, the CI will pass.
//!
//! std only, no dependency. Build and run:
//!
//! ```text
//! rustc -O scripts/patch_coverage.rs -o /tmp/patch_coverage
//! /tmp/patch_coverage lcov.info --base origin/main --min 95
//! ```
//!
//! Self-test: `rustc --test scripts/patch_coverage.rs -o /tmp/pc_test && /tmp/pc_test`.
//!
//! Reading: a line of the diff counts only if the report INSTRUMENTS it (an
//! `DA:` record). Comments, blank lines and declarations are not counted, by
//! either this tool or Codecov. A line in a file the report does not mention is
//! counted as `unknown`, never as covered: a file no test binary compiles must
//! not be able to raise a percentage.

use std::collections::BTreeMap;
use std::process::{Command, ExitCode};

/// Hits per line for each file of the report, merged over every record.
type Record = BTreeMap<u32, u64>;
type Report = BTreeMap<String, Record>;

fn blank(c: char) -> char {
    if c == '\n' || c == '\r' { c } else { ' ' }
}

fn prev_is_ident(c: &[char], i: usize) -> bool {
    i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_')
}

/// `'x'` or `'\n'` as opposed to a lifetime such as `'a`.
fn is_char_literal(c: &[char], i: usize) -> bool {
    match (c.get(i + 1), c.get(i + 2)) {
        (Some('\\'), _) => true,
        (Some(_), Some('\'')) => true,
        _ => false,
    }
}

/// Replace every string, char literal and comment body by spaces, keeping the
/// line structure: braces and quotes are counted from code only.
fn mask_literals(source: &str) -> String {
    let c: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let mut i = 0usize;
    while i < c.len() {
        let ch = c[i];
        // line comment
        if ch == '/' && c.get(i + 1) == Some(&'/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if ch == '/' && c.get(i + 1) == Some(&'*') {
            out.push(' ');
            out.push(' ');
            i += 2;
            while i < c.len() {
                if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    break;
                }
                out.push(blank(c[i]));
                i += 1;
            }
        } else if ch == 'r' && matches!(c.get(i + 1), Some('"') | Some('#')) && !prev_is_ident(&c, i) {
            let mut j = i + 1;
            while c.get(j) == Some(&'#') {
                j += 1;
            }
            if c.get(j) != Some(&'"') {
                out.push(ch);
                i += 1;
                continue;
            }
            let hashes = j - i - 1;
            for _ in i..=j {
                out.push(' ');
            }
            i = j + 1;
            while i < c.len() {
                if c[i] == '"' && (1..=hashes).all(|k| c.get(i + k) == Some(&'#')) {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    i += hashes + 1;
                    break;
                }
                out.push(blank(c[i]));
                i += 1;
            }
        } else if ch == '"' {
            out.push(' ');
            i += 1;
            while i < c.len() {
                if c[i] == '\\' {
                    out.push(' ');
                    if let Some(&n) = c.get(i + 1) {
                        out.push(blank(n));
                    }
                    i += 2;
                } else if c[i] == '"' {
                    out.push(' ');
                    i += 1;
                    break;
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
        } else if ch == '\'' && is_char_literal(&c, i) {
            out.push(' ');
            i += 1;
            while i < c.len() {
                let end = c[i] == '\'';
                if c[i] == '\\' {
                    out.push(' ');
                    i += 1;
                }
                out.push(' ');
                i += 1;
                if end {
                    break;
                }
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// First line (1-based) of the file's trailing test module, or `u32::MAX`.
///
/// The patch gate of `src/refs` already refuses to let a module's tests grade
/// themselves; this tool does the same for the whole diff, because a test line
/// is executed by construction and would otherwise fill the denominator with
/// free points.
fn tests_start(source: &str) -> u32 {
    let masked = mask_literals(source);
    let lines: Vec<&str> = masked.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if l.trim_end() != "#[cfg(test)]" {
            continue;
        }
        let Some(next) = lines[i + 1..].iter().find(|l| !l.trim().is_empty()) else {
            continue;
        };
        if !(next.starts_with("mod ") || next.starts_with("pub mod ")) {
            continue;
        }
        let rest = lines[i + 1..].join("\n");
        let mut depth = 0usize;
        let mut end = None;
        for (at, ch) in rest.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = Some(at + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if end.is_some_and(|e| rest[e..].trim().is_empty()) {
            return i as u32 + 1;
        }
    }
    u32::MAX
}

fn parse_lcov(text: &str) -> Report {
    let mut report: Report = BTreeMap::new();
    let mut current: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(path) = line.strip_prefix("SF:") {
            report.entry(path.to_string()).or_default();
            current = Some(path.to_string());
        } else if let Some(datum) = line.strip_prefix("DA:") {
            let Some(file) = &current else { continue };
            let mut parts = datum.split(',');
            let (Some(n), Some(h)) = (parts.next(), parts.next()) else {
                continue;
            };
            if let (Ok(n), Ok(h)) = (n.parse::<u32>(), h.parse::<u64>()) {
                *report.get_mut(file).unwrap().entry(n).or_insert(0) += h;
            }
        } else if line == "end_of_record" {
            current = None;
        }
    }
    report
}

/// `@@ -a,b +c,d @@` → the NEW side range `(c, c+d-1)`, `None` when nothing is
/// added (`d == 0`). A bare `+c` means a length of 1.
fn hunk_new_range(line: &str) -> Option<(u32, u32)> {
    let rest = line.strip_prefix("@@ ")?;
    let plus = rest.split(' ').find(|s| s.starts_with('+'))?;
    let body = plus.trim_start_matches('+');
    let (start, len) = match body.split_once(',') {
        Some((s, l)) => (s.parse::<u32>().ok()?, l.parse::<u32>().ok()?),
        None => (body.parse::<u32>().ok()?, 1),
    };
    if len == 0 { None } else { Some((start, start + len - 1)) }
}

/// Added line ranges per NEW-side path, from a `git diff -U0` text.
fn parse_unified_diff(text: &str) -> BTreeMap<String, Vec<(u32, u32)>> {
    let mut out: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            current = if path == "/dev/null" {
                None
            } else {
                Some(path.trim_start_matches("b/").to_string())
            };
            continue;
        }
        if line.starts_with("--- ") || line.starts_with("diff --git ") {
            continue;
        }
        if line.starts_with("@@") {
            if let (Some(file), Some(range)) = (&current, hunk_new_range(line)) {
                out.entry(file.clone()).or_default().push(range);
            }
        }
    }
    out
}

/// The report record of `rel`: the exact key, else the single key that ends
/// with `/rel` (llvm-cov writes absolute paths on CI, relative ones locally).
fn find_record<'a>(report: &'a Report, rel: &str) -> Option<&'a Record> {
    if let Some(r) = report.get(rel) {
        return Some(r);
    }
    let suffix = format!("/{rel}");
    let mut hits = report.iter().filter(|(k, _)| k.ends_with(&suffix));
    let first = hits.next()?;
    if hits.next().is_some() {
        return None;
    }
    Some(first.1)
}

#[derive(Debug, PartialEq, Default)]
struct Tally {
    covered: usize,
    instrumented: usize,
    unknown: usize,
}

#[derive(Debug, PartialEq)]
struct FileTally {
    path: String,
    tally: Tally,
    uncovered: Vec<u32>,
}

/// Count, per added line, whether the report instruments it and whether it ran.
/// `test_from` cuts the file's trailing test module (never counted).
fn tally_file(record: Option<&Record>, ranges: &[(u32, u32)], test_from: u32) -> (Tally, Vec<u32>) {
    let mut t = Tally::default();
    let mut uncovered = Vec::new();
    for (start, end) in ranges {
        for line in *start..=*end {
            if line >= test_from {
                continue;
            }
            match record.and_then(|r| r.get(&line)) {
                Some(hits) => {
                    t.instrumented += 1;
                    if *hits > 0 {
                        t.covered += 1;
                    } else {
                        uncovered.push(line);
                    }
                }
                None => t.unknown += 1,
            }
        }
    }
    (t, uncovered)
}

fn gather(
    report: &Report,
    diff: &BTreeMap<String, Vec<(u32, u32)>>,
    root: &str,
    exclude_tests: bool,
) -> Vec<FileTally> {
    let mut out = Vec::new();
    for (path, ranges) in diff {
        if !path.ends_with(".rs") {
            continue;
        }
        let source = std::fs::read_to_string(format!("{root}/{path}")).unwrap_or_default();
        let cut = if exclude_tests { tests_start(&source) } else { u32::MAX };
        let (tally, uncovered) = tally_file(find_record(report, path), ranges, cut);
        if tally.instrumented == 0 && tally.unknown == 0 {
            continue;
        }
        out.push(FileTally { path: path.clone(), tally, uncovered });
    }
    out
}

fn percent(covered: usize, total: usize) -> f64 {
    if total == 0 { 100.0 } else { covered as f64 * 100.0 / total as f64 }
}

fn compact(lines: &[u32]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let start = lines[i];
        let mut end = start;
        while i + 1 < lines.len() && lines[i + 1] == end + 1 {
            i += 1;
            end = lines[i];
        }
        parts.push(if start == end { start.to_string() } else { format!("{start}-{end}") });
        i += 1;
    }
    parts.join(",")
}

struct Args {
    lcov: String,
    base: String,
    min: f64,
    root: String,
    strict: bool,
    exclude_tests: bool,
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut args = Args {
        lcov: String::new(),
        base: "origin/main".to_string(),
        min: 95.0,
        root: ".".to_string(),
        strict: false,
        exclude_tests: false,
    };
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base" => args.base = it.next().ok_or("--base wants a ref")?.clone(),
            "--min" => {
                let v = it.next().ok_or("--min wants a number")?;
                args.min = v.parse().map_err(|_| format!("--min: '{v}' is not a number"))?;
            }
            "--repo" => args.root = it.next().ok_or("--repo wants a directory")?.clone(),
            "--strict" => args.strict = true,
            "--exclude-tests" => args.exclude_tests = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => {
                if !args.lcov.is_empty() {
                    return Err(format!("unexpected argument {other}"));
                }
                args.lcov = other.to_string();
            }
        }
    }
    if args.lcov.is_empty() {
        return Err("usage: patch_coverage <lcov> [--base <ref>] [--min <pct>] [--repo <dir>] [--strict]".to_string());
    }
    Ok(args)
}

fn git_diff(root: &str, base: &str) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "-U0", &format!("{base}...HEAD"), "--", "*.rs"])
        .output()
        .map_err(|e| format!("git diff: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git diff {base}...HEAD failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run(args: &Args) -> Result<bool, String> {
    let text = std::fs::read_to_string(&args.lcov).map_err(|e| format!("{}: {e}", args.lcov))?;
    let report = parse_lcov(&text);
    if report.is_empty() {
        return Err(format!("{}: no SF: record — empty or truncated report", args.lcov));
    }
    let diff = parse_unified_diff(&git_diff(&args.root, &args.base)?);
    let files = gather(&report, &diff, &args.root, args.exclude_tests);

    let covered: usize = files.iter().map(|f| f.tally.covered).sum();
    let instrumented: usize = files.iter().map(|f| f.tally.instrumented).sum();
    let unknown: usize = files.iter().map(|f| f.tally.unknown).sum();
    let pct = percent(covered, instrumented);

    println!(
        "patch coverage: {pct:.2}% ({covered}/{instrumented} instrumented lines added) — target {:.2}% over {}{}{}",
        args.min,
        args.base,
        if args.exclude_tests { " [test modules excluded]" } else { "" },
        if args.strict { " [strict]" } else { "" }
    );
    if unknown > 0 {
        println!(
            "  {unknown} added line(s) are in no record of this report: not instrumented (blank, comment, declaration), \
             or in a file this lcov does not cover yet — never counted as covered"
        );
    }
    let mut worst: Vec<&FileTally> = files.iter().filter(|f| f.tally.instrumented > 0).collect();
    worst.sort_by(|a, b| {
        let ka = percent(a.tally.covered, a.tally.instrumented);
        let kb = percent(b.tally.covered, b.tally.instrumented);
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for f in &worst {
        if f.uncovered.is_empty() {
            continue;
        }
        println!(
            "  {:>6.2}%  {}  ({}/{} lines)  uncovered: {}",
            percent(f.tally.covered, f.tally.instrumented),
            f.path,
            f.tally.covered,
            f.tally.instrumented,
            compact(&f.uncovered)
        );
    }
    if instrumented == 0 {
        println!("  nothing instrumented in the diff: nothing to gate");
    }
    let ok = covered as f64 * 100.0 >= instrumented as f64 * args.min;
    if !ok {
        println!(
            "FAIL: {:.2}% < {:.2}% — {}/{instrumented} covered, {} line(s) to write tests for",
            pct,
            args.min,
            covered,
            instrumented - covered
        );
    }
    if args.strict {
        let ghosts: Vec<&FileTally> = files
            .iter()
            .filter(|f| f.tally.instrumented == 0 && f.tally.unknown > 0)
            .map(|f| f)
            .collect();
        if !ghosts.is_empty() {
            let names: Vec<&str> = ghosts.iter().map(|f| f.path.as_str()).collect();
            println!("FAIL [strict]: file(s) absent from the report: {}", names.join(", "));
            return Ok(false);
        }
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("patch_coverage: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFF: &str = "\
diff --git a/src/a.rs b/src/a.rs
index 111..222 100644
--- a/src/a.rs
+++ b/src/a.rs
@@ -10,2 +10,3 @@ fn f() {
@@ -40 +41,0 @@
diff --git a/docs/x.md b/docs/x.md
--- a/docs/x.md
+++ b/docs/x.md
@@ -1 +1,2 @@
+hello
diff --git a/src/new.rs b/src/new.rs
new file mode 100644
--- /dev/null
+++ b/src/new.rs
@@ -0,0 +1,2 @@
+fn a() {}
";

    #[test]
    fn hunk_ranges_follow_the_new_side() {
        assert_eq!(hunk_new_range("@@ -10,2 +10,3 @@ fn f() {"), Some((10, 12)));
        assert_eq!(hunk_new_range("@@ -40 +41,0 @@"), None);
        assert_eq!(hunk_new_range("@@ -1 +7 @@"), Some((7, 7)));
        assert_eq!(hunk_new_range("not a hunk"), None);
    }

    #[test]
    fn diff_collects_new_side_per_file_and_skips_deleted() {
        let d = parse_unified_diff(DIFF);
        assert_eq!(d.get("src/a.rs"), Some(&vec![(10, 12)]));
        assert_eq!(d.get("src/new.rs"), Some(&vec![(1, 2)]));
        assert_eq!(d.get("docs/x.md"), Some(&vec![(1, 2)]));
    }

    #[test]
    fn a_line_the_report_does_not_instrument_is_never_covered() {
        let mut record = Record::new();
        record.insert(10, 3);
        record.insert(12, 0);
        let (tally, uncovered) = tally_file(Some(&record), &[(10, 12)], u32::MAX);
        assert_eq!(tally, Tally { covered: 1, instrumented: 2, unknown: 1 });
        assert_eq!(uncovered, vec![12]);
    }

    #[test]
    fn a_file_absent_from_the_report_counts_as_unknown_not_covered() {
        let (tally, uncovered) = tally_file(None, &[(1, 2)], u32::MAX);
        assert_eq!(tally, Tally { covered: 0, instrumented: 0, unknown: 2 });
        assert!(uncovered.is_empty());
    }

    #[test]
    fn the_trailing_test_module_never_grades_itself() {
        let mut record = Record::new();
        record.insert(1, 1);
        record.insert(5, 1);
        let (tally, _) = tally_file(Some(&record), &[(1, 5)], 4);
        assert_eq!(tally.instrumented, 1);
        assert_eq!(tally.covered, 1);
    }

    #[test]
    fn lcov_sums_hits_over_records() {
        let text = "SF:src/a.rs\nDA:1,0\nDA:2,1\nend_of_record\nSF:src/a.rs\nDA:1,2\nend_of_record\n";
        let r = parse_lcov(text);
        let rec = r.get("src/a.rs").unwrap();
        assert_eq!(rec.get(&1), Some(&2));
        assert_eq!(rec.get(&2), Some(&1));
    }

    #[test]
    fn absolute_report_paths_match_the_repo_relative_diff_path() {
        let mut r = Report::new();
        r.insert("/ci/work/backend/src/a.rs".to_string(), Record::new());
        assert!(find_record(&r, "src/a.rs").is_some());
        assert!(find_record(&r, "src/b.rs").is_none());
    }

    #[test]
    fn an_ambiguous_suffix_matches_nothing() {
        let mut r = Report::new();
        r.insert("/one/src/a.rs".to_string(), Record::new());
        r.insert("/two/src/a.rs".to_string(), Record::new());
        assert!(find_record(&r, "src/a.rs").is_none());
    }

    #[test]
    fn ranges_are_compacted_for_display() {
        assert_eq!(compact(&[3, 4, 5, 9, 11, 12]), "3-5,9,11-12");
        assert_eq!(compact(&[7]), "7");
    }

    #[test]
    fn the_bar_is_reached_exactly_at_the_target() {
        assert!(95.0 * 100.0 >= 100.0 * 95.0);
        assert!(!(94.0 * 100.0 >= 100.0 * 95.0));
    }

    #[test]
    fn tests_start_finds_only_a_trailing_cfg_test_module() {
        let src = "fn a() {}\n#[cfg(test)]\nmod tests {\n    fn b() {}\n}\n";
        assert_eq!(tests_start(src), 2);
        let mid = "fn a() {}\n#[cfg(test)]\nfn c() {}\n";
        assert_eq!(tests_start(mid), u32::MAX);
    }
}
