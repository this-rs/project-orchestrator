//! Blocking coverage gate for a directory, from a merged lcov report.
//!
//! Codecov's patch status is advisory in this repository (`codecov.yml` sets
//! `require_ci_to_pass: false` and the upload uses `fail_ci_if_error: false`),
//! and a file that no test binary ever compiles does not appear in the lcov at
//! all — so it can never lower a percentage. This gate closes both holes for
//! the directories it is pointed at:
//!
//! * every `.rs` file under `--dir` is enumerated from DISK, not from the
//!   report: a file missing from the report fails, unless it holds no `fn` at
//!   all (a file of `pub mod` lines has nothing to instrument);
//! * every file must reach `--min` percent of its instrumented lines;
//! * files named with `--exact` must reach 100 %;
//! * lines of the file's trailing `#[cfg(test)] mod` (the last item, found on code only, never in a string) are not counted:
//!   a module's tests do not get to grade themselves.
//!
//! std only, no dependency. Build and run:
//!
//! ```text
//! rustc -O scripts/lcov_gate.rs -o /tmp/lcov_gate
//! /tmp/lcov_gate lcov.info --dir src/refs --min 90 \
//!     --exact src/refs/access.rs --exact src/refs/validate.rs
//! ```
//!
//! Self-test: `rustc --test scripts/lcov_gate.rs -o /tmp/lcov_gate_test && /tmp/lcov_gate_test`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Hits per line for each file of the report, merged over every record.
type Report = BTreeMap<String, BTreeMap<u32, u64>>;

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

/// Whether `report_path` (an `SF:` value) is `rel` of the crate at `root`:
/// either the same relative path, or exactly `<root>/<rel>`. A suffix match
/// would also accept `crates/other/src/refs/a.rs` from another workspace member.
fn same_file(report_path: &str, rel: &str, roots: &[PathBuf]) -> bool {
    report_path == rel || roots.iter().any(|r| Path::new(report_path) == r.join(rel))
}

/// The record of `rel` (a path relative to the crate root `roots[..]`).
fn record_for<'a>(
    report: &'a Report,
    rel: &str,
    roots: &[PathBuf],
) -> Option<&'a BTreeMap<u32, u64>> {
    report
        .iter()
        .find(|(k, _)| same_file(k, rel, roots))
        .map(|(_, v)| v)
}

/// `source` with the inside of comments, string literals (plain and raw) and
/// char literals blanked out (newlines kept), so that what is left is code only.
fn mask_literals(source: &str) -> String {
    let c: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let blank = |ch: char| if ch == '\n' { '\n' } else { ' ' };
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        let next = c.get(i + 1).copied();
        if ch == '/' && next == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if ch == '/' && next == Some('*') {
            let mut depth = 0;
            while i < c.len() {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
        } else if ch == 'r' && matches!(next, Some('"') | Some('#')) && !prev_is_ident(&c, i) {
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

/// First line (1-based) of the file's test module, or `u32::MAX` when it has none.
///
/// Only a `#[cfg(test)]` at column 0, in code (not inside a string or a
/// comment), directly followed by a `mod` item that is the LAST item of the
/// file, starts the test part. Any other `#[cfg(test)]` (on a `use`, on a `fn`,
/// in the middle of the file) does not hide anything: what follows is counted.
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
        // The module's braces must close, and nothing but blanks may follow.
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

/// Whether the non-test part of `source` defines a function.
fn has_production_fn(source: &str) -> bool {
    let cut = tests_start(source);
    mask_literals(source)
        .lines()
        .take_while({
            let mut n = 0u32;
            move |_| {
                n += 1;
                n < cut
            }
        })
        .any(|l| l.contains("fn "))
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Ok { covered: usize, total: usize },
    /// Not in the report although it holds code: no test binary compiled it.
    Ghost,
    Below { covered: usize, total: usize, need: u32 },
}

fn judge(source: &str, record: Option<&BTreeMap<u32, u64>>, need: u32) -> Verdict {
    let Some(record) = record else {
        return if has_production_fn(source) {
            Verdict::Ghost
        } else {
            Verdict::Ok { covered: 0, total: 0 }
        };
    };
    let cut = tests_start(source);
    let lines: Vec<u64> = record
        .iter()
        .filter(|(n, _)| **n < cut)
        .map(|(_, h)| *h)
        .collect();
    let total = lines.len();
    let covered = lines.iter().filter(|h| **h > 0).count();
    // Code but no instrumented line: nothing was measured, which is not coverage.
    if total == 0 && has_production_fn(source) {
        return Verdict::Below { covered, total, need };
    }
    // covered / total >= need / 100, in integers.
    if covered * 100 >= total * need as usize {
        Verdict::Ok { covered, total }
    } else {
        Verdict::Below { covered, total, need }
    }
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

struct Args {
    lcov: String,
    dir: String,
    min: u32,
    exact: Vec<String>,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut lcov = None;
    let (mut dir, mut min, mut exact) = (None, 90, Vec::new());
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dir" => dir = it.next().cloned(),
            "--min" => {
                min = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|v| *v <= 100)
                    .ok_or("--min needs 0..=100")?
            }
            "--exact" => exact.push(it.next().cloned().ok_or("--exact needs a path")?),
            other if lcov.is_none() && !other.starts_with("--") => lcov = Some(other.to_string()),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Args {
        lcov: lcov.ok_or("usage: lcov_gate <lcov> --dir <dir> [--min N] [--exact file]...")?,
        dir: dir.ok_or("--dir is required")?,
        min,
        exact,
    })
}

fn run(argv: &[String]) -> Result<bool, String> {
    let args = parse_args(argv)?;
    let text = fs::read_to_string(&args.lcov).map_err(|e| format!("{}: {e}", args.lcov))?;
    let report = parse_lcov(&text);
    let mut files = Vec::new();
    rust_files(Path::new(&args.dir), &mut files);
    if files.is_empty() {
        return Err(format!("no .rs file under {}", args.dir));
    }
    for e in &args.exact {
        if !files.iter().any(|f| f.to_string_lossy() == e.as_str()) {
            return Err(format!("--exact {e}: no such file under {}", args.dir));
        }
    }
    let roots: Vec<PathBuf> = std::env::current_dir()
        .ok()
        .into_iter()
        .flat_map(|d| [d.canonicalize().unwrap_or_else(|_| d.clone()), d])
        .collect();
    let mut ok = true;
    for f in &files {
        let rel = f.to_string_lossy().to_string();
        let need = if args.exact.contains(&rel) { 100 } else { args.min };
        let source = fs::read_to_string(f).map_err(|e| format!("{rel}: {e}"))?;
        match judge(&source, record_for(&report, &rel, &roots), need) {
            Verdict::Ok { covered, total } => println!("ok    {rel}: {covered}/{total} (need {need}%)"),
            Verdict::Ghost => {
                ok = false;
                println!("FAIL  {rel}: absent from the report - no test binary compiles it");
            }
            Verdict::Below { covered, total, need } => {
                ok = false;
                println!("FAIL  {rel}: {covered}/{total} lines, need {need}%");
            }
        }
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("lcov_gate: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "pub fn a() {}\npub fn b() {}\n#[cfg(test)]\nmod tests {}\n";

    fn rec(pairs: &[(u32, u64)]) -> BTreeMap<u32, u64> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn records_of_one_file_are_merged() {
        let r = parse_lcov("SF:/w/src/x.rs\nDA:1,0\nDA:2,3\nend_of_record\nSF:/w/src/x.rs\nDA:1,2\nend_of_record\n");
        assert_eq!(r["/w/src/x.rs"], rec(&[(1, 2), (2, 3)]));
    }

    #[test]
    fn absolute_and_relative_paths_both_match() {
        let r = parse_lcov("SF:/w/src/refs/a.rs\nDA:1,1\nend_of_record\n");
        let roots = [PathBuf::from("/w")];
        assert!(record_for(&r, "src/refs/a.rs", &roots).is_some());
        assert!(record_for(&r, "src/refs/b.rs", &roots).is_none());
    }

    #[test]
    fn test_lines_do_not_count() {
        // lines 3 and 4 are test code: a hit there must not hide line 2.
        let v = judge(SRC, Some(&rec(&[(1, 1), (2, 0), (4, 9)])), 100);
        assert_eq!(v, Verdict::Below { covered: 1, total: 2, need: 100 });
    }

    #[test]
    fn threshold_is_inclusive() {
        let r = rec(&[(1, 1), (2, 0)]);
        assert_eq!(judge(SRC, Some(&r), 50), Verdict::Ok { covered: 1, total: 2 });
        assert!(matches!(judge(SRC, Some(&r), 51), Verdict::Below { .. }));
    }

    #[test]
    fn a_file_with_code_missing_from_the_report_is_a_ghost() {
        assert_eq!(judge(SRC, None, 90), Verdict::Ghost);
    }

    #[test]
    fn a_file_without_any_fn_missing_from_the_report_is_fine() {
        assert_eq!(judge("pub mod a;\npub mod b;\n", None, 90), Verdict::Ok { covered: 0, total: 0 });
        // a fn that only lives in the test part does not make it a ghost
        assert_eq!(judge("pub mod a;\n#[cfg(test)]\nmod t {\n    fn t() {}\n}\n", None, 90), Verdict::Ok { covered: 0, total: 0 });
    }

    #[test]
    fn args() {
        let a = parse_args(&["l.info".into(), "--dir".into(), "src/refs".into(), "--min".into(), "80".into(), "--exact".into(), "src/refs/a.rs".into()]).unwrap();
        assert_eq!((a.min, a.exact.len(), a.dir.as_str()), (80, 1, "src/refs"));
        assert!(parse_args(&["l.info".into()]).is_err());
        assert!(parse_args(&["--dir".into(), "d".into()]).is_err());
        assert!(parse_args(&["l".into(), "--dir".into(), "d".into(), "--min".into(), "101".into()]).is_err());
        assert!(parse_args(&["l".into(), "--dir".into(), "d".into(), "--bogus".into()]).is_err());
    }

    #[test]
    fn code_with_no_instrumented_line_does_not_pass() {
        let src = "pub fn a() { 1 }\npub fn b() { 2 }\n#[cfg(test)]\nmod t {}\n";
        assert_eq!(judge(src, Some(&rec(&[])), 100), Verdict::Below { covered: 0, total: 0, need: 100 });
        // only test lines in the report: still nothing measured
        assert!(matches!(judge(src, Some(&rec(&[(4, 1)])), 90), Verdict::Below { .. }));
        // a file without any fn has nothing to instrument
        assert_eq!(judge("pub mod a;\n", Some(&rec(&[])), 100), Verdict::Ok { covered: 0, total: 0 });
    }

    #[test]
    fn an_early_cfg_test_does_not_hide_the_code_after_it() {
        for src in [
            "#[cfg(test)]\nuse foo::bar;\npub fn a() { 1 }\npub fn b() { 2 }\n",
            "pub fn a() { 1 }\n#[cfg(test)]\nfn helper() {}\npub fn b() { 2 }\n",
            "pub fn a() { 1 }\n#[cfg(test)]\nfn helper() { 2 }\n",
            // a test module that is not the last item
            "pub fn a() { 1 }\n#[cfg(test)]\nmod t {}\npub fn b() { 2 }\n",
        ] {
            let v = judge(src, Some(&rec(&[(1, 1), (2, 0), (3, 0), (4, 0)])), 100);
            assert!(matches!(v, Verdict::Below { .. }), "{:?} -> {:?}", src, v);
        }
    }

    #[test]
    fn a_cfg_test_inside_a_literal_or_a_comment_is_not_one() {
        for src in [
            "pub fn a() {\n    let s = \"\n#[cfg(test)]\nmod t {}\n\";\n}\npub fn b() { 2 }\n",
            "pub fn a() {\n    let s = r#\"\n#[cfg(test)]\nmod t {}\n\"#;\n}\npub fn b() { 2 }\n",
            "/*\n#[cfg(test)]\nmod t {}\n*/\npub fn b() { 2 }\n",
            "// x\n    #[cfg(test)]\npub fn b() { 2 }\n",
        ] {
            let last = src.lines().count() as u32;
            let v = judge(src, Some(&rec(&[(1, 1), (last, 0)])), 100);
            assert!(matches!(v, Verdict::Below { .. }), "{:?} -> {:?}", src, v);
        }
    }

    #[test]
    fn the_trailing_test_module_is_cut_even_with_braces_in_literals() {
        let src = "pub fn a() { 1 }\n#[cfg(test)]\nmod t {\n    const S: &str = \"}\";\n    fn x() { let c = '{'; }\n}\n";
        assert_eq!(tests_start(src), 2);
        assert_eq!(judge(src, Some(&rec(&[(1, 1), (5, 0)])), 100), Verdict::Ok { covered: 1, total: 1 });
        assert_eq!(tests_start("fn a() {}\n"), u32::MAX);
        assert_eq!(tests_start("#[cfg(test)]\n"), u32::MAX);
        assert_eq!(tests_start("#[cfg(test)]\nmod t {\n"), u32::MAX);
    }

    #[test]
    fn a_report_path_must_be_the_crate_file_not_a_suffix_match() {
        let r = parse_lcov("SF:/w/crates/other/src/refs/a.rs\nDA:1,1\nend_of_record\n");
        let roots = [PathBuf::from("/w/backend")];
        assert!(record_for(&r, "src/refs/a.rs", &roots).is_none());
        let r = parse_lcov("SF:/w/backend/src/refs/a.rs\nDA:1,1\nend_of_record\nSF:src/refs/b.rs\nDA:1,1\nend_of_record\n");
        assert!(record_for(&r, "src/refs/a.rs", &roots).is_some());
        assert!(record_for(&r, "src/refs/b.rs", &roots).is_some());
        assert!(record_for(&r, "src/refs/a.rs", &[PathBuf::from("/w")]).is_none());
    }
}
