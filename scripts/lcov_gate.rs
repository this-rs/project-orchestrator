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
//! * lines from the first `#[cfg(test)]` of a file onwards are not counted:
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

/// The record of `rel` (a repo-relative path): the report may hold absolute paths.
fn record_for<'a>(report: &'a Report, rel: &str) -> Option<&'a BTreeMap<u32, u64>> {
    let suffix = format!("/{rel}");
    report
        .iter()
        .find(|(k, _)| k.as_str() == rel || k.ends_with(&suffix))
        .map(|(_, v)| v)
}

/// First line (1-based) of the file's test code, or `u32::MAX` when it has none.
fn tests_start(source: &str) -> u32 {
    source
        .lines()
        .position(|l| l.trim_start().starts_with("#[cfg(test)]"))
        .map(|i| i as u32 + 1)
        .unwrap_or(u32::MAX)
}

/// Non-test source text.
fn production_part(source: &str) -> String {
    let cut = tests_start(source);
    source
        .lines()
        .enumerate()
        .take_while(|(i, _)| (*i as u32 + 1) < cut)
        .map(|(_, l)| l)
        .collect::<Vec<_>>()
        .join("\n")
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
        return if production_part(source).contains("fn ") {
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
    let mut ok = true;
    for f in &files {
        let rel = f.to_string_lossy().to_string();
        let need = if args.exact.contains(&rel) { 100 } else { args.min };
        let source = fs::read_to_string(f).map_err(|e| format!("{rel}: {e}"))?;
        match judge(&source, record_for(&report, &rel), need) {
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
        assert!(record_for(&r, "src/refs/a.rs").is_some());
        assert!(record_for(&r, "src/refs/b.rs").is_none());
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
        assert_eq!(judge("pub mod a;\n#[cfg(test)]\nfn t() {}\n", None, 90), Verdict::Ok { covered: 0, total: 0 });
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
}
