//! Deterministic digests of tool results (first brick: no activation).
//!
//! The context of an agent grows mostly through tool output — `Bash` results
//! above all — and every byte added is later re-read hundreds of times. This
//! module is the first stage of a system that keeps the *full* result in a
//! content-addressed [`ArtifactStore`], while exposing a short, deterministic
//! [`Digest`] for the formats it understands. Nothing here guesses: a format it
//! does not recognize is kept in full and summarized as [`Kind::Unknown`], and
//! an artifact it cannot find is a named error, never a silent empty read.
//!
//! This brick is a library only — it is not wired into the request path yet.

use std::collections::BTreeMap;

pub use store::{ArtifactStore, Ref, StoreError};

mod formats;
mod store;

/// What produced a tool result, as far as this brick can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `cargo test` output (test binaries, doc-tests, `test result:` lines).
    CargoTest,
    /// `cargo check` / `cargo build` diagnostics (`error[…]` / `warning:` lines).
    CargoCheck,
    /// `git` output: short `status` / `diff` / `log`.
    Git,
    /// Any other output, kept in full with no summary guessed.
    Unknown,
}

impl Kind {
    /// Stable machine name of the kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::CargoTest => "cargo_test",
            Kind::CargoCheck => "cargo_check",
            Kind::Git => "git",
            Kind::Unknown => "unknown",
        }
    }
}

/// A deterministic, lossy view of one tool result. The artifact referenced by
/// [`Digest::artefact`] always holds the *whole* result; `summary` and `counts`
/// are what this brick extracts, and `truncated` says whether even the summary
/// had to drop lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest {
    pub kind: Kind,
    /// Short, human-readable summary lines (bounded by `formats::MAX_SUMMARY_LINES`).
    pub summary: Vec<String>,
    /// Stable, deterministic counts for the kind (e.g. `passed`, `failed`,
    /// `warnings`, `errors`, `files`, `commits`, …).
    pub counts: BTreeMap<String, usize>,
    /// Address of the full result in the store — never dropped.
    pub artefact: Ref,
    /// `true` when `summary` was capped and dropped lines (read the artifact).
    pub truncated: bool,
}

/// Split `text` into lines, 0-based, with no synthetic trailing empty line.
///
/// A trailing `\n` terminates the last line without starting a new one; an
/// interior `\n` separates; `\r\n` is normalized to `\n`. So `"a\nb\n"` has two
/// lines and `"a\nb\n\n"` has three (`"a"`, `"b"`, `""`). This is the same
/// notion of "line" that [`ArtifactStore::read_portion`] exposes.
pub(crate) fn split_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// Store `bytes` in `store` and digest them. The result is always preserved in
/// full (as `artefact`); a known format gets a real summary and counts, an
/// unknown one is reported as [`Kind::Unknown`] with an empty summary — never a
/// guessed one.
pub fn digest(store: &ArtifactStore, bytes: &[u8]) -> Result<Digest, StoreError> {
    let artefact = store.put(bytes)?;
    let text = String::from_utf8_lossy(bytes);
    match formats::parse(&text) {
        Some(p) => Ok(Digest {
            kind: p.kind,
            summary: p.summary,
            counts: p.counts,
            artefact,
            truncated: p.truncated,
        }),
        None => Ok(Digest {
            kind: Kind::Unknown,
            summary: Vec::new(),
            counts: BTreeMap::new(),
            artefact,
            truncated: false,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ArtifactStore {
        ArtifactStore::new(tempfile::tempdir().unwrap().path())
    }

    fn digest_str(s: &str) -> Digest {
        digest(&store(), s.as_bytes()).unwrap()
    }

    // Real outputs, captured from a scratch crate (`/tmp/geo-fixtures/tinycrate`)
    // and a scratch git repository (`/tmp/geo-fixtures/gitrepo`). Kept verbatim
    // (including cargo's absolute paths) as fixture files.

    #[test]
    fn cargo_test_success_is_summarized() {
        let d = digest_str(include_str!("fixtures/cargo_test_pass.txt"));
        assert_eq!(d.kind, Kind::CargoTest);
        assert_eq!(d.counts["passed"], 2);
        assert_eq!(d.counts["failed"], 0);
        assert_eq!(d.counts["ignored"], 0);
        // One unit-test binary + one doc-test binary.
        assert_eq!(d.counts["binaries"], 2);
        assert!(!d.truncated);
    }

    #[test]
    fn cargo_test_failure_is_summarized_with_diagnostics() {
        let d = digest_str(include_str!("fixtures/cargo_test_fail.txt"));
        assert_eq!(d.kind, Kind::CargoTest);
        assert_eq!(d.counts["passed"], 1);
        assert_eq!(d.counts["failed"], 1);
        assert_eq!(d.counts["binaries"], 1);
        assert!(
            d.summary.iter().any(|l| l.contains("test result: FAILED.")),
            "the failing result line is summarized: {:?}",
            d.summary
        );
        assert!(
            d.summary
                .iter()
                .any(|l| l == "test tests::failing_on_purpose ... FAILED"),
            "the failing test line is summarized: {:?}",
            d.summary
        );
    }

    #[test]
    fn cargo_check_extracts_a_warning_with_its_location() {
        let d = digest_str(include_str!("fixtures/cargo_check_warning.txt"));
        assert_eq!(d.kind, Kind::CargoCheck);
        assert_eq!(
            d.counts["warnings"], 1,
            "the `generated 1 warning` summary line is not a diagnostic"
        );
        assert_eq!(d.counts.get("errors"), Some(&0));
        assert_eq!(
            d.summary[0],
            "warning: unused variable: `unused` --> src/lib.rs:6:9"
        );
        assert!(
            d.summary.iter().any(|l| l.contains("generated 1 warning")),
            "the build-level warning summary stays in the digest: {:?}",
            d.summary
        );
    }

    #[test]
    fn cargo_check_reports_a_compile_error() {
        let d = digest_str(include_str!("fixtures/cargo_check_error.txt"));
        assert_eq!(d.kind, Kind::CargoCheck);
        assert_eq!(
            d.counts["errors"], 1,
            "the `could not compile` summary line is not a diagnostic"
        );
        assert_eq!(d.counts.get("warnings"), Some(&0));
        assert_eq!(
            d.summary[0],
            "error[E0308]: mismatched types --> src/lib.rs:2:5"
        );
        assert!(d.summary.iter().any(|l| l.contains("could not compile")));
    }

    #[test]
    fn git_status_counts_each_change() {
        let d = digest_str(include_str!("fixtures/git_status.txt"));
        assert_eq!(d.kind, Kind::Git);
        assert_eq!(d.counts["modified"], 2);
        assert_eq!(d.counts["untracked"], 1);
    }

    #[test]
    fn git_diff_stat_counts_files_and_lines() {
        let d = digest_str(include_str!("fixtures/git_diff_stat.txt"));
        assert_eq!(d.kind, Kind::Git);
        assert_eq!(d.counts["files"], 2);
        assert_eq!(d.counts["insertions"], 3);
        assert_eq!(d.counts["deletions"], 1);
    }

    #[test]
    fn git_diff_counts_headers_without_a_stat_line() {
        let d = digest_str(include_str!("fixtures/git_diff.txt"));
        assert_eq!(d.kind, Kind::Git);
        assert_eq!(d.counts["files"], 2);
        assert_eq!(d.counts["insertions"], 0);
        assert_eq!(d.counts["deletions"], 0);
    }

    #[test]
    fn git_log_counts_commits() {
        let d = digest_str(include_str!("fixtures/git_log.txt"));
        assert_eq!(d.kind, Kind::Git);
        assert_eq!(d.counts["commits"], 1);
    }

    #[test]
    fn unknown_format_is_kept_in_full_and_not_guessed() {
        let content = "this is not a known tool output\njust some arbitrary bytes\n";
        let s = store();
        let d = digest(&s, content.as_bytes()).unwrap();
        assert_eq!(d.kind, Kind::Unknown);
        assert!(d.summary.is_empty(), "no guessed summary");
        assert!(d.counts.is_empty(), "no guessed counts");
        assert!(!d.truncated);
        // The full artifact is preserved and recoverable.
        assert_eq!(
            s.read_portion(&d.artefact, 0, 2).unwrap(),
            "this is not a known tool output\njust some arbitrary bytes"
        );
        assert_eq!(s.len(&d.artefact).unwrap(), content.len());
    }

    #[test]
    fn empty_output_digests_to_unknown_with_an_empty_artifact() {
        let s = store();
        let d = digest(&s, b"").unwrap();
        assert_eq!(d.kind, Kind::Unknown);
        assert!(d.summary.is_empty());
        assert!(d.counts.is_empty());
        assert_eq!(s.len(&d.artefact).unwrap(), 0);
        assert_eq!(s.read_portion(&d.artefact, 0, 1).unwrap(), "");
    }

    #[test]
    fn a_long_log_is_truncated_but_the_artifact_is_not() {
        // 50 oneline commits: more than formats::MAX_SUMMARY_LINES (40).
        let mut body = String::new();
        for i in 0..50 {
            body.push_str(&format!("{:07x} commit message {}\n", i, i));
        }
        let s = store();
        let d = digest(&s, body.as_bytes()).unwrap();
        assert_eq!(d.kind, Kind::Git);
        assert_eq!(d.counts["commits"], 50);
        assert!(d.truncated, "summary was capped");
        assert_eq!(d.summary.len(), 40);
        // Nothing is lost: the artifact still holds all 50 lines.
        let full = s.read_portion(&d.artefact, 0, 100).unwrap();
        assert_eq!(full.lines().count(), 50);
    }
}
