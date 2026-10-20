//! Thin `git` wrappers (process side) and the pure parsers of their output.
//!
//! Everything that spawns a process is in `impl Repo`; everything that reads
//! text is a free function with unit tests. The two never mix, so a
//! subcommand's verdict can be tested from captured strings.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One commit as `git log --format=%H%x09%s` prints it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub sha: String,
    pub subject: String,
}

/// Parse `git log --format=%H%x09%s` output (one `<sha>\t<subject>` per line).
/// Blank lines are skipped; a line without a tab is a commit with an empty
/// subject rather than an error, so one odd line never hides the others.
pub fn parse_log(output: &str) -> Vec<Commit> {
    output
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(|l| match l.split_once('\t') {
            Some((sha, subject)) => Commit {
                sha: sha.trim().to_string(),
                subject: subject.trim().to_string(),
            },
            None => Commit {
                sha: l.trim().to_string(),
                subject: String::new(),
            },
        })
        .collect()
}

/// Parse the single line `git rev-parse` / `git rev-list --count` print.
pub fn first_line(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
}

/// `true` when `s` looks like a (possibly abbreviated) commit sha: 7 to 40
/// lowercase hex characters.
pub fn looks_like_sha(s: &str) -> bool {
    (7..=40).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Abbreviation printed in verdicts (8 characters, as this repo's logs do).
pub fn short(sha: &str) -> &str {
    let n = sha.len().min(8);
    &sha[..n]
}

/// Remote name of a remote-tracking ref (`origin/main` → `origin`); `None`
/// for a local branch or a bare sha, which nothing can fetch.
pub fn remote_of(reference: &str) -> Option<&str> {
    reference.split_once('/').map(|(remote, _)| remote)
}

/// A local git repository (a clone or a worktree). Only `Repo` spawns `git`.
#[derive(Debug, Clone)]
pub struct Repo {
    pub path: PathBuf,
}

impl Repo {
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The toplevel of the repository containing `from`, if any.
    pub fn discover(from: &Path) -> Option<Self> {
        let out = Command::new("git")
            .arg("-C")
            .arg(from)
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        first_line(&String::from_utf8_lossy(&out.stdout)).map(Self::at)
    }

    pub fn is_repo(&self) -> bool {
        self.git(&["rev-parse", "--git-dir"]).is_ok()
    }

    /// Run `git <args>` in the repo; `Ok(stdout)` on success, `Err(stderr)`
    /// otherwise. Never panics on a missing `git`: that is an `Err` too.
    pub fn git(&self, args: &[&str]) -> Result<String, String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.path)
            .args(args)
            .output()
            .map_err(|e| format!("git introuvable : {e}"))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
        }
    }

    /// `git fetch <remote>` (quiet). Failure is reported, not fatal: the
    /// caller keeps going with the refs it has, offline.
    pub fn fetch(&self, remote: &str) -> Result<(), String> {
        self.git(&["fetch", "--quiet", remote]).map(|_| ())
    }

    /// Full sha of `rev` when the repo knows it as a commit.
    pub fn resolve(&self, rev: &str) -> Option<String> {
        let spec = format!("{rev}^{{commit}}");
        self.git(&["rev-parse", "--verify", "--quiet", &spec])
            .ok()
            .and_then(|s| first_line(&s))
    }

    /// Is `ancestor` reachable from `descendant`?
    pub fn is_ancestor(&self, ancestor: &str, descendant: &str) -> bool {
        self.git(&["merge-base", "--is-ancestor", ancestor, descendant])
            .is_ok()
    }

    /// First-parent commits in `from..to`: on a branch merged by squash or
    /// merge commits, that is the list of merges `to` has and `from` lacks.
    pub fn merges_between(&self, from: &str, to: &str) -> Result<Vec<Commit>, String> {
        let range = format!("{from}..{to}");
        self.git(&["log", "--first-parent", "--format=%H%x09%s", &range])
            .map(|s| parse_log(&s))
    }

    /// Number of commits in `from..to` (all parents).
    pub fn count_between(&self, from: &str, to: &str) -> Result<usize, String> {
        let range = format!("{from}..{to}");
        let out = self.git(&["rev-list", "--count", &range])?;
        first_line(&out)
            .and_then(|l| l.parse().ok())
            .ok_or_else(|| format!("rev-list --count {range} : sortie illisible « {out} »"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_log_reads_sha_tab_subject_per_line() {
        let out =
            "860ed17a\tdocs(diagrams): restore (#124)\nec7a8c5b\tfeat(agent): hooks (#123)\n\n";
        let commits = parse_log(out);
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].sha, "860ed17a");
        assert_eq!(commits[0].subject, "docs(diagrams): restore (#124)");
        assert_eq!(commits[1].subject, "feat(agent): hooks (#123)");
    }

    #[test]
    fn parse_log_keeps_a_commit_without_subject() {
        let commits = parse_log("abc1234\n");
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].sha, "abc1234");
        assert_eq!(commits[0].subject, "");
    }

    #[test]
    fn parse_log_of_empty_output_is_empty() {
        assert!(parse_log("").is_empty());
        assert!(parse_log("\n\n").is_empty());
    }

    #[test]
    fn first_line_skips_blank_lines_and_trims() {
        assert_eq!(first_line("\n  abc  \ndef\n").as_deref(), Some("abc"));
        assert_eq!(first_line("   \n"), None);
    }

    #[test]
    fn looks_like_sha_accepts_abbreviated_and_full_lowercase_hex() {
        assert!(looks_like_sha("c94fa37d"));
        assert!(looks_like_sha("b3afe82fc576e80e7ffd7c39b0c25b8f90d72904"));
        assert!(!looks_like_sha("C94FA37D"));
        assert!(!looks_like_sha("abc12"));
        assert!(!looks_like_sha("0.0.17"));
        assert!(!looks_like_sha(""));
    }

    #[test]
    fn short_and_remote_of() {
        assert_eq!(
            short("b3afe82fc576e80e7ffd7c39b0c25b8f90d72904"),
            "b3afe82f"
        );
        assert_eq!(short("abc"), "abc");
        assert_eq!(remote_of("origin/main"), Some("origin"));
        assert_eq!(remote_of("main"), None);
    }
}
