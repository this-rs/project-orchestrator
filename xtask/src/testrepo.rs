//! Throwaway git repositories for the red/green tests of the subcommands:
//! the "simulated trees" of AM-1 and the stale/up-to-date clones of AM-2.
//! Test-only; removed on drop.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

pub struct TempRepo {
    pub path: PathBuf,
}

impl TempRepo {
    /// An empty repository whose default branch is `main`.
    pub fn new() -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path =
            std::env::temp_dir().join(format!("xtask-test-{}-{n}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).expect("temp dir");
        let repo = Self { path };
        repo.git(&["init", "--quiet"]);
        repo.git(&["symbolic-ref", "HEAD", "refs/heads/main"]);
        repo
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Run `git <args>` and return trimmed stdout; panics on failure (a test
    /// fixture that cannot be built is a test failure, not a verdict).
    pub fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.path)
            .args([
                "-c",
                "user.name=xtask",
                "-c",
                "user.email=xtask@example.invalid",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// An empty commit on the current branch; returns its full sha.
    pub fn commit(&self, subject: &str) -> String {
        self.git(&["commit", "--quiet", "--allow-empty", "-m", subject]);
        self.git(&["rev-parse", "HEAD"])
    }

    /// A branch from `start`, with one commit; back on `main` afterwards.
    pub fn side_branch(&self, name: &str, start: &str, subject: &str) -> String {
        self.git(&["checkout", "--quiet", "-b", name, start]);
        let sha = self.commit(subject);
        self.git(&["checkout", "--quiet", "main"]);
        sha
    }
}

impl Drop for TempRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
