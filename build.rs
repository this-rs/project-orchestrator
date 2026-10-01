//! Build script: capture `git describe` so the update checker can tell which
//! release a source build is based on (see `src/update/version.rs`).
//!
//! Never fails the build: without git (Docker context, crates tarball) it
//! emits an empty `PO_GIT_DESCRIBE`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PO_GIT_DESCRIBE");

    // Allow packagers to pin the value explicitly.
    if let Ok(v) = std::env::var("PO_GIT_DESCRIBE") {
        println!("cargo:rustc-env=PO_GIT_DESCRIBE={v}");
        return;
    }

    let describe = git(&["describe", "--tags", "--long", "--match", "v[0-9]*"]).unwrap_or_default();
    println!("cargo:rustc-env=PO_GIT_DESCRIBE={describe}");

    // Re-run when HEAD moves (checkout / commit) or tags change. Only watch
    // files that exist: a missing path makes Cargo re-run the script on every
    // build. Works for linked worktrees (separate git-dir, shared common dir).
    let (Some(git_dir), Some(common_dir)) = (
        git(&["rev-parse", "--git-dir"]),
        git(&["rev-parse", "--git-common-dir"]),
    ) else {
        return;
    };
    let git_dir = PathBuf::from(git_dir);
    let common_dir = PathBuf::from(common_dir);
    rerun_if_exists(&git_dir.join("HEAD"));
    rerun_if_exists(&common_dir.join("packed-refs"));
    rerun_if_exists(&common_dir.join("refs").join("tags"));
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
        rerun_if_exists(&common_dir.join(head_ref));
    }
}
