//! Structured validation tool: run a bounded command and return a short,
//! deterministic digest instead of the raw output.
//!
//! This is the second brick of the tool-output economy (after `tool_digest`):
//! it *executes* the repetitive commands an agent keeps running through `Bash`
//! (`cargo test`, `cargo check`, `git status`) and returns the same thing every
//! time — a [`Digest`] of at most 40 summary lines plus the address of the
//! immutable artifact that holds the *whole* output. The agent can read an
//! exact slice of that artifact later with [`Validator::read_portion`], without
//! re-running the command.
//!
//! Nothing here guesses and nothing is lost silently:
//!
//! * a format the digest layer does not recognize is kept in full and reported
//!   `kind: "unknown"`;
//! * the exit code is propagated (`ok` is true only for exit 0);
//! * a command that does not finish in time is killed and reported as a named
//!   [`ValidateError::Timeout`];
//! * a command whose binary is missing is a named [`ValidateError::Spawn`],
//!   never a panic;
//! * a working directory outside the authorized roots is refused *before* any
//!   execution ([`ValidateError::WorkingDirNotAuthorized`]).

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::tool_digest::{self, ArtifactStore, Digest, StoreError};

/// A validation action the tool understands. Each maps to a fixed command; the
/// agent only ever supplies extra arguments, never an arbitrary program, so the
/// tool cannot be turned into a general shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// `cargo test <extra_args…>`
    Test,
    /// `cargo check <extra_args…>`
    Check,
    /// `git status --short` (a clean tree yields an empty, `ok` artifact)
    GitStatus,
}

impl Action {
    /// The action name as published in the MCP tool's `action` enum.
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Test => "test",
            Action::Check => "check",
            Action::GitStatus => "git_status",
        }
    }

    /// Parse a published action name.
    pub fn parse_action(s: &str) -> Result<Action, ValidateError> {
        match s {
            "test" => Ok(Action::Test),
            "check" => Ok(Action::Check),
            "git_status" => Ok(Action::GitStatus),
            other => Err(ValidateError::UnknownAction(other.to_string())),
        }
    }

    /// The argv the action runs (extra args are appended by the caller).
    fn command(&self) -> Vec<&'static str> {
        match self {
            Action::Test => vec!["cargo", "test"],
            Action::Check => vec!["cargo", "check"],
            Action::GitStatus => vec!["git", "status", "--short"],
        }
    }
}

/// A validation runner: an artifact store plus the working directories it may
/// execute in. Clonable and cheap, so the async handler can move a copy into a
/// blocking task.
#[derive(Debug, Clone)]
pub struct Validator {
    store: ArtifactStore,
    /// Canonicalized absolute roots. A `cwd` is executed only when it resolves
    /// inside one of them. Empty means "nothing is authorized" (safe default).
    allowed_roots: Vec<PathBuf>,
}

/// The compact, stable payload an agent receives. `summary` and `counts` are
/// omitted when empty (no empty fields, no duplication of raw output); the raw
/// output lives only in the artifact addressed by `artefact`.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub kind: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<String>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub counts: std::collections::BTreeMap<String, usize>,
    pub artefact: String,
    pub bytes: usize,
    pub truncated: bool,
    pub command: String,
    pub cwd: String,
    pub duration_ms: u64,
}

/// The full result of one execution: the digest plus the execution facts.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub ok: bool,
    pub exit_code: Option<i32>,
    pub digest: Digest,
    pub command: String,
    pub cwd: String,
    pub duration_ms: u64,
    pub bytes: usize,
}

impl Outcome {
    /// The compact payload an agent receives (what the MCP tool returns).
    pub fn response(&self) -> Response {
        Response {
            ok: self.ok,
            exit_code: self.exit_code,
            kind: self.digest.kind.as_str().to_string(),
            summary: self.digest.summary.clone(),
            counts: self.digest.counts.clone(),
            artefact: self.digest.artefact.as_str().to_string(),
            bytes: self.bytes,
            truncated: self.digest.truncated,
            command: self.command.clone(),
            cwd: self.cwd.clone(),
            duration_ms: self.duration_ms,
        }
    }
}

/// Every failure is a name, never a silence.
#[derive(Debug, thiserror::Error)]
pub enum ValidateError {
    #[error("unknown validate action '{0}'")]
    UnknownAction(String),
    #[error("working directory '{0}' is not authorized (allowed roots: {1})")]
    WorkingDirNotAuthorized(String, String),
    #[error("working directory '{0}' does not exist or cannot be opened")]
    WorkingDirMissing(String),
    #[error("validate command timed out after {0} ms")]
    Timeout(u64),
    #[error("failed to spawn command '{0}': {1}")]
    Spawn(String, String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl Validator {
    /// A validator rooted at `store_dir`, executing only inside `allowed_roots`.
    /// Roots are canonicalized; ones that cannot be resolved are dropped.
    pub fn new(store_dir: impl Into<PathBuf>, allowed_roots: Vec<PathBuf>) -> Self {
        let allowed_roots = allowed_roots
            .into_iter()
            .filter_map(|p| p.canonicalize().ok())
            .collect();
        Validator {
            store: ArtifactStore::new(store_dir),
            allowed_roots,
        }
    }

    /// A validator seeded from the environment, for the MCP server process:
    /// `PO_VALIDATE_STORE_DIR` (artifact store directory, default the system
    /// temp dir under `po-validate-artifacts`) and `PO_VALIDATE_ROOTS`
    /// (colon-separated absolute roots), plus the process's own working
    /// directory. With no roots configured, only the launch directory is
    /// authorized.
    pub fn from_env() -> Self {
        let store_dir = std::env::var_os("PO_VALIDATE_STORE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("po-validate-artifacts"));
        let mut roots: Vec<PathBuf> = std::env::var_os("PO_VALIDATE_ROOTS")
            .map(|v| std::env::split_paths(&v).collect())
            .unwrap_or_default();
        if let Ok(cwd) = std::env::current_dir() {
            roots.push(cwd);
        }
        Validator::new(store_dir, roots)
    }

    /// Run a validation action in `cwd` with the given extra arguments and a
    /// per-command timeout, then digest the combined output. This is the public
    /// entry point of the MCP tool.
    pub fn run(
        &self,
        action: Action,
        cwd: &str,
        command_args: &[String],
        timeout_ms: u64,
    ) -> Result<Outcome, ValidateError> {
        let cwd_path = self.authorized_cwd(cwd)?;
        let mut argv: Vec<String> = action.command().into_iter().map(str::to_string).collect();
        argv.extend(command_args.iter().cloned());
        self.execute(&argv, &cwd_path, timeout_ms)
    }

    /// Execute an arbitrary `argv` in an already-authorized `cwd` and digest the
    /// combined output. The action-based [`Self::run`] is built on this; tests
    /// use it to exercise the capture, timeout and missing-binary paths without
    /// a real cargo. It performs no authorization: the caller (or `run`) has
    /// already decided `cwd` is allowed.
    pub fn execute(
        &self,
        argv: &[String],
        cwd: &Path,
        timeout_ms: u64,
    ) -> Result<Outcome, ValidateError> {
        let start = std::time::Instant::now();
        let mut cmd = std::process::Command::new(&argv[0]);
        cmd.args(&argv[1..]).current_dir(cwd);
        let (combined, exit_code) =
            capture_with_timeout(&mut cmd, Duration::from_millis(timeout_ms))?;
        let bytes = combined.len();
        let digest = tool_digest::digest(&self.store, &combined)?;
        Ok(Outcome {
            ok: exit_code == Some(0),
            exit_code,
            digest,
            command: argv.join(" "),
            cwd: cwd.to_string_lossy().into_owned(),
            duration_ms: start.elapsed().as_millis() as u64,
            bytes,
        })
    }

    /// Read `[from_line, to_line)` of a stored artifact (0-based, half-open),
    /// addressed by its hex ref. Never re-runs the command.
    pub fn read_portion(
        &self,
        artefact: &str,
        from_line: usize,
        to_line: usize,
    ) -> Result<String, ValidateError> {
        let r = crate::tool_digest::Ref::from_hex(artefact)?;
        Ok(self.store.read_portion(&r, from_line, to_line)?)
    }

    /// The byte length of a stored artifact (validates the ref exists).
    pub fn artefact_len(&self, artefact: &str) -> Result<usize, ValidateError> {
        let r = crate::tool_digest::Ref::from_hex(artefact)?;
        Ok(self.store.len(&r)?)
    }

    /// The artifact store directory (exposed for tests to inspect persistence).
    pub fn store_dir(&self) -> &Path {
        self.store.dir()
    }

    /// Canonicalize `cwd` and refuse it unless it sits inside an authorized root.
    fn authorized_cwd(&self, cwd: &str) -> Result<PathBuf, ValidateError> {
        let canon = Path::new(cwd)
            .canonicalize()
            .map_err(|_| ValidateError::WorkingDirMissing(cwd.to_string()))?;
        if self.allowed_roots.iter().any(|r| canon.starts_with(r)) {
            Ok(canon)
        } else {
            let roots = self
                .allowed_roots
                .iter()
                .map(|r| r.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(", ");
            Err(ValidateError::WorkingDirNotAuthorized(
                canon.to_string_lossy().into_owned(),
                roots,
            ))
        }
    }
}

/// Spawn `cmd` (with piped stdout+stderr), read both streams concurrently, and
/// wait with a deadline. Returns the combined bytes and the exit code (None when
/// the process was terminated by a signal). Kills the child on timeout.
fn capture_with_timeout(
    cmd: &mut std::process::Command,
    timeout: Duration,
) -> Result<(Vec<u8>, Option<i32>), ValidateError> {
    use std::io::Read;
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| ValidateError::Spawn(format!("{cmd:?}"), e.to_string()))?;
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    let out = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let err = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (o, e) = (out.clone(), err.clone());
    let t1 = std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => o.lock().unwrap().extend_from_slice(&buf[..n]),
            }
        }
    });
    let t2 = std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match stderr.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => e.lock().unwrap().extend_from_slice(&buf[..n]),
            }
        }
    });

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = t1.join();
                let _ = t2.join();
                return Err(ValidateError::Spawn(format!("{cmd:?}"), err.to_string()));
            }
        }
    };
    let _ = t1.join();
    let _ = t2.join();

    if status.is_none() {
        return Err(ValidateError::Timeout(timeout.as_millis() as u64));
    }
    let exit_code = status.as_ref().and_then(|s| s.code());
    let mut combined = Vec::new();
    combined.extend_from_slice(&out.lock().unwrap());
    combined.extend_from_slice(&err.lock().unwrap());
    Ok((combined, exit_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A validator whose store and authorized root live in a fresh temp dir.
    fn validator() -> (Validator, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let v = Validator::new(dir.path().join("artifacts"), vec![dir.path().to_path_buf()]);
        (v, dir)
    }

    #[test]
    fn git_status_success_is_a_short_digest_with_a_recoverable_artifact() {
        let (v, dir) = validator();
        // A real, dirty git repository: one modified tracked file + one untracked.
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&repo)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?} failed: {:?}",
                out.stderr
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "a@b.c"]);
        git(&["config", "user.name", "a"]);
        std::fs::write(repo.join("tracked.txt"), "v1").unwrap();
        git(&["add", "tracked.txt"]);
        git(&["commit", "-q", "-m", "init"]);
        std::fs::write(repo.join("tracked.txt"), "v2").unwrap();
        std::fs::write(repo.join("new.txt"), "untracked").unwrap();

        let out = v
            .run(Action::GitStatus, repo.to_str().unwrap(), &[], 30_000)
            .unwrap();
        assert!(out.ok);
        assert_eq!(out.exit_code, Some(0));
        assert_eq!(out.digest.kind, crate::tool_digest::Kind::Git);
        assert_eq!(out.digest.counts["modified"], 1);
        assert_eq!(out.digest.counts["untracked"], 1);
        assert!(!out.digest.truncated);
        assert!(out.digest.summary.len() <= 40, "short digest");

        // The artifact is present and a slice reads back exact lines.
        let artefact = out.digest.artefact.as_str();
        assert!(v.artefact_len(artefact).unwrap() > 0);
        let slice = v.read_portion(artefact, 0, 100).unwrap();
        assert!(slice.contains("M tracked.txt"), "slice: {slice:?}");
        assert!(slice.contains("?? new.txt"), "slice: {slice:?}");

        // The compact response carries no raw duplication and no empty fields.
        let resp = out.response();
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["kind"], "git");
        assert_eq!(json["ok"], true);
        assert!(!json["summary"].is_null());
        assert!(!json["summary"].as_array().unwrap().is_empty());
    }

    #[test]
    fn a_failing_compile_is_counted_and_localized() {
        let (v, _dir) = validator();
        let fixture = include_bytes!("../tool_digest/fixtures/cargo_check_error.txt");
        let d = tool_digest::digest(&v.store, fixture).unwrap();
        assert_eq!(d.kind, crate::tool_digest::Kind::CargoCheck);
        assert_eq!(d.counts["errors"], 1);
        assert_eq!(d.counts.get("warnings"), Some(&0));
        // `error[E0308]: mismatched types --> src/lib.rs:2:5` — file:line present.
        assert_eq!(
            d.summary[0],
            "error[E0308]: mismatched types --> src/lib.rs:2:5"
        );
        assert!(d.summary.iter().any(|l| l.contains("could not compile")));
        assert!(d.summary.len() <= 40);
    }

    #[test]
    fn a_missing_binary_is_a_named_error_not_a_panic() {
        let (v, dir) = validator();
        let err = v
            .execute(&["/no/such/binary/anywhere".to_string()], dir.path(), 5_000)
            .unwrap_err();
        assert!(
            matches!(err, ValidateError::Spawn(_, _)),
            "expected a spawn error, got {err:?}"
        );
        assert!(err.to_string().contains("failed to spawn"), "{err}");
    }

    #[test]
    fn a_huge_output_stays_bounded_while_the_artifact_holds_everything() {
        let (v, dir) = validator();
        // 20 000 one-line commits: far beyond the digest's 40-line summary cap.
        let cmd =
            "i=0; while [ $i -lt 20000 ]; do printf '%07x commit %d\\n' $i $i; i=$((i+1)); done";
        let out = v
            .execute(
                &["sh".to_string(), "-c".to_string(), cmd.to_string()],
                dir.path(),
                30_000,
            )
            .unwrap();
        assert_eq!(out.digest.kind, crate::tool_digest::Kind::Git);
        assert_eq!(out.digest.counts["commits"], 20_000);
        assert!(out.digest.truncated, "summary was capped");
        assert_eq!(out.digest.summary.len(), 40, "bounded at 40 lines");
        // Nothing lost: the whole 20 000 lines are recoverable from the artifact.
        assert_eq!(
            v.artefact_len(out.digest.artefact.as_str()).unwrap(),
            out.bytes
        );
        let full = v
            .read_portion(out.digest.artefact.as_str(), 0, 100_000)
            .unwrap();
        assert_eq!(full.lines().count(), 20_000);
    }

    #[test]
    fn an_unathorized_working_directory_is_refused_before_execution() {
        let (v, _dir) = validator();
        let outside = tempfile::tempdir().unwrap();
        let err = v
            .run(
                Action::GitStatus,
                outside.path().to_str().unwrap(),
                &[],
                5_000,
            )
            .unwrap_err();
        assert!(
            matches!(err, ValidateError::WorkingDirNotAuthorized(_, _)),
            "expected a refusal, got {err:?}"
        );
        assert!(err.to_string().contains("not authorized"), "{err}");
    }

    #[test]
    fn a_timeout_is_a_named_error_and_kills_the_child() {
        let (v, dir) = validator();
        let err = v
            .execute(
                &["sh".to_string(), "-c".to_string(), "sleep 30".to_string()],
                dir.path(),
                300,
            )
            .unwrap_err();
        assert!(
            matches!(err, ValidateError::Timeout(300)),
            "expected a timeout, got {err:?}"
        );
    }

    #[test]
    fn unknown_output_is_preserved_and_announced() {
        let (v, dir) = validator();
        let out = v
            .execute(
                &[
                    "sh".to_string(),
                    "-c".to_string(),
                    "echo not-a-known-format".to_string(),
                ],
                dir.path(),
                5_000,
            )
            .unwrap();
        assert_eq!(out.digest.kind, crate::tool_digest::Kind::Unknown);
        assert!(out.digest.summary.is_empty());
        assert!(out.digest.counts.is_empty());
        // The full bytes are still recoverable.
        assert_eq!(
            v.read_portion(out.digest.artefact.as_str(), 0, 10).unwrap(),
            "not-a-known-format"
        );
    }
}
