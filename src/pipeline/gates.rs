//! # Quality Gate Library — Reusable verification gates
//!
//! Each gate implements the [`QualityGate`] trait and can be composed into pipelines.
//! Gates wrap external CLI tools (`cargo`, `gh`, `npx`, etc.) and parse their output
//! into structured [`GateResult`] values with metrics.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Context passed to quality gates for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateContext {
    /// Working directory for command execution.
    pub cwd: String,
    /// Project slug (for MCP calls).
    pub project_slug: Option<String>,
    /// Arbitrary parameters (gate-specific).
    pub params: HashMap<String, serde_json::Value>,
}

/// Result of a quality gate check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate name.
    pub gate_name: String,
    /// Whether the gate passed.
    pub status: GateStatus,
    /// Structured metrics from the gate.
    pub metrics: HashMap<String, f64>,
    /// Human-readable message.
    pub message: String,
    /// How long the gate took (milliseconds).
    pub duration_ms: u64,
}

/// Outcome status for a gate execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    Pass,
    Fail,
    Skip,
    Error,
}

/// Trait for quality gates.
#[async_trait::async_trait]
pub trait QualityGate: Send + Sync {
    /// Gate name for identification.
    fn name(&self) -> &str;

    /// Execute the gate check.
    async fn check(&self, context: &GateContext) -> GateResult;
}

// ---------------------------------------------------------------------------
// Helper: run a shell command
// ---------------------------------------------------------------------------

/// Run a command and return `(stdout, stderr, exit_code)`.
///
/// Returns `Err` if the command could not be spawned at all (e.g. binary not
/// found).
async fn run_command(cmd: &str, args: &[&str], cwd: &str) -> Result<(String, String, i32), String> {
    let output = tokio::process::Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| format!("failed to spawn `{cmd}`: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let code = output.status.code().unwrap_or(-1);

    Ok((stdout, stderr, code))
}

/// Convenience: build a [`GateResult`] for the "command not found" / spawn
/// failure case — the gate is skipped.
fn skip_result(gate_name: &str, reason: String, duration_ms: u64) -> GateResult {
    GateResult {
        gate_name: gate_name.to_string(),
        status: GateStatus::Skip,
        metrics: HashMap::new(),
        message: reason,
        duration_ms,
    }
}

/// Convenience: build a [`GateResult`] for an unexpected error.
fn error_result(gate_name: &str, reason: String, duration_ms: u64) -> GateResult {
    GateResult {
        gate_name: gate_name.to_string(),
        status: GateStatus::Error,
        metrics: HashMap::new(),
        message: reason,
        duration_ms,
    }
}

// ---------------------------------------------------------------------------
// 1. CargoCheckGate
// ---------------------------------------------------------------------------

/// Runs `cargo check` and reports error / warning counts.
#[derive(Debug, Clone)]
pub struct CargoCheckGate;

#[async_trait::async_trait]
impl QualityGate for CargoCheckGate {
    fn name(&self) -> &str {
        "cargo-check"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        let (stdout, stderr, code) = match run_command("cargo", &["check", "--message-format=short"], &ctx.cwd).await {
            Ok(v) => v,
            Err(e) => return skip_result(self.name(), e, start.elapsed().as_millis() as u64),
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let combined = format!("{stdout}\n{stderr}");
        let error_count = combined.matches("error[").count() as f64
            + combined.matches("error:").count() as f64;
        let warning_count = combined.matches("warning:").count() as f64;

        let mut metrics = HashMap::new();
        metrics.insert("error_count".into(), error_count);
        metrics.insert("warning_count".into(), warning_count);

        let status = if code == 0 {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };

        GateResult {
            gate_name: self.name().to_string(),
            status,
            metrics,
            message: if code == 0 {
                format!("cargo check passed ({warning_count} warnings)")
            } else {
                format!("cargo check failed ({error_count} errors, {warning_count} warnings)")
            },
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. CargoTestGate
// ---------------------------------------------------------------------------

/// Runs `cargo test` and reports pass / fail / ignored counts.
#[derive(Debug, Clone)]
pub struct CargoTestGate;

#[async_trait::async_trait]
impl QualityGate for CargoTestGate {
    fn name(&self) -> &str {
        "cargo-test"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        let (stdout, stderr, code) = match run_command("cargo", &["test", "--", "--format=terse"], &ctx.cwd).await {
            Ok(v) => v,
            Err(e) => return skip_result(self.name(), e, start.elapsed().as_millis() as u64),
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let combined = format!("{stdout}\n{stderr}");

        // Parse lines like: "test result: ok. 42 passed; 0 failed; 3 ignored; ..."
        let (mut passed, mut failed, mut ignored) = (0.0_f64, 0.0_f64, 0.0_f64);
        for line in combined.lines() {
            if line.starts_with("test result:") {
                if let Some(p) = extract_number_before(line, "passed") {
                    passed += p;
                }
                if let Some(f) = extract_number_before(line, "failed") {
                    failed += f;
                }
                if let Some(i) = extract_number_before(line, "ignored") {
                    ignored += i;
                }
            }
        }
        let total = passed + failed + ignored;

        let mut metrics = HashMap::new();
        metrics.insert("tests_passed".into(), passed);
        metrics.insert("tests_failed".into(), failed);
        metrics.insert("tests_ignored".into(), ignored);
        metrics.insert("tests_total".into(), total);

        let status = if code == 0 {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };

        GateResult {
            gate_name: self.name().to_string(),
            status,
            metrics,
            message: if code == 0 {
                format!("tests passed: {passed} ok, {ignored} ignored")
            } else {
                format!("tests failed: {failed} failures out of {total}")
            },
            duration_ms,
        }
    }
}

/// Extract the integer that appears immediately before `label` in a string.
/// E.g. `extract_number_before("42 passed", "passed")` → `Some(42.0)`.
fn extract_number_before(haystack: &str, label: &str) -> Option<f64> {
    let idx = haystack.find(label)?;
    let prefix = haystack[..idx].trim_end();
    let token = prefix.rsplit(|c: char| !c.is_ascii_digit()).next()?;
    token.parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
// 3. CoverageGate
// ---------------------------------------------------------------------------

/// Runs `cargo tarpaulin --out json` and checks coverage against a threshold.
#[derive(Debug, Clone)]
pub struct CoverageGate {
    /// Minimum coverage percentage to pass (0–100). Default: 60.0.
    pub threshold: f64,
}

impl Default for CoverageGate {
    fn default() -> Self {
        Self { threshold: 60.0 }
    }
}

#[async_trait::async_trait]
impl QualityGate for CoverageGate {
    fn name(&self) -> &str {
        "coverage"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        let (stdout, _stderr, code) =
            match run_command("cargo", &["tarpaulin", "--out", "json", "--output-dir", "/tmp"], &ctx.cwd).await {
                Ok(v) => v,
                Err(e) => return skip_result(self.name(), e, start.elapsed().as_millis() as u64),
            };
        let duration_ms = start.elapsed().as_millis() as u64;

        if code != 0 {
            return error_result(
                self.name(),
                "cargo tarpaulin exited with non-zero status".into(),
                duration_ms,
            );
        }

        // tarpaulin JSON contains top-level keys "coverage" (as a percentage),
        // "covered" and "coverable".
        let (coverage_pct, lines_covered, lines_total) = parse_tarpaulin_json(&stdout);

        let mut metrics = HashMap::new();
        metrics.insert("coverage_pct".into(), coverage_pct);
        metrics.insert("lines_covered".into(), lines_covered);
        metrics.insert("lines_total".into(), lines_total);

        let status = if coverage_pct >= self.threshold {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };

        GateResult {
            gate_name: self.name().to_string(),
            status,
            metrics,
            message: format!(
                "coverage {coverage_pct:.1}% (threshold {:.1}%) — {lines_covered}/{lines_total} lines",
                self.threshold,
            ),
            duration_ms,
        }
    }
}

/// Best-effort extraction of coverage numbers from tarpaulin JSON output.
fn parse_tarpaulin_json(json_str: &str) -> (f64, f64, f64) {
    // tarpaulin may emit non-JSON lines before the actual JSON; find the first `{`.
    let start = json_str.find('{').unwrap_or(0);
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str[start..]) {
        let pct = val.get("coverage").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let covered = val.get("covered").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let coverable = val.get("coverable").and_then(|v| v.as_f64()).unwrap_or(0.0);
        (pct, covered, coverable)
    } else {
        (0.0, 0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// 4. NpmTypeCheckGate
// ---------------------------------------------------------------------------

/// Runs `npx tsc --noEmit` and reports error count.
#[derive(Debug, Clone)]
pub struct NpmTypeCheckGate;

#[async_trait::async_trait]
impl QualityGate for NpmTypeCheckGate {
    fn name(&self) -> &str {
        "npm-typecheck"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        let (stdout, stderr, code) = match run_command("npx", &["tsc", "--noEmit"], &ctx.cwd).await {
            Ok(v) => v,
            Err(e) => return skip_result(self.name(), e, start.elapsed().as_millis() as u64),
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let combined = format!("{stdout}\n{stderr}");
        // TypeScript errors look like: "src/foo.ts(12,5): error TS2322: ..."
        let error_count = combined.matches("): error TS").count() as f64;

        let mut metrics = HashMap::new();
        metrics.insert("error_count".into(), error_count);

        let status = if code == 0 {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };

        GateResult {
            gate_name: self.name().to_string(),
            status,
            metrics,
            message: if code == 0 {
                "tsc --noEmit passed".into()
            } else {
                format!("tsc --noEmit failed ({error_count} errors)")
            },
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// 5. CiWatchGate
// ---------------------------------------------------------------------------

/// Runs `gh run watch` to wait for a CI run to complete.
#[derive(Debug, Clone)]
pub struct CiWatchGate {
    /// Maximum time to wait for CI (seconds). Default: 600.
    pub timeout_secs: u64,
}

impl Default for CiWatchGate {
    fn default() -> Self {
        Self { timeout_secs: 600 }
    }
}

#[async_trait::async_trait]
impl QualityGate for CiWatchGate {
    fn name(&self) -> &str {
        "ci-watch"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();

        let timeout_str = self.timeout_secs.to_string();
        let args: Vec<&str> = vec![
            "run", "watch", "--exit-status",
        ];

        // We use `timeout` on the tokio side rather than passing --interval
        // to gh so that we can enforce a hard ceiling.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            run_command("gh", &args, &ctx.cwd),
        )
        .await;

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(Ok((_stdout, _stderr, code))) => {
                let ci_success = code == 0;
                let mut metrics = HashMap::new();
                metrics.insert("ci_status".into(), if ci_success { 1.0 } else { 0.0 });

                GateResult {
                    gate_name: self.name().to_string(),
                    status: if ci_success { GateStatus::Pass } else { GateStatus::Fail },
                    metrics,
                    message: if ci_success {
                        "CI run succeeded".into()
                    } else {
                        "CI run failed".into()
                    },
                    duration_ms,
                }
            }
            Ok(Err(e)) => skip_result(self.name(), e, duration_ms),
            Err(_) => {
                let mut metrics: HashMap<String, f64> = HashMap::new();
                metrics.insert("ci_status".into(), 0.0);
                error_result(
                    self.name(),
                    format!("CI watch timed out after {timeout_str}s"),
                    duration_ms,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. PrChecksGate
// ---------------------------------------------------------------------------

/// Runs `gh pr checks` to verify PR status checks.
#[derive(Debug, Clone)]
pub struct PrChecksGate;

#[async_trait::async_trait]
impl QualityGate for PrChecksGate {
    fn name(&self) -> &str {
        "pr-checks"
    }

    async fn check(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        let (stdout, stderr, code) = match run_command("gh", &["pr", "checks"], &ctx.cwd).await {
            Ok(v) => v,
            Err(e) => return skip_result(self.name(), e, start.elapsed().as_millis() as u64),
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let combined = format!("{stdout}\n{stderr}");

        // `gh pr checks` output has one line per check; the second column is
        // "pass", "fail", "pending", etc.
        let mut passed = 0.0_f64;
        let mut failed = 0.0_f64;
        let mut total = 0.0_f64;

        for line in combined.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // columns are tab-separated
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() >= 2 {
                total += 1.0;
                match cols[1].trim().to_lowercase().as_str() {
                    "pass" => passed += 1.0,
                    "fail" => failed += 1.0,
                    _ => {} // pending, skipped, etc.
                }
            }
        }

        let mut metrics = HashMap::new();
        metrics.insert("checks_passed".into(), passed);
        metrics.insert("checks_failed".into(), failed);
        metrics.insert("checks_total".into(), total);

        let status = if code == 0 && failed == 0.0 {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };

        GateResult {
            gate_name: self.name().to_string(),
            status,
            metrics,
            message: format!("PR checks: {passed}/{total} passed, {failed} failed"),
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GateStatus serialization -------------------------------------------

    #[test]
    fn gate_status_serializes_to_expected_strings() {
        assert_eq!(serde_json::to_string(&GateStatus::Pass).unwrap(), "\"Pass\"");
        assert_eq!(serde_json::to_string(&GateStatus::Fail).unwrap(), "\"Fail\"");
        assert_eq!(serde_json::to_string(&GateStatus::Skip).unwrap(), "\"Skip\"");
        assert_eq!(serde_json::to_string(&GateStatus::Error).unwrap(), "\"Error\"");
    }

    #[test]
    fn gate_status_roundtrips() {
        for status in [GateStatus::Pass, GateStatus::Fail, GateStatus::Skip, GateStatus::Error] {
            let json = serde_json::to_string(&status).unwrap();
            let back: GateStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    // -- GateContext / GateResult construction --------------------------------

    #[test]
    fn gate_context_construction() {
        let ctx = GateContext {
            cwd: "/tmp".into(),
            project_slug: Some("my-project".into()),
            params: HashMap::new(),
        };
        assert_eq!(ctx.cwd, "/tmp");
        assert_eq!(ctx.project_slug.as_deref(), Some("my-project"));

        // Serialization round-trip
        let json = serde_json::to_string(&ctx).unwrap();
        let back: GateContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cwd, ctx.cwd);
    }

    #[test]
    fn gate_result_construction() {
        let mut metrics = HashMap::new();
        metrics.insert("error_count".into(), 3.0);

        let res = GateResult {
            gate_name: "test-gate".into(),
            status: GateStatus::Fail,
            metrics,
            message: "3 errors found".into(),
            duration_ms: 123,
        };

        assert_eq!(res.gate_name, "test-gate");
        assert_eq!(res.status, GateStatus::Fail);
        assert_eq!(res.metrics["error_count"], 3.0);

        // Serialization round-trip
        let json = serde_json::to_string(&res).unwrap();
        let back: GateResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.gate_name, "test-gate");
        assert_eq!(back.duration_ms, 123);
    }

    // -- Gate name() methods ------------------------------------------------

    #[test]
    fn gate_names() {
        assert_eq!(CargoCheckGate.name(), "cargo-check");
        assert_eq!(CargoTestGate.name(), "cargo-test");
        assert_eq!(CoverageGate::default().name(), "coverage");
        assert_eq!(NpmTypeCheckGate.name(), "npm-typecheck");
        assert_eq!(CiWatchGate::default().name(), "ci-watch");
        assert_eq!(PrChecksGate.name(), "pr-checks");
    }

    // -- run_command helper -------------------------------------------------

    #[tokio::test]
    async fn run_command_echo() {
        let (stdout, _stderr, code) = run_command("echo", &["hello", "world"], "/tmp")
            .await
            .expect("echo should succeed");
        assert_eq!(code, 0);
        assert!(stdout.contains("hello world"));
    }

    #[tokio::test]
    async fn run_command_true() {
        let (_stdout, _stderr, code) = run_command("true", &[], "/tmp")
            .await
            .expect("true should succeed");
        assert_eq!(code, 0);
    }

    #[tokio::test]
    async fn run_command_false() {
        let (_stdout, _stderr, code) = run_command("false", &[], "/tmp")
            .await
            .expect("false should succeed (spawn-wise)");
        assert_ne!(code, 0);
    }

    #[tokio::test]
    async fn run_command_not_found() {
        let result = run_command("this_binary_does_not_exist_xyz", &[], "/tmp").await;
        assert!(result.is_err(), "non-existent binary should return Err");
    }

    // -- extract_number_before helper ---------------------------------------

    #[test]
    fn extract_number_before_works() {
        assert_eq!(extract_number_before("42 passed; 1 failed", "passed"), Some(42.0));
        assert_eq!(extract_number_before("42 passed; 1 failed", "failed"), Some(1.0));
        assert_eq!(extract_number_before("no match here", "passed"), None);
    }

    // -- parse_tarpaulin_json helper ----------------------------------------

    #[test]
    fn parse_tarpaulin_json_works() {
        let json = r#"{"coverage": 75.5, "covered": 151, "coverable": 200}"#;
        let (pct, covered, total) = parse_tarpaulin_json(json);
        assert!((pct - 75.5).abs() < f64::EPSILON);
        assert!((covered - 151.0).abs() < f64::EPSILON);
        assert!((total - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_tarpaulin_json_garbage() {
        let (pct, covered, total) = parse_tarpaulin_json("not json at all");
        assert_eq!(pct, 0.0);
        assert_eq!(covered, 0.0);
        assert_eq!(total, 0.0);
    }
}
