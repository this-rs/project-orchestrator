//! TaskVerifier — Post-task verification: build check, step completion, git sanity.
//!
//! After each task executed by the agent, the Verifier automatically checks:
//! - Build compiles (cargo check, npm run build, etc.)
//! - All steps are completed or skipped
//! - Git diff is clean (no sensitive files, reasonable size)
//! - Tests pass (optional, if configured)

use crate::neo4j::models::StepStatus;
use crate::neo4j::traits::GraphStore;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================================
// VerifyResult — outcome of post-task verification
// ============================================================================

/// Result of post-task verification.
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// All checks passed
    Pass,
    /// One or more checks failed
    Fail {
        /// List of failure reasons
        reasons: Vec<String>,
    },
}

impl VerifyResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, VerifyResult::Pass)
    }
}

// ============================================================================
// ProjectLanguage — detected from project files
// ============================================================================

/// Detected project language for build commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectLanguage {
    Rust,
    Node,
    Go,
    Python,
    Unknown,
}

impl ProjectLanguage {
    /// Detect the project language from files in the given directory.
    pub fn detect(cwd: &Path) -> Self {
        if cwd.join("Cargo.toml").exists() {
            ProjectLanguage::Rust
        } else if cwd.join("package.json").exists() {
            ProjectLanguage::Node
        } else if cwd.join("go.mod").exists() {
            ProjectLanguage::Go
        } else if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
            ProjectLanguage::Python
        } else {
            ProjectLanguage::Unknown
        }
    }

    /// Get the build check command for this language.
    pub fn build_command(&self) -> Option<(&str, &[&str])> {
        match self {
            ProjectLanguage::Rust => Some(("cargo", &["check"])),
            ProjectLanguage::Node => Some(("npm", &["run", "build"])),
            ProjectLanguage::Go => Some(("go", &["build", "./..."])),
            ProjectLanguage::Python => None, // No universal build check
            ProjectLanguage::Unknown => None,
        }
    }

    /// Get the test command for this language.
    pub fn test_command(&self) -> Option<(&str, &[&str])> {
        match self {
            ProjectLanguage::Rust => Some(("cargo", &["test", "--lib"])),
            ProjectLanguage::Node => Some(("npm", &["test"])),
            ProjectLanguage::Go => Some(("go", &["test", "./..."])),
            ProjectLanguage::Python => Some(("python", &["-m", "pytest"])),
            ProjectLanguage::Unknown => None,
        }
    }
}

// ============================================================================
// Sensitive file patterns
// ============================================================================

/// File patterns that should never be committed.
const SENSITIVE_PATTERNS: &[&str] = &[
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
    "secrets.yaml",
    "secrets.yml",
];

/// File extensions that indicate sensitive files.
const SENSITIVE_EXTENSIONS: &[&str] = &[".key", ".pem", ".p12", ".pfx", ".secret"];

/// Check if a file path is sensitive.
fn is_sensitive_file(path: &str) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path);

    // Check exact filename matches
    if SENSITIVE_PATTERNS.contains(&filename) {
        return true;
    }

    // Check extension matches
    if SENSITIVE_EXTENSIONS
        .iter()
        .any(|ext| filename.ends_with(ext))
    {
        return true;
    }

    false
}

// ============================================================================
// TaskVerifier — orchestrates all post-task checks
// ============================================================================

/// Post-task verification engine.
///
/// Runs build checks, step completion validation, and git sanity checks
/// after each task to catch issues before marking the task as completed.
pub struct TaskVerifier {
    graph: Arc<dyn GraphStore>,
    /// Whether to run build checks
    build_check_enabled: bool,
    /// Whether to run tests
    test_runner_enabled: bool,
    /// Build/test timeout
    command_timeout: Duration,
}

impl TaskVerifier {
    /// Create a new TaskVerifier.
    pub fn new(
        graph: Arc<dyn GraphStore>,
        build_check_enabled: bool,
        test_runner_enabled: bool,
    ) -> Self {
        Self {
            graph,
            build_check_enabled,
            test_runner_enabled,
            command_timeout: Duration::from_secs(600),
        }
    }

    /// Run all verifications for a completed task.
    ///
    /// Returns `VerifyResult::Pass` if all checks pass, or
    /// `VerifyResult::Fail` with reasons if any check fails.
    pub async fn verify(&self, task_id: Uuid, cwd: &str) -> VerifyResult {
        let mut reasons = Vec::new();
        let cwd_path = Path::new(cwd);

        // 1. Build check
        if self.build_check_enabled {
            if let Err(e) = self.verify_build(cwd_path).await {
                reasons.push(format!("Build check failed: {}", e));
            }
        }

        // 2. Step completion
        if let Err(e) = self.verify_steps(task_id).await {
            reasons.push(format!("Step completion check failed: {}", e));
        }

        // 3. Git sanity
        if let Err(e) = self.verify_git_sanity(cwd_path).await {
            reasons.push(format!("Git sanity check failed: {}", e));
        }

        // 4. Tests (optional)
        if self.test_runner_enabled {
            if let Err(e) = self.verify_tests(cwd_path).await {
                reasons.push(format!("Tests failed: {}", e));
            }
        }

        if reasons.is_empty() {
            info!("Task {} passed all verifications", task_id);
            VerifyResult::Pass
        } else {
            warn!("Task {} failed verification: {:?}", task_id, reasons);
            VerifyResult::Fail { reasons }
        }
    }

    /// Check that the project builds without errors.
    async fn verify_build(&self, cwd: &Path) -> Result<()> {
        let lang = ProjectLanguage::detect(cwd);
        let (cmd, args) = match lang.build_command() {
            Some(c) => c,
            None => {
                info!("No build command for language {:?}, skipping", lang);
                return Ok(());
            }
        };

        info!("Running build check: {} {:?}", cmd, args);
        let output = tokio::time::timeout(
            self.command_timeout,
            tokio::process::Command::new(cmd)
                .args(args)
                .current_dir(cwd)
                .output(),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Build command timed out after {}s",
                self.command_timeout.as_secs()
            )
        })?
        .map_err(|e| anyhow::anyhow!("Failed to run build command: {}", e))?;

        if output.status.success() {
            info!("Build check passed");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Truncate stderr to avoid huge error messages
            let truncated = if stderr.len() > 2000 {
                format!("{}...(truncated)", &stderr[..2000])
            } else {
                stderr.to_string()
            };
            Err(anyhow::anyhow!(
                "{} {} failed:\n{}",
                cmd,
                args.join(" "),
                truncated
            ))
        }
    }

    /// Verify all steps of the task are completed or skipped.
    async fn verify_steps(&self, task_id: Uuid) -> Result<()> {
        let steps = self.graph.get_task_steps(task_id).await?;

        let incomplete: Vec<String> = steps
            .iter()
            .filter(|s| s.status != StepStatus::Completed && s.status != StepStatus::Skipped)
            .map(|s| format!("Step '{}' is still '{:?}'", s.description, s.status))
            .collect();

        if incomplete.is_empty() {
            info!("All {} steps completed/skipped", steps.len());
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "{} step(s) not completed:\n- {}",
                incomplete.len(),
                incomplete.join("\n- ")
            ))
        }
    }

    /// Check git diff for sensitive files and unreasonable sizes.
    async fn verify_git_sanity(&self, cwd: &Path) -> Result<()> {
        // Get list of changed files
        let output = tokio::process::Command::new("git")
            .args(["diff", "--name-only", "HEAD~1..HEAD"])
            .current_dir(cwd)
            .output()
            .await;

        let output = match output {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                // git diff might fail if no commits yet or not a git repo — not an error
                let stderr = String::from_utf8_lossy(&o.stderr);
                // Check common non-error cases (locale-independent: also check exit code)
                if stderr.contains("unknown revision")
                    || stderr.contains("not a git repository")
                    || stderr.contains("dépôt git") // French locale
                    || o.status.code().is_some_and(|c| c > 1)
                // git exits >1 for usage errors
                {
                    info!("No git context to diff against, skipping git sanity");
                    return Ok(());
                }
                return Err(anyhow::anyhow!("git diff failed: {}", stderr));
            }
            Err(e) => {
                warn!("Failed to run git diff: {}", e);
                return Ok(()); // Don't fail the task if git isn't available
            }
        };

        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
        let changed_files: Vec<&str> = stdout_str.lines().collect();

        // Check for sensitive files
        let sensitive: Vec<&str> = changed_files
            .iter()
            .copied()
            .filter(|f| is_sensitive_file(f))
            .collect();

        if !sensitive.is_empty() {
            return Err(anyhow::anyhow!(
                "Sensitive files detected in commit: {}",
                sensitive
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Check diff size (stat)
        let stat_output = tokio::process::Command::new("git")
            .args(["diff", "--stat", "HEAD~1..HEAD"])
            .current_dir(cwd)
            .output()
            .await;

        if let Ok(stat) = stat_output {
            let stat_str = String::from_utf8_lossy(&stat.stdout);
            // Parse the last line for total insertions/deletions
            if let Some(last_line) = stat_str.lines().last() {
                // Check for unreasonably large changes (>10000 lines)
                if let Some(insertions) = extract_stat_number(last_line, "insertion") {
                    if insertions > 10000 {
                        warn!(
                            "Very large diff ({} insertions) — might contain generated/binary files",
                            insertions
                        );
                    }
                }
            }
        }

        info!(
            "Git sanity check passed ({} files changed)",
            changed_files.len()
        );
        Ok(())
    }

    /// Run tests if configured.
    async fn verify_tests(&self, cwd: &Path) -> Result<()> {
        let lang = ProjectLanguage::detect(cwd);
        let (cmd, args) = match lang.test_command() {
            Some(c) => c,
            None => {
                info!("No test command for language {:?}, skipping", lang);
                return Ok(());
            }
        };

        info!("Running tests: {} {:?}", cmd, args);
        let output = tokio::time::timeout(
            self.command_timeout,
            tokio::process::Command::new(cmd)
                .args(args)
                .current_dir(cwd)
                .output(),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Test command timed out after {}s",
                self.command_timeout.as_secs()
            )
        })?
        .map_err(|e| anyhow::anyhow!("Failed to run test command: {}", e))?;

        if output.status.success() {
            info!("Tests passed");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("{}\n{}", stdout, stderr);
            let truncated = if combined.len() > 2000 {
                format!("{}...(truncated)", &combined[..2000])
            } else {
                combined
            };
            Err(anyhow::anyhow!("Tests failed:\n{}", truncated))
        }
    }
}

/// Extract a number from git stat output (e.g., "3 insertions(+)")
fn extract_stat_number(line: &str, keyword: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if part.starts_with(keyword) && i > 0 {
            return parts[i - 1].parse().ok();
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::StepNode;

    #[test]
    fn test_project_language_detect_rust() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Rust);
    }

    #[test]
    fn test_project_language_detect_node() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("package.json"), "{}").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Node);
    }

    #[test]
    fn test_project_language_detect_go() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("go.mod"), "module test").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Go);
    }

    #[test]
    fn test_project_language_detect_python() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("pyproject.toml"), "").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Python);
    }

    #[test]
    fn test_project_language_detect_unknown() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            ProjectLanguage::detect(tmp.path()),
            ProjectLanguage::Unknown
        );
    }

    #[test]
    fn test_project_language_build_commands() {
        assert!(ProjectLanguage::Rust.build_command().is_some());
        assert!(ProjectLanguage::Node.build_command().is_some());
        assert!(ProjectLanguage::Go.build_command().is_some());
        assert!(ProjectLanguage::Python.build_command().is_none());
        assert!(ProjectLanguage::Unknown.build_command().is_none());
    }

    #[test]
    fn test_is_sensitive_file() {
        assert!(is_sensitive_file(".env"));
        assert!(is_sensitive_file(".env.local"));
        assert!(is_sensitive_file(".env.production"));
        assert!(is_sensitive_file("credentials.json"));
        assert!(is_sensitive_file("server.key"));
        assert!(is_sensitive_file("cert.pem"));
        assert!(is_sensitive_file("keystore.p12"));
        assert!(is_sensitive_file("my.secret"));

        assert!(!is_sensitive_file("main.rs"));
        assert!(!is_sensitive_file("config.yaml"));
        assert!(!is_sensitive_file("README.md"));
        assert!(!is_sensitive_file(".env.example")); // Not exact match
    }

    #[test]
    fn test_is_sensitive_file_with_path() {
        assert!(is_sensitive_file("src/config/.env"));
        assert!(is_sensitive_file("deploy/credentials.json"));
        assert!(is_sensitive_file("certs/server.key"));
    }

    #[test]
    fn test_extract_stat_number() {
        assert_eq!(
            extract_stat_number(
                " 5 files changed, 120 insertions(+), 30 deletions(-)",
                "insertion"
            ),
            Some(120)
        );
        assert_eq!(
            extract_stat_number(
                " 5 files changed, 120 insertions(+), 30 deletions(-)",
                "deletion"
            ),
            Some(30)
        );
        assert_eq!(extract_stat_number("no changes", "insertion"), None);
    }

    #[test]
    fn test_verify_result_is_pass() {
        assert!(VerifyResult::Pass.is_pass());
        assert!(!VerifyResult::Fail {
            reasons: vec!["test".to_string()]
        }
        .is_pass());
    }

    // Helper to create a step node
    fn make_step(description: &str, status: StepStatus) -> StepNode {
        StepNode {
            id: Uuid::new_v4(),
            order: 0,
            description: description.to_string(),
            status,
            verification: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            completed_at: None,
        }
    }

    // Helper to setup plan + task in mock, returns (graph as Arc<dyn GraphStore>, task_id)
    async fn setup_mock_with_task() -> (Arc<dyn GraphStore>, Uuid) {
        use crate::neo4j::mock::MockGraphStore;
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        (graph, task_id)
    }

    #[tokio::test]
    async fn test_verify_steps_all_completed() {
        let (graph, task_id) = setup_mock_with_task().await;

        // Add completed steps
        let step1 = make_step("Step 1", StepStatus::Completed);
        let step2 = make_step("Step 2", StepStatus::Completed);
        graph.create_step(task_id, &step1).await.unwrap();
        graph.create_step(task_id, &step2).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_verify_steps_with_pending() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step1 = make_step("Done step", StepStatus::Completed);
        let step2 = make_step("Pending step", StepStatus::Pending);
        graph.create_step(task_id, &step1).await.unwrap();
        graph.create_step(task_id, &step2).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Pending step"),
            "Error should mention the pending step: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_verify_steps_skipped_counts_as_ok() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step1 = make_step("Completed step", StepStatus::Completed);
        let step2 = make_step("Skipped step", StepStatus::Skipped);
        graph.create_step(task_id, &step1).await.unwrap();
        graph.create_step(task_id, &step2).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_verify_full_pass() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step = make_step("Single step", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        // Disable build and test — only check steps
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify(task_id, "/tmp").await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_verify_full_fail_steps() {
        let (graph, task_id) = setup_mock_with_task().await;

        // Add a step but leave it pending
        let step = make_step("Incomplete step", StepStatus::Pending);
        graph.create_step(task_id, &step).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify(task_id, "/tmp").await;
        assert!(!result.is_pass());
        if let VerifyResult::Fail { reasons } = result {
            assert!(reasons.iter().any(|r| r.contains("Incomplete step")));
        }
    }
}
