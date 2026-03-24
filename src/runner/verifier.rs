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
    /// SHA of HEAD before the agent started — if set, verify that new commits exist
    base_commit: Option<String>,
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
            base_commit: None,
        }
    }

    /// Create a TaskVerifier that checks for new commits since `base_sha`.
    pub fn with_base_commit(
        graph: Arc<dyn GraphStore>,
        build_check_enabled: bool,
        test_runner_enabled: bool,
        base_sha: String,
    ) -> Self {
        Self {
            graph,
            build_check_enabled,
            test_runner_enabled,
            command_timeout: Duration::from_secs(600),
            base_commit: Some(base_sha),
        }
    }

    /// Run all verifications for a completed task.
    ///
    /// Returns `VerifyResult::Pass` if all checks pass, or
    /// `VerifyResult::Fail` with reasons if any check fails.
    pub async fn verify(&self, task_id: Uuid, cwd: &str) -> VerifyResult {
        let mut reasons = Vec::new();
        let cwd_path = Path::new(cwd);

        // 0. Verify the agent produced commits (critical — prevents ghost completions)
        if let Err(e) = self.verify_has_commits(cwd_path).await {
            reasons.push(format!("No code produced: {}", e));
        }

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

    /// Verify the agent produced at least one new commit since the base commit.
    ///
    /// This prevents "ghost completions" where the agent updates MCP statuses
    /// but produces zero code changes. If `base_commit` is None (legacy callers),
    /// we fall back to checking for uncommitted changes (dirty worktree).
    async fn verify_has_commits(&self, cwd: &Path) -> Result<()> {
        if let Some(ref base_sha) = self.base_commit {
            // Count commits since base
            let output = tokio::process::Command::new("git")
                .args(["rev-list", "--count", &format!("{}..HEAD", base_sha)])
                .current_dir(cwd)
                .output()
                .await;

            match output {
                Ok(o) if o.status.success() => {
                    let count_str = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    let count: usize = count_str.parse().unwrap_or(0);
                    if count == 0 {
                        // Also check for uncommitted changes — agent might have written
                        // code but forgotten to commit
                        let has_changes = self.has_uncommitted_changes(cwd).await;
                        if has_changes {
                            return Err(anyhow::anyhow!(
                                "Agent wrote code but did NOT commit (0 new commits since {}, \
                                 but working tree has uncommitted changes). \
                                 The agent must run `git add` + `git commit`.",
                                &base_sha[..8]
                            ));
                        }
                        return Err(anyhow::anyhow!(
                            "Agent produced 0 new commits since {} — no code was written. \
                             Task cannot be marked as completed without code changes.",
                            &base_sha[..8]
                        ));
                    }
                    info!(
                        "Agent produced {} commit(s) since {}",
                        count,
                        &base_sha[..base_sha.len().min(8)]
                    );
                    Ok(())
                }
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    warn!("git rev-list failed (non-fatal): {}", stderr);
                    Ok(()) // Don't fail on git errors — might be a shallow clone
                }
                Err(e) => {
                    warn!("Failed to run git rev-list: {}", e);
                    Ok(())
                }
            }
        } else {
            // No base commit — skip this check (legacy verifier callers)
            Ok(())
        }
    }

    /// Check if the working tree has uncommitted changes.
    async fn has_uncommitted_changes(&self, cwd: &Path) -> bool {
        let output = tokio::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(cwd)
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                !stdout.trim().is_empty()
            }
            _ => false,
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
            execution_context: None,
            persona: None,
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

    #[tokio::test]
    async fn test_verify_has_commits_detects_ghost_completion() {
        // Create a temp git repo with a single commit
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();

        // Init repo with one commit
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::fs::write(cwd.join("file.txt"), "initial").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(cwd)
            .output()
            .unwrap();

        // Get the base SHA
        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(cwd)
            .output()
            .unwrap();
        let base_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();

        // Create a verifier with base_commit — NO new commits exist
        let (graph, task_id) = setup_mock_with_task().await;
        let step = make_step("Step 1", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify_has_commits(cwd).await;

        // Should fail — 0 new commits
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("0 new commits"),
            "Error should mention 0 new commits: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_verify_has_commits_passes_with_new_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();

        // Init repo with one commit
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::fs::write(cwd.join("file.txt"), "initial").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(cwd)
            .output()
            .unwrap();
        let base_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();

        // Make a NEW commit after the base
        std::fs::write(cwd.join("new_file.rs"), "fn main() {}").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "feat: new feature"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify_has_commits(cwd).await;

        // Should pass — 1 new commit
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_verify_without_base_commit_skips_check() {
        let (graph, _task_id) = setup_mock_with_task().await;

        // No base_commit — should always pass (legacy behavior)
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_has_commits(Path::new("/tmp")).await;
        assert!(result.is_ok());
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_project_language_detect_python_setup_py() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("setup.py"), "from setuptools import setup").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Python);
    }

    #[test]
    fn test_project_language_test_commands() {
        // Rust
        let (cmd, args) = ProjectLanguage::Rust.test_command().unwrap();
        assert_eq!(cmd, "cargo");
        assert!(args.contains(&"test"));

        // Node
        let (cmd, args) = ProjectLanguage::Node.test_command().unwrap();
        assert_eq!(cmd, "npm");
        assert!(args.contains(&"test"));

        // Go
        let (cmd, args) = ProjectLanguage::Go.test_command().unwrap();
        assert_eq!(cmd, "go");
        assert!(args.contains(&"test"));

        // Python
        let (cmd, args) = ProjectLanguage::Python.test_command().unwrap();
        assert_eq!(cmd, "python");
        assert!(args.contains(&"pytest"));

        // Unknown
        assert!(ProjectLanguage::Unknown.test_command().is_none());
    }

    #[test]
    fn test_project_language_build_command_values() {
        let (cmd, args) = ProjectLanguage::Rust.build_command().unwrap();
        assert_eq!(cmd, "cargo");
        assert_eq!(args, &["check"]);

        let (cmd, args) = ProjectLanguage::Node.build_command().unwrap();
        assert_eq!(cmd, "npm");
        assert_eq!(args, &["run", "build"]);

        let (cmd, args) = ProjectLanguage::Go.build_command().unwrap();
        assert_eq!(cmd, "go");
        assert_eq!(args, &["build", "./..."]);
    }

    #[test]
    fn test_is_sensitive_file_remaining_patterns() {
        // Exact filename patterns not yet tested
        assert!(is_sensitive_file("secrets.yaml"));
        assert!(is_sensitive_file("secrets.yml"));
        assert!(is_sensitive_file("service-account.json"));

        // Extension patterns not yet fully tested
        assert!(is_sensitive_file("keystore.pfx"));

        // With directory paths
        assert!(is_sensitive_file("deploy/secrets.yaml"));
        assert!(is_sensitive_file("config/secrets.yml"));
        assert!(is_sensitive_file("gcp/service-account.json"));
        assert!(is_sensitive_file("certs/client.pfx"));
    }

    #[test]
    fn test_is_sensitive_file_negative_cases() {
        assert!(!is_sensitive_file("Cargo.toml"));
        assert!(!is_sensitive_file("package.json"));
        assert!(!is_sensitive_file("src/main.rs"));
        assert!(!is_sensitive_file("secrets_test.go")); // Not exact match
        assert!(!is_sensitive_file("my-env-file.txt")); // Not .env
        assert!(!is_sensitive_file(".envrc")); // Not exact match
    }

    #[test]
    fn test_extract_stat_number_edge_cases() {
        // Single insertion
        assert_eq!(
            extract_stat_number(" 1 file changed, 1 insertion(+)", "insertion"),
            Some(1)
        );

        // No number before keyword (keyword at start)
        assert_eq!(extract_stat_number("insertion(+)", "insertion"), None);

        // Empty string
        assert_eq!(extract_stat_number("", "insertion"), None);

        // Keyword not present
        assert_eq!(
            extract_stat_number(" 5 files changed, 120 foobar", "insertion"),
            None
        );

        // Non-numeric value before keyword
        assert_eq!(extract_stat_number("abc insertions(+)", "insertion"), None);

        // Extract file count
        assert_eq!(
            extract_stat_number(
                " 5 files changed, 120 insertions(+), 30 deletions(-)",
                "files"
            ),
            Some(5)
        );
    }

    #[test]
    fn test_verify_result_fail_reasons() {
        let result = VerifyResult::Fail {
            reasons: vec!["build failed".to_string(), "tests failed".to_string()],
        };
        assert!(!result.is_pass());

        // Verify Debug is implemented
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("build failed"));
        assert!(debug_str.contains("tests failed"));

        // Verify Clone is implemented
        let cloned = result.clone();
        assert!(!cloned.is_pass());
    }

    #[test]
    fn test_task_verifier_construction() {
        use crate::neo4j::mock::MockGraphStore;

        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());

        // Test new()
        let v = TaskVerifier::new(graph.clone(), true, true);
        assert!(v.build_check_enabled);
        assert!(v.test_runner_enabled);
        assert!(v.base_commit.is_none());
        assert_eq!(v.command_timeout, Duration::from_secs(600));

        // Test with_base_commit()
        let v2 =
            TaskVerifier::with_base_commit(graph.clone(), false, true, "abc123def456".to_string());
        assert!(!v2.build_check_enabled);
        assert!(v2.test_runner_enabled);
        assert_eq!(v2.base_commit, Some("abc123def456".to_string()));

        // Test new() with both disabled
        let v3 = TaskVerifier::new(graph, false, false);
        assert!(!v3.build_check_enabled);
        assert!(!v3.test_runner_enabled);
    }

    #[tokio::test]
    async fn test_verify_steps_empty_steps_passes() {
        let (graph, task_id) = setup_mock_with_task().await;

        // No steps at all — should pass (0 incomplete)
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_verify_steps_in_progress_fails() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step = make_step("WIP step", StepStatus::InProgress);
        graph.create_step(task_id, &step).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("WIP step"),
            "Error should mention InProgress step: {}",
            err
        );
        assert!(err.contains("InProgress"), "Should mention status: {}", err);
    }

    #[tokio::test]
    async fn test_verify_steps_mixed_statuses() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step1 = make_step("Done", StepStatus::Completed);
        let step2 = make_step("Skipped", StepStatus::Skipped);
        let step3 = make_step("Still pending", StepStatus::Pending);
        let step4 = make_step("Still in progress", StepStatus::InProgress);
        graph.create_step(task_id, &step1).await.unwrap();
        graph.create_step(task_id, &step2).await.unwrap();
        graph.create_step(task_id, &step3).await.unwrap();
        graph.create_step(task_id, &step4).await.unwrap();

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_steps(task_id).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should report 2 incomplete steps
        assert!(
            err.contains("2 step(s)"),
            "Should count 2 failures: {}",
            err
        );
        assert!(err.contains("Still pending"));
        assert!(err.contains("Still in progress"));
    }

    #[tokio::test]
    async fn test_verify_full_with_no_steps_passes() {
        let (graph, task_id) = setup_mock_with_task().await;

        // No steps, build/test disabled — should pass
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify(task_id, "/tmp").await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_verify_has_commits_detects_uncommitted_changes() {
        // Create a temp git repo with one commit, then leave dirty working tree
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::fs::write(cwd.join("file.txt"), "initial").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(cwd)
            .output()
            .unwrap();
        let base_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();

        // Create uncommitted changes (dirty working tree, no new commits)
        std::fs::write(cwd.join("new_file.rs"), "fn main() {}").unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify_has_commits(cwd).await;

        // Should fail with specific message about uncommitted changes
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("uncommitted changes"),
            "Should mention uncommitted changes: {}",
            err
        );
        assert!(
            err.contains("git add"),
            "Should hint about git add: {}",
            err
        );
    }

    #[test]
    fn test_sensitive_patterns_and_extensions_constants() {
        // Verify the constant arrays contain expected entries
        assert!(SENSITIVE_PATTERNS.contains(&".env"));
        assert!(SENSITIVE_PATTERNS.contains(&"credentials.json"));
        assert!(SENSITIVE_PATTERNS.contains(&"service-account.json"));
        assert!(SENSITIVE_PATTERNS.contains(&"secrets.yaml"));
        assert!(SENSITIVE_PATTERNS.contains(&"secrets.yml"));

        assert!(SENSITIVE_EXTENSIONS.contains(&".key"));
        assert!(SENSITIVE_EXTENSIONS.contains(&".pem"));
        assert!(SENSITIVE_EXTENSIONS.contains(&".p12"));
        assert!(SENSITIVE_EXTENSIONS.contains(&".pfx"));
        assert!(SENSITIVE_EXTENSIONS.contains(&".secret"));
    }

    #[test]
    fn test_project_language_debug_clone_eq() {
        let lang = ProjectLanguage::Rust;
        let cloned = lang.clone();
        assert_eq!(lang, cloned);

        // Verify Debug
        let debug = format!("{:?}", ProjectLanguage::Unknown);
        assert_eq!(debug, "Unknown");

        // All variants are Eq
        assert_eq!(ProjectLanguage::Node, ProjectLanguage::Node);
        assert_ne!(ProjectLanguage::Rust, ProjectLanguage::Go);
    }

    #[test]
    fn test_project_language_detect_priority() {
        // When both Cargo.toml and package.json exist, Rust should win
        // (it's checked first)
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(tmp.path().join("package.json"), "{}").unwrap();
        assert_eq!(ProjectLanguage::detect(tmp.path()), ProjectLanguage::Rust);
    }

    // ========================================================================
    // Git sanity check tests
    // ========================================================================

    /// Helper: create a temp git repo with one commit and return (tempdir, base_sha)
    fn init_git_repo_with_commit() -> (tempfile::TempDir, String) {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::fs::write(cwd.join("file.txt"), "initial").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(cwd)
            .output()
            .unwrap();
        let base_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();

        (tmp, base_sha)
    }

    /// Helper: add and commit a file in an existing git repo
    fn commit_file(cwd: &Path, name: &str, content: &str, msg: &str) {
        std::fs::write(cwd.join(name), content).unwrap();
        std::process::Command::new("git")
            .args(["add", name])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", msg])
            .current_dir(cwd)
            .output()
            .unwrap();
    }

    #[tokio::test]
    async fn test_verify_git_sanity_clean_commit() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Add a second commit with a non-sensitive file
        commit_file(cwd, "src_main.rs", "fn main() {}", "feat: add main");

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        assert!(
            result.is_ok(),
            "Clean commit should pass git sanity: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_git_sanity_sensitive_file_env() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Commit a sensitive file
        commit_file(cwd, ".env", "SECRET_KEY=abc123", "add env");

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Sensitive files"),
            "Should mention sensitive files: {}",
            err
        );
        assert!(err.contains(".env"), "Should mention .env: {}", err);
    }

    #[tokio::test]
    async fn test_verify_git_sanity_sensitive_file_key() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        commit_file(
            cwd,
            "server.key",
            "-----BEGIN RSA PRIVATE KEY-----",
            "add key",
        );

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("server.key"));
    }

    #[tokio::test]
    async fn test_verify_git_sanity_sensitive_file_credentials_json() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        commit_file(cwd, "credentials.json", "{\"key\":\"secret\"}", "add creds");

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("credentials.json"));
    }

    #[tokio::test]
    async fn test_verify_git_sanity_not_a_git_repo() {
        // A temp dir that is NOT a git repo
        let tmp = tempfile::tempdir().unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(tmp.path()).await;
        // Should not fail — we gracefully skip when not a git repo
        assert!(
            result.is_ok(),
            "Non-git dir should not fail: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_git_sanity_single_commit_repo() {
        // A repo with only one commit — HEAD~1 doesn't exist
        let (tmp, _base_sha) = init_git_repo_with_commit();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(tmp.path()).await;
        // HEAD~1 fails on single-commit repo; should be handled gracefully
        assert!(
            result.is_ok(),
            "Single commit repo should be handled: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_git_sanity_multiple_sensitive_files() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Commit multiple sensitive files in one commit
        std::fs::write(cwd.join(".env.local"), "DB_PASS=x").unwrap();
        std::fs::write(cwd.join("secrets.yaml"), "key: value").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(cwd)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "add secrets"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains(".env.local") || err.contains("secrets.yaml"));
    }

    // ========================================================================
    // Build verification tests (using shell scripts as fake build tools)
    // ========================================================================

    #[tokio::test]
    async fn test_verify_build_unknown_language_skips() {
        let tmp = tempfile::tempdir().unwrap();
        // No project files — Unknown language, build_command returns None

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, true, false);
        let result = verifier.verify_build(tmp.path()).await;
        assert!(
            result.is_ok(),
            "Unknown language should skip build: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_build_python_skips() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("pyproject.toml"), "[build-system]").unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, true, false);
        let result = verifier.verify_build(tmp.path()).await;
        assert!(
            result.is_ok(),
            "Python build should skip (no universal build): {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_tests_unknown_language_skips() {
        let tmp = tempfile::tempdir().unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, true);
        let result = verifier.verify_tests(tmp.path()).await;
        assert!(
            result.is_ok(),
            "Unknown language should skip tests: {:?}",
            result.err()
        );
    }

    // ========================================================================
    // has_uncommitted_changes tests
    // ========================================================================

    #[tokio::test]
    async fn test_has_uncommitted_changes_clean_repo() {
        let (tmp, _base_sha) = init_git_repo_with_commit();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let has_changes = verifier.has_uncommitted_changes(tmp.path()).await;
        assert!(
            !has_changes,
            "Clean repo should have no uncommitted changes"
        );
    }

    #[tokio::test]
    async fn test_has_uncommitted_changes_dirty_repo() {
        let (tmp, _base_sha) = init_git_repo_with_commit();

        // Create an untracked file
        std::fs::write(tmp.path().join("new.rs"), "fn new() {}").unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let has_changes = verifier.has_uncommitted_changes(tmp.path()).await;
        assert!(has_changes, "Dirty repo should have uncommitted changes");
    }

    #[tokio::test]
    async fn test_has_uncommitted_changes_modified_file() {
        let (tmp, _base_sha) = init_git_repo_with_commit();

        // Modify an existing tracked file
        std::fs::write(tmp.path().join("file.txt"), "modified content").unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let has_changes = verifier.has_uncommitted_changes(tmp.path()).await;
        assert!(
            has_changes,
            "Modified tracked file should count as uncommitted"
        );
    }

    #[tokio::test]
    async fn test_has_uncommitted_changes_not_git_repo() {
        let tmp = tempfile::tempdir().unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let has_changes = verifier.has_uncommitted_changes(tmp.path()).await;
        // Not a git repo — git status fails, returns false
        assert!(!has_changes, "Non-git dir should return false");
    }

    // ========================================================================
    // Full verify() integration tests with various flag combos
    // ========================================================================

    #[tokio::test]
    async fn test_verify_full_with_build_enabled_unknown_lang() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step = make_step("Done", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let tmp = tempfile::tempdir().unwrap();
        // No project files — Unknown language, build skipped
        let verifier = TaskVerifier::new(graph, true, false);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(
            result.is_pass(),
            "Unknown lang build should be skipped: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_verify_full_with_test_enabled_unknown_lang() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step = make_step("Done", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let verifier = TaskVerifier::new(graph, false, true);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(
            result.is_pass(),
            "Unknown lang test should be skipped: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_verify_full_with_both_enabled_unknown_lang() {
        let (graph, task_id) = setup_mock_with_task().await;

        let step = make_step("Done", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let verifier = TaskVerifier::new(graph, true, true);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_verify_full_collects_multiple_failures() {
        let (graph, task_id) = setup_mock_with_task().await;

        // Leave steps pending — this will cause a step failure
        let step = make_step("Pending step", StepStatus::Pending);
        graph.create_step(task_id, &step).await.unwrap();

        // Use a git repo with sensitive files for git sanity failure
        let (tmp, _base_sha) = init_git_repo_with_commit();
        commit_file(tmp.path(), ".env", "SECRET=x", "add env");

        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(!result.is_pass());
        if let VerifyResult::Fail { reasons } = result {
            // Should have at least step failure and git sanity failure
            assert!(
                reasons.len() >= 2,
                "Expected at least 2 failures, got {}: {:?}",
                reasons.len(),
                reasons
            );
            assert!(reasons.iter().any(|r| r.contains("Step completion")));
            assert!(reasons.iter().any(|r| r.contains("Git sanity")));
        } else {
            panic!("Expected Fail variant");
        }
    }

    #[tokio::test]
    async fn test_verify_full_with_base_commit_and_new_commits_passes() {
        let (tmp, base_sha) = init_git_repo_with_commit();
        commit_file(tmp.path(), "new.rs", "fn new() {}", "feat: new");

        let (graph, task_id) = setup_mock_with_task().await;
        let step = make_step("Done", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(result.is_pass());
    }

    #[tokio::test]
    async fn test_verify_full_with_base_commit_no_new_commits_fails() {
        let (tmp, base_sha) = init_git_repo_with_commit();

        let (graph, task_id) = setup_mock_with_task().await;
        let step = make_step("Done", StepStatus::Completed);
        graph.create_step(task_id, &step).await.unwrap();

        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify(task_id, tmp.path().to_str().unwrap()).await;
        assert!(!result.is_pass());
        if let VerifyResult::Fail { reasons } = result {
            assert!(reasons.iter().any(|r| r.contains("No code produced")));
        }
    }

    // ========================================================================
    // extract_stat_number additional edge cases
    // ========================================================================

    #[test]
    fn test_extract_stat_number_large_numbers() {
        assert_eq!(
            extract_stat_number(" 3 files changed, 15000 insertions(+)", "insertion"),
            Some(15000)
        );
        assert_eq!(
            extract_stat_number(" 1 file changed, 99999 deletions(-)", "deletion"),
            Some(99999)
        );
    }

    #[test]
    fn test_extract_stat_number_single_word_line() {
        assert_eq!(extract_stat_number("insertions", "insertion"), None);
    }

    #[test]
    fn test_extract_stat_number_keyword_at_index_zero() {
        // keyword at index 0, so i-1 would underflow
        assert_eq!(
            extract_stat_number("insertions(+) 5 files", "insertion"),
            None
        );
    }

    // ========================================================================
    // is_sensitive_file additional edge cases
    // ========================================================================

    #[test]
    fn test_is_sensitive_file_empty_string() {
        assert!(!is_sensitive_file(""));
    }

    #[test]
    fn test_is_sensitive_file_just_slash() {
        assert!(!is_sensitive_file("/"));
    }

    #[test]
    fn test_is_sensitive_file_deep_nested_path() {
        assert!(is_sensitive_file("a/b/c/d/.env"));
        assert!(is_sensitive_file("a/b/c/d/server.pem"));
        assert!(!is_sensitive_file("a/b/c/d/main.go"));
    }

    #[test]
    fn test_is_sensitive_file_env_production() {
        assert!(is_sensitive_file(".env.production"));
        assert!(is_sensitive_file("config/.env.production"));
    }

    #[test]
    fn test_is_sensitive_file_all_sensitive_extensions() {
        assert!(is_sensitive_file("test.key"));
        assert!(is_sensitive_file("test.pem"));
        assert!(is_sensitive_file("test.p12"));
        assert!(is_sensitive_file("test.pfx"));
        assert!(is_sensitive_file("test.secret"));
    }

    #[test]
    fn test_is_sensitive_file_partial_matches_negative() {
        // These should NOT match
        assert!(!is_sensitive_file(".env.example"));
        assert!(!is_sensitive_file(".envrc"));
        assert!(!is_sensitive_file("credential.json")); // singular, not credentials
        assert!(!is_sensitive_file("secrets.json")); // wrong extension
        assert!(!is_sensitive_file("my.keys")); // .keys != .key
    }

    // ========================================================================
    // VerifyResult Clone + Debug coverage
    // ========================================================================

    #[test]
    fn test_verify_result_pass_clone_debug() {
        let result = VerifyResult::Pass;
        let cloned = result.clone();
        assert!(cloned.is_pass());
        let debug = format!("{:?}", result);
        assert_eq!(debug, "Pass");
    }

    #[test]
    fn test_verify_result_fail_empty_reasons() {
        let result = VerifyResult::Fail { reasons: vec![] };
        assert!(!result.is_pass());
    }

    // ========================================================================
    // verify_has_commits edge cases
    // ========================================================================

    #[tokio::test]
    async fn test_verify_has_commits_with_multiple_new_commits() {
        let (tmp, base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Add multiple commits after base
        commit_file(cwd, "a.rs", "// a", "commit a");
        commit_file(cwd, "b.rs", "// b", "commit b");
        commit_file(cwd, "c.rs", "// c", "commit c");

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify_has_commits(cwd).await;
        assert!(
            result.is_ok(),
            "Multiple commits should pass: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_has_commits_invalid_base_sha() {
        let (tmp, _base_sha) = init_git_repo_with_commit();

        let (graph, _task_id) = setup_mock_with_task().await;
        // Use a bogus SHA that git won't recognize
        let verifier = TaskVerifier::with_base_commit(
            graph,
            false,
            false,
            "0000000000000000000000000000000000000000".to_string(),
        );
        let result = verifier.verify_has_commits(tmp.path()).await;
        // git rev-list with invalid ref should fail gracefully (non-fatal)
        assert!(
            result.is_ok(),
            "Invalid base SHA should be non-fatal: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_verify_has_commits_staged_but_uncommitted() {
        let (tmp, base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Stage a file but don't commit
        std::fs::write(cwd.join("staged.rs"), "fn staged() {}").unwrap();
        std::process::Command::new("git")
            .args(["add", "staged.rs"])
            .current_dir(cwd)
            .output()
            .unwrap();

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::with_base_commit(graph, false, false, base_sha);
        let result = verifier.verify_has_commits(cwd).await;

        // Should fail — 0 new commits, but has uncommitted changes
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("uncommitted changes"));
    }

    // ========================================================================
    // verify_git_sanity with large diff (stat parsing path)
    // ========================================================================

    #[tokio::test]
    async fn test_verify_git_sanity_large_diff_warns_but_passes() {
        let (tmp, _base_sha) = init_git_repo_with_commit();
        let cwd = tmp.path();

        // Generate a file with many lines to test the large-diff warning path
        let content: String = (0..500).map(|i| format!("line {}\n", i)).collect();
        commit_file(cwd, "big_file.txt", &content, "add big file");

        let (graph, _task_id) = setup_mock_with_task().await;
        let verifier = TaskVerifier::new(graph, false, false);
        let result = verifier.verify_git_sanity(cwd).await;
        // Large diff only warns, doesn't fail
        assert!(result.is_ok(), "Large diff should pass: {:?}", result.err());
    }
}
