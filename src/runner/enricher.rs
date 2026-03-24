//! TaskEnricher — Post-task automatic knowledge capture.
//!
//! After each completed task, the Enricher automatically:
//! - Links git commits to the task and plan
//! - Creates a context note summarizing the work (V1: from git log, no LLM)
//! - Anchors architectural decisions to affected files (AFFECTS)
//! - Marks modified files as discussed in the chat session
//!
//! V1 is entirely LLM-free — builds knowledge from git log + diff.
//! V2 will add LLM-enriched summaries via ChatManager oneshot mode.

use crate::neo4j::models::{CommitNode, FileChangedInfo};
use crate::neo4j::traits::GraphStore;
use crate::notes::models::{EntityType, Note, NoteType};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================================
// TaskEnricher — orchestrates post-task knowledge capture
// ============================================================================

/// Post-task knowledge enrichment engine.
///
/// Runs after each successfully completed task to capture knowledge
/// in the project graph: commits, notes, decisions, discussed entities.
pub struct TaskEnricher {
    graph: Arc<dyn GraphStore>,
}

/// Info about a git commit parsed from `git log`.
#[derive(Debug, Clone)]
struct GitCommitInfo {
    hash: String,
    message: String,
    author: String,
    timestamp: DateTime<Utc>,
    files: Vec<FileChangedInfo>,
}

/// Summary of all enrichment performed.
#[derive(Debug, Clone, Default)]
pub struct EnrichResult {
    /// Number of commits linked
    pub commits_linked: usize,
    /// Whether an auto-note was created
    pub note_created: bool,
    /// Number of AFFECTS relations added
    pub affects_added: usize,
    /// Number of entities marked as discussed
    pub discussed_added: usize,
}

impl TaskEnricher {
    /// Create a new TaskEnricher.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Run all enrichments for a completed task.
    ///
    /// This is fire-and-forget — errors are logged but don't propagate.
    pub async fn enrich(
        &self,
        task_id: Uuid,
        plan_id: Uuid,
        project_id: Option<Uuid>,
        session_id: Option<Uuid>,
        task_start_time: DateTime<Utc>,
        cwd: &str,
    ) -> EnrichResult {
        let mut result = EnrichResult::default();

        // 1. Get commits since task started
        let commits = match self.get_commits_since(cwd, task_start_time).await {
            Ok(c) => c,
            Err(e) => {
                warn!("Enricher: failed to get commits: {}", e);
                return result;
            }
        };

        if commits.is_empty() {
            info!("Enricher: no commits found for task {}", task_id);
            return result;
        }

        // Collect all modified files across all commits
        let all_modified_files: Vec<String> = commits
            .iter()
            .flat_map(|c| c.files.iter().map(|f| f.path.clone()))
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();

        // 2. Link commits to task and plan
        result.commits_linked = self
            .enrich_commits(&commits, task_id, plan_id, project_id)
            .await;

        // 3. Create auto-note from git info
        if let Some(note_id) = self
            .enrich_auto_note(&commits, &all_modified_files, task_id, project_id, cwd)
            .await
        {
            result.note_created = true;

            // Link note to task
            if let Err(e) = self
                .graph
                .link_note_to_entity(note_id, &EntityType::Task, &task_id.to_string(), None, None)
                .await
            {
                warn!("Enricher: failed to link note to task: {}", e);
            }
        }

        // 4. Anchor decisions to affected files (AFFECTS)
        result.affects_added = self
            .enrich_decision_affects(task_id, &all_modified_files)
            .await;

        // 5. Mark files as discussed in the session
        if let Some(sid) = session_id {
            result.discussed_added = self.enrich_discussed(sid, &all_modified_files).await;
        }

        info!(
            "Enricher: task {} — {} commits linked, note={}, {} affects, {} discussed",
            task_id,
            result.commits_linked,
            result.note_created,
            result.affects_added,
            result.discussed_added
        );

        result
    }

    // ========================================================================
    // Step 1: Commit linking
    // ========================================================================

    /// Register commits in the graph and link them to task + plan.
    async fn enrich_commits(
        &self,
        commits: &[GitCommitInfo],
        task_id: Uuid,
        plan_id: Uuid,
        project_id: Option<Uuid>,
    ) -> usize {
        let mut linked = 0;

        for commit in commits {
            // Create commit node
            let commit_node = CommitNode {
                hash: commit.hash.clone(),
                message: commit.message.clone(),
                author: commit.author.clone(),
                timestamp: commit.timestamp,
            };

            if let Err(e) = self.graph.create_commit(&commit_node).await {
                warn!("Enricher: failed to create commit {}: {}", commit.hash, e);
                continue;
            }

            // Create TOUCHES relations for files
            if !commit.files.is_empty() {
                if let Err(e) = self
                    .graph
                    .create_commit_touches(&commit.hash, &commit.files)
                    .await
                {
                    warn!(
                        "Enricher: failed to create touches for {}: {}",
                        commit.hash, e
                    );
                }
            }

            // Link to task
            if let Err(e) = self.graph.link_commit_to_task(&commit.hash, task_id).await {
                warn!(
                    "Enricher: failed to link commit {} to task: {}",
                    commit.hash, e
                );
            }

            // Link to plan (all commits, not just last — they're cheap)
            if let Err(e) = self.graph.link_commit_to_plan(&commit.hash, plan_id).await {
                warn!(
                    "Enricher: failed to link commit {} to plan: {}",
                    commit.hash, e
                );
            }

            // Trigger incremental sync if project_id available
            // (commit create with files_changed + project_id does this automatically
            //  via the MCP handler, but here we're calling GraphStore directly)
            let _ = project_id; // Reserved for future incremental sync call

            linked += 1;
        }

        linked
    }

    // ========================================================================
    // Step 2: Auto-note (V1: git-based, no LLM)
    // ========================================================================

    /// Create an auto-note summarizing the task's work from git info.
    ///
    /// V1: Builds the note content from git log + diff stat. No LLM call.
    /// V2 will add LLM-enriched summaries via ChatManager oneshot mode.
    async fn enrich_auto_note(
        &self,
        commits: &[GitCommitInfo],
        all_modified_files: &[String],
        task_id: Uuid,
        project_id: Option<Uuid>,
        cwd: &str,
    ) -> Option<Uuid> {
        // Build note content from git info
        let content = self
            .build_note_content(commits, all_modified_files, cwd)
            .await;

        let mut note = Note::new(
            project_id,
            NoteType::Context,
            content,
            "runner-enricher".to_string(),
        );
        note.tags = vec![
            "auto-generated".to_string(),
            "post-task".to_string(),
            format!("task:{}", task_id),
        ];

        let note_id = note.id;

        if let Err(e) = self.graph.create_note(&note).await {
            warn!("Enricher: failed to create auto-note: {}", e);
            return None;
        }

        // Link note to each modified file (defensive relativize for edge cases)
        for file_path in all_modified_files {
            let normalized = crate::utils::paths::relativize(file_path, cwd);
            if let Err(e) = self
                .graph
                .link_note_to_entity(note_id, &EntityType::File, &normalized, None, None)
                .await
            {
                warn!(
                    "Enricher: failed to link note to file {}: {}",
                    normalized, e
                );
            }
        }

        Some(note_id)
    }

    /// Build note content from git commits and diff stat.
    async fn build_note_content(
        &self,
        commits: &[GitCommitInfo],
        all_modified_files: &[String],
        cwd: &str,
    ) -> String {
        let mut content = String::new();

        content.push_str("## Task Implementation Summary (auto-generated)\n\n");

        // Commit messages
        content.push_str("### Commits\n");
        for commit in commits {
            content.push_str(&format!(
                "- `{}` {}\n",
                &commit.hash[..7.min(commit.hash.len())],
                commit.message
            ));
        }

        // Modified files
        content.push_str(&format!(
            "\n### Files Modified ({})\n",
            all_modified_files.len()
        ));
        for file in all_modified_files {
            content.push_str(&format!("- `{}`\n", file));
        }

        // Diff stat summary
        if let Ok(stat) = self.get_diff_stat(cwd, commits).await {
            content.push_str(&format!("\n### Diff Summary\n{}\n", stat));
        }

        content
    }

    /// Get diff stat from git.
    async fn get_diff_stat(&self, cwd: &str, commits: &[GitCommitInfo]) -> Result<String> {
        if commits.is_empty() {
            return Ok(String::new());
        }

        // Use range from first to last commit
        let first_hash = &commits.last().unwrap().hash; // git log returns newest first
        let last_hash = &commits.first().unwrap().hash;

        let range = if commits.len() == 1 {
            format!("{}~1..{}", first_hash, first_hash)
        } else {
            format!("{}~1..{}", first_hash, last_hash)
        };

        let output = tokio::process::Command::new("git")
            .args(["diff", "--stat", &range])
            .current_dir(cwd)
            .output()
            .await?;

        if output.status.success() {
            // Get just the summary line (last line)
            let stat_str = String::from_utf8_lossy(&output.stdout);
            if let Some(last_line) = stat_str.lines().last() {
                Ok(last_line.trim().to_string())
            } else {
                Ok(String::new())
            }
        } else {
            Ok(String::new()) // Don't fail on diff stat errors
        }
    }

    // ========================================================================
    // Step 3: Decision AFFECTS anchoring (V1: no LLM)
    // ========================================================================

    /// For decisions created during this task, add AFFECTS relations
    /// to all files modified in the diff.
    async fn enrich_decision_affects(&self, task_id: Uuid, modified_files: &[String]) -> usize {
        if modified_files.is_empty() {
            return 0;
        }

        // Get decisions created for this task
        let decisions = match self
            .graph
            .get_decisions_for_entity("task", &task_id.to_string(), 50)
            .await
        {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "Enricher: failed to get decisions for task {}: {}",
                    task_id, e
                );
                return 0;
            }
        };

        if decisions.is_empty() {
            return 0;
        }

        let mut added = 0;

        for decision in &decisions {
            // Check existing affects to avoid duplicates
            let existing: Vec<crate::neo4j::models::AffectsRelation> = self
                .graph
                .list_decision_affects(decision.id)
                .await
                .unwrap_or_default();
            let existing_ids: std::collections::HashSet<String> =
                existing.iter().map(|a| a.entity_id.clone()).collect();

            for file_path in modified_files {
                if !existing_ids.contains(file_path) {
                    if let Err(e) = self
                        .graph
                        .add_decision_affects(
                            decision.id,
                            "File",
                            file_path,
                            Some("Modified during task implementation"),
                        )
                        .await
                    {
                        warn!(
                            "Enricher: failed to add AFFECTS {}->{}: {}",
                            decision.id, file_path, e
                        );
                    } else {
                        added += 1;
                    }
                }
            }
        }

        added
    }

    // ========================================================================
    // Step 4: Mark files as discussed in the chat session
    // ========================================================================

    /// Mark all modified files as discussed in the chat session.
    async fn enrich_discussed(&self, session_id: Uuid, modified_files: &[String]) -> usize {
        if modified_files.is_empty() {
            return 0;
        }

        let entities: Vec<(String, String)> = modified_files
            .iter()
            .map(|f| ("file".to_string(), f.clone()))
            .collect();

        match self.graph.add_discussed(session_id, &entities).await {
            Ok(count) => count,
            Err(e) => {
                warn!("Enricher: failed to add discussed entities: {}", e);
                0
            }
        }
    }

    // ========================================================================
    // Post-run sweep — catch commits missed by per-task enrichment
    // ========================================================================

    /// Sweep all commits since `run_start` and link any that are orphaned.
    ///
    /// When agents produce a single mega-commit at the end of the plan (instead
    /// of per-task atomic commits), the per-task enricher finds 0 commits for
    /// T1..T(N-1). This sweep runs ONCE after the full plan completes and links
    /// every unlinked commit to the plan + assigns to tasks by file overlap.
    pub async fn post_run_sweep(
        &self,
        plan_id: Uuid,
        completed_tasks: &[Uuid],
        run_start: DateTime<Utc>,
        _project_id: Option<Uuid>,
        cwd: &str,
    ) -> usize {
        // 1. Get ALL commits since the run started
        let commits = match self.get_commits_since(cwd, run_start).await {
            Ok(c) => c,
            Err(e) => {
                warn!("Post-run sweep: failed to get commits: {}", e);
                return 0;
            }
        };

        if commits.is_empty() {
            info!("Post-run sweep: no commits found since run start");
            return 0;
        }

        let mut linked = 0;

        for commit in &commits {
            // Create commit node (idempotent — skips if already exists)
            let commit_node = CommitNode {
                hash: commit.hash.clone(),
                message: commit.message.clone(),
                author: commit.author.clone(),
                timestamp: commit.timestamp,
            };
            let _ = self.graph.create_commit(&commit_node).await;

            // Create TOUCHES relations
            if !commit.files.is_empty() {
                let _ = self
                    .graph
                    .create_commit_touches(&commit.hash, &commit.files)
                    .await;
            }

            // Link to plan (idempotent)
            let _ = self.graph.link_commit_to_plan(&commit.hash, plan_id).await;

            // Try to assign to a task by matching affected_files
            // For each completed task, check if the commit touches any of its affected files
            let commit_files: std::collections::HashSet<&str> =
                commit.files.iter().map(|f| f.path.as_str()).collect();

            for &task_id in completed_tasks {
                // Check if this task already has this commit linked
                let existing = self
                    .graph
                    .get_task_commits(task_id)
                    .await
                    .unwrap_or_default();
                if existing.iter().any(|c| c.hash == commit.hash) {
                    continue; // Already linked
                }

                // Get task's affected_files to match against commit
                if let Ok(Some(task_node)) = self.graph.get_task(task_id).await {
                    let task_files: std::collections::HashSet<&str> = task_node
                        .affected_files
                        .iter()
                        .map(|f| f.as_str())
                        .collect();

                    let overlap = commit_files.intersection(&task_files).count();
                    if overlap > 0 || task_node.affected_files.is_empty() {
                        // Match found (or task has no affected_files → link anyway)
                        let _ = self.graph.link_commit_to_task(&commit.hash, task_id).await;
                        linked += 1;
                    }
                }
            }
        }

        info!(
            "Post-run sweep: {} commits processed, {} task links created",
            commits.len(),
            linked
        );

        linked
    }

    // ========================================================================
    // Git helpers
    // ========================================================================

    /// Get all commits since a given time in the working directory.
    async fn get_commits_since(
        &self,
        cwd: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<GitCommitInfo>> {
        let since_str = since.format("%Y-%m-%dT%H:%M:%S").to_string();

        // Get commit hashes + messages + authors + timestamps
        let output = tokio::process::Command::new("git")
            .args([
                "log",
                "--format=%H|%s|%an|%aI",
                &format!("--since={}", since_str),
                "--no-merges",
            ])
            .current_dir(cwd)
            .output()
            .await;

        let output = match output {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                if stderr.contains("not a git repository")
                    || stderr.contains("dépôt git")
                    || o.status.code().is_some_and(|c| c > 1)
                {
                    return Ok(vec![]);
                }
                return Err(anyhow::anyhow!("git log failed: {}", stderr));
            }
            Err(e) => {
                warn!("Failed to run git log: {}", e);
                return Ok(vec![]);
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.splitn(4, '|').collect();
            if parts.len() < 4 {
                continue;
            }

            let hash = parts[0].to_string();
            let message = parts[1].to_string();
            let author = parts[2].to_string();
            let timestamp = chrono::DateTime::parse_from_rfc3339(parts[3])
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            // Get files changed for this commit
            let files = self.get_commit_files(cwd, &hash).await.unwrap_or_default();

            commits.push(GitCommitInfo {
                hash,
                message,
                author,
                timestamp,
                files,
            });
        }

        Ok(commits)
    }

    /// Get files changed in a specific commit with additions/deletions.
    async fn get_commit_files(&self, cwd: &str, hash: &str) -> Result<Vec<FileChangedInfo>> {
        let output = tokio::process::Command::new("git")
            .args(["diff-tree", "--no-commit-id", "-r", "--numstat", hash])
            .current_dir(cwd)
            .output()
            .await?;

        if !output.status.success() {
            return Ok(vec![]);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut files = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let additions = parts[0].parse::<i64>().ok();
                let deletions = parts[1].parse::<i64>().ok();
                let path = parts[2].to_string();

                files.push(FileChangedInfo {
                    path,
                    additions,
                    deletions,
                });
            }
        }

        Ok(files)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enrich_result_default() {
        let result = EnrichResult::default();
        assert_eq!(result.commits_linked, 0);
        assert!(!result.note_created);
        assert_eq!(result.affects_added, 0);
        assert_eq!(result.discussed_added, 0);
    }

    #[test]
    fn test_git_commit_info_struct() {
        let info = GitCommitInfo {
            hash: "abc123".to_string(),
            message: "feat: add something".to_string(),
            author: "Test User".to_string(),
            timestamp: Utc::now(),
            files: vec![FileChangedInfo {
                path: "src/main.rs".to_string(),
                additions: Some(10),
                deletions: Some(2),
            }],
        };
        assert_eq!(info.files.len(), 1);
        assert_eq!(info.files[0].additions, Some(10));
    }

    #[tokio::test]
    async fn test_build_note_content() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph);

        let commits = vec![
            GitCommitInfo {
                hash: "abc1234567890".to_string(),
                message: "feat(runner): add enricher".to_string(),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![FileChangedInfo {
                    path: "src/runner/enricher.rs".to_string(),
                    additions: Some(100),
                    deletions: Some(0),
                }],
            },
            GitCommitInfo {
                hash: "def4567890123".to_string(),
                message: "fix(runner): handle edge case".to_string(),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![FileChangedInfo {
                    path: "src/runner/runner.rs".to_string(),
                    additions: Some(5),
                    deletions: Some(2),
                }],
            },
        ];
        let modified_files = vec![
            "src/runner/enricher.rs".to_string(),
            "src/runner/runner.rs".to_string(),
        ];

        let content = enricher
            .build_note_content(&commits, &modified_files, "/tmp")
            .await;

        assert!(content.contains("## Task Implementation Summary"));
        assert!(content.contains("abc1234")); // short hash
        assert!(content.contains("feat(runner): add enricher"));
        assert!(content.contains("fix(runner): handle edge case"));
        assert!(content.contains("src/runner/enricher.rs"));
        assert!(content.contains("src/runner/runner.rs"));
        assert!(content.contains("Files Modified (2)"));
    }

    #[tokio::test]
    async fn test_enrich_commits_creates_and_links() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        // Setup plan + task
        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "abc123def456".to_string(),
            message: "feat: test commit".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![FileChangedInfo {
                path: "src/main.rs".to_string(),
                additions: Some(10),
                deletions: Some(0),
            }],
        }];

        let linked = enricher
            .enrich_commits(&commits, task_id, plan_id, None)
            .await;
        assert_eq!(linked, 1);

        // Verify the commit was created and linked
        let task_commits = graph.get_task_commits(task_id).await.unwrap();
        assert_eq!(task_commits.len(), 1);
        assert_eq!(task_commits[0].hash, "abc123def456");
    }

    #[tokio::test]
    async fn test_enrich_auto_note_creates_note() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        // Setup plan + task + project
        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "abc123def456".to_string(),
            message: "feat: something".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        }];
        let modified_files = vec!["src/main.rs".to_string()];

        let note_id = enricher
            .enrich_auto_note(&commits, &modified_files, task_id, None, "/tmp")
            .await;
        assert!(note_id.is_some());

        // Verify the note was created
        let note = graph.get_note(note_id.unwrap()).await.unwrap();
        assert!(note.is_some());
        let note = note.unwrap();
        assert_eq!(note.note_type, NoteType::Context);
        assert!(note.tags.contains(&"auto-generated".to_string()));
        assert!(note.tags.contains(&"post-task".to_string()));
        assert!(note.content.contains("abc123d"));
    }

    #[tokio::test]
    async fn test_enrich_decision_affects_links_files() {
        use crate::neo4j::models::{DecisionNode, DecisionStatus};
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Create a decision for this task
        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: "Use async for I/O".to_string(),
            rationale: "Better performance".to_string(),
            alternatives: vec![],
            chosen_option: Some("tokio async".to_string()),
            status: DecisionStatus::Accepted,
            decided_at: Utc::now(),
            decided_by: "agent".to_string(),
            scar_intensity: 0.0,
            embedding: None,
            embedding_model: None,
        };
        graph.create_decision(task_id, &decision).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let modified_files = vec![
            "src/runner/enricher.rs".to_string(),
            "src/runner/runner.rs".to_string(),
        ];

        let added = enricher
            .enrich_decision_affects(task_id, &modified_files)
            .await;
        assert_eq!(added, 2);

        // Verify AFFECTS were created
        let affects = graph.list_decision_affects(decision.id).await.unwrap();
        assert_eq!(affects.len(), 2);
    }

    #[tokio::test]
    async fn test_enrich_discussed_marks_files() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let enricher = TaskEnricher::new(graph.clone());

        let session_id = Uuid::new_v4();
        let modified_files = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];

        let count = enricher.enrich_discussed(session_id, &modified_files).await;
        // MockGraphStore may return 0 if add_discussed isn't fully implemented,
        // but the call should not error
        assert!(count <= modified_files.len());
    }

    #[tokio::test]
    async fn test_enrich_decision_affects_no_duplicates() {
        use crate::neo4j::models::{DecisionNode, DecisionStatus};
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: "Test decision".to_string(),
            rationale: String::new(),
            alternatives: vec![],
            chosen_option: None,
            status: DecisionStatus::Accepted,
            decided_at: Utc::now(),
            decided_by: "agent".to_string(),
            scar_intensity: 0.0,
            embedding: None,
            embedding_model: None,
        };
        graph.create_decision(task_id, &decision).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let files = vec!["src/main.rs".to_string()];

        // First call: should add 1
        let added1 = enricher.enrich_decision_affects(task_id, &files).await;
        assert_eq!(added1, 1);

        // Second call: should add 0 (already exists)
        let added2 = enricher.enrich_decision_affects(task_id, &files).await;
        assert_eq!(added2, 0);
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[tokio::test]
    async fn test_enrich_decision_affects_empty_files() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let empty_files: Vec<String> = vec![];

        // Should short-circuit and return 0 when no modified files
        let added = enricher
            .enrich_decision_affects(task_id, &empty_files)
            .await;
        assert_eq!(added, 0);
    }

    #[tokio::test]
    async fn test_enrich_decision_affects_no_decisions() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let files = vec!["src/main.rs".to_string()];

        // No decisions exist for this task, should return 0
        let added = enricher.enrich_decision_affects(task_id, &files).await;
        assert_eq!(added, 0);
    }

    #[tokio::test]
    async fn test_enrich_discussed_empty_files() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph.clone());

        let session_id = Uuid::new_v4();
        let empty_files: Vec<String> = vec![];

        // Should short-circuit and return 0
        let count = enricher.enrich_discussed(session_id, &empty_files).await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_enrich_discussed_returns_entity_count() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph.clone());

        let session_id = Uuid::new_v4();
        let files = vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "Cargo.toml".to_string(),
        ];

        // MockGraphStore returns entities.len() for add_discussed
        let count = enricher.enrich_discussed(session_id, &files).await;
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_build_note_content_empty_commits() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph);

        let commits: Vec<GitCommitInfo> = vec![];
        let modified_files: Vec<String> = vec![];

        let content = enricher
            .build_note_content(&commits, &modified_files, "/tmp")
            .await;

        assert!(content.contains("## Task Implementation Summary"));
        assert!(content.contains("### Commits"));
        assert!(content.contains("Files Modified (0)"));
    }

    #[tokio::test]
    async fn test_build_note_content_short_hash() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph);

        // Hash shorter than 7 characters — should not panic
        let commits = vec![GitCommitInfo {
            hash: "abc".to_string(),
            message: "short hash commit".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        }];
        let modified_files = vec!["src/foo.rs".to_string()];

        let content = enricher
            .build_note_content(&commits, &modified_files, "/tmp")
            .await;

        assert!(content.contains("`abc`"));
        assert!(content.contains("short hash commit"));
    }

    #[tokio::test]
    async fn test_build_note_content_single_file() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph);

        let commits = vec![GitCommitInfo {
            hash: "1234567890abcdef".to_string(),
            message: "feat: add feature".to_string(),
            author: "Alice".to_string(),
            timestamp: Utc::now(),
            files: vec![FileChangedInfo {
                path: "src/feature.rs".to_string(),
                additions: Some(50),
                deletions: Some(10),
            }],
        }];
        let modified_files = vec!["src/feature.rs".to_string()];

        let content = enricher
            .build_note_content(&commits, &modified_files, "/tmp")
            .await;

        assert!(content.contains("`1234567`")); // truncated to 7
        assert!(content.contains("feat: add feature"));
        assert!(content.contains("Files Modified (1)"));
        assert!(content.contains("`src/feature.rs`"));
    }

    #[tokio::test]
    async fn test_enrich_commits_multiple() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![
            GitCommitInfo {
                hash: "aaa111bbb222".to_string(),
                message: "feat: first commit".to_string(),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![FileChangedInfo {
                    path: "src/a.rs".to_string(),
                    additions: Some(10),
                    deletions: Some(0),
                }],
            },
            GitCommitInfo {
                hash: "ccc333ddd444".to_string(),
                message: "fix: second commit".to_string(),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![FileChangedInfo {
                    path: "src/b.rs".to_string(),
                    additions: Some(5),
                    deletions: Some(3),
                }],
            },
            GitCommitInfo {
                hash: "eee555fff666".to_string(),
                message: "refactor: third commit".to_string(),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![], // no files changed
            },
        ];

        let linked = enricher
            .enrich_commits(&commits, task_id, plan_id, None)
            .await;
        assert_eq!(linked, 3);

        // All three should be linked to the task
        let task_commits = graph.get_task_commits(task_id).await.unwrap();
        assert_eq!(task_commits.len(), 3);
    }

    #[tokio::test]
    async fn test_enrich_commits_with_project_id() {
        use crate::test_helpers::{test_plan, test_project, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let project = test_project();
        let project_id = project.id;
        graph.create_project(&project).await.unwrap();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "proj123456789".to_string(),
            message: "feat: with project".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        }];

        let linked = enricher
            .enrich_commits(&commits, task_id, plan_id, Some(project_id))
            .await;
        assert_eq!(linked, 1);
    }

    #[tokio::test]
    async fn test_enrich_auto_note_with_multiple_files() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "multi123456789".to_string(),
            message: "feat: multi-file change".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![
                FileChangedInfo {
                    path: "src/a.rs".to_string(),
                    additions: Some(10),
                    deletions: Some(0),
                },
                FileChangedInfo {
                    path: "src/b.rs".to_string(),
                    additions: Some(5),
                    deletions: Some(2),
                },
            ],
        }];
        let modified_files = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];

        let note_id = enricher
            .enrich_auto_note(&commits, &modified_files, task_id, None, "/tmp")
            .await;
        assert!(note_id.is_some());

        let note = graph.get_note(note_id.unwrap()).await.unwrap().unwrap();
        assert!(note.content.contains("src/a.rs"));
        assert!(note.content.contains("src/b.rs"));
        assert!(note.tags.contains(&format!("task:{}", task_id)));
    }

    #[tokio::test]
    async fn test_enrich_auto_note_with_project_id() {
        use crate::test_helpers::{test_plan, test_project, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let project = test_project();
        let project_id = project.id;
        graph.create_project(&project).await.unwrap();

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "projnote123456".to_string(),
            message: "feat: with project note".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        }];
        let modified_files = vec!["src/main.rs".to_string()];

        let note_id = enricher
            .enrich_auto_note(&commits, &modified_files, task_id, Some(project_id), "/tmp")
            .await;
        assert!(note_id.is_some());

        let note = graph.get_note(note_id.unwrap()).await.unwrap().unwrap();
        assert_eq!(note.project_id, Some(project_id));
    }

    #[tokio::test]
    async fn test_enrich_decision_affects_multiple_decisions() {
        use crate::neo4j::models::{DecisionNode, DecisionStatus};
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        // Create two decisions for the same task
        let decision1 = DecisionNode {
            id: Uuid::new_v4(),
            description: "Decision one".to_string(),
            rationale: "Reason one".to_string(),
            alternatives: vec![],
            chosen_option: None,
            status: DecisionStatus::Accepted,
            decided_at: Utc::now(),
            decided_by: "agent".to_string(),
            scar_intensity: 0.0,
            embedding: None,
            embedding_model: None,
        };
        let decision2 = DecisionNode {
            id: Uuid::new_v4(),
            description: "Decision two".to_string(),
            rationale: "Reason two".to_string(),
            alternatives: vec![],
            chosen_option: None,
            status: DecisionStatus::Accepted,
            decided_at: Utc::now(),
            decided_by: "agent".to_string(),
            scar_intensity: 0.0,
            embedding: None,
            embedding_model: None,
        };
        graph.create_decision(task_id, &decision1).await.unwrap();
        graph.create_decision(task_id, &decision2).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let files = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];

        // Each decision should get AFFECTS for each file = 2 * 2 = 4
        let added = enricher.enrich_decision_affects(task_id, &files).await;
        assert_eq!(added, 4);

        // Verify both decisions have affects
        let affects1 = graph.list_decision_affects(decision1.id).await.unwrap();
        assert_eq!(affects1.len(), 2);
        let affects2 = graph.list_decision_affects(decision2.id).await.unwrap();
        assert_eq!(affects2.len(), 2);
    }

    #[test]
    fn test_enrich_result_fields() {
        let result = EnrichResult {
            commits_linked: 5,
            note_created: true,
            affects_added: 10,
            discussed_added: 3,
        };

        assert_eq!(result.commits_linked, 5);
        assert!(result.note_created);
        assert_eq!(result.affects_added, 10);
        assert_eq!(result.discussed_added, 3);
    }

    #[test]
    fn test_enrich_result_clone() {
        let result = EnrichResult {
            commits_linked: 2,
            note_created: true,
            affects_added: 3,
            discussed_added: 1,
        };
        let cloned = result.clone();
        assert_eq!(cloned.commits_linked, 2);
        assert!(cloned.note_created);
        assert_eq!(cloned.affects_added, 3);
        assert_eq!(cloned.discussed_added, 1);
    }

    #[test]
    fn test_enrich_result_debug() {
        let result = EnrichResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("EnrichResult"));
        assert!(debug_str.contains("commits_linked"));
    }

    #[test]
    fn test_git_commit_info_clone() {
        let info = GitCommitInfo {
            hash: "abc123".to_string(),
            message: "test".to_string(),
            author: "Author".to_string(),
            timestamp: Utc::now(),
            files: vec![
                FileChangedInfo {
                    path: "a.rs".to_string(),
                    additions: Some(1),
                    deletions: None,
                },
                FileChangedInfo {
                    path: "b.rs".to_string(),
                    additions: None,
                    deletions: Some(5),
                },
            ],
        };
        let cloned = info.clone();
        assert_eq!(cloned.hash, "abc123");
        assert_eq!(cloned.files.len(), 2);
        assert_eq!(cloned.files[1].deletions, Some(5));
    }

    #[test]
    fn test_git_commit_info_debug() {
        let info = GitCommitInfo {
            hash: "xyz".to_string(),
            message: "msg".to_string(),
            author: "author".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("GitCommitInfo"));
        assert!(debug_str.contains("xyz"));
    }

    #[test]
    fn test_task_enricher_new() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let _enricher = TaskEnricher::new(graph);
        // Just verify construction does not panic
    }

    #[tokio::test]
    async fn test_enrich_auto_note_empty_modified_files() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());

        let commits = vec![GitCommitInfo {
            hash: "empty123456789".to_string(),
            message: "chore: no files".to_string(),
            author: "Test".to_string(),
            timestamp: Utc::now(),
            files: vec![],
        }];
        let modified_files: Vec<String> = vec![];

        // Should still create a note even with no modified files
        let note_id = enricher
            .enrich_auto_note(&commits, &modified_files, task_id, None, "/tmp")
            .await;
        assert!(note_id.is_some());

        let note = graph.get_note(note_id.unwrap()).await.unwrap().unwrap();
        assert!(note.content.contains("Files Modified (0)"));
    }

    #[tokio::test]
    async fn test_build_note_content_many_commits() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let enricher = TaskEnricher::new(graph);

        let commits: Vec<GitCommitInfo> = (0..5)
            .map(|i| GitCommitInfo {
                hash: format!("hash{:03}abcdef0123", i),
                message: format!("commit number {}", i),
                author: "Test".to_string(),
                timestamp: Utc::now(),
                files: vec![FileChangedInfo {
                    path: format!("src/file{}.rs", i),
                    additions: Some(i as i64 * 10),
                    deletions: Some(i as i64),
                }],
            })
            .collect();

        let modified_files: Vec<String> = (0..5).map(|i| format!("src/file{}.rs", i)).collect();

        let content = enricher
            .build_note_content(&commits, &modified_files, "/tmp")
            .await;

        // Verify all commits are listed
        for i in 0..5 {
            assert!(content.contains(&format!("commit number {}", i)));
        }
        assert!(content.contains("Files Modified (5)"));
    }

    #[tokio::test]
    async fn test_enrich_commits_empty_list() {
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let commits: Vec<GitCommitInfo> = vec![];

        let linked = enricher
            .enrich_commits(&commits, task_id, plan_id, None)
            .await;
        assert_eq!(linked, 0);
    }

    #[tokio::test]
    async fn test_enrich_decision_affects_with_existing_affects() {
        use crate::neo4j::models::{DecisionNode, DecisionStatus};
        use crate::test_helpers::{test_plan, test_task};

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let plan = test_plan();
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        let task = test_task();
        let task_id = task.id;
        graph.create_task(plan_id, &task).await.unwrap();

        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: "Test with pre-existing".to_string(),
            rationale: String::new(),
            alternatives: vec![],
            chosen_option: None,
            status: DecisionStatus::Accepted,
            decided_at: Utc::now(),
            decided_by: "agent".to_string(),
            scar_intensity: 0.0,
            embedding: None,
            embedding_model: None,
        };
        graph.create_decision(task_id, &decision).await.unwrap();

        // Pre-add an AFFECTS relation
        graph
            .add_decision_affects(decision.id, "File", "src/existing.rs", Some("pre-existing"))
            .await
            .unwrap();

        let enricher = TaskEnricher::new(graph.clone());
        let files = vec![
            "src/existing.rs".to_string(), // already exists
            "src/new.rs".to_string(),      // new
        ];

        let added = enricher.enrich_decision_affects(task_id, &files).await;
        // Only the new file should be added
        assert_eq!(added, 1);

        let affects = graph.list_decision_affects(decision.id).await.unwrap();
        assert_eq!(affects.len(), 2);
    }
}
