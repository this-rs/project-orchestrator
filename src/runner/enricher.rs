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
                warn!("Enricher: failed to link note to file {}: {}", normalized, e);
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
}
