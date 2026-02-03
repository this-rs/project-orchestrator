//! Main orchestrator runner

use crate::neo4j::models::*;
use crate::parser::{CodeParser, ParsedFile};
use crate::plan::models::*;
use crate::plan::PlanManager;
use crate::AppState;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use walkdir::WalkDir;

use super::context::ContextBuilder;

/// Main orchestrator for coordinating AI agents
pub struct Orchestrator {
    state: AppState,
    plan_manager: Arc<PlanManager>,
    context_builder: Arc<ContextBuilder>,
    parser: Arc<RwLock<CodeParser>>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub async fn new(state: AppState) -> Result<Self> {
        let plan_manager = Arc::new(PlanManager::new(
            state.neo4j.clone(),
            state.meili.clone(),
        ));

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
        })
    }

    /// Get the plan manager
    pub fn plan_manager(&self) -> &Arc<PlanManager> {
        &self.plan_manager
    }

    /// Get the context builder
    pub fn context_builder(&self) -> &Arc<ContextBuilder> {
        &self.context_builder
    }

    /// Get the Neo4j client
    pub fn neo4j(&self) -> &crate::neo4j::client::Neo4jClient {
        &self.state.neo4j
    }

    /// Get the Meilisearch client
    pub fn meili(&self) -> &crate::meilisearch::client::MeiliClient {
        &self.state.meili
    }

    // ========================================================================
    // Sync operations
    // ========================================================================

    /// Sync a directory to the knowledge base (legacy, no project)
    pub async fn sync_directory(&self, dir_path: &Path) -> Result<SyncResult> {
        self.sync_directory_for_project(dir_path, None, None).await
    }

    /// Sync a directory to the knowledge base for a specific project
    pub async fn sync_directory_for_project(
        &self,
        dir_path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
    ) -> Result<SyncResult> {
        let project_slug = project_slug.map(|s| s.to_string());
        let mut result = SyncResult::default();

        let extensions = ["rs", "ts", "tsx", "js", "jsx", "py", "go"];

        for entry in WalkDir::new(dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();

            if !extensions.contains(&ext) {
                continue;
            }

            // Skip node_modules, target, etc.
            let path_str = path.to_string_lossy();
            if path_str.contains("node_modules")
                || path_str.contains("/target/")
                || path_str.contains("/.git/")
                || path_str.contains("__pycache__")
            {
                continue;
            }

            match self
                .sync_file_for_project(path, project_id, project_slug.as_deref())
                .await
            {
                Ok(synced) => {
                    if synced {
                        result.files_synced += 1;
                    } else {
                        result.files_skipped += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to sync {}: {}", path.display(), e);
                    result.errors += 1;
                }
            }
        }

        Ok(result)
    }

    /// Sync a single file to the knowledge base (legacy, no project)
    pub async fn sync_file(&self, path: &Path) -> Result<bool> {
        self.sync_file_for_project(path, None, None).await
    }

    /// Sync a single file to the knowledge base for a specific project
    pub async fn sync_file_for_project(
        &self,
        path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
    ) -> Result<bool> {
        let content = tokio::fs::read_to_string(path)
            .await
            .context("Failed to read file")?;

        // Check if file has changed
        let path_str = path.to_string_lossy().to_string();
        if let Some(existing) = self.state.neo4j.get_file(&path_str).await? {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            let hash = hex::encode(hasher.finalize());

            if existing.hash == hash {
                return Ok(false); // File unchanged
            }
        }

        // Parse the file
        let parsed = {
            let mut parser = self.parser.write().await;
            parser.parse_file(path, &content)?
        };

        // Store in Neo4j with project association
        self.store_parsed_file_for_project(&parsed, project_id).await?;

        // Index in Meilisearch with project association
        let mut doc = CodeParser::to_code_document(&parsed, &content);
        doc.project_id = project_id.map(|id| id.to_string());
        doc.project_slug = project_slug.map(|s| s.to_string());
        self.state.meili.index_code(&doc).await?;

        Ok(true)
    }

    /// Store a parsed file in Neo4j with project association
    async fn store_parsed_file_for_project(
        &self,
        parsed: &ParsedFile,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        // Store file node
        let file_node = FileNode {
            path: parsed.path.clone(),
            language: parsed.language.clone(),
            hash: parsed.hash.clone(),
            last_parsed: chrono::Utc::now(),
            project_id,
        };
        self.state.neo4j.upsert_file(&file_node).await?;

        // Store functions
        for func in &parsed.functions {
            self.state.neo4j.upsert_function(func).await?;
        }

        // Store structs
        for s in &parsed.structs {
            self.state.neo4j.upsert_struct(s).await?;
        }

        // Store traits
        for t in &parsed.traits {
            self.state.neo4j.upsert_trait(t).await?;
        }

        // Store enums
        for e in &parsed.enums {
            self.state.neo4j.upsert_enum(e).await?;
        }

        // Store impl blocks with relationships
        for impl_block in &parsed.impl_blocks {
            self.state.neo4j.upsert_impl(impl_block).await?;
        }

        // Store imports
        for import in &parsed.imports {
            self.state.neo4j.upsert_import(import).await?;
        }

        // Store function call relationships
        for call in &parsed.function_calls {
            self.state
                .neo4j
                .create_call_relationship(&call.caller_id, &call.callee_name)
                .await?;
        }

        Ok(())
    }

    // ========================================================================
    // Agent dispatch
    // ========================================================================

    /// Dispatch a task to an agent
    pub async fn dispatch_task(
        &self,
        task_id: Uuid,
        plan_id: Uuid,
        agent_id: &str,
    ) -> Result<String> {
        // Mark task as in progress
        self.plan_manager
            .update_task(
                task_id,
                UpdateTaskRequest {
                    status: Some(TaskStatus::InProgress),
                    assigned_to: Some(agent_id.to_string()),
                    ..Default::default()
                },
            )
            .await?;

        // Build context
        let context = self.context_builder.build_context(task_id, plan_id).await?;

        // Generate prompt
        let prompt = self.context_builder.generate_prompt(&context);

        Ok(prompt)
    }

    /// Handle task completion from an agent
    pub async fn handle_task_completion(
        &self,
        task_id: Uuid,
        success: bool,
        summary: &str,
        files_modified: &[String],
    ) -> Result<()> {
        let status = if success {
            TaskStatus::Completed
        } else {
            TaskStatus::Failed
        };

        // Update task status
        self.plan_manager
            .update_task(
                task_id,
                UpdateTaskRequest {
                    status: Some(status),
                    ..Default::default()
                },
            )
            .await?;

        // Link modified files
        if !files_modified.is_empty() {
            self.plan_manager
                .link_task_to_files(task_id, files_modified)
                .await?;
        }

        // Re-sync modified files
        for file_path in files_modified {
            let path = Path::new(file_path);
            if path.exists() {
                if let Err(e) = self.sync_file(path).await {
                    tracing::warn!("Failed to re-sync {}: {}", file_path, e);
                }
            }
        }

        tracing::info!("Task {} completed: {}", task_id, summary);
        Ok(())
    }

    // ========================================================================
    // Orchestration loop
    // ========================================================================

    /// Run the main orchestration loop
    pub async fn run_loop(&self, plan_id: Uuid) -> Result<()> {
        loop {
            // Check for next available task
            let next_task = self.plan_manager.get_next_available_task(plan_id).await?;

            match next_task {
                Some(task) => {
                    tracing::info!("Found available task: {}", task.description);
                    // In a real implementation, this would dispatch to an actual agent
                    // For now, we just log it
                }
                None => {
                    // Check if plan is complete
                    let details = self.plan_manager.get_plan_details(plan_id).await?;
                    if let Some(d) = details {
                        let all_complete = d
                            .tasks
                            .iter()
                            .all(|t| t.task.status == TaskStatus::Completed);

                        if all_complete {
                            tracing::info!("Plan {} completed!", plan_id);
                            self.plan_manager
                                .update_plan_status(plan_id, PlanStatus::Completed)
                                .await?;
                            break;
                        }
                    }

                    // Wait before checking again
                    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                }
            }
        }

        Ok(())
    }
}

/// Result of a sync operation
#[derive(Debug, Default)]
pub struct SyncResult {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub errors: usize,
}
