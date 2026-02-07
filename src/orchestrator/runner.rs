//! Main orchestrator runner

use crate::events::{CrudAction, CrudEvent, EntityType as EventEntityType, EventBus};
use crate::neo4j::models::*;
use crate::notes::{EntityType, NoteLifecycleManager, NoteManager};
use crate::parser::{CodeParser, ParsedFile};
use crate::plan::models::*;
use crate::plan::PlanManager;
use crate::AppState;
use anyhow::{Context, Result};
use std::collections::HashSet;
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
    note_manager: Arc<NoteManager>,
    note_lifecycle: Arc<NoteLifecycleManager>,
    event_bus: Option<Arc<EventBus>>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub async fn new(state: AppState) -> Result<Self> {
        let plan_manager = Arc::new(PlanManager::new(state.neo4j.clone(), state.meili.clone()));

        let note_manager = Arc::new(NoteManager::new(state.neo4j.clone(), state.meili.clone()));

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));
        let note_lifecycle = Arc::new(NoteLifecycleManager::new());

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
            note_manager,
            note_lifecycle,
            event_bus: None,
        })
    }

    /// Create a new orchestrator with an EventBus for CRUD notifications
    pub async fn with_event_bus(state: AppState, event_bus: Arc<EventBus>) -> Result<Self> {
        let plan_manager = Arc::new(PlanManager::with_event_bus(
            state.neo4j.clone(),
            state.meili.clone(),
            event_bus.clone(),
        ));

        let note_manager = Arc::new(NoteManager::with_event_bus(
            state.neo4j.clone(),
            state.meili.clone(),
            event_bus.clone(),
        ));

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));
        let note_lifecycle = Arc::new(NoteLifecycleManager::new());

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
            note_manager,
            note_lifecycle,
            event_bus: Some(event_bus),
        })
    }

    /// Get the event bus (if configured)
    pub fn event_bus(&self) -> Option<&Arc<EventBus>> {
        self.event_bus.as_ref()
    }

    /// Emit a CRUD event (no-op if event_bus is None)
    fn emit(&self, event: CrudEvent) {
        if let Some(bus) = &self.event_bus {
            bus.emit(event);
        }
    }

    /// Get the plan manager
    pub fn plan_manager(&self) -> &Arc<PlanManager> {
        &self.plan_manager
    }

    /// Get the context builder
    pub fn context_builder(&self) -> &Arc<ContextBuilder> {
        &self.context_builder
    }

    /// Get the graph store
    pub fn neo4j(&self) -> &dyn crate::neo4j::GraphStore {
        self.state.neo4j.as_ref()
    }

    /// Get the graph store as Arc (for sharing with ChatManager etc.)
    pub fn neo4j_arc(&self) -> Arc<dyn crate::neo4j::GraphStore> {
        self.state.neo4j.clone()
    }

    /// Get the search store
    pub fn meili(&self) -> &dyn crate::meilisearch::SearchStore {
        self.state.meili.as_ref()
    }

    /// Get the search store as Arc (for sharing with ChatManager etc.)
    pub fn meili_arc(&self) -> Arc<dyn crate::meilisearch::SearchStore> {
        self.state.meili.clone()
    }

    /// Get the note manager
    pub fn note_manager(&self) -> &Arc<NoteManager> {
        &self.note_manager
    }

    /// Get the note lifecycle manager
    pub fn note_lifecycle(&self) -> &Arc<NoteLifecycleManager> {
        &self.note_lifecycle
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
        let mut synced_paths: HashSet<String> = HashSet::new();

        // All supported languages - must match SupportedLanguage::from_extension()
        let extensions = [
            "rs", // Rust
            "ts", "tsx", "js", "jsx",  // TypeScript/JavaScript
            "py",   // Python
            "go",   // Go
            "java", // Java
            "c", "h", // C
            "cpp", "cc", "cxx", "hpp", "hxx", // C++
            "rb",  // Ruby
            "php", // PHP
            "kt", "kts",   // Kotlin
            "swift", // Swift
            "sh", "bash", // Bash
        ];

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

            // Track the path for cleanup
            synced_paths.insert(path_str.to_string());

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

        // Clean up stale files if we have a project_id
        if let Some(pid) = project_id {
            let valid_paths: Vec<String> = synced_paths.into_iter().collect();
            match self.neo4j().delete_stale_files(pid, &valid_paths).await {
                Ok((files_deleted, symbols_deleted)) => {
                    result.files_deleted = files_deleted;
                    result.symbols_deleted = symbols_deleted;
                }
                Err(e) => {
                    tracing::warn!("Failed to clean up stale files: {}", e);
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
        self.store_parsed_file_for_project(&parsed, project_id)
            .await?;

        // Index in Meilisearch only if project context is available
        if let (Some(pid), Some(slug)) = (project_id, project_slug) {
            let doc = CodeParser::to_code_document(&parsed, &pid.to_string(), slug);
            self.state.meili.index_code(&doc).await?;
        }

        // Verify notes attached to this file
        self.verify_notes_for_file(&path_str, &parsed, &content)
            .await?;

        Ok(true)
    }

    /// Verify notes attached to a file after it has been modified
    ///
    /// This checks if any notes anchored to entities in this file need
    /// status updates due to code changes.
    async fn verify_notes_for_file(
        &self,
        file_path: &str,
        parsed: &ParsedFile,
        source: &str,
    ) -> Result<()> {
        // Get all notes attached to this file
        let notes = self
            .state
            .neo4j
            .get_notes_for_entity(&EntityType::File, file_path)
            .await?;

        if notes.is_empty() {
            return Ok(());
        }

        tracing::debug!("Verifying {} notes for file: {}", notes.len(), file_path);

        // Create FileInfo from parsed data
        let file_info = NoteLifecycleManager::create_file_info(parsed, source);

        // Verify each note's anchors
        let results = self
            .note_lifecycle
            .verify_notes_for_file(&notes, &file_info);

        // Process verification results
        for result in results {
            if !result.all_valid {
                if let Some(update) = result.suggested_update {
                    // Update note status
                    self.state
                        .neo4j
                        .update_note(
                            result.note_id,
                            None,
                            None,
                            Some(update.new_status),
                            None,
                            None,
                        )
                        .await?;

                    // Update Meilisearch index
                    self.state
                        .meili
                        .update_note_status(
                            &result.note_id.to_string(),
                            &update.new_status.to_string(),
                        )
                        .await?;

                    tracing::info!(
                        "Note {} status changed to {:?}: {}",
                        result.note_id,
                        update.new_status,
                        update.reason
                    );
                }
            }
        }

        // Also verify notes attached to functions/structs in this file
        self.verify_notes_for_file_symbols(file_path, &file_info)
            .await?;

        // Verify assertion notes
        self.verify_assertions_for_file(file_path, &file_info)
            .await?;

        Ok(())
    }

    /// Verify assertion notes that apply to a file
    async fn verify_assertions_for_file(
        &self,
        file_path: &str,
        file_info: &crate::notes::FileInfo,
    ) -> Result<()> {
        use crate::notes::{NoteStatus, NoteType, ViolationAction};

        // Get all assertion notes that might apply to this file
        let notes = self
            .state
            .neo4j
            .get_notes_for_entity(&EntityType::File, file_path)
            .await?;

        let assertion_notes: Vec<_> = notes
            .into_iter()
            .filter(|n| n.note_type == NoteType::Assertion)
            .collect();

        if assertion_notes.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "Verifying {} assertion notes for file: {}",
            assertion_notes.len(),
            file_path
        );

        // Verify each assertion
        let results = self
            .note_lifecycle
            .verify_assertions_for_file(&assertion_notes, file_info);

        for result in results {
            if !result.passed {
                // Find the note to get the violation action
                let note = assertion_notes.iter().find(|n| n.id == result.note_id);

                if let Some(note) = note {
                    if let Some(ref rule) = note.assertion_rule {
                        match rule.on_violation {
                            ViolationAction::Warn => {
                                tracing::warn!(
                                    "Assertion failed (warning): note {} - {}",
                                    result.note_id,
                                    result.message
                                );
                            }
                            ViolationAction::FlagNote | ViolationAction::Block => {
                                // Update note status to needs_review
                                self.state
                                    .neo4j
                                    .update_note(
                                        result.note_id,
                                        None,
                                        None,
                                        Some(NoteStatus::NeedsReview),
                                        None,
                                        None,
                                    )
                                    .await?;

                                self.state
                                    .meili
                                    .update_note_status(&result.note_id.to_string(), "needs_review")
                                    .await?;

                                tracing::warn!(
                                    "Assertion failed: note {} flagged for review - {}",
                                    result.note_id,
                                    result.message
                                );
                            }
                        }
                    }
                }
            } else {
                tracing::debug!(
                    "Assertion passed: note {} - {}",
                    result.note_id,
                    result.message
                );
            }
        }

        Ok(())
    }

    /// Verify notes attached to symbols (functions, structs) within a file
    async fn verify_notes_for_file_symbols(
        &self,
        file_path: &str,
        file_info: &crate::notes::FileInfo,
    ) -> Result<()> {
        // Verify notes attached to functions
        for func in &file_info.functions {
            let func_id = format!("{}::{}", file_path, func.name);
            let notes = self
                .state
                .neo4j
                .get_notes_for_entity(&EntityType::Function, &func_id)
                .await?;

            if notes.is_empty() {
                continue;
            }

            let results = self.note_lifecycle.verify_notes_for_file(&notes, file_info);

            for result in results {
                if !result.all_valid {
                    if let Some(update) = result.suggested_update {
                        self.state
                            .neo4j
                            .update_note(
                                result.note_id,
                                None,
                                None,
                                Some(update.new_status),
                                None,
                                None,
                            )
                            .await?;

                        self.state
                            .meili
                            .update_note_status(
                                &result.note_id.to_string(),
                                &update.new_status.to_string(),
                            )
                            .await?;

                        tracing::info!(
                            "Note {} (on {}) status changed to {:?}: {}",
                            result.note_id,
                            func.name,
                            update.new_status,
                            update.reason
                        );
                    }
                }
            }
        }

        // Verify notes attached to structs
        for s in &file_info.structs {
            let struct_id = format!("{}::{}", file_path, s.name);
            let notes = self
                .state
                .neo4j
                .get_notes_for_entity(&EntityType::Struct, &struct_id)
                .await?;

            if notes.is_empty() {
                continue;
            }

            let results = self.note_lifecycle.verify_notes_for_file(&notes, file_info);

            for result in results {
                if !result.all_valid {
                    if let Some(update) = result.suggested_update {
                        self.state
                            .neo4j
                            .update_note(
                                result.note_id,
                                None,
                                None,
                                Some(update.new_status),
                                None,
                                None,
                            )
                            .await?;

                        self.state
                            .meili
                            .update_note_status(
                                &result.note_id.to_string(),
                                &update.new_status.to_string(),
                            )
                            .await?;

                        tracing::info!(
                            "Note {} (on {}) status changed to {:?}: {}",
                            result.note_id,
                            s.name,
                            update.new_status,
                            update.reason
                        );
                    }
                }
            }
        }

        Ok(())
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

        // Store imports and create File→IMPORTS→File relationships
        for import in &parsed.imports {
            self.state.neo4j.upsert_import(import).await?;

            // Try to resolve import to a file path and create relationship
            if let Some(target_file) = self.resolve_rust_import(&import.path, &parsed.path) {
                // Only create relationship if target file exists in our graph
                self.state
                    .neo4j
                    .create_import_relationship(&parsed.path, &target_file, &import.path)
                    .await
                    .ok(); // Ignore errors (target file might not exist yet)
            }
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

    /// Resolve a Rust import path to an actual file path
    ///
    /// Examples:
    /// - `crate::neo4j::client` → `src/neo4j/client.rs` or `src/neo4j/client/mod.rs`
    /// - `super::models` → parent_dir/models.rs
    /// - `self::utils` → current_dir/utils.rs
    /// - External crates (std::, serde::) → None
    fn resolve_rust_import(&self, import_path: &str, source_file: &str) -> Option<String> {
        // Skip external crates (no :: prefix or starts with known external)
        let path = import_path.split("::").collect::<Vec<_>>();
        if path.is_empty() {
            return None;
        }

        let first = path[0];

        // External crates - skip
        if !matches!(first, "crate" | "super" | "self") {
            return None;
        }

        // Get the source file's directory
        let source_path = Path::new(source_file);
        let source_dir = source_path.parent()?;

        // Find the project root (where src/ is)
        let mut project_root = source_dir;
        while !project_root.join("Cargo.toml").exists() {
            project_root = project_root.parent()?;
            // Safety limit
            if project_root.as_os_str().is_empty() {
                return None;
            }
        }

        let src_dir = project_root.join("src");

        // Build the target path based on import type
        let target_path = match first {
            "crate" => {
                // crate::foo::bar → src/foo/bar.rs or src/foo/bar/mod.rs
                if path.len() < 2 {
                    return None;
                }
                let module_path = &path[1..path.len().saturating_sub(1)]; // Exclude the final item (might be a type)
                if module_path.is_empty() {
                    return None;
                }
                let mut target = src_dir.clone();
                for part in module_path {
                    target = target.join(part);
                }
                target
            }
            "super" => {
                // super::foo → ../foo.rs relative to current file
                let mut target = source_dir.to_path_buf();
                for part in &path[1..path.len().saturating_sub(1)] {
                    if *part == "super" {
                        target = target.parent()?.to_path_buf();
                    } else {
                        target = target.join(part);
                    }
                }
                target
            }
            "self" => {
                // self::foo → ./foo.rs relative to current file's module
                let mut target = source_dir.to_path_buf();
                for part in &path[1..path.len().saturating_sub(1)] {
                    target = target.join(part);
                }
                target
            }
            _ => return None,
        };

        // Try .rs file first, then mod.rs
        let rs_file = target_path.with_extension("rs");
        if rs_file.exists() {
            return Some(rs_file.to_string_lossy().to_string());
        }

        let mod_file = target_path.join("mod.rs");
        if mod_file.exists() {
            return Some(mod_file.to_string_lossy().to_string());
        }

        // Also try without removing the last segment (in case it's a module not a type)
        let full_path = match first {
            "crate" => {
                let module_path = &path[1..];
                let mut target = src_dir;
                for part in module_path {
                    target = target.join(part);
                }
                target
            }
            _ => return None,
        };

        let rs_file = full_path.with_extension("rs");
        if rs_file.exists() {
            return Some(rs_file.to_string_lossy().to_string());
        }

        let mod_file = full_path.join("mod.rs");
        if mod_file.exists() {
            return Some(mod_file.to_string_lossy().to_string());
        }

        None
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

    // ========================================================================
    // CRUD wrappers — mutation + event emission
    // ========================================================================

    // --- Projects ---

    /// Create a project and emit event
    pub async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        self.neo4j().create_project(project).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Project, CrudAction::Created, project.id.to_string())
                .with_payload(serde_json::json!({"name": &project.name, "slug": &project.slug})),
        );
        Ok(())
    }

    /// Update a project and emit event
    pub async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_project(id, name, description, root_path).await?;
        self.emit(CrudEvent::new(EventEntityType::Project, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a project and emit event
    pub async fn delete_project(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_project(id).await?;
        self.emit(CrudEvent::new(EventEntityType::Project, CrudAction::Deleted, id.to_string()));
        Ok(())
    }

    // --- Plans (link/unlink only — CRUD is in PlanManager) ---

    /// Link a plan to a project and emit event
    pub async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        self.neo4j().link_plan_to_project(plan_id, project_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Plan, CrudAction::Linked, plan_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    /// Unlink a plan from its project and emit event
    pub async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        self.neo4j().unlink_plan_from_project(plan_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Plan, CrudAction::Unlinked, plan_id.to_string()));
        Ok(())
    }

    // --- Task dependencies ---

    /// Add a task dependency and emit event
    pub async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.neo4j().add_task_dependency(task_id, depends_on_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Task, CrudAction::Linked, task_id.to_string())
                .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a task dependency and emit event
    pub async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.neo4j().remove_task_dependency(task_id, depends_on_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Task, CrudAction::Unlinked, task_id.to_string())
                .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    // --- Steps/Decisions/Constraints (delete + update only — create is in PlanManager) ---

    /// Delete a step and emit event
    pub async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        self.neo4j().delete_step(step_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Step, CrudAction::Deleted, step_id.to_string()));
        Ok(())
    }

    /// Update a decision and emit event
    pub async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_decision(decision_id, description, rationale, chosen_option).await?;
        self.emit(CrudEvent::new(EventEntityType::Decision, CrudAction::Updated, decision_id.to_string()));
        Ok(())
    }

    /// Delete a decision and emit event
    pub async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        self.neo4j().delete_decision(decision_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Decision, CrudAction::Deleted, decision_id.to_string()));
        Ok(())
    }

    /// Update a constraint and emit event
    pub async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_constraint(constraint_id, description, constraint_type, enforced_by).await?;
        self.emit(CrudEvent::new(EventEntityType::Constraint, CrudAction::Updated, constraint_id.to_string()));
        Ok(())
    }

    /// Delete a constraint and emit event
    pub async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        self.neo4j().delete_constraint(constraint_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Constraint, CrudAction::Deleted, constraint_id.to_string()));
        Ok(())
    }

    // --- Commits ---

    /// Create a commit and emit event
    pub async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        self.neo4j().create_commit(commit).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Created, &commit.hash)
                .with_payload(serde_json::json!({"message": &commit.message})),
        );
        Ok(())
    }

    /// Link a commit to a task and emit event
    pub async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        self.neo4j().link_commit_to_task(commit_hash, task_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Linked, commit_hash)
                .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Link a commit to a plan and emit event
    pub async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        self.neo4j().link_commit_to_plan(commit_hash, plan_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Linked, commit_hash)
                .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    // --- Releases ---

    /// Create a release and emit event
    pub async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        self.neo4j().create_release(release).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Release, CrudAction::Created, release.id.to_string())
                .with_payload(serde_json::json!({"version": &release.version}))
                .with_project_id(release.project_id.to_string()),
        );
        Ok(())
    }

    /// Update a release and emit event
    pub async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_release(id, status, target_date, released_at, title, description).await?;
        self.emit(CrudEvent::new(EventEntityType::Release, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a release and emit event
    pub async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        self.neo4j().delete_release(release_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Release, CrudAction::Deleted, release_id.to_string()));
        Ok(())
    }

    /// Add a task to a release and emit event
    pub async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()> {
        self.neo4j().add_task_to_release(release_id, task_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Release, CrudAction::Linked, release_id.to_string())
                .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Add a commit to a release and emit event
    pub async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()> {
        self.neo4j().add_commit_to_release(release_id, commit_hash).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Release, CrudAction::Linked, release_id.to_string())
                .with_payload(serde_json::json!({"commit_hash": commit_hash})),
        );
        Ok(())
    }

    // --- Milestones ---

    /// Create a milestone and emit event
    pub async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        self.neo4j().create_milestone(milestone).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Milestone, CrudAction::Created, milestone.id.to_string())
                .with_payload(serde_json::json!({"title": &milestone.title}))
                .with_project_id(milestone.project_id.to_string()),
        );
        Ok(())
    }

    /// Update a milestone and emit event
    pub async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_milestone(id, status, target_date, closed_at, title, description).await?;
        self.emit(CrudEvent::new(EventEntityType::Milestone, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a milestone and emit event
    pub async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        self.neo4j().delete_milestone(milestone_id).await?;
        self.emit(CrudEvent::new(EventEntityType::Milestone, CrudAction::Deleted, milestone_id.to_string()));
        Ok(())
    }

    /// Add a task to a milestone and emit event
    pub async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        self.neo4j().add_task_to_milestone(milestone_id, task_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Milestone, CrudAction::Linked, milestone_id.to_string())
                .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    // --- Workspaces ---

    /// Create a workspace and emit event
    pub async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        self.neo4j().create_workspace(workspace).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Workspace, CrudAction::Created, workspace.id.to_string())
                .with_payload(serde_json::json!({"name": &workspace.name, "slug": &workspace.slug})),
        );
        Ok(())
    }

    /// Update a workspace and emit event
    pub async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.neo4j().update_workspace(id, name, description, metadata).await?;
        self.emit(CrudEvent::new(EventEntityType::Workspace, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a workspace and emit event
    pub async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_workspace(id).await?;
        self.emit(CrudEvent::new(EventEntityType::Workspace, CrudAction::Deleted, id.to_string()));
        Ok(())
    }

    /// Add a project to a workspace and emit event
    pub async fn add_project_to_workspace(&self, workspace_id: Uuid, project_id: Uuid) -> Result<()> {
        self.neo4j().add_project_to_workspace(workspace_id, project_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Workspace, CrudAction::Linked, workspace_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a project from a workspace and emit event
    pub async fn remove_project_from_workspace(&self, workspace_id: Uuid, project_id: Uuid) -> Result<()> {
        self.neo4j().remove_project_from_workspace(workspace_id, project_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Workspace, CrudAction::Unlinked, workspace_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    // --- Workspace Milestones ---

    /// Create a workspace milestone and emit event
    pub async fn create_workspace_milestone(&self, milestone: &WorkspaceMilestoneNode) -> Result<()> {
        self.neo4j().create_workspace_milestone(milestone).await?;
        self.emit(
            CrudEvent::new(EventEntityType::WorkspaceMilestone, CrudAction::Created, milestone.id.to_string())
                .with_payload(serde_json::json!({"title": &milestone.title})),
        );
        Ok(())
    }

    /// Update a workspace milestone and emit event
    pub async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        self.neo4j().update_workspace_milestone(id, title, description, status, target_date).await?;
        self.emit(CrudEvent::new(EventEntityType::WorkspaceMilestone, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a workspace milestone and emit event
    pub async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_workspace_milestone(id).await?;
        self.emit(CrudEvent::new(EventEntityType::WorkspaceMilestone, CrudAction::Deleted, id.to_string()));
        Ok(())
    }

    /// Add a task to a workspace milestone and emit event
    pub async fn add_task_to_workspace_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        self.neo4j().add_task_to_workspace_milestone(milestone_id, task_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::WorkspaceMilestone, CrudAction::Linked, milestone_id.to_string())
                .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    // --- Resources ---

    /// Create a resource and emit event
    pub async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        self.neo4j().create_resource(resource).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Resource, CrudAction::Created, resource.id.to_string())
                .with_payload(serde_json::json!({"name": &resource.name})),
        );
        Ok(())
    }

    /// Update a resource and emit event
    pub async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j().update_resource(id, name, file_path, url, version, description).await?;
        self.emit(CrudEvent::new(EventEntityType::Resource, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a resource and emit event
    pub async fn delete_resource(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_resource(id).await?;
        self.emit(CrudEvent::new(EventEntityType::Resource, CrudAction::Deleted, id.to_string()));
        Ok(())
    }

    /// Link a project to a resource (implements) and emit event
    pub async fn link_project_implements_resource(&self, project_id: Uuid, resource_id: Uuid) -> Result<()> {
        self.neo4j().link_project_implements_resource(project_id, resource_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Resource, CrudAction::Linked, resource_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string(), "link_type": "implements"})),
        );
        Ok(())
    }

    /// Link a project to a resource (uses) and emit event
    pub async fn link_project_uses_resource(&self, project_id: Uuid, resource_id: Uuid) -> Result<()> {
        self.neo4j().link_project_uses_resource(project_id, resource_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Resource, CrudAction::Linked, resource_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string(), "link_type": "uses"})),
        );
        Ok(())
    }

    // --- Components ---

    /// Create a component and emit event
    pub async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        self.neo4j().create_component(component).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Component, CrudAction::Created, component.id.to_string())
                .with_payload(serde_json::json!({"name": &component.name})),
        );
        Ok(())
    }

    /// Update a component and emit event
    pub async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        self.neo4j().update_component(id, name, description, runtime, config, tags).await?;
        self.emit(CrudEvent::new(EventEntityType::Component, CrudAction::Updated, id.to_string()));
        Ok(())
    }

    /// Delete a component and emit event
    pub async fn delete_component(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_component(id).await?;
        self.emit(CrudEvent::new(EventEntityType::Component, CrudAction::Deleted, id.to_string()));
        Ok(())
    }

    /// Add a component dependency and emit event
    pub async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        self.neo4j().add_component_dependency(component_id, depends_on_id, protocol, required).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Component, CrudAction::Linked, component_id.to_string())
                .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a component dependency and emit event
    pub async fn remove_component_dependency(&self, component_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.neo4j().remove_component_dependency(component_id, depends_on_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Component, CrudAction::Unlinked, component_id.to_string())
                .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Map a component to a project and emit event
    pub async fn map_component_to_project(&self, component_id: Uuid, project_id: Uuid) -> Result<()> {
        self.neo4j().map_component_to_project(component_id, project_id).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Component, CrudAction::Linked, component_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }
}

/// Result of a sync operation
#[derive(Debug, Default)]
pub struct SyncResult {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub symbols_deleted: usize,
    pub errors: usize,
}
