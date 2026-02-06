//! Trait abstraction for Meilisearch search operations

use super::indexes::*;
use anyhow::Result;
use async_trait::async_trait;

/// Trait abstracting all Meilisearch search and indexing operations.
///
/// This trait covers every concrete (non-generic) public async method on `MeiliClient`,
/// excluding `new`, `init_indexes`, and the two generic helpers (`search<T>`,
/// `index_document<T>`) which are not dyn-compatible.
#[async_trait]
pub trait SearchStore: Send + Sync {
    // ========================================================================
    // Code indexing
    // ========================================================================

    /// Index a code document
    async fn index_code(&self, doc: &CodeDocument) -> Result<()>;

    /// Index multiple code documents
    async fn index_code_batch(&self, docs: &[CodeDocument]) -> Result<()>;

    /// Search code
    async fn search_code(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
    ) -> Result<Vec<CodeDocument>>;

    /// Search code within a specific project
    async fn search_code_in_project(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<CodeDocument>>;

    /// Search code with ranking scores
    async fn search_code_with_scores(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<SearchHit<CodeDocument>>>;

    /// Delete code document by path
    async fn delete_code(&self, path: &str) -> Result<()>;

    /// Delete all code documents for a project
    async fn delete_code_for_project(&self, project_slug: &str) -> Result<()>;

    /// Delete orphan code documents (documents without project_id or with empty project_id)
    async fn delete_orphan_code_documents(&self) -> Result<()>;

    /// Get statistics for the code index
    async fn get_code_stats(&self) -> Result<IndexStats>;

    // ========================================================================
    // Decision indexing
    // ========================================================================

    /// Index a decision document
    async fn index_decision(&self, doc: &DecisionDocument) -> Result<()>;

    /// Search decisions
    async fn search_decisions(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DecisionDocument>>;

    /// Search decisions within a specific project
    async fn search_decisions_in_project(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<DecisionDocument>>;

    /// Delete a decision document by ID
    async fn delete_decision(&self, id: &str) -> Result<()>;

    /// Delete all decision documents for a project
    async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()>;

    /// Delete all decision documents for a task
    async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()>;

    // ========================================================================
    // Note indexing
    // ========================================================================

    /// Index a note document
    async fn index_note(&self, doc: &NoteDocument) -> Result<()>;

    /// Index multiple note documents
    async fn index_notes_batch(&self, docs: &[NoteDocument]) -> Result<()>;

    /// Search notes
    async fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<NoteDocument>>;

    /// Search notes with filters
    async fn search_notes_with_filters(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<NoteDocument>>;

    /// Search notes with ranking scores
    async fn search_notes_with_scores(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<SearchHit<NoteDocument>>>;

    /// Delete a note document by ID
    async fn delete_note(&self, id: &str) -> Result<()>;

    /// Delete all note documents for a project
    async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()>;

    /// Update note status in the index
    async fn update_note_status(&self, id: &str, status: &str) -> Result<()>;

    /// Get statistics for the notes index
    async fn get_notes_stats(&self) -> Result<IndexStats>;
}
