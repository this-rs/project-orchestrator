//! SearchStore trait implementation for MeiliClient
//!
//! Each trait method delegates directly to the corresponding inherent method
//! on `MeiliClient`.

use anyhow::Result;
use async_trait::async_trait;

use super::client::MeiliClient;
use super::indexes::{CodeDocument, DecisionDocument, IndexStats, NoteDocument, SearchHit};
use super::traits::SearchStore;

#[async_trait]
impl SearchStore for MeiliClient {
    // ========================================================================
    // Code indexing
    // ========================================================================

    async fn index_code(&self, doc: &CodeDocument) -> Result<()> {
        self.index_code(doc).await
    }

    async fn index_code_batch(&self, docs: &[CodeDocument]) -> Result<()> {
        self.index_code_batch(docs).await
    }

    async fn search_code(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        self.search_code(query, limit, language_filter).await
    }

    async fn search_code_in_project(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        self.search_code_in_project(query, limit, language_filter, project_slug)
            .await
    }

    async fn search_code_with_scores(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<SearchHit<CodeDocument>>> {
        self.search_code_with_scores(query, limit, language_filter, project_slug)
            .await
    }

    async fn delete_code(&self, path: &str) -> Result<()> {
        self.delete_code(path).await
    }

    async fn delete_code_for_project(&self, project_slug: &str) -> Result<()> {
        self.delete_code_for_project(project_slug).await
    }

    async fn delete_orphan_code_documents(&self) -> Result<()> {
        self.delete_orphan_code_documents().await
    }

    async fn get_code_stats(&self) -> Result<IndexStats> {
        self.get_code_stats().await
    }

    // ========================================================================
    // Decision indexing
    // ========================================================================

    async fn index_decision(&self, doc: &DecisionDocument) -> Result<()> {
        self.index_decision(doc).await
    }

    async fn search_decisions(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DecisionDocument>> {
        self.search_decisions(query, limit).await
    }

    async fn search_decisions_in_project(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<DecisionDocument>> {
        self.search_decisions_in_project(query, limit, project_slug)
            .await
    }

    async fn delete_decision(&self, id: &str) -> Result<()> {
        self.delete_decision(id).await
    }

    async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()> {
        self.delete_decisions_for_project(project_slug).await
    }

    async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()> {
        self.delete_decisions_for_task(task_id).await
    }

    // ========================================================================
    // Note indexing
    // ========================================================================

    async fn index_note(&self, doc: &NoteDocument) -> Result<()> {
        self.index_note(doc).await
    }

    async fn index_notes_batch(&self, docs: &[NoteDocument]) -> Result<()> {
        self.index_notes_batch(docs).await
    }

    async fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<NoteDocument>> {
        self.search_notes(query, limit).await
    }

    async fn search_notes_with_filters(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<NoteDocument>> {
        self.search_notes_with_filters(query, limit, project_slug, note_type, status, importance)
            .await
    }

    async fn search_notes_with_scores(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<SearchHit<NoteDocument>>> {
        self.search_notes_with_scores(query, limit, project_slug, note_type, status, importance)
            .await
    }

    async fn delete_note(&self, id: &str) -> Result<()> {
        self.delete_note(id).await
    }

    async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()> {
        self.delete_notes_for_project(project_slug).await
    }

    async fn update_note_status(&self, id: &str, status: &str) -> Result<()> {
        self.update_note_status(id, status).await
    }

    async fn get_notes_stats(&self) -> Result<IndexStats> {
        self.get_notes_stats().await
    }
}
