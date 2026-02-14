//! Meilisearch client for search operations

use super::indexes::*;
use anyhow::{Context, Result};
use meilisearch_sdk::{client::Client, indexes::Index, search::SearchResults, settings::Settings};
use serde::{de::DeserializeOwned, Serialize};

/// Client for Meilisearch operations
pub struct MeiliClient {
    client: Client,
}

impl MeiliClient {
    /// Check if Meilisearch is reachable and healthy.
    pub async fn is_healthy(&self) -> bool {
        match self.client.health().await {
            Ok(h) if h.status == "available" => true,
            _ => false,
        }
    }

    /// Create a new Meilisearch client
    pub async fn new(url: &str, api_key: &str) -> Result<Self> {
        let client =
            Client::new(url, Some(api_key)).context("Failed to create Meilisearch client")?;

        let meili = Self { client };
        meili.init_indexes().await?;

        Ok(meili)
    }

    /// Initialize all required indexes
    async fn init_indexes(&self) -> Result<()> {
        // Create indexes if they don't exist
        let indexes = [
            index_names::CODE,
            index_names::DECISIONS,
            index_names::NOTES,
        ];

        for index_name in indexes {
            let task = self
                .client
                .create_index(index_name, Some("id"))
                .await
                .context(format!("Failed to create index {}", index_name))?;

            // Wait for index creation
            task.wait_for_completion(&self.client, None, None).await?;
        }

        // Configure code index
        self.configure_code_index().await?;

        // Configure decisions index
        self.configure_decisions_index().await?;

        // Configure notes index
        self.configure_notes_index().await?;

        Ok(())
    }

    /// Configure the code index settings
    async fn configure_code_index(&self) -> Result<()> {
        let index = self.client.index(index_names::CODE);

        let settings = Settings::new()
            .with_searchable_attributes([
                "symbols",    // Function/struct/trait names (highest priority)
                "docstrings", // Documentation for semantic search
                "signatures", // Function signatures
                "path",       // File path
                "imports",    // Import paths
            ])
            .with_filterable_attributes(["language", "path", "project_id", "project_slug"])
            .with_sortable_attributes(["path"]);

        let task = index.set_settings(&settings).await?;
        task.wait_for_completion(&self.client, None, None).await?;

        Ok(())
    }

    /// Configure the decisions index settings
    async fn configure_decisions_index(&self) -> Result<()> {
        let index = self.client.index(index_names::DECISIONS);

        let settings = Settings::new()
            .with_searchable_attributes(["description", "rationale", "tags"])
            .with_filterable_attributes([
                "task_id",
                "agent",
                "timestamp",
                "project_id",
                "project_slug",
            ])
            .with_sortable_attributes(["timestamp"]);

        let task = index.set_settings(&settings).await?;
        task.wait_for_completion(&self.client, None, None).await?;

        Ok(())
    }

    /// Configure the notes index settings
    async fn configure_notes_index(&self) -> Result<()> {
        let index = self.client.index(index_names::NOTES);

        let settings = Settings::new()
            .with_searchable_attributes([
                "content",         // Main note content (highest priority)
                "tags",            // Tags for categorization
                "scope_path",      // Scope path (e.g., function name)
                "anchor_entities", // Linked entities
            ])
            .with_filterable_attributes([
                "project_id",
                "project_slug",
                "note_type",
                "status",
                "importance",
                "scope_type",
                "tags",
                "created_by",
                "staleness_score",
            ])
            .with_sortable_attributes(["created_at", "staleness_score", "importance"]);

        let task = index.set_settings(&settings).await?;
        task.wait_for_completion(&self.client, None, None).await?;

        Ok(())
    }

    /// Get an index by name
    pub fn index(&self, name: &str) -> Index {
        self.client.index(name)
    }

    // ========================================================================
    // Code indexing
    // ========================================================================

    /// Index a code document
    pub async fn index_code(&self, doc: &CodeDocument) -> Result<()> {
        let index = self.client.index(index_names::CODE);
        let task = index.add_documents(&[doc], Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Index multiple code documents
    pub async fn index_code_batch(&self, docs: &[CodeDocument]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let index = self.client.index(index_names::CODE);
        let task = index.add_documents(docs, Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Search code
    pub async fn search_code(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        self.search_code_in_project(query, limit, language_filter, None)
            .await
    }

    /// Search code within a specific project
    pub async fn search_code_in_project(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        let hits = self
            .search_code_with_scores(query, limit, language_filter, project_slug, None)
            .await?;
        Ok(hits.into_iter().map(|h| h.document).collect())
    }

    /// Search code with ranking scores
    pub async fn search_code_with_scores(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit<CodeDocument>>> {
        let index = self.client.index(index_names::CODE);

        let mut filters = Vec::new();
        if let Some(lang) = language_filter {
            filters.push(format!("language = \"{}\"", lang));
        }
        if let Some(slug) = project_slug {
            filters.push(format!("project_slug = \"{}\"", slug));
        }
        if let Some(prefix) = path_prefix {
            filters.push(format!("path STARTS WITH \"{}\"", prefix));
        }

        let filter_str = if filters.is_empty() {
            None
        } else {
            Some(filters.join(" AND "))
        };

        let mut search = index.search();
        search
            .with_query(query)
            .with_limit(limit)
            .with_show_ranking_score(true);

        if let Some(ref filter) = filter_str {
            search.with_filter(filter);
        }

        let results: SearchResults<CodeDocument> = search.execute().await?;
        Ok(results
            .hits
            .into_iter()
            .map(|h| SearchHit {
                document: h.result,
                score: h.ranking_score.unwrap_or(0.0),
            })
            .collect())
    }

    /// Delete code document by path
    pub async fn delete_code(&self, path: &str) -> Result<()> {
        let index = self.client.index(index_names::CODE);
        let id = Self::path_to_id(path);
        let task = index.delete_document(&id).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Delete all code documents for a project
    pub async fn delete_code_for_project(&self, project_slug: &str) -> Result<()> {
        use meilisearch_sdk::documents::DocumentDeletionQuery;

        let index = self.client.index(index_names::CODE);
        let mut query = DocumentDeletionQuery::new(&index);
        let filter = format!("project_slug = \"{}\"", project_slug);
        query.with_filter(&filter);

        let task = index.delete_documents_with(&query).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Delete orphan code documents (documents without project_id or with empty project_id)
    pub async fn delete_orphan_code_documents(&self) -> Result<()> {
        use meilisearch_sdk::documents::DocumentDeletionQuery;

        let index = self.client.index(index_names::CODE);
        let mut query = DocumentDeletionQuery::new(&index);
        // Delete documents where project_id is empty or not set
        query.with_filter("project_id IS EMPTY OR project_slug IS EMPTY");

        let task = index.delete_documents_with(&query).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Get statistics for the code index
    pub async fn get_code_stats(&self) -> Result<IndexStats> {
        let index = self.client.index(index_names::CODE);
        let stats = index.get_stats().await?;
        Ok(IndexStats {
            total_documents: stats.number_of_documents,
            is_indexing: stats.is_indexing,
        })
    }

    // ========================================================================
    // Decision indexing
    // ========================================================================

    /// Index a decision document
    pub async fn index_decision(&self, doc: &DecisionDocument) -> Result<()> {
        let index = self.client.index(index_names::DECISIONS);
        let task = index.add_documents(&[doc], Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Search decisions
    pub async fn search_decisions(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DecisionDocument>> {
        self.search_decisions_in_project(query, limit, None).await
    }

    /// Search decisions within a specific project
    pub async fn search_decisions_in_project(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<DecisionDocument>> {
        let index = self.client.index(index_names::DECISIONS);

        let filter_str = project_slug.map(|slug| format!("project_slug = \"{}\"", slug));

        let mut search = index.search();
        search.with_query(query).with_limit(limit);

        if let Some(ref filter) = filter_str {
            search.with_filter(filter);
        }

        let results: SearchResults<DecisionDocument> = search.execute().await?;
        Ok(results.hits.into_iter().map(|h| h.result).collect())
    }

    /// Delete a decision document by ID
    pub async fn delete_decision(&self, id: &str) -> Result<()> {
        let index = self.client.index(index_names::DECISIONS);
        let task = index.delete_document(id).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Delete all decision documents for a project
    pub async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()> {
        use meilisearch_sdk::documents::DocumentDeletionQuery;

        let index = self.client.index(index_names::DECISIONS);
        let mut query = DocumentDeletionQuery::new(&index);
        let filter = format!("project_slug = \"{}\"", project_slug);
        query.with_filter(&filter);

        let task = index.delete_documents_with(&query).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Delete all decision documents for a task
    pub async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()> {
        use meilisearch_sdk::documents::DocumentDeletionQuery;

        let index = self.client.index(index_names::DECISIONS);
        let mut query = DocumentDeletionQuery::new(&index);
        let filter = format!("task_id = \"{}\"", task_id);
        query.with_filter(&filter);

        let task = index.delete_documents_with(&query).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    // ========================================================================
    // Note indexing
    // ========================================================================

    /// Index a note document
    pub async fn index_note(&self, doc: &NoteDocument) -> Result<()> {
        let index = self.client.index(index_names::NOTES);
        let task = index.add_documents(&[doc], Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Index multiple note documents
    pub async fn index_notes_batch(&self, docs: &[NoteDocument]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let index = self.client.index(index_names::NOTES);
        let task = index.add_documents(docs, Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Search notes
    pub async fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<NoteDocument>> {
        self.search_notes_with_filters(query, limit, None, None, None, None)
            .await
    }

    /// Search notes with filters
    pub async fn search_notes_with_filters(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<NoteDocument>> {
        let hits = self
            .search_notes_with_scores(query, limit, project_slug, note_type, status, importance)
            .await?;
        Ok(hits.into_iter().map(|h| h.document).collect())
    }

    /// Search notes with ranking scores
    pub async fn search_notes_with_scores(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<SearchHit<NoteDocument>>> {
        let index = self.client.index(index_names::NOTES);

        let mut filters = Vec::new();

        if let Some(slug) = project_slug {
            filters.push(format!("project_slug = \"{}\"", slug));
        }
        if let Some(nt) = note_type {
            filters.push(format!("note_type = \"{}\"", nt));
        }
        if let Some(s) = status {
            filters.push(format!("status = \"{}\"", s));
        } else {
            // By default, only search active and needs_review notes
            filters.push("status IN [\"active\", \"needs_review\"]".to_string());
        }
        if let Some(imp) = importance {
            filters.push(format!("importance = \"{}\"", imp));
        }

        let filter_str = if filters.is_empty() {
            None
        } else {
            Some(filters.join(" AND "))
        };

        let mut search = index.search();
        search
            .with_query(query)
            .with_limit(limit)
            .with_show_ranking_score(true);

        if let Some(ref filter) = filter_str {
            search.with_filter(filter);
        }

        let results: SearchResults<NoteDocument> = search.execute().await?;
        Ok(results
            .hits
            .into_iter()
            .map(|h| SearchHit {
                document: h.result,
                score: h.ranking_score.unwrap_or(0.0),
            })
            .collect())
    }

    /// Delete a note document by ID
    pub async fn delete_note(&self, id: &str) -> Result<()> {
        let index = self.client.index(index_names::NOTES);
        let task = index.delete_document(id).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Delete all note documents for a project
    pub async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()> {
        use meilisearch_sdk::documents::DocumentDeletionQuery;

        let index = self.client.index(index_names::NOTES);
        let mut query = DocumentDeletionQuery::new(&index);
        let filter = format!("project_slug = \"{}\"", project_slug);
        query.with_filter(&filter);

        let task = index.delete_documents_with(&query).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    /// Update note status in the index
    pub async fn update_note_status(&self, id: &str, status: &str) -> Result<()> {
        // Meilisearch uses add_documents with same ID to update
        // We need to fetch the document first, update it, and re-index
        // For simplicity, we'll use a partial update approach
        let index = self.client.index(index_names::NOTES);

        // Fetch the existing document
        let doc: Option<NoteDocument> = index.get_document(id).await.ok();

        if let Some(mut doc) = doc {
            doc.status = status.to_string();
            let task = index.add_documents(&[doc], Some("id")).await?;
            task.wait_for_completion(&self.client, None, None).await?;
        }

        Ok(())
    }

    /// Get statistics for the notes index
    pub async fn get_notes_stats(&self) -> Result<IndexStats> {
        let index = self.client.index(index_names::NOTES);
        let stats = index.get_stats().await?;
        Ok(IndexStats {
            total_documents: stats.number_of_documents,
            is_indexing: stats.is_indexing,
        })
    }

    // ========================================================================
    // Generic operations
    // ========================================================================

    /// Search any index
    pub async fn search<T: DeserializeOwned + Send + Sync + 'static>(
        &self,
        index_name: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<T>> {
        let index = self.client.index(index_name);

        let results: SearchResults<T> = index
            .search()
            .with_query(query)
            .with_limit(limit)
            .execute()
            .await?;

        Ok(results.hits.into_iter().map(|h| h.result).collect())
    }

    /// Index any document
    pub async fn index_document<T: Serialize + Send + Sync>(
        &self,
        index_name: &str,
        doc: &T,
    ) -> Result<()> {
        let index = self.client.index(index_name);
        let task = index.add_documents(&[doc], Some("id")).await?;
        task.wait_for_completion(&self.client, None, None).await?;
        Ok(())
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    /// Convert a file path to a document ID
    pub fn path_to_id(path: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(path.as_bytes());
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_id_consistent() {
        let path = "src/main.rs";
        let id1 = MeiliClient::path_to_id(path);
        let id2 = MeiliClient::path_to_id(path);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_path_to_id_different_paths() {
        let id1 = MeiliClient::path_to_id("src/main.rs");
        let id2 = MeiliClient::path_to_id("src/lib.rs");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_path_to_id_is_hex() {
        let id = MeiliClient::path_to_id("test.rs");
        // SHA256 hex is 64 characters
        assert_eq!(id.len(), 64);
        // All characters should be valid hex
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_path_to_id_handles_special_chars() {
        let id = MeiliClient::path_to_id("src/path with spaces/file.rs");
        assert_eq!(id.len(), 64);

        let id2 = MeiliClient::path_to_id("src/über/файл.rs");
        assert_eq!(id2.len(), 64);
    }

    #[test]
    fn test_path_to_id_empty_path() {
        let id = MeiliClient::path_to_id("");
        assert_eq!(id.len(), 64);
    }

    #[test]
    fn test_index_names_constants() {
        assert_eq!(index_names::CODE, "code");
        assert_eq!(index_names::DECISIONS, "decisions");
        assert_eq!(index_names::NOTES, "notes");
    }
}
