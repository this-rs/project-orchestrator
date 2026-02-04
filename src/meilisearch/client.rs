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
        // Note: Only CODE and DECISIONS are used - LOGS and CONVERSATIONS were removed
        let indexes = [index_names::CODE, index_names::DECISIONS];

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
            .search_code_with_scores(query, limit, language_filter, project_slug)
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
    ) -> Result<Vec<SearchHit<CodeDocument>>> {
        let index = self.client.index(index_names::CODE);

        let mut filters = Vec::new();
        if let Some(lang) = language_filter {
            filters.push(format!("language = \"{}\"", lang));
        }
        if let Some(slug) = project_slug {
            filters.push(format!("project_slug = \"{}\"", slug));
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
            total_documents: stats.number_of_documents as usize,
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
