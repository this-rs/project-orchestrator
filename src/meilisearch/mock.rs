//! In-memory mock implementation of SearchStore for testing without a real Meilisearch instance.

use super::indexes::*;
use super::traits::SearchStore;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::RwLock;

/// In-memory mock implementation of SearchStore for testing.
///
/// Stores documents in `Vec`s behind async `RwLock`s. Search operations use
/// simple substring matching; scores are 1.0 for exact matches and 0.5 for
/// substring matches.
pub struct MockSearchStore {
    code_documents: RwLock<Vec<CodeDocument>>,
    decision_documents: RwLock<Vec<DecisionDocument>>,
    note_documents: RwLock<Vec<NoteDocument>>,
}

impl MockSearchStore {
    /// Create a new empty mock store.
    pub fn new() -> Self {
        Self {
            code_documents: RwLock::new(Vec::new()),
            decision_documents: RwLock::new(Vec::new()),
            note_documents: RwLock::new(Vec::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a score based on how the query matches the text.
/// - 1.0 if the text equals the query (case-insensitive)
/// - 0.5 if the text contains the query as a substring (case-insensitive)
/// - 0.0 (no match)
fn match_score(text: &str, query: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    if text_lower == query_lower {
        1.0
    } else if text_lower.contains(&query_lower) {
        0.5
    } else {
        0.0
    }
}

/// Check if any of the given fields contain the query as a substring (case-insensitive).
fn any_field_matches(fields: &[&str], query: &str) -> bool {
    let query_lower = query.to_lowercase();
    fields
        .iter()
        .any(|f| f.to_lowercase().contains(&query_lower))
}

/// Best score across multiple text fields.
fn best_score(fields: &[&str], query: &str) -> f64 {
    fields
        .iter()
        .map(|f| match_score(f, query))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// SearchStore implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl SearchStore for MockSearchStore {
    // ======================================================================
    // Code
    // ======================================================================

    async fn index_code(&self, doc: &CodeDocument) -> Result<()> {
        let mut docs = self.code_documents.write().await;
        if let Some(existing) = docs.iter_mut().find(|d| d.path == doc.path) {
            *existing = doc.clone();
        } else {
            docs.push(doc.clone());
        }
        Ok(())
    }

    async fn index_code_batch(&self, docs: &[CodeDocument]) -> Result<()> {
        for doc in docs {
            self.index_code(doc).await?;
        }
        Ok(())
    }

    async fn search_code(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        let docs = self.code_documents.read().await;
        let results: Vec<CodeDocument> = docs
            .iter()
            .filter(|d| {
                if let Some(lang) = language_filter {
                    if !d.language.eq_ignore_ascii_case(lang) {
                        return false;
                    }
                }
                let symbols_text = d.symbols.join(" ");
                any_field_matches(&[&symbols_text, &d.docstrings, &d.path], query)
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn search_code_in_project(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        let docs = self.code_documents.read().await;
        let results: Vec<CodeDocument> = docs
            .iter()
            .filter(|d| {
                if let Some(slug) = project_slug {
                    if d.project_slug != slug {
                        return false;
                    }
                }
                if let Some(lang) = language_filter {
                    if !d.language.eq_ignore_ascii_case(lang) {
                        return false;
                    }
                }
                let symbols_text = d.symbols.join(" ");
                any_field_matches(&[&symbols_text, &d.docstrings, &d.path], query)
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn search_code_with_scores(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit<CodeDocument>>> {
        let docs = self.code_documents.read().await;
        let mut results: Vec<SearchHit<CodeDocument>> = docs
            .iter()
            .filter_map(|d| {
                if let Some(slug) = project_slug {
                    if d.project_slug != slug {
                        return None;
                    }
                }
                if let Some(lang) = language_filter {
                    if !d.language.eq_ignore_ascii_case(lang) {
                        return None;
                    }
                }
                if let Some(prefix) = path_prefix {
                    if !d.path.starts_with(prefix) {
                        return None;
                    }
                }
                let symbols_text = d.symbols.join(" ");
                let score = best_score(&[&symbols_text, &d.docstrings, &d.path], query);
                if score > 0.0 {
                    Some(SearchHit {
                        document: d.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        // Sort by descending score for consistent ordering
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn delete_code(&self, path: &str) -> Result<()> {
        let mut docs = self.code_documents.write().await;
        docs.retain(|d| d.path != path);
        Ok(())
    }

    async fn delete_code_for_project(&self, project_slug: &str) -> Result<()> {
        let mut docs = self.code_documents.write().await;
        docs.retain(|d| d.project_slug != project_slug);
        Ok(())
    }

    async fn delete_orphan_code_documents(&self) -> Result<()> {
        let mut docs = self.code_documents.write().await;
        docs.retain(|d| !d.project_id.is_empty());
        Ok(())
    }

    async fn get_code_stats(&self) -> Result<IndexStats> {
        let docs = self.code_documents.read().await;
        Ok(IndexStats {
            total_documents: docs.len(),
            is_indexing: false,
        })
    }

    // ======================================================================
    // Decision
    // ======================================================================

    async fn index_decision(&self, doc: &DecisionDocument) -> Result<()> {
        let mut docs = self.decision_documents.write().await;
        if let Some(existing) = docs.iter_mut().find(|d| d.id == doc.id) {
            *existing = doc.clone();
        } else {
            docs.push(doc.clone());
        }
        Ok(())
    }

    async fn search_decisions(&self, query: &str, limit: usize) -> Result<Vec<DecisionDocument>> {
        let docs = self.decision_documents.read().await;
        let results: Vec<DecisionDocument> = docs
            .iter()
            .filter(|d| any_field_matches(&[&d.description, &d.rationale], query))
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn search_decisions_in_project(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<DecisionDocument>> {
        let docs = self.decision_documents.read().await;
        let results: Vec<DecisionDocument> = docs
            .iter()
            .filter(|d| {
                if let Some(slug) = project_slug {
                    if d.project_slug.as_deref() != Some(slug) {
                        return false;
                    }
                }
                any_field_matches(&[&d.description, &d.rationale], query)
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn delete_decision(&self, id: &str) -> Result<()> {
        let mut docs = self.decision_documents.write().await;
        docs.retain(|d| d.id != id);
        Ok(())
    }

    async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()> {
        let mut docs = self.decision_documents.write().await;
        docs.retain(|d| d.project_slug.as_deref() != Some(project_slug));
        Ok(())
    }

    async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()> {
        let mut docs = self.decision_documents.write().await;
        docs.retain(|d| d.task_id != task_id);
        Ok(())
    }

    // ======================================================================
    // Note
    // ======================================================================

    async fn index_note(&self, doc: &NoteDocument) -> Result<()> {
        let mut docs = self.note_documents.write().await;
        if let Some(existing) = docs.iter_mut().find(|d| d.id == doc.id) {
            *existing = doc.clone();
        } else {
            docs.push(doc.clone());
        }
        Ok(())
    }

    async fn index_notes_batch(&self, docs: &[NoteDocument]) -> Result<()> {
        for doc in docs {
            self.index_note(doc).await?;
        }
        Ok(())
    }

    async fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<NoteDocument>> {
        let docs = self.note_documents.read().await;
        let tags_joined: Vec<String> = docs.iter().map(|d| d.tags.join(" ")).collect();
        let results: Vec<NoteDocument> = docs
            .iter()
            .enumerate()
            .filter(|(i, d)| any_field_matches(&[&d.content, &tags_joined[*i]], query))
            .map(|(_, d)| d.clone())
            .take(limit)
            .collect();
        Ok(results)
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
        let docs = self.note_documents.read().await;
        let tags_joined: Vec<String> = docs.iter().map(|d| d.tags.join(" ")).collect();
        let results: Vec<NoteDocument> = docs
            .iter()
            .enumerate()
            .filter(|(i, d)| {
                if let Some(slug) = project_slug {
                    if d.project_slug != slug {
                        return false;
                    }
                }
                if let Some(nt) = note_type {
                    if d.note_type != nt {
                        return false;
                    }
                }
                if let Some(s) = status {
                    if d.status != s {
                        return false;
                    }
                }
                if let Some(imp) = importance {
                    if d.importance != imp {
                        return false;
                    }
                }
                any_field_matches(&[&d.content, &tags_joined[*i]], query)
            })
            .map(|(_, d)| d.clone())
            .take(limit)
            .collect();
        Ok(results)
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
        let docs = self.note_documents.read().await;
        let mut results: Vec<SearchHit<NoteDocument>> = docs
            .iter()
            .filter_map(|d| {
                if let Some(slug) = project_slug {
                    if d.project_slug != slug {
                        return None;
                    }
                }
                if let Some(nt) = note_type {
                    if d.note_type != nt {
                        return None;
                    }
                }
                if let Some(s) = status {
                    if d.status != s {
                        return None;
                    }
                }
                if let Some(imp) = importance {
                    if d.importance != imp {
                        return None;
                    }
                }
                let tags_text = d.tags.join(" ");
                let score = best_score(&[&d.content, &tags_text], query);
                if score > 0.0 {
                    Some(SearchHit {
                        document: d.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn delete_note(&self, id: &str) -> Result<()> {
        let mut docs = self.note_documents.write().await;
        docs.retain(|d| d.id != id);
        Ok(())
    }

    async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()> {
        let mut docs = self.note_documents.write().await;
        docs.retain(|d| d.project_slug != project_slug);
        Ok(())
    }

    async fn update_note_status(&self, id: &str, status: &str) -> Result<()> {
        let mut docs = self.note_documents.write().await;
        if let Some(doc) = docs.iter_mut().find(|d| d.id == id) {
            doc.status = status.to_string();
        }
        Ok(())
    }

    async fn get_notes_stats(&self) -> Result<IndexStats> {
        let docs = self.note_documents.read().await;
        Ok(IndexStats {
            total_documents: docs.len(),
            is_indexing: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_code_doc(path: &str, project_slug: &str) -> CodeDocument {
        CodeDocument {
            id: path.replace('/', "_"),
            path: path.to_string(),
            language: "rust".to_string(),
            symbols: vec!["main".to_string(), "Config".to_string()],
            docstrings: "Main entry point for the application".to_string(),
            signatures: vec!["fn main()".to_string()],
            imports: vec!["std::io".to_string()],
            project_id: "proj-1".to_string(),
            project_slug: project_slug.to_string(),
        }
    }

    fn sample_decision_doc(id: &str, task_id: &str) -> DecisionDocument {
        DecisionDocument {
            id: id.to_string(),
            description: "Use async/await for all IO operations".to_string(),
            rationale: "Better performance under load".to_string(),
            task_id: task_id.to_string(),
            agent: "claude".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            tags: vec!["architecture".to_string()],
            project_id: Some("proj-1".to_string()),
            project_slug: Some("my-project".to_string()),
        }
    }

    fn sample_note_doc(id: &str, project_slug: &str) -> NoteDocument {
        NoteDocument {
            id: id.to_string(),
            project_id: "proj-1".to_string(),
            project_slug: project_slug.to_string(),
            note_type: "guideline".to_string(),
            status: "active".to_string(),
            importance: "high".to_string(),
            scope_type: "file".to_string(),
            scope_path: "src/main.rs".to_string(),
            content: "Always handle errors with proper context".to_string(),
            tags: vec!["error-handling".to_string(), "best-practice".to_string()],
            anchor_entities: vec!["file:src/main.rs".to_string()],
            created_at: 1704067200,
            created_by: "claude".to_string(),
            staleness_score: 0.0,
        }
    }

    // ------------------------------------------------------------------
    // Code tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_index_and_search_code() {
        let store = MockSearchStore::new();
        let doc = sample_code_doc("src/main.rs", "my-project");
        store.index_code(&doc).await.unwrap();

        let results = store.search_code("main", 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_index_code_upsert() {
        let store = MockSearchStore::new();
        let mut doc = sample_code_doc("src/main.rs", "my-project");
        store.index_code(&doc).await.unwrap();

        doc.symbols = vec!["run".to_string()];
        store.index_code(&doc).await.unwrap();

        let stats = store.get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);

        let results = store.search_code("run", 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_code_language_filter() {
        let store = MockSearchStore::new();
        store
            .index_code(&sample_code_doc("src/main.rs", "proj"))
            .await
            .unwrap();

        let results = store.search_code("main", 10, Some("python")).await.unwrap();
        assert!(results.is_empty());

        let results = store.search_code("main", 10, Some("rust")).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_code_in_project() {
        let store = MockSearchStore::new();
        store
            .index_code(&sample_code_doc("src/a.rs", "alpha"))
            .await
            .unwrap();
        store
            .index_code(&sample_code_doc("src/b.rs", "beta"))
            .await
            .unwrap();

        let results = store
            .search_code_in_project("main", 10, None, Some("alpha"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_slug, "alpha");
    }

    #[tokio::test]
    async fn test_search_code_with_scores() {
        let store = MockSearchStore::new();
        store
            .index_code(&sample_code_doc("src/main.rs", "proj"))
            .await
            .unwrap();

        let hits = store
            .search_code_with_scores("main", 10, None, None, None)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_delete_code() {
        let store = MockSearchStore::new();
        store
            .index_code(&sample_code_doc("src/main.rs", "proj"))
            .await
            .unwrap();
        store.delete_code("src/main.rs").await.unwrap();

        let stats = store.get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 0);
    }

    #[tokio::test]
    async fn test_delete_code_for_project() {
        let store = MockSearchStore::new();
        store
            .index_code(&sample_code_doc("src/a.rs", "alpha"))
            .await
            .unwrap();
        store
            .index_code(&sample_code_doc("src/b.rs", "beta"))
            .await
            .unwrap();
        store.delete_code_for_project("alpha").await.unwrap();

        let stats = store.get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);
    }

    #[tokio::test]
    async fn test_delete_orphan_code_documents() {
        let store = MockSearchStore::new();
        let mut orphan = sample_code_doc("src/orphan.rs", "proj");
        orphan.project_id = String::new();
        store.index_code(&orphan).await.unwrap();
        store
            .index_code(&sample_code_doc("src/ok.rs", "proj"))
            .await
            .unwrap();

        store.delete_orphan_code_documents().await.unwrap();

        let stats = store.get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);
    }

    #[tokio::test]
    async fn test_index_code_batch() {
        let store = MockSearchStore::new();
        let docs = vec![
            sample_code_doc("src/a.rs", "proj"),
            sample_code_doc("src/b.rs", "proj"),
        ];
        store.index_code_batch(&docs).await.unwrap();

        let stats = store.get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 2);
    }

    // ------------------------------------------------------------------
    // Decision tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_index_and_search_decisions() {
        let store = MockSearchStore::new();
        store
            .index_decision(&sample_decision_doc("d1", "t1"))
            .await
            .unwrap();

        let results = store.search_decisions("async", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_decisions_in_project() {
        let store = MockSearchStore::new();
        store
            .index_decision(&sample_decision_doc("d1", "t1"))
            .await
            .unwrap();

        let results = store
            .search_decisions_in_project("async", 10, Some("my-project"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        let results = store
            .search_decisions_in_project("async", 10, Some("other"))
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_decision() {
        let store = MockSearchStore::new();
        store
            .index_decision(&sample_decision_doc("d1", "t1"))
            .await
            .unwrap();
        store.delete_decision("d1").await.unwrap();

        let results = store.search_decisions("async", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_decisions_for_project() {
        let store = MockSearchStore::new();
        store
            .index_decision(&sample_decision_doc("d1", "t1"))
            .await
            .unwrap();
        store
            .delete_decisions_for_project("my-project")
            .await
            .unwrap();

        let results = store.search_decisions("async", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_decisions_for_task() {
        let store = MockSearchStore::new();
        store
            .index_decision(&sample_decision_doc("d1", "task-99"))
            .await
            .unwrap();
        store.delete_decisions_for_task("task-99").await.unwrap();

        let results = store.search_decisions("async", 10).await.unwrap();
        assert!(results.is_empty());
    }

    // ------------------------------------------------------------------
    // Note tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_index_and_search_notes() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "my-project"))
            .await
            .unwrap();

        let results = store.search_notes("error", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_notes_by_tag() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "my-project"))
            .await
            .unwrap();

        let results = store.search_notes("best-practice", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_notes_with_filters() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "alpha"))
            .await
            .unwrap();
        store
            .index_note(&sample_note_doc("n2", "beta"))
            .await
            .unwrap();

        let results = store
            .search_notes_with_filters("error", 10, Some("alpha"), None, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_slug, "alpha");

        let results = store
            .search_notes_with_filters("error", 10, None, Some("gotcha"), None, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_notes_with_scores() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "proj"))
            .await
            .unwrap();

        let hits = store
            .search_notes_with_scores("error", 10, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_delete_note() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "proj"))
            .await
            .unwrap();
        store.delete_note("n1").await.unwrap();

        let stats = store.get_notes_stats().await.unwrap();
        assert_eq!(stats.total_documents, 0);
    }

    #[tokio::test]
    async fn test_delete_notes_for_project() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "alpha"))
            .await
            .unwrap();
        store
            .index_note(&sample_note_doc("n2", "beta"))
            .await
            .unwrap();
        store.delete_notes_for_project("alpha").await.unwrap();

        let stats = store.get_notes_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);
    }

    #[tokio::test]
    async fn test_update_note_status() {
        let store = MockSearchStore::new();
        store
            .index_note(&sample_note_doc("n1", "proj"))
            .await
            .unwrap();

        store.update_note_status("n1", "obsolete").await.unwrap();

        let results = store
            .search_notes_with_filters("error", 10, None, None, Some("obsolete"), None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, "obsolete");
    }

    #[tokio::test]
    async fn test_index_notes_batch() {
        let store = MockSearchStore::new();
        let docs = vec![sample_note_doc("n1", "proj"), sample_note_doc("n2", "proj")];
        store.index_notes_batch(&docs).await.unwrap();

        let stats = store.get_notes_stats().await.unwrap();
        assert_eq!(stats.total_documents, 2);
    }

    #[tokio::test]
    async fn test_note_upsert() {
        let store = MockSearchStore::new();
        let mut doc = sample_note_doc("n1", "proj");
        store.index_note(&doc).await.unwrap();

        doc.content = "Updated content about concurrency".to_string();
        store.index_note(&doc).await.unwrap();

        let stats = store.get_notes_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);

        let results = store.search_notes("concurrency", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
