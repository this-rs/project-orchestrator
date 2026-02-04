//! Index definitions for Meilisearch

use serde::{Deserialize, Serialize};

/// Search result with ranking score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit<T> {
    pub document: T,
    /// Ranking score from Meilisearch (0.0 to 1.0, higher is better)
    pub score: f64,
}

/// Code document for indexing
///
/// Lightweight document for semantic search - does NOT store full file content.
/// Use Neo4j for structural queries, file system for actual code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDocument {
    pub id: String,
    pub path: String,
    pub language: String,
    /// Symbol names (functions, structs, traits, enums)
    pub symbols: Vec<String>,
    /// Concatenated docstrings for semantic search
    pub docstrings: String,
    /// Function signatures for quick reference (e.g., "fn new(url: &str) -> Result<Self>")
    pub signatures: Vec<String>,
    /// Import paths
    pub imports: Vec<String>,
    /// Project ID (required for multi-project support)
    pub project_id: String,
    /// Project slug (required for filtering)
    pub project_slug: String,
}

/// Decision document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDocument {
    pub id: String,
    pub description: String,
    pub rationale: String,
    pub task_id: String,
    pub agent: String,
    pub timestamp: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub project_slug: Option<String>,
}

/// Knowledge Note document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteDocument {
    /// Unique identifier (UUID)
    pub id: String,
    /// Project ID (UUID)
    pub project_id: String,
    /// Project slug for filtering
    pub project_slug: String,
    /// Type of note (guideline, gotcha, pattern, context, tip, observation, assertion)
    pub note_type: String,
    /// Status (active, needs_review, stale, obsolete, archived)
    pub status: String,
    /// Importance level (low, medium, high, critical)
    pub importance: String,
    /// Scope type (project, module, file, function, struct, trait)
    pub scope_type: String,
    /// Scope path (e.g., "src/auth/jwt.rs::validate_token")
    pub scope_path: String,
    /// The full content/text of the note (main searchable field)
    pub content: String,
    /// Tags for categorization and search
    pub tags: Vec<String>,
    /// Entity identifiers this note is attached to
    pub anchor_entities: Vec<String>,
    /// Unix timestamp for creation
    pub created_at: i64,
    /// Who created the note
    pub created_by: String,
    /// Staleness score (0.0 - 1.0)
    pub staleness_score: f64,
}

/// Statistics for a Meilisearch index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_documents: usize,
    pub is_indexing: bool,
}

/// Index names
pub mod index_names {
    pub const CODE: &str = "code";
    pub const DECISIONS: &str = "decisions";
    pub const NOTES: &str = "notes";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_document_serialization() {
        let doc = CodeDocument {
            id: "abc123".to_string(),
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            symbols: vec!["main".to_string(), "Config".to_string()],
            docstrings: "Main entry point".to_string(),
            signatures: vec!["fn main()".to_string()],
            imports: vec!["std::io".to_string()],
            project_id: "proj-1".to_string(),
            project_slug: "my-project".to_string(),
        };

        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: CodeDocument = serde_json::from_str(&json).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.path, deserialized.path);
        assert_eq!(doc.symbols, deserialized.symbols);
        assert_eq!(doc.project_slug, deserialized.project_slug);
    }

    #[test]
    fn test_decision_document_serialization() {
        let doc = DecisionDocument {
            id: "dec-1".to_string(),
            description: "Use async/await".to_string(),
            rationale: "Better performance".to_string(),
            task_id: "task-1".to_string(),
            agent: "claude".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            tags: vec!["architecture".to_string()],
            project_id: Some("proj-1".to_string()),
            project_slug: Some("my-project".to_string()),
        };

        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: DecisionDocument = serde_json::from_str(&json).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.description, deserialized.description);
        assert_eq!(doc.project_id, deserialized.project_id);
    }

    #[test]
    fn test_decision_document_without_project() {
        // Test that project_id and project_slug default to None
        let json = r#"{
            "id": "dec-1",
            "description": "Test",
            "rationale": "Test reason",
            "task_id": "task-1",
            "agent": "test",
            "timestamp": "2024-01-01",
            "tags": []
        }"#;

        let doc: DecisionDocument = serde_json::from_str(json).unwrap();
        assert!(doc.project_id.is_none());
        assert!(doc.project_slug.is_none());
    }

    #[test]
    fn test_note_document_serialization() {
        let doc = NoteDocument {
            id: "note-1".to_string(),
            project_id: "proj-1".to_string(),
            project_slug: "my-project".to_string(),
            note_type: "guideline".to_string(),
            status: "active".to_string(),
            importance: "high".to_string(),
            scope_type: "file".to_string(),
            scope_path: "src/main.rs".to_string(),
            content: "Always handle errors properly".to_string(),
            tags: vec!["error-handling".to_string()],
            anchor_entities: vec!["file:src/main.rs".to_string()],
            created_at: 1704067200,
            created_by: "claude".to_string(),
            staleness_score: 0.0,
        };

        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: NoteDocument = serde_json::from_str(&json).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.note_type, deserialized.note_type);
        assert_eq!(doc.content, deserialized.content);
        assert_eq!(doc.staleness_score, deserialized.staleness_score);
    }

    #[test]
    fn test_search_hit_serialization() {
        let hit = SearchHit {
            document: CodeDocument {
                id: "test".to_string(),
                path: "test.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec![],
                docstrings: "".to_string(),
                signatures: vec![],
                imports: vec![],
                project_id: "".to_string(),
                project_slug: "".to_string(),
            },
            score: 0.85,
        };

        let json = serde_json::to_string(&hit).unwrap();
        let deserialized: SearchHit<CodeDocument> = serde_json::from_str(&json).unwrap();

        assert_eq!(hit.score, deserialized.score);
        assert_eq!(hit.document.id, deserialized.document.id);
    }

    #[test]
    fn test_index_stats_serialization() {
        let stats = IndexStats {
            total_documents: 1000,
            is_indexing: false,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: IndexStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.total_documents, deserialized.total_documents);
        assert_eq!(stats.is_indexing, deserialized.is_indexing);
    }
}
