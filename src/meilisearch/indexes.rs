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
