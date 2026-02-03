//! Index definitions for Meilisearch

use serde::{Deserialize, Serialize};

/// Code document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDocument {
    pub id: String,
    pub path: String,
    pub language: String,
    pub content: String,
    pub symbols: Vec<String>,
    pub imports: Vec<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub project_slug: Option<String>,
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

/// Log document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogDocument {
    pub id: String,
    pub agent_id: String,
    pub task_id: Option<String>,
    pub level: String,
    pub message: String,
    pub timestamp: String,
}

/// Conversation document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationDocument {
    pub id: String,
    pub participants: Vec<String>,
    pub content: String,
    pub topic: Option<String>,
    pub timestamp: String,
}

/// Index names
pub mod index_names {
    pub const CODE: &str = "code";
    pub const DECISIONS: &str = "decisions";
    pub const LOGS: &str = "logs";
    pub const CONVERSATIONS: &str = "conversations";
}
