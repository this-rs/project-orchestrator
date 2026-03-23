//! Tool Discovery & Introspection — understand external MCP tools automatically.
//!
//! When PO connects to an external MCP server, we don't just store the raw tool
//! definitions. We analyze them to understand what each tool does:
//!
//! 1. **Classification** — Is this a Query, Mutation, Create, Delete, or Search?
//!    (Critical for safe probing: we NEVER probe mutations.)
//! 2. **Embedding** — Generate a vector representation of the tool's description
//!    + schema for semantic similarity search.
//! 3. **Internal matching** — Find which of our own MCP tools are similar to
//!    this external tool (useful for mapping and substitution).
//!
//! ## Architecture
//!
//! ```text
//! tools/list response (Vec<McpToolDef>)
//!   │
//!   ▼
//! ToolIntrospector.introspect()
//!   ├── classify_tool()       → InferredCategory
//!   ├── embed_tool()          → Option<Vec<f32>> (768d)
//!   └── find_similar_internal → Vec<(fqn, score)>
//!   │
//!   ▼
//! Vec<DiscoveredTool>
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};

use crate::embeddings::EmbeddingProvider;

use super::client::McpToolDef;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Inferred category of an external tool, based on name + description analysis.
///
/// This classification drives critical safety decisions:
/// - `Query` / `Search` → safe to probe
/// - `Create` / `Mutation` / `Delete` → NEVER probe automatically
/// - `Unknown` → treated as Mutation for safety (no probing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferredCategory {
    /// Read-only data retrieval (get, list, find, fetch, describe, show).
    Query,
    /// Search/lookup operations (search, find, lookup, filter).
    Search,
    /// Creates new resources (create, add, insert, new, register, post).
    Create,
    /// Modifies existing resources (update, patch, set, modify, edit, rename, move).
    Mutation,
    /// Removes resources (delete, remove, destroy, drop, purge, unlink).
    Delete,
    /// Cannot determine — treated as Mutation for safety (no probing).
    Unknown,
}

impl InferredCategory {
    /// Whether this category is safe for automatic probing.
    ///
    /// Only `Query` and `Search` are safe — they don't mutate state.
    pub fn is_safe_to_probe(&self) -> bool {
        matches!(self, Self::Query | Self::Search)
    }
}

/// Response shape observed from probing a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseShape {
    /// Tool returns a JSON object.
    Object,
    /// Tool returns a JSON array.
    Array,
    /// Tool returns a scalar (string, number, bool, null).
    Scalar,
    /// Tool returned an error (shape unknown).
    Error,
    /// Tool was not probed.
    NotProbed,
}

/// Profile of an external tool, gathered from probing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProfile {
    /// Response latency in milliseconds.
    pub latency_ms: u64,
    /// Shape of the response data.
    pub response_shape: ResponseShape,
    /// Whether the tool supports pagination (detected from response or schema).
    pub pagination: bool,
    /// Error format hint (e.g., "json_rpc", "plain_text", "structured").
    pub error_format: Option<String>,
    /// When the tool was probed.
    pub probed_at: DateTime<Utc>,
}

impl ToolProfile {
    /// Create a profile for a tool that was NOT probed (unsafe category).
    pub fn not_probed() -> Self {
        Self {
            latency_ms: 0,
            response_shape: ResponseShape::NotProbed,
            pagination: false,
            error_format: None,
            probed_at: Utc::now(),
        }
    }
}

/// A tool discovered from an external MCP server, enriched with introspection data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredTool {
    /// Tool name as reported by the server (e.g., "run_cypher").
    pub name: String,
    /// Fully-qualified name: "server_id::tool_name".
    pub fqn: String,
    /// Human-readable description from the server.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
    /// Inferred category (Query, Mutation, etc.).
    pub category: InferredCategory,
    /// Embedding vector (768d from EmbeddingProvider, None if unavailable).
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Similar internal PO tools: (fqn, cosine_score), sorted by score desc.
    pub similar_internal: Vec<(String, f32)>,
    /// Optional profile from probing (only for Query/Search tools).
    pub profile: Option<ToolProfile>,
}

/// Internal tool descriptor for similarity matching.
#[derive(Debug, Clone)]
pub struct InternalToolDescriptor {
    /// Tool FQN (e.g., "note::search_semantic").
    pub fqn: String,
    /// Canonical text for embedding (name + description + param names).
    pub canonical_text: String,
    /// Cached embedding (768d).
    pub embedding: Option<Vec<f32>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classify a tool based on its name and description.
///
/// Uses a two-pass approach:
/// 1. **Name patterns** — strongest signal (most tools follow naming conventions)
/// 2. **Description keywords** — fallback when name is ambiguous
///
/// If both passes disagree or neither matches → `Unknown`.
pub fn classify_tool(name: &str, description: &str) -> InferredCategory {
    let name_lower = name.to_lowercase();
    let desc_lower = description.to_lowercase();

    // Pass 1: Name-based classification (strongest signal)
    let name_category = classify_by_name(&name_lower);

    // Pass 2: Description-based classification (fallback)
    let desc_category = classify_by_description(&desc_lower);

    match (name_category, desc_category) {
        // Name match is authoritative
        (Some(cat), _) => cat,
        // Fallback to description
        (None, Some(cat)) => cat,
        // No signal → Unknown (treated as unsafe)
        (None, None) => InferredCategory::Unknown,
    }
}

/// Classify by tool name patterns.
fn classify_by_name(name: &str) -> Option<InferredCategory> {
    // Delete patterns (check first — "remove_from_list" should be Delete, not Query)
    let delete_patterns = [
        "delete", "remove", "destroy", "drop", "purge", "unlink", "detach",
        "erase", "clear", "clean",
    ];
    for pat in &delete_patterns {
        if name.contains(pat) {
            return Some(InferredCategory::Delete);
        }
    }

    // Create patterns
    let create_patterns = [
        "create", "add", "insert", "new", "register", "post", "generate",
        "init", "bootstrap", "setup", "provision",
    ];
    for pat in &create_patterns {
        if name.contains(pat) {
            return Some(InferredCategory::Create);
        }
    }

    // Mutation patterns
    let mutation_patterns = [
        "update", "patch", "set", "modify", "edit", "rename", "move",
        "assign", "reassign", "change", "toggle", "enable", "disable",
        "approve", "reject", "merge", "close", "reopen", "archive",
        "restore", "sync", "refresh", "reset", "configure", "link",
        "connect", "disconnect", "start", "stop", "pause", "resume",
        "cancel", "retry", "replay", "advance", "supersede",
        "confirm", "invalidate", "publish", "unpublish",
    ];
    for pat in &mutation_patterns {
        if name.contains(pat) {
            return Some(InferredCategory::Mutation);
        }
    }

    // Search patterns (check before Query — "search_notes" is Search, not Query)
    let search_patterns = [
        "search", "find", "lookup", "filter", "query", "match",
        "discover", "explore", "scan", "detect",
    ];
    for pat in &search_patterns {
        if name.contains(pat) {
            return Some(InferredCategory::Search);
        }
    }

    // Query patterns
    let query_patterns = [
        "get", "list", "fetch", "describe", "show", "read", "view",
        "inspect", "check", "count", "exists", "is_", "has_",
        "status", "info", "health", "ping", "version", "whoami",
        "analyze", "compute", "calculate", "estimate", "predict",
        "compare", "diff", "export", "download",
    ];
    for pat in &query_patterns {
        if name.contains(pat) {
            return Some(InferredCategory::Query);
        }
    }

    None
}

/// Classify by description keywords (weaker signal than name).
fn classify_by_description(desc: &str) -> Option<InferredCategory> {
    // Score each category by keyword presence
    let mut scores: [(InferredCategory, i32); 5] = [
        (InferredCategory::Query, 0),
        (InferredCategory::Search, 0),
        (InferredCategory::Create, 0),
        (InferredCategory::Mutation, 0),
        (InferredCategory::Delete, 0),
    ];

    let query_keywords = [
        "retriev", "return", "get", "list", "fetch", "read-only", "readonly",
        "inspect", "view", "show", "display",
    ];
    let search_keywords = [
        "search", "find", "look up", "filter", "semantic", "full-text",
        "vector", "similar",
    ];
    let create_keywords = [
        "create", "add", "insert", "new", "register", "generate",
    ];
    let mutation_keywords = [
        "update", "modify", "change", "set", "edit", "patch", "mutate",
        "write",
    ];
    let delete_keywords = [
        "delete", "remove", "destroy", "drop", "purge", "erase",
    ];

    for kw in &query_keywords {
        if desc.contains(kw) { scores[0].1 += 1; }
    }
    for kw in &search_keywords {
        if desc.contains(kw) { scores[1].1 += 1; }
    }
    for kw in &create_keywords {
        if desc.contains(kw) { scores[2].1 += 1; }
    }
    for kw in &mutation_keywords {
        if desc.contains(kw) { scores[3].1 += 1; }
    }
    for kw in &delete_keywords {
        if desc.contains(kw) { scores[4].1 += 1; }
    }

    // Find the max
    scores.sort_by(|a, b| b.1.cmp(&a.1));

    if scores[0].1 == 0 {
        return None;
    }

    // Require clear winner (at least 1 point ahead of second)
    if scores[0].1 > scores[1].1 {
        Some(scores[0].0)
    } else {
        None // Ambiguous → Unknown
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Build a canonical text representation from a tool's metadata for embedding.
///
/// Format: "tool_name: description. Parameters: param1 (type), param2 (type), ..."
pub fn build_canonical_text(name: &str, description: &str, input_schema: &serde_json::Value) -> String {
    let mut text = format!("{}: {}", name, description);

    // Extract parameter names and types from JSON Schema
    if let Some(properties) = input_schema.get("properties").and_then(|p| p.as_object()) {
        let params: Vec<String> = properties
            .iter()
            .map(|(param_name, param_def)| {
                let param_type = param_def
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("any");
                let param_desc = param_def
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                if param_desc.is_empty() {
                    format!("{} ({})", param_name, param_type)
                } else {
                    // Truncate long descriptions
                    let short = if param_desc.len() > 60 {
                        &param_desc[..60]
                    } else {
                        param_desc
                    };
                    format!("{} ({}, {})", param_name, param_type, short)
                }
            })
            .collect();

        if !params.is_empty() {
            text.push_str(". Parameters: ");
            text.push_str(&params.join(", "));
        }
    }

    text
}

/// Embed a single tool description using the embedding provider.
///
/// Returns `None` if no provider is available.
pub async fn embed_tool(
    provider: &dyn EmbeddingProvider,
    name: &str,
    description: &str,
    input_schema: &serde_json::Value,
) -> Result<Vec<f32>> {
    let text = build_canonical_text(name, description, input_schema);
    provider.embed_text(&text).await
}

/// Embed multiple tools in batch (more efficient than individual calls).
pub async fn embed_tools_batch(
    provider: &dyn EmbeddingProvider,
    tools: &[McpToolDef],
) -> Result<Vec<Vec<f32>>> {
    let texts: Vec<String> = tools
        .iter()
        .map(|t| {
            build_canonical_text(
                &t.name,
                t.description.as_deref().unwrap_or(""),
                &t.input_schema,
            )
        })
        .collect();

    provider.embed_batch(&texts).await
}

// ─────────────────────────────────────────────────────────────────────────────
// Similarity matching
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity between two vectors.
///
/// Assumes both vectors are L2-normalized (returns dot product directly).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Find the top-K most similar internal tools for a given embedding.
///
/// Returns `(fqn, cosine_score)` pairs, filtered by `min_score` threshold.
pub fn find_similar_internal(
    embedding: &[f32],
    internal_tools: &[InternalToolDescriptor],
    top_k: usize,
    min_score: f32,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = internal_tools
        .iter()
        .filter_map(|tool| {
            let tool_emb = tool.embedding.as_ref()?;
            let score = cosine_similarity(embedding, tool_emb);
            if score >= min_score {
                Some((tool.fqn.clone(), score))
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspector
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the tool introspector.
#[derive(Debug, Clone)]
pub struct IntrospectorConfig {
    /// Minimum cosine similarity for internal tool matching.
    pub similarity_threshold: f32,
    /// Maximum number of similar internal tools to return per external tool.
    pub top_k: usize,
}

impl Default for IntrospectorConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            top_k: 5,
        }
    }
}

/// Orchestrates the full introspection pipeline for discovered tools.
pub struct ToolIntrospector {
    /// Embedding provider (None = skip embedding + similarity).
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Pre-computed descriptors for our internal MCP tools.
    internal_tools: Vec<InternalToolDescriptor>,
    /// Configuration.
    config: IntrospectorConfig,
}

impl ToolIntrospector {
    /// Create a new introspector.
    ///
    /// If `embedding_provider` is None, embedding and similarity matching are skipped.
    pub fn new(
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
        internal_tools: Vec<InternalToolDescriptor>,
        config: IntrospectorConfig,
    ) -> Self {
        Self {
            embedding_provider,
            internal_tools,
            config,
        }
    }

    /// Run the full introspection pipeline on a batch of tools from an external server.
    ///
    /// For each tool: classify → embed → find_similar → build DiscoveredTool.
    pub async fn introspect(
        &self,
        server_id: &str,
        tools: &[McpToolDef],
    ) -> Vec<DiscoveredTool> {
        // 1. Classify all tools (instant, no I/O)
        let categories: Vec<InferredCategory> = tools
            .iter()
            .map(|t| classify_tool(&t.name, t.description.as_deref().unwrap_or("")))
            .collect();

        // 2. Batch embed all tools (single API call if provider available)
        let embeddings: Vec<Option<Vec<f32>>> = if let Some(provider) = &self.embedding_provider {
            match embed_tools_batch(provider.as_ref(), tools).await {
                Ok(batch) => batch.into_iter().map(Some).collect(),
                Err(e) => {
                    warn!(error = %e, "Failed to batch-embed tools, skipping embeddings");
                    vec![None; tools.len()]
                }
            }
        } else {
            debug!("No embedding provider, skipping tool embeddings");
            vec![None; tools.len()]
        };

        // 3. Build DiscoveredTool for each
        tools
            .iter()
            .zip(categories.iter())
            .zip(embeddings.into_iter())
            .map(|((tool, &category), embedding)| {
                let fqn = format!("{}::{}", server_id, tool.name);

                // Find similar internal tools (only if we have an embedding)
                let similar_internal = match &embedding {
                    Some(emb) => find_similar_internal(
                        emb,
                        &self.internal_tools,
                        self.config.top_k,
                        self.config.similarity_threshold,
                    ),
                    None => vec![],
                };

                debug!(
                    tool = %tool.name,
                    fqn = %fqn,
                    category = ?category,
                    similar_count = similar_internal.len(),
                    has_embedding = embedding.is_some(),
                    "Introspected tool"
                );

                DiscoveredTool {
                    name: tool.name.clone(),
                    fqn,
                    description: tool.description.clone().unwrap_or_default(),
                    input_schema: tool.input_schema.clone(),
                    category,
                    embedding,
                    similar_internal,
                    profile: None, // Filled by ToolProber later
                }
            })
            .collect()
    }

    /// Build internal tool descriptors from our MCP tool definitions.
    ///
    /// This pre-computes embeddings for all our internal tools, enabling
    /// similarity matching with external tools.
    pub async fn build_internal_descriptors(
        provider: &dyn EmbeddingProvider,
        tools: &[(String, String, serde_json::Value)], // (name, description, schema)
    ) -> Result<Vec<InternalToolDescriptor>> {
        let texts: Vec<String> = tools
            .iter()
            .map(|(name, desc, schema)| build_canonical_text(name, desc, schema))
            .collect();

        let fqns: Vec<String> = tools
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect();

        let embeddings = provider.embed_batch(&texts).await?;

        Ok(fqns
            .into_iter()
            .zip(texts.into_iter())
            .zip(embeddings.into_iter())
            .map(|((fqn, canonical_text), embedding)| InternalToolDescriptor {
                fqn,
                canonical_text,
                embedding: Some(embedding),
            })
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingProvider;
    use serde_json::json;

    // ── Classification tests ──────────────────────────────────────────────

    #[test]
    fn test_classify_query_tools() {
        assert_eq!(classify_tool("get_user", "Get a user by ID"), InferredCategory::Query);
        assert_eq!(classify_tool("list_projects", "List all projects"), InferredCategory::Query);
        assert_eq!(classify_tool("fetch_data", "Fetch data from source"), InferredCategory::Query);
        assert_eq!(classify_tool("describe_table", "Describe table schema"), InferredCategory::Query);
        assert_eq!(classify_tool("show_status", "Show current status"), InferredCategory::Query);
        assert_eq!(classify_tool("count_items", "Count items in collection"), InferredCategory::Query);
        assert_eq!(classify_tool("check_health", "Health check endpoint"), InferredCategory::Query);
        assert_eq!(classify_tool("analyze_impact", "Analyze code impact"), InferredCategory::Query);
    }

    #[test]
    fn test_classify_search_tools() {
        assert_eq!(classify_tool("search_notes", "Search notes by query"), InferredCategory::Search);
        assert_eq!(classify_tool("find_references", "Find symbol references"), InferredCategory::Search);
        assert_eq!(classify_tool("lookup_user", "Look up user by email"), InferredCategory::Search);
        assert_eq!(classify_tool("filter_tasks", "Filter tasks by criteria"), InferredCategory::Search);
        assert_eq!(classify_tool("discover_tools", "Discover available tools"), InferredCategory::Search);
    }

    #[test]
    fn test_classify_create_tools() {
        assert_eq!(classify_tool("create_issue", "Create a new issue"), InferredCategory::Create);
        assert_eq!(classify_tool("add_comment", "Add a comment"), InferredCategory::Create);
        assert_eq!(classify_tool("insert_record", "Insert a new record"), InferredCategory::Create);
        assert_eq!(classify_tool("register_webhook", "Register a webhook"), InferredCategory::Create);
        assert_eq!(classify_tool("generate_report", "Generate a report"), InferredCategory::Create);
    }

    #[test]
    fn test_classify_mutation_tools() {
        assert_eq!(classify_tool("update_task", "Update task fields"), InferredCategory::Mutation);
        assert_eq!(classify_tool("patch_config", "Patch configuration"), InferredCategory::Mutation);
        assert_eq!(classify_tool("set_status", "Set the status"), InferredCategory::Mutation);
        assert_eq!(classify_tool("modify_permissions", "Modify permissions"), InferredCategory::Mutation);
        assert_eq!(classify_tool("rename_file", "Rename a file"), InferredCategory::Mutation);
        assert_eq!(classify_tool("enable_feature", "Enable a feature flag"), InferredCategory::Mutation);
        assert_eq!(classify_tool("sync_data", "Sync data from remote"), InferredCategory::Mutation);
    }

    #[test]
    fn test_classify_delete_tools() {
        assert_eq!(classify_tool("delete_user", "Delete a user"), InferredCategory::Delete);
        assert_eq!(classify_tool("remove_member", "Remove a team member"), InferredCategory::Delete);
        assert_eq!(classify_tool("purge_cache", "Purge the cache"), InferredCategory::Delete);
        assert_eq!(classify_tool("drop_index", "Drop a database index"), InferredCategory::Delete);
    }

    #[test]
    fn test_classify_unknown_tools() {
        // Completely ambiguous names with no description
        assert_eq!(classify_tool("run", ""), InferredCategory::Unknown);
        assert_eq!(classify_tool("process", ""), InferredCategory::Unknown);
        assert_eq!(classify_tool("execute", ""), InferredCategory::Unknown);
    }

    #[test]
    fn test_classify_description_fallback() {
        // Name is ambiguous but description provides signal
        assert_eq!(
            classify_tool("do_stuff", "Retrieves data from the database"),
            InferredCategory::Query
        );
        assert_eq!(
            classify_tool("action", "Deletes the specified resource permanently"),
            InferredCategory::Delete
        );
    }

    #[test]
    fn test_category_safety() {
        assert!(InferredCategory::Query.is_safe_to_probe());
        assert!(InferredCategory::Search.is_safe_to_probe());
        assert!(!InferredCategory::Create.is_safe_to_probe());
        assert!(!InferredCategory::Mutation.is_safe_to_probe());
        assert!(!InferredCategory::Delete.is_safe_to_probe());
        assert!(!InferredCategory::Unknown.is_safe_to_probe());
    }

    // ── Canonical text tests ──────────────────────────────────────────────

    #[test]
    fn test_canonical_text_with_params() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "limit": { "type": "integer" }
            }
        });
        let text = build_canonical_text("search_notes", "Search notes semantically", &schema);
        assert!(text.contains("search_notes: Search notes semantically"));
        assert!(text.contains("Parameters:"));
        assert!(text.contains("query"));
        assert!(text.contains("limit"));
    }

    #[test]
    fn test_canonical_text_without_params() {
        let text = build_canonical_text("ping", "Ping the server", &json!({}));
        assert_eq!(text, "ping: Ping the server");
    }

    // ── Cosine similarity tests ───────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── find_similar_internal tests ───────────────────────────────────────

    #[test]
    fn test_find_similar_filters_by_threshold() {
        let tools = vec![
            InternalToolDescriptor {
                fqn: "note::search".into(),
                canonical_text: "".into(),
                embedding: Some(vec![1.0, 0.0, 0.0]),
            },
            InternalToolDescriptor {
                fqn: "task::list".into(),
                canonical_text: "".into(),
                embedding: Some(vec![0.0, 1.0, 0.0]),
            },
        ];

        let query = vec![1.0, 0.0, 0.0]; // Identical to note::search
        let results = find_similar_internal(&query, &tools, 5, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "note::search");
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_find_similar_top_k_limit() {
        let tools: Vec<InternalToolDescriptor> = (0..10)
            .map(|i| InternalToolDescriptor {
                fqn: format!("tool_{}", i),
                canonical_text: "".into(),
                embedding: Some(vec![1.0, i as f32 * 0.01, 0.0]),
            })
            .collect();

        let query = vec![1.0, 0.0, 0.0];
        let results = find_similar_internal(&query, &tools, 3, 0.0);
        assert_eq!(results.len(), 3);
    }

    // ── Introspector tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_introspect_classifies_all_tools() {
        let introspector = ToolIntrospector::new(None, vec![], IntrospectorConfig::default());

        let tools = vec![
            McpToolDef {
                name: "search_notes".into(),
                description: Some("Search notes".into()),
                input_schema: json!({}),
            },
            McpToolDef {
                name: "create_issue".into(),
                description: Some("Create an issue".into()),
                input_schema: json!({}),
            },
            McpToolDef {
                name: "delete_user".into(),
                description: Some("Delete a user".into()),
                input_schema: json!({}),
            },
        ];

        let discovered = introspector.introspect("test-server", &tools).await;
        assert_eq!(discovered.len(), 3);
        assert_eq!(discovered[0].category, InferredCategory::Search);
        assert_eq!(discovered[1].category, InferredCategory::Create);
        assert_eq!(discovered[2].category, InferredCategory::Delete);
    }

    #[tokio::test]
    async fn test_introspect_builds_fqn() {
        let introspector = ToolIntrospector::new(None, vec![], IntrospectorConfig::default());

        let tools = vec![McpToolDef {
            name: "run_cypher".into(),
            description: Some("Run a Cypher query".into()),
            input_schema: json!({}),
        }];

        let discovered = introspector.introspect("grafeo", &tools).await;
        assert_eq!(discovered[0].fqn, "grafeo::run_cypher");
    }

    #[tokio::test]
    async fn test_introspect_with_embeddings() {
        let provider = Arc::new(MockEmbeddingProvider::new(768));

        // Build some internal tools
        let internal = vec![InternalToolDescriptor {
            fqn: "note::search_semantic".into(),
            canonical_text: "search_semantic: Semantic search notes".into(),
            embedding: Some(
                provider
                    .embed_text("search_semantic: Semantic search notes")
                    .await
                    .unwrap(),
            ),
        }];

        let introspector = ToolIntrospector::new(
            Some(provider),
            internal,
            IntrospectorConfig {
                similarity_threshold: 0.0, // Accept all for test
                top_k: 5,
            },
        );

        let tools = vec![McpToolDef {
            name: "vector_search".into(),
            description: Some("Search by vector similarity".into()),
            input_schema: json!({}),
        }];

        let discovered = introspector.introspect("external", &tools).await;
        assert_eq!(discovered.len(), 1);
        assert!(discovered[0].embedding.is_some());
        // With mock embeddings, the similar_internal won't be empty
        // (mock generates deterministic vectors from text)
    }

    #[tokio::test]
    async fn test_introspect_ten_tools() {
        let provider = Arc::new(MockEmbeddingProvider::new(768));
        let introspector = ToolIntrospector::new(
            Some(provider),
            vec![],
            IntrospectorConfig::default(),
        );

        let tools: Vec<McpToolDef> = (0..10)
            .map(|i| McpToolDef {
                name: format!("get_item_{}", i),
                description: Some(format!("Get item {}", i)),
                input_schema: json!({"type": "object", "properties": {"id": {"type": "string"}}}),
            })
            .collect();

        let discovered = introspector.introspect("batch-server", &tools).await;
        assert_eq!(discovered.len(), 10);
        for tool in &discovered {
            assert!(tool.embedding.is_some());
            assert_eq!(tool.category, InferredCategory::Query);
            assert!(tool.fqn.starts_with("batch-server::"));
        }
    }

    // ── build_internal_descriptors tests ──────────────────────────────────

    #[tokio::test]
    async fn test_build_internal_descriptors() {
        let provider = MockEmbeddingProvider::new(768);
        let tools = vec![
            (
                "note::search_semantic".to_string(),
                "Semantic search notes".to_string(),
                json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            ),
            (
                "task::create".to_string(),
                "Create a new task".to_string(),
                json!({"type": "object", "properties": {"title": {"type": "string"}}}),
            ),
        ];

        let descriptors = ToolIntrospector::build_internal_descriptors(&provider, &tools)
            .await
            .unwrap();

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].fqn, "note::search_semantic");
        assert!(descriptors[0].embedding.is_some());
        assert_eq!(descriptors[0].embedding.as_ref().unwrap().len(), 768);
    }
}
