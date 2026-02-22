//! Implementation Flow Planner — analyzes the knowledge graph to produce
//! a DAG of implementation phases (sequential + parallel branches).

use crate::meilisearch::SearchStore;
use crate::neo4j::models::{StepNode as StepNodeModel, StepStatus};
use crate::neo4j::GraphStore;
use crate::notes::models::EntityType;
use crate::notes::NoteManager;
use crate::plan::models::CreatePlanRequest;
use crate::plan::models::CreateTaskRequest;
use crate::plan::PlanManager;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Input types
// ============================================================================

/// Scope of the implementation analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlanScope {
    File,
    #[default]
    Module,
    Project,
}

/// Request to plan an implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Project UUID
    pub project_id: Uuid,
    /// Project slug (for SearchStore queries)
    pub project_slug: Option<String>,
    /// Human description of what to implement
    pub description: String,
    /// Explicit entry points (file paths or function names)
    #[serde(default)]
    pub entry_points: Option<Vec<String>>,
    /// Scope of analysis
    #[serde(default)]
    pub scope: Option<PlanScope>,
    /// If true, auto-create a Plan MCP with Tasks/Steps
    #[serde(default)]
    pub auto_create_plan: Option<bool>,
    /// Project root path (for resolving relative file paths to absolute)
    #[serde(default)]
    pub root_path: Option<String>,
}

// ============================================================================
// Output types
// ============================================================================

/// Risk level for a modification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    #[default]
    Low,
    Medium,
    High,
}

impl RiskLevel {
    /// Compare risk levels, returning the higher one
    pub fn max(self, other: Self) -> Self {
        match (&self, &other) {
            (Self::High, _) | (_, Self::High) => Self::High,
            (Self::Medium, _) | (_, Self::Medium) => Self::Medium,
            _ => Self::Low,
        }
    }
}

/// A single file modification in a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modification {
    /// File path to modify
    pub file: String,
    /// Reason for the modification
    pub reason: String,
    /// Symbols affected in this file
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols_affected: Vec<String>,
    /// Risk level for this modification
    pub risk: RiskLevel,
    /// Number of files that depend on this file
    pub dependents_count: usize,
}

/// A parallel branch within a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Files in this branch
    pub files: Vec<String>,
    /// Reason for this branch
    pub reason: String,
}

/// A phase of implementation (sequential step in the DAG)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    /// Phase number (1-based)
    pub phase_number: usize,
    /// Human-readable description
    pub description: String,
    /// Whether modifications in this phase can be done in parallel
    pub parallel: bool,
    /// Modifications (when parallel=false or single modification)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modifications: Vec<Modification>,
    /// Parallel branches (when parallel=true)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<Branch>,
}

/// A contextual note relevant to the implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerNote {
    /// Note type (gotcha, guideline, pattern, etc.)
    pub note_type: String,
    /// Note content
    pub content: String,
    /// Importance level
    pub importance: String,
    /// Source entity (file/function the note is attached to)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_entity: Option<String>,
}

/// The complete implementation plan output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationPlan {
    /// Human-readable summary
    pub summary: String,
    /// Ordered phases of implementation
    pub phases: Vec<Phase>,
    /// Test files that should be run
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_files: Vec<String>,
    /// Relevant notes (gotchas, guidelines)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<PlannerNote>,
    /// Overall risk level
    pub total_risk: RiskLevel,
    /// Plan ID if auto_create_plan was true
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
}

// ============================================================================
// Internal types (DAG construction)
// ============================================================================

/// Source of a relevant zone discovery
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ZoneSource {
    /// User-provided explicit entry point
    ExplicitEntry,
    /// Found via semantic code search
    SemanticSearch,
    /// Found via note reference
    NoteReference,
}

/// A relevant zone of code identified for modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevantZone {
    /// File path
    pub file_path: String,
    /// Functions in this zone
    #[serde(default)]
    pub functions: Vec<String>,
    /// Relevance score (0.0 - 1.0)
    pub relevance_score: f64,
    /// How this zone was discovered
    pub source: ZoneSource,
}

/// A node in the modification DAG
#[derive(Debug, Clone)]
pub struct DagNode {
    /// File path
    pub file_path: String,
    /// Symbols in this file
    pub symbols: Vec<String>,
    /// Risk level
    pub risk: RiskLevel,
    /// Test files that depend on this file
    pub test_files: Vec<String>,
    /// Number of dependent files
    pub dependents_count: usize,
    /// Community ID from graph analytics (Louvain clustering).
    /// Used to group files into parallel branches in compute_phases().
    pub community_id: Option<u32>,
    /// Files from the same community that may need co-modification.
    /// Populated when a file has 0 structural dependents but has community peers.
    pub community_peers: Vec<String>,
}

/// The modification DAG (Directed Acyclic Graph)
#[derive(Debug, Clone)]
pub struct ModificationDag {
    /// Nodes indexed by file path
    pub nodes: HashMap<String, DagNode>,
    /// Edges: (from, to) meaning "from must be modified before to"
    pub edges: Vec<(String, String)>,
}

// ============================================================================
// Path resolution helper
// ============================================================================

/// Resolve a potentially relative file path to absolute using the project root_path.
/// - If `path` is already absolute (starts with `/`), return it as-is.
/// - If `root_path` is provided and `path` is relative, prepend the expanded root.
/// - Otherwise return `path` unchanged.
fn resolve_path(root_path: Option<&str>, path: &str) -> String {
    if path.starts_with('/') {
        return path.to_string();
    }
    if let Some(root) = root_path {
        let expanded = crate::expand_tilde(root);
        format!("{}/{}", expanded.trim_end_matches('/'), path)
    } else {
        path.to_string()
    }
}

// ============================================================================
// Planner
// ============================================================================

/// Analyzes the knowledge graph to produce implementation plans
pub struct ImplementationPlanner {
    neo4j: Arc<dyn GraphStore>,
    meili: Arc<dyn SearchStore>,
    plan_manager: Arc<PlanManager>,
    note_manager: Arc<NoteManager>,
    /// Optional embedding provider for hybrid vector search in search_semantic_zones
    embedding_provider: Option<Arc<dyn crate::embeddings::EmbeddingProvider>>,
}

/// Maximum number of zones to consider
const MAX_ZONES: usize = 20;

/// Common stop words (French + English) that never appear in code and
/// dilute Meilisearch keyword matching when the user's description is
/// in natural language.
const STOP_WORDS: &[&str] = &[
    // French
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "d",
    "l",
    "et",
    "ou",
    "en",
    "au",
    "aux",
    "ce",
    "ces",
    "cette",
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "son",
    "sa",
    "ses",
    "notre",
    "nos",
    "votre",
    "vos",
    "leur",
    "leurs",
    "qui",
    "que",
    "quoi",
    "dont",
    "dans",
    "sur",
    "sous",
    "avec",
    "sans",
    "pour",
    "par",
    "vers",
    "chez",
    "entre",
    "comme",
    "plus",
    "moins",
    "très",
    "trop",
    "je",
    "tu",
    "il",
    "elle",
    "nous",
    "vous",
    "ils",
    "elles",
    "on",
    "ne",
    "pas",
    "est",
    "sont",
    "être",
    "avoir",
    "fait",
    "faire",
    "faut",
    "peut",
    "doit",
    "dois",
    "quand",
    "si",
    "mais",
    "car",
    "donc",
    "ni",
    "puis",
    "aussi",
    "bien",
    "tout",
    "tous",
    "toute",
    "toutes",
    "même",
    "autre",
    "autres",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "chaque",
    "quelque",
    "quelques",
    "comment",
    "combien",
    "où",
    "ya",
    "y",
    "a",
    "à",
    // French verbs commonly used in feature descriptions
    "ajouter",
    "créer",
    "modifier",
    "supprimer",
    "mettre",
    "jour",
    "implémenter",
    "implementer",
    "corriger",
    "refactorer",
    // English
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "up",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "and",
    "but",
    "or",
    "if",
    "while",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    // English verbs commonly used in feature descriptions
    "add",
    "create",
    "update",
    "delete",
    "remove",
    "fix",
    "implement",
    "change",
    "modify",
    "refactor",
    "move",
    "need",
    "want",
    "new",
    "make",
    "get",
    "set",
    "use",
    "using",
    "like",
];

/// Extract code-relevant keywords from a natural language description.
///
/// Strips stop words (FR/EN), short tokens (< 2 chars), and returns
/// the remaining terms joined by spaces. Falls back to the original
/// description if filtering removes everything.
fn extract_search_keywords(description: &str) -> String {
    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    // Trim only common natural-language punctuation from edges.
    // Preserve code-relevant chars: _ - / : # . @ (paths, URLs, identifiers)
    let trim_chars = |c: char| {
        matches!(
            c,
            ',' | ';'
                | '!'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '"'
                | '\''
                | '«'
                | '»'
                | '—'
                | '–'
                | '\u{2019}'
        )
    };

    let keywords: Vec<&str> = description
        .split_whitespace()
        // Split on apostrophes within words (e.g. "l'endpoint" → ["l", "endpoint"])
        .flat_map(|w| w.split('\''))
        .flat_map(|w| w.split('\u{2019}')) // curly apostrophe
        .map(|w| w.trim_matches(trim_chars))
        .filter(|w| !w.is_empty())
        .filter(|w| w.len() >= 2)
        .filter(|w| !stop_set.contains(&w.to_lowercase().as_str()))
        .collect();

    if keywords.is_empty() {
        description.to_string()
    } else {
        keywords.join(" ")
    }
}
/// Maximum depth for dependency expansion
const MAX_DEPENDENCY_DEPTH: u32 = 2;
/// Threshold for high risk (dependents count)
const HIGH_RISK_THRESHOLD: usize = 10;
/// Threshold for medium risk (dependents count)
const MEDIUM_RISK_THRESHOLD: usize = 3;
/// RRF constant (standard value from the original paper)
const RRF_K: f64 = 60.0;

/// Reciprocal Rank Fusion: merges two ranked result lists into one.
///
/// For each result appearing in either list, computes:
///   score = Σ 1/(k + rank_i)
/// Results appearing in both lists are naturally boosted.
fn reciprocal_rank_fusion(
    bm25_results: &[(String, f64)],
    vector_results: &[(String, f64)],
    k: usize,
) -> Vec<(String, f64)> {
    use std::collections::HashMap;
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (path, _score)) in bm25_results.iter().enumerate() {
        *scores.entry(path.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    for (rank, (path, _score)) in vector_results.iter().enumerate() {
        *scores.entry(path.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

impl ImplementationPlanner {
    /// Create a new implementation planner
    pub fn new(
        neo4j: Arc<dyn GraphStore>,
        meili: Arc<dyn SearchStore>,
        plan_manager: Arc<PlanManager>,
        note_manager: Arc<NoteManager>,
    ) -> Self {
        Self {
            neo4j,
            meili,
            plan_manager,
            note_manager,
            embedding_provider: None,
        }
    }

    /// Add an embedding provider for hybrid vector search
    pub fn with_embedding_provider(
        mut self,
        provider: Arc<dyn crate::embeddings::EmbeddingProvider>,
    ) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    // ========================================================================
    // Phase 1 — Zone identification (Task 2)
    // ========================================================================

    /// Resolve explicit entry points to RelevantZones.
    /// If the entry point looks like a file path (contains `/` or `.`), resolve via
    /// `get_file_symbol_names`. Otherwise treat it as a symbol name and use
    /// `find_symbol_references` to locate the file.
    ///
    /// Relative file paths are resolved to absolute using the project `root_path`.
    async fn resolve_explicit_entries(
        &self,
        entry_points: &[String],
        project_id: Uuid,
        root_path: Option<&str>,
    ) -> Result<Vec<RelevantZone>> {
        let mut zones = Vec::new();
        for entry in entry_points {
            if entry.contains('/') || entry.contains('.') {
                // File path — resolve to absolute, then get symbols
                let resolved = resolve_path(root_path, entry);
                match self.neo4j.get_file_symbol_names(&resolved).await {
                    Ok(symbols) => {
                        let mut functions = symbols.functions;
                        functions.extend(symbols.structs);
                        functions.extend(symbols.traits);
                        functions.extend(symbols.enums);
                        zones.push(RelevantZone {
                            file_path: resolved,
                            functions,
                            relevance_score: 1.0,
                            source: ZoneSource::ExplicitEntry,
                        });
                    }
                    Err(_) => {
                        // File not in graph — still add it with resolved path
                        zones.push(RelevantZone {
                            file_path: resolved,
                            functions: vec![],
                            relevance_score: 1.0,
                            source: ZoneSource::ExplicitEntry,
                        });
                    }
                }
            } else {
                // Symbol name — find references to locate the file
                let refs = self
                    .neo4j
                    .find_symbol_references(entry, 5, Some(project_id))
                    .await?;
                if let Some(first) = refs.first() {
                    zones.push(RelevantZone {
                        file_path: first.file_path.clone(),
                        functions: vec![entry.clone()],
                        relevance_score: 1.0,
                        source: ZoneSource::ExplicitEntry,
                    });
                }
            }
        }
        Ok(zones)
    }

    /// Fallback: search for relevant zones via hybrid BM25 + vector search + note search.
    ///
    /// Uses `tokio::join!` to parallelize independent queries, then merges code results
    /// via Reciprocal Rank Fusion (RRF) when an embedding provider is available.
    ///
    /// The description is pre-processed with `extract_search_keywords` to strip
    /// natural-language stop words (FR/EN) that would dilute Meilisearch matching.
    async fn search_semantic_zones(
        &self,
        description: &str,
        project_slug: Option<&str>,
        project_id: Option<Uuid>,
    ) -> Result<Vec<RelevantZone>> {
        let keywords = extract_search_keywords(description);
        tracing::debug!(
            "Planner: semantic search — raw={:?}, keywords={:?}, has_embedding={}",
            description,
            keywords,
            self.embedding_provider.is_some()
        );

        // BM25 search (always available) + note search in parallel
        let (code_results, note_results) = tokio::join!(
            self.meili
                .search_code_with_scores(&keywords, 10, None, project_slug, None),
            self.meili
                .search_notes_with_filters(description, 10, project_slug, None, None, None),
        );

        // Vector search (optional — only when embedding provider is available)
        let vector_results: Vec<(String, f64)> =
            if let (Some(ref provider), Some(pid)) = (&self.embedding_provider, project_id) {
                match provider.embed_text(description).await {
                    Ok(query_embedding) => {
                        match self
                            .neo4j
                            .vector_search_files(&query_embedding, 10, Some(pid))
                            .await
                        {
                            Ok(results) => results,
                            Err(e) => {
                                tracing::warn!("Planner: vector file search failed: {}", e);
                                vec![]
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Planner: embedding query failed: {}", e);
                        vec![]
                    }
                }
            } else {
                vec![]
            };

        let mut zones = Vec::new();

        // Merge BM25 + vector results via RRF (or use BM25 alone as fallback)
        let bm25_list: Vec<(String, f64)> = match &code_results {
            Ok(hits) => hits
                .iter()
                .map(|h| (h.document.path.clone(), h.score.min(1.0)))
                .collect(),
            Err(e) => {
                tracing::warn!("Planner: semantic code search failed: {}", e);
                vec![]
            }
        };

        // Build a map of path → symbols from BM25 results for later lookup
        let symbols_map: std::collections::HashMap<String, Vec<String>> = match &code_results {
            Ok(hits) => hits
                .iter()
                .map(|h| (h.document.path.clone(), h.document.symbols.clone()))
                .collect(),
            Err(_) => std::collections::HashMap::new(),
        };

        if !vector_results.is_empty() {
            // Hybrid: merge BM25 + vector via Reciprocal Rank Fusion
            let fused = reciprocal_rank_fusion(&bm25_list, &vector_results, 15);
            tracing::debug!(
                "Planner: hybrid search — bm25={}, vector={}, fused={}",
                bm25_list.len(),
                vector_results.len(),
                fused.len()
            );
            for (path, rrf_score) in fused {
                let symbols = symbols_map.get(&path).cloned().unwrap_or_default();
                zones.push(RelevantZone {
                    file_path: path,
                    functions: symbols,
                    relevance_score: rrf_score,
                    source: ZoneSource::SemanticSearch,
                });
            }
        } else {
            // Fallback: BM25 only
            for (path, score) in bm25_list {
                let symbols = symbols_map.get(&path).cloned().unwrap_or_default();
                zones.push(RelevantZone {
                    file_path: path,
                    functions: symbols,
                    relevance_score: score,
                    source: ZoneSource::SemanticSearch,
                });
            }
        }

        // Note search results — extract file paths from anchor entities
        match note_results {
            Ok(notes) => {
                for note in notes {
                    for anchor in &note.anchor_entities {
                        // Anchor entities that look like file paths
                        if anchor.contains('/') || anchor.contains('.') {
                            zones.push(RelevantZone {
                                file_path: anchor.clone(),
                                functions: vec![],
                                relevance_score: 0.5, // Lower score for note-sourced zones
                                source: ZoneSource::NoteReference,
                            });
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Planner: note search failed: {}", e);
            }
        }

        Ok(zones)
    }

    /// Orchestrate zone identification: explicit entries take priority,
    /// otherwise fall back to semantic search. Dedup by file_path (keep best score),
    /// sort by score descending, limit to MAX_ZONES.
    async fn identify_zones(&self, request: &PlanRequest) -> Result<Vec<RelevantZone>> {
        let root_path = request.root_path.as_deref();
        let zones = if let Some(ref entries) = request.entry_points {
            if !entries.is_empty() {
                self.resolve_explicit_entries(entries, request.project_id, root_path)
                    .await?
            } else {
                self.search_semantic_zones(&request.description, request.project_slug.as_deref(), Some(request.project_id))
                    .await?
            }
        } else {
            self.search_semantic_zones(&request.description, request.project_slug.as_deref(), Some(request.project_id))
                .await?
        };

        // Normalize all zone file paths to absolute using project root_path.
        // This ensures dedup works correctly even when Meilisearch returns
        // both relative and absolute paths for the same file.
        let zones: Vec<RelevantZone> = zones
            .into_iter()
            .map(|mut zone| {
                zone.file_path = resolve_path(root_path, &zone.file_path);
                zone
            })
            .collect();

        // Dedup by file_path — keep the highest relevance_score
        let mut best: HashMap<String, RelevantZone> = HashMap::new();
        for zone in zones {
            let entry = best.entry(zone.file_path.clone()).or_insert(zone.clone());
            if zone.relevance_score > entry.relevance_score {
                *entry = zone;
            }
        }

        let mut deduped: Vec<RelevantZone> = best.into_values().collect();
        deduped.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deduped.truncate(MAX_ZONES);
        Ok(deduped)
    }

    // ========================================================================
    // Phase 2 — DAG construction (Task 3)
    // ========================================================================

    /// Expand dependencies for each zone: find impacted files (files that import
    /// OR call functions from this zone's file) and direct imports.
    /// Also discovers community peers for files with 0 structural dependents.
    /// Returns (dependents_map, imports_map, community_peers_map).
    async fn expand_dependencies(
        &self,
        zones: &[RelevantZone],
        project_id: Uuid,
    ) -> Result<(
        HashMap<String, Vec<String>>,
        HashMap<String, Vec<String>>,
        HashMap<String, Vec<String>>,
    )> {
        let mut dependents_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut imports_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut community_peers_map: HashMap<String, Vec<String>> = HashMap::new();

        for zone in zones {
            let (dependents, imports) = tokio::join!(
                self.neo4j.find_impacted_files(
                    &zone.file_path,
                    MAX_DEPENDENCY_DEPTH,
                    Some(project_id)
                ),
                self.neo4j.get_file_direct_imports(&zone.file_path),
            );

            let dep_count = dependents.as_ref().map(|d| d.len()).unwrap_or(0);
            if let Ok(deps) = dependents {
                dependents_map.insert(zone.file_path.clone(), deps);
            }
            if let Ok(imps) = imports {
                imports_map.insert(
                    zone.file_path.clone(),
                    imps.into_iter().map(|i| i.path).collect(),
                );
            }

            // Community expansion: if file has 0 structural dependents and has a
            // community_id, find peer files in the same community (best-effort).
            if dep_count == 0 {
                if let Ok(Some(analytics)) = self
                    .neo4j
                    .get_node_analytics(&zone.file_path, "file")
                    .await
                {
                    if let Some(community_id) = analytics.community_id {
                        // Fetch all communities for the project, find the matching one
                        if let Ok(communities) =
                            self.neo4j.get_project_communities(project_id).await
                        {
                            let zone_files: HashSet<&str> =
                                zones.iter().map(|z| z.file_path.as_str()).collect();
                            if let Some(community) =
                                communities.iter().find(|c| c.community_id == community_id)
                            {
                                let peers: Vec<String> = community
                                    .key_files
                                    .iter()
                                    .filter(|f| {
                                        *f != &zone.file_path && !zone_files.contains(f.as_str())
                                    })
                                    .cloned()
                                    .collect();
                                if !peers.is_empty() {
                                    tracing::debug!(
                                        "Planner: community expansion for {} (community_id={}) → {:?}",
                                        zone.file_path,
                                        community_id,
                                        peers
                                    );
                                    community_peers_map
                                        .insert(zone.file_path.clone(), peers);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((dependents_map, imports_map, community_peers_map))
    }

    /// Build the modification DAG from zones and their dependencies.
    /// Edge A→B means "A must be modified before B" (B depends on A).
    /// Cycles are detected and broken by skipping the back-edge.
    async fn build_dag(
        &self,
        zones: &[RelevantZone],
        dependents_map: &HashMap<String, Vec<String>>,
        community_peers_map: &HashMap<String, Vec<String>>,
        project_id: Uuid,
    ) -> Result<ModificationDag> {
        let zone_files: HashSet<String> = zones.iter().map(|z| z.file_path.clone()).collect();

        // Fetch project percentiles once for GDS-based risk scoring (best-effort)
        let percentiles = self
            .neo4j
            .get_project_percentiles(project_id)
            .await
            .ok();

        // Build nodes for all zone files
        let mut nodes: HashMap<String, DagNode> = HashMap::new();
        for zone in zones {
            let symbols = match self.neo4j.get_file_symbol_names(&zone.file_path).await {
                Ok(s) => {
                    let mut all = s.functions;
                    all.extend(s.structs);
                    all.extend(s.traits);
                    all.extend(s.enums);
                    all
                }
                Err(_) => zone.functions.clone(),
            };

            let deps = dependents_map
                .get(&zone.file_path)
                .cloned()
                .unwrap_or_default();
            let dependents_count = deps.len();
            let test_files: Vec<String> = deps
                .iter()
                .filter(|p| p.contains("test"))
                .cloned()
                .collect();

            // Calculate risk based on caller count for main symbols
            let mut risk = if dependents_count > HIGH_RISK_THRESHOLD {
                RiskLevel::High
            } else if dependents_count > MEDIUM_RISK_THRESHOLD {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            };

            // Also check function caller counts
            for symbol in symbols.iter().take(3) {
                if let Ok(count) = self
                    .neo4j
                    .get_function_caller_count(symbol, Some(project_id))
                    .await
                {
                    let symbol_risk = if count > HIGH_RISK_THRESHOLD as i64 {
                        RiskLevel::High
                    } else if count > MEDIUM_RISK_THRESHOLD as i64 {
                        RiskLevel::Medium
                    } else {
                        RiskLevel::Low
                    };
                    risk = risk.max(symbol_risk);
                }
            }

            // Fetch GDS analytics: community_id + pagerank/betweenness for risk scoring
            let analytics = self
                .neo4j
                .get_node_analytics(&zone.file_path, "file")
                .await
                .ok()
                .flatten();

            let community_id = analytics
                .as_ref()
                .and_then(|a| a.community_id)
                .map(|c| c as u32);

            // GDS risk boost: if pagerank or betweenness reaches/exceeds project p95, boost risk
            if let (Some(ref a), Some(ref p)) = (&analytics, &percentiles) {
                if let Some(pr) = a.pagerank {
                    if p.pagerank_p95 > 0.0 && pr >= p.pagerank_p95 {
                        tracing::debug!(
                            "Planner: GDS risk boost for {} — pagerank {:.6} >= p95 {:.6}",
                            zone.file_path,
                            pr,
                            p.pagerank_p95
                        );
                        risk = risk.max(RiskLevel::Medium);
                    }
                }
                if let Some(bw) = a.betweenness {
                    if p.betweenness_p95 > 0.0 && bw >= p.betweenness_p95 {
                        tracing::debug!(
                            "Planner: GDS bridge risk boost for {} — betweenness {:.6} >= p95 {:.6}",
                            zone.file_path,
                            bw,
                            p.betweenness_p95
                        );
                        risk = risk.max(RiskLevel::High);
                    }
                }
            }

            // Community peers from expand_dependencies
            let community_peers = community_peers_map
                .get(&zone.file_path)
                .cloned()
                .unwrap_or_default();

            nodes.insert(
                zone.file_path.clone(),
                DagNode {
                    file_path: zone.file_path.clone(),
                    symbols,
                    risk,
                    test_files,
                    dependents_count,
                    community_id,
                    community_peers,
                },
            );
        }

        // Build edges: if zone A has dependent B (B imports A), and B is also
        // a zone file, then A→B (A must be modified before B)
        let mut edges = Vec::new();
        for zone in zones {
            if let Some(deps) = dependents_map.get(&zone.file_path) {
                for dep in deps {
                    if zone_files.contains(dep) && dep != &zone.file_path {
                        edges.push((zone.file_path.clone(), dep.clone()));
                    }
                }
            }
        }

        // Detect and break cycles via DFS
        let edges = Self::break_cycles(&nodes, edges);

        Ok(ModificationDag { nodes, edges })
    }

    /// Break cycles in the edge list using DFS. Returns edges with back-edges removed.
    fn break_cycles(
        nodes: &HashMap<String, DagNode>,
        edges: Vec<(String, String)>,
    ) -> Vec<(String, String)> {
        // Build adjacency list
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, to) in &edges {
            adj.entry(from.as_str()).or_default().push(to.as_str());
        }

        // DFS cycle detection
        let mut visited: HashSet<&str> = HashSet::new();
        let mut in_stack: HashSet<&str> = HashSet::new();
        let mut back_edges: HashSet<(String, String)> = HashSet::new();

        fn dfs<'a>(
            node: &'a str,
            adj: &HashMap<&'a str, Vec<&'a str>>,
            visited: &mut HashSet<&'a str>,
            in_stack: &mut HashSet<&'a str>,
            back_edges: &mut HashSet<(String, String)>,
        ) {
            visited.insert(node);
            in_stack.insert(node);

            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        dfs(neighbor, adj, visited, in_stack, back_edges);
                    } else if in_stack.contains(neighbor) {
                        // Back edge found — this creates a cycle
                        back_edges.insert((node.to_string(), neighbor.to_string()));
                    }
                }
            }

            in_stack.remove(node);
        }

        for node_key in nodes.keys() {
            if !visited.contains(node_key.as_str()) {
                dfs(
                    node_key.as_str(),
                    &adj,
                    &mut visited,
                    &mut in_stack,
                    &mut back_edges,
                );
            }
        }

        // Filter out back edges
        edges
            .into_iter()
            .filter(|e| !back_edges.contains(e))
            .collect()
    }

    // ========================================================================
    // Phase 3 — Topological sort and phase computation (Task 4)
    // ========================================================================

    /// Kahn's algorithm: topological sort by levels.
    /// Returns Vec<Vec<String>> where each inner vec is a level of nodes
    /// that can be processed in parallel.
    fn topological_sort(dag: &ModificationDag) -> Vec<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        // Initialize in-degrees to 0 for all nodes
        for key in dag.nodes.keys() {
            in_degree.entry(key.as_str()).or_insert(0);
            adj.entry(key.as_str()).or_default();
        }

        // Compute in-degrees from edges
        for (from, to) in &dag.edges {
            adj.entry(from.as_str()).or_default().push(to.as_str());
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
        }

        let mut levels = Vec::new();
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&node, _)| node)
            .collect();

        // Sort initial queue for deterministic ordering
        let mut initial: Vec<&str> = queue.drain(..).collect();
        initial.sort();
        queue.extend(initial);

        while !queue.is_empty() {
            let level: Vec<&str> = queue.drain(..).collect();
            let mut next_queue = Vec::new();

            for &node in &level {
                if let Some(neighbors) = adj.get(node) {
                    for &neighbor in neighbors {
                        if let Some(deg) = in_degree.get_mut(neighbor) {
                            *deg -= 1;
                            if *deg == 0 {
                                next_queue.push(neighbor);
                            }
                        }
                    }
                }
            }

            levels.push(level.into_iter().map(|s| s.to_string()).collect());

            // Sort next level for deterministic ordering
            next_queue.sort();
            queue.extend(next_queue);
        }

        levels
    }

    /// Generate a human-readable description for a phase based on its files.
    fn generate_phase_description(files: &[String]) -> String {
        if files.len() == 1 {
            let file = &files[0];
            let basename = file.rsplit('/').next().unwrap_or(file);
            format!("Modify {}", basename)
        } else if files.iter().all(|f| f.contains("test")) {
            "Update test files".to_string()
        } else if files.iter().all(|f| f.contains("model")) {
            "Update data models".to_string()
        } else if files.iter().any(|f| f.contains("handler")) {
            "Update handlers and endpoints".to_string()
        } else {
            format!("Modify {} files in parallel", files.len())
        }
    }

    /// Group files in a topo-sort level by community_id.
    ///
    /// Files in the same community are grouped into a single branch.
    /// If NO files have community_id (analytics not computed), falls back to
    /// one file per group (preserving old parallel behavior).
    /// Two community groups are only kept separate (parallel) if there's no
    /// direct edge between them in the DAG (truly independent).
    fn group_by_community(level: &[String], dag: &ModificationDag) -> Vec<Vec<String>> {
        // Check if any file has community_id; if none do, fall back to one-per-group
        let has_any_community = level
            .iter()
            .any(|f| dag.nodes.get(f).and_then(|n| n.community_id).is_some());

        if !has_any_community {
            // No community data → each file is its own group (old behavior)
            let mut groups: Vec<Vec<String>> = level.iter().map(|f| vec![f.clone()]).collect();
            groups.sort_by(|a, b| a[0].cmp(&b[0]));
            return groups;
        }

        // Build edge set for quick lookup
        let edge_set: HashSet<(&str, &str)> = dag
            .edges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();

        // Group files by community_id
        let mut community_groups: HashMap<Option<u32>, Vec<String>> = HashMap::new();
        for file in level {
            let cid = dag.nodes.get(file).and_then(|n| n.community_id);
            community_groups.entry(cid).or_default().push(file.clone());
        }

        let mut groups: Vec<Vec<String>> = community_groups.into_values().collect();

        // Merge groups that have cross-community edges (not truly independent)
        // Use a union-find approach: if any edge connects files from two groups, merge them
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..groups.len() {
                for j in (i + 1)..groups.len() {
                    let has_cross_edge = groups[i].iter().any(|a| {
                        groups[j].iter().any(|b| {
                            edge_set.contains(&(a.as_str(), b.as_str()))
                                || edge_set.contains(&(b.as_str(), a.as_str()))
                        })
                    });
                    if has_cross_edge {
                        let to_merge = groups.remove(j);
                        groups[i].extend(to_merge);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }

        // Sort groups deterministically (by first file path)
        for group in &mut groups {
            group.sort();
        }
        groups.sort_by(|a, b| a[0].cmp(&b[0]));

        groups
    }

    /// Compute phases from topological sort levels + DAG nodes.
    /// Uses community-aware grouping: files from independent communities
    /// become separate parallel branches, while same-community or
    /// cross-dependent files are grouped together.
    fn compute_phases(levels: &[Vec<String>], dag: &ModificationDag) -> Vec<Phase> {
        let mut phases = Vec::new();

        for (i, level) in levels.iter().enumerate() {
            let description = Self::generate_phase_description(level);

            if level.len() == 1 {
                // Single node — sequential phase
                let file = &level[0];
                let node = dag.nodes.get(file);
                phases.push(Phase {
                    phase_number: i + 1,
                    description,
                    parallel: false,
                    modifications: vec![Modification {
                        file: file.clone(),
                        reason: format!(
                            "Modify {} ({})",
                            file.rsplit('/').next().unwrap_or(file),
                            node.map(|n| format!("{} symbols", n.symbols.len()))
                                .unwrap_or_default()
                        ),
                        symbols_affected: node.map(|n| n.symbols.clone()).unwrap_or_default(),
                        risk: node.map(|n| n.risk.clone()).unwrap_or_default(),
                        dependents_count: node.map(|n| n.dependents_count).unwrap_or(0),
                    }],
                    branches: vec![],
                });
            } else {
                // Multiple nodes — use community-aware grouping
                let groups = Self::group_by_community(level, dag);

                if groups.len() == 1 {
                    // All in one group — sequential phase (same community or cross-linked)
                    let modifications: Vec<Modification> = groups[0]
                        .iter()
                        .map(|file| {
                            let node = dag.nodes.get(file);
                            Modification {
                                file: file.clone(),
                                reason: format!(
                                    "Modify {} ({})",
                                    file.rsplit('/').next().unwrap_or(file),
                                    node.map(|n| format!("{} dependents", n.dependents_count))
                                        .unwrap_or_default()
                                ),
                                symbols_affected: node
                                    .map(|n| n.symbols.clone())
                                    .unwrap_or_default(),
                                risk: node.map(|n| n.risk.clone()).unwrap_or_default(),
                                dependents_count: node.map(|n| n.dependents_count).unwrap_or(0),
                            }
                        })
                        .collect();
                    phases.push(Phase {
                        phase_number: i + 1,
                        description,
                        parallel: false,
                        modifications,
                        branches: vec![],
                    });
                } else {
                    // Multiple independent groups — parallel branches
                    let branches: Vec<Branch> = groups
                        .iter()
                        .map(|group| {
                            let reason = if group.len() == 1 {
                                let file = &group[0];
                                let node = dag.nodes.get(file);
                                format!(
                                    "Modify {} ({})",
                                    file.rsplit('/').next().unwrap_or(file),
                                    node.map(|n| format!("{} dependents", n.dependents_count))
                                        .unwrap_or_default()
                                )
                            } else {
                                // Try to get community label from first node
                                let cid = dag.nodes.get(&group[0]).and_then(|n| n.community_id);
                                match cid {
                                    Some(id) => format!("Community {} ({} files)", id, group.len()),
                                    None => format!("{} independent files", group.len()),
                                }
                            };
                            Branch {
                                files: group.clone(),
                                reason,
                            }
                        })
                        .collect();
                    phases.push(Phase {
                        phase_number: i + 1,
                        description,
                        parallel: true,
                        modifications: vec![],
                        branches,
                    });
                }
            }
        }

        phases
    }

    /// Collect relevant notes (gotchas, guidelines) for the files in the DAG.
    async fn collect_notes(&self, dag: &ModificationDag) -> Vec<PlannerNote> {
        let mut planner_notes = Vec::new();
        let mut seen_note_ids: HashSet<String> = HashSet::new();

        for file_path in dag.nodes.keys() {
            if let Ok(ctx) = self
                .note_manager
                .get_context_notes(&EntityType::File, file_path, 2, 0.3)
                .await
            {
                for note in ctx.direct_notes {
                    let note_id = note.id.to_string();
                    if seen_note_ids.contains(&note_id) {
                        continue;
                    }
                    seen_note_ids.insert(note_id);
                    planner_notes.push(PlannerNote {
                        note_type: format!("{:?}", note.note_type).to_lowercase(),
                        content: note.content.clone(),
                        importance: format!("{:?}", note.importance).to_lowercase(),
                        source_entity: Some(file_path.clone()),
                    });
                }
                for pnote in ctx.propagated_notes {
                    let note_id = pnote.note.id.to_string();
                    if seen_note_ids.contains(&note_id) {
                        continue;
                    }
                    seen_note_ids.insert(note_id);
                    planner_notes.push(PlannerNote {
                        note_type: format!("{:?}", pnote.note.note_type).to_lowercase(),
                        content: pnote.note.content.clone(),
                        importance: format!("{:?}", pnote.note.importance).to_lowercase(),
                        source_entity: Some(file_path.clone()),
                    });
                }
            }
        }

        planner_notes
    }

    // ========================================================================
    // Phase 4 — Auto-create Plan MCP (Task 6)
    // ========================================================================

    /// Auto-create a Plan MCP with Tasks/Steps from the ImplementationPlan.
    async fn auto_create_plan(
        &self,
        plan: &ImplementationPlan,
        project_id: Uuid,
        description: &str,
    ) -> Result<Uuid> {
        // Create the plan
        let create_req = CreatePlanRequest {
            title: plan.summary.clone(),
            description: format!(
                "Auto-generated by ImplementationPlanner.\n\nOriginal request: {}\n\n{} phases, {} test files, risk: {:?}",
                description,
                plan.phases.len(),
                plan.test_files.len(),
                plan.total_risk
            ),
            priority: Some(5),
            constraints: None,
            project_id: Some(project_id),
        };

        let plan_node = self.plan_manager.create_plan(create_req, "planner").await?;
        let plan_id = plan_node.id;

        // Link to project
        self.neo4j.link_plan_to_project(plan_id, project_id).await?;

        // Create tasks for each phase, chained by dependencies
        let mut prev_task_id: Option<Uuid> = None;
        for phase in &plan.phases {
            let affected_files: Vec<String> = if phase.parallel {
                phase
                    .branches
                    .iter()
                    .flat_map(|b| b.files.clone())
                    .collect()
            } else {
                phase.modifications.iter().map(|m| m.file.clone()).collect()
            };

            let task_req = CreateTaskRequest {
                title: Some(phase.description.clone()),
                description: format!("Phase {} — {}", phase.phase_number, phase.description),
                priority: Some(5),
                tags: Some(vec!["auto-generated".to_string(), "planner".to_string()]),
                acceptance_criteria: Some(vec!["All modifications applied".to_string()]),
                affected_files: Some(affected_files),
                depends_on: prev_task_id.map(|id| vec![id]),
                steps: None,
                estimated_complexity: None,
            };

            let task_node = self.plan_manager.add_task(plan_id, task_req).await?;

            // Create steps for each modification or branch
            let mut step_order = 0u32;
            if phase.parallel {
                for branch in &phase.branches {
                    let step = StepNodeModel {
                        id: Uuid::new_v4(),
                        order: step_order,
                        description: branch.reason.clone(),
                        status: StepStatus::Pending,
                        verification: Some("cargo check".to_string()),
                        created_at: chrono::Utc::now(),
                        updated_at: None,
                        completed_at: None,
                    };
                    self.plan_manager.add_step(task_node.id, &step).await?;
                    step_order += 1;
                }
            } else {
                for modification in &phase.modifications {
                    let step = StepNodeModel {
                        id: Uuid::new_v4(),
                        order: step_order,
                        description: modification.reason.clone(),
                        status: StepStatus::Pending,
                        verification: Some("cargo check".to_string()),
                        created_at: chrono::Utc::now(),
                        updated_at: None,
                        completed_at: None,
                    };
                    self.plan_manager.add_step(task_node.id, &step).await?;
                    step_order += 1;
                }
            }

            prev_task_id = Some(task_node.id);
        }

        Ok(plan_id)
    }

    // ========================================================================
    // Main entry point
    // ========================================================================

    /// Plan an implementation based on the knowledge graph.
    /// Orchestrates: identify_zones → build_dag → topo_sort → compute_phases → collect_notes.
    pub async fn plan_implementation(&self, request: PlanRequest) -> Result<ImplementationPlan> {
        // Step 1: Identify relevant zones
        let zones = self.identify_zones(&request).await?;
        if zones.is_empty() {
            return Ok(ImplementationPlan {
                summary: "No relevant code zones found for the given description.".to_string(),
                phases: vec![],
                test_files: vec![],
                notes: vec![],
                total_risk: RiskLevel::Low,
                plan_id: None,
            });
        }

        // Step 2: Expand dependencies (includes community peer discovery)
        let (dependents_map, _imports_map, community_peers_map) =
            self.expand_dependencies(&zones, request.project_id).await?;

        // Step 3: Build DAG (with GDS risk scoring)
        let dag = self
            .build_dag(&zones, &dependents_map, &community_peers_map, request.project_id)
            .await?;

        // Step 4: Topological sort
        let levels = Self::topological_sort(&dag);

        // Step 5: Compute phases
        let phases = Self::compute_phases(&levels, &dag);

        // Step 6: Collect notes
        let notes = self.collect_notes(&dag).await;

        // Aggregate test files and risk
        let mut all_test_files: Vec<String> = dag
            .nodes
            .values()
            .flat_map(|n| n.test_files.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_test_files.sort();

        let total_risk = dag
            .nodes
            .values()
            .fold(RiskLevel::Low, |acc, n| acc.max(n.risk.clone()));

        let summary = format!(
            "Modifying {} files across {} phases",
            dag.nodes.len(),
            phases.len()
        );

        let mut plan = ImplementationPlan {
            summary,
            phases,
            test_files: all_test_files,
            notes,
            total_risk,
            plan_id: None,
        };

        // Step 7: Auto-create Plan MCP if requested
        if request.auto_create_plan == Some(true) {
            match self
                .auto_create_plan(&plan, request.project_id, &request.description)
                .await
            {
                Ok(plan_id) => {
                    plan.plan_id = Some(plan_id.to_string());
                }
                Err(e) => {
                    tracing::warn!("Failed to auto-create plan: {}", e);
                }
            }
        }

        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_request_serde_roundtrip() {
        let request = PlanRequest {
            project_id: Uuid::new_v4(),
            project_slug: Some("my-project".to_string()),
            description: "Add WebSocket support".to_string(),
            entry_points: Some(vec!["src/chat/mod.rs".to_string()]),
            scope: Some(PlanScope::Module),
            auto_create_plan: Some(false),
            root_path: Some("~/.openclaw/workspace/my-project".to_string()),
        };
        let json = serde_json::to_string(&request).unwrap();
        let parsed: PlanRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.description, request.description);
        assert_eq!(parsed.scope, request.scope);
    }

    #[test]
    fn test_implementation_plan_serde_roundtrip() {
        let plan = ImplementationPlan {
            summary: "Modify 3 files across 2 phases".to_string(),
            phases: vec![
                Phase {
                    phase_number: 1,
                    description: "Update data models".to_string(),
                    parallel: false,
                    modifications: vec![Modification {
                        file: "src/models.rs".to_string(),
                        reason: "Add WsMessage struct".to_string(),
                        symbols_affected: vec!["ChatMessage".to_string()],
                        risk: RiskLevel::Low,
                        dependents_count: 2,
                    }],
                    branches: vec![],
                },
                Phase {
                    phase_number: 2,
                    description: "Implement endpoints".to_string(),
                    parallel: true,
                    modifications: vec![],
                    branches: vec![
                        Branch {
                            files: vec!["src/api/ws.rs".to_string()],
                            reason: "REST WebSocket upgrade".to_string(),
                        },
                        Branch {
                            files: vec!["src/mcp/handlers.rs".to_string()],
                            reason: "MCP tool for WS".to_string(),
                        },
                    ],
                },
            ],
            test_files: vec!["src/tests/chat_tests.rs".to_string()],
            notes: vec![PlannerNote {
                note_type: "gotcha".to_string(),
                content: "Two backend instances share DB".to_string(),
                importance: "high".to_string(),
                source_entity: Some("src/db.rs".to_string()),
            }],
            total_risk: RiskLevel::Medium,
            plan_id: None,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let parsed: ImplementationPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.phases.len(), 2);
        assert_eq!(parsed.total_risk, RiskLevel::Medium);
        assert!(parsed.phases[1].parallel);
        assert_eq!(parsed.phases[1].branches.len(), 2);
    }

    #[test]
    fn test_risk_level_max() {
        assert_eq!(RiskLevel::Low.max(RiskLevel::Low), RiskLevel::Low);
        assert_eq!(RiskLevel::Low.max(RiskLevel::Medium), RiskLevel::Medium);
        assert_eq!(RiskLevel::Medium.max(RiskLevel::High), RiskLevel::High);
        assert_eq!(RiskLevel::High.max(RiskLevel::Low), RiskLevel::High);
    }

    #[test]
    fn test_plan_scope_default() {
        assert_eq!(PlanScope::default(), PlanScope::Module);
    }

    // ========================================================================
    // Helper to create a planner with seeded mock data
    // ========================================================================

    async fn setup_planner_with_data() -> (ImplementationPlanner, Uuid) {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FileNode, FunctionNode, ProjectNode, Visibility};

        let graph = MockGraphStore::new();

        // Create a project
        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
        };
        let pid = project.id;
        graph.create_project(&project).await.unwrap();

        // Seed files
        let files = vec![
            ("src/models.rs", "rust"),
            ("src/handlers.rs", "rust"),
            ("src/routes.rs", "rust"),
            ("src/tests/handler_test.rs", "rust"),
        ];
        for (path, lang) in &files {
            graph
                .upsert_file(&FileNode {
                    path: path.to_string(),
                    language: lang.to_string(),
                    hash: "abc".to_string(),
                    last_parsed: chrono::Utc::now(),
                    project_id: Some(pid),
                })
                .await
                .unwrap();
        }

        // Register files to project
        for (path, _) in &files {
            graph.link_file_to_project(path, pid).await.unwrap();
        }

        // Seed functions
        let fns = vec![
            ("fn_models_create", "src/models.rs", 10u32, 30u32),
            ("fn_handler_get", "src/handlers.rs", 5, 25),
            ("fn_routes_setup", "src/routes.rs", 1, 50),
        ];
        for (name, file_path, start, end) in &fns {
            graph
                .upsert_function(&FunctionNode {
                    name: name.to_string(),
                    file_path: file_path.to_string(),
                    line_start: *start,
                    line_end: *end,
                    visibility: Visibility::Public,
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    docstring: None,
                })
                .await
                .unwrap();
        }

        // Seed import relationships:
        // handlers.rs imports models.rs
        // routes.rs imports handlers.rs
        // tests/handler_test.rs imports handlers.rs
        graph
            .create_import_relationship("src/handlers.rs", "src/models.rs", "crate::models")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/routes.rs", "src/handlers.rs", "crate::handlers")
            .await
            .unwrap();
        graph
            .create_import_relationship(
                "src/tests/handler_test.rs",
                "src/handlers.rs",
                "crate::handlers",
            )
            .await
            .unwrap();

        let neo4j: Arc<dyn crate::neo4j::GraphStore> = Arc::new(graph);
        let meili: Arc<dyn crate::meilisearch::SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        let plan_manager = Arc::new(PlanManager::new(neo4j.clone(), meili.clone()));
        let note_manager = Arc::new(NoteManager::new(neo4j.clone(), meili.clone()));

        let planner = ImplementationPlanner::new(neo4j, meili, plan_manager, note_manager);
        (planner, pid)
    }

    // ========================================================================
    // Task 2 tests — Zone identification
    // ========================================================================

    #[tokio::test]
    async fn test_resolve_explicit_entries_file_path() {
        let (planner, pid) = setup_planner_with_data().await;

        let zones = planner
            .resolve_explicit_entries(&["src/models.rs".to_string()], pid, None)
            .await
            .unwrap();

        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].file_path, "src/models.rs");
        assert_eq!(zones[0].source, ZoneSource::ExplicitEntry);
        assert!((zones[0].relevance_score - 1.0).abs() < f64::EPSILON);
        // Should contain the function from that file
        assert!(zones[0].functions.contains(&"fn_models_create".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_explicit_entries_unknown_file() {
        let (planner, pid) = setup_planner_with_data().await;

        let zones = planner
            .resolve_explicit_entries(&["src/unknown.rs".to_string()], pid, None)
            .await
            .unwrap();

        // Should still return a zone even if file not in graph
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].file_path, "src/unknown.rs");
        assert!(zones[0].functions.is_empty());
    }

    #[tokio::test]
    async fn test_identify_zones_with_explicit_entries() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "Modify models".to_string(),
            entry_points: Some(vec![
                "src/models.rs".to_string(),
                "src/handlers.rs".to_string(),
            ]),
            scope: None,
            auto_create_plan: None,
            root_path: None,
        };

        let zones = planner.identify_zones(&request).await.unwrap();
        assert_eq!(zones.len(), 2);
        // Should be sorted by score (all 1.0, so order by dedup)
        let file_paths: Vec<&str> = zones.iter().map(|z| z.file_path.as_str()).collect();
        assert!(file_paths.contains(&"src/models.rs"));
        assert!(file_paths.contains(&"src/handlers.rs"));
    }

    #[tokio::test]
    async fn test_identify_zones_dedup_by_file() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "Modify models".to_string(),
            entry_points: Some(vec![
                "src/models.rs".to_string(),
                "src/models.rs".to_string(), // duplicate
            ]),
            scope: None,
            auto_create_plan: None,
            root_path: None,
        };

        let zones = planner.identify_zones(&request).await.unwrap();
        assert_eq!(zones.len(), 1); // Deduped
        assert_eq!(zones[0].file_path, "src/models.rs");
    }

    #[tokio::test]
    async fn test_identify_zones_no_entries_fallback_semantic() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "something that does not match".to_string(),
            entry_points: None,
            scope: None,
            auto_create_plan: None,
            root_path: None,
        };

        // With empty mocks, should return empty (no semantic matches)
        let zones = planner.identify_zones(&request).await.unwrap();
        assert!(zones.is_empty());
    }

    // ========================================================================
    // Task 3 tests — DAG construction
    // ========================================================================

    #[tokio::test]
    async fn test_build_dag_linear() {
        let (planner, pid) = setup_planner_with_data().await;

        // Create zones for models → handlers → routes (linear dependency chain)
        let zones = vec![
            RelevantZone {
                file_path: "src/models.rs".to_string(),
                functions: vec![],
                relevance_score: 1.0,
                source: ZoneSource::ExplicitEntry,
            },
            RelevantZone {
                file_path: "src/handlers.rs".to_string(),
                functions: vec![],
                relevance_score: 1.0,
                source: ZoneSource::ExplicitEntry,
            },
            RelevantZone {
                file_path: "src/routes.rs".to_string(),
                functions: vec![],
                relevance_score: 1.0,
                source: ZoneSource::ExplicitEntry,
            },
        ];

        let (dependents_map, _, community_peers_map) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, &community_peers_map, pid)
            .await
            .unwrap();

        assert_eq!(dag.nodes.len(), 3);
        // models.rs → handlers.rs (handlers imports models)
        // handlers.rs → routes.rs (routes imports handlers)
        assert_eq!(dag.edges.len(), 2);
        assert!(dag
            .edges
            .contains(&("src/models.rs".to_string(), "src/handlers.rs".to_string())));
        assert!(dag
            .edges
            .contains(&("src/handlers.rs".to_string(), "src/routes.rs".to_string())));
    }

    #[tokio::test]
    async fn test_build_dag_test_files_detected() {
        let (planner, pid) = setup_planner_with_data().await;

        let zones = vec![RelevantZone {
            file_path: "src/handlers.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, &community_peers_map, pid)
            .await
            .unwrap();

        let node = dag.nodes.get("src/handlers.rs").unwrap();
        // tests/handler_test.rs imports handlers.rs, should be in test_files
        assert!(node
            .test_files
            .contains(&"src/tests/handler_test.rs".to_string()));
    }

    #[tokio::test]
    async fn test_break_cycles() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "a.rs".to_string(),
            DagNode {
                file_path: "a.rs".to_string(),
                symbols: vec![],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 0,
                community_id: None,
                    community_peers: vec![],
            },
        );
        nodes.insert(
            "b.rs".to_string(),
            DagNode {
                file_path: "b.rs".to_string(),
                symbols: vec![],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 0,
                community_id: None,
                    community_peers: vec![],
            },
        );

        // Cycle: a → b → a
        let edges = vec![
            ("a.rs".to_string(), "b.rs".to_string()),
            ("b.rs".to_string(), "a.rs".to_string()),
        ];

        let result = ImplementationPlanner::break_cycles(&nodes, edges);
        // One back-edge should be removed
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_risk_calculation() {
        let (planner, pid) = setup_planner_with_data().await;

        let zones = vec![RelevantZone {
            file_path: "src/handlers.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, &community_peers_map, pid)
            .await
            .unwrap();

        let node = dag.nodes.get("src/handlers.rs").unwrap();
        // 2 dependents (routes.rs + tests/handler_test.rs) — below medium threshold
        assert_eq!(node.risk, RiskLevel::Low);
    }

    // ========================================================================
    // Task 4 tests — Topological sort and phases
    // ========================================================================

    #[test]
    fn test_topological_sort_linear() {
        let mut nodes = HashMap::new();
        for name in &["a.rs", "b.rs", "c.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: None,
                    community_peers: vec![],
                },
            );
        }
        // a → b → c
        let edges = vec![
            ("a.rs".to_string(), "b.rs".to_string()),
            ("b.rs".to_string(), "c.rs".to_string()),
        ];
        let dag = ModificationDag { nodes, edges };
        let levels = ImplementationPlanner::topological_sort(&dag);

        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec!["a.rs"]);
        assert_eq!(levels[1], vec!["b.rs"]);
        assert_eq!(levels[2], vec!["c.rs"]);
    }

    #[test]
    fn test_topological_sort_fan_out() {
        let mut nodes = HashMap::new();
        for name in &["a.rs", "b.rs", "c.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: None,
                    community_peers: vec![],
                },
            );
        }
        // a → b, a → c (fan-out)
        let edges = vec![
            ("a.rs".to_string(), "b.rs".to_string()),
            ("a.rs".to_string(), "c.rs".to_string()),
        ];
        let dag = ModificationDag { nodes, edges };
        let levels = ImplementationPlanner::topological_sort(&dag);

        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec!["a.rs"]);
        // b and c should be in the same level (parallel)
        let mut level2 = levels[1].clone();
        level2.sort();
        assert_eq!(level2, vec!["b.rs", "c.rs"]);
    }

    #[test]
    fn test_compute_phases_sequential() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "a.rs".to_string(),
            DagNode {
                file_path: "a.rs".to_string(),
                symbols: vec!["Foo".to_string()],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 1,
                community_id: None,
                    community_peers: vec![],
            },
        );
        nodes.insert(
            "b.rs".to_string(),
            DagNode {
                file_path: "b.rs".to_string(),
                symbols: vec!["Bar".to_string()],
                risk: RiskLevel::Medium,
                test_files: vec![],
                dependents_count: 0,
                community_id: None,
                    community_peers: vec![],
            },
        );
        let dag = ModificationDag {
            nodes,
            edges: vec![("a.rs".to_string(), "b.rs".to_string())],
        };
        let levels = vec![vec!["a.rs".to_string()], vec!["b.rs".to_string()]];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);

        assert_eq!(phases.len(), 2);
        assert!(!phases[0].parallel);
        assert_eq!(phases[0].modifications.len(), 1);
        assert_eq!(phases[0].modifications[0].file, "a.rs");
        assert!(!phases[1].parallel);
        assert_eq!(phases[1].modifications.len(), 1);
        assert_eq!(phases[1].modifications[0].file, "b.rs");
    }

    #[test]
    fn test_compute_phases_parallel() {
        let mut nodes = HashMap::new();
        for name in &["a.rs", "b.rs", "c.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: None,
                    community_peers: vec![],
                },
            );
        }
        let dag = ModificationDag {
            nodes,
            edges: vec![
                ("a.rs".to_string(), "b.rs".to_string()),
                ("a.rs".to_string(), "c.rs".to_string()),
            ],
        };
        let levels = vec![
            vec!["a.rs".to_string()],
            vec!["b.rs".to_string(), "c.rs".to_string()],
        ];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);

        assert_eq!(phases.len(), 2);
        assert!(!phases[0].parallel);
        assert!(phases[1].parallel);
        assert_eq!(phases[1].branches.len(), 2);
    }

    // ========================================================================
    // Task 3.4 — Community-aware phase tests
    // ========================================================================

    #[test]
    fn test_community_grouping_two_independent_communities() {
        // 6 files in 2 independent communities → 2 parallel branches
        let mut nodes = HashMap::new();
        // Community 1: a1.rs, a2.rs, a3.rs
        for name in &["a1.rs", "a2.rs", "a3.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: Some(1),
                    community_peers: vec![],
                },
            );
        }
        // Community 2: b1.rs, b2.rs, b3.rs
        for name in &["b1.rs", "b2.rs", "b3.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: Some(2),
                    community_peers: vec![],
                },
            );
        }

        // No edges between communities — all in same topo level
        let dag = ModificationDag {
            nodes,
            edges: vec![],
        };
        let level: Vec<String> = vec!["a1.rs", "a2.rs", "a3.rs", "b1.rs", "b2.rs", "b3.rs"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let groups = ImplementationPlanner::group_by_community(&level, &dag);
        assert_eq!(
            groups.len(),
            2,
            "Should have 2 independent community groups"
        );
        // Each group should have 3 files
        assert_eq!(groups[0].len(), 3);
        assert_eq!(groups[1].len(), 3);

        // Compute phases — should produce parallel branches
        let levels = vec![level];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);
        assert_eq!(phases.len(), 1);
        assert!(
            phases[0].parallel,
            "Independent communities should produce parallel phase"
        );
        assert_eq!(phases[0].branches.len(), 2);
        // Branch reason should mention community ID
        assert!(
            phases[0]
                .branches
                .iter()
                .any(|b| b.reason.contains("Community")),
            "Branch reason should reference community"
        );
    }

    #[test]
    fn test_community_grouping_same_community_sequential() {
        // 4 files all in same community → single group → sequential (not parallel)
        let mut nodes = HashMap::new();
        for name in &["x1.rs", "x2.rs", "x3.rs", "x4.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: Some(5),
                    community_peers: vec![],
                },
            );
        }

        let dag = ModificationDag {
            nodes,
            edges: vec![],
        };
        let level: Vec<String> = vec!["x1.rs", "x2.rs", "x3.rs", "x4.rs"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let groups = ImplementationPlanner::group_by_community(&level, &dag);
        assert_eq!(
            groups.len(),
            1,
            "Same community files should be in one group"
        );
        assert_eq!(groups[0].len(), 4);

        // Compute phases — single group → not parallel
        let levels = vec![level];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);
        assert_eq!(phases.len(), 1);
        assert!(
            !phases[0].parallel,
            "Same community should produce sequential phase"
        );
        assert_eq!(phases[0].modifications.len(), 4);
    }

    #[test]
    fn test_community_grouping_no_community_data_fallback() {
        // No files have community_id → old behavior: each file is its own group
        let mut nodes = HashMap::new();
        for name in &["p.rs", "q.rs", "r.rs"] {
            nodes.insert(
                name.to_string(),
                DagNode {
                    file_path: name.to_string(),
                    symbols: vec![],
                    risk: RiskLevel::Low,
                    test_files: vec![],
                    dependents_count: 0,
                    community_id: None,
                    community_peers: vec![],
                },
            );
        }

        let dag = ModificationDag {
            nodes,
            edges: vec![],
        };
        let level: Vec<String> = vec!["p.rs", "q.rs", "r.rs"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let groups = ImplementationPlanner::group_by_community(&level, &dag);
        // Without community data, each file is its own group (old parallel behavior)
        assert_eq!(
            groups.len(),
            3,
            "No community data → one file per group (old behavior)"
        );
        for group in &groups {
            assert_eq!(group.len(), 1);
        }

        // Compute phases — 3 groups → parallel with 3 branches
        let levels = vec![level];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);
        assert_eq!(phases.len(), 1);
        assert!(
            phases[0].parallel,
            "No community data → parallel branches (old behavior)"
        );
        assert_eq!(phases[0].branches.len(), 3);
    }

    #[test]
    fn test_community_grouping_cross_community_edge_merges() {
        // 2 communities with a cross-community edge → should merge into 1 group
        let mut nodes = HashMap::new();
        nodes.insert(
            "alpha.rs".to_string(),
            DagNode {
                file_path: "alpha.rs".to_string(),
                symbols: vec![],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 1,
                community_id: Some(10),
                community_peers: vec![],
            },
        );
        nodes.insert(
            "beta.rs".to_string(),
            DagNode {
                file_path: "beta.rs".to_string(),
                symbols: vec![],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 0,
                community_id: Some(20),
                community_peers: vec![],
            },
        );
        nodes.insert(
            "gamma.rs".to_string(),
            DagNode {
                file_path: "gamma.rs".to_string(),
                symbols: vec![],
                risk: RiskLevel::Low,
                test_files: vec![],
                dependents_count: 0,
                community_id: Some(20),
                community_peers: vec![],
            },
        );

        // Cross-community edge: alpha (community 10) → beta (community 20)
        let dag = ModificationDag {
            nodes,
            edges: vec![("alpha.rs".to_string(), "beta.rs".to_string())],
        };
        let level: Vec<String> = vec!["alpha.rs", "beta.rs", "gamma.rs"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let groups = ImplementationPlanner::group_by_community(&level, &dag);
        // Community 10 (alpha) and community 20 (beta, gamma) should merge
        // because alpha→beta edge connects them
        assert_eq!(groups.len(), 1, "Cross-community edge should merge groups");
        assert_eq!(groups[0].len(), 3);

        // Compute phases — single group → not parallel
        let levels = vec![level];
        let phases = ImplementationPlanner::compute_phases(&levels, &dag);
        assert_eq!(phases.len(), 1);
        assert!(
            !phases[0].parallel,
            "Merged communities should be sequential"
        );
        assert_eq!(phases[0].modifications.len(), 3);
    }

    // ========================================================================
    // End-to-end test
    // ========================================================================

    #[tokio::test]
    async fn test_plan_implementation_end_to_end() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "Modify the model and handlers".to_string(),
            entry_points: Some(vec![
                "src/models.rs".to_string(),
                "src/handlers.rs".to_string(),
                "src/routes.rs".to_string(),
            ]),
            scope: Some(PlanScope::Module),
            auto_create_plan: None,
            root_path: None,
        };

        let plan = planner.plan_implementation(request).await.unwrap();

        // Should have phases
        assert!(!plan.phases.is_empty());
        // models.rs should come first (handlers and routes depend on it)
        assert_eq!(plan.phases[0].modifications[0].file, "src/models.rs");
        // test files should include the handler test
        assert!(plan
            .test_files
            .contains(&"src/tests/handler_test.rs".to_string()));
        // plan_id should be None (auto_create_plan not set)
        assert!(plan.plan_id.is_none());
    }

    #[tokio::test]
    async fn test_plan_implementation_empty_zones() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "something nonexistent".to_string(),
            entry_points: None, // No explicit entries, semantic search won't match
            scope: None,
            auto_create_plan: None,
            root_path: None,
        };

        let plan = planner.plan_implementation(request).await.unwrap();
        assert!(plan.phases.is_empty());
        assert_eq!(plan.total_risk, RiskLevel::Low);
    }

    #[tokio::test]
    async fn test_plan_implementation_with_auto_create() {
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "Add new endpoint".to_string(),
            entry_points: Some(vec![
                "src/models.rs".to_string(),
                "src/handlers.rs".to_string(),
            ]),
            scope: None,
            auto_create_plan: Some(true),
            root_path: None,
        };

        let plan = planner.plan_implementation(request).await.unwrap();
        // Should have a plan_id
        assert!(plan.plan_id.is_some());
        let plan_id_str = plan.plan_id.unwrap();
        assert!(!plan_id_str.is_empty());
        // Verify it's a valid UUID
        Uuid::parse_str(&plan_id_str).unwrap();
    }

    #[test]
    fn test_extract_search_keywords_french() {
        let result = extract_search_keywords("ajouter un endpoint REST pour les releases");
        assert_eq!(result, "endpoint REST releases");
    }

    #[test]
    fn test_extract_search_keywords_english() {
        let result = extract_search_keywords("add a new REST endpoint for releases");
        assert_eq!(result, "REST endpoint releases");
    }

    #[test]
    fn test_extract_search_keywords_technical() {
        // Technical terms should be preserved
        let result = extract_search_keywords("implement WebSocket streaming for chat sessions");
        assert_eq!(result, "WebSocket streaming chat sessions");
    }

    #[test]
    fn test_extract_search_keywords_code_terms_only() {
        let result = extract_search_keywords("get_release create_milestone");
        assert_eq!(result, "get_release create_milestone");
    }

    #[test]
    fn test_extract_search_keywords_all_stopwords_fallback() {
        // If all words are stop words, fall back to original
        let result = extract_search_keywords("un de la");
        assert_eq!(result, "un de la");
    }

    #[test]
    fn test_extract_search_keywords_mixed_punctuation() {
        let result = extract_search_keywords("ajouter l'endpoint /api/releases/:id");
        assert!(result.contains("endpoint"));
        assert!(result.contains("/api/releases/:id"));
    }

    // ========================================================================
    // Task 3 Step 5 — Vector search, community expansion, GDS risk tests
    // ========================================================================

    #[test]
    fn test_reciprocal_rank_fusion_basic() {
        // Two lists with some overlap
        let bm25 = vec![
            ("a.rs".to_string(), 1.0),
            ("b.rs".to_string(), 0.8),
            ("c.rs".to_string(), 0.6),
        ];
        let vector = vec![
            ("b.rs".to_string(), 0.95),
            ("d.rs".to_string(), 0.9),
            ("a.rs".to_string(), 0.7),
        ];

        let fused = reciprocal_rank_fusion(&bm25, &vector, 10);

        // b.rs appears in both lists → should be ranked highest (boosted by RRF)
        assert_eq!(fused[0].0, "b.rs", "b.rs should be top result (in both lists)");

        // a.rs also appears in both → should be second
        assert_eq!(fused[1].0, "a.rs", "a.rs should be second (in both lists)");

        // All 4 unique files should appear
        assert_eq!(fused.len(), 4);
        let paths: Vec<&str> = fused.iter().map(|r| r.0.as_str()).collect();
        assert!(paths.contains(&"c.rs"));
        assert!(paths.contains(&"d.rs"));
    }

    #[test]
    fn test_reciprocal_rank_fusion_no_overlap() {
        let bm25 = vec![("a.rs".to_string(), 1.0)];
        let vector = vec![("b.rs".to_string(), 0.9)];

        let fused = reciprocal_rank_fusion(&bm25, &vector, 10);
        assert_eq!(fused.len(), 2);
        // Scores should be equal (both rank 1 in their respective list)
        assert!((fused[0].1 - fused[1].1).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_rank_fusion_empty_input() {
        let empty: Vec<(String, f64)> = vec![];
        let bm25 = vec![("a.rs".to_string(), 1.0)];

        // One empty, one populated
        let fused = reciprocal_rank_fusion(&bm25, &empty, 10);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].0, "a.rs");

        // Both empty
        let fused = reciprocal_rank_fusion(&empty, &empty, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_reciprocal_rank_fusion_truncates_to_k() {
        let bm25: Vec<(String, f64)> = (0..20)
            .map(|i| (format!("file{}.rs", i), 1.0 - i as f64 * 0.01))
            .collect();
        let vector: Vec<(String, f64)> = vec![];

        let fused = reciprocal_rank_fusion(&bm25, &vector, 5);
        assert_eq!(fused.len(), 5);
    }

    /// Helper to create a planner with embedding provider and seeded analytics data
    async fn setup_planner_with_analytics() -> (ImplementationPlanner, Uuid) {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FileNode, FunctionNode, ProjectNode, Visibility};

        let graph = MockGraphStore::new();

        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: "test-project-gds".to_string(),
            slug: "test-project-gds".to_string(),
            root_path: "/tmp/test-gds".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
        };
        let pid = project.id;
        graph.create_project(&project).await.unwrap();

        // Seed files
        let files = vec![
            ("src/models.rs", "rust"),
            ("src/handlers.rs", "rust"),
            ("src/routes.rs", "rust"),
            ("src/client.rs", "rust"),
            ("src/utils.rs", "rust"),
            ("src/tests/handler_test.rs", "rust"),
        ];
        for (path, lang) in &files {
            graph
                .upsert_file(&FileNode {
                    path: path.to_string(),
                    language: lang.to_string(),
                    hash: "abc".to_string(),
                    last_parsed: chrono::Utc::now(),
                    project_id: Some(pid),
                })
                .await
                .unwrap();
            graph.link_file_to_project(path, pid).await.unwrap();
        }

        // Seed functions
        let fns = vec![
            ("fn_models_create", "src/models.rs", 10u32, 30u32),
            ("fn_handler_get", "src/handlers.rs", 5, 25),
            ("fn_routes_setup", "src/routes.rs", 1, 50),
            ("fn_client_fetch", "src/client.rs", 1, 40),
        ];
        for (name, file_path, start, end) in &fns {
            graph
                .upsert_function(&FunctionNode {
                    name: name.to_string(),
                    file_path: file_path.to_string(),
                    line_start: *start,
                    line_end: *end,
                    visibility: Visibility::Public,
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    docstring: None,
                })
                .await
                .unwrap();
        }

        // Import relationships:
        // handlers.rs imports models.rs
        // routes.rs imports handlers.rs
        // tests/handler_test.rs imports handlers.rs
        graph
            .create_import_relationship("src/handlers.rs", "src/models.rs", "crate::models")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/routes.rs", "src/handlers.rs", "crate::handlers")
            .await
            .unwrap();
        graph
            .create_import_relationship(
                "src/tests/handler_test.rs",
                "src/handlers.rs",
                "crate::handlers",
            )
            .await
            .unwrap();
        // Note: routes.rs has 0 dependents (nothing imports routes.rs)
        // Note: client.rs also has 0 dependents

        // Seed analytics: put routes.rs, handlers.rs, client.rs in same community (42)
        // models.rs in a different community (1)
        // routes.rs gets high pagerank, utils.rs gets high betweenness
        use crate::graph::models::FileAnalyticsUpdate;
        graph
            .batch_update_file_analytics(&[
                FileAnalyticsUpdate {
                    path: "src/models.rs".to_string(),
                    pagerank: 0.01,
                    betweenness: 0.001,
                    community_id: 1,
                    community_label: "data-models".to_string(),
                    clustering_coefficient: 0.0,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/handlers.rs".to_string(),
                    pagerank: 0.05,
                    betweenness: 0.01,
                    community_id: 42,
                    community_label: "api-layer".to_string(),
                    clustering_coefficient: 0.0,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/routes.rs".to_string(),
                    pagerank: 0.02,
                    betweenness: 0.005,
                    community_id: 42,
                    community_label: "api-layer".to_string(),
                    clustering_coefficient: 0.0,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/client.rs".to_string(),
                    pagerank: 0.03,
                    betweenness: 0.008,
                    community_id: 42,
                    community_label: "api-layer".to_string(),
                    clustering_coefficient: 0.0,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/utils.rs".to_string(),
                    pagerank: 0.50,       // Very high — above p95
                    betweenness: 0.90,    // Very high — bridge file
                    community_id: 99,
                    community_label: "utilities".to_string(),
                    clustering_coefficient: 0.0,
                    component_id: 0,
                },
            ])
            .await
            .unwrap();

        let neo4j: Arc<dyn crate::neo4j::GraphStore> = Arc::new(graph);
        let meili: Arc<dyn crate::meilisearch::SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        let plan_manager = Arc::new(PlanManager::new(neo4j.clone(), meili.clone()));
        let note_manager = Arc::new(NoteManager::new(neo4j.clone(), meili.clone()));

        let planner = ImplementationPlanner::new(neo4j, meili, plan_manager, note_manager);
        (planner, pid)
    }

    #[tokio::test]
    async fn test_community_expansion_for_zero_dependents() {
        let (planner, pid) = setup_planner_with_analytics().await;

        // routes.rs has 0 structural dependents but is in community 42 (api-layer)
        // with handlers.rs and client.rs as key_files
        let zones = vec![RelevantZone {
            file_path: "src/routes.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) =
            planner.expand_dependencies(&zones, pid).await.unwrap();

        // routes.rs should have 0 structural dependents
        let deps = dependents_map.get("src/routes.rs").unwrap();
        assert!(deps.is_empty(), "routes.rs should have 0 dependents");

        // Community expansion should find peers in community 42
        let peers = community_peers_map.get("src/routes.rs");
        assert!(
            peers.is_some(),
            "routes.rs should have community peers (same community 42)"
        );
        let peers = peers.unwrap();
        // handlers.rs and client.rs are in the same community
        // (they appear as key_files since the mock computes top files by pagerank)
        assert!(
            !peers.is_empty(),
            "Should find at least one community peer"
        );
        // Peers should NOT include routes.rs itself
        assert!(
            !peers.contains(&"src/routes.rs".to_string()),
            "Peers should not include the file itself"
        );
    }

    #[tokio::test]
    async fn test_community_expansion_skipped_when_has_dependents() {
        let (planner, pid) = setup_planner_with_analytics().await;

        // models.rs has dependents (handlers.rs imports it)
        // → community expansion should NOT trigger
        let zones = vec![RelevantZone {
            file_path: "src/models.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) =
            planner.expand_dependencies(&zones, pid).await.unwrap();

        // models.rs has dependents
        let deps = dependents_map.get("src/models.rs").unwrap();
        assert!(
            !deps.is_empty(),
            "models.rs should have dependents (handlers.rs)"
        );

        // No community expansion since it has dependents
        assert!(
            community_peers_map.get("src/models.rs").is_none(),
            "models.rs should NOT have community peers (has dependents)"
        );
    }

    #[tokio::test]
    async fn test_pagerank_boosts_risk() {
        let (planner, pid) = setup_planner_with_analytics().await;

        // utils.rs has very high pagerank (0.50) and betweenness (0.90)
        // which should exceed p95 thresholds computed from the project
        let zones = vec![RelevantZone {
            file_path: "src/utils.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) =
            planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, &community_peers_map, pid)
            .await
            .unwrap();

        let node = dag.nodes.get("src/utils.rs").unwrap();
        // utils.rs has 0 structural dependents → base risk = Low
        // BUT its pagerank (0.50) >> p95 → boosted to Medium
        // AND its betweenness (0.90) >> p95 → boosted to High (bridge)
        assert_eq!(
            node.risk,
            RiskLevel::High,
            "utils.rs risk should be High (betweenness > p95 = bridge file)"
        );
    }

    #[tokio::test]
    async fn test_gds_risk_no_boost_for_normal_files() {
        let (planner, pid) = setup_planner_with_analytics().await;

        // models.rs has normal pagerank (0.01) and betweenness (0.001)
        // → should NOT get a GDS risk boost
        let zones = vec![RelevantZone {
            file_path: "src/models.rs".to_string(),
            functions: vec![],
            relevance_score: 1.0,
            source: ZoneSource::ExplicitEntry,
        }];

        let (dependents_map, _, community_peers_map) =
            planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, &community_peers_map, pid)
            .await
            .unwrap();

        let node = dag.nodes.get("src/models.rs").unwrap();
        // models.rs has 1 dependent (handlers.rs) → below MEDIUM_RISK_THRESHOLD (3)
        // Its low pagerank/betweenness should NOT boost the risk
        assert_eq!(
            node.risk,
            RiskLevel::Low,
            "models.rs risk should remain Low (normal GDS metrics)"
        );
    }

    #[tokio::test]
    async fn test_planner_works_without_embeddings() {
        // Use the basic setup (no embedding provider, no analytics data)
        let (planner, pid) = setup_planner_with_data().await;

        let request = PlanRequest {
            project_id: pid,
            project_slug: Some("test-project".to_string()),
            description: "Modify models and handlers".to_string(),
            entry_points: Some(vec![
                "src/models.rs".to_string(),
                "src/handlers.rs".to_string(),
                "src/routes.rs".to_string(),
            ]),
            scope: None,
            auto_create_plan: None,
            root_path: None,
        };

        // Should work perfectly without embedding provider (graceful fallback)
        let plan = planner.plan_implementation(request).await.unwrap();

        assert!(!plan.phases.is_empty(), "Should produce phases");
        assert_eq!(
            plan.phases[0].modifications[0].file, "src/models.rs",
            "models.rs should still be first (dependency order)"
        );
        assert!(
            plan.test_files
                .contains(&"src/tests/handler_test.rs".to_string()),
            "Should still find test files"
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_merges_bm25_and_vector() {
        use crate::embeddings::mock::MockEmbeddingProvider;
        use crate::embeddings::EmbeddingProvider;
        use crate::meilisearch::indexes::CodeDocument;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FileNode, ProjectNode};

        let graph = MockGraphStore::new();
        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: "hybrid-test".to_string(),
            slug: "hybrid-test".to_string(),
            root_path: "/tmp/hybrid".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
        };
        let pid = project.id;
        graph.create_project(&project).await.unwrap();

        // File A: in Meilisearch (BM25 match for "webhook")
        // File B: has a similar embedding to "webhook" (vector match)
        // File C: in BOTH (BM25 + vector) → should be top result via RRF
        let files = vec!["src/webhook.rs", "src/events.rs", "src/notify.rs"];
        for path in &files {
            graph
                .upsert_file(&FileNode {
                    path: path.to_string(),
                    language: "rust".to_string(),
                    hash: "abc".to_string(),
                    last_parsed: chrono::Utc::now(),
                    project_id: Some(pid),
                })
                .await
                .unwrap();
            graph.link_file_to_project(path, pid).await.unwrap();
        }

        // Seed file embeddings for vector search
        let embedding_provider = MockEmbeddingProvider::new(384);
        let query_embedding = embedding_provider.embed_text("webhook notifications").await.unwrap();
        // Give notify.rs an embedding identical to the query → highest cosine similarity
        graph
            .set_file_embedding("src/notify.rs", &query_embedding, "mock")
            .await
            .unwrap();
        // Give events.rs a slightly different embedding
        let events_emb = embedding_provider.embed_text("event system hooks").await.unwrap();
        graph
            .set_file_embedding("src/events.rs", &events_emb, "mock")
            .await
            .unwrap();

        // Seed Meilisearch with BM25-matching documents
        let meili = crate::meilisearch::mock::MockSearchStore::new();
        meili
            .index_code(&CodeDocument {
                id: format!("{}:src/webhook.rs", pid),
                path: "src/webhook.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec!["send_webhook".to_string()],
                docstrings: "Send webhook notifications".to_string(),
                signatures: vec![],
                imports: vec![],
                project_id: pid.to_string(),
                project_slug: "hybrid-test".to_string(),
            })
            .await
            .unwrap();
        meili
            .index_code(&CodeDocument {
                id: format!("{}:src/notify.rs", pid),
                path: "src/notify.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec!["notify_webhook".to_string()],
                docstrings: "Handle webhook notifications".to_string(),
                signatures: vec![],
                imports: vec![],
                project_id: pid.to_string(),
                project_slug: "hybrid-test".to_string(),
            })
            .await
            .unwrap();

        let neo4j: Arc<dyn crate::neo4j::GraphStore> = Arc::new(graph);
        let meili: Arc<dyn crate::meilisearch::SearchStore> = Arc::new(meili);
        let plan_manager = Arc::new(PlanManager::new(neo4j.clone(), meili.clone()));
        let note_manager = Arc::new(NoteManager::new(neo4j.clone(), meili.clone()));

        let planner = ImplementationPlanner::new(neo4j, meili, plan_manager, note_manager)
            .with_embedding_provider(Arc::new(embedding_provider));

        // Search should fuse BM25 + vector results
        let zones = planner
            .search_semantic_zones("webhook notifications", Some("hybrid-test"), Some(pid))
            .await
            .unwrap();

        // notify.rs appears in BOTH BM25 (substring match "webhook") AND vector (identical embedding)
        // → should be boosted by RRF to top or near-top
        let paths: Vec<&str> = zones.iter().map(|z| z.file_path.as_str()).collect();
        assert!(
            !zones.is_empty(),
            "Hybrid search should return results"
        );
        assert!(
            paths.contains(&"src/notify.rs"),
            "notify.rs should appear (in both BM25 and vector)"
        );
        assert!(
            paths.contains(&"src/webhook.rs"),
            "webhook.rs should appear (BM25 match)"
        );

        // If notify.rs is in both lists, it should be ranked higher than webhook.rs (BM25 only)
        if let (Some(notify_idx), Some(webhook_idx)) = (
            paths.iter().position(|p| *p == "src/notify.rs"),
            paths.iter().position(|p| *p == "src/webhook.rs"),
        ) {
            assert!(
                notify_idx <= webhook_idx,
                "notify.rs (BM25+vector) should rank >= webhook.rs (BM25 only)"
            );
        }
    }
}
