//! Code exploration API handlers
//!
//! These endpoints provide intelligent code exploration using Neo4j graph queries
//! and Meilisearch semantic search - much more powerful than reading files directly.

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use super::handlers::{AppError, OrchestratorState};
use crate::neo4j::models::ConnectedFileNode;

// ============================================================================
// Code Search (Meilisearch)
// ============================================================================

#[derive(Deserialize)]
pub struct CodeSearchQuery {
    /// Search query (semantic search across code content, symbols, comments)
    pub query: String,
    /// Max results (default 10)
    pub limit: Option<usize>,
    /// Filter by language (rust, typescript, python, go)
    pub language: Option<String>,
}

/// Search code semantically across the codebase
///
/// Returns `SearchHit<CodeDocument>` directly so the frontend gets
/// the `{ document, score }` shape it expects.
pub async fn search_code(
    State(state): State<OrchestratorState>,
    Query(params): Query<CodeSearchQuery>,
) -> Result<
    Json<Vec<crate::meilisearch::indexes::SearchHit<crate::meilisearch::indexes::CodeDocument>>>,
    AppError,
> {
    let hits = state
        .orchestrator
        .meili()
        .search_code_with_scores(
            &params.query,
            params.limit.unwrap_or(10),
            params.language.as_deref(),
            None,
        )
        .await?;

    Ok(Json(hits))
}

// ============================================================================
// Symbol Lookup
// ============================================================================

#[derive(Serialize)]
pub struct FileSymbols {
    pub path: String,
    pub language: String,
    pub functions: Vec<FunctionSummary>,
    pub structs: Vec<StructSummary>,
    pub imports: Vec<String>,
}

#[derive(Serialize)]
pub struct FunctionSummary {
    pub name: String,
    pub signature: String,
    pub line: u32,
    pub is_async: bool,
    pub is_public: bool,
    pub complexity: u32,
    pub docstring: Option<String>,
}

#[derive(Serialize)]
pub struct StructSummary {
    pub name: String,
    pub line: u32,
    pub is_public: bool,
    pub docstring: Option<String>,
}

/// Get all symbols defined in a file without reading the entire file
pub async fn get_file_symbols(
    State(state): State<OrchestratorState>,
    Path(file_path): Path<String>,
) -> Result<Json<FileSymbols>, AppError> {
    let file_path = urlencoding::decode(&file_path)
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .to_string();

    let language = state
        .orchestrator
        .neo4j()
        .get_file_language(&file_path)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("File not found: {}", file_path)))?;

    let func_nodes = state
        .orchestrator
        .neo4j()
        .get_file_functions_summary(&file_path)
        .await?;

    let functions: Vec<FunctionSummary> = func_nodes
        .into_iter()
        .map(|f| FunctionSummary {
            name: f.name,
            signature: f.signature,
            line: f.line,
            is_async: f.is_async,
            is_public: f.is_public,
            complexity: f.complexity,
            docstring: f.docstring,
        })
        .collect();

    let struct_nodes = state
        .orchestrator
        .neo4j()
        .get_file_structs_summary(&file_path)
        .await?;

    let structs: Vec<StructSummary> = struct_nodes
        .into_iter()
        .map(|s| StructSummary {
            name: s.name,
            line: s.line,
            is_public: s.is_public,
            docstring: s.docstring,
        })
        .collect();

    let imports = state
        .orchestrator
        .neo4j()
        .get_file_import_paths_list(&file_path)
        .await?;

    Ok(Json(FileSymbols {
        path: file_path,
        language,
        functions,
        structs,
        imports,
    }))
}

// ============================================================================
// Find References
// ============================================================================

#[derive(Deserialize)]
pub struct FindReferencesQuery {
    /// Symbol name to find references for
    pub symbol: String,
    /// Limit results
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct SymbolReference {
    pub file_path: String,
    pub line: u32,
    pub context: String,
    pub reference_type: String, // "call", "import", "type_usage"
}

/// Find all references to a symbol across the codebase
pub async fn find_references(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindReferencesQuery>,
) -> Result<Json<Vec<SymbolReference>>, AppError> {
    let limit = query.limit.unwrap_or(20);

    let ref_nodes = state
        .orchestrator
        .neo4j()
        .find_symbol_references(&query.symbol, limit)
        .await?;

    let references: Vec<SymbolReference> = ref_nodes
        .into_iter()
        .map(|r| SymbolReference {
            file_path: r.file_path,
            line: r.line,
            context: r.context,
            reference_type: r.reference_type,
        })
        .collect();

    Ok(Json(references))
}

// ============================================================================
// Dependency Analysis
// ============================================================================

#[derive(Serialize)]
pub struct FileDependencies {
    /// Files this file imports/depends on
    pub imports: Vec<DependencyInfo>,
    /// Files that import/depend on this file
    pub imported_by: Vec<DependencyInfo>,
    /// Transitive dependencies (files affected if this changes)
    pub impact_radius: Vec<String>,
}

#[derive(Serialize)]
pub struct DependencyInfo {
    pub path: String,
    pub language: String,
    pub symbols_used: Vec<String>,
}

/// Get dependency graph for a file
pub async fn get_file_dependencies(
    State(state): State<OrchestratorState>,
    Path(file_path): Path<String>,
) -> Result<Json<FileDependencies>, AppError> {
    let file_path = urlencoding::decode(&file_path)
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .to_string();

    let dependents = state
        .orchestrator
        .neo4j()
        .find_dependent_files(&file_path, 3)
        .await?;

    let import_nodes = state
        .orchestrator
        .neo4j()
        .get_file_direct_imports(&file_path)
        .await?;

    let imports: Vec<DependencyInfo> = import_nodes
        .into_iter()
        .map(|i| DependencyInfo {
            path: i.path,
            language: i.language,
            symbols_used: vec![],
        })
        .collect();

    Ok(Json(FileDependencies {
        imports,
        imported_by: dependents
            .iter()
            .map(|p| DependencyInfo {
                path: p.clone(),
                language: String::new(),
                symbols_used: vec![],
            })
            .collect(),
        impact_radius: dependents,
    }))
}

// ============================================================================
// Call Graph
// ============================================================================

#[derive(Deserialize)]
pub struct CallGraphQuery {
    /// Function name
    pub function: String,
    /// Depth of call graph (default 2)
    pub depth: Option<u32>,
    /// Direction: "callers" (who calls this), "callees" (what this calls), "both"
    pub direction: Option<String>,
}

#[derive(Serialize)]
pub struct CallGraphNode {
    pub name: String,
    pub file_path: String,
    pub line: u32,
    pub callers: Vec<String>,
    pub callees: Vec<String>,
}

/// Get call graph for a function
pub async fn get_call_graph(
    State(state): State<OrchestratorState>,
    Query(query): Query<CallGraphQuery>,
) -> Result<Json<CallGraphNode>, AppError> {
    let depth = query.depth.unwrap_or(2);
    let direction = query.direction.as_deref().unwrap_or("both");

    let mut callers = vec![];
    let mut callees = vec![];

    if direction == "callers" || direction == "both" {
        callers = state
            .orchestrator
            .neo4j()
            .get_function_callers_by_name(&query.function, depth)
            .await?;
    }

    if direction == "callees" || direction == "both" {
        callees = state
            .orchestrator
            .neo4j()
            .get_function_callees_by_name(&query.function, depth)
            .await?;
    }

    Ok(Json(CallGraphNode {
        name: query.function,
        file_path: String::new(),
        line: 0,
        callers,
        callees,
    }))
}

// ============================================================================
// Impact Analysis
// ============================================================================

#[derive(Deserialize)]
pub struct ImpactQuery {
    /// File or function to analyze
    pub target: String,
    /// "file" or "function"
    pub target_type: Option<String>,
}

#[derive(Serialize)]
pub struct ImpactAnalysis {
    pub target: String,
    pub directly_affected: Vec<String>,
    pub transitively_affected: Vec<String>,
    pub test_files_affected: Vec<String>,
    pub risk_level: String, // "low", "medium", "high"
    pub suggestion: String,
}

/// Analyze impact of changing a file or function
pub async fn analyze_impact(
    State(state): State<OrchestratorState>,
    Query(query): Query<ImpactQuery>,
) -> Result<Json<ImpactAnalysis>, AppError> {
    let target_type = query.target_type.as_deref().unwrap_or("file");

    let (directly_affected, transitively_affected) = if target_type == "file" {
        let direct = state
            .orchestrator
            .neo4j()
            .find_dependent_files(&query.target, 1)
            .await?;
        let transitive = state
            .orchestrator
            .neo4j()
            .find_dependent_files(&query.target, 3)
            .await?;
        (direct, transitive)
    } else {
        let callers = state
            .orchestrator
            .neo4j()
            .find_callers(&query.target)
            .await?;
        let direct: Vec<String> = callers.iter().map(|f| f.file_path.clone()).collect();
        (direct.clone(), direct)
    };

    let test_files: Vec<String> = transitively_affected
        .iter()
        .filter(|p| p.contains("test") || p.contains("_test") || p.ends_with("_tests.rs"))
        .cloned()
        .collect();

    let risk_level = if transitively_affected.len() > 10 {
        "high"
    } else if transitively_affected.len() > 3 {
        "medium"
    } else {
        "low"
    };

    let suggestion = format!(
        "Run tests in: {}. Consider reviewing: {}",
        test_files.join(", "),
        directly_affected
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(Json(ImpactAnalysis {
        target: query.target,
        directly_affected,
        transitively_affected,
        test_files_affected: test_files,
        risk_level: risk_level.to_string(),
        suggestion,
    }))
}

// ============================================================================
// Architecture Overview
// ============================================================================

#[derive(Serialize)]
pub struct ArchitectureOverview {
    pub total_files: usize,
    pub languages: Vec<LanguageStats>,
    pub modules: Vec<ModuleInfo>,
    pub key_files: Vec<ConnectedFileNode>,
    pub orphan_files: Vec<String>,
}

#[derive(Serialize)]
pub struct LanguageStats {
    pub language: String,
    pub file_count: usize,
    pub function_count: usize,
    pub struct_count: usize,
}

#[derive(Serialize)]
pub struct ModuleInfo {
    pub path: String,
    pub files: usize,
    pub public_api: Vec<String>,
}

/// Get high-level architecture overview
pub async fn get_architecture(
    State(state): State<OrchestratorState>,
) -> Result<Json<ArchitectureOverview>, AppError> {
    let lang_stats = state.orchestrator.neo4j().get_language_stats().await?;

    let languages: Vec<LanguageStats> = lang_stats
        .into_iter()
        .map(|s| LanguageStats {
            language: s.language,
            file_count: s.file_count,
            function_count: 0,
            struct_count: 0,
        })
        .collect();

    let key_files = state
        .orchestrator
        .neo4j()
        .get_most_connected_files_detailed(10)
        .await?;

    let total_files: usize = languages.iter().map(|l| l.file_count).sum();

    Ok(Json(ArchitectureOverview {
        total_files,
        languages,
        modules: vec![],
        key_files,
        orphan_files: vec![],
    }))
}

// ============================================================================
// Similar Code
// ============================================================================

#[derive(Deserialize)]
pub struct SimilarCodeQuery {
    /// Code snippet to find similar code for
    pub snippet: String,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct SimilarCode {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub snippet: String,
    pub similarity: f64,
}

/// Find code similar to a given snippet
pub async fn find_similar_code(
    State(state): State<OrchestratorState>,
    Json(query): Json<SimilarCodeQuery>,
) -> Result<Json<Vec<SimilarCode>>, AppError> {
    let hits = state
        .orchestrator
        .meili()
        .search_code_with_scores(&query.snippet, query.limit.unwrap_or(5), None, None)
        .await?;

    let similar: Vec<SimilarCode> = hits
        .into_iter()
        .map(|hit| SimilarCode {
            path: hit.document.path,
            line_start: 0,
            line_end: 0,
            snippet: hit.document.docstrings.chars().take(300).collect(),
            similarity: hit.score,
        })
        .collect();

    Ok(Json(similar))
}

// ============================================================================
// Trait Implementations
// ============================================================================

#[derive(Deserialize)]
pub struct TraitImplQuery {
    /// Trait name to find implementations for
    pub trait_name: String,
}

#[derive(Serialize)]
pub struct TraitImplementors {
    pub trait_name: String,
    pub is_external: bool,
    pub source: Option<String>,
    pub implementors: Vec<TypeImplementation>,
}

#[derive(Serialize)]
pub struct TypeImplementation {
    pub type_name: String,
    pub file_path: String,
    pub line: u32,
}

/// Find all types that implement a specific trait
pub async fn find_trait_implementations(
    State(state): State<OrchestratorState>,
    Query(query): Query<TraitImplQuery>,
) -> Result<Json<TraitImplementors>, AppError> {
    let trait_info = state
        .orchestrator
        .neo4j()
        .get_trait_info(&query.trait_name)
        .await?;

    let (is_external, source) = trait_info
        .map(|t| (t.is_external, t.source))
        .unwrap_or((false, None));

    let impl_nodes = state
        .orchestrator
        .neo4j()
        .get_trait_implementors_detailed(&query.trait_name)
        .await?;

    let implementors: Vec<TypeImplementation> = impl_nodes
        .into_iter()
        .map(|i| TypeImplementation {
            type_name: i.type_name,
            file_path: i.file_path,
            line: i.line,
        })
        .collect();

    Ok(Json(TraitImplementors {
        trait_name: query.trait_name,
        is_external,
        source,
        implementors,
    }))
}

#[derive(Deserialize)]
pub struct TypeTraitsQuery {
    /// Type name to find implemented traits for
    pub type_name: String,
}

#[derive(Serialize)]
pub struct TypeTraits {
    pub type_name: String,
    pub traits: Vec<TraitInfo>,
}

#[derive(Serialize)]
pub struct TraitInfo {
    pub name: String,
    pub full_path: Option<String>,
    pub file_path: String,
    pub is_external: bool,
    pub source: Option<String>,
}

/// Find all traits implemented by a specific type
pub async fn find_type_traits(
    State(state): State<OrchestratorState>,
    Query(query): Query<TypeTraitsQuery>,
) -> Result<Json<TypeTraits>, AppError> {
    let trait_nodes = state
        .orchestrator
        .neo4j()
        .get_type_trait_implementations(&query.type_name)
        .await?;

    let traits: Vec<TraitInfo> = trait_nodes
        .into_iter()
        .map(|t| TraitInfo {
            name: t.name,
            full_path: t.full_path,
            file_path: t.file_path,
            is_external: t.is_external,
            source: t.source,
        })
        .collect();

    Ok(Json(TypeTraits {
        type_name: query.type_name,
        traits,
    }))
}

// ============================================================================
// Impl Blocks
// ============================================================================

#[derive(Deserialize)]
pub struct ImplBlocksQuery {
    /// Type name to find impl blocks for
    pub type_name: String,
}

#[derive(Serialize)]
pub struct TypeImplBlocks {
    pub type_name: String,
    pub impl_blocks: Vec<ImplBlockInfo>,
}

#[derive(Serialize)]
pub struct ImplBlockInfo {
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub trait_name: Option<String>,
    pub methods: Vec<String>,
}

/// Get all impl blocks for a type
pub async fn get_impl_blocks(
    State(state): State<OrchestratorState>,
    Query(query): Query<ImplBlocksQuery>,
) -> Result<Json<TypeImplBlocks>, AppError> {
    let block_nodes = state
        .orchestrator
        .neo4j()
        .get_type_impl_blocks_detailed(&query.type_name)
        .await?;

    let impl_blocks: Vec<ImplBlockInfo> = block_nodes
        .into_iter()
        .map(|b| ImplBlockInfo {
            file_path: b.file_path,
            line_start: b.line_start,
            line_end: b.line_end,
            trait_name: b.trait_name,
            methods: b.methods,
        })
        .collect();

    Ok(Json(TypeImplBlocks {
        type_name: query.type_name,
        impl_blocks,
    }))
}
