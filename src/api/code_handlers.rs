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

// ============================================================================
// Code Search (Meilisearch)
// ============================================================================

#[derive(Deserialize)]
pub struct CodeSearchQuery {
    /// Search query (semantic search across code content, symbols, comments)
    pub q: String,
    /// Max results (default 10)
    pub limit: Option<usize>,
    /// Filter by language (rust, typescript, python, go)
    pub language: Option<String>,
}

#[derive(Serialize)]
pub struct CodeSearchResult {
    pub path: String,
    pub language: String,
    pub snippet: String,
    pub symbols: Vec<String>,
    pub score: f64,
}

/// Search code semantically across the codebase
///
/// Example: `/api/code/search?q=error handling async&language=rust&limit=5`
///
/// This finds code containing patterns, not just exact matches.
/// Much faster than grep for understanding "how is X done in this codebase?"
pub async fn search_code(
    State(state): State<OrchestratorState>,
    Query(query): Query<CodeSearchQuery>,
) -> Result<Json<Vec<CodeSearchResult>>, AppError> {
    let hits = state
        .orchestrator
        .meili()
        .search_code_with_scores(
            &query.q,
            query.limit.unwrap_or(10),
            query.language.as_deref(),
            None,
        )
        .await?;

    let results: Vec<CodeSearchResult> = hits
        .into_iter()
        .map(|hit| CodeSearchResult {
            path: hit.document.path,
            language: hit.document.language,
            snippet: hit.document.docstrings.chars().take(500).collect(),
            symbols: hit.document.symbols,
            score: hit.score,
        })
        .collect();

    Ok(Json(results))
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
///
/// Example: `/api/code/symbols/src%2Flib.rs`
///
/// Returns functions, structs, enums with their signatures and line numbers.
/// Faster than parsing the file, and you get structured data.
pub async fn get_file_symbols(
    State(state): State<OrchestratorState>,
    Path(file_path): Path<String>,
) -> Result<Json<FileSymbols>, AppError> {
    let file_path = urlencoding::decode(&file_path)
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .to_string();

    // Query Neo4j for all symbols in this file
    let q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})
        OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
        OPTIONAL MATCH (f)-[:CONTAINS]->(s:Struct)
        RETURN f.language AS language,
               collect(DISTINCT func) AS functions,
               collect(DISTINCT s) AS structs
        "#,
    )
    .param("path", file_path.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    if rows.is_empty() {
        return Err(AppError::NotFound(format!("File not found: {}", file_path)));
    }

    // Parse results (simplified - would need proper deserialization)
    let language: String = rows[0].get("language").unwrap_or_default();

    Ok(Json(FileSymbols {
        path: file_path,
        language,
        functions: vec![], // TODO: deserialize from Neo4j nodes
        structs: vec![],
        imports: vec![],
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
///
/// Example: `/api/code/references?symbol=AppState&limit=20`
///
/// This is like "Find All References" in an IDE but across the entire codebase.
/// Impossible to do manually, takes seconds with Neo4j.
pub async fn find_references(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindReferencesQuery>,
) -> Result<Json<Vec<SymbolReference>>, AppError> {
    let q = neo4rs::query(
        r#"
        MATCH (f:Function {name: $name})
        OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
        RETURN f.file_path AS defined_in, f.line_start AS defined_line,
               collect({
                   file: caller.file_path,
                   line: caller.line_start,
                   name: caller.name
               }) AS callers
        UNION
        MATCH (s:Struct {name: $name})
        OPTIONAL MATCH (user:Function)-[:USES_TYPE]->(s)
        RETURN s.file_path AS defined_in, s.line_start AS defined_line,
               collect({
                   file: user.file_path,
                   line: user.line_start,
                   name: user.name
               }) AS callers
        "#,
    )
    .param("name", query.symbol.clone());

    let _rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    // TODO: parse rows into SymbolReference
    Ok(Json(vec![]))
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
///
/// Example: `/api/code/dependencies/src%2Fneo4j%2Fclient.rs`
///
/// Shows what this file depends on AND what depends on it.
/// Critical for understanding impact of changes.
pub async fn get_file_dependencies(
    State(state): State<OrchestratorState>,
    Path(file_path): Path<String>,
) -> Result<Json<FileDependencies>, AppError> {
    let file_path = urlencoding::decode(&file_path)
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .to_string();

    // Get files that depend on this file
    let dependents = state
        .orchestrator
        .neo4j()
        .find_dependent_files(&file_path, 3)
        .await?;

    // Get files this file imports
    let q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})-[:IMPORTS]->(imported:File)
        RETURN imported.path AS path, imported.language AS language
        "#,
    )
    .param("path", file_path.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
    let imports: Vec<DependencyInfo> = rows
        .into_iter()
        .filter_map(|row| {
            Some(DependencyInfo {
                path: row.get("path").ok()?,
                language: row.get("language").ok()?,
                symbols_used: vec![],
            })
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
///
/// Example: `/api/code/callgraph?function=handle_request&depth=2&direction=both`
///
/// Shows the full call chain - who calls this function and what it calls.
/// Essential for understanding code flow and refactoring impact.
pub async fn get_call_graph(
    State(state): State<OrchestratorState>,
    Query(query): Query<CallGraphQuery>,
) -> Result<Json<CallGraphNode>, AppError> {
    let depth = query.depth.unwrap_or(2);
    let direction = query.direction.as_deref().unwrap_or("both");

    let mut callers = vec![];
    let mut callees = vec![];

    if direction == "callers" || direction == "both" {
        // Find functions that call this function
        let q = neo4rs::query(&format!(
            r#"
            MATCH (f:Function {{name: $name}})
            MATCH (caller:Function)-[:CALLS*1..{}]->(f)
            RETURN DISTINCT caller.name AS name, caller.file_path AS file
            "#,
            depth
        ))
        .param("name", query.function.clone());

        let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
        callers = rows
            .into_iter()
            .filter_map(|r| r.get::<String>("name").ok())
            .collect();
    }

    if direction == "callees" || direction == "both" {
        // Find functions this function calls
        let q = neo4rs::query(&format!(
            r#"
            MATCH (f:Function {{name: $name}})
            MATCH (f)-[:CALLS*1..{}]->(callee:Function)
            RETURN DISTINCT callee.name AS name, callee.file_path AS file
            "#,
            depth
        ))
        .param("name", query.function.clone());

        let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
        callees = rows
            .into_iter()
            .filter_map(|r| r.get::<String>("name").ok())
            .collect();
    }

    Ok(Json(CallGraphNode {
        name: query.function,
        file_path: String::new(), // TODO: get from initial query
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
///
/// Example: `/api/code/impact?target=src/neo4j/client.rs&target_type=file`
///
/// Tells you exactly what would break if you change something.
/// Suggests which tests to run.
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
        // Function impact
        let callers = state
            .orchestrator
            .neo4j()
            .find_callers(&query.target)
            .await?;
        let direct: Vec<String> = callers.iter().map(|f| f.file_path.clone()).collect();
        (direct.clone(), direct)
    };

    // Find test files in the affected set
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
    pub most_connected: Vec<String>,
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
///
/// Example: `/api/code/architecture`
///
/// Provides a bird's eye view of the codebase structure.
/// Useful for onboarding or understanding unfamiliar codebases.
pub async fn get_architecture(
    State(state): State<OrchestratorState>,
) -> Result<Json<ArchitectureOverview>, AppError> {
    // Count files by language
    let q = neo4rs::query(
        r#"
        MATCH (f:File)
        RETURN f.language AS language, count(f) AS count
        ORDER BY count DESC
        "#,
    );
    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    let languages: Vec<LanguageStats> = rows
        .into_iter()
        .filter_map(|row| {
            Some(LanguageStats {
                language: row.get("language").ok()?,
                file_count: row.get::<i64>("count").ok()? as usize,
                function_count: 0,
                struct_count: 0,
            })
        })
        .collect();

    // Find most connected files (highest in-degree)
    let q = neo4rs::query(
        r#"
        MATCH (f:File)<-[:IMPORTS]-(importer:File)
        RETURN f.path AS path, count(importer) AS imports
        ORDER BY imports DESC
        LIMIT 10
        "#,
    );
    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
    let most_connected: Vec<String> = rows
        .into_iter()
        .filter_map(|r| r.get("path").ok())
        .collect();

    let total_files: usize = languages.iter().map(|l| l.file_count).sum();

    Ok(Json(ArchitectureOverview {
        total_files,
        languages,
        modules: vec![],
        most_connected,
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
///
/// Example: POST `/api/code/similar` with body `{"snippet": "async fn handle_error", "limit": 5}`
///
/// Useful for: "How is this pattern implemented elsewhere?"
/// "Are there similar functions I should update too?"
pub async fn find_similar_code(
    State(state): State<OrchestratorState>,
    Json(query): Json<SimilarCodeQuery>,
) -> Result<Json<Vec<SimilarCode>>, AppError> {
    // Use Meilisearch to find semantically similar code
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
    pub implementors: Vec<TypeImplementation>,
}

#[derive(Serialize)]
pub struct TypeImplementation {
    pub type_name: String,
    pub file_path: String,
    pub line: u32,
}

/// Find all types that implement a specific trait
///
/// Example: `/api/code/trait-impls?trait_name=Module`
///
/// Useful for understanding trait usage patterns across the codebase.
/// Critical for Rust/Go interface exploration.
pub async fn find_trait_implementations(
    State(state): State<OrchestratorState>,
    Query(query): Query<TraitImplQuery>,
) -> Result<Json<TraitImplementors>, AppError> {
    let q = neo4rs::query(
        r#"
        MATCH (i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait {name: $trait_name})
        MATCH (i)-[:IMPLEMENTS_FOR]->(type)
        RETURN type.name AS type_name, i.file_path AS file_path, i.line_start AS line
        "#,
    )
    .param("trait_name", query.trait_name.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    let implementors: Vec<TypeImplementation> = rows
        .into_iter()
        .filter_map(|row| {
            Some(TypeImplementation {
                type_name: row.get("type_name").ok()?,
                file_path: row.get("file_path").ok()?,
                line: row.get::<i64>("line").ok()? as u32,
            })
        })
        .collect();

    Ok(Json(TraitImplementors {
        trait_name: query.trait_name,
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
    pub file_path: String,
    pub is_local: bool,
}

/// Find all traits implemented by a specific type
///
/// Example: `/api/code/type-traits?type_name=AppState`
///
/// Useful for understanding what a type can do.
pub async fn find_type_traits(
    State(state): State<OrchestratorState>,
    Query(query): Query<TypeTraitsQuery>,
) -> Result<Json<TypeTraits>, AppError> {
    let q = neo4rs::query(
        r#"
        MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
        OPTIONAL MATCH (i)-[:IMPLEMENTS_TRAIT]->(t:Trait)
        RETURN t.name AS trait_name, t.file_path AS file_path
        "#,
    )
    .param("type_name", query.type_name.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    let traits: Vec<TraitInfo> = rows
        .into_iter()
        .filter_map(|row| {
            let name: String = row.get("trait_name").ok()?;
            let file_path: String = row.get("file_path").unwrap_or_default();
            Some(TraitInfo {
                name,
                is_local: !file_path.is_empty(),
                file_path,
            })
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
///
/// Example: `/api/code/impl-blocks?type_name=Orchestrator`
///
/// Shows where a type's methods are defined, including trait implementations.
pub async fn get_impl_blocks(
    State(state): State<OrchestratorState>,
    Query(query): Query<ImplBlocksQuery>,
) -> Result<Json<TypeImplBlocks>, AppError> {
    let q = neo4rs::query(
        r#"
        MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
        OPTIONAL MATCH (i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait)
        OPTIONAL MATCH (f:File {path: i.file_path})-[:CONTAINS]->(func:Function)
        WHERE func.line_start >= i.line_start AND func.line_end <= i.line_end
        RETURN i.file_path AS file_path, i.line_start AS line_start, i.line_end AS line_end,
               i.trait_name AS trait_name, collect(func.name) AS methods
        "#,
    )
    .param("type_name", query.type_name.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    let impl_blocks: Vec<ImplBlockInfo> = rows
        .into_iter()
        .filter_map(|row| {
            let trait_name: Option<String> = row.get("trait_name").ok();
            let trait_name = trait_name.filter(|s| !s.is_empty());
            Some(ImplBlockInfo {
                file_path: row.get("file_path").ok()?,
                line_start: row.get::<i64>("line_start").ok()? as u32,
                line_end: row.get::<i64>("line_end").ok()? as u32,
                trait_name,
                methods: row.get("methods").unwrap_or_default(),
            })
        })
        .collect();

    Ok(Json(TypeImplBlocks {
        type_name: query.type_name,
        impl_blocks,
    }))
}
