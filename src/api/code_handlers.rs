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

    // First, get the file language
    let file_q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})
        RETURN f.language AS language
        "#,
    )
    .param("path", file_path.clone());

    let file_rows = state.orchestrator.neo4j().execute_with_params(file_q).await?;

    if file_rows.is_empty() {
        return Err(AppError::NotFound(format!("File not found: {}", file_path)));
    }

    let language: String = file_rows[0].get("language").unwrap_or_default();

    // Query functions separately to avoid cartesian products
    let func_q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})-[:CONTAINS]->(func:Function)
        RETURN func
        ORDER BY func.line_start
        "#,
    )
    .param("path", file_path.clone());

    let func_rows = state.orchestrator.neo4j().execute_with_params(func_q).await?;
    let functions: Vec<FunctionSummary> = func_rows
        .iter()
        .filter_map(|row| {
            let node: neo4rs::Node = row.get("func").ok()?;
            let name: String = node.get("name").ok()?;
            let is_async: bool = node.get("is_async").unwrap_or(false);
            let visibility: String = node.get("visibility").unwrap_or_default();
            let is_public = visibility == "public";
            let line: i64 = node.get("line_start").unwrap_or(0);
            let complexity: i64 = node.get("complexity").unwrap_or(1);
            let docstring: Option<String> = node.get("docstring").ok();

            // Build signature from params and return type
            let params: Vec<String> = node.get("params").unwrap_or_default();
            let return_type: String = node.get("return_type").unwrap_or_default();
            let async_prefix = if is_async { "async " } else { "" };
            let signature = format!(
                "{}fn {}({}){}",
                async_prefix,
                name,
                params.join(", "),
                if return_type.is_empty() {
                    String::new()
                } else {
                    format!(" -> {}", return_type)
                }
            );

            Some(FunctionSummary {
                name,
                signature,
                line: line as u32,
                is_async,
                is_public,
                complexity: complexity as u32,
                docstring,
            })
        })
        .collect();

    // Query structs separately
    let struct_q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})-[:CONTAINS]->(s:Struct)
        RETURN s
        ORDER BY s.line_start
        "#,
    )
    .param("path", file_path.clone());

    let struct_rows = state.orchestrator.neo4j().execute_with_params(struct_q).await?;
    let structs: Vec<StructSummary> = struct_rows
        .iter()
        .filter_map(|row| {
            let node: neo4rs::Node = row.get("s").ok()?;
            let name: String = node.get("name").ok()?;
            let visibility: String = node.get("visibility").unwrap_or_default();
            let is_public = visibility == "public";
            let line: i64 = node.get("line_start").unwrap_or(0);
            let docstring: Option<String> = node.get("docstring").ok();

            Some(StructSummary {
                name,
                line: line as u32,
                is_public,
                docstring,
            })
        })
        .collect();

    // Query imports separately
    let import_q = neo4rs::query(
        r#"
        MATCH (f:File {path: $path})-[:CONTAINS]->(i:Import)
        RETURN i.path AS path
        ORDER BY i.line
        "#,
    )
    .param("path", file_path.clone());

    let import_rows = state.orchestrator.neo4j().execute_with_params(import_q).await?;
    let imports: Vec<String> = import_rows
        .iter()
        .filter_map(|row| row.get("path").ok())
        .collect();

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
///
/// Example: `/api/code/references?symbol=AppState&limit=20`
///
/// This is like "Find All References" in an IDE but across the entire codebase.
/// Impossible to do manually, takes seconds with Neo4j.
pub async fn find_references(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindReferencesQuery>,
) -> Result<Json<Vec<SymbolReference>>, AppError> {
    let limit = query.limit.unwrap_or(20);
    let mut references = Vec::new();

    // Find function callers
    let q = neo4rs::query(
        r#"
        MATCH (f:Function {name: $name})
        OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
        WHERE caller IS NOT NULL
        RETURN 'call' AS ref_type,
               caller.file_path AS file_path,
               caller.line_start AS line,
               caller.name AS context
        LIMIT $limit
        "#,
    )
    .param("name", query.symbol.clone())
    .param("limit", limit as i64);

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
    for row in rows {
        if let (Ok(file_path), Ok(line), Ok(context)) = (
            row.get::<String>("file_path"),
            row.get::<i64>("line"),
            row.get::<String>("context"),
        ) {
            references.push(SymbolReference {
                file_path,
                line: line as u32,
                context: format!("called from {}", context),
                reference_type: "call".to_string(),
            });
        }
    }

    // Find struct usages (via imports or type references)
    let q = neo4rs::query(
        r#"
        MATCH (s:Struct {name: $name})
        OPTIONAL MATCH (i:Import)-[:IMPORTS_SYMBOL]->(s)
        WHERE i IS NOT NULL
        RETURN 'import' AS ref_type,
               i.file_path AS file_path,
               i.line AS line,
               i.path AS context
        LIMIT $limit
        "#,
    )
    .param("name", query.symbol.clone())
    .param("limit", limit as i64);

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
    for row in rows {
        if let (Ok(file_path), Ok(line), Ok(context)) = (
            row.get::<String>("file_path"),
            row.get::<i64>("line"),
            row.get::<String>("context"),
        ) {
            references.push(SymbolReference {
                file_path,
                line: line as u32,
                context: format!("imported via {}", context),
                reference_type: "import".to_string(),
            });
        }
    }

    // Find files that import the symbol's module
    let q = neo4rs::query(
        r#"
        MATCH (s {name: $name})
        WHERE s:Function OR s:Struct OR s:Trait OR s:Enum
        MATCH (f:File {path: s.file_path})
        OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
        WHERE importer IS NOT NULL
        RETURN 'file_import' AS ref_type,
               importer.path AS file_path,
               0 AS line,
               f.path AS context
        LIMIT $limit
        "#,
    )
    .param("name", query.symbol.clone())
    .param("limit", limit as i64);

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;
    for row in rows {
        if let (Ok(file_path), Ok(context)) = (
            row.get::<String>("file_path"),
            row.get::<String>("context"),
        ) {
            references.push(SymbolReference {
                file_path,
                line: 0,
                context: format!("imports module {}", context),
                reference_type: "file_import".to_string(),
            });
        }
    }

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
///
/// Example: `/api/code/trait-impls?trait_name=Debug`
///
/// Useful for understanding trait usage patterns across the codebase.
/// Works for both local traits and external traits (std, serde, etc.)
pub async fn find_trait_implementations(
    State(state): State<OrchestratorState>,
    Query(query): Query<TraitImplQuery>,
) -> Result<Json<TraitImplementors>, AppError> {
    // First get trait info
    let trait_q = neo4rs::query(
        r#"
        MATCH (t:Trait {name: $trait_name})
        RETURN t.is_external AS is_external, t.source AS source
        LIMIT 1
        "#,
    )
    .param("trait_name", query.trait_name.clone());

    let trait_rows = state.orchestrator.neo4j().execute_with_params(trait_q).await?;
    let (is_external, source) = trait_rows
        .first()
        .map(|r| {
            let ext: bool = r.get("is_external").unwrap_or(false);
            let src: Option<String> = r.get("source").ok();
            (ext, src)
        })
        .unwrap_or((false, None));

    // Get implementors
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
///
/// Example: `/api/code/type-traits?type_name=AppState`
///
/// Useful for understanding what a type can do.
/// Returns both local traits (defined in codebase) and external traits (std, serde, etc.)
pub async fn find_type_traits(
    State(state): State<OrchestratorState>,
    Query(query): Query<TypeTraitsQuery>,
) -> Result<Json<TypeTraits>, AppError> {
    let q = neo4rs::query(
        r#"
        MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
        OPTIONAL MATCH (i)-[:IMPLEMENTS_TRAIT]->(t:Trait)
        RETURN t.name AS trait_name,
               t.full_path AS full_path,
               t.file_path AS file_path,
               t.is_external AS is_external,
               t.source AS source
        "#,
    )
    .param("type_name", query.type_name.clone());

    let rows = state.orchestrator.neo4j().execute_with_params(q).await?;

    let traits: Vec<TraitInfo> = rows
        .into_iter()
        .filter_map(|row| {
            let name: String = row.get("trait_name").ok()?;
            let full_path: Option<String> = row.get("full_path").ok();
            let file_path: String = row.get("file_path").unwrap_or_default();
            let is_external: bool = row.get("is_external").unwrap_or(false);
            let source: Option<String> = row.get("source").ok();
            Some(TraitInfo {
                name,
                full_path,
                file_path,
                is_external,
                source,
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
