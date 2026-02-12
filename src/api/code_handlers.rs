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
    /// Filter by project slug
    pub project_slug: Option<String>,
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

    let project_id = if let Some(ref slug) = query.project_slug {
        Some(
            state
                .orchestrator
                .neo4j()
                .get_project_by_slug(slug)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?
                .id,
        )
    } else {
        None
    };

    let ref_nodes = state
        .orchestrator
        .neo4j()
        .find_symbol_references(&query.symbol, limit, project_id)
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
        .find_dependent_files(&file_path, 3, None)
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
    /// Filter by project slug
    pub project_slug: Option<String>,
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

    let project_id = if let Some(ref slug) = query.project_slug {
        Some(
            state
                .orchestrator
                .neo4j()
                .get_project_by_slug(slug)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?
                .id,
        )
    } else {
        None
    };

    let mut callers = vec![];
    let mut callees = vec![];

    if direction == "callers" || direction == "both" {
        callers = state
            .orchestrator
            .neo4j()
            .get_function_callers_by_name(&query.function, depth, project_id)
            .await?;
    }

    if direction == "callees" || direction == "both" {
        callees = state
            .orchestrator
            .neo4j()
            .get_function_callees_by_name(&query.function, depth, project_id)
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
    /// Filter by project slug
    pub project_slug: Option<String>,
}

#[derive(Serialize)]
pub struct ImpactAnalysis {
    pub target: String,
    pub directly_affected: Vec<String>,
    pub transitively_affected: Vec<String>,
    pub test_files_affected: Vec<String>,
    pub caller_count: i64,
    pub risk_level: String, // "low", "medium", "high"
    pub suggestion: String,
}

/// Analyze impact of changing a file or function
pub async fn analyze_impact(
    State(state): State<OrchestratorState>,
    Query(query): Query<ImpactQuery>,
) -> Result<Json<ImpactAnalysis>, AppError> {
    let target_type = query.target_type.as_deref().unwrap_or("file");

    let project_id = if let Some(ref slug) = query.project_slug {
        Some(
            state
                .orchestrator
                .neo4j()
                .get_project_by_slug(slug)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?
                .id,
        )
    } else {
        None
    };

    let (directly_affected, transitively_affected, caller_count) = if target_type == "file" {
        let direct = state
            .orchestrator
            .neo4j()
            .find_dependent_files(&query.target, 1, project_id)
            .await?;
        let transitive = state
            .orchestrator
            .neo4j()
            .find_dependent_files(&query.target, 3, project_id)
            .await?;
        let count = direct.len() as i64;
        (direct, transitive, count)
    } else {
        let callers = state
            .orchestrator
            .neo4j()
            .find_callers(&query.target, project_id)
            .await?;
        let direct: Vec<String> = callers.iter().map(|f| f.file_path.clone()).collect();
        let count = state
            .orchestrator
            .neo4j()
            .get_function_caller_count(&query.target, project_id)
            .await?;
        (direct.clone(), direct, count)
    };

    let test_files: Vec<String> = transitively_affected
        .iter()
        .filter(|p| p.contains("test") || p.contains("_test") || p.ends_with("_tests.rs"))
        .cloned()
        .collect();

    let risk_level = if transitively_affected.len() > 10 || caller_count > 10 {
        "high"
    } else if transitively_affected.len() > 3 || caller_count > 3 {
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
        caller_count,
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
        .search_code_with_scores(&query.snippet, query.limit.unwrap_or(5), None, None, None)
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

// ============================================================================
// Feature Graphs
// ============================================================================

#[derive(Deserialize)]
pub struct CreateFeatureGraphBody {
    pub name: String,
    pub description: Option<String>,
    pub project_id: uuid::Uuid,
}

#[derive(Deserialize)]
pub struct ListFeatureGraphsQuery {
    pub project_id: Option<uuid::Uuid>,
}

#[derive(Deserialize)]
pub struct AddEntityBody {
    pub entity_type: String,
    pub entity_id: String,
    pub role: Option<String>,
}

#[derive(Deserialize)]
pub struct AutoBuildBody {
    pub name: String,
    pub description: Option<String>,
    pub project_id: uuid::Uuid,
    pub entry_function: String,
    pub depth: Option<u32>,
    pub include_relations: Option<Vec<String>>,
}

/// POST /api/feature-graphs
pub async fn create_feature_graph(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateFeatureGraphBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let fg = crate::neo4j::models::FeatureGraphNode {
        id: uuid::Uuid::new_v4(),
        name: body.name,
        description: body.description,
        project_id: body.project_id,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    state.orchestrator.neo4j().create_feature_graph(&fg).await?;
    Ok(Json(serde_json::json!({
        "id": fg.id.to_string(),
        "name": fg.name,
        "project_id": fg.project_id.to_string(),
        "created_at": fg.created_at.to_rfc3339(),
    })))
}

/// GET /api/feature-graphs
pub async fn list_feature_graphs(
    State(state): State<OrchestratorState>,
    Query(query): Query<ListFeatureGraphsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphs = state
        .orchestrator
        .neo4j()
        .list_feature_graphs(query.project_id)
        .await?;
    let items: Vec<serde_json::Value> = graphs
        .iter()
        .map(|fg| {
            serde_json::json!({
                "id": fg.id.to_string(),
                "name": fg.name,
                "description": fg.description,
                "project_id": fg.project_id.to_string(),
                "created_at": fg.created_at.to_rfc3339(),
            })
        })
        .collect();
    Ok(Json(
        serde_json::json!({ "feature_graphs": items, "count": items.len() }),
    ))
}

/// GET /api/feature-graphs/:id
pub async fn get_feature_graph(
    State(state): State<OrchestratorState>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let detail = state
        .orchestrator
        .neo4j()
        .get_feature_graph_detail(id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Feature graph not found: {}", id)))?;
    let entities: Vec<serde_json::Value> = detail
        .entities
        .iter()
        .map(|e| {
            serde_json::json!({
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "name": e.name,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "id": detail.graph.id.to_string(),
        "name": detail.graph.name,
        "description": detail.graph.description,
        "project_id": detail.graph.project_id.to_string(),
        "entities": entities,
        "entity_count": entities.len(),
    })))
}

/// DELETE /api/feature-graphs/:id
pub async fn delete_feature_graph(
    State(state): State<OrchestratorState>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.orchestrator.neo4j().delete_feature_graph(id).await?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

/// POST /api/feature-graphs/:id/entities
pub async fn add_entity_to_feature_graph(
    State(state): State<OrchestratorState>,
    Path(id): Path<uuid::Uuid>,
    Json(body): Json<AddEntityBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .add_entity_to_feature_graph(id, &body.entity_type, &body.entity_id, body.role.as_deref())
        .await?;
    Ok(Json(serde_json::json!({
        "added": true,
        "feature_graph_id": id.to_string(),
        "entity_type": body.entity_type,
        "entity_id": body.entity_id,
        "role": body.role,
    })))
}

/// POST /api/feature-graphs/auto-build
pub async fn auto_build_feature_graph(
    State(state): State<OrchestratorState>,
    Json(body): Json<AutoBuildBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let depth = body.depth.unwrap_or(2);
    let detail = state
        .orchestrator
        .neo4j()
        .auto_build_feature_graph(
            &body.name,
            body.description.as_deref(),
            body.project_id,
            &body.entry_function,
            depth,
            body.include_relations.as_deref(),
        )
        .await?;
    let entities: Vec<serde_json::Value> = detail
        .entities
        .iter()
        .map(|e| {
            serde_json::json!({
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "name": e.name,
                "role": e.role,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "id": detail.graph.id.to_string(),
        "name": detail.graph.name,
        "description": detail.graph.description,
        "project_id": detail.graph.project_id.to_string(),
        "entities": entities,
        "entity_count": entities.len(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::meilisearch::indexes::CodeDocument;
    use crate::neo4j::models::FileNode;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Create an authenticated GET request for a given URI
    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Build a test router with mock backends
    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
        });
        create_router(state)
    }

    /// Build a test router with pre-seeded code documents and files
    async fn test_app_with_code() -> axum::Router {
        let app_state = mock_app_state();

        // Seed a code document in the mock search store
        let doc = CodeDocument {
            id: "src/main.rs".to_string(),
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            symbols: vec!["main".to_string(), "Config".to_string()],
            docstrings: "Main entry point for the application".to_string(),
            signatures: vec!["fn main()".to_string()],
            imports: vec!["std::io".to_string()],
            project_id: "proj-1".to_string(),
            project_slug: "test-project".to_string(),
        };
        app_state.meili.index_code(&doc).await.unwrap();

        // Seed files in the mock graph store for architecture endpoint
        let file1 = FileNode {
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: None,
        };
        let file2 = FileNode {
            path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            hash: "def456".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: None,
        };
        app_state.neo4j.upsert_file(&file1).await.unwrap();
        app_state.neo4j.upsert_file(&file2).await.unwrap();

        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
        });
        create_router(state)
    }

    // ====================================================================
    // GET /api/code/search
    // ====================================================================

    #[tokio::test]
    async fn test_search_code_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/search?query=something&limit=5"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_search_code_with_results() {
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(auth_get("/api/code/search?query=main&limit=10"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.is_array());
        let results = json.as_array().unwrap();
        assert!(!results.is_empty());
        // Verify SearchHit<CodeDocument> shape: { document, score }
        let first = &results[0];
        assert!(first["document"].is_object());
        assert!(first["score"].is_number());
        assert_eq!(first["document"]["path"], "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_code_with_language_filter() {
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/search?query=main&limit=10&language=rust",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let results = json.as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["document"]["language"], "rust");
    }

    #[tokio::test]
    async fn test_search_code_language_filter_no_match() {
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/search?query=main&limit=10&language=python",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_search_code_missing_query_param() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/search?limit=5"))
            .await
            .unwrap();

        // Missing required 'query' param → 400
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/code/architecture
    // ====================================================================

    #[tokio::test]
    async fn test_get_architecture_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/architecture"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total_files"], 0);
        assert!(json["languages"].is_array());
        assert!(json["key_files"].is_array());
        assert!(json["orphan_files"].is_array());
    }

    #[tokio::test]
    async fn test_get_architecture_with_files() {
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(auth_get("/api/code/architecture"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // We seeded 2 rust files
        assert_eq!(json["total_files"], 2);
        let languages = json["languages"].as_array().unwrap();
        assert_eq!(languages.len(), 1);
        assert_eq!(languages[0]["language"], "rust");
        assert_eq!(languages[0]["file_count"], 2);
        // key_files is a Vec<ConnectedFileNode> with { path, imports, dependents }
        assert!(json["key_files"].is_array());
    }

    // ====================================================================
    // CodeSearchQuery serde
    // ====================================================================

    #[test]
    fn test_code_search_query_deserialization() {
        let json = r#"{"query":"fn main","limit":5,"language":"rust"}"#;
        let q: CodeSearchQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "fn main");
        assert_eq!(q.limit, Some(5));
        assert_eq!(q.language, Some("rust".to_string()));
    }

    #[test]
    fn test_code_search_query_minimal() {
        let json = r#"{"query":"test"}"#;
        let q: CodeSearchQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "test");
        assert_eq!(q.limit, None);
        assert_eq!(q.language, None);
    }

    // ====================================================================
    // GET /api/code/impact — analyze_impact with caller_count
    // ====================================================================

    /// Build a test router with seeded import relationships and a project.
    /// Uses MockGraphStore directly (before Arc wrapping) to access project_files.
    async fn test_app_with_imports() -> axum::Router {
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FunctionNode, ProjectNode, Visibility};
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();

        // Create project
        let project = ProjectNode {
            id: uuid::Uuid::new_v4(),
            name: "test-proj".to_string(),
            slug: "test-proj".to_string(),
            description: None,
            root_path: "/tmp/test".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
        };
        graph.create_project(&project).await.unwrap();

        // Seed files and register in project_files
        let files = [
            "src/lib.rs",
            "src/handler.rs",
            "src/utils.rs",
            "tests/test_lib.rs",
        ];
        for path in &files {
            let file = FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "h".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            graph.upsert_file(&file).await.unwrap();
            graph
                .project_files
                .write()
                .await
                .entry(project.id)
                .or_default()
                .push(path.to_string());
        }

        // Seed a function for symbol mode
        let func = FunctionNode {
            name: "handle".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/handler.rs".to_string(),
            line_start: 0,
            line_end: 10,
            docstring: None,
        };
        graph.upsert_function(&func).await.unwrap();

        // Import relationships: handler.rs → lib.rs, utils.rs → lib.rs, test_lib.rs → lib.rs
        graph
            .create_import_relationship("src/handler.rs", "src/lib.rs", "lib")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/utils.rs", "src/lib.rs", "lib")
            .await
            .unwrap();
        graph
            .create_import_relationship("tests/test_lib.rs", "src/lib.rs", "lib")
            .await
            .unwrap();

        let app_state = mock_app_state_with(graph, MockSearchStore::new());
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(
            crate::orchestrator::FileWatcher::new(orchestrator.clone()),
        ));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
        });
        create_router(state)
    }

    #[tokio::test]
    async fn test_rest_analyze_impact_includes_caller_count() {
        let app = test_app_with_imports().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/impact?target=src/lib.rs&project_slug=test-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // caller_count field must be present
        assert!(
            json.get("caller_count").is_some(),
            "Response must include caller_count field"
        );
        let caller_count = json["caller_count"].as_i64().unwrap();
        // 3 files import lib.rs → direct dependents = 3
        assert_eq!(caller_count, 3);

        // directly_affected should contain the importers
        let direct = json["directly_affected"].as_array().unwrap();
        assert_eq!(direct.len(), 3);

        // test_files_affected should contain the test file
        let test_files = json["test_files_affected"].as_array().unwrap();
        assert!(test_files
            .iter()
            .any(|f| f.as_str().unwrap().contains("test")));

        // risk_level should reflect the count
        assert!(json.get("risk_level").is_some());
    }

    #[tokio::test]
    async fn test_rest_analyze_impact_scoped_dependent_files() {
        let app = test_app_with_imports().await;

        // With project_slug scoping, only files from the project are returned
        let resp = app
            .oneshot(auth_get(
                "/api/code/impact?target=src/lib.rs&project_slug=test-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let direct = json["directly_affected"].as_array().unwrap();
        let transitive = json["transitively_affected"].as_array().unwrap();

        // All returned files should be from the seeded project
        for file in direct.iter().chain(transitive.iter()) {
            let path = file.as_str().unwrap();
            assert!(
                path.starts_with("src/") || path.starts_with("tests/"),
                "File {} should be from the seeded project",
                path
            );
        }
    }
}
